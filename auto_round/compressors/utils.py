# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import random
from typing import Optional, Union

import torch
from torch.amp import autocast

from auto_round.compressors.config_resolution import LayerConfigResolutionError

# Explicit compatibility exports for callers that historically imported GGUF and
# ignore-layer helpers from compressors.utils.
from auto_round.compressors.layer_config_resolver import get_fp_layer_names
from auto_round.export.formats.backends.gguf import (
    _apply_gguf_shape_fallback,
    _infer_gguf_n_layers_from_model,
    _resolve_gguf_n_layers,
    get_layer_config_by_gguf_format,
    gguf_type_fallback,
)
from auto_round.logger import logger
from auto_round.schemes import BackendDataType  # re-exported: qlinear_fp/qlinear_int import it from here
from auto_round.schemes import (
    QuantizationScheme,
    is_mx_fp,
    is_mx_int,
    is_nv_fp,
    is_standard_fp,
)
from auto_round.utils import (
    check_to_quantized,
    get_layer_names_in_block,
    get_module,
)
from auto_round.utils.device_manager import device_manager


def _as_scheme(ar_or_scheme) -> "QuantizationScheme":
    """Resolve a compressor-like object or QuantizationScheme to a QuantizationScheme.

    `ar` (the compressor) exposes the same flat attribute names as QuantizationScheme
    (bits, data_type, act_bits, ...), so QuantizationScheme.from_dict can read them
    directly without needing ar.scheme to already be resolved.
    """
    if isinstance(ar_or_scheme, QuantizationScheme):
        return ar_or_scheme
    return QuantizationScheme(
        bits=ar_or_scheme.bits,
        group_size=ar_or_scheme.group_size,
        sym=getattr(ar_or_scheme, "sym", True),
        data_type=ar_or_scheme.data_type,
        act_bits=ar_or_scheme.act_bits,
        act_group_size=getattr(ar_or_scheme, "act_group_size", None),
        act_sym=getattr(ar_or_scheme, "act_sym", None),
        act_data_type=ar_or_scheme.act_data_type,
        act_dynamic=getattr(ar_or_scheme, "act_dynamic", None),
        super_bits=getattr(ar_or_scheme, "super_bits", None),
        super_group_size=getattr(ar_or_scheme, "super_group_size", None),
    )


# ``is_standard_fp`` / ``is_mx_fp`` / ``is_nv_fp`` / ``is_mx_int`` (data_type-string
# classifiers) now live in ``auto_round.schemes`` as the single authority and are
# re-exported above so existing ``from auto_round.compressors.utils import is_mx_fp``
# call sites keep working.


def is_wint_woq(ar):
    """Returns True for integer weight-only quantization with non-quantized activations (`act_bits >= 16`)."""
    return _as_scheme(ar).is_wint_woq()


def is_wfp8afp8(ar):
    return _as_scheme(ar).is_wfp8afp8()


def is_wint8aint8(ar):
    return _as_scheme(ar).is_wint8aint8()


def is_act_static(ar_or_format):
    if isinstance(ar_or_format, str):
        return "fp8_static" in ar_or_format.lower()
    return _as_scheme(ar_or_format).is_act_static()


def is_dynamic_wint8aint8(ar_or_format):
    if isinstance(ar_or_format, str):
        return "int8_w8a8" in ar_or_format.lower()
    return _as_scheme(ar_or_format).is_dynamic_wint8aint8()


def is_wint4aint4(ar_or_scheme):
    if isinstance(ar_or_scheme, str):
        return "int4" in ar_or_scheme.lower()
    return _as_scheme(ar_or_scheme).is_wint4aint4()


def is_dynamic_afp8(ar_or_format):
    return _as_scheme(ar_or_format).is_dynamic_afp8()


def is_block_wfp8(ar_or_format):
    return _as_scheme(ar_or_format).is_block_wfp8()


def block_forward(
    block: torch.nn.Module,
    input_ids: torch.Tensor,
    input_others: dict,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    device: torch.device = torch.device("cpu"),
    output_return_id: int = 0,
) -> Union[torch.Tensor, dict]:
    """Performs a forward pass through a block with the given inputs.

    Args:
    block: The block to perform the forward pass on.
    input_ids: The input IDs.
    input_others: A dictionary containing other input data.
    amp: A boolean indicating whether to use automatic mixed precision.
    amp_dtype: The data type for automatic mixed precision.
    device: The target device.
    output_return_id: if the output has more than one tenor, return the specified idx tensor.

    Returns:
    output: The output of the forward pass.
    """
    from auto_round.utils.model import to_device

    if input_ids.device != device:
        input_ids = to_device(input_ids, device)
        input_others = to_device(input_others, device)
    input_tuple = input_others.pop("positional_inputs", None)
    if "alibi" in input_others.keys() and input_others["alibi"] is not None:
        alibi = input_others["alibi"]
        input_others["alibi"] = alibi.reshape(-1, alibi.shape[2], alibi.shape[3])

    from auto_round.special_model_handler import prepare_special_model_block_inputs

    input_others, input_tuple = prepare_special_model_block_inputs(block, input_ids, input_others, input_tuple)

    # Use the block's actual parameter name for the first positional argument.
    import inspect as _inspect

    # For DDP-wrapped blocks, inspect the underlying module's signature
    _actual_block = getattr(block, "module", block)
    param_names = [p for p in _inspect.signature(_actual_block.forward).parameters.keys() if p != "self"]
    block_input_kwarg = param_names[0] if param_names else "hidden_states"
    if block_input_kwarg not in input_others:
        input_others[block_input_kwarg] = input_ids

    # Convert positional inputs to keyword args for any remaining positional parameters.
    positional_inputs = input_tuple or ()
    if positional_inputs:
        for i, val in enumerate(positional_inputs):
            param_idx = i + 1  # hidden_states is params[0]
            if param_idx < len(param_names):
                param_name = param_names[param_idx]
                if param_name not in input_others:
                    input_others[param_name] = val
        positional_inputs = ()

    if amp:
        with autocast(device_type=str(device).split(":")[0], dtype=amp_dtype):  # pragma: no cover
            output = block(**input_others)
    else:
        output = block(**input_others)
    if isinstance(output_return_id, int) and (isinstance(output, list) or isinstance(output, tuple)):
        output = output[output_return_id]
    return output


def check_skippable_keywords(key):
    """
    Prints a reminder if a key is not stored during quantization fine-tuning.
    """
    skippable_cache_keys = ("past_key_value",)
    for cache_key in skippable_cache_keys:
        if cache_key not in key:
            return True
    return False


def check_need_act_calibration(
    is_act_dynamic: Union[bool, None],
    act_data_type: Union[str, None] = None,
    act_bits: Union[int, None] = 16,
    static_kv_dtype: Union[str, None] = None,
    static_attention_dtype: Union[str, None] = None,
) -> bool:
    if static_kv_dtype is not None or static_attention_dtype is not None:
        return True
    if act_bits is None or act_bits > 8:
        return False
    # None is dynamic
    if is_act_dynamic is not None and not is_act_dynamic:
        return True
    if act_data_type is not None and "static" in act_data_type:
        return True
    return False


def collect_best_params(block, cache_device="cpu"):
    """Collect the best parameters from the block to the specified device."""
    params = {}
    if hasattr(block, "orig_layer"):
        for key in block.params.keys():
            params[key] = block.params[key].data.to(cache_device, copy=True)
    else:
        for n, m in block.named_modules():
            if hasattr(m, "orig_layer"):
                params[n] = {}
                for key in m.params.keys():
                    params[n][key] = m.params[key].data.to(cache_device, copy=True)
    return params


def collect_best_params_local(block):
    """Best-params snapshot duplicated on each parameter's own device.

    With a CPU cache device this is not used (host parking is the point of
    ``low_gpu_mem_usage``). With a non-CPU cache device the historical path
    copied every parameter to that single device -- concentrating all devices'
    snapshot bytes on one GPU and paying cross-device copies on every improving
    iteration. Duplicating each parameter on the device that already hosts it
    keeps the footprint spread like the weights themselves, removes the
    cross-device traffic, and makes the unwrap copy-back local. Values and the
    restore path are unchanged. Raises on failure (typically an out of memory
    during the clone itself): the snapshot ladder's attempt-and-catch then
    walks the next rung - swallowing the error here would skip the freest-peer
    rung, mislabel the sticky route as local while the data sits on the host,
    and re-attempt the doomed clone every improving iteration.
    """
    params = {}
    if hasattr(block, "orig_layer"):
        for key, p_ in block.params.items():
            params[key] = p_.data.to(p_.data.device, copy=True)
    else:
        for n, m in block.named_modules():
            if hasattr(m, "orig_layer"):
                params[n] = {key: p_.data.to(p_.data.device, copy=True) for key, p_ in m.params.items()}
    return params


def _accel_mem_get_info_(device):
    """``(free_bytes, total_bytes)`` for any accelerator device, or ``None``.

    Routes through the device-manager abstraction (``get_ar_device``), so
    cuda, xpu and hpu behave identically; ``None`` means "unknown" and every
    caller keeps its fallback instead of guessing.
    """
    if isinstance(device, str):
        # device_manager.device may be a str ("cuda:0"); strings have no
        # .type attribute, so the getattr below would misclassify them as
        # cpu (the same class as the census str-device guard)
        device = torch.device(device)
    if device is None or getattr(device, "type", "cpu") == "cpu":
        return None
    try:
        from auto_round.utils.device_manager import get_ar_device

        ar_device = get_ar_device(device.type)
        if not ar_device.is_available():
            return None
        free_b, total_b = ar_device.mem_get_info(device.index or 0)
        return int(free_b), int(total_b)
    except Exception:  # pylint: disable=broad-except
        return None


def _snapshot_param_needs(block) -> list[tuple[torch.device, int]]:
    """Per-device snapshot need (bytes) across the block's wrappers."""
    needs = {}

    def _add(params):
        for tensor in params.values():
            if isinstance(tensor, torch.Tensor):
                home = tensor.device
                needs[home] = needs.get(home, 0) + tensor.numel() * tensor.element_size()

    if hasattr(block, "orig_layer"):
        _add(block.params)
    else:
        for _, module in block.named_modules():
            if hasattr(module, "orig_layer"):
                _add(module.params)
    return sorted(needs.items(), key=lambda kv: str(kv[0]))


def _idle_peer_for_(need_bytes, home) -> Optional[torch.device]:
    """An idle CUDA peer with headroom for ``need_bytes``, or ``None``.

    Shared by the snapshot ladders (the huge-layer wrapper path and the
    block path). An idle peer still carries a CUDA context and allocator
    fragmentation; a tenth of its free memory stays untouched.
    """
    if home is None or getattr(home, "type", "cpu") == "cpu":
        return None
    try:
        from auto_round.utils.device_manager import get_ar_device

        count = get_ar_device(home.type).device_count
        count = count() if callable(count) else count  # property or method
    except Exception:  # pylint: disable=broad-except - exotic devices
        return None
    for index in range(count):
        candidate = torch.device(home.type, index)
        if candidate == home:
            continue
        info = _accel_mem_get_info_(candidate)
        if info is not None and info[0] * 0.9 >= need_bytes:
            return candidate
    return None


def _snapshot_route_log_(block, route, msg, *args, warn_first=False, silent=False) -> None:
    """Log a snapshot routing decision once per change; repeats log nothing.

    The ladder re-runs on every best-params update because free memory moves
    between iterations; the re-evaluation stays, but an unchanged route (the
    common case - parameters improving again) says nothing worth reading at
    any level. ``silent`` records the route for stickiness without logging
    (the unremarkable beside-the-weights path never logs); ``warn_first``
    keeps a fallback's first occurrence at WARNING."""
    last = getattr(block, "_snapshot_route", None)
    if last == route:
        return
    if not silent:
        (logger.warning if warn_first else logger.info)(msg, *args)
    block._snapshot_route = route


def _snapshot_free_devices_(dev_type: str, need_bytes: int, exclude=()) -> list:
    """Visible accelerator devices of ``dev_type`` with room, most free first.

    Same introspection as ``_idle_peer_for_`` but returns the full ordering:
    the attempt ladder tries the freest device first, and every candidate is
    still guarded by a real attempt-and-catch at clone time. Devices already
    hosting the block's params are excluded - the local rung covers them, and
    re-attempting a just-failed same-device clone is pure churn."""
    devices = []
    try:
        from auto_round.utils.device_manager import get_ar_device

        count = get_ar_device(dev_type).device_count
        count = count() if callable(count) else count  # property or method
    except Exception:  # pylint: disable=broad-except - exotic devices
        return devices
    excluded = {torch.device(d) if not isinstance(d, torch.device) else d for d in exclude}
    for index in range(count):
        candidate = torch.device(dev_type, index)
        if candidate in excluded:
            continue
        info = _accel_mem_get_info_(candidate)
        if info is not None and info[0] * 0.9 >= need_bytes:
            devices.append((info[0], candidate))
    devices.sort(key=lambda kv: kv[0], reverse=True)
    return [d for _b, d in devices]


def snapshot_best_params(block, cache_device="cpu"):
    """Collect the best-params snapshot by attempt, falling back device by device.

    Sticky per block: duplicate beside the weights first (per-parameter local
    copies), then the freest visible accelerator of the same type, then the
    host with a WARNING. Every step is attempted for real - a failure
    (typically an out of memory during the clone itself) falls through to
    the next candidate; a successful route is retried first on later
    improving iterations, and the host is terminal. A CPU ``cache_device``
    (``low_gpu_mem_usage``) snapshots on the host directly - VRAM was never
    budgeted for caching.
    """
    try:
        non_cpu = torch.device(str(cache_device)).type != "cpu"
    except (ValueError, RuntimeError):
        non_cpu = False  # unrecognized label: keep the historical path (it will raise as before)
    if not non_cpu:
        return collect_best_params(block, cache_device)

    last = getattr(block, "_snapshot_route", None)
    if last == "host":
        # terminal: no mid-tune flip-flopping back onto accelerators
        return collect_best_params(block, "cpu")

    total_need = sum(need for _home, need in _snapshot_param_needs(block)) or 0
    candidates: list = []
    if last == "local":
        candidates.append("local")
    elif isinstance(last, str) and last.startswith("peer:"):
        try:
            candidates.append(torch.device(last[5:]))
        except (ValueError, RuntimeError):
            pass
    if "local" not in candidates:
        candidates.append("local")
    _home = {
        p_.data.device
        for m in block.named_modules()
        if hasattr(m, "orig_layer")
        for p_ in m.params.values()
        if isinstance(p_, torch.Tensor)
    }
    for dev in _snapshot_free_devices_(torch.device(str(cache_device)).type, total_need, exclude=_home):
        if dev not in candidates:
            candidates.append(dev)

    for cand in candidates:
        try:
            if cand == "local":
                out = collect_best_params_local(block)
                _snapshot_route_log_(
                    block, "local", "[snapshot] %.2fGiB stays beside the weights", total_need / 2**30, silent=True
                )
                return out
            out = collect_best_params(block, cand)
            _snapshot_route_log_(block, f"peer:{cand}", "[snapshot] cloning %.2fGiB to %s", total_need / 2**30, cand)
            return out
        except Exception as e:  # pylint: disable=broad-except - the attempt IS the test
            logger.debug(
                "[snapshot] placement on %s failed (%s: %s); trying the next candidate", cand, type(e).__name__, e
            )
            continue
    _snapshot_route_log_(
        block,
        "host",
        "[snapshot] no accelerator could hold the %.2fGiB snapshot; parking on host",
        total_need / 2**30,
        warn_first=True,
    )
    return collect_best_params(block, "cpu")


def snapshot_window_floor_bytes(wrapper) -> int:
    """Minimum free bytes the row-window machinery needs beside the snapshot.

    ``WrapperLinear.row_block_bounds`` never shrinks a window below 1024 rows
    and keeps about six fp32 row-block-sized arrays live in the quantize and
    backward math, so a home-resident snapshot must leave that working set
    room. Both anchors come from the row-block machinery itself; the value is
    computed from the layer's real shapes.
    """
    params = getattr(wrapper, "params", None)
    value = params.get("value") if params else None
    if not isinstance(value, torch.Tensor) or value.dim() < 2:
        return 0
    in_features = value.shape[-1]
    return int(1024 * in_features * 4 * 6)


def select_snapshot_device(wrapper) -> torch.device:
    """Device for a huge layer's best-params snapshot; host fallback.

    Ladder: the layer's own device when the snapshot provably fits beside the
    row-window floor (``free - snapshot >= window floor``) -- the row windows
    then size themselves against the reduced free pool, trading window size
    for keeping the snapshot on-device; otherwise an idle CUDA peer with
    headroom (peer-to-peer refreshes replace the host round trip); otherwise
    the host, the previously shipped behavior. Never raises.
    """
    need = sum(t.numel() * t.element_size() for t in getattr(wrapper, "params", {}).values())
    home = getattr(wrapper, "device", None)
    if not isinstance(home, torch.device):
        value = getattr(wrapper, "params", {}).get("value")
        home = value.device if isinstance(value, torch.Tensor) else None
    if need <= 0 or home is None or home.type == "cpu":
        return torch.device("cpu")

    info = _accel_mem_get_info_(home)
    if info is not None:
        free = info[0]
        floor = snapshot_window_floor_bytes(wrapper)
        if free - need >= floor:
            return home

    peer = _idle_peer_for_(need, home)
    if peer is not None:
        return peer

    return torch.device("cpu")


class BestParamsSlot:
    """Pre-reserved best-params snapshot for a huge tuning layer.

    Reserved BEFORE the tune loop starts, so the row-window machinery's
    per-forward free-memory probe sees the reduced pool from iteration 0 and
    windows shrink in favor of keeping the snapshot -- instead of sizing
    windows on the full pool and overflowing on the first improving
    iteration. Refreshing an on-device slot is a copy into existing buffers
    (an intra-device blip, or a peer-to-peer transfer on a parked peer);
    the host fallback keeps the historical collect-per-improvement path.
    """

    def __init__(self, wrapper):
        self.device = select_snapshot_device(wrapper)
        self.buffers = None
        if self.device.type != "cpu":  # any accelerator; the ladder may pick xpu/hpu peers
            try:
                self.buffers = {
                    key: torch.empty_like(t.data, device=self.device)
                    for key, t in getattr(wrapper, "params", {}).items()
                }
            except RuntimeError as e:  # pragma: no cover - reservation OOM
                logger.warning("[snapshot] slot reservation on %s failed (%s); using the host", self.device, e)
                self.device = torch.device("cpu")
                self.buffers = None
        if self.device.type != "cpu":  # any accelerator; the ladder may pick xpu/hpu peers
            need = sum(t.numel() * t.element_size() for t in self.buffers.values())
            logger.info(
                "[snapshot] best-params slot reserved on %s (%.2f GiB); "
                "row windows size against the reduced free pool",
                self.device,
                need / 2**30,
            )

    def refresh(self, wrapper):
        """Copy the current best parameters into the slot; returns the snapshot."""
        if self.buffers is not None:
            for key, tensor in self.buffers.items():
                tensor.copy_(wrapper.params[key].data)
            return self.buffers
        return collect_best_params(wrapper, "cpu")


def infer_bits_by_data_type(data_type: str):
    """Infer bits by data_type

    Args:
        data_type (str): data_type

    Returns:
        int: bits inferred by data_type, None means cannot infer correct bits by data_type
    """
    from auto_round.utils import SUPPORTED_DTYPES

    if data_type is None:
        return 16
    for supported_dtype in SUPPORTED_DTYPES:
        if data_type.startswith(supported_dtype) and len(data_type) > len(supported_dtype):
            ##first check the following two bits
            suc_2str = data_type[len(supported_dtype) : len(supported_dtype) + 2]
            if str.isdigit(suc_2str):
                return int(suc_2str)
            if str.isdigit(data_type[len(supported_dtype)]):
                return int(data_type[len(supported_dtype)])
    return None


def set_layer_config(
    model: torch.nn.Module,
    layer_config: dict[str, Union[str, dict, "QuantizationScheme"]],
    default_scheme: Union[str, "QuantizationScheme"],
    default_scale_dtype: torch.dtype | str,
    supported_types: tuple,
    inner_supported_types: tuple,
    quant_block_list=None,
    ignore_layers: str = "",
    quant_lm_head: bool = False,
    enable_gguf_official_mixed: bool = True,
    is_mllm: bool = False,
    fill_default_value=True,
) -> tuple[dict, bool, dict]:
    """Compatibility adapter for the pure layer-config resolver and explicit apply phase."""
    from auto_round.compressors.config_resolution import (
        ResolvedQuantizationConfig,
        ResolvedScheme,
        resolve_scheme_value,
    )
    from auto_round.compressors.layer_config_resolver import (
        apply_plan_to_model,
        extract_regex_config,
        has_quantized_layer_outside_blocks,
        resolve_layer_config,
    )
    from auto_round.schemes import get_gguf_scheme

    if isinstance(default_scheme, ResolvedScheme):
        resolved_scheme = default_scheme
    elif isinstance(default_scheme, QuantizationScheme):
        resolved_scheme = ResolvedScheme.from_scheme(
            default_scheme,
            preset_name=get_gguf_scheme(default_scheme) or None,
        )
    else:
        resolved_scheme = resolve_scheme_value(default_scheme, {})

    resolved = resolve_layer_config(
        model=model,
        scheme=resolved_scheme,
        layer_config=layer_config,
        scale_dtype=default_scale_dtype,
        supported_types=supported_types,
        inner_supported_types=inner_supported_types,
        quant_block_list=quant_block_list,
        ignore_layers=ignore_layers,
        quant_lm_head=quant_lm_head,
        enable_gguf_official_mixed=enable_gguf_official_mixed,
        is_mllm=is_mllm,
        fill_default_value=fill_default_value,
    )
    regex_config = extract_regex_config(
        model=model,
        scheme=resolved_scheme,
        layer_config=layer_config,
        scale_dtype=default_scale_dtype,
        supported_types=supported_types,
        inner_supported_types=inner_supported_types,
        ignore_layers=ignore_layers,
        fill_default_value=fill_default_value,
    )
    has_outside = has_quantized_layer_outside_blocks(resolved)
    plan = ResolvedQuantizationConfig(
        scheme=resolved_scheme,
        formats=(),
        layer_config=resolved,
        regex_config=regex_config,
        has_qlayer_outside_block=has_outside,
        scale_dtype=default_scale_dtype,
        quant_block_list=quant_block_list,
    )
    apply_plan_to_model(model, plan)
    return (
        {name: dict(config) for name, config in plan.layer_config.items()},
        plan.has_qlayer_outside_block,
        {name: dict(config) for name, config in plan.regex_config.items()},
    )


def get_shared_keys(model):
    """
    Retrieves shared keys from the model's state dictionary.

    Args:
        model (torch.nn.Module): The model to retrieve shared keys from.

    Returns:
        tuple: tuple of shared keys.
    """
    from auto_round.special_model_handler import SPECIAL_SHARED_CACHE_KEYS
    from auto_round.utils import SHARED_CACHE_KEYS

    shared_keys = SHARED_CACHE_KEYS
    shared_keys += SPECIAL_SHARED_CACHE_KEYS.get(model.__class__.__name__, ())
    return shared_keys


def init_cache(positional_inputs, inputs):
    """
    Initializes special model inputs by adding positional inputs if missing.

    Args:
        positional_inputs (list): List of positional inputs to add to inputs.
        inputs (dict): Dictionary of model inputs.

    Modifies:
        inputs (dict): Adds "positional_inputs" key if not present.
    """
    from auto_round.utils.model import to_device

    if "positional_inputs" not in inputs:  # for chatglm Series
        inputs["positional_inputs"] = []
    for idx, item in enumerate(positional_inputs):
        inputs["positional_inputs"] = to_device(positional_inputs)


def reset_params(inputs):
    """
    Resets specific input parameters to avoid saving the key-value cache during fine-tuning.

    Args:
        inputs (dict): Dictionary of model inputs.

    Modifies:
        inputs (dict): Sets "use_cache" to False if the key is present.
    """
    if "use_cache" in inputs.keys():  # Not storing kv cache
        inputs["use_cache"] = False


class IndexSampler:
    """A cyclic sampler that returns shuffled index batches.

    This sampler maintains internal state so that each call to `next_batch()`
    continues from where it left off. When the remaining number of samples is
    less than `batch_size`, the sampler reshuffles all indices and starts from
    the beginning, discarding the last incomplete batch.

    Attributes:
        nsamples (int): Total number of samples.
        batch_size (int): Number of indices to return in each batch.
        index (int): Current position in the index list.
        indices (List[int]): Shuffled list of indices.
    """

    def __init__(self, nsamples: int, batch_size: int) -> None:
        """Initializes the sampler.

        Args:
            nsamples (int): Total number of samples (must be >= batch_size).
            batch_size (int): Number of indices per batch.

        Raises:
            ValueError: If batch_size is not in the range (0, nsamples].
        """
        if batch_size <= 0 or batch_size > nsamples:
            raise ValueError("batch_size must be > 0 and <= nsamples")

        self.nsamples: int = nsamples
        self.batch_size: int = batch_size
        self.index: int = 0

        self.indices: list[int] = list(range(nsamples))
        random.shuffle(self.indices)

    def next_batch(self) -> list[int]:
        """Returns the next batch of shuffled indices.

        If the remaining indices are fewer than `batch_size`, the sampler
        reshuffles the entire list and starts from the beginning.

        Returns:
            list[int]: A list of size `batch_size` containing sample indices.
        """
        if self.index + self.batch_size > self.nsamples:
            random.shuffle(self.indices)
            self.index = 0

        batch = self.indices[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return batch


def _get_quantized_layer_names_outside_blocks(model, layer_config, supported_types, quant_block_list) -> list:
    """Gets the names of quantized layers outside blocks in the model.

    Returns:
        list: List of layer names outside blocks.
    """
    if layer_config is None or len(layer_config) == 0:
        return []

    layer_names = []
    all_layers_in_block = get_layer_names_in_block(model, supported_types, quant_block_list)

    for key in layer_config.keys():
        if key in all_layers_in_block:
            continue
        layer = get_module(model, key)
        if layer is None:
            raise LayerConfigResolutionError(f"could not find layer '{key}' in the model")
        if type(layer) in supported_types and check_to_quantized(layer_config[key]):
            layer_names.append(key)

    return layer_names


def _get_diffusion_save_folder_name(format) -> str:
    """Generates the save folder name based on the provided format string.

    If there are multiple formats to handle, the function creates a subfolder
    named after the format string with special characters replaced. If there's
    only one format, it returns the original output directory directly.

    Args:
        format_str (str): The format identifier (e.g., 'gguf:q2_k_s').

    Returns:
        str: The path to the folder where results should be saved.
    """
    from auto_round.context.compress import CompressContext
    from auto_round.context.model import ModelContext

    compress_context = CompressContext.get_context()
    model_context = ModelContext.get_context()

    # Replace special characters to make the folder name filesystem-safe
    sanitized_format = format.get_backend_name().replace(":", "-").replace("_", "-")

    formats = compress_context.formats
    # Use a subfolder only if there are multiple formats
    if len(formats) > 1:
        return (
            os.path.join(compress_context.output_dir, sanitized_format, "transformer")
            if compress_context.is_immediate_saving
            else os.path.join(compress_context.output_dir, sanitized_format, "transformer")
        )

    # if use is_immediate_saving, we need to save model in self.output_dir/transformer folder
    return (
        os.path.join(compress_context.output_dir, "transformer")
        if compress_context.is_immediate_saving
        else compress_context.output_dir
    )


def _get_save_folder_name(format, *args, **kwargs) -> str:
    """Generates the save folder name based on the provided format string.

    If there are multiple formats to handle, the function creates a subfolder
    named after the format string with special characters replaced. If there's
    only one format, it returns the original output directory directly.

    Args:
        format_str (str): The format identifier (e.g., 'gguf:q2_k_s').

    Returns:
        str: The path to the folder where results should be saved.
    """
    from auto_round.context.compress import CompressContext
    from auto_round.context.model import ModelContext

    compress_context = CompressContext.get_context()
    model_context = ModelContext.get_context()
    if model_context.is_diffusion:
        return _get_diffusion_save_folder_name(format)
    # Replace special characters to make the folder name filesystem-safe
    sanitized_format = format.get_backend_name().replace(":", "-").replace("_", "-")

    # Use a subfolder only if there are multiple formats
    if len(compress_context.formats) > 1:
        return os.path.join(compress_context.output_dir, sanitized_format)

    return compress_context.output_dir


def _resolve_pack_device_(weight, default_device):
    """Pack device for a layer: the host when int64 intermediates can't fit VRAM.

    Huge layers (a 248k-vocab lm_head) pack into int64 intermediates of ~8-16
    bytes/elem: right after its tune the GPU still holds tuning remnants, and
    the pack OOMs even though the tune itself fit. Probe the live free VRAM
    and drop to the host when the intermediates cannot fit - the packed
    output serializes to disk from the host just as well. Body-sized linears
    stay on the device. Caller-specified devices bypass the probe entirely.
    """
    info = _accel_mem_get_info_(
        torch.device(default_device) if not isinstance(default_device, torch.device) else default_device
    )
    if weight is None or info is None:
        return default_device
    free_b = info[0]
    need_b = weight.numel() * 16
    if need_b > free_b * 0.9:
        logger.info(
            "[pack] packing on cpu: intermediates ~%.1fGiB vs %.1fGiB free on %s",
            need_b / 2**30,
            free_b / 2**30,
            default_device,
        )
        return torch.device("cpu")
    return default_device


def immediate_pack(name: str, layer_config: dict, device=None):
    from auto_round.context.compress import CompressContext
    from auto_round.context.model import ModelContext

    compress_context = CompressContext.get_context()
    model_context = ModelContext.get_context()

    if not compress_context.is_immediate_packing:
        return
    pack_device = _resolve_pack_device_(
        getattr(get_module(model_context.model, name), "weight", None),
        device if device is not None else device_manager.device,
    )
    compress_context.formats[0].immediate_pack(
        name=name,
        model=model_context.model,
        device=pack_device,
        output_dir=_get_save_folder_name(compress_context.formats[0]),
        layer_config=layer_config,
        tokenizer=model_context.tokenizer,
        mllm=model_context.is_mllm,
        processor=getattr(model_context, "processor", None),
        image_processor=getattr(model_context, "image_processor", None),
        quant_nontext_module=getattr(model_context, "quant_nontext_module", False),
    )
