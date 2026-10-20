# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""Batched zero-shot (iters=0) search driver.

Layers whose staged key -- (device, weight shape/dtype, bits, group size,
symmetry, threshold, resolved quant callable) -- matches are stacked along a
leading dim and quantized with ONE ``weight_quant_func`` call. The quant math
is row-independent, and the per-column imatrix is pre-expanded to each
module's full weight shape before stacking so the quant function's internal
flatten-expand becomes an identity: results are bit-identical to the serial
``unwrapper({})`` path. Write-back goes through the wrapper's ``_apply_qdq``
(the unwrapper's own conventions), so serial and batched outputs can never
drift apart.
"""

from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import torch

from auto_round.algorithms.quantization.search_dispatch import (
    _fn_key,
    _plan_capacity_chunks,
    _search_fn_stackable,
    _wrap_batch_max_elems,
    dump_oom_tensor_census_,
    group_items_by_device,
    run_items_by_device,
    search_sweep_budget,
)
from auto_round.logger import logger
from auto_round.utils import set_module
from auto_round.utils.perf_counters import perf_counters_enabled
from auto_round.wrapper import WrapperLinear


def _staged_weight(wrapper: WrapperLinear) -> torch.Tensor:
    """The weight exactly as _qdq_weight would pass it to the quant func."""
    weight = wrapper.orig_layer.weight
    import transformers

    if type(wrapper.orig_layer) == transformers.pytorch_utils.Conv1D:
        weight = weight.t()
    return weight


def _staged_imatrix(wrapper: WrapperLinear, weight: torch.Tensor) -> torch.Tensor | None:
    """The RAW per-column imatrix (or None) -- expansion happens chunk-time.

    Staging the expanded full-size copy would pin N x weight bytes across the
    whole staging phase; the chunk-time expansion below is transient and
    bounded by the batch cap. Groups are homogeneous on imatrix presence (it
    is part of the staged key), so chunks are all-None or all-tensor.
    """
    del weight
    return getattr(wrapper.orig_layer, "imatrix", None)


def _expand_imatrix(im: torch.Tensor | None, weight: torch.Tensor, group_size: int) -> torch.Tensor | None:
    """Expand one module's raw imatrix to its full weight shape (chunk-time).

    Mirrors the quant funcs' own imatrix handling exactly (data_type/int.py):
    the raw per-column vector is group-padded (fill 1e-5) BEFORE being row
    expanded, so the stacked flat layout byte-matches what the serial
    per-module call produces after its internal flatten-pad-expand. Skipping
    the pad misplaces the group alignment for every module after the first
    when ``in % group_size != 0``; casting to the weight dtype (bf16/fp16
    weights with an fp32 hook-collected imatrix) silently diverges the
    search loss weighting from the serial path.
    """
    if im is None:
        return None
    im = im.to(weight.device)  # device only: keep the collected precision
    if im.dim() == 1 and weight.dim() == 2:
        out_f, in_f = weight.shape
        pad_len = 0
        if isinstance(group_size, int) and 0 < group_size <= in_f and in_f % group_size != 0:
            pad_len = ((in_f + group_size - 1) // group_size) * group_size - in_f
        if pad_len:
            im = torch.nn.functional.pad(im, (0, pad_len), value=1e-5)
        return im.unsqueeze(0).expand(out_f, -1).contiguous()
    if im.dim() == 2 and weight.dim() == 2 and im.shape[0] == 1:
        # row vector form: same treatment
        out_f, in_f = weight.shape
        pad_len = 0
        if isinstance(group_size, int) and 0 < group_size <= in_f and in_f % group_size != 0:
            pad_len = ((in_f + group_size - 1) // group_size) * group_size - in_f
        if pad_len:
            im = torch.nn.functional.pad(im, (0, pad_len), value=1e-5)
        return im.expand(out_f, -1).contiguous()
    if im.shape == weight.shape:
        return im.contiguous()
    return im


def _extra_quant_kwargs_key(wrapper: WrapperLinear) -> tuple[tuple[str, str], ...] | None:
    """Per-module _extra_quant_kwargs hook values as a hashable key term.

    Any wrapper subclass defining the hook injects arbitrary per-module kwargs
    into the quant call; two modules with differing extras must never share a
    stacked chunk (same hazard class as differing global_scale).
    """
    hook = getattr(wrapper, "_extra_quant_kwargs", None)
    if not callable(hook):
        return None
    return tuple(sorted((str(k), repr(v)) for k, v in hook().items()))


def _staged_key(wrapper: WrapperLinear, weight: torch.Tensor, imatrix: torch.Tensor | None) -> tuple[
    str,
    tuple[int, ...],
    str,
    int | None,
    int | None,
    bool | None,
    float,
    bool,
    str | tuple[Any, ...],
    str,
    int | None,
    int | None,
    float | None,
    tuple[tuple[str, str], ...] | None,
]:
    layer = wrapper.orig_layer
    # The stacked call assembles quant kwargs once from chunk[0], so every
    # per-layer kwarg _quant_call_kwargs injects must be part of the key:
    # modules differing in any of them never share a batch (nv-fp4 computes
    # weight_global_scale per layer; scale_dtype/super_* come from
    # per-layer config pins). imatrix is passed as a stacked override.
    global_scale = getattr(wrapper, "weight_global_scale", None)
    return (
        str(weight.device),
        tuple(weight.shape),
        str(weight.dtype),
        getattr(layer, "bits", None),
        getattr(layer, "group_size", None),
        getattr(layer, "sym", None),
        float(getattr(wrapper, "q_scale_thresh", 1e-5)),
        imatrix is not None,
        # normalize away per-wrapper torch.compile wrappers: the underlying
        # eager original is shared per config, while each compiled wrapper is
        # unique (repr-keyed) and would silently collapse every group to size 1
        _fn_key(getattr(wrapper.weight_quant_func, "_torchdynamo_orig_callable", None) or wrapper.weight_quant_func),
        str(getattr(layer, "scale_dtype", None)),
        getattr(layer, "super_bits", None),
        getattr(layer, "super_group_size", None),
        None if global_scale is None else float(global_scale),
        _extra_quant_kwargs_key(wrapper),
    )


def _stackable_layout(wrapper: WrapperLinear) -> bool:
    """Whether the wrapper's weight layout is safe for a stacked batch call.

    The stacked call bypasses ``_qdq_weight`` and hands the quant callable a
    3-D ``(N, out, in)`` tensor, so only row-group layouts are stackable:
    ``group_size`` -1 (per-channel) or a positive int never straddle the
    leading-dim module boundary, while ``0`` (per-tensor) would share ONE
    scale across every module in the chunk and 2-D block tuples (e.g. fp8
    ``(128, 128)``) expect a 2-D weight. Activation-only layers (``bits >=
    16``) must finish through their own ``unwrapper`` so their weights stay
    untouched and the act metadata is still processed.
    """
    layer = wrapper.orig_layer
    bits = getattr(layer, "bits", None)
    if bits is None or int(bits) >= 16:
        return False
    group_size = getattr(layer, "group_size", None)
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size == 0:
        return False
    return group_size == -1 or group_size > 0


def swap_wrapper_callables_to_eager(wrapper: WrapperLinear) -> Callable[[], None] | None:
    """Point compiled wrapper callables at their eager originals.

    Returns a restore fn or None. ``unwrapper({})`` may call BOTH the
    (possibly torch.compile-wrapped) weight_quant_func and -- for
    act-quantized layers, act_bits <= 8 -- the compiled act_quant_func (the
    act tail's first-ever call in the iters=0 lane); on worker threads
    either first call would race dynamo's trace lock, exactly like the
    stacked path.
    """
    restores = []

    for attr in ("weight_quant_func", "act_quant_func"):
        fn = getattr(wrapper, attr, None)
        if fn is None:
            continue
        orig = getattr(fn, "_torchdynamo_orig_callable", None)
        if orig is None:
            continue
        setattr(wrapper, attr, orig)
        restores.append((attr, fn))

    if not restores:
        return None

    def _restore() -> None:
        for attr, fn in restores:
            setattr(wrapper, attr, fn)

    return _restore


def _unwrap_with_cpu_fallback(w: WrapperLinear, name: str) -> torch.nn.Module:
    """unwrapper({}) under the serial lane's OOM->CPU contract.

    The per-module search can OOM on the device exactly like the serial
    path's own search (base.py); there the run falls back to CPU instead of
    crashing. The batched lane previously called unwrapper({}) unguarded --
    a module that OOMed took the whole run down, violating the progressive
    batched -> per-module -> CPU ladder.

    The rebuilt wrapper keeps the failed wrapper's optimized-RTN and NeuQI
    settings: the fallback must only change WHERE the search runs, not
    WHICH search runs (the WrapperLinear defaults would silently downgrade
    an optimized or NeuQI layer to plain RTN).
    """
    try:
        return w.unwrapper({})
    except torch.OutOfMemoryError:
        dump_oom_tensor_census_("rtn batched unwrapper")
        logger.warning("[rtn-batch] per-module search OOM for %s; falling back to CPU", name)
        layer = w.orig_layer if hasattr(w, "orig_layer") else w
        layer = layer.to("cpu")
        layer = WrapperLinear(
            layer,
            enable_minmax_tuning=False,
            enable_norm_bias_tuning=False,
            enable_round_tuning=False,
            enable_torch_compile=False,
            disable_opt_rtn=getattr(w, "disable_opt_rtn", True),
            enable_neuqi=getattr(w, "enable_neuqi", False),
            iters=0,
        )
        return layer.unwrapper({})


@torch.no_grad()
def run_batched_rtn_search(
    model: torch.nn.Module, staged: list[tuple[str, WrapperLinear]], max_batch: int | None = None
) -> None:
    """Finish deferred zero-shot wrappers on stacked same-shape batches.

    Args:
        model: The model tree (for set_module re-attachment).
        staged: list of (layer_name, wrapper) with the search deferred.
        max_batch: optional explicit chunk size.

    Every staged module is finished by this function itself -- grouped and
    stacked, singleton-per-module, or via the per-module OOM fallback -- so
    the caller never has leftover wrappers to finish.
    """
    entries = []
    for layer_name, wrapper in staged:
        weight = _staged_weight(wrapper)
        imatrix = _staged_imatrix(wrapper, weight)
        entries.append({"name": layer_name, "w": wrapper, "weight": weight, "im": imatrix})

    device_groups = OrderedDict()
    for e in entries:
        device_groups.setdefault(str(e["weight"].device), []).append(e)

    # Offload pass: the searches are weight-local, so chunks may run on any
    # device with headroom (the zero-shot lane leaves every non-home GPU idle).
    # Chunks are assigned round-robin over viable worker devices; the weights
    # and imatrices move to the worker for the stacked call and results write
    # back cross-device through _apply_qdq.
    plan_entries = []  # (home_device, per-module chain, sweep budget, max modules, group)
    for dev, batch in device_groups.items():
        by_key = OrderedDict()
        for e in batch:
            by_key.setdefault(_staged_key(e["w"], e["weight"], e["im"]), []).append(e)
        for group in by_key.values():
            fn0 = group[0]["w"].weight_quant_func
            elements = group[0]["weight"].numel() + (group[0]["im"].numel() if group[0]["im"] is not None else 0)
            per_module = elements * 4 * 4  # the fp32 copy chain per module
            max_n = 1024 if max_batch is None else max(1, max_batch)
            elem_cap = _wrap_batch_max_elems()
            if elem_cap is not None:
                max_n = min(max_n, max(1, elem_cap // max(elements, 1)))
            # layouts that must not share a stacked call (per-tensor scales,
            # 2-D block groups, activation-only layers) plan as singletons:
            # _run_chunk finishes each through the module's own unwrapper,
            # exactly like the serial lane
            if not _stackable_layout(group[0]["w"]):
                max_n = 1
            # singleton and non-stackable groups plan like any other group:
            # the RTN lane runs them through the same per-module path, and a
            # roomy idle device may still take them
            plan_entries.append((dev, per_module, search_sweep_budget(fn0), max_n, group))

    # capacity-weighted planning: chunks form directly on the device that
    # runs them, sized to its tracked headroom (see _plan_capacity_chunks)
    planned = _plan_capacity_chunks(plan_entries)

    buckets = OrderedDict()
    for worker, _home, chunk in planned:
        buckets.setdefault(str(worker), []).append(chunk)

    # per-worker execution stats for the wrap-grammar summary line
    stats: OrderedDict[str, dict] = OrderedDict()

    def _worker_of(chunk: list[dict[str, Any]]) -> str:
        for wk, cs in buckets.items():
            if any(c is chunk for c in cs):
                return str(wk)
        return str(chunk[0]["weight"].device)  # not found (should not happen): stay home

    def _chunk_working_set(chunk: list[dict[str, Any]]) -> int:
        e0 = chunk[0]
        elements = e0["weight"].numel() + (e0["im"].numel() if e0["im"] is not None else 0)
        from auto_round.algorithms.quantization.search_dispatch import search_transient_bytes

        return search_transient_bytes(e0["w"].weight_quant_func, elements * len(chunk))

    @torch.no_grad()
    def _run_chunk_guarded(chunk: list[dict[str, Any]], worker: str, threaded: bool = False) -> None:
        """_run_chunk with an execution-time headroom check: halve an oversized
        chunk before running it (plan-time estimates can drift)."""
        if len(chunk) > 1:
            from auto_round.utils.device import probe_usable_bytes

            usable = probe_usable_bytes(str(worker))
            if usable is not None and usable - 512 * 2**20 < _chunk_working_set(chunk):
                mid = len(chunk) // 2
                _run_chunk_guarded(chunk[:mid], worker, threaded)
                _run_chunk_guarded(chunk[mid:], worker, threaded)
                return
        _run_chunk(chunk, worker, threaded)

    @torch.no_grad()
    def _run_chunk(chunk: list[dict[str, Any]], worker: str, threaded: bool = False) -> None:
        st = stats.setdefault(str(worker), {"wall": 0.0, "modules": 0, "batches": 0, "singletons": 0})
        st["modules"] += len(chunk)
        if len(chunk) > 1:
            st["batches"] += 1
        else:
            st["singletons"] += 1
        w0 = chunk[0]["w"]
        dev = str(chunk[0]["weight"].device)
        worker = str(worker)
        if len(chunk) == 1 and (not _search_fn_stackable(w0.weight_quant_func) or not _stackable_layout(w0)):
            # non-stackable callable or non-stackable layout: the module's
            # own serial path, on the weight's device (the stacked call is
            # unsafe for it)
            _restore = swap_wrapper_callables_to_eager(chunk[0]["w"]) if (threaded or worker != dev) else None
            try:
                layer = _unwrap_with_cpu_fallback(chunk[0]["w"], chunk[0]["name"])
            finally:
                if _restore is not None:
                    _restore()
            set_module(model, chunk[0]["name"], layer)
            return
        # stackable chunks of ANY size (including 1-row: singleton groups or
        # max_batch=1 per-module spreading) ride the stacked machinery, which
        # moves weights+imatrix to the assigned worker and writes results
        # back through finalize -- identical search semantics for
        # row-independent fns
        weights = [e["weight"] for e in chunk]
        ims = [e["im"] for e in chunk]
        fn = w0.weight_quant_func
        if worker != dev or threaded:
            # the compiled path runs slower than eager for this bandwidth-bound
            # grid search; eager also sidesteps the cross-thread dynamo trace
            # race entirely -- including for home-staying chunks, which the
            # multi-device path also executes on worker threads
            fn = getattr(fn, "_torchdynamo_orig_callable", None) or fn
        try:
            # materialization is inside the guard on purpose: an OOM while
            # moving or stacking the batch must take the same per-module
            # ladder as an OOM inside the stacked call, not abort the run
            if worker != dev:
                weights = [w.to(worker) for w in weights]
                ims = [im.to(worker) if im is not None else None for im in ims]
            stacked_w = torch.stack(weights)
            stacked_im = None
            if ims and ims[0] is not None:
                _gs = w0.orig_layer.group_size
                stacked_im = torch.stack([_expand_imatrix(im, w, _gs) for im, w in zip(ims, weights)])
            kwargs = w0._quant_call_kwargs(
                torch.tensor(0.0), torch.tensor(1.0), torch.tensor(1.0), imatrix_override=stacked_im
            )
            kwargs = _relocate_tensor_kwargs(kwargs, str(stacked_w.device))
            qdq, scale, zp = fn(stacked_w, **kwargs)
        except torch.OutOfMemoryError:
            # release the partial batch before the retry: the ladder needs
            # the headroom the failed materialization consumed
            weights = ims = stacked_w = stacked_im = kwargs = None
            logger.warning(
                "[rtn-batch] stacked search OOM (%d modules); finishing this chunk per-module "
                "(shrink batches with AR_SEARCH_BATCH_GB or disable with AR_DISABLE_BATCHED_SEARCH=1)",
                len(chunk),
            )
            dump_oom_tensor_census_("rtn batched search")
            for e in chunk:
                _restore = swap_wrapper_callables_to_eager(e["w"]) if (threaded or worker != dev) else None
                try:
                    layer = _unwrap_with_cpu_fallback(e["w"], e["name"])
                finally:
                    if _restore is not None:
                        _restore()
                set_module(model, e["name"], layer)
            return
        n = len(chunk)
        scale_parts = _split_leading(scale, n)
        zp_parts = _split_leading(zp, n)
        import transformers

        for i, e in enumerate(chunk):
            res = qdq[i]
            w = e["w"]
            # mirror the serial _qdq_weight output contract: cast back to the
            # stored dtype and restore the HF Conv1D [in, out] layout (staging
            # transposed it)
            res = res.to(w.orig_layer.weight.dtype)
            if type(w.orig_layer) == transformers.pytorch_utils.Conv1D:
                res = res.t()
            # route the write-back through unwrapper({}) with the precomputed
            # search result injected: _apply_qdq alone would skip the
            # unwrapper tail (bias/meta update, static-act rescale, act
            # metadata, WrapperWALayer attachment) and leave act-quantized
            # layers (act_bits <= 8) silently inconsistent with the serial
            # path. The injection short-circuits _qdq_weight's recompute so
            # no compiled weight_quant_func runs here; the act tail may still
            # call its compiled act_quant_func, so swap BOTH callables to
            # eager on worker threads before unwrapping.
            _restore = swap_wrapper_callables_to_eager(w) if (threaded or worker != dev) else None
            try:
                w._presolved_qdq = (res, scale_parts[i], zp_parts[i])
                layer = w.unwrapper({})
            finally:
                if _restore is not None:
                    _restore()
            set_module(model, e["name"], layer)

    def _run_chunks(wk: str, cs: list[list[dict[str, Any]]]) -> None:
        import time as _ptime

        _t0 = _ptime.perf_counter()
        try:
            for c in cs:
                _run_chunk_guarded(c, _worker_of(c), threaded=True)
        finally:
            stats.setdefault(str(wk), {"wall": 0.0, "modules": 0, "batches": 0, "singletons": 0})["wall"] += (
                _ptime.perf_counter() - _t0
            )

    if len(buckets) > 1:
        keyed = OrderedDict()
        for wk, cs in buckets.items():
            keyed.setdefault(str(wk), []).append((len(keyed), (str(wk), cs)))
        run_items_by_device(keyed, lambda _idx, item: _run_chunks(item[0], item[1]))
    else:
        for wk, cs in buckets.items():
            _run_chunks(str(wk), cs)

    if stats and perf_counters_enabled():
        # one timing line, same grammar and gating as the wrap-search lane:
        # calls are chunks (each chunk is one stacked or per-module search
        # call), so the counts cannot read as "one call per module"
        from auto_round.utils.pool_placement import _short_device_key

        _n_mod = sum(st["modules"] for st in stats.values())
        _n_batch = sum(st["batches"] for st in stats.values())
        _n_single = sum(st["singletons"] for st in stats.values())
        _n_offloaded = sum(1 for worker, home, _chunk in planned if worker != home)
        _walls = sum(st.get("wall", 0.0) for st in stats.values())
        per_device = (
            "{" + ", ".join(f"'{_short_device_key(dev)}': {st.get('wall', 0.0):.2f}" for dev, st in stats.items()) + "}"
        )
        logger.info(
            "[rtn-batch] %d modules, %d batch + %d singleton search calls"
            + (f" ({_n_offloaded} chunk(s) offloaded to idle devices)" if _n_offloaded else "")
            + ", %.2fs device-wall (%.2fs summed) walls %s",
            _n_mod,
            _n_batch,
            _n_single,
            max((st.get("wall", 0.0) for st in stats.values()), default=0.0),
            _walls,
            per_device,
        )


def _relocate_tensor_kwargs(kwargs: dict[str, Any], device: str) -> dict[str, Any]:
    """Move accelerator-valued kwargs onto the compute device.

    The weight-local searches normally run on the weight's own device, so
    tensor kwargs (e.g. an nv-fp4 ``global_scale``) live wherever the layer
    lives; under worker offload the stacked call runs elsewhere and a
    left-behind operand would fault at the first cross-device op (the mapped-
    lane init_scale crash class). CPU scalars broadcast and are left alone.
    """
    for key, val in kwargs.items():
        if isinstance(val, torch.Tensor) and val.device.type != "cpu" and str(val.device) != device:
            kwargs[key] = val.to(device)
    return kwargs


def _split_leading(result: Any, n: int) -> list[Any]:
    """Split a quant result along the (possibly flattened) leading batch dim.

    Stacked calls may return per-module rows flattened into the leading dim
    (e.g. scale ``[N*out, 1]``) or keep a batch dim (``[N, out, groups]``);
    both reshape to ``[N, -1, ...]`` identically. Scalars pass through.
    Dicts are split recursively: GGUF K-quant funcs return scale/zp as dicts
    (``{"scale": ..., "d_scale": ...}``) whose tensor values carry the
    flattened batch dim -- sharing the whole dict per module would give every
    layer N-times the expected metadata (caught at GGUF export validation).
    Non-tensor dict values (e.g. ``None``) are shared as-is.
    """
    if isinstance(result, dict):
        split_values = {key: _split_leading(value, n) for key, value in result.items()}
        return [{key: values[i] for key, values in split_values.items()} for i in range(n)]
    if not isinstance(result, torch.Tensor):
        return [result] * n
    if result.dim() == 0 or result.numel() == 1:
        return [result] * n
    return result.reshape(n, -1, *result.shape[1:])
