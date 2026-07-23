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
import re
from typing import Union

import torch
from torch.amp import autocast

from auto_round.logger import logger
from auto_round.planning import LayerConfigResolutionError
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


def is_static_wfp8afp8(ar_or_format):
    if isinstance(ar_or_format, str):
        return "fp8_static" in ar_or_format.lower()
    return _as_scheme(ar_or_format).is_static_wfp8afp8()


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

    param_names = [p for p in _inspect.signature(block.forward).parameters.keys() if p != "self"]
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
    from auto_round.layer_config import (
        apply_plan_to_model,
        extract_regex_config,
        has_quantized_layer_outside_blocks,
        resolve_layer_config,
    )
    from auto_round.planning import CompressionPlan, ResolvedScheme, resolve_scheme_value
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
    plan = CompressionPlan(
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


# Explicit compatibility exports for callers that historically imported GGUF and
# ignore-layer helpers from compressors.utils.
from auto_round.formats.backends.gguf import (
    _apply_gguf_shape_fallback,
    _infer_gguf_n_layers_from_model,
    _resolve_gguf_n_layers,
    get_layer_config_by_gguf_format,
    gguf_type_fallback,
)
from auto_round.layer_config.resolver import get_fp_layer_names


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


def immediate_pack(name: str, layer_config: dict):
    from auto_round.context.compress import CompressContext
    from auto_round.context.model import ModelContext

    compress_context = CompressContext.get_context()
    model_context = ModelContext.get_context()

    if not compress_context.is_immediate_packing:
        return
    compress_context.formats[0].immediate_pack(
        name=name,
        model=model_context.model,
        device=device_manager.device,
        output_dir=_get_save_folder_name(compress_context.formats[0]),
        layer_config=layer_config,
        tokenizer=model_context.tokenizer,
        mllm=model_context.is_mllm,
        processor=getattr(model_context, "processor", None),
        image_processor=getattr(model_context, "image_processor", None),
        quant_nontext_module=getattr(model_context, "quant_nontext_module", False),
    )
