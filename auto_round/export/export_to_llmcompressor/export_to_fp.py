# Copyright (c) 2023 Intel Corporation
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

import copy
import inspect
import json
import os
import sys
from typing import Callable, Union

import torch
import torch.nn as nn
import transformers
from tqdm import tqdm

from auto_round.compressors.utils import is_mx_fp, is_nv_fp
from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear
from auto_round.export.export_to_llmcompressor.config import initialize_quantization
from auto_round.export.export_to_llmcompressor.utils import generate_ignore_regex_list
from auto_round.export.utils import (
    filter_quantization_config,
    is_immediate_saving_mode,
    release_layer_safely,
    save_model,
    save_pretrained_artifact,
)
from auto_round.logger import logger
from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    check_start_with_block_name,
    check_to_quantized,
    copy_python_files_from_model_cache,
    get_block_names,
    get_lm_head_name,
    get_module,
    set_amax_for_all_moe_layers,
    set_module,
    unsupported_meta_device,
)
from auto_round.wrapper import WrapperWALayer

from .config import check_compressed_tensors_supported
from .export_to_static_fp import (
    _append_attention_group,
    _configure_gaudi2_fp8_dtype,
    _construct_kv_scheme,
    _use_fp8_attention,
    _use_fp8_kv,
)

__all__ = [
    "pack_layer",
    "save_quantized_as_fp",
]

# Lazy import: vLLM LinearBase (None when vLLM is not installed).
_VLLMLinearBase = None
try:
    from vllm.model_executor.layers.linear import LinearBase as _VLLMLinearBase
except Exception:
    pass


def _unfuse_vllm_packed_layer(name: str, model: torch.nn.Module, qlayer) -> bool:
    """Split a packed fused vLLM layer (qkv_proj / gate_up_proj) into separate
    sub-layers stored under the corresponding HF-checkpoint key names.

    For qkv_proj  → injects q_proj / k_proj / v_proj into the parent module.
    For gate_up_proj → injects gate_proj / up_proj into the parent module.

    Returns True if the layer was unfused, False otherwise.
    """
    layer_base = name.rsplit(".", 1)[-1]

    # Determine the sub-layer names and output dimension split
    if layer_base == "qkv_proj":
        sub_names = ["q_proj", "k_proj", "v_proj"]
    elif layer_base == "gate_up_proj":
        sub_names = ["gate_proj", "up_proj"]
    else:
        return False

    # Get the split sizes from the original vLLM LinearBase (stored as output_sizes)
    # We stored output_sizes on qlayer before replacing the original layer.
    output_sizes = getattr(qlayer, "_vllm_output_sizes", None)
    if output_sizes is None or len(output_sizes) != len(sub_names):
        logger.warning(
            "Cannot unfuse vLLM layer %s: _vllm_output_sizes not set or length mismatch "
            "(expected %d, got %s). Saving as fused.",
            name,
            len(sub_names),
            output_sizes,
        )
        return False

    # Compute cumulative offsets
    offsets = [0]
    for s in output_sizes:
        offsets.append(offsets[-1] + s)

    # bits=8 (MXFP8): weight attr is "weight"   [out, in]        dtype float8_e4m3fn
    # bits=4 (MXFP4/NVFP4): weight attr is "weight_packed" [out, in//2] dtype uint8
    if hasattr(qlayer, "weight_packed"):
        fused_weight = qlayer.weight_packed
        weight_attr = "weight_packed"
    else:
        fused_weight = qlayer.weight
        weight_attr = "weight"
    fused_scale = qlayer.weight_scale  # uint8 (mxfp8 exponents) or float16/fp8 (mxfp4/nvfp4)
    fused_bias = getattr(qlayer, "bias", None)
    global_scale = getattr(qlayer, "weight_global_scale", None)
    input_global_scale = getattr(qlayer, "input_global_scale", None)

    # Resolve parent module
    parts = name.rsplit(".", 1)
    parent_path = parts[0] if len(parts) == 2 else ""
    parent = get_module(model, parent_path) if parent_path else model

    # For bits=4 the weight is packed 2 fp4 per byte along in-dim,
    # but output-dim slicing is unaffected.
    in_features_packed = fused_weight.shape[1]

    for i, sub_name in enumerate(sub_names):
        row_start, row_end = offsets[i], offsets[i + 1]
        sub_out = row_end - row_start

        # Slice weight and scale along dim=0 (output dimension)
        sub_weight = fused_weight[row_start:row_end, :].contiguous()
        # weight_scale rows correspond 1:1 with weight rows
        sub_scale = fused_scale[row_start:row_end, :].contiguous()
        sub_bias = fused_bias[row_start:row_end].contiguous() if fused_bias is not None else None

        # infeatures for the QuantLinear constructor: un-pack for bits=4
        sub_infeatures = in_features_packed * 2 if qlayer.bits == 4 else in_features_packed

        # Build a fresh QuantLinear shell and assign buffers directly (skip .pack())
        sub_qlayer = QuantLinear(
            qlayer.bits,
            qlayer.group_size,
            sub_infeatures,
            sub_out,
            sub_bias is not None,
            weight_dtype=sub_weight.dtype,
            sym=qlayer.sym,
            data_type=qlayer.data_type,
            act_bits=qlayer.act_bits,
            act_data_type=qlayer.act_data_type,
        )
        sub_qlayer.device = qlayer.device
        setattr(sub_qlayer, weight_attr, sub_weight)
        sub_qlayer.weight_scale = sub_scale
        if sub_bias is not None:
            sub_qlayer.bias = sub_bias
        if global_scale is not None:
            sub_qlayer.weight_global_scale = global_scale
        if input_global_scale is not None:
            sub_qlayer.input_global_scale = input_global_scale

        # Register under the parent module with the HF-style name
        setattr(parent, sub_name, sub_qlayer)

    # Remove the original fused layer from the parent
    delattr(parent, layer_base)
    logger.info("Unfused vLLM layer %s → %s", name, sub_names)
    return True


def pack_layer(name, model, device=None):
    layer = get_module(model, name)
    _is_vllm_linear = _VLLMLinearBase is not None and isinstance(layer, _VLLMLinearBase)
    if type(layer) not in SUPPORTED_LAYER_TYPES and not isinstance(layer, WrapperWALayer) and not _is_vllm_linear:
        return  ##already packed

    if isinstance(layer, WrapperWALayer):  # revert WrapperWALayer for offline usage
        wp_layer = layer
        layer = wp_layer.orig_layer
        set_module(model, name, layer)
        _is_vllm_linear = _VLLMLinearBase is not None and isinstance(layer, _VLLMLinearBase)

    orig_device = layer.weight.device
    data_type = layer.data_type
    act_bits = layer.act_bits
    act_data_type = layer.act_data_type
    bits = layer.bits
    if bits > 8:
        return
    group_size = layer.group_size
    sym = layer.sym

    if is_nv_fp(act_data_type) and act_bits <= 8:
        input_global_scale = getattr(layer, "input_global_scale", None)
        if input_global_scale is None:
            if not hasattr(layer, "act_max"):
                from auto_round.logger import logger as _logger

                _logger.error(
                    f"act_max missing for layer '{name}' "
                    f"(type={type(layer).__name__}, act_bits={act_bits}, act_data_type={act_data_type}). "
                    f"attrs: {[a for a in dir(layer) if not a.startswith('_')][:20]}"
                )
            assert hasattr(layer, "act_max")
            from auto_round.data_type.nvfp import calculate_gparam

            input_global_scale = calculate_gparam(layer.act_max, layer.group_size)  # , model.device
            setattr(layer, "input_global_scale", input_global_scale)
            delattr(layer, "act_max")

    # QuantLinear = get_fp_qlinear(backend, bits, group_size, sym)

    if type(layer) == nn.Linear:
        in_features = layer.in_features
        out_features = layer.out_features
    elif type(layer) == nn.Conv2d:
        in_features = layer.in_channels
        out_features = layer.out_channels
    elif type(layer) == transformers.pytorch_utils.Conv1D:
        in_features = layer.weight.shape[0]
        out_features = layer.weight.shape[1]
    elif _is_vllm_linear:
        # vLLM LinearBase uses input_size / output_size instead of in_features / out_features
        in_features = layer.input_size
        out_features = layer.output_size

    bias = layer.bias is not None
    ##bias = True  ## if using the above, llama3 lambada RTN will be NAN , TODO why?
    qlayer = QuantLinear(  ##pylint: disable=E1123
        bits,
        group_size,
        in_features,
        out_features,
        bias,
        weight_dtype=layer.weight.dtype,
        sym=sym,
        data_type=data_type,
        act_bits=act_bits,
        act_data_type=act_data_type,
    )

    qlayer.device = orig_device
    scale = layer.scale
    global_scale = getattr(layer, "weight_global_scale", None)
    input_global_scale = getattr(layer, "input_global_scale", None)
    # zero = layer.zp # no zeros to handle, as mxfp/nvfp do not support asym quantization
    qlayer.pack(layer, scale, global_scale=global_scale, input_global_scale=input_global_scale, device=device)
    qlayer.to(orig_device)

    # For vLLM fused layers (qkv_proj, gate_up_proj), store the per-part output
    # sizes so _unfuse_vllm_packed_layer can slice them into HF-style sub-layers.
    if _is_vllm_linear and hasattr(layer, "output_sizes"):
        qlayer._vllm_output_sizes = list(layer.output_sizes)
        # QuantLinear stores act_data_type in **kwargs but doesn't set self.act_data_type;
        # store it explicitly so _unfuse_vllm_packed_layer can read it.
        qlayer.act_data_type = act_data_type

    set_module(model, name, qlayer)

    # Attempt to unfuse vLLM fused layers into HF-style separate sub-layers.
    if _is_vllm_linear:
        _unfuse_vllm_packed_layer(name, model, qlayer)

    # Note: release weight and bias explicitly, in case they are referenced elsewhere
    release_layer_safely(layer)


def _get_scheme(bits, data_type):
    """Determine the compressed-tensors format string for a given data type and bit-width."""
    if is_mx_fp(data_type):
        return "MXFP4" if bits == 4 else "MXFP8"
    if is_nv_fp(data_type):
        return "NVFP4"
    return None


def _get_group_format(bits, data_type):
    """Determine the compressed-tensors format string for a given data type and bit-width."""
    if is_mx_fp(data_type):
        return "mxfp4-pack-quantized" if bits == 4 else "mxfp8-quantized"
    if is_nv_fp(data_type):
        return "nvfp4-pack-quantized"
    return "float-quantized"


def _build_mixed_fp_quantization_config(
    scheme_groups,
    layer_config,
    ignore,
    global_bits,
    global_data_type,
    model,
    static_kv_dtype=None,
    static_attention_dtype=None,
):
    """Build a quantization config dict for mixed-precision scenarios.

    Creates multiple config groups with per-group format strings using compressed_tensors
    QuantizationScheme objects. Override groups (non-default schemes with specific layer
    targets) come first, and the default group (matching global bits/data_type with
    targets=["Linear"]) comes last. Top-level format is set to "mixed-precision".

    Args:
        scheme_groups: dict mapping (bits, data_type) -> list of layer names
        layer_config: per-layer quantization configs
        ignore: list of layers/patterns to ignore
        global_bits: global quantization bit-width
        global_data_type: global quantization data type
    Returns:
        quantization_config dict
    """
    global_key = (global_bits, global_data_type)

    # Override groups first, default group last
    override_groups = []
    default_group = None
    for key, layer_names in scheme_groups.items():
        if key == global_key:
            default_group = (key, layer_names)
        else:
            override_groups.append((key, layer_names))
    ordered = override_groups + ([default_group] if default_group else [])

    config_groups = {}
    group_formats = {}
    for idx, ((lbits, ldata_type), layer_names) in enumerate(ordered):
        group_name = f"group_{idx}"
        scheme = _get_scheme(lbits, ldata_type)
        tmp_quantization_config = initialize_quantization(scheme=scheme)
        tmp_quantization_config = tmp_quantization_config.config_groups["group_0"]
        is_default = (lbits, ldata_type) == global_key
        tmp_quantization_config.targets = ["Linear"] if is_default else layer_names
        config_groups[group_name] = tmp_quantization_config
        group_formats[group_name] = _get_group_format(lbits, ldata_type)

    use_fp8_attention = _use_fp8_attention(static_attention_dtype)
    if use_fp8_attention:
        _append_attention_group(config_groups, model)
    quantization_config = initialize_quantization(
        scheme=None,
        config_groups=config_groups,
        kv_cache_scheme=_construct_kv_scheme() if (_use_fp8_kv(static_kv_dtype) or use_fp8_attention) else None,
        ignore=ignore,
    )
    quantization_config = quantization_config.to_dict()

    # Set per-group format and top-level mixed-precision format
    quantization_config["format"] = "mixed-precision"
    for group_name, fmt in group_formats.items():
        quantization_config["config_groups"][group_name]["format"] = fmt
    quantization_config["provider"] = "auto-round"
    _configure_gaudi2_fp8_dtype(quantization_config)

    return quantization_config


def save_quantized_as_fp(
    output_dir: str,
    model: torch.nn.Module = None,
    tokenizer: Callable = None,
    layer_config: dict = None,
    inplace: bool = True,
    device: Union[str, torch.device] = "cpu",
    backend: str = None,
    serialization_dict: dict = None,
    **kwargs,
) -> torch.nn.Module:
    """
    Saves a quantized model of mxfp/nvfp data_type in the llm-compressor format.

    Args:
        output_dir (str): The directory where the quantized model will be saved.
        inplace (bool, optional): If True, modifies the model in place. Otherwise, creates a deepcopy of the model.
                                Default is True.
        backend (str, optional): The backend to be used for quantization.
                                  Default is "autoround:exllamav2".
        **kwargs: Additional keyword arguments including:
            - model (nn.Module): The model to be quantized.
            - layer_config (dict): The layer configuration for each layer.
            - serialization_dict (dict): The serialization configuration.
            - tokenizer (Tokenizer, optional): The tokenizer to be saved.

    Returns:
        None

    Raises:
        ValueError: If the backend is not supported.
    """
    bits = serialization_dict.get("bits", None)
    data_type = serialization_dict.get("data_type", None)
    act_bits = serialization_dict.get("act_bits", None)
    act_data_type = serialization_dict.get("act_data_type", None)
    safe_serialization = True if "safe_serialization" not in kwargs.keys() else kwargs["safe_serialization"]
    if not inplace:
        model = copy.deepcopy(model.to("cpu"))
    processor = kwargs.get("processor", None)
    regex_config = serialization_dict.pop("regex_config")
    extra_config = {}

    if act_bits <= 8:
        # revert WrapperWALayer for offline usage
        for n, m in model.named_modules():
            if isinstance(m, WrapperWALayer):
                orig_layer = m.orig_layer
                set_module(model, n, orig_layer)

    if is_nv_fp(act_data_type) and "static_gs" in str(act_data_type).lower():
        # generate static input_global_scale
        for n, m in model.named_modules():
            if type(m) in SUPPORTED_LAYER_TYPES:
                layer = m
                if hasattr(layer, "act_bits") and layer.act_bits < 8 and not getattr(layer, "input_global_scale", None):
                    assert hasattr(layer, "act_max")
                    from auto_round.data_type.nvfp import calculate_gparam

                    input_global_scale = calculate_gparam(layer.act_max, layer.group_size, model.device)
                    setattr(layer, "input_global_scale", input_global_scale)
                    delattr(layer, "act_max")
        # update fused input_global_scale
        from auto_round.data_type.utils import update_fused_layer_global_scales

        modules = list(model.modules())
        for module in tqdm(modules, desc="Update input global scale for fuse modules"):
            update_fused_layer_global_scales(module, base_name="input")

    names = list(layer_config.keys())
    if not unsupported_meta_device(model):
        for name in tqdm(names, desc="packing", leave=True):
            pack_layer(name, model, device)

    ignore = generate_ignore_regex_list(regex_config=regex_config, layer_config=layer_config)
    lm_head_name = get_lm_head_name(model)
    if lm_head_name and lm_head_name not in layer_config and lm_head_name not in ignore:
        ignore.append(lm_head_name)

    # get llm-compressor format config
    check_compressed_tensors_supported(raise_error=True)

    # Detect mixed precision by grouping quantized layers by (bits, data_type)
    scheme_groups = {}  # (bits, data_type) -> list of layer names
    for name, cfg in layer_config.items():
        layer_bits = cfg.get("bits", bits)
        layer_dt = cfg.get("data_type", data_type)
        if layer_bits > 8:
            continue
        key = (layer_bits, layer_dt)
        scheme_groups.setdefault(key, []).append(name)

    is_mixed = len(scheme_groups) > 1

    use_fp8_attention = _use_fp8_attention(serialization_dict.get("static_attention_dtype", None))
    kv_cache_scheme = (
        _construct_kv_scheme()
        if (_use_fp8_kv(serialization_dict.get("static_kv_dtype", None)) or use_fp8_attention)
        else None
    )

    if is_mixed:
        quantization_config = _build_mixed_fp_quantization_config(
            scheme_groups,
            layer_config,
            ignore,
            bits,
            data_type,
            model,
            static_kv_dtype=serialization_dict.get("static_kv_dtype", None),
            static_attention_dtype=serialization_dict.get("static_attention_dtype", None),
        )
    else:
        scheme = _get_scheme(bits, data_type)
        if scheme is None:
            raise ValueError(f"Unsupported combination of data_type={data_type} and bits={bits}.")

        format = _get_group_format(bits, data_type)
        quantization_config = initialize_quantization(
            scheme=scheme,
            kv_cache_scheme=kv_cache_scheme,
            ignore=ignore,
        )
        if use_fp8_attention:
            _append_attention_group(quantization_config.config_groups, model)
        setattr(quantization_config, "format", format)
        quantization_config = quantization_config.to_dict()
        quantization_config["provider"] = "auto-round"
        _configure_gaudi2_fp8_dtype(quantization_config)
    if hasattr(model, "config"):
        model.config.quantization_config = quantization_config
    if output_dir is None:
        return model

    if output_dir is None:
        model.tokenizer = tokenizer
        return model
    immediate_saving = is_immediate_saving_mode(model, serialization_dict)
    if os.path.exists(output_dir) and not immediate_saving:
        logger.warning(f"{output_dir} already exists, this may cause model conflict")
    save_pretrained_artifact(tokenizer, output_dir, artifact_name="tokenizer")

    if processor is not None:
        processor.save_pretrained(output_dir)

    dtype = None
    save_model(model, output_dir, safe_serialization=safe_serialization, dtype=dtype, immediate_saving=immediate_saving)

    return model
