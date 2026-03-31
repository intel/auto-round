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
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Union

import threadpoolctl as tctl
import torch
import torch.nn as nn
import transformers
from tqdm import tqdm

from auto_round.compressors.utils import is_mx_fp, is_nv_fp
from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear
from auto_round.export.export_to_llmcompressor.utils import generate_ignore_regex_list
from auto_round.export.utils import filter_quantization_config, release_layer_safely, save_model
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

__all__ = [
    "pack_layer",
    "save_quantized_as_fp",
]


def pack_layer(name, model, device=None):
    layer = get_module(model, name)
    if type(layer) not in SUPPORTED_LAYER_TYPES and not isinstance(layer, WrapperWALayer):  ##already packed
        return

    if isinstance(layer, WrapperWALayer):  # revert WrapperWALayer for offline usage
        wp_layer = layer
        layer = wp_layer.orig_layer
        set_module(model, name, layer)

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
    set_module(model, name, qlayer)
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


def _build_mixed_fp_quantization_config(scheme_groups, layer_config, ignore, global_bits, global_data_type):
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
    from compressed_tensors.quantization import (
        QuantizationArgs,
    )
    from compressed_tensors.quantization import QuantizationScheme as CTScheme  # pylint: disable=E0401

    from .config import initialize_quantization
    from .export import _get_act_scheme_strategy, _get_weight_scheme_strategy

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

    quantization_config = initialize_quantization(scheme=None, config_groups=config_groups, ignore=ignore)
    quantization_config = quantization_config.to_dict()

    # Set per-group format and top-level mixed-precision format
    quantization_config["format"] = "mixed-precision"
    for group_name, fmt in group_formats.items():
        quantization_config["config_groups"][group_name]["format"] = fmt
    quantization_config["provider"] = "auto-round"

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
    max_workers = 1
    if not torch.cuda.is_available() or not torch.xpu.is_available():
        max_workers = 2  ## 2 with cuda packing will cause hang occasionally
    if not unsupported_meta_device(model):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(total=len(names), leave=True) as pbar:

                def wrapper(name):
                    pbar.set_description(f"packing {name}")
                    with tctl.threadpool_limits(limits=1):
                        pack_layer(name, model, device)
                    pbar.update(1)

                for _ in executor.map(wrapper, names):
                    pass

    ignore = generate_ignore_regex_list(regex_config=regex_config, layer_config=layer_config)
    lm_head_name = get_lm_head_name(model)
    if lm_head_name not in layer_config:
        ignore.append(lm_head_name)

    # get llm-compressor format config
    check_compressed_tensors_supported()
    from .config import initialize_quantization

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

    if is_mixed:
        quantization_config = _build_mixed_fp_quantization_config(scheme_groups, layer_config, ignore, bits, data_type)
    else:
        scheme = _get_scheme(bits, data_type)
        if scheme is None:
            logger.error(f"Got unpexcted {data_type} with bits {bits}, please check.")
            sys.exit(-1)

        format = _get_group_format(bits, data_type)
        quantization_config = initialize_quantization(scheme=scheme)
        setattr(quantization_config, "format", format)
        setattr(quantization_config, "ignore", ignore)
        quantization_config = quantization_config.to_dict()
        quantization_config["provider"] = "auto-round"
    if hasattr(model, "config"):
        model.config.quantization_config = quantization_config
    if output_dir is None:
        return model

    if output_dir is None:
        model.tokenizer = tokenizer
        return model
    if os.path.exists(output_dir):
        logger.warning(f"{output_dir} already exists, this may cause model conflict")
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

    if processor is not None:
        processor.save_pretrained(output_dir)

    dtype = None
    save_model(model, output_dir, safe_serialization=safe_serialization, dtype=dtype)

    return model
