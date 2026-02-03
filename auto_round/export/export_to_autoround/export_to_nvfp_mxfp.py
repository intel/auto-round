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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from typing import Callable, Union

import threadpoolctl as tctl
import torch
import torch.nn as nn
import transformers
from tqdm import tqdm

from auto_round.compressors.utils import is_mx_fp, is_nv_fp
from auto_round.export.export_to_autoround.utils import check_neq_config
from auto_round.export.utils import filter_quantization_config, release_layer_safely, save_model
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme
from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    check_start_with_block_name,
    check_to_quantized,
    copy_python_files_from_model_cache,
    get_module,
    get_packing_device,
    set_amax_for_all_moe_layers,
    set_module,
    to_standard_regex,
)
from auto_round.wrapper import WrapperWALayer

from .qlinear_fp import QuantLinear

__all__ = [
    "pack_layer",
    "save_quantized_as_fp",
]


def pack_layer(name, model, backend, device=None):
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
            input_global_scale = calculate_gparam(layer.act_max, layer.group_size, "cpu")
            setattr(layer, "input_global_scale", input_global_scale)
            delattr(layer, "act_max")

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
    ## no zeros to handle, as mxfp/nvfp do not support asym quantization
    # zero = layer.zp
    qlayer.pack(layer, scale, global_scale=global_scale, input_global_scale=input_global_scale, device=device)
    qlayer.to(orig_device)
    set_module(model, name, qlayer)
    # Note: release weight and bias explicitly, in case they are referenced elsewhere
    release_layer_safely(layer)


def save_quantized_as_fp(
    output_dir: str,
    model: torch.nn.Module = None,
    tokenizer: Callable = None,
    layer_config: dict = None,
    inplace: bool = True,
    device: Union[str, torch.device] = "cpu",
    backend: str = "autoround:exllamav2",
    serialization_dict: dict = None,
    **kwargs,
) -> torch.nn.Module:
    """
    Saves a quantized model of mxfp/nvfp data_type in the auto-round format.

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
    quantization_config = serialization_dict
    quantization_config["block_name_to_quantize"] = quantization_config.pop("to_quant_block_names", None)
    quantization_config["quant_method"] = "auto-round"
    quantization_config["packing_format"] = backend

    processor = kwargs.get("processor", None)
    image_processor = kwargs.get("image_processor", None)
    extra_config = {}

    if act_bits <= 8:
        # revert WrapperWALayer for offline usage
        for n, m in model.named_modules():
            if isinstance(m, WrapperWALayer):
                orig_layer = m.orig_layer
                set_module(model, n, orig_layer)

    if is_nv_fp(act_data_type) and "static_gs" in str(act_data_type).lower():
        # Ensure all MOE layers have act_max set (needed after deep copy or for uncalibrated layers)
        from auto_round.utils.model import is_moe_model, set_amax_for_all_moe_layers

        if is_moe_model(model):
            set_amax_for_all_moe_layers(model)

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

    block_name_to_quantize = quantization_config["block_name_to_quantize"]
    if isinstance(block_name_to_quantize, str):
        block_name_to_quantize = block_name_to_quantize.split(",")
    elif isinstance(block_name_to_quantize, list):
        for i in range(len(block_name_to_quantize)):
            block_name_to_quantize[i] = os.path.commonprefix(block_name_to_quantize[i]).rstrip(".")

    scheme_keys = [f.name for f in fields(QuantizationScheme)]
    for layer_name, cfg in layer_config.items():
        if not cfg["in_blocks"] and cfg["bits"] <= 8:  # lm head
            extra_config[layer_name] = {key: cfg.get(key) for key in scheme_keys}
        elif cfg["in_blocks"] or (
            block_name_to_quantize is not None and check_start_with_block_name(layer_name, block_name_to_quantize)
        ):
            neq_keys = check_neq_config(cfg, **{k: quantization_config[k] for k in scheme_keys})
            if len(neq_keys) > 0:
                extra_config[layer_name] = {}
                for key in neq_keys:
                    if cfg.get(key, None) is not None:
                        extra_config[layer_name][key] = cfg.get(key, None)

    regex_config = quantization_config.pop("regex_config")
    if regex_config is not None:
        for name, cfg in regex_config.items():
            regex_name = to_standard_regex(name)
            neq_keys = check_neq_config(cfg, **{k: quantization_config[k] for k in scheme_keys})
            if len(neq_keys) > 0:
                extra_config[regex_name] = {}
                for key in neq_keys:
                    if cfg.get(key) is not None:
                        extra_config[regex_name][key] = cfg[key]

    if len(extra_config) > 0:
        quantization_config["extra_config"] = extra_config
    names = list(layer_config.keys())
    max_workers = 1
    if not torch.cuda.is_available() or not torch.xpu.is_available():
        max_workers = 2  ## 2 with cuda packing will cause hang occasionally
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(names), leave=True) as pbar:

            def wrapper(name):
                pbar.set_description(f"packing {name}")
                with tctl.threadpool_limits(limits=1):
                    pack_layer(name, model, backend, device)
                pbar.update(1)

            for _ in executor.map(wrapper, names):
                pass
    filter_quantization_config(quantization_config)

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
    if image_processor is not None:
        image_processor.save_pretrained(output_dir)

    dtype = None
    save_model(model, output_dir, safe_serialization=safe_serialization, dtype=dtype)

    return model
