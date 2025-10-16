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

import threadpoolctl as tctl
import torch
import torch.nn as nn
import transformers
from tqdm import tqdm

from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear
from auto_round.export.export_to_llmcompressor.utils import generate_ignore_regex_list
from auto_round.export.utils import save_model
from auto_round.logger import logger
from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    check_start_with_block_name,
    check_to_quantized,
    copy_python_files_from_model_cache,
    filter_quantization_config,
    get_block_names,
    get_module,
    is_mx_fp,
    is_nv_fp,
    set_amax_for_all_moe_layers,
    set_module,
)
from auto_round.wrapper import WrapperWALayer

from .config import check_compressed_tensors_supported

__all__ = [
    "pack_layer",
    "save_quantized_as_fp",
]


def pack_layer(name, model, backend, device=None):
    if name == "lm_head":  # TODO: Check vLLM inference status to determine whether to enable this feature
        return
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
        if not getattr(layer, "input_global_scale", None):
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
    new_layer = QuantLinear(  ##pylint: disable=E1123
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

    new_layer.device = orig_device
    set_module(model, name, new_layer)
    qlayer = new_layer
    scale = layer.scale
    global_scale = getattr(layer, "weight_global_scale", None)
    input_global_scale = getattr(layer, "input_global_scale", None)
    # zero = layer.zp # no zeros to handle, as mxfp not support asym quantization
    qlayer.pack(layer, scale, global_scale=global_scale, input_global_scale=input_global_scale, device=device)
    qlayer.to(orig_device)


def save_quantized_as_fp(output_dir, inplace=True, **kwargs):
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
    model = kwargs["model"]
    backend = kwargs.get("backend", None)
    bits = kwargs.get("bits", None)
    data_type = kwargs.get("data_type", None)
    act_bits = kwargs.get("act_bits", None)
    act_data_type = kwargs.get("act_data_type", None)
    safe_serialization = True if "safe_serialization" not in kwargs.keys() else kwargs["safe_serialization"]
    if not inplace:
        model = copy.deepcopy(model.to("cpu"))
    layer_config = kwargs["layer_config"]
    device = kwargs.get("device", None)
    tokenizer = kwargs.get("tokenizer", None)
    processor = kwargs.get("processor", None)
    ar_quantization_config = kwargs["serialization_dict"]
    regex_config = ar_quantization_config.pop("regex_config")
    layer_config = kwargs["layer_config"]
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
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(names), leave=True) as pbar:

            def wrapper(name):
                pbar.set_description(f"packing {name}")
                with tctl.threadpool_limits(limits=1):
                    pack_layer(name, model, backend, device)
                pbar.update(1)

            for _ in executor.map(wrapper, names):
                pass

    ignore = generate_ignore_regex_list(regex_config=regex_config, layer_config=layer_config)

    # get llm-compressor format config
    check_compressed_tensors_supported()
    from .config import initialize_quantization

    scheme = "NVFP4"
    quantization_config = initialize_quantization(scheme=scheme)
    setattr(quantization_config, "format", "nvfp4-pack-quantized")
    setattr(quantization_config, "ignore", ignore)
    quantization_config = quantization_config.to_dict()
    quantization_config["provider"] = "auto-round"
    if is_mx_fp(data_type):  # Manually replace some parameters, as compressed-tensor is not support MXFP scheme yet
        quantization_config["config_groups"]["group_0"]["weights"]["num_bits"] = bits
        quantization_config["config_groups"]["group_0"]["input_activations"]["num_bits"] = (
            act_bits if act_bits <= 8 else bits
        )
        quantization_config["config_groups"]["group_0"]["weights"]["is_mx"] = True
        quantization_config["config_groups"]["group_0"]["input_activations"]["is_mx"] = True
        quantization_config["config_groups"]["group_0"]["weights"]["group_size"] = 32
        quantization_config["config_groups"]["group_0"]["input_activations"]["group_size"] = 32
        quantization_config["format"] = "float-quantized"
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
