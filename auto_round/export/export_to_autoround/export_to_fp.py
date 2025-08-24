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

from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    check_start_with_block_name,
    check_to_quantized,
    filter_quantization_config,
    get_module,
    is_mx_fp,
    is_nv_fp,
    logger,
    set_amax_for_all_moe_layers,
    set_module,
)
from auto_round.wrapper import WrapperWALayer

from .qlinear_fp import QuantLinear
from .utils import check_neq_config

__all__ = [
    "pack_layer",
    "save_quantized_as_fp",
]


def pack_layer(name, model, backend):
    if name == "lm_head":  # TODO: Check vLLM inference status to determine whether to enable this feature
        return
    layer = get_module(model, name)

    if not isinstance(layer, SUPPORTED_LAYER_TYPES) and not isinstance(layer, WrapperWALayer):  ##already packed
        return

    if isinstance(layer, WrapperWALayer):  # revert WrapperWALayer for offline usage
        wp_layer = layer
        layer = wp_layer.orig_layer
        set_module(model, name, layer)

    data_type = layer.data_type
    act_bits = layer.act_bits
    act_data_type = layer.act_data_type
    bits = layer.bits
    if bits > 8:
        return
    group_size = layer.group_size
    sym = layer.sym
    device = layer.weight.device

    if is_nv_fp(act_data_type) and act_bits <= 8:
        if not getattr(layer, "input_global_scale", None):
            assert hasattr(layer, "act_max")
            from auto_round.data_type.nvfp import calculate_gparam

            input_global_scale = calculate_gparam(layer.act_max, layer.group_size, "cpu")  # , model.device
            setattr(layer, "input_global_scale", input_global_scale)
            delattr(layer, "act_max")

    # QuantLinear = get_fp_qlinear(backend, bits, group_size, sym)

    if isinstance(layer, nn.Linear):
        in_features = layer.in_features
        out_features = layer.out_features
    elif isinstance(layer, nn.Conv2d):
        in_features = layer.in_channels
        out_features = layer.out_channels
    elif isinstance(layer, transformers.pytorch_utils.Conv1D):
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

    new_layer.device = device
    set_module(model, name, new_layer)
    qlayer = new_layer
    scale = layer.scale
    global_scale = getattr(layer, "weight_global_scale", None)
    input_global_scale = getattr(layer, "input_global_scale", None)
    # zero = layer.zp
    # so far can only pack layer on CPU
    qlayer.to("cpu")
    layer, scale = layer.to("cpu"), scale.to("cpu")
    qlayer.pack(layer, scale, global_scale=global_scale, input_global_scale=input_global_scale)
    ## no zeros to handle, as mxfp not support asym quantization
    qlayer.to(device)


def save_quantized_as_fp(output_dir, inplace=True, **kwargs):
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
    quantization_config = kwargs["serialization_dict"]
    quantization_config["block_name_to_quantize"] = quantization_config.pop("to_quant_block_names", None)
    quantization_config["quant_method"] = "auto-round"
    quantization_config["packing_format"] = backend
    quantization_config["scale_format"] = ("e8m0",)
    quantization_config["scale_calculation_mode"] = ("even",)

    tokenizer = kwargs.get("tokenizer", None)
    processor = kwargs.get("processor", None)
    extra_config = {}

    if act_bits <= 8:
        # revert WrapperWALayer for offline usage
        for n, m in model.named_modules():
            if isinstance(m, WrapperWALayer):
                orig_layer = m.orig_layer
                if not getattr(orig_layer, "input_global_scale", None):
                    assert hasattr(orig_layer, "act_max")
                    from auto_round.data_type.nvfp import calculate_gparam

                    input_global_scale = calculate_gparam(orig_layer.act_max, orig_layer.group_size, model.device)
                    setattr(orig_layer, "input_global_scale", input_global_scale)
                    delattr(orig_layer, "act_max")
                set_module(model, n, orig_layer)

        # update input_global_scale
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

    for layer_name in layer_config:
        if (
            not layer_config[layer_name]["in_blocks"] and layer_config[layer_name]["bits"] <= 8
        ):  ##lm head ##TODO fix act and so on
            extra_config[layer_name] = {}
            extra_config[layer_name]["bits"] = layer_config[layer_name]["bits"]
            extra_config[layer_name]["data_type"] = layer_config[layer_name]["data_type"]
            extra_config[layer_name]["group_size"] = layer_config[layer_name]["group_size"]
            extra_config[layer_name]["sym"] = layer_config[layer_name]["sym"]
        elif layer_config[layer_name]["in_blocks"] or (
            block_name_to_quantize is not None and check_start_with_block_name(layer_name, block_name_to_quantize)
        ):
            neq_keys = check_neq_config(
                layer_config[layer_name],
                data_type=quantization_config["data_type"],
                bits=quantization_config["bits"],
                act_bits=quantization_config["act_bits"],
                group_size=quantization_config["group_size"],
                sym=quantization_config["sym"],
            )
            if len(neq_keys) > 0:
                extra_config[layer_name] = {}
            for key in neq_keys:
                if layer_config[layer_name][key] is not None:
                    extra_config[layer_name][key] = layer_config[layer_name][key]
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
                    pack_layer(name, model, backend)
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

    dtype = None
    save(model, output_dir, safe_serialization=safe_serialization, dtype=dtype)

    return model


def save(model: nn.Module, save_dir: str, max_shard_size: str = "5GB", safe_serialization: bool = True, dtype=None):
    """Save model state dict and configs.

    Args:
        model (`nn.Module`):
            Model to be saved. The model can be wrapped or unwrapped.
        save_dir (`str`):
            Directory to which to save. Will be created if it doesn't exist.
        max_shard_size (`str`, defaults to `"10GB"`):
            The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
            lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).
            <Tip warning={true}>

            If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
            which will be bigger than `max_shard_size`.

            </Tip>
        safe_serialization (`bool`, defaults to `True`):
            Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
    """
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir, max_shard_size=max_shard_size, safe_serialization=safe_serialization)
    config_path = os.path.join(save_dir, "config.json")
    if dtype is not None and dtype != model.dtype and os.path.exists(os.path.join(save_dir, "config.json")):
        with open(config_path, "r") as file:
            data = json.load(file)
        data["torch_dtype"] = str(dtype).split(".")[-1]
        with open(config_path, "w") as file:
            json.dump(data, file, indent=2)
    config_file = "quantization_config.json"
    if hasattr(model, "config") and hasattr(model.config, "quantization_config"):
        with open(os.path.join(save_dir, config_file), "w", encoding="utf-8") as f:
            json.dump(model.config.quantization_config, f, indent=2)
