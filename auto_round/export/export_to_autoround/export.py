# Copyright (c) 2024 Intel Corporation
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
import json
import os

import torch
import torch.nn as nn
import transformers

import auto_round.export.export_to_autoround.qlinear_triton_act

import auto_round_extension.cuda.qlinear_tritonv2
from auto_round.utils import get_module, logger, set_module, supported_layer_types, check_to_quantized
import threadpoolctl as tctl
import inspect
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from auto_round.utils import get_autogptq_packing_qlinear


def check_neq_config(config, data_type, bits, group_size, sym):
    """
    Checks if the provided configuration parameters are not equal to the values in the config dictionary.

    Args:
        config (dict): A dictionary containing the configuration parameters.
        data_type (str): The expected data type.
        bits (int): The expected number of bits.
        group_size (int): The expected group size.
        sym (bool): The expected symmetry flag.

    Returns:
        list: A list of strings indicating which configuration parameters do not match.
    """
    expected_config = {"data_type": data_type,
                       "bits": bits,
                       "group_size": group_size,
                       "sym": sym
                       }
    return [key for key, expected_value in expected_config.items() if config.get(key) != expected_value]


def dynamic_import_quant_linear_for_packing(backend, bits, group_size, sym, act_bits=16):
    """
    Dynamically imports and returns the appropriate QuantLinear class based on the specified backend and parameters.

    Args:
        backend (str): The backend to be used for quantization. Supported values include "auto_round" "awq" and "gptq".
        bits (int): The number of bits for quantization.
        group_size (int): The group size for quantization.
        sym (bool): Flag indicating whether to use symmetric quantization.

    Returns:
        class: The dynamically imported QuantLinear class configured according to the specified parameters.

    Raises:
        AssertionError: If the backend is not supported.
    """
    if "auto_round" in backend and "awq" not in backend and "gptq" not in backend:
        if act_bits <= 8:  ##easily have bug for other configuration, need to refine code later
            return auto_round.export.export_to_autoround.qlinear_triton_act.QuantLinear

        from auto_round_extension.cuda.qlinear_tritonv2 import QuantLinear
        return QuantLinear
    elif "auto_round" in backend and "gptq" in backend and bits in (2, 4, 8):
        from auto_round.export.export_to_autoround.qlinear_triton import QuantLinear  ##no g_idx
        return QuantLinear
    elif "awq" in backend:
        from ..export_to_awq.utils import WQLinear_GEMM
        return WQLinear_GEMM
    elif "gptqmodel" in backend:
        return auto_round_extension.cuda.qlinear_tritonv2.QuantLinear
    elif "gptq" in backend and not "gptqmodel" in backend:  ## have g_idx
        return get_autogptq_packing_qlinear(backend, bits, group_size, sym)
    else:
        assert False, f"only support auto_gptq, auto_awq and auto_round backend"


def pack_qact_layer(name, model):
    layer = get_module(model, name)
    if hasattr(layer, "orig_layer"):
        layer = layer.orig_layer

    if layer.bits > 8:
        return

    device = layer.weight.device
    bits = layer.bits
    group_size = layer.group_size
    act_bits = layer.act_bits

    act_scale = layer.act_scale if hasattr(layer, "act_scale") else None
    w_bf16_to_fp8_scale = layer.w_bf16_to_fp8_scale if hasattr(layer, "w_bf16_to_fp8_scale") else None
    scale = layer.scale
    zp = layer.zp
    QuantLinear = auto_round.export.export_to_autoround.qlinear_triton_act.QuantLinear

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
    use_pc = False
    new_layer = QuantLinear(  ##pylint: disable=E1123
        bits, group_size, in_features, out_features, bias, weight_dtype=layer.weight.dtype, use_pc=use_pc
    )
    new_layer.device = device
    set_module(model, name, new_layer)
    qlayer = new_layer

    qlayer.to("cpu")

    qlayer.pack(layer, scale, zp, act_scale, w_bf16_to_fp8_scale)
    qlayer.to(device)


def pack_layer(layer_name, model, backend):
    """
     Packs a model layer for quantization based on its type and configuration.

    This function retrieves the specified layer from the model, checks its
    compatibility for quantization, and replaces it with a quantized version
    if applicable. The quantization process depends on the layer's bit-width,
    group size, symmetry, and activation bits.

    Args:
        layer_name (str): The name of the layer to be packed.
        model (torch.nn.Module): The model containing the layer.
        backend (str): The backend framework to be used for quantization.

    Returns:
        None: The function modifies the model in place.
    """
    layer = get_module(model, layer_name)
    if hasattr(layer, "orig_layer"):
        layer = layer.orig_layer

    if not isinstance(layer, supported_layer_types):  ##already packed
        return

    if int(layer.act_bits) <= 8:
        return pack_qact_layer(layer_name, model)

    if not check_to_quantized(layer):
        return

    device = layer.weight.device
    bits = layer.bits
    group_size = layer.group_size
    sym = layer.sym
    act_bits = layer.act_bits

    scale = layer.scale
    zp = layer.zp
    QuantLinear = dynamic_import_quant_linear_for_packing(backend, bits, group_size, sym, act_bits)

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

    if "awq" not in backend:
        new_layer = QuantLinear(  ##pylint: disable=E1123
            bits, group_size, in_features, out_features, bias, weight_dtype=layer.weight.dtype
        )
        new_layer.device = device
        set_module(model, layer_name, new_layer)
        qlayer = new_layer
        import auto_round.export.export_to_autoround.qlinear_triton
        if sym and isinstance(QuantLinear, (auto_round.export.export_to_autoround.qlinear_triton.QuantLinear,
                                            auto_round_extension.cuda.qlinear_tritonv2.QuantLinear)):
            zp = int(zp.flatten()[0])

        qlayer.to("cpu")
        ##force to float32 to be compatible with torch 2.0
        sig = inspect.signature(qlayer.pack)
        param_count = len(sig.parameters)
        if param_count == 2:
            qlayer.pack(layer, scale)
        else:
            qlayer.pack(layer, scale, zp, None)
        qlayer.to(device)
    else:
        scale, zp = scale.to(torch.float32), zp.to(torch.float32)
        scale = scale.t().contiguous()
        zp = zp.t().contiguous()
        if sym:
            zp = int(zp.flatten()[0])

        if bits != 4:
            logger.error("AutoAWQ format only supports 4-bits quantization.")
        qlayer = QuantLinear.from_linear(
            linear=layer,
            w_bit=bits,
            group_size=group_size,
            init_only=False,
            scales=scale,
            zeros=zp,
        )
        qlayer.to(device)
        set_module(model, layer_name, qlayer)


def save_quantized_as_autoround(output_dir, inplace=True, backend="auto_round:exllamav2", **kwargs):
    """
    Saves a quantized model in the auto-round format.

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
        AssertionError: If the backend is not supported.
    """

    ##if using sym, we change to gptq sym kernel to avoid compiling from auto_round source
    if (kwargs.get("sym") is None or kwargs.get("sym") == True) and ("gptq" not in backend and "awq" not in backend):
        backend = backend.replace('auto_round', 'auto_round:auto_gptq')

    model = kwargs["model"]
    safe_serialization = True if 'safe_serialization' not in kwargs.keys() else kwargs["safe_serialization"]
    if not inplace:
        model = copy.deepcopy(model.to("cpu"))

    layer_config = kwargs["layer_config"]
    quantization_config = kwargs["serialization_dict"]
    quantization_config["quant_method"] = "auto-round"
    if quantization_config["bits"] == 3:
        backend = "auto_round:auto_gptq"
    quantization_config["packing_format"] = backend

    tokenizer = kwargs.get("tokenizer", None)
    processor = kwargs.get("processor", None)
    extra_config = {}
    for layer_name in layer_config:
        if not layer_config[layer_name]["in_blocks"] and layer_config[layer_name][
            "bits"] <= 8:  ##lm head ##TODO fix act and so on
            extra_config[layer_name] = {}
            extra_config[layer_name]["bits"] = layer_config[layer_name]["bits"]
            extra_config[layer_name]["data_type"] = layer_config[layer_name]["data_type"]
            extra_config[layer_name]["group_size"] = layer_config[layer_name]["group_size"]
            extra_config[layer_name]["sym"] = layer_config[layer_name]["sym"]
        elif layer_config[layer_name]["in_blocks"]:
            neq_keys = check_neq_config(
                layer_config[layer_name],
                data_type=quantization_config["data_type"],
                bits=quantization_config["bits"],
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
    if not torch.cuda.is_available():
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
    if quantization_config.get("act_bits", 16) <= 8:
        dtype = torch.bfloat16
    else:
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
        with open(config_path, 'r') as file:
            data = json.load(file)
        data["torch_dtype"] = str(dtype).split(".")[-1]
        with open(config_path, 'w') as file:
            json.dump(data, file, indent=2)
    config_file = "quantization_config.json"
    if hasattr(model, "config") and hasattr(model.config, "quantization_config"):
        with open(os.path.join(save_dir, config_file), "w", encoding="utf-8") as f:
            json.dump(model.config.quantization_config, f, indent=2)
