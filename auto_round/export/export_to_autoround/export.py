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
import inspect
import json
import os
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import threadpoolctl as tctl
import torch
import torch.nn as nn
import transformers
from tqdm import tqdm

from auto_round.export.export_to_autoround.utils import REQUIRED_CONFIG_KEYS, check_neq_config
from auto_round.logger import logger
from auto_round.utils import (
    SUPPORTED_FORMATS,
    SUPPORTED_LAYER_TYPES,
    check_start_with_block_name,
    check_to_quantized,
    copy_python_files_from_model_cache,
    filter_quantization_config,
    get_autogptq_packing_qlinear,
    get_module,
    is_mx_fp,
    is_nv_fp,
    is_standard_fp,
    set_module,
    to_standard_regex,
)


class AutoRoundFormat(str, Enum):
    # Weight: FP8, per-channel, may be extended to per-tensor in future
    # Activation: FP8, per-tensor
    TORCH_FP8_STATIC = "fp8_static"


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
        ValueError: If the backend is not supported.
    """
    if "auto_round" in backend and "awq" not in backend and "gptq" not in backend:
        if act_bits <= 8:  ##easily have bug for other configuration, need to refine code later
            import auto_round.export.export_to_autoround.qlinear_triton_act

            return auto_round.export.export_to_autoround.qlinear_triton_act.QuantLinear
        from auto_round_extension.torch.qlinear_torch import QuantLinear

        return QuantLinear
    elif "gptqmodel" in backend:
        from auto_round_extension.torch.qlinear_torch import QuantLinear

        return QuantLinear
    elif "auto_round" in backend and "gptq" in backend and "gptqmodel" not in backend:
        from auto_round_extension.torch.qlinear_torch_zp import QuantLinear

        return QuantLinear
    elif "awq" in backend:
        from ..export_to_awq.utils import WQLinear_GEMM

        return WQLinear_GEMM
    elif "gptq" in backend and "gptqmodel" not in backend:  ## have g_idx
        return get_autogptq_packing_qlinear(backend, bits, group_size, sym)
    else:
        raise ValueError(f"unsupported backend: '{backend}'. Supported backends are: {', '.join(SUPPORTED_FORMATS)}")


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
    import auto_round.export.export_to_autoround.qlinear_triton_act

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

    qlayer.pack(layer, scale, zp, act_scale, w_bf16_to_fp8_scale, device)
    qlayer.to(device)


def pack_layer(layer_name, model, backend, device=None):
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
    if is_nv_fp(backend) or is_mx_fp(backend):
        from auto_round.export.export_to_autoround.export_to_nvfp_mxfp import pack_layer

        return pack_layer(layer_name, model, backend, device)

    if backend == "auto_round:fp8" or backend == f"auto_round:{AutoRoundFormat.TORCH_FP8_STATIC.value}":
        from auto_round.export.export_to_autoround.export_to_fp8 import pack_layer

        return pack_layer(layer_name, model, backend, device)

    layer = get_module(model, layer_name)
    if hasattr(layer, "orig_layer"):
        layer = layer.orig_layer

    if not isinstance(layer, SUPPORTED_LAYER_TYPES):  ##already packed
        return

    if int(layer.act_bits) <= 8:
        return pack_qact_layer(layer_name, model)

    if not check_to_quantized(layer):
        return

    orig_device = layer.weight.device
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
            bits, group_size, in_features, out_features, bias=bias, weight_dtype=layer.weight.dtype
        )
        new_layer.device = orig_device
        set_module(model, layer_name, new_layer)
        qlayer = new_layer
        import auto_round_extension.torch.qlinear_torch

        if (
            sym
            and isinstance(zp, torch.Tensor)
            and isinstance(QuantLinear, (auto_round_extension.torch.qlinear_torch.QuantLinear))
        ):
            zp = int(zp.flatten()[0])

        qlayer.to("cpu")
        ##force to float32 to be compatible with torch 2.0
        sig = inspect.signature(qlayer.pack)
        param_count = len(sig.parameters)
        if param_count == 2:
            qlayer.pack(layer, scale, device=device)
        else:
            qlayer.pack(layer, scale, zp, None, device=device)
        qlayer.to(device)
    else:
        scale = scale.to(torch.float32).t().contiguous()
        if isinstance(zp, torch.Tensor):
            zp = zp.to(torch.float32).t().contiguous()
            if sym:
                zp = int(zp.flatten()[0])

        if bits != 4:
            logger.error("AutoAWQ format only supports 4-bits quantization.")
        qlayer = QuantLinear.from_linear(
            linear=layer, w_bit=bits, group_size=group_size, init_only=False, scales=scale, zeros=zp, device=device
        )
        qlayer.to(orig_device)
        set_module(model, layer_name, qlayer)
    if hasattr(layer, "weight"):
        layer.weight = None
    if hasattr(layer, "bias"):
        layer.bias = None


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
        ValueError: If the backend is not supported.
    """
    data_type = kwargs.get("data_type", None)
    if is_nv_fp(data_type) or is_mx_fp(data_type):  ## detect nvfp & mxfp first
        from auto_round.export.export_to_autoround.export_to_nvfp_mxfp import save_quantized_as_fp

        return save_quantized_as_fp(output_dir, inplace=inplace, backend="auto_round:llm_compressor", **kwargs)

    if kwargs.get("data_type", "int") == "fp" and kwargs.get("bits", 16) == 8 and kwargs.get("act_bits", 16) >= 16:
        from auto_round.export.export_to_autoround.export_to_fp8 import save_quantized_as_autoround

        return save_quantized_as_autoround(output_dir, inplace=inplace, backend="auto_round", **kwargs)
    from auto_round.autoround import AutoRoundFormat

    ##if using sym, we change to gptq sym kernel to avoid compiling from auto_round source
    if (
        (kwargs.get("sym") is None or kwargs.get("sym"))
        and ("gptq" not in backend and "awq" not in backend)
        and (AutoRoundFormat.TORCH_FP8_STATIC.value not in backend)
    ):
        backend = backend.replace("auto_round", "auto_round:auto_gptq")

    model = kwargs["model"]
    safe_serialization = True if "safe_serialization" not in kwargs.keys() else kwargs["safe_serialization"]
    if not inplace:
        model = copy.deepcopy(model.to("cpu"))

    layer_config = kwargs["layer_config"]
    quantization_config = kwargs["serialization_dict"]
    quantization_config["block_name_to_quantize"] = quantization_config.pop("to_quant_block_names", None)
    quantization_config["quant_method"] = "auto-round"
    quantization_config["packing_format"] = backend
    device = kwargs.get("device", None)
    tokenizer = kwargs.get("tokenizer", None)
    processor = kwargs.get("processor", None)
    image_processor = kwargs.get("image_processor", None)
    extra_config = {}
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
            extra_config[layer_name]["act_bits"] = layer_config[layer_name]["act_bits"]
            extra_config[layer_name]["act_data_type"] = layer_config[layer_name]["act_data_type"]
            extra_config[layer_name]["act_group_size"] = layer_config[layer_name]["act_group_size"]
            extra_config[layer_name]["act_sym"] = layer_config[layer_name]["act_sym"]
        elif layer_config[layer_name]["in_blocks"] or (
            block_name_to_quantize is not None and check_start_with_block_name(layer_name, block_name_to_quantize)
        ):
            neq_keys = check_neq_config(
                layer_config[layer_name], **{k: quantization_config[k] for k in REQUIRED_CONFIG_KEYS}
            )
            if len(neq_keys) > 0:
                extra_config[layer_name] = {}
            for key in neq_keys:
                if layer_config[layer_name][key] is not None:
                    extra_config[layer_name][key] = layer_config[layer_name][key]

    dynamic_config = quantization_config.pop("dynamic_config")
    if dynamic_config is not None:
        for name in dynamic_config.keys():
            regex_name = to_standard_regex(name)
            extra_config[regex_name] = {**{k: dynamic_config[name][k] for k in REQUIRED_CONFIG_KEYS}}

    if len(extra_config) > 0:
        quantization_config["extra_config"] = extra_config
    names = list(layer_config.keys())
    max_workers = 1
    if not torch.cuda.is_available() and not torch.xpu.is_available():
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
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(output_dir)

    if processor is not None:
        processor.save_pretrained(output_dir)
    if image_processor is not None:
        image_processor.save_pretrained(output_dir)
    if quantization_config.get("act_bits", 16) <= 8:
        dtype = torch.bfloat16
    elif "awq" in quantization_config.get("packing_format", "auto_round:auto_gptq"):
        dtype = torch.float16  ## awq kernel only supports float16 on cuda
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
    try:
        model.save_pretrained(save_dir, max_shard_size=max_shard_size, safe_serialization=safe_serialization)
    except ValueError as e:
        if hasattr(model, "generation_config"):
            setattr(model.generation_config, "do_sample", True)
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

    try:
        copy_python_files_from_model_cache(model, save_dir)
    except Exception as e:
        logger.warning("Skipping source model Python file copy due to error: %s", e)
