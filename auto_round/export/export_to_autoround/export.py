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

from auto_round.export.register import register_format
from auto_round.utils import get_layer_names_in_block, get_module, logger, set_module
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


def dynamic_import_quantLinear_for_packing(backend, bits, group_size, sym):
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
        ##only support triton and exllamav2
        if not ("triton" in backend or "exllamav2" in backend):
            logger.warning_once(f"auto_round format does not support {backend}, try to pack each layer with auto_gptq")
            return get_autogptq_packing_qlinear(backend, bits, group_size, sym)
        from auto_round_extension.cuda.qlinear_triton import QuantLinear
        return QuantLinear
    elif "awq" in backend:
        from ..export_to_awq.utils import WQLinear_GEMM
        return WQLinear_GEMM
    elif "gptq" in backend:
        return get_autogptq_packing_qlinear(backend, bits, group_size, sym)
    else:
        assert False, f"only support auto_gptq, auto_awq and auto_round backend"


def pack_layer(name, model, layer_config, backend, pbar):
    with tctl.threadpool_limits(limits=1):
        pbar.set_description(f"packing {name}")
        config = layer_config[name]
        if config["bits"] > 8:
            pbar.update(1)
            return

        bits = config["bits"]
        group_size = config["group_size"]
        sym = config["sym"]

        layer = get_module(model, name)
        device = layer.weight.device

        QuantLinear = dynamic_import_quantLinear_for_packing(backend, bits, group_size, sym)

        if isinstance(layer, nn.Linear):
            in_features = layer.in_features
            out_features = layer.out_features
        elif isinstance(layer, nn.Conv2d):
            in_features = layer.in_channels
            out_features = layer.out_channels
        elif isinstance(layer, transformers.pytorch_utils.Conv1D):
            in_features = layer.weight.shape[0]
            out_features = layer.weight.shape[1]
        bias = layer.bias is not None and torch.any(layer.bias)

        if "awq" not in backend:
            new_layer = QuantLinear(  ##pylint: disable=E1123
                bits, group_size, in_features, out_features, bias, weight_dtype=layer.weight.dtype
            )
            new_layer.device = device
            set_module(model, name, new_layer)
            qlayer = new_layer
            scale = layer_config[name]["scale"]
            zero = layer_config[name]["zp"]
            # so far can only pack layer on CPU
            qlayer.to("cpu")
            ##force to float32 to be compatible with torch 2.0
            layer, scale, zero = layer.to("cpu"), scale.to("cpu"), zero.to("cpu").to(torch.float32)
            sig = inspect.signature(qlayer.pack)
            param_count = len(sig.parameters)
            if param_count == 2:
                qlayer.pack(layer, scale)
            else:
                qlayer.pack(layer, scale, zero, None)
            qlayer.to(device)
        else:
            from ..export_to_awq.utils import clear_memory
            scale, zp = layer_config[name]["scale"].to(torch.float32), layer_config[name]["zp"].to(torch.float32)
            scale = scale.t().contiguous()
            zp = zp.t().contiguous()
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
            set_module(model, name, qlayer)
            clear_memory()
        pbar.update(1)


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
    if ":" not in backend:
        backend = "auto_round:exllamav2"
    backend = backend.replace("autoround", "auto_round")
    backend = backend.replace("auto-round", "auto_round")
    ##if using sym, we change to gptq sym kernel to avoid compiling from auto_round source
    if (kwargs.get("sym") is None or kwargs.get("sym") == True) and ("gptq" not in backend and "awq" not in backend):
        backend = backend.replace('auto_round', 'auto_round:gptq')

    if not ("triton" in backend or "exllamav2" in backend or "awq" in backend or "gptq" in backend):
        logger.info(f"AutoRound format does not support {backend}, try to pack each layer with AutoGPTQ")
        backend = backend.replace("auto_round", "auto_gptq")

    model = kwargs["model"]
    to_quant_block_names = kwargs["to_quant_block_names"]
    safe_serialization = True if 'safe_serialization' not in kwargs.keys() else kwargs["safe_serialization"]
    if not inplace:
        model = copy.deepcopy(model.to("cpu"))
    layer_names_in_block = get_layer_names_in_block(model, to_quant_block_names=to_quant_block_names)

    layer_config = kwargs["layer_config"]
    quantization_config = kwargs["serialization_dict"]
    quantization_config["quant_method"] = "intel/auto-round"

    quantization_config["backend"] = backend
    tokenizer = kwargs.get("tokenizer", None)
    processor = kwargs.get("processor", None)
    extra_config = {}
    for layer_name in layer_config:
        if layer_name not in layer_names_in_block and layer_config[layer_name]["bits"] <= 8:  ##lm head
            extra_config[layer_name] = {}
            extra_config[layer_name]["bits"] = layer_config[layer_name]["bits"]
            extra_config[layer_name]["data_type"] = layer_config[layer_name]["data_type"]
            extra_config[layer_name]["group_size"] = layer_config[layer_name]["group_size"]
            extra_config[layer_name]["sym"] = layer_config[layer_name]["sym"]
        elif layer_name in layer_names_in_block:
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
    with ThreadPoolExecutor(max_workers=2) as executor:
        with tqdm(total=len(names), leave=True) as pbar:
            def wrapper(name):
                pack_layer(name, model, layer_config, backend, pbar)

            for _ in executor.map(wrapper, names):
                pass

    if hasattr(model, "config"):
        model.config.quantization_config = quantization_config
    if output_dir is None:
        return model
    
    if output_dir is None:
        model.tokenizer = tokenizer
        return model
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

    if processor is not None:
        processor.save_pretrained(output_dir)
    save(model, output_dir, safe_serialization=safe_serialization)

    return model


def save(model: nn.Module, save_dir: str, max_shard_size: str = "5GB", safe_serialization: bool = True):
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
    config_file = "quantization_config.json"
    if hasattr(model, "config") and hasattr(model.config, "quantization_config"):
        with open(os.path.join(save_dir, config_file), "w", encoding="utf-8") as f:
            json.dump(model.config.quantization_config, f, indent=2)


