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


def check_neq_config(config, data_type, bits, group_size, sym):
    res = []
    if data_type != config["data_type"]:
        res.append("data_type")
        return res
    if bits != config["bits"]:
        res.append("bits")
    if group_size != config["group_size"]:
        res.append("group_size")
    if sym != config["sym"]:
        res.append("sym")
    return res


def get_autogptq_backend_config(backend, bits=4):
    use_triton = False
    disable_exllamav2 = False
    disable_exllamav1 = False
    disable_marlin = True
    use_qigen = False
    if backend == "gptq:qigen":
        use_qigen = True
    if backend == "gptq:triton":  ##TODO refine the code
        use_triton = True
    if backend == "gptq:marlin":
        use_triton = False
        disable_marlin = True
    if backend == "gptq:exllamav2":  ##need v1 code to export
        use_triton = False
        disable_marlin = True
    if backend == "gptq:exllamav1":
        use_triton = False
        disable_marlin = True
    if backend == "gptq:cuda":
        use_triton = False
        disable_marlin = True
        disable_exllamav2 = True
        disable_exllamav1 = True
    if bits not in [2, 4, 8]:
        use_qigen = False
    if bits not in [2, 4]:
        use_triton = False
    return use_triton, disable_exllamav1, disable_exllamav2, use_qigen, disable_marlin


def dynamic_QuantLienar_for_packing(backend, bits, group_size):
    if "gptq" in backend:
        use_triton, disable_exllamav1, disable_exllamav2, use_qigen, disable_marlin = get_autogptq_backend_config(
            backend, bits
        )
        from auto_gptq.utils.import_utils import dynamically_import_QuantLinear # pylint: disable=E0401
        QuantLinear = dynamically_import_QuantLinear(
            use_triton=use_triton,
            desc_act=False,
            group_size=group_size,
            bits=bits,
            disable_exllama=disable_exllamav1,
            disable_exllamav2=disable_exllamav2,
            use_qigen=use_qigen,
            disable_marlin=disable_marlin,
        )
        return QuantLinear
    ##export all use trition, inference use exllamav2
    elif "autoround" in backend or "auto-round" in backend or "auto_round" in backend:
        from auto_round_extension.cuda.qliner_triton import QuantLinear
        return QuantLinear

    else:
        assert False, f"only support gptq and autoround backend"


@register_format("auto_round")
def save_quantized_as_autoround(output_dir, inplace=True, backend="autoround:exllamav2", **kwargs):
    model = kwargs["model"]
    if not inplace:
        model = copy.deepcopy(model.to("cpu"))
    layer_names_in_block = get_layer_names_in_block(model)

    weight_config = kwargs["weight_config"]
    for name in weight_config.keys():

        config = kwargs["weight_config"][name]
        if config["data_type"] != "int" and config["bits"] >= 16:
            continue
        logger.info(f"packing {name}")

        bits = config["bits"]
        group_size = config["group_size"]

        layer = get_module(model, name)
        device = layer.weight.device

        QuantLinear = dynamic_QuantLienar_for_packing(backend, bits, group_size)

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

        new_layer = QuantLinear(  ##pylint: disable=E1123
            bits, group_size, in_features, out_features, bias, weight_dtype=layer.weight.dtype
        )

        new_layer.device = device
        set_module(model, name, new_layer)
        qlayer = new_layer
        scale = weight_config[name]["scale"]
        zero = weight_config[name]["zp"]
        # so far can only pack layer on CPU
        qlayer.to("cpu")
        ##force to float32 to be compatible with torch 2.0
        layer, scale, zero = layer.to("cpu"), scale.to("cpu"), zero.to("cpu").to(torch.float32)
        qlayer.pack(layer, scale, zero, None)
        qlayer.to(device)
    quantization_config = kwargs["serialization_dict"]
    quantization_config["quant_method"] = "intel/auto-round"
    quantization_config["backend"] = backend
    extra_config = {}
    for layer_name in weight_config:
        if weight_config[layer_name]["bits"] >= 16:
            continue
        if layer_name not in layer_names_in_block:
            extra_config[layer_name] = {}
            extra_config[layer_name]["bits"] = weight_config[layer_name]["bits"]
            extra_config[layer_name]["data_type"] = weight_config[layer_name]["data_type"]
            extra_config[layer_name]["group_size"] = weight_config[layer_name]["group_size"]
            extra_config[layer_name]["sym"] = weight_config[layer_name]["sym"]
        else:
            neq_keys = check_neq_config(
                weight_config[layer_name],
                data_type=quantization_config["data_type"],
                bits=quantization_config["bits"],
                group_size=quantization_config["group_size"],
                sym=quantization_config["sym"],
            )
            if len(neq_keys) > 0:
                extra_config[layer_name] = {}
            for key in neq_keys:
                extra_config[layer_name][key] = weight_config[layer_name][key]
    if len(extra_config) > 0:
        quantization_config["extra_config"] = extra_config
    if hasattr(model, "config"):
        model.config.quantization_config = quantization_config
    tokenizer = kwargs["tokenizer"]
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    save(model, output_dir)


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
