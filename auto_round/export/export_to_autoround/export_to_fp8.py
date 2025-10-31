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

import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields

import threadpoolctl as tctl
import torch
import transformers
from tqdm import tqdm

from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad
from auto_round.export.export_to_autoround.utils import check_neq_config
from auto_round.export.utils import filter_quantization_config, save_model
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme
from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    check_start_with_block_name,
    check_to_quantized,
    copy_python_files_from_model_cache,
    get_module,
    get_packing_device,
    set_module,
)


class FP8QLinear(torch.nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        weight,
        weight_scale,
        bias=None,
        weight_zp=None,
        input_scale=None,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

        if bias is not None:
            self.bias = torch.nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.register_buffer("weight_scale", weight_scale.to(dtype))

        if weight_zp:
            self.register_buffer("weight_zp", weight_zp.to(dtype))

        if input_scale is not None:
            self.register_buffer("input_scale", input_scale.to(dtype))


def pack_layer(layer_name, model, data_type, device=None):
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
    packing_device = get_packing_device(device)
    layer = get_module(model, layer_name)
    if hasattr(layer, "orig_layer"):
        layer = layer.orig_layer

    if type(layer) not in SUPPORTED_LAYER_TYPES:  ##already packed
        return

    if not check_to_quantized(layer):
        return

    orig_device = layer.weight.device
    scale = layer.scale.view(-1)
    zp = layer.zp
    weight = layer.weight
    weight, orig_shape, pad_len = reshape_pad_tensor_by_group_size(weight, layer.group_size)
    act_scale = layer.act_scale.view(-1) if hasattr(layer, "act_scale") else None
    dtype = torch.float8_e4m3fn
    if "fp8_e5m2" in data_type:
        dtype = torch.float8_e5m2
    info = torch.finfo(dtype)
    if zp is not None:
        if isinstance(zp, torch.Tensor):
            zp = zp.to(packing_device)
        q_weight = weight.to(packing_device) / scale.to(packing_device).unsqueeze(-1) + zp
    else:
        q_weight = weight.to(packing_device) / scale.to(packing_device).unsqueeze(-1)
    q_weight = revert_tensor_by_pad(q_weight, orig_shape=orig_shape, pad_len=pad_len)
    q_weight = torch.clamp(q_weight, info.min, info.max)
    q_weight = q_weight.to(dtype)
    if type(layer) == torch.nn.Linear:
        in_features = layer.in_features
        out_features = layer.out_features
    # elif isinstance(layer, nn.Conv2d):
    #     in_features = layer.in_channels
    #     out_features = layer.out_channels
    elif type(layer) == transformers.pytorch_utils.Conv1D:
        in_features = layer.weight.shape[0]
        out_features = layer.weight.shape[1]
    bias = layer.bias
    my_linear = FP8QLinear(
        in_features,
        out_features,
        weight=q_weight,
        weight_scale=scale,
        bias=bias,
        weight_zp=zp,
        input_scale=act_scale,
        dtype=model.dtype,
    )

    my_linear.to(orig_device)
    set_module(model, layer_name, my_linear)


def save_quantized_as_autoround(output_dir, inplace=True, backend="auto_round", **kwargs):
    model = kwargs["model"]
    safe_serialization = True if "safe_serialization" not in kwargs.keys() else kwargs["safe_serialization"]
    if not inplace:
        model = copy.deepcopy(model.to("cpu"))
    layer_config = kwargs["layer_config"]
    quantization_config = kwargs["serialization_dict"]
    quantization_config["block_name_to_quantize"] = quantization_config.pop("to_quant_block_names", None)
    quantization_config["quant_method"] = "auto-round"
    if "e5m2" in kwargs.get("data_type", "fp8"):
        quantization_config["fmt"] = "e5m2"
    else:
        quantization_config["fmt"] = "e4m3"
    quantization_config["activation_scheme"] = "dynamic" if quantization_config["act_dynamic"] else "static"

    tokenizer = kwargs.get("tokenizer", None)
    processor = kwargs.get("processor", None)
    device = kwargs.get("device", None)
    image_processor = kwargs.get("image_processor", None)
    extra_config = {}
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
                for key in scheme_keys:
                    if cfg[key] is not None:
                        extra_config[layer_name][key] = cfg[key]

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
                    pack_layer(name, model, kwargs.get("data_type", "fp8"), device)
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
    if quantization_config.get("act_bits", 16) <= 8:
        dtype = torch.bfloat16
    elif "awq" in quantization_config.get("packing_format", "auto_round:auto_gptq"):
        dtype = torch.float16  ## awq kernel only supports float16 on cuda
    else:
        dtype = None
    save_model(model, output_dir, safe_serialization=safe_serialization, dtype=dtype)

    return model
