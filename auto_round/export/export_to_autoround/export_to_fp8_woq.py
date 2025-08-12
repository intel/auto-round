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
from typing import Optional, Union

import threadpoolctl as tctl
import torch
import transformers
from tqdm import tqdm

from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    check_start_with_block_name,
    check_to_quantized,
    filter_quantization_config,
    get_module,
    logger,
    set_module,
)


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
    expected_config = {"data_type": data_type, "bits": bits, "group_size": group_size, "sym": sym}
    return [key for key, expected_value in expected_config.items() if config.get(key) != expected_value]


class FP8WOQLinear(torch.nn.Module):

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


def quant_tensor_with_scale(tensor, scale):
    FULL_RANGE = torch.finfo(torch.float8_e4m3fn).max
    qtensor = tensor / scale
    cliped_qtensor = torch.clamp(qtensor, -FULL_RANGE, FULL_RANGE)
    cliped_qtensor_fp8 = cliped_qtensor.to(torch.float8_e4m3fn)
    return scale, cliped_qtensor_fp8


class WeightFP8ActFP8StaticQuantLinear(torch.nn.Module):
    hp_dtype = torch.bfloat16
    fp8_dtype = torch.float8_e4m3fn

    def __init__(
        self,
        in_features,
        out_features,
        weight: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        bias: Union[torch.Tensor, bool, None] = None,
        weight_zp: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_weight = torch.empty((out_features, in_features), dtype=dtype) if weight is None else weight
        self.weight = torch.nn.Parameter(init_weight, requires_grad=False)
        self.dtype = dtype
        if bias is not None:
            if isinstance(bias, bool):
                bias = torch.zeros((out_features,), dtype=dtype)
            self.bias = torch.nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter("bias", None)
        init_weight_scale = torch.empty((out_features, 1), dtype=dtype) if weight_scale is None else weight_scale
        self.register_buffer("weight_scale", init_weight_scale.to(dtype))

        init_weight_zp = torch.zeros((out_features, 1), dtype=dtype) if weight_zp is None else weight_zp
        if weight_zp:
            self.register_buffer("weight_zp", init_weight_zp.to(dtype))

        init_input_scale = torch.zeros((1, 1), dtype=dtype) if input_scale is None else input_scale
        self.register_buffer("input_scale", init_input_scale.to(dtype))
        self.pre_dequantized = False

    @classmethod
    def from_original(cls, config, original_layer):
        """
        Create an FP8WOQLinear layer from an original linear layer.
        """
        device = original_layer.weight.device
        with torch.device(device):
            qdq_linear = cls(
                in_features=original_layer.in_features,
                out_features=original_layer.out_features,
                bias=original_layer.bias,
            )
            return qdq_linear

    def dequant_weight_online(self):
        if self.pre_dequantized:
            return self.weight
        fp8_weight = self.weight
        qdq_weight = fp8_weight.to(self.dtype) * self.weight_scale
        return qdq_weight

    def pre_dequantize(self):
        if self.pre_dequantized:
            return
        dequant_weight = self.dequant_weight_online()
        del self.weight
        del self.weight_scale
        self.weight = torch.nn.Parameter(dequant_weight, requires_grad=False)
        self.pre_dequantized = True

    def qdq_input(self, bf16_input: torch.Tensor):
        input_scale, input_fp8 = quant_tensor_with_scale(bf16_input, self.input_scale.data)
        qdq_input_bf16 = input_fp8.to(self.dtype) * input_scale
        return qdq_input_bf16

    def forward(self, bf16_input: torch.Tensor) -> torch.Tensor:
        qdq_input = self.qdq_input(bf16_input)
        qdq_weight = self.dequant_weight_online()
        out = torch.nn.functional.linear(qdq_input, qdq_weight, self.bias)
        return out


def pack_layer(layer_name, model, data_type, packing_device=None):
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
    if packing_device is None:
        packing_device = "cpu"
        if torch.cuda.is_available():
            packing_device = "cuda"
        elif torch.xpu.is_available():
            packing_device = "xpu"
    layer = get_module(model, layer_name)
    if hasattr(layer, "orig_layer"):
        layer = layer.orig_layer

    if not isinstance(layer, SUPPORTED_LAYER_TYPES):  ##already packed
        return

    if not check_to_quantized(layer):
        return

    device = layer.weight.device
    scale = layer.scale
    zp = layer.zp
    weight = layer.weight
    act_scale = layer.act_scale if hasattr(layer, "act_scale") else None
    torch_dtype = torch.float8_e4m3fn
    if "fp8_e5m2" in data_type:
        torch_dtype = torch.float8_e5m2
    info = torch.finfo(torch_dtype)
    if zp is not None:
        q_weight = weight.to(packing_device) / scale.to(packing_device) + zp.to(packing_device)
    else:
        q_weight = weight.to(packing_device) / scale.to(packing_device)
    q_weight = torch.clamp(q_weight, info.min, info.max)
    q_weight = q_weight.to(torch_dtype)
    if isinstance(layer, torch.nn.Linear):
        in_features = layer.in_features
        out_features = layer.out_features
    # elif isinstance(layer, nn.Conv2d):
    #     in_features = layer.in_channels
    #     out_features = layer.out_channels
    elif isinstance(layer, transformers.pytorch_utils.Conv1D):
        in_features = layer.weight.shape[0]
        out_features = layer.weight.shape[1]
    bias = layer.bias
    my_linear = FP8WOQLinear(
        in_features,
        out_features,
        weight=q_weight,
        weight_scale=scale,
        bias=bias,
        weight_zp=zp,
        input_scale=act_scale,
        dtype=model.dtype,
    )

    my_linear.to(device)
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
        elif layer_config[layer_name]["in_blocks"] or (
            block_name_to_quantize is not None and check_start_with_block_name(layer_name, block_name_to_quantize)
        ):
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
    if not torch.cuda.is_available() and not torch.xpu.is_available():
        max_workers = 2  ## 2 with cuda packing will cause hang occasionally
    packing_device = "cpu"
    if torch.cuda.is_available():
        packing_device = "cuda"
    elif torch.xpu.is_available():
        packing_device = "xpu"
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(names), leave=True) as pbar:

            def wrapper(name):
                pbar.set_description(f"packing {name}")
                with tctl.threadpool_limits(limits=1):
                    pack_layer(name, model, kwargs.get("data_type", "fp8"), packing_device)
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
    save(model, output_dir, safe_serialization=safe_serialization, dtype=dtype)

    return model


def save(
    model: torch.nn.Module, save_dir: str, max_shard_size: str = "5GB", safe_serialization: bool = True, dtype=None
):
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
