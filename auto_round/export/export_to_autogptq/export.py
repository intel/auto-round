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

# MIT License
#
# Copyright (c) 2023 潘其威(William)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import torch
import torch.nn as nn
import transformers
from tqdm import tqdm

import auto_round.export.export_to_autogptq.qlinear_triton
from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    check_to_quantized,
    filter_quantization_config,
    get_autogptq_packing_qlinear,
    get_block_names,
    get_module,
    logger,
    set_module,
)

BLOCK_PATTERNS = [  ## copy from transformers optimum
    "transformer.h",
    "model.decoder.layers",
    "gpt_neox.layers",
    "model.layers",
]


def pack_layer(name, model, backend):
    if name == "lm_head":  ##dese not support lm-head
        return
    layer = get_module(model, name)

    if not isinstance(layer, SUPPORTED_LAYER_TYPES):  ##already packed
        return

    bits = layer.bits
    if bits > 8:
        return

    group_size = layer.group_size
    sym = layer.sym

    device = layer.weight.device

    QuantLinear = get_autogptq_packing_qlinear(backend, bits, group_size, sym)

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
        bits, group_size, in_features, out_features, bias, weight_dtype=layer.weight.dtype
    )

    new_layer.device = device
    set_module(model, name, new_layer)
    qlayer = new_layer
    scale = layer.scale
    zero = layer.zp
    # so far can only pack layer on CPU
    qlayer.to("cpu")
    ##force to float32 to be compatible with torch 2.0
    if sym and isinstance(new_layer, auto_round.export.export_to_autogptq.qlinear_triton.QuantLinear):
        layer, scale = layer.to("cpu"), scale.to("cpu")
        zero = int(zero.flatten()[0])
    else:
        layer, scale, zero = layer.to("cpu"), scale.to("cpu"), zero.to("cpu").to(torch.float32)
    sig = inspect.signature(qlayer.pack)
    param_count = len(sig.parameters)
    if param_count == 2:
        qlayer.pack(layer, scale)
    else:
        qlayer.pack(layer, scale, zero, None)
    qlayer.to(device)
    if hasattr(layer, "weight"):
        layer.weight = None
    if hasattr(layer, "bias"):
        layer.bias = None


def save_quantized_as_autogptq(output_dir, inplace=True, backend="auto_gptq:exllamav2", **kwargs):
    """Export the model to autogptq format to easily leverage cuda kernel."""

    model = kwargs["model"]
    safe_serialization = True if "safe_serialization" not in kwargs.keys() else kwargs["safe_serialization"]
    quant_block_list = kwargs.get("quant_block_list", get_block_names(model))
    tokenizer = kwargs.get("tokenizer", None)
    processor = kwargs.get("processor", None)
    if output_dir is not None and os.path.exists(output_dir):
        logger.warning(f"{output_dir} already exists, this may cause model conflict")
    if output_dir is not None and tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    if output_dir is not None and processor is not None:
        processor.save_pretrained(output_dir)
    ##check module quantized in block, this may have bug for mixed precision quantization
    quantization_config = kwargs["serialization_dict"]
    all_blocks = quant_block_list
    flattened_list = [item for sublist in all_blocks for item in sublist]
    common_prefix = os.path.commonprefix(flattened_list).rstrip(".")
    if common_prefix not in BLOCK_PATTERNS:
        logger.error("auto-gptq format may not support loading this quantized model")
        quantization_config["block_name_to_quantize"] = common_prefix
    quantization_config.pop("to_quant_block_names", None)

    ## as layers maybe already packed, we need to check in layer_config
    layer_config = kwargs["layer_config"]
    for n, m in model.named_modules():
        m.tmp_name = n

    all_to_quantized = True
    modules_in_block_to_quantize = []
    for block_names in all_blocks:
        first_block = get_module(model, block_names[0])
        for n, m in first_block.named_modules():
            if m.tmp_name not in layer_config.keys():
                continue
            if not check_to_quantized(layer_config[m.tmp_name]):
                all_to_quantized = False
            else:
                modules_in_block_to_quantize.append(n)
    modules_in_block_to_quantize = [modules_in_block_to_quantize]
    if all_to_quantized:
        modules_in_block_to_quantize = None

    for n, m in model.named_modules():
        delattr(m, "tmp_name")

    if not inplace:
        model = copy.deepcopy(model.to("cpu"))

    layer_config = kwargs["layer_config"]
    names = list(layer_config.keys())
    max_workers = 1
    if not torch.cuda.is_available() and not torch.xpu.is_available():
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
    if output_dir is None:
        return model

    quantization_config["quant_method"] = "gptq"
    quantization_config.pop("dataset", None)  ## pile-10k is not supported in gptq
    quantization_config["desc_act"] = False  ## for autogptq API
    quantization_config["true_sequential"] = False
    quantization_config["damp_percent"] = 0.01
    if modules_in_block_to_quantize is not None:
        quantization_config["modules_in_block_to_quantize"] = modules_in_block_to_quantize
    filter_quantization_config(quantization_config)
    if hasattr(model, "config"):
        model.config.quantization_config = quantization_config

    dtype = torch.float16  ##force dtype to fp16
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
    ##max_shard_size = "10000GB"  ## API of auto-gptq with marlin does not support shard size
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir, max_shard_size=max_shard_size, safe_serialization=safe_serialization)
    config_path = os.path.join(save_dir, "config.json")
    if dtype is not None and dtype != model.dtype and os.path.exists(os.path.join(save_dir, "config.json")):
        with open(config_path, "r") as file:
            data = json.load(file)
        data["torch_dtype"] = str(dtype).split(".")[-1]
        with open(config_path, "w") as file:
            json.dump(data, file, indent=2)

    config_file = "quantize_config.json"
    if hasattr(model, "config") and hasattr(model.config, "quantization_config"):
        with open(os.path.join(save_dir, config_file), "w", encoding="utf-8") as f:
            json.dump(model.config.quantization_config, f, indent=2)
