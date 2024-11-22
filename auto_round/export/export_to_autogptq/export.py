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

from auto_round.utils import check_to_quantized, get_block_names, \
    get_module, logger, set_module
import copy
import json
import os

import torch.nn as nn
import transformers

from auto_round.export.register import register_format
import threadpoolctl as tctl
import inspect
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from auto_round.utils import get_autogptq_packing_qlinear

BLOCK_PATTERNS = [  ## copy from transformers optimum
    "transformer.h",
    "model.decoder.layers",
    "gpt_neox.layers",
    "model.layers",
]


def pack_layer(name, model, layer_config, backend, pbar):
    with tctl.threadpool_limits(limits=1):
        pbar.set_description(f"packing {name}")
        if name == "lm_head":  ##dese not support lm-head
            pbar.update(1)
            return
        config = layer_config[name]
        if config["bits"] > 8:
            pbar.update(1)
            return
        bits = config["bits"]
        group_size = config["group_size"]
        sym = config["sym"]

        layer = get_module(model, name)
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

        ##bias = layer.bias is not None and torch.any(layer.bias)
        bias = True  ## if using the above, llama3 lambada RTN will be NAN , TODO why?
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
        pbar.update(1)


def save_quantized_as_autogptq(output_dir, inplace=True, backend="auto_gptq:exllamav2",
                               **kwargs):
    """Export the model to autogptq format to easily leverage cuda kernel."""

    model = kwargs["model"]
    supported_types = kwargs["supported_types"]
    safe_serialization = True if 'safe_serialization' not in kwargs.keys() else kwargs["safe_serialization"]
    to_quant_block_names = kwargs["to_quant_block_names"]
    logger.info("Saving quantized model to autogptq format, this may take a while...")
    tokenizer = kwargs.get("tokenizer", None)
    processor = kwargs.get("processor", None)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    if processor is not None:
        processor.save_pretrained(output_dir)
    ##check module quantized in block, this may have bug for mixed precision quantization
    quantization_config = kwargs["serialization_dict"]
    if bool(to_quant_block_names):
        all_blocks = to_quant_block_names
        flattened_list = [item for sublist in all_blocks for item in sublist]
        common_prefix = os.path.commonprefix(flattened_list).rstrip('.')
        if common_prefix not in BLOCK_PATTERNS:
            logger.error(f"auto-gptq format may not support loading this quantized model")
            quantization_config['block_name_to_quantize'] = common_prefix
    else:
        all_blocks = get_block_names(model)
        flattened_list = [item for sublist in all_blocks for item in sublist]
        common_prefix = os.path.commonprefix(flattened_list).rstrip('.')
        if common_prefix not in BLOCK_PATTERNS:
            quantization_config['block_name_to_quantize'] = common_prefix

    all_to_quantized = True
    modules_in_block_to_quantize = []
    for block_names in all_blocks:
        first_block = get_module(model, block_names[0])
        for n, m in first_block.named_modules():
            is_supported_type = False
            for supported_type in supported_types:
                if isinstance(m, supported_type):
                    is_supported_type = True
                    break
            if not is_supported_type:
                continue
            if not check_to_quantized(m):
                all_to_quantized = False
            else:
                modules_in_block_to_quantize.append(n)
    modules_in_block_to_quantize = [modules_in_block_to_quantize]
    if all_to_quantized:
        modules_in_block_to_quantize = None

    if not inplace:
        model = copy.deepcopy(model.to("cpu"))

    layer_config = kwargs["layer_config"]
    names = list(layer_config.keys())
    with ThreadPoolExecutor(max_workers=2) as executor:
        with tqdm(total=len(names), leave=True) as pbar:
            def wrapper(name):
                pack_layer(name, model, layer_config, backend, pbar)

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
    if hasattr(model, "config"):
        model.config.quantization_config = quantization_config
    save(model, output_dir, safe_serialization=safe_serialization)
    return model


def save(model: torch.nn.Module, save_dir: str, max_shard_size: str = "5GB", safe_serialization: bool = True):
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
    config_file = "quantize_config.json"
    if hasattr(model, "config") and hasattr(model.config, "quantization_config"):
        with open(os.path.join(save_dir, config_file), "w", encoding="utf-8") as f:
            json.dump(model.config.quantization_config, f, indent=2)

