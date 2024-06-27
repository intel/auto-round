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
import json
import os
from os.path import isdir, isfile, join
from typing import Dict, List, Optional, Union

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
from safetensors.torch import save_file as safe_save

from auto_round.export.register import register_format
from auto_round.utils import check_to_quantized, get_block_names, get_module, logger

from ..utils import convert_dtype_torch2str_hf


@register_format("auto_gptq")
def save_quantized_as_autogptq(output_dir, use_triton=True, inplace=True,
                               **kwargs):
    """Export the model to autogptq format to easily leverage cuda kernel."""
    try:
        import auto_gptq
    except ImportError:
        raise ImportError("export to autogptq requires autogptq library. Please run 'pip install auto-gptq'")
    model = kwargs["model"]
    weight_config = kwargs["weight_config"]
    bits = kwargs["bits"]
    group_size = kwargs["group_size"]
    tokenizer = kwargs["tokenizer"]
    supported_types = kwargs["supported_types"]

    logger.info("Saving quantized model to autogptq format, this may take a while...")
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    ##check module quantized in block, this may have bug for mixed precision quantization
    block_name = get_block_names(model)[0]
    first_block = get_module(model, block_name)
    all_to_quantized = True
    modules_in_block_to_quantize = []
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

    if inplace:
        compressed_model = model.to("cpu")
    else:
        compressed_model = copy.deepcopy(model.to("cpu"))

    from auto_gptq.modeling._utils import pack_model  # pylint: disable=E0401

    if bits == 3 or use_triton is False:
        use_triton = False
    quantizers = {}
    for key in weight_config:
        if key == "lm_head":  ##TODO remove this after pr 87 is merged
            continue
        info = weight_config[key]
        if not check_to_quantized(info):
            continue
        quantizers[key] = (None, info["scale"], info["zp"].to(torch.float32), info["g_idx"])
    pack_model(
        compressed_model,
        quantizers,
        bits,
        group_size,
        use_cuda_fp16=True,
        desc_act=False,
        force_layer_back_to_cpu=True,
        use_triton=use_triton
    )
    if output_dir is None:
        return compressed_model
    quantization_config = kwargs["serialization_dict"]
    quantization_config["quant_method"] = "gptq"
    quantization_config.pop("dataset", None) ## pile-10k is not supported in gptq
    if modules_in_block_to_quantize is not None:
        quantization_config["modules_in_block_to_quantize"] = modules_in_block_to_quantize
    if hasattr(model, "config"):
        model.config.quantization_config = quantization_config
    save(compressed_model, output_dir)


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
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir, max_shard_size=max_shard_size, safe_serialization=safe_serialization)
    config_file = "quantization_config.json"
    if hasattr(model, "config") and hasattr(model.config, "quantization_config"):
        model.config.quantization_config["quant_method"] = "intel/auto-round"
        with open(os.path.join(save_dir, config_file), "w", encoding="utf-8") as f:
            json.dump(model.config.quantization_config, f, indent=2)
