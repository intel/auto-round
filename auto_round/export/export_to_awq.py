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

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import torch
import torch.nn as nn

from auto_round.export.register import register_format
from auto_round.utils import check_to_quantized, convert_dtype_torch2str_hf, get_block_names, get_module, logger

# MIT License

# Copyright (c) 2023 MIT HAN Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.


@register_format("auto_awq")
def save_quantized_as_autoawq(output_dir, model_path, **kwargs):
    """Export the model to autogptq format to easily leverage cuda kernel."""

    try:
        from awq import AutoAWQForCausalLM  # pylint: disable=E0401
        from awq.modules.linear import WQLinear_GEMM  # pylint: disable=E0401
        from awq.utils.utils import clear_memory  # pylint: disable=E0401
    except:
        logger.error("autoawq is required. Please install it to support auto_awq format.")
        return


    model = kwargs["model"]
    weight_config = kwargs["weight_config"]
    sym = kwargs["sym"]
    bits = kwargs["bits"]
    group_size = kwargs["group_size"]
    iters = kwargs["iters"]
    lr = kwargs["lr"]
    minmax_lr = kwargs["minmax_lr"]
    enable_minmax_tuning = kwargs["enable_minmax_tuning"]
    enable_quanted_input = kwargs["enable_quanted_input"]
    scale_dtype = kwargs["scale_dtype"]
    tokenizer = kwargs["tokenizer"]
    supported_types = kwargs["supported_types"]

    logger.info("Saving quantized model to autoawq format")
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

    compressed_model = copy.deepcopy(model.to("cpu"))

    q_linear_module = WQLinear_GEMM
    awq_model = AutoAWQForCausalLM.from_pretrained(model_path)
    self_modules = awq_model.get_model_layers(compressed_model)
    del awq_model  # release memory
    for i in range(len(self_modules)):
        module = self_modules[i]
        named_linears = get_named_linears(module)
        for name, linear_layer in named_linears.items():
            key = get_module_name(compressed_model, linear_layer)
            info = weight_config[key]
            if not check_to_quantized(info):
                continue
            info["zp"] = info["zp"].to(torch.float32)
            scale, zp = info["scale"], info["zp"]
            scale = scale.t().contiguous()
            zp = zp.t().contiguous()
            q_linear = q_linear_module.from_linear(
                linear=linear_layer,
                w_bit=bits,
                group_size=group_size,
                init_only=False,
                scales=scale,
                zeros=zp,
            )
            linear_layer.cpu()
            q_linear.to(next(module.parameters()).device)
            set_op_by_name(module, name, q_linear)
            clear_memory()

    quant_config = {}
    quant_config["quant_method"] = "awq"
    quant_config["modules_to_not_convert"] = None
    quant_config["version"] = "gemm"
    quant_config["iters"] = iters
    quant_config["lr"] = lr
    quant_config["minmax_lr"] = minmax_lr
    quant_config["enable_minmax_tuning"] = enable_minmax_tuning
    quant_config["enable_quanted_input"] = enable_quanted_input
    quant_config["scale_dtype"] = convert_dtype_torch2str_hf(scale_dtype)
    quant_config["sym"] = sym
    quant_config["bits"] = bits
    quant_config["group_size"] = group_size
    quant_config["zero_point"] = not sym

    save_quantized(compressed_model, save_dir=output_dir, quant_config=quant_config)


from safetensors.torch import save_file
from transformers.modeling_utils import shard_checkpoint


def save_quantized(
    model,
    save_dir,
    quant_config,
    safetensors=True,
    shard_size="10GB",
):
    save_dir = save_dir[:-1] if save_dir[-1] == "/" else save_dir

    # Save model
    class EmptyModule(nn.Module):
        def __init__(self):
            super(EmptyModule, self).__init__()

        def forward(self, x):
            return x

    # Save model and config files with empty state dict
    awq_quant_config = {
        "quant_method": "awq",
        "zero_point": quant_config["zero_point"],
        "group_size": quant_config["group_size"],
        "bits": quant_config["bits"],
        "version": "gemm",
        "modules_to_not_convert": None,
    }

    model.config.quantization_config = awq_quant_config
    model.generation_config.do_sample = True
    model.save_pretrained(save_dir, state_dict=EmptyModule().state_dict())

    # Remove empty state dict
    default_paths = [
        f"{save_dir}/model.safetensors",
        f"{save_dir}/pytorch_model.bin",
    ]
    for path in default_paths:
        if os.path.exists(path):
            os.remove(path)

    # model_name has no extension, add it when saving state_dict
    model_name = "model.safetensors" if safetensors else "pytorch_model.bin"

    # shard checkpoint into chunks (10GB default)
    shards, index = shard_checkpoint(model.state_dict(), max_shard_size=shard_size, weights_name=model_name)

    for shard_file, shard in shards.items():
        if safetensors:
            # safetensors must be in the same memory, so we duplicate and use contiguous memory
            shard = {k: v.clone().contiguous() for k, v in shard.items()}
            save_file(shard, os.path.join(save_dir, shard_file), metadata={"format": "pt"})
        else:
            torch.save(shard, os.path.join(save_dir, shard_file))

    # save shard index
    if index is not None:
        with open(f"{save_dir}/{model_name}.index.json", "w+") as file:
            file.write(json.dumps(index, indent=4))

    # save quantize_config
    with open(join(save_dir, "quantize_config.json"), "w", encoding="utf-8") as f:
        json.dump(quant_config, f, indent=2)


def get_named_linears(module):
    """Get the name, linear_op pairs of a given module.

    Args:
    module: A module to be searched.
    """
    return {name: m for name, m in module.named_modules() if isinstance(m, torch.nn.Linear)}


def set_op_by_name(layer, name, new_module):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def get_module_name(model, module_to_find):
    """Get the name of a given module in a model.

    Args:
    model: The model.
    module_to_find: A module to be found.

    Returns:
    name: The corresponding name of the given module.
    """
    for name, module in model.named_modules():
        if module is module_to_find:
            return name
    return None
