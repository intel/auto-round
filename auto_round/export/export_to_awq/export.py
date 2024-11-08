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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
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


import os
from os.path import isdir, isfile, join
import torch
import torch.nn as nn
from auto_round.export.register import register_format
from auto_round.utils import convert_dtype_torch2str_hf, logger, get_module, set_module
import copy
import json
from typing import Dict, List, Optional, Union
from .utils import WQLinear_GEMM, clear_memory, get_self_modules
from concurrent.futures import ThreadPoolExecutor
import threadpoolctl as tctl
from tqdm import tqdm


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
        scale, zp = config["scale"], config["zp"]
        scale = scale.t().contiguous()
        zp = zp.t().contiguous()
        config["zp"] = config["zp"].to(torch.float32)
        bits = config["bits"]
        group_size = config["group_size"]
        linear_layer = get_module(model, name)
        q_linear = WQLinear_GEMM.from_linear(
            linear=linear_layer,
            w_bit=bits,
            group_size=group_size,
            init_only=False,
            scales=scale,
            zeros=zp,
        )
        linear_layer.cpu()
        q_linear.to("cpu")
        set_module(model, name, q_linear)
        clear_memory()
        pbar.update(1)


def save_quantized_as_autoawq(output_dir, inplace=True, **kwargs):
    """Export the model to autogptq format to easily leverage cuda kernel."""
    model = kwargs["model"]
    layer_config = kwargs["layer_config"]
    sym = kwargs["sym"]
    bits = kwargs["bits"]
    group_size = kwargs["group_size"]
    iters = kwargs["iters"]
    lr = kwargs["lr"]
    minmax_lr = kwargs["minmax_lr"]
    enable_minmax_tuning = kwargs["enable_minmax_tuning"]
    enable_quanted_input = kwargs["enable_quanted_input"]
    scale_dtype = kwargs["scale_dtype"]
    tokenizer = kwargs.get("tokenizer", None)
    processor = kwargs.get("processor", None)

    logger.info("Saving quantized model to auto_awq format")
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    if processor is not None:
        processor.save_pretrained(output_dir)
    ##check module quantized in block, this may have bug for mixed precision quantization
    modules_to_not_convert = []
    if inplace:
        compressed_model = model.to("cpu")
    else:
        compressed_model = copy.deepcopy(model.to("cpu"))

    names = list(layer_config.keys())

    self_modules = get_self_modules(compressed_model)
    layers = []
    for i in range(len(self_modules)):
        module = self_modules[i]
        named_linears = get_named_linears(module)
        for name, linear_layer in named_linears.items():
            key = get_module_name(compressed_model, linear_layer)
            layers.append(key)
            config = layer_config[key]
            if config["bits"] > 8:
                modules_to_not_convert.append(name)

    backend = None
    with ThreadPoolExecutor(max_workers=2) as executor:
        with tqdm(total=len(names), leave=True) as pbar:
            def wrapper(name):
                pack_layer(name, model, layer_config, backend, pbar)

            for _ in executor.map(wrapper, names):
                pass
    if output_dir is None:
        return model

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
    if output_dir is None:
        return compressed_model
    save_quantized(compressed_model, save_dir=output_dir, quant_config=quant_config)
    return compressed_model


from safetensors.torch import save_file
from transformers.modeling_utils import shard_checkpoint


def save_quantized(
        model,
        save_dir,
        quant_config,
        safetensors=True,
        shard_size="5GB",
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
        "modules_to_not_convert": quant_config["modules_to_not_convert"],
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
    with open(join(save_dir, "quantization_config.json"), "w", encoding="utf-8") as f:
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

