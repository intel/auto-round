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
from dataclasses import fields
from typing import Any, Dict

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
from auto_round.export.utils import filter_quantization_config, get_autogptq_packing_qlinear, save_model
from auto_round.schemes import QuantizationScheme

GPTQ_REQUIRED_CONFIG_KEYS = (
    "bits",
    "group_size",
    "sym",
)

from auto_round.logger import logger
from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    check_start_with_block_name,
    check_to_quantized,
    copy_python_files_from_model_cache,
    get_block_names,
    get_module,
    unsupported_meta_device,
    json_serialize,
    matches_any_regex,
    set_module,
    to_standard_regex,
)

BLOCK_PATTERNS = [  ## copy from transformers optimum
    "transformer.h",
    "model.decoder.layers",
    "gpt_neox.layers",
    "model.layers",
]
from auto_round.export.export_to_autoround.utils import check_neq_config


def convert_to_autogptq_dynamic(regex_config: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Convert AutoRound-style regex_config into AutoGPTQ-style QuantizerConfig.dynamic.

    Rules:
    - bits < 16 -> quantize -> positive match `+:regex`
    - bits == 16 -> skip quantize -> negative match `-:regex`
    """
    converted = {}
    for name, cfg in regex_config.items():
        bits = cfg.get("bits")
        regex = to_standard_regex(name)

        if bits is None:
            continue  # ignore invalid entries
        elif bits < 16:
            converted[f"+:{regex}"] = {"bits": bits}
            for key in GPTQ_REQUIRED_CONFIG_KEYS:  # only save keys gptq supported
                converted[f"+:{regex}"][key] = regex_config[name][key]
        else:
            # skip quantization
            converted[f"-:{regex}"] = {}
    return converted


def convert_from_autogptq_dynamic(dynamic_config: dict) -> dict:
    """
    Convert AutoGPTQ-style QuantizerConfig.dynamic into AutoRound-style extra_config.

    Rules:
    - '+:regex' => quantize => keep bits and other quantization keys
    - '-:regex' => skip quantize => set bits to 16 (FP16 passthrough)
    """
    converted = {}
    for name, cfg in dynamic_config.items():
        # Strip the +: or -:
        if name.startswith("+:"):
            regex = name[2:]
            # keep all config fields (bits, group_size, sym, etc.)
            converted[regex] = dict(cfg)
        elif name.startswith("-:"):
            regex = name[2:]
            # mark skipped layers with bits=16
            converted[regex] = {"bits": 16, "act_bits": 16}
    return converted


def pack_layer(name, model, backend, device=None):
    if name == "lm_head":  ##dese not support lm-head
        return
    layer = get_module(model, name)

    if type(layer) not in SUPPORTED_LAYER_TYPES:  # already packed
        return

    orig_device = layer.weight.device  # must place after 74
    bits = layer.bits
    if bits > 8:
        return

    group_size = layer.group_size
    sym = layer.sym

    QuantLinear = get_autogptq_packing_qlinear(backend, bits, group_size, sym)

    if type(layer) == nn.Linear:
        in_features = layer.in_features
        out_features = layer.out_features
    elif type(layer) == nn.Conv2d:
        in_features = layer.in_channels
        out_features = layer.out_channels
    elif type(layer) == transformers.pytorch_utils.Conv1D:
        in_features = layer.weight.shape[0]
        out_features = layer.weight.shape[1]

    bias = layer.bias is not None
    ##bias = True  ## if using the above, llama3 lambada RTN will be NAN , TODO why?
    new_layer = QuantLinear(  ##pylint: disable=E1123
        bits, group_size, in_features, out_features, bias, weight_dtype=layer.weight.dtype
    )

    new_layer.device = orig_device
    set_module(model, name, new_layer)
    qlayer = new_layer
    scale = layer.scale
    zero = layer.zp
    # so far can only pack layer on CPU
    qlayer.to("cpu")
    ##force to float32 to be compatible with torch 2.0
    if sym and isinstance(zero, torch.Tensor):
        layer, scale, zero = layer.to("cpu"), scale.to("cpu"), zero.to("cpu")
        if isinstance(new_layer, auto_round.export.export_to_autogptq.qlinear_triton.QuantLinear):
            zero = int(zero.flatten()[0])
    else:
        layer, scale, zero = layer.to("cpu"), scale.to("cpu"), zero
    sig = inspect.signature(qlayer.pack)
    param_count = len(sig.parameters)
    if param_count == 2:
        qlayer.pack(layer, scale, device)
    else:
        qlayer.pack(layer, scale, zero, None, device)
    qlayer.to(orig_device)
    if hasattr(layer, "weight"):
        layer.weight = None
    if hasattr(layer, "bias"):
        layer.bias = None


def save_quantized_as_autogptq(output_dir, inplace=True, backend="auto_gptq:exllamav2", **kwargs):
    """Export the model to autogptq format to easily leverage cuda kernel."""

    # --- 1️⃣ Extract inputs & configs ---
    model = kwargs["model"]
    quantization_config = kwargs["serialization_dict"]
    layer_config = kwargs["layer_config"]
    quant_block_list = kwargs.get("quant_block_list", get_block_names(model))
    tokenizer = kwargs.get("tokenizer")
    processor = kwargs.get("processor")
    image_processor = kwargs.get("image_processor")
    device = kwargs.get("device")
    safe_serialization = kwargs.get("safe_serialization", True)

    # --- Save metadata (tokenizer, processor, etc.) ---
    if output_dir:
        if os.path.exists(output_dir):
            logger.warning(f"{output_dir} already exists, may cause overwrite conflicts.")
        for comp in (tokenizer, processor, image_processor):
            if comp is not None and hasattr(comp, "save_pretrained"):
                comp.save_pretrained(output_dir)

    # --- Handle quantization structure ---
    all_blocks = quant_block_list
    flattened = [x for sub in all_blocks for x in sub]
    common_prefix = os.path.commonprefix(flattened).rstrip(".")

    if "BLOCK_PATTERNS" in kwargs and common_prefix not in kwargs["BLOCK_PATTERNS"]:
        logger.error(f"Unsupported block prefix '{common_prefix}' for AutoGPTQ format.")
        quantization_config["block_name_to_quantize"] = common_prefix
    quantization_config.pop("to_quant_block_names", None)

    # --- Build per-layer dynamic overrides ---
    regex_config = quantization_config.pop("regex_config", {})
    block_name_to_quantize = quantization_config.get("block_name_to_quantize")
    extra_config = {}
    lm_head_quantized = False
    scheme_keys = [f.name for f in fields(QuantizationScheme)]
    for layer_name, cfg in layer_config.items():
        bits = cfg.get("bits", 16)
        in_blocks = cfg.get("in_blocks", False)
        # Handle non-block layers (e.g., LM head)
        if not in_blocks and bits <= 8:
            lm_head_quantized = True
            extra_config[layer_name] = {k: cfg[k] for k in GPTQ_REQUIRED_CONFIG_KEYS}
            continue
        # Handle block layers
        if in_blocks or (block_name_to_quantize and check_start_with_block_name(layer_name, block_name_to_quantize)):
            neq_keys = check_neq_config(cfg, **{k: quantization_config[k] for k in scheme_keys})
            if neq_keys:
                if matches_any_regex(layer_name, regex_config):
                    continue
                extra_config[layer_name] = {k: cfg[k] for k in GPTQ_REQUIRED_CONFIG_KEYS}

    # --- Merge regex_config + extra_config into GPTQ dynamic config ---
    dynamic = {}
    if regex_config:
        dynamic.update(convert_to_autogptq_dynamic(regex_config))
    if extra_config:
        dynamic.update(convert_to_autogptq_dynamic(extra_config))
    if dynamic:
        quantization_config["dynamic"] = dynamic

    # --- Block-wise quantization verification ---
    for n, m in model.named_modules():
        m.tmp_name = n

    all_to_quantized = True
    modules_in_block_to_quantize = []
    if not dynamic:  # Only uniform precision
        for block_names in all_blocks:
            first_block = get_module(model, block_names[0])
            for n, m in first_block.named_modules():
                if m.tmp_name not in layer_config:
                    continue
                if not check_to_quantized(layer_config[m.tmp_name]):
                    all_to_quantized = False
                else:
                    modules_in_block_to_quantize.append(n)
        modules_in_block_to_quantize = [modules_in_block_to_quantize]

    if all_to_quantized:
        modules_in_block_to_quantize = None

    for _, m in model.named_modules():
        delattr(m, "tmp_name")

    if not inplace:
        model = copy.deepcopy(model.to("cpu"))

    names = list(layer_config.keys())
    max_workers = 1
    if not torch.cuda.is_available() and not torch.xpu.is_available():
        max_workers = 2  ## 2 with cuda packing will cause hang occasionally
    if not unsupported_meta_device(model):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(total=len(names), leave=True) as pbar:

                def wrapper(name):
                    pbar.set_description(f"packing {name}")
                    with tctl.threadpool_limits(limits=1):
                        pack_layer(name, model, backend, device)
                    pbar.update(1)

                for _ in executor.map(wrapper, names):
                    pass
    if output_dir is None:
        return model
    quantization_config["lm_head"] = lm_head_quantized
    quantization_config["provider"] = "auto-round"
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
    save_model(
        model, output_dir, safe_serialization=safe_serialization, dtype=dtype, config_file="quantize_config.json"
    )
    return model
