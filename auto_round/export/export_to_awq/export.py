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


import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor

import threadpoolctl as tctl
import torch
import torch.nn as nn
from tqdm import tqdm

from auto_round.export.export_to_awq.utils import WQLinear_GEMM
from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    check_to_quantized,
    extract_block_names_to_str,
    filter_quantization_config,
    get_block_names,
    get_module,
    logger,
    set_module,
)


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
    linear_layer = get_module(model, name)
    scale, zp = linear_layer.scale, linear_layer.zp
    scale = scale.t().contiguous()
    zp = zp.t().contiguous().to(torch.float32)
    if sym:
        zp = int(zp.flatten()[0])
    q_linear = WQLinear_GEMM.from_linear(
        linear=linear_layer,
        w_bit=bits,
        group_size=group_size,
        init_only=False,
        scales=scale,
        zeros=zp,
    )
    set_module(model, name, q_linear)
    if hasattr(layer, "weight"):
        layer.weight = None
    if hasattr(layer, "bias"):
        layer.bias = None


def save_quantized_as_autoawq(output_dir, inplace=True, **kwargs):
    """Export the model to autogptq format to easily leverage cuda kernel."""
    model = kwargs["model"]
    layer_config = kwargs["layer_config"]
    to_quant_block_names = kwargs.get("to_quant_block_names", None)
    tokenizer = kwargs.get("tokenizer", None)
    processor = kwargs.get("processor", None)
    modules_to_not_convert = []

    if output_dir is not None and os.path.exists(output_dir):
        logger.warning(f"{output_dir} already exists, this may cause model conflict")

    logger.info("Saving quantized model to auto_awq format")
    if tokenizer is not None and output_dir is not None:
        tokenizer.save_pretrained(output_dir)
    if processor is not None and output_dir is not None:
        processor.save_pretrained(output_dir)
        # mllm models
        all_blocks = get_block_names(model, quant_vision=True)
        all_block_names = extract_block_names_to_str(all_blocks)
        all_block_names = all_block_names.split(",")
        to_quant_block_names = to_quant_block_names.split(",")
        modules_to_not_convert = list(set(all_block_names) - set(to_quant_block_names))

    if inplace:
        compressed_model = model.to("cpu")
    else:
        compressed_model = copy.deepcopy(model.to("cpu"))

    names = list(layer_config.keys())

    backend = None
    max_workers = 1
    if not torch.cuda.is_available() and not torch.xpu.is_available():
        max_workers = 2  ## 2 with cuda packing will cause hang occasionally
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(names), leave=True) as pbar:

            def wrapper(name):
                pbar.set_description(f"packing {name}")
                with tctl.threadpool_limits(limits=1):
                    pack_layer(name, compressed_model, backend)
                pbar.update(1)

            for _ in executor.map(wrapper, names):
                pass
    if output_dir is None:
        return model

    quantization_config = kwargs["serialization_dict"]

    if output_dir is None:
        return compressed_model

    layer_config = kwargs["layer_config"]
    for key in layer_config.keys():
        if not check_to_quantized(layer_config[key]) and not any(name in key for name in modules_to_not_convert):
            modules_to_not_convert.append(key)

    quantization_config["quant_method"] = "awq"
    quantization_config["zero_point"] = not quantization_config["sym"]
    quantization_config["version"] = "gemm"

    quantization_config["modules_to_not_convert"] = modules_to_not_convert
    ##check module quantized in block, this may have bug for mixed precision quantization
    filter_quantization_config(quantization_config)
    if hasattr(compressed_model, "config"):
        compressed_model.config.quantization_config = quantization_config
    safe_serialization = kwargs.get("safe_serialization", True)
    dtype = torch.float16  ##force dtype to fp16
    save(compressed_model, output_dir, safe_serialization=safe_serialization, dtype=dtype)

    return compressed_model


def save(model: nn.Module, save_dir: str, max_shard_size: str = "5GB", safe_serialization: bool = True, dtype=None):
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
