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

# Copyright (c) 2023-2024 The ggml authors
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

from __future__ import annotations

import json
import os
import re
import sys
from functools import partial
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
import requests
import torch
from transformers import AutoConfig

from auto_round.export.export_to_gguf.config import ModelType
from auto_round.export.export_to_gguf.packing import ggml_quant
from auto_round.utils import (
    LazyImport,
    clean_module_parameter,
    clear_memory,
    get_module,
    get_packing_device,
    is_quantized_input_module,
    is_separate_tensor,
    logger,
)

gguf = LazyImport("gguf")

if TYPE_CHECKING:
    from torch import Tensor


def download_convert_file(redownload=False):
    CONVERT_URL = "https://raw.githubusercontent.com/ggml-org/llama.cpp/refs/heads/master/convert_hf_to_gguf.py"
    FILE_NAME = "convert_hf_to_gguf.py"
    gguf_export_dir = os.path.dirname(__file__)
    if redownload is False and FILE_NAME in os.listdir(gguf_export_dir):
        return
    try:
        response = requests.get(CONVERT_URL)
    except:
        logger.error(
            f"Fail to download the dependency file, please try downloading the convert_hf_to_gguf.py"
            f" from https://github.com/ggml-org/llama.cpp manually and move it to {gguf_export_dir}."
        )
        sys.exit(-1)
    with open(os.path.join(gguf_export_dir, FILE_NAME), "w", encoding="utf-8") as f:
        f.write(response.text)


def wrapper_model_instance(
    model_instance, model, layer_config, low_cpu_mem_usage=False, device=None, quant_nontext_module=False
):
    if model_instance.model_arch == gguf.MODEL_ARCH.MMPROJ and model_instance.fname_out.is_dir():
        model_instance.fname_out = model_instance.fname_out / "mmproj-model.gguf"
    model_instance.model = model
    model_instance.layer_config = layer_config
    model_instance.low_cpu_mem_usage = _need_low_cpu_mem(low_cpu_mem_usage)

    model_instance.get_tensors = partial(get_tensors, model_instance)
    model_instance.prepare_tensors = partial(prepare_tensors, model_instance)

    model_instance.device = device
    model_instance.quant_nontext_module = quant_nontext_module

    return model_instance


def _need_low_cpu_mem(low_cpu_mem_usage):
    if not low_cpu_mem_usage:
        return False

    # process = psutil.Process(os.getpid())
    # mem_usage = process.memory_info().rss
    # memory_info = psutil.virtual_memory()
    # if memory_info.available > mem_usage / 3:
    #     return False
    # else:
    #     logger.info("use low cpu memory mode.")
    return True


def get_moe_name(cls, name, new_name):
    type_mapping = {
        "FFN_GATE_EXP": ["gate_proj", "w1", "linear"],
        "FFN_DOWN_EXP": ["down_proj", "w2", "linear_1"],
        "FFN_UP_EXP": ["up_proj", "w3", "linear_v"],
    }
    nums = re.findall(r"\.(\d+)\.", name)
    if len(nums) != 2:
        return name
    name_tmp = name[: -len(".weight")].replace(f".{nums[1]}", "")
    new_name_tmp = new_name[: -len(".weight")]
    if cls.tensor_map.get_type(name_tmp) == cls.tensor_map.get_type(new_name_tmp):
        return name

    tensor_type = cls.tensor_map.get_type(new_name_tmp).name
    experts_name = name_tmp.split(".")[-1]
    for k, v in type_mapping.items():
        if experts_name in v:
            idx = v.index(experts_name)
            name = name.replace(experts_name, type_mapping[tensor_type][idx])
            return name
    return name


def get_tensors(cls) -> Iterator[tuple[str, Tensor]]:
    if not hasattr(cls.model, "tensor_name_list"):
        cls.model.tensor_name_list = []

    reverse_key_mapping = {v: k for k, v in cls.model._checkpoint_conversion_mapping.items()}
    for name, tensor in cls.model.named_parameters():
        if name not in cls.model.tensor_name_list:
            cls.model.tensor_name_list.append(name)
            for pattern, replacement in reverse_key_mapping.items():
                replacement = replacement.lstrip("^")  # strip off un-needed chars and patterns
                replacement = re.sub(r"\(.*\)", "", replacement)
                key, n_replace = re.subn(pattern, replacement, name)
                cls.model.tensor_name_list.append(key)
                if n_replace > 0:
                    break
        yield name, tensor

    def is_extra_tensor(tensor_name):
        if getattr(cls.model, "_is_quantized_input_module", False) and "scale" in tensor_name.split(".")[-1]:
            return False
        if tensor_name not in cls.model.tensor_name_list:
            return True
        return False

    extra_tensor = {}
    if hasattr(cls.model, "name_or_path"):
        from safetensors import safe_open

        from auto_round.export.export_to_gguf.special_handle import get_tensor_from_file
        from auto_round.utils import download_or_get_path

        dir_path = cls.model.name_or_path
        if not os.path.isdir(dir_path):
            dir_path = download_or_get_path(dir_path)
        INDEX_FILE = "model.safetensors.index.json"
        if INDEX_FILE in os.listdir(dir_path):
            with open(os.path.join(dir_path, INDEX_FILE)) as f:
                tensor_index = json.load(f)
            for tensor_name in tensor_index["weight_map"]:
                if is_extra_tensor(tensor_name):
                    extra_tensor[tensor_name] = get_tensor_from_file(dir_path, tensor_name)
        else:
            f = safe_open(os.path.join(dir_path, "model.safetensors"), framework="pt")
            for tensor_name in f.keys():
                if is_extra_tensor(tensor_name):
                    extra_tensor[tensor_name] = get_tensor_from_file(dir_path, tensor_name)

    for name, tensor in extra_tensor.items():
        yield name, tensor


def _quant_data_with_args(
    data_torch, data_qtype, scale, zp, d_scale=None, wmin=None, d_wmin=None, imatrix=None, device=None
):
    device = data_torch.device if device is None else device
    data_torch = data_torch.to(torch.float32)
    scale = scale.to(torch.float32) if isinstance(scale, torch.Tensor) else scale
    zp = zp.to(torch.float32) if isinstance(zp, torch.Tensor) else zp
    d_scale = d_scale.to(torch.float32) if isinstance(d_scale, torch.Tensor) else d_scale
    d_wmin = d_wmin.to(torch.float32) if isinstance(d_wmin, torch.Tensor) else d_wmin
    wmin = wmin.to(torch.float32) if isinstance(wmin, torch.Tensor) else wmin
    imatrix = imatrix.to(torch.float32) if isinstance(imatrix, torch.Tensor) else imatrix

    data = ggml_quant(
        data_torch,
        data_qtype.name.lower(),
        scale,
        zp,
        wmin=wmin,
        d_scale=d_scale,
        d_wmin=d_wmin,
        imatrix=imatrix,
        device=device,
    )
    return data


def _quant_data(cls, data_torch, data_qtype, name, modify_name, new_name, bid, device=None):
    """

    Args:
        data_torch: original data tensor
        data_qtype: quantization type
        name: original tensor name, for getting auto_round config, model.language_model.layers.0.input_linear.weight
        modify_name: modified tensor name, for gguf mapping, model.layers.0.input_linear.weight
        new_name: name after modify_tensors, gguf using this to save tensor, like blk.0.ffn_gate.weight
        bid: block id
        device: device to perform quantization. Defaults to None.

    """
    suffix = ".weight"
    device = data_torch.device if device is None else device

    if name.endswith(suffix):
        layer_name = name[: -len(suffix)]
    else:
        layer_name = name
    module = get_module(cls.model, layer_name)
    kwargs = {
        "scale": None,
        "zp": None,
        "d_scale": None,
        "d_wmin": None,
        "wmin": None,
        "imatrix": None,
    }
    # support for MOE model with cls eexperts not linear
    # if hasattr(module, "scale") or ("exps" in new_name and len(data_torch.shape) == 3):
    for attr in ["scale", "zp", "w_d_scale", "w_d_wmin", "w_wmin"]:
        if hasattr(module, attr) and getattr(module, attr) is not None:
            attr_tensor = getattr(module, attr)
            if not isinstance(attr_tensor, torch.Tensor):
                continue
            if hasattr(cls, "permute"):
                bs = module.weight.shape[0]
                attr_tensors_dict = dict(cls.modify_tensors(attr_tensor.reshape(bs, -1), modify_name, bid))
                attr_tensor = attr_tensors_dict[new_name]
            if attr in kwargs:
                kwargs[attr] = attr_tensor.to(torch.float32)
            else:
                kwargs[attr.replace("w_", "")] = attr_tensor.to(torch.float32)
    data_torch = data_torch.to(torch.float32)

    data = ggml_quant(data_torch, data_qtype.name.lower(), device=device, **kwargs)
    # else:
    #     # if data_torch.dtype ==torch.float32:
    #     #     data_qtype = gguf.GGMLQuantizationType.F32
    #     # else:
    #     #     data_qtype = gguf.GGMLQuantizationType.F16
    #     data_qtype = gguf.GGMLQuantizationType.F32  ##FP16 has issues at inference
    #     data = data_torch.to(torch.float32).squeeze().cpu().numpy()
    return data, data_qtype


def get_qtype_by_layer_config(layer_config, name, data_qtype):
    name = name[: -len(".weight")]
    if name not in layer_config or layer_config[name]["bits"] >= 16:
        return data_qtype
    bits = layer_config[name].get("bits")
    super_bits = layer_config[name].get("super_bits")
    sym = layer_config[name].get("sym")
    if bits == 2:
        return gguf.GGMLQuantizationType.Q2_K
    if bits == 3:
        return gguf.GGMLQuantizationType.Q3_K
    if bits == 4:
        if super_bits is not None:
            return gguf.GGMLQuantizationType.Q4_K
        if super_bits is None and sym:
            return gguf.GGMLQuantizationType.Q4_0
        if super_bits is None and not sym:
            return gguf.GGMLQuantizationType.Q4_1
    if bits == 5:
        if super_bits is not None:
            return gguf.GGMLQuantizationType.Q5_K
        if super_bits is None and sym:
            return gguf.GGMLQuantizationType.Q5_0
        if super_bits is None and not sym:
            return gguf.GGMLQuantizationType.Q5_1
    if bits == 6:
        return gguf.GGMLQuantizationType.Q6_K
    if bits == 8:
        return gguf.GGMLQuantizationType.Q8_0
    raise ValueError(f"Unknown file type: {data_qtype}")

    # data_qtype = re.sub(r"Q\d", f"Q{bits}", cls.ftype.name)
    #
    # if data_qtype == "MOSTLY_Q8_0":
    #     data_qtype = gguf.GGMLQuantizationType.Q8_0
    # elif data_qtype == "MOSTLY_Q4_0":
    #     data_qtype = gguf.GGMLQuantizationType.Q4_0
    # elif data_qtype == "MOSTLY_Q4_1":
    #     data_qtype = gguf.GGMLQuantizationType.Q4_1
    # elif data_qtype == "MOSTLY_Q5_0":
    #     data_qtype = gguf.GGMLQuantizationType.Q5_0
    # elif data_qtype == "MOSTLY_Q5_1":
    #     data_qtype = gguf.GGMLQuantizationType.Q5_1
    # elif data_qtype.startswith("MOSTLY_Q2_K"):
    #     data_qtype = gguf.GGMLQuantizationType.Q2_K
    # elif data_qtype.startswith("MOSTLY_Q3_K"):
    #     data_qtype = gguf.GGMLQuantizationType.Q3_K
    # elif data_qtype.startswith("MOSTLY_Q4_K"):
    #     data_qtype = gguf.GGMLQuantizationType.Q4_K
    # elif data_qtype.startswith("MOSTLY_Q5_K"):
    #     data_qtype = gguf.GGMLQuantizationType.Q5_K
    # elif data_qtype.startswith("MOSTLY_Q6_K"):
    #     data_qtype = gguf.GGMLQuantizationType.Q6_K
    # else:
    #     raise ValueError(f"Unknown file type: {data_qtype}")
    # return data_qtype


def _special_name_handle(cls, name):
    # for Qwen2VL
    def remove_prefix(name, key_list):
        for key in key_list:
            if key in name and not name.startswith(key):
                name = ".".join(name.split(".")[1:])
                break
        return name

    if cls.model_arch == gguf.MODEL_ARCH.QWEN2VL:
        name = name.replace("language_model.", "")
        visual_keys = ["thinker", "visual", "audio", "talker", "token2wav"]
        name = remove_prefix(name, visual_keys)
    if cls.__class__.__name__ == "Qwen2VLVisionModel":
        if "visual." in name and not name.startswith("visual."):
            name = ".".join(name.split(".")[1:])

    # # for gemma3
    if cls.model_arch == gguf.MODEL_ARCH.GEMMA3 or cls.__class__.__name__ == "Gemma3VisionModel":
        visual_keys = ["multi_modal_projector", "vision_tower", "multimodal_projector"]
        name = remove_prefix(name, visual_keys)

    # for LlavaForConditionalGeneration
    if cls.model_arch == gguf.MODEL_ARCH.LLAMA:
        name = name.replace("language_model.", "")
        # name = name.replace("model.", "")
    if cls.__class__.__name__ == "LlavaVisionModel":
        visual_keys = ["multi_modal_projector", "vision_tower"]
        name = remove_prefix(name, visual_keys)

    # for InternVisionModel
    if cls.__class__.__name__ == "InternVisionModel":
        visual_keys = ["vision_model"]
        name = remove_prefix(name, visual_keys)

    return name


def prepare_tensors(cls):
    max_name_len = max(len(s) for _, s in cls.tensor_map.mapping.values()) + len(".weight,")
    device = get_packing_device(cls.device)

    for name, data_torch in chain(cls.generate_extra_tensors(), cls.get_tensors()):
        if name in getattr(cls.model, "_tied_weights_keys", []) and not is_separate_tensor(cls.model, name):
            continue
        if data_torch is None or data_torch.numel() == 0:
            continue
        # we don't need these
        if name.endswith((".attention.masked_bias", ".attention.bias", ".rotary_emb.inv_freq")):
            continue
        if hasattr(cls, "current_packing_block") and cls.current_packing_block is not None:  # pylint: disable=E1101
            current_packing_block_split = cls.current_packing_block.split(".")  # pylint: disable=E1101
            name_split = name.split(".")
            if (
                len(name_split) < len(current_packing_block_split)
                or name_split[: len(current_packing_block_split)] != current_packing_block_split
            ):
                continue

        old_dtype = data_torch.dtype

        # convert any unsupported data types to float32
        if data_torch.dtype not in (torch.float16, torch.float32):
            data_torch = data_torch.to(torch.float32)

        # use the first number-like part of the tensor name as the block id
        bid = None
        for part in name.split("."):
            if part.isdecimal():
                bid = int(part)
                break

        clean_weight_list = []

        modify_name = _special_name_handle(cls, name)
        orig_device = data_torch.device
        data_torch = data_torch.to("cpu")
        for new_name, data_torch in cls.modify_tensors(data_torch, modify_name, bid):
            skip = False
            for tensor_info in cls.gguf_writer.tensors:
                if new_name in tensor_info:
                    logger.info(f"{new_name} already add to gguf_writer, skip")
                    skip = True
                    break
            if skip:
                continue
            # sync with new version of gguf
            # data = data_torch.squeeze()
            data = data_torch
            n_dims = len(data.shape)
            data_qtype: gguf.GGMLQuantizationType | bool = cls.tensor_force_quant(name, new_name, bid, n_dims)

            # Most of the codebase that takes in 1D tensors or norms only handles F32 tensors
            if n_dims <= 1 or new_name.endswith("_norm.weight"):
                data_qtype = gguf.GGMLQuantizationType.F32

            # Conditions should closely match those in llama_model_quantize_internal in llama.cpp
            # Some tensor types are always in float32
            if data_qtype is False and (
                any(
                    cls.match_model_tensor_name(new_name, key, bid)
                    for key in (
                        gguf.MODEL_TENSOR.FFN_GATE_INP,
                        gguf.MODEL_TENSOR.POS_EMBD,
                        gguf.MODEL_TENSOR.TOKEN_TYPES,
                        gguf.MODEL_TENSOR.SSM_CONV1D,
                        gguf.MODEL_TENSOR.SHORTCONV_CONV,
                        gguf.MODEL_TENSOR.TIME_MIX_FIRST,
                        gguf.MODEL_TENSOR.TIME_MIX_W1,
                        gguf.MODEL_TENSOR.TIME_MIX_W2,
                        gguf.MODEL_TENSOR.TIME_MIX_DECAY_W1,
                        gguf.MODEL_TENSOR.TIME_MIX_DECAY_W2,
                        gguf.MODEL_TENSOR.TIME_MIX_LERP_FUSED,
                        gguf.MODEL_TENSOR.POSNET_NORM1,
                        gguf.MODEL_TENSOR.POSNET_NORM2,
                        gguf.MODEL_TENSOR.V_ENC_EMBD_POS,
                        gguf.MODEL_TENSOR.A_ENC_EMBD_POS,
                        gguf.MODEL_TENSOR.ALTUP_CORRECT_COEF,
                        gguf.MODEL_TENSOR.ALTUP_PREDICT_COEF,
                    )
                )
                or not new_name.endswith(".weight")
            ):
                data_qtype = gguf.GGMLQuantizationType.F32

            if data_qtype is False and any(
                cls.match_model_tensor_name(new_name, key, bid)
                for key in (
                    gguf.MODEL_TENSOR.TOKEN_EMBD,
                    gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD,
                    gguf.MODEL_TENSOR.OUTPUT,
                    gguf.MODEL_TENSOR.ALTUP_ROUTER,
                    gguf.MODEL_TENSOR.LAUREL_L,
                    gguf.MODEL_TENSOR.LAUREL_R,
                )
            ):
                if cls.ftype in (
                    gguf.LlamaFileType.MOSTLY_TQ1_0,
                    gguf.LlamaFileType.MOSTLY_TQ2_0,
                ):
                    # TODO: use Q4_K and Q6_K
                    data_qtype = gguf.GGMLQuantizationType.F16
                    # data_qtype = gguf.GGMLQuantizationType.Q8_0  # llama.cpp:llama_tensor_get_type

            # get name by new_name (for experts),
            name = get_moe_name(cls, name, new_name)
            clean_weight_list.append(name)
            # get data_qtype by layer_config
            if isinstance(data_qtype, bool):
                data_qtype = get_qtype_by_layer_config(cls.layer_config, name, data_qtype)

            # # No override (data_qtype is False), or wants to be quantized (data_qtype is True)
            if isinstance(data_qtype, bool):
                if cls.ftype == gguf.LlamaFileType.ALL_F32:
                    data_qtype = gguf.GGMLQuantizationType.F32
                elif cls.ftype == gguf.LlamaFileType.MOSTLY_F16:
                    data_qtype = gguf.GGMLQuantizationType.F16
                elif cls.ftype == gguf.LlamaFileType.MOSTLY_BF16:
                    data_qtype = gguf.GGMLQuantizationType.BF16
                elif cls.ftype == gguf.LlamaFileType.MOSTLY_Q8_0:
                    data_qtype = gguf.GGMLQuantizationType.Q8_0
                elif cls.ftype == gguf.LlamaFileType.MOSTLY_Q4_0:
                    data_qtype = gguf.GGMLQuantizationType.Q4_0
                elif cls.ftype == gguf.LlamaFileType.MOSTLY_Q4_1:
                    data_qtype = gguf.GGMLQuantizationType.Q4_1
                elif cls.ftype == gguf.LlamaFileType.MOSTLY_Q5_0:
                    data_qtype = gguf.GGMLQuantizationType.Q5_0
                elif cls.ftype == gguf.LlamaFileType.MOSTLY_Q5_1:
                    data_qtype = gguf.GGMLQuantizationType.Q5_1
                elif cls.ftype == gguf.LlamaFileType.MOSTLY_Q2_K_S or cls.ftype == gguf.LlamaFileType.MOSTLY_Q2_K:
                    data_qtype = gguf.GGMLQuantizationType.Q2_K
                elif (
                    cls.ftype == gguf.LlamaFileType.MOSTLY_Q3_K_S
                    or cls.ftype == gguf.LlamaFileType.MOSTLY_Q3_K_M
                    or cls.ftype == gguf.LlamaFileType.MOSTLY_Q3_K_L
                ):
                    data_qtype = gguf.GGMLQuantizationType.Q3_K
                elif cls.ftype == gguf.LlamaFileType.MOSTLY_Q4_K_S or cls.ftype == gguf.LlamaFileType.MOSTLY_Q4_K_M:
                    data_qtype = gguf.GGMLQuantizationType.Q4_K
                elif cls.ftype == gguf.LlamaFileType.MOSTLY_Q5_K_S or cls.ftype == gguf.LlamaFileType.MOSTLY_Q5_K_M:
                    data_qtype = gguf.GGMLQuantizationType.Q5_K
                elif cls.ftype == gguf.LlamaFileType.MOSTLY_Q6_K:
                    data_qtype = gguf.GGMLQuantizationType.Q6_K
                else:
                    raise ValueError(f"Unknown file type: {cls.ftype.name}")

            if data_qtype.name.endswith("_K") and data_torch.shape[-1] % 256 != 0:
                if data_qtype in [
                    gguf.GGMLQuantizationType.Q2_K,
                    gguf.GGMLQuantizationType.Q3_K,
                    gguf.GGMLQuantizationType.Q4_K,
                ]:
                    data_qtype = gguf.GGMLQuantizationType.Q5_0
                elif data_qtype == gguf.GGMLQuantizationType.Q5_K:
                    data_qtype = gguf.GGMLQuantizationType.Q5_0
                elif data_qtype == gguf.GGMLQuantizationType.Q6_K:
                    data_qtype = gguf.GGMLQuantizationType.Q8_0

            if cls.model_arch == gguf.MODEL_ARCH.MMPROJ and cls.quant_nontext_module is False:
                data_qtype = gguf.GGMLQuantizationType.F32

            from auto_round.export.export_to_gguf.config import GGML_QUANT_SIZES

            if data_qtype.name.lower() in GGML_QUANT_SIZES:
                block_size, type_size = GGML_QUANT_SIZES[data_qtype.name.lower()]
                if data_torch.shape[-1] % block_size != 0:
                    logger.warning(
                        f"{new_name}: Can't quantize tensor with shape {data_torch.shape} to {data_qtype.name},"
                        " fallback to F16"
                    )
                    data_qtype = gguf.GGMLQuantizationType.F16

            if isinstance(data_qtype, bool) or data_qtype in [
                gguf.GGMLQuantizationType.F16,
                gguf.GGMLQuantizationType.BF16,
                gguf.GGMLQuantizationType.F32,
            ]:
                # sync with new version of gguf
                # data = data_torch.squeeze().cpu().numpy()

                # if data ends up empty, it means data_torch was a scalar tensor -> restore
                if len(data_torch.shape) == 0:
                    data = data_torch.numpy()
                try:
                    data = data_torch.squeeze().cpu().numpy()
                    data = gguf.quants.quantize(data, data_qtype)
                except gguf.QuantError as e:
                    logger.warning("%s, %s", e, "falling back to F16")
                    data_qtype = gguf.GGMLQuantizationType.F16
                    data = gguf.quants.quantize(data, data_qtype)
            else:
                # for deepseek v2
                if name.endswith("kv_b_proj.weight") and cls.model_arch.name == "DEEPSEEK2":
                    layer_name = name[: -len(".weight")]
                    module = get_module(cls.model, layer_name)
                    n_head_kv = cls.hparams["num_key_value_heads"]
                    v_head_dim = cls.hparams["v_head_dim"]
                    qk_nope_head_dim = cls.hparams["qk_nope_head_dim"]

                    attr_list = {"scale": None, "zp": None, "d_scale": None, "wmin": None, "d_wmin": None}
                    for attr in attr_list:
                        if hasattr(module, attr) or hasattr(module, "w_" + attr):
                            if attr in ["scale", "zp"]:
                                attr_tensor = getattr(module, attr)
                            else:
                                attr_tensor = getattr(module, "w_" + attr)
                            if attr_tensor is None or not isinstance(attr_tensor, torch.Tensor):
                                continue
                            kv_b = attr_tensor.view(n_head_kv, v_head_dim + qk_nope_head_dim, -1)
                            k_b, v_b = torch.split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
                            k_b = k_b.transpose(1, 2)

                            name_kb = modify_name.replace("kv_b_proj", "k_b_proj")
                            if new_name == cls.map_tensor_name(name_kb):
                                attr_list[attr] = k_b
                            else:
                                attr_list[attr] = v_b
                    attr_list["imatrix"] = module.imatrix if hasattr(module, "imatrix") else None
                    data = _quant_data_with_args(data_torch, data_qtype, device=device, **attr_list)

                # for MOE model
                elif len(data_torch.shape) == 3 and len(re.findall(r"\d+", name)) == 2:
                    new_data = []
                    for idx, arr in enumerate(data_torch):
                        arr_name = name.split(".")
                        for i in range(len(arr_name) - 1, -1, -1):
                            if arr_name[i].isdecimal() and int(arr_name[i]) == (data_torch.shape[0] - 1):
                                arr_name[i] = str(idx)
                        arr_name = ".".join(arr_name)
                        arr, data_qtype = _quant_data(
                            cls, arr, data_qtype, arr_name, modify_name, new_name, bid, device=device
                        )
                        new_data.append(arr)
                    data = np.array(new_data)
                    del new_data
                else:
                    data, data_qtype = _quant_data(
                        cls, data_torch, data_qtype, name, modify_name, new_name, bid, device=device
                    )

            shape = gguf.quant_shape_from_byte_shape(data.shape, data_qtype) if data.dtype == np.uint8 else data.shape

            # reverse shape to make it similar to the internal ggml dimension order
            shape_str = f"{{{', '.join(str(n) for n in reversed(shape))}}}"

            # n_dims is implicit in the shape
            logger.info(
                f"{f'%-{max_name_len}s' % f'{new_name},'} {old_dtype}" f" --> {data_qtype.name}, shape = {shape_str}"
            )
            if not (hasattr(cls, "current_packing_block") and cls.current_packing_block is not None):
                clear_memory(device_list=[orig_device])

            cls.gguf_writer.add_tensor(new_name, data, raw_dtype=data_qtype)

        # # save cpu memory, but slow
        if cls.low_cpu_mem_usage:
            for weight_name in clean_weight_list:
                if cls.model_arch == gguf.MODEL_ARCH.GEMMA:
                    continue
                module = get_module(cls.model, ".".join(weight_name.split(".")[:-1]))
                for key in ["scale", "zp", "d_scale", "wmin", "d_wmin", "imatrix"]:
                    if hasattr(module, key):
                        setattr(module, key, None)
                    if hasattr(module, "w_" + key):
                        setattr(module, "w_" + key, None)
                if cls.model_arch == gguf.MODEL_ARCH.LLAMA and "embed_tokens.weight" in weight_name:
                    continue
                clean_module_parameter(module, weight_name.split(".")[-1])
