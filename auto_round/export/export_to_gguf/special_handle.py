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
import json
import os
from functools import partial
from typing import Iterable

import torch
from safetensors import safe_open
from torch import Tensor

from auto_round.utils import download_or_get_path


def handle_special_model(cls, model_architecture):
    if model_architecture == "GptOssForCausalLM":
        cls.generate_extra_tensors = partial(gptoss_generate_extra_tensors, cls)
        cls.modify_tensors = partial(gptoss_modify_tensors, cls)
    return cls


def get_tensor_from_file(dir_path, tensor_name):
    if not os.path.isdir(dir_path):
        dir_path = download_or_get_path(dir_path)
    INDEX_FILE = "model.safetensors.index.json"
    # get filename
    if INDEX_FILE in os.listdir(dir_path):
        with open(os.path.join(dir_path, INDEX_FILE)) as f:
            tensor_index = json.load(f)
            filename = tensor_index["weight_map"][tensor_name]
    else:
        filename = "model.safetensors"

    # get tensor
    f = safe_open(os.path.join(dir_path, filename), framework="pt")
    return f.get_tensor(tensor_name)


GPTOSS_RELOAD = True


def gptoss_generate_extra_tensors(cls) -> Iterable[tuple[str, Tensor]]:
    blocks0: Tensor = torch.zeros(1)
    blocks1: Tensor = torch.zeros(1)
    found_mxfp4_tensors = False

    # we assume that tensors are loaded in the correct order
    def repack(name, data_torch, blocks0, blocks1):
        if "mlp.experts.down_proj_blocks" in name:
            blocks0 = data_torch
        elif "mlp.experts.down_proj_scales" in name:
            new_name = cls.map_tensor_name(name.replace("_scales", ".weight"))
            for t in cls.gguf_writer.tensors:
                if new_name in t.keys():
                    return blocks0, blocks1
            cls.repack_mxfp4(new_name, blocks0, data_torch)
            found_mxfp4_tensors = True
        elif "mlp.experts.gate_up_proj_blocks" in name:
            blocks0, blocks1 = data_torch[:, ::2, :, :], data_torch[:, 1::2, :, :]
        elif "mlp.experts.gate_up_proj_scales" in name:
            scales0, scales1 = data_torch[:, ::2, :], data_torch[:, 1::2, :]
            new_name_gate = cls.map_tensor_name(name.replace("gate_up_proj_scales", "gate_proj.weight"))
            new_name_up = cls.map_tensor_name(name.replace("gate_up_proj_scales", "up_proj.weight"))
            for t in cls.gguf_writer.tensors:
                if new_name_gate in t.keys() or new_name_up in t.keys():
                    return blocks0, blocks1
            cls.repack_mxfp4(new_name_gate, blocks0, scales0)
            cls.repack_mxfp4(new_name_up, blocks1, scales1)
            found_mxfp4_tensors = True
        return blocks0, blocks1

    for name, data_torch in cls.get_tensors():
        if GPTOSS_RELOAD and (name.endswith("mlp.experts.down_proj") or name.endswith("mlp.experts.gate_up_proj")):
            block_name = name + "_blocks"
            block_data_torch = get_tensor_from_file(cls.model.name_or_path, block_name)
            blocks0, blocks1 = repack(block_name, block_data_torch, blocks0, blocks1)
            scale_name = name + "_scales"
            scale_data_torch = get_tensor_from_file(cls.model.name_or_path, scale_name)
            blocks0, blocks1 = repack(scale_name, scale_data_torch, blocks0, blocks1)
        else:
            blocks0, blocks1 = repack(name, data_torch, blocks0, blocks1)

    # convert to bf16 model, foud_mxfp4_tensors is False
    # if not found_mxfp4_tensors:
    #     raise ValueError("No MXFP4 tensors found in the model. Please make sure you are using MXFP4 model.")
    return []


def gptoss_modify_tensors(cls, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
    del bid  # unused

    if "sinks" in name:
        name += ".weight"

    # correct naming for down_proj
    if "down_proj" in name:
        if name.endswith("_bias"):
            name = name.replace("down_proj_bias", "down_proj.bias")
        else:
            if GPTOSS_RELOAD:
                return []
            if data_torch.dtype != torch.uint8:
                return [(cls.map_tensor_name(name + ".weight"), data_torch)]
            return []

    # split the gate_up into gate and up
    if "gate_up_proj" in name:
        if name.endswith("_bias"):
            name_up = name.replace("gate_up_proj_bias", "up_proj.bias")
            name_gate = name.replace("gate_up_proj_bias", "gate_proj.bias")
            gate_proj_bias, up_proj_bias = data_torch[..., ::2], data_torch[..., 1::2]
            return [(cls.map_tensor_name(name_gate), gate_proj_bias), (cls.map_tensor_name(name_up), up_proj_bias)]
        else:
            if GPTOSS_RELOAD:
                return []
            if data_torch.dtype != torch.uint8:
                new_name_gate = cls.map_tensor_name(name.replace("gate_up_proj", "gate_proj.weight"))
                new_name_up = cls.map_tensor_name(name.replace("gate_up_proj", "up_proj.weight"))
                gate_data, up_data = data_torch[:, ::2, :], data_torch[:, 1::2, :]
                # gate_data = gate_data.reshape(-1, 2880, 2880)
                # up_data = up_data.reshape(-1, 2880, 2880)
                return [(new_name_gate, gate_data), (new_name_up, up_data)]
            return []

    return [(cls.map_tensor_name(name), data_torch)]
