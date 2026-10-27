# Copyright (c) 2026 Intel Corporation
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

import torch.nn as nn
from tqdm import tqdm

from auto_round.utils import unsupported_meta_device


def prepare_model(model, inplace, *, device=None):
    """Copy before moving, keeping the caller's model isolated when requested."""
    prepared = model if inplace else copy.deepcopy(model)
    if device is not None and not unsupported_meta_device(prepared):
        prepared = prepared.to(device)
    return prepared


def pack_layers(model, layer_names, pack_layer, device=None):
    """Pack a stable snapshot of names unless the model contains meta tensors."""
    if not unsupported_meta_device(model):
        for name in tqdm(list(layer_names), desc="packing", leave=True):
            pack_layer(name, model, device=device)
    return model


def get_autogptq_packing_qlinear(backend, bits=4, group_size=128, sym=False):
    """
    Configures and returns a QuantLinear class based on the specified backend and parameters.

    Args:
        backend (str): The backend to be used for quantization. Supported values include "qigen", "triton", "marlin",
                       "exllama", and "cuda".
        bits (int, optional): The number of bits for quantization. Default is 4.
        group_size (int, optional): The group size for quantization. Default is 128.
        sym (bool, optional): Flag indicating whether to use symmetric quantization. Default is False.

    Returns:
        class: The dynamically imported QuantLinear class configured according to the specified parameters.
    """

    from auto_round_extension.torch.qlinear_torch_zp import QuantLinear

    return QuantLinear


def release_layer_safely(layer: nn.Module):
    """
    Safely releases the weight and bias tensors of a layer to free memory.
    Handles the case where attributes might not exist or are already None.
    """
    for attr in ["weight", "bias"]:
        if hasattr(layer, attr):
            tensor = getattr(layer, attr)
            if tensor is not None:
                # Detach and delete to avoid memory leaks
                tensor.detach_()
                setattr(layer, attr, None)
