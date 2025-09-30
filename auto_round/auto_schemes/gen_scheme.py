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

from typing import Iterable, Union

import torch

from auto_round import AutoScheme
from auto_round.utils import get_layer_features


class GenScheme:
    def __init__(
        self,
        auto_scheme: AutoScheme,
        model: torch.nn.Module,
        quant_layer_names: Iterable[str],
        fixed_layer_scheme: dict[str, dict],
        scale_dtype: str = "fp16",
        dataset="pile-10k",
    ):
        self.auto_scheme = auto_scheme
        self.model = model
        self.quant_layer_names = quant_layer_names
        self.fixed_layer_scheme = fixed_layer_scheme
        self.scale_dtype = scale_dtype
        self.dataset = dataset

    def _get_min_max_avg_bits(self) -> tuple[float, float]:
        pass

    # not validate yet
    def get_layer_bits(self, layer):
        weight = layer.weight
        n_param = weight.numel()
        weight_bits = getattr(layer, "bits", 16)
        group_size = getattr(layer, "group_size", 128)
        super_group_size = getattr(layer, "super_group_size", None)
        super_weight_bits = getattr(layer, "super_bits", None)

        # Main quantization cost
        weight_total_bits = weight_bits * n_param
        if weight_bits >= 16:  # Unquantized layer
            return weight_total_bits, 16

        in_features, output_features = get_layer_features(layer)
        # Determine number of groups
        if group_size > 0:  # group-wise
            n_group = output_features * (in_features + group_size - 1) // group_size
        elif group_size == 0:  # per-tensor
            n_group = 1
        elif group_size == -1:  # per-channel
            n_group = output_features  # out_channels
        else:
            raise ValueError(f"Invalid group_size {group_size}")
        aux_total_bits = 0
        if not super_group_size:
            # Scale and zero point bitwidths
            scale_bits = 16
            zp_bits = weight_bits if not super_group_size else 32  # default: same as weight_bits
            # Overhead from scales and zero points
            aux_total_bits = n_group * (scale_bits + zp_bits)

        # Double quantization case
        if super_group_size:
            # Number of super-groups
            aux_total_bits += n_group * super_weight_bits * 2  # sclae and min int count
            n_super_group = (n_group + super_group_size - 1) // super_group_size
            aux_total_bits += n_super_group * 32 * 2  # double quant scale and min_v

        total_bits = weight_total_bits + aux_total_bits
        avg_bits = total_bits / n_param
        return total_bits, avg_bits
