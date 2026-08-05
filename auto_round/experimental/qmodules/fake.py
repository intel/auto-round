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

from typing import Optional

import torch

from auto_round.data_type.utils import get_quant_func
from auto_round.experimental.qmodules.base import QModuleBase
from auto_round.schemes import QuantizationScheme

__all__ = ["FakeActQuantLinear"]


class FakeActQuantLinear(QModuleBase):
    """Linear with high-precision QDQ weights and runtime activation QDQ."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: QuantizationScheme,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        if weight is None:
            weight = torch.empty((out_features, in_features), dtype=dtype)
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = torch.nn.Parameter(bias, requires_grad=False)

    @classmethod
    def from_original(cls, config: QuantizationScheme, original_layer: torch.nn.Linear):
        return cls(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            config=config,
            weight=original_layer.weight,
            bias=original_layer.bias,
            dtype=original_layer.weight.dtype,
        )

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    def process_weights_after_loading(self, layer: torch.nn.Module):
        return

    def post_init(self):
        return

    def qdq_input(self, activation: torch.Tensor) -> torch.Tensor:
        quant_func, _ = get_quant_func(
            dtype=self.config.act_data_type,
            bits=self.config.act_bits,
            sym=self.config.act_sym,
        )
        qdq_activation, _, _ = quant_func(
            tensor=activation,
            bits=self.config.act_bits,
            group_size=self.config.act_group_size,
        )
        return qdq_activation.to(activation.dtype)

    @torch.inference_mode()
    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        qdq_activation = self.qdq_input(activation)
        return torch.nn.functional.linear(qdq_activation, self.weight.to(qdq_activation.dtype), self.bias)