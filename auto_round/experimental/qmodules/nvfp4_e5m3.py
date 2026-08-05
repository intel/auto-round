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

from typing import Optional, Union

import torch

from auto_round.data_type.nvfp import e5m3_to_float_tensor, fp4_v2
from auto_round.experimental.qmodules.base import QModuleBase
from auto_round.experimental.qmodules.fp4_utils import unpack_fp4_from_uint8
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme

__all__ = ["NVFP4E5M3QuantLinear"]


class NVFP4E5M3QuantLinear(QModuleBase):
    """FP4 E2M1 weights and activations with unsigned E5M3 block scales."""

    SUPPORTED_COMPUTE_DTYPE = [torch.bfloat16, torch.float16, torch.float32]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: QuantizationScheme,
        weight: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        bias: Union[torch.Tensor, bool, None] = None,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        assert dtype in self.SUPPORTED_COMPUTE_DTYPE
        assert config.group_size == 16 and config.act_group_size == 16
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = config.group_size
        self.config = config
        self.dtype = dtype
        self._cached_weight = None

        packed_weight = (
            torch.zeros((out_features, in_features // 2), dtype=torch.uint8) if weight is None else weight
        )
        self.register_buffer("weight_packed", packed_weight)
        scale = (
            torch.empty((out_features, in_features // self.group_size), dtype=torch.uint8)
            if weight_scale is None
            else weight_scale
        )
        self.register_buffer("weight_scale", scale)

        if bias is not None:
            if isinstance(bias, bool):
                bias = torch.zeros((out_features,), dtype=dtype)
            self.bias = torch.nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @classmethod
    def get_min_capability(cls) -> int:
        logger.warning_once("NVFP4 E5M3 quantization uses reference PyTorch inference and may be slow.")
        return 0

    def dequant_weight_online(self) -> torch.Tensor:
        unpacked = unpack_fp4_from_uint8(
            self.weight_packed, self.out_features, self.in_features, dtype=self.dtype
        ).to(torch.float32)
        scale = e5m3_to_float_tensor(self.weight_scale).reshape(-1, 1)
        return (unpacked.reshape(-1, self.group_size) * scale).reshape(self.out_features, self.in_features)

    @property
    def weight(self) -> torch.Tensor:
        if self._cached_weight is None:
            self._cached_weight = self.dequant_weight_online()
        return self._cached_weight

    def qdq_input(self, activation: torch.Tensor) -> torch.Tensor:
        original_dtype = activation.dtype
        qdq_activation, _, _ = fp4_v2(
            activation.to(torch.float32), bits=self.config.act_bits, group_size=self.config.act_group_size
        )
        return qdq_activation.to(original_dtype)

    @torch.inference_mode()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        qdq_input = self.qdq_input(input)
        return torch.nn.functional.linear(qdq_input, self.dequant_weight_online().to(qdq_input.dtype), self.bias)

    @classmethod
    def from_original(cls, config: QuantizationScheme, original_layer: torch.nn.Linear):
        return cls(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            config=config,
            bias=original_layer.bias is not None,
            dtype=original_layer.weight.dtype,
        )
