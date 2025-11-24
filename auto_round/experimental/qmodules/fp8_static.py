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


from typing import Optional, Union

import torch

from auto_round.experimental.qmodules.base import QModuleBase
from auto_round.utils import logger

__all__ = ["WeightFP8ActFP8StaticQuantLinear"]


def _quant_tensor_to_fp8_with_scale(tensor: torch.Tensor, scale: torch.Tensor):
    FULL_RANGE = torch.finfo(torch.float8_e4m3fn).max
    qtensor = tensor / scale
    clipped_qtensor = torch.clamp(qtensor, -FULL_RANGE, FULL_RANGE)
    clipped_qtensor_fp8 = clipped_qtensor.to(torch.float8_e4m3fn)
    return scale, clipped_qtensor_fp8


class WeightFP8ActFP8StaticQuantLinear(QModuleBase):
    hp_dtype = torch.bfloat16
    fp8_dtype = torch.float8_e4m3fn

    def __init__(
        self,
        in_features,
        out_features,
        weight: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        bias: Union[torch.Tensor, bool, None] = None,
        input_scale: Optional[torch.Tensor] = None,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_weight = torch.zeros((out_features, in_features), dtype=dtype) if weight is None else weight
        self.weight = torch.nn.Parameter(init_weight, requires_grad=False)
        self.dtype = dtype
        if bias is not None:
            if isinstance(bias, bool):
                bias = torch.zeros((out_features,), dtype=dtype)
            self.bias = torch.nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter("bias", None)
        init_weight_scale = torch.empty((out_features), dtype=dtype) if weight_scale is None else weight_scale
        self.register_buffer("weight_scale", init_weight_scale.to(dtype))

        init_input_scale = torch.zeros((1), dtype=dtype) if input_scale is None else input_scale
        self.register_buffer("input_scale", init_input_scale.to(dtype))
        self.pre_dequantized = False

    @classmethod
    def get_min_capability(cls) -> int:
        """
        Get minimum device capability.
        """
        # TODO: correct that config once we add fp8 op support.
        logger.warning_once("FP8 ops are not yet supported. Using capability 0.")
        return 0

    def process_weights_after_loading(self, layer: torch.nn.Module):
        pass

    @classmethod
    def from_original(cls, config, original_layer):
        """
        Create an `WeightFP8ActFP8StaticQuantLinear` layer from an original linear layer.
        """
        logger.warning_once(
            "FP8 static quantization is still in experimental stage, the inference speed might be slow."
        )
        device = original_layer.weight.device
        with torch.device(device):
            qdq_linear = cls(
                in_features=original_layer.in_features,
                out_features=original_layer.out_features,
                bias=original_layer.bias,
            )
            return qdq_linear

    def dequant_weight_online(self):
        if self.pre_dequantized:
            return self.weight
        qdq_weight = self.weight.to(self.dtype) * self.weight_scale.unsqueeze(1)
        return qdq_weight

    def pre_dequantize(self):
        if self.pre_dequantized:
            return
        dequant_weight = self.dequant_weight_online()
        del self.weight
        del self.weight_scale
        self.weight = torch.nn.Parameter(dequant_weight, requires_grad=False)
        self.pre_dequantized = True

    def qdq_input(self, bf16_input: torch.Tensor):
        input_scale, input_fp8 = _quant_tensor_to_fp8_with_scale(bf16_input, self.input_scale.data)
        qdq_input_bf16 = input_fp8.to(self.dtype) * input_scale
        return qdq_input_bf16

    @torch.no_grad()
    def forward(self, bf16_input: torch.Tensor) -> torch.Tensor:
        original_dtype = bf16_input.dtype
        qdq_input = self.qdq_input(bf16_input)
        qdq_weight = self.dequant_weight_online()
        out = torch.nn.functional.linear(qdq_input.to(original_dtype), qdq_weight.to(original_dtype), self.bias)
        return out
