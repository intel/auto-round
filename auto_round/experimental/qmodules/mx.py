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

__all__ = ["MXQuantLinear"]

SUPPORTED_HIGHER_DTYPE = [torch.bfloat16, torch.float16, torch.float32]


def _mx_qdq(tensor: torch.Tensor):
    # FIXME: (Yi) handle difference roudning method
    from auto_round.data_type.mxfp import quant_mx

    qdq_tesnor, _, _ = quant_mx(
        tensor=tensor,
        bits=4,
        group_size=32,
    )
    return qdq_tesnor


E8M0_EXPONENT_BIAS = 127


# https://github.com/pytorch/ao/blob/994a4ba6c869854fcaa6ca7e118fcbd75e6c28cc/torchao/prototype/mx_formats/mx_tensor.py#L337
def get_fp_scale(scale_e8m0):
    scale_e8m0 = scale_e8m0.view(torch.uint8)
    s_offset = scale_e8m0.to(torch.int16) - E8M0_EXPONENT_BIAS
    # TODO(later): it would be nice if there was a way to do the 2^x operation
    # in PyTorch without creating a tensor of twos
    two = torch.full(s_offset.size(), 2.0, device=scale_e8m0.device)
    # pow(two, s_offset) can be out of range of floating point formats.
    # TODO(later): handle this for float16 if we decide to support float16
    # scales.
    s_fp = torch.pow(two, s_offset)

    return s_fp


class MXQuantLinear(QModuleBase):
    hp_dtype = torch.bfloat16
    fp8_dtype = torch.float8_e4m3fn

    def __init__(
        self,
        in_features,
        out_features,
        weight: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        bias: Union[torch.Tensor, bool, None] = None,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = 32
        init_weight = torch.zeros((out_features, in_features), dtype=torch.float8_e4m3fn) if weight is None else weight
        self.weight = torch.nn.Parameter(init_weight, requires_grad=False)
        assert (
            dtype in SUPPORTED_HIGHER_DTYPE
        ), f"Expected dtype to be one of {SUPPORTED_HIGHER_DTYPE}, but got {dtype}."
        self.dtype = dtype
        if bias is not None:
            if isinstance(bias, bool):
                bias = torch.zeros((out_features,), dtype=dtype)
            self.bias = torch.nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter("bias", None)
        # weight: [out_features, in_features]
        # weight_scale: [out_features, in_features//32]
        # FIXME: Yi handle the padding case
        init_weight_scale = (
            torch.empty((out_features, in_features // self.group_size), dtype=torch.uint8)
            if weight_scale is None
            else weight_scale
        )
        self.register_buffer("weight_scale", init_weight_scale)

        self.pre_dequantized = False

    @classmethod
    def get_min_capability(cls) -> int:
        """
        Get minimum device capability.
        """
        # TODO: correct that config once we add mxfp8 op support.
        logger.warning_once("MXFP8 ops are not yet supported. Using capability 0.")
        return 0

    def process_weights_after_loading(self, layer: torch.nn.Module):
        pass

    @classmethod
    def from_original(cls, config, original_layer):
        """
        Create an `MXQuantLinear` layer from an original linear layer.
        """
        logger.warning_once("MXFP8 quantization is still in experimental stage, the inference speed might be slow.")
        device = original_layer.weight.device
        with torch.device(device):
            qdq_linear = cls(
                in_features=original_layer.in_features,
                out_features=original_layer.out_features,
                bias=original_layer.bias,
                dtype=original_layer.weight.dtype,
            )
            return qdq_linear

    @classmethod
    def _get_float_scale(cls, scale_e8m0: torch.Tensor) -> torch.Tensor:
        return get_fp_scale(scale_e8m0)

    def _dequant_mxfp_tensor(self, tensor_packed: torch.Tensor, tensor_scale: torch.Tensor):
        tensor_scale_float = self._get_float_scale(tensor_scale).to(self.dtype)
        tensor_packed = tensor_packed.to(self.dtype)
        original_shape = tensor_packed.shape
        tensor_packed = tensor_packed.reshape(-1, self.group_size)
        tensor_scale_float = tensor_scale_float.reshape(-1, 1)
        tensor_float = tensor_packed.to(self.dtype)
        tensor_dequant = tensor_float * tensor_scale_float
        tensor_dequant = tensor_dequant.reshape(original_shape)
        return tensor_dequant

    def dequant_weight_online(self):
        if self.pre_dequantized:
            return self.weight
        dq_weight = self._dequant_mxfp_tensor(self.weight, self.weight_scale)
        return dq_weight

    def pre_dequantize(self):
        if self.pre_dequantized:
            return
        dequant_weight = self.dequant_weight_online()
        del self.weight
        del self.weight_scale
        self.weight = torch.nn.Parameter(dequant_weight, requires_grad=False)
        self.pre_dequantized = True

    def qdq_input(self, activation: torch.Tensor):
        return _mx_qdq(activation)

    @torch.no_grad()
    def forward(self, bf16_input: torch.Tensor) -> torch.Tensor:
        qdq_input = self.qdq_input(bf16_input)
        qdq_weight = self.dequant_weight_online()
        out = torch.nn.functional.linear(qdq_input, qdq_weight, self.bias)
        return out
