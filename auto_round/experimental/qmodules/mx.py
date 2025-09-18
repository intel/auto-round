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

from auto_round.data_type.fp4_utils import unpack_fp4_from_uint8
from auto_round.data_type.mxfp import QUANT_FUNC_WITH_DTYPE
from auto_round.data_type.utils import get_quant_func
from auto_round.experimental.qmodules.base import QModuleBase
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme

__all__ = ["MXQuantLinear"]

SUPPORTED_HIGHER_DTYPE = [torch.bfloat16, torch.float16, torch.float32]
E8M0_EXPONENT_BIAS = 127


def _mx_qdq(tensor: torch.Tensor, config: QuantizationScheme):
    qdq_func, _ = get_quant_func(dtype=config.act_data_type, bits=config.act_bits, sym=True)
    qdq_tesnor, shared_exp, _ = qdq_func(tensor=tensor, bits=config.act_bits, group_size=config.act_group_size)
    return qdq_tesnor


# https://github.com/pytorch/ao/blob/994a4ba6c869854fcaa6ca7e118fcbd75e6c28cc/torchao/prototype/mx_formats/mx_tensor.py#L337
def get_fp_scale(scale_e8m0):
    scale_e8m0 = scale_e8m0.view(torch.uint8)
    s_offset = scale_e8m0.to(torch.int16) - E8M0_EXPONENT_BIAS
    two = torch.full(s_offset.size(), 2.0, device=scale_e8m0.device)
    # TODO(later): handle this for float16 if we decide to support float16
    s_fp = torch.pow(two, s_offset)

    return s_fp


class MXQuantLinear(QModuleBase):
    """
    Quantized linear layer using MXFP8/MXFP4 quantization scheme.
    """

    def __init__(
        self,
        in_features,
        out_features,
        config: QuantizationScheme,
        weight: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        bias: Union[torch.Tensor, bool, None] = None,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = 32
        if config.act_bits == 4:
            weight_dtype = torch.uint8
            weight_in_features = in_features // 2
        else:
            weight_dtype = torch.float8_e4m3fn
            weight_in_features = in_features
        init_weight = torch.zeros((out_features, weight_in_features), dtype=weight_dtype) if weight is None else weight
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
        # FIXME: Yi handle the padding case
        init_weight_scale = (
            torch.empty((out_features, in_features // self.group_size), dtype=torch.uint8)
            if weight_scale is None
            else weight_scale
        )
        self.register_buffer("weight_scale", init_weight_scale)
        self.config = config
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
    def from_original(cls, config: Optional[QuantizationScheme], original_layer: torch.nn.Linear):
        """
        Create an `MXQuantLinear` layer from an original linear layer.
        """
        logger.warning_once("MXFP quantization is still in experimental stage, the inference speed might be slow.")
        device = original_layer.weight.device
        with torch.device(device):
            qdq_linear = cls(
                in_features=original_layer.in_features,
                out_features=original_layer.out_features,
                config=config,
                bias=original_layer.bias,
                dtype=original_layer.weight.dtype,
            )
            return qdq_linear

    @classmethod
    def _get_float_scale(cls, scale_e8m0: torch.Tensor) -> torch.Tensor:
        return get_fp_scale(scale_e8m0)

    def unpack_data(self, packed_data: torch.Tensor):
        # For MXFP8
        if self.config.bits == 8:
            return packed_data.to(self.dtype)
        elif self.config.bits == 4:
            # For MXFP4
            m, half_n = packed_data.shape
            unpacked_data = unpack_fp4_from_uint8(packed_data, m, half_n * 2, dtype=self.dtype)
            return unpacked_data
        else:
            raise NotImplementedError("Only 4 and 8 bits are supported for MXFP.")

    def dequant_mx_tensor(
        self, packed_data: torch.Tensor, scale: torch.Tensor, target_dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        scale_float = self._get_float_scale(scale).to(target_dtype)
        unpacked_data = self.unpack_data(packed_data)
        original_shape = unpacked_data.shape
        unpacked_data = unpacked_data.reshape(-1, self.group_size)
        scale_float = scale_float.reshape(-1, 1)
        data_float = unpacked_data.to(target_dtype)
        data_dequant = data_float * scale_float
        data_dequant = data_dequant.reshape(original_shape)
        return data_dequant

    def dequant_weight_online(self):
        if self.pre_dequantized:
            return self.weight
        dq_weight = self.dequant_mx_tensor(self.weight, self.weight_scale)
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
        return _mx_qdq(activation, self.config)

    @torch.inference_mode()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        qdq_input = self.qdq_input(input)
        qdq_weight = self.dequant_weight_online()
        out = torch.nn.functional.linear(qdq_input, qdq_weight, self.bias)
        return out
