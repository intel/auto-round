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

from auto_round.data_type.nvfp import get_reciprocal, ref_nvfp4_quant
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad
from auto_round.experimental.qmodules.base import QModuleBase
from auto_round.experimental.qmodules.fp4_utils import unpack_fp4_from_uint8
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme

__all__ = ["NVFP4QuantLinear"]


# Adapted from auto_round/data_type/nvfp.py
def _nv_fp4_with_static_gs(
    tensor: torch.Tensor, global_scale: torch.Tensor, bits: int = 4, group_size: int = 16
) -> tuple[torch.Tensor, torch.Tensor, None]:
    orig_dtype = tensor.dtype
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)

    global_scale = global_scale.to(tensor.device)
    qdq_res, scale = ref_nvfp4_quant(tensor, global_scale, group_size, v=0)
    qdq_res = revert_tensor_by_pad(qdq_res, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_res.to(orig_dtype), scale, None


def _nvfp4_qdq(tensor: torch.Tensor, config: QuantizationScheme, global_scale: torch.Tensor) -> torch.Tensor:
    qdq_tensor, scales, _ = _nv_fp4_with_static_gs(
        tensor=tensor, global_scale=global_scale, bits=config.act_bits, group_size=config.act_group_size
    )
    return qdq_tensor


class NVFP4QuantLinear(QModuleBase):
    """
    Quantized linear layer using NVFP4 quantization scheme.
    """

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
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = 16
        self.config = config
        self.dtype = dtype
        self.pre_dequantized = False

        # Validate dtype
        assert (
            dtype in self.SUPPORTED_COMPUTE_DTYPE
        ), f"Expected dtype to be one of {self.SUPPORTED_COMPUTE_DTYPE}, but got {dtype}."

        # check group size
        assert self.group_size == config.group_size, f"Group size mismatch: {self.group_size} vs {config.group_size}"
        assert (
            self.group_size == config.act_group_size
        ), f"Group size mismatch: {self.group_size} vs {config.act_group_size}"

        # Initialize weights
        init_weight = self.initialize_weights(weight)
        self.register_buffer("weight_packed", init_weight)

        # Initialize bias
        if bias is not None:
            if isinstance(bias, bool):
                bias = torch.zeros((out_features,), dtype=dtype)
            self.bias = torch.nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # Initialize weight scale
        init_weight_scale = (
            torch.empty((out_features, in_features // self.group_size), dtype=torch.float8_e4m3fn)
            if weight_scale is None
            else weight_scale
        )
        self.register_buffer("weight_scale", init_weight_scale)

        self.register_buffer(
            "weight_global_scale",
            torch.zeros(
                (1,),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "input_global_scale",
            torch.zeros(
                (1,),
                dtype=torch.float32,
            ),
        )

    @staticmethod
    def _convert_global_scale_to_float32(state_dict: dict[str, torch.Tensor], name: str):
        if name not in state_dict or state_dict[name].dtype == torch.float32:
            return
        original_scale = state_dict[name]
        state_dict[name] = original_scale.to(torch.float32)
        logger.warning_once("Forcing global scale to float32 for better precision.")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self._convert_global_scale_to_float32(state_dict, "weight_global_scale")
        self._convert_global_scale_to_float32(state_dict, "input_global_scale")
        return super().load_state_dict(state_dict, strict, assign)

    def initialize_weights(self, weight: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Initialize weights.
        """
        weight_dtype = torch.uint8
        weight_in_features = self.in_features // 2
        return torch.zeros((self.out_features, weight_in_features), dtype=weight_dtype) if weight is None else weight

    @classmethod
    def get_min_capability(cls) -> int:
        """
        Get minimum device capability.
        """
        logger.warning_once("NVFP4 quantization is still in experimental stage, the inference speed might be slow.")
        return 0

    def _dequant_nvfp4_tensor(
        self, packed_data: torch.Tensor, scale: torch.Tensor, target_dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        weight_global_scale = self.weight_global_scale
        unpacked_data = self.unpack_data(packed_data)
        unpacked_data = unpacked_data.to(target_dtype) * get_reciprocal(weight_global_scale).to(target_dtype)
        scale_float = scale.to(target_dtype)
        original_shape = unpacked_data.shape
        unpacked_data = unpacked_data.reshape(-1, self.group_size)
        scale_float = scale_float.reshape(-1, 1)
        data_dequant = unpacked_data * scale_float
        data_dequant = data_dequant.reshape(original_shape)
        return data_dequant

    def dequant_weight_online(self) -> torch.Tensor:
        dq_weight = self._dequant_nvfp4_tensor(self.weight_packed, self.weight_scale)
        return dq_weight

    @property
    def weight(self) -> torch.Tensor:
        if not hasattr(self, '_cached_weight') or self._cached_weight is None:
            self._cached_weight = self.dequant_weight_online()
        return self._cached_weight

    def qdq_input(self, activation: torch.Tensor):
        original_dtype = activation.dtype
        temp_qdq_act = _nvfp4_qdq(activation.to(torch.float32), self.config, self.input_global_scale)
        return temp_qdq_act.to(original_dtype)

    @torch.inference_mode()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        qdq_input = self.qdq_input(input)
        qdq_weight = self.dequant_weight_online()
        qdq_weight = qdq_weight.to(qdq_input.dtype)
        out = torch.nn.functional.linear(qdq_input, qdq_weight, self.bias)
        return out

    @classmethod
    def from_original(cls, config: Optional[QuantizationScheme], original_layer: torch.nn.Linear):
        """
        Create an `NVFPQuantLinear` layer from an original linear layer.
        """
        logger.warning_once("NVFP4 quantization is still in experimental stage, the inference speed might be slow.")
        qdq_linear = cls(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            config=config,
            bias=original_layer.bias,
            dtype=original_layer.weight.dtype,
        )
        return qdq_linear

    def unpack_data(self, packed_data: torch.Tensor) -> torch.Tensor:
        m, half_n = packed_data.shape
        unpacked_data = unpack_fp4_from_uint8(packed_data, m, half_n * 2, dtype=self.dtype)
        return unpacked_data
