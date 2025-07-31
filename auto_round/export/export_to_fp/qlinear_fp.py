# Copyright (c) 2023 Intel Corporation
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


# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import transformers

from auto_round.data_type.mxfp import FP32_EXPONENT_BIAS, FP32_MIN_NORMAL
from auto_round.data_type.nvfp import cast_to_fp4, get_reciprocal
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad

# from auto_round.utils import get_weight_compress_dtype
logger = getLogger(__name__)
E8M0_EXPONENT_BIAS = 127
E8M0_EXPONENT_NAN_VAL = 255

__all__ = ["QuantLinear"]

FLOAT_TO_E2M1 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
]


class QuantLinear(nn.Module):
    """
    MXFP quantized linear layer.
    """

    QUANT_TYPE = "MXFP"

    def __init__(
        self, bits, group_size, infeatures, outfeatures, bias, trainable=False, data_type="mx_fp8e4m3", **kwargs
    ):
        super().__init__()
        if bits not in [4, 8]:
            raise NotImplementedError("Only 4,8 bits are supported.")
        if "mx" in data_type and group_size != 32:
            raise NotImplementedError("Only group_size 32 are supported for mxfp.")
        if "nv" in data_type and group_size not in [16, 32]:
            raise NotImplementedError("Only group_size 16 are supported for nvfp.")

        if infeatures % 32 != 0 or outfeatures % 32 != 0:
            raise NotImplementedError("in_feature and out_feature must be divisible by 32.")
        self.is_mx = "mx" in data_type
        self.is_nv = "nv" in data_type
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.data_type = data_type
        self.sym = kwargs.get("sym", True)
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2**self.bits - 1
        self.act_bits = kwargs.get("act_bits", None)

        weight_name = "weight" if self.bits == 8 and self.is_mx else "weight_packed"
        ## TODO check the dtype of weight_packed and weight_scale
        self.register_buffer(
            weight_name,
            torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32),
        )
        if self.sym == False:
            ## TODO Currently only sym quant is supported for mxfp dtype. Is weight_zero_point param necessary？
            self.register_buffer(
                "weight_zero_point",
                torch.zeros(
                    (
                        math.ceil(infeatures / self.group_size),
                        outfeatures // 32 * self.bits,
                    ),
                    dtype=torch.int32,
                ),
            )
        self.register_buffer(
            "weight_scale",
            torch.zeros(
                (math.ceil(infeatures / self.group_size), outfeatures),
                dtype=torch.float16,  ## TODO update to correct scale dtype for different bits
            ),
        )
        if self.is_nv and self.bits == 4:
            self.register_buffer(
                "weight_global_scale",
                torch.zeros(
                    (1),
                    dtype=torch.float32,
                ),
            )
        if self.is_nv and self.act_bits <= 8:
            self.register_buffer(
                "input_global_scale",
                torch.zeros(
                    (1),
                    dtype=torch.float32,
                ),
            )
        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None

        self.trainable = trainable

    def post_init(self):
        pass

    def pack(self, linear, scales, zeros=None, g_idx=None, global_scale=None, input_global_scale=None):
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.xpu.is_available():
            device = "xpu:0"

        W = linear.weight.data.to(device).clone()  # TODO check is nesscessory
        if isinstance(linear, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.t()

        tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(linear.weight, self.group_size)
        if self.is_nv:
            assert global_scale is not None and global_scale.numel() == 1
            scaled_tensor = tensor.to(global_scale.dtype) * get_reciprocal(
                scales.reshape(tensor.shape[0], -1) * get_reciprocal(global_scale)
            )
            scaled_tensor = cast_to_fp4(torch.clamp(scaled_tensor, -6.0, 6.0))
        else:
            scaled_tensor = tensor / (2 ** scales.reshape(tensor.shape[0], -1))
        scaled_tensor = revert_tensor_by_pad(scaled_tensor, orig_shape=orig_shape, pad_len=pad_len)
        if self.is_mx:
            final_scale = (scales + E8M0_EXPONENT_BIAS).clamp(0, E8M0_EXPONENT_NAN_VAL).to(torch.uint8)
        else:
            final_scale = scales.to(torch.float8_e4m3fn)
        self.weight_scale = final_scale
        # self.weight =  get_compressed_weight(scaled_tensor, self.bits, self.data_type) ## TODO
        if self.bits == 8:
            compress_dtype = torch.float8_e4m3fn
            self.weight = scaled_tensor.to(compress_dtype)
        else:
            compress_dtype = torch.uint8
            self.weight_packed = self.pack_fp4_to_uint8(scaled_tensor)

        if global_scale is not None:
            self.weight_global_scale = global_scale.to(torch.float32)

        if input_global_scale is not None:
            self.input_global_scale = input_global_scale.to(torch.float32)
        return

    def pack_fp4_to_uint8(self, x: torch.Tensor) -> torch.Tensor:
        """
        Packs a tensor with values in the fp4 range into uint8.
        As there are 16 valid fp4 values, two fp4 values can be
        packed into one uint8. Each fp4 value is mapped to its
        particular index (e.g. 0.5 is mapped to index 1, 6.0 is mapped
        to index 7) which is then represented using 4 bits. Consecutive
        pairs of 4 bits are then packed into an uint8.

        :param x: tensor to pack
        returns: a packed tensor in uint8
        """

        m, n = x.shape
        device = x.device

        # Create lookup table for FP4 values to indices
        # Map the absolute values to 0-7 indices
        kE2M1 = torch.tensor(FLOAT_TO_E2M1, device=device, dtype=x.dtype)

        # Find closest valid FP4 value index for each element
        abs_x = torch.abs(x)
        abs_indices = torch.zeros_like(abs_x, dtype=torch.long)
        for i, val in enumerate(kE2M1):  # TODO any optimize?
            abs_indices = torch.where(torch.isclose(abs_x, val), i, abs_indices)

        # Apply sign bit (bit 3) to get final 4-bit representation
        indices = abs_indices + (torch.signbit(x) << 3).to(torch.long)

        # Reshape to prepare for packing pairs of values
        indices = indices.reshape(-1)

        # Handle odd length by padding if necessary
        if indices.numel() % 2 != 0:
            indices = torch.cat([indices, torch.zeros(1, dtype=torch.long, device=device)])

        # Reshape to pair consecutive elements
        indices = indices.reshape(-1, 2)

        # Pack pairs of 4-bit values into 8-bit values
        packed = (indices[:, 0] | (indices[:, 1] << 4)).to(torch.uint8)

        return packed.reshape(m, n // 2)

    # def get_fp_scale(self, scale_e8m0):
    #     E8M0_EXPONENT_BIAS = 127
    #     E8M0_EXPONENT_NAN_VAL = 255

    #     scale_e8m0 = scale_e8m0.view(torch.uint8)
    #     s_offset = scale_e8m0.to(torch.int16) - E8M0_EXPONENT_BIAS
    #     # TODO(later): it would be nice if there was a way to do the 2^x operation
    #     # in PyTorch without creating a tensor of twos
    #     two = torch.full(s_offset.size(), 2.0, device=scale_e8m0.device)
    #     # pow(two, s_offset) can be out of range of floating point formats.
    #     # TODO(later): handle this for float16 if we decide to support float16
    #     # scales.
    #     s_fp = torch.pow(two, s_offset)

    #     # If a block exponent was 255, set values of that block to NaN
    #     s_fp = torch.where(scale_e8m0 != E8M0_EXPONENT_NAN_VAL, s_fp, float("nan"))

    #     return s_fp

    # def get_compressed_weight(self, tensor, bits, data_type):
    #     data_type = str(data_type)
    #     compress_dtype = None
    #     if self.is_mx:
    #         if bits == 4:

    #     if self.is_nv:
    #         pass ## TODO
