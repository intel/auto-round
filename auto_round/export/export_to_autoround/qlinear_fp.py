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

import numpy as np
import torch
import torch.nn as nn
import transformers

import auto_round.envs as envs
from auto_round.compressors.utils import BackendDataType, is_mx_fp, is_nv_fp
from auto_round.data_type.mxfp import FP32_EXPONENT_BIAS, FP32_MIN_NORMAL
from auto_round.data_type.nvfp import cast_to_fp4, get_reciprocal
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad
from auto_round.utils import get_packing_device, logger

# from auto_round.utils import get_weight_compress_dtype
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
        self.is_mx = is_mx_fp(data_type)
        self.is_nv = is_nv_fp(data_type)
        if self.is_mx:
            if group_size != 32:
                raise NotImplementedError(f"Only group_size 32 are supported for {BackendDataType.MX_FP} data type.")
            if infeatures % group_size != 0:
                raise NotImplementedError(
                    f"in_feature must be divisible by {group_size} for {BackendDataType.MX_FP} data type."
                )
        if self.is_nv:
            if group_size % 16 != 0:
                raise NotImplementedError(f"Only group_size 16 are supported for {BackendDataType.NV_FP} data type.")
            if infeatures % group_size != 0:
                raise NotImplementedError(
                    f"in_feature must be divisible by {group_size} for {BackendDataType.NV_FP} data type."
                )
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.data_type = data_type
        self.sym = kwargs.get("sym", True)
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2**self.bits - 1
        self.act_bits = kwargs.get("act_bits", None)

        weight_name = "weight" if self.bits == 8 and self.is_mx else "weight_packed"
        weight_infeatures = infeatures if self.bits == 8 else infeatures // 2
        weight_dtype = torch.float8_e4m3fn if self.bits == 8 else torch.uint8
        ## TODO check the dtype of weight_packed and weight_scale
        self.register_buffer(
            weight_name,
            torch.zeros((outfeatures, weight_infeatures), dtype=weight_dtype),
        )
        self.register_buffer(
            "weight_scale",
            torch.zeros(
                (outfeatures, math.ceil(infeatures / self.group_size)),
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

    def pack(self, linear, scales, zeros=None, g_idx=None, global_scale=None, input_global_scale=None, device=None):
        device = get_packing_device(device)
        if getattr(linear, "bias", None) is not None:
            self.bias = linear.bias.detach().to(torch.float16)

        W = linear.weight.data.detach().to(device)
        if type(linear) == nn.Conv2d:
            W = W.flatten(1)
        if type(linear) == transformers.pytorch_utils.Conv1D:
            W = W.t()

        tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(W, self.group_size)
        scales = scales.to(device)
        if self.is_nv:
            assert global_scale is not None and global_scale.numel() == 1
            global_scale = global_scale.reshape([1])
            global_scale = global_scale.to(device)
            scaled_tensor = tensor.to(global_scale.dtype) * get_reciprocal(
                scales.reshape(tensor.shape[0], -1) * get_reciprocal(global_scale)
            )
            scaled_tensor.clamp_(-6.0, 6.0)
            scaled_tensor = cast_to_fp4(scaled_tensor)
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
            self.weight_packed = pack_fp4_to_uint8(scaled_tensor)

        if global_scale is not None:
            self.weight_global_scale = global_scale.to(torch.float32).to(device)

        if input_global_scale is not None:
            # TODO: the shape of `input_global_scale` is [] in some cases — need to investigate why.
            self.input_global_scale = input_global_scale.to(torch.float32).to(device).reshape([1])
        return


def pack_fp4_to_uint8(scaled_tensor: torch.Tensor):
    if scaled_tensor.device.type == "cuda":
        return pack_fp4_to_uint8_cuda(scaled_tensor)
    else:
        return pack_fp4_to_uint8_cpu(scaled_tensor)


# The torch.compile with dynamic=True is incompatible with multiple threads
# https://github.com/pytorch/pytorch/issues/126024
@torch.compiler.disable()
def pack_fp4_to_uint8_cpu(x: torch.Tensor) -> torch.Tensor:
    return _pack_fp4_to_uint8(x)


# Adapted from https://github.com/neuralmagic/compressed-tensors/pull/400


def _get_packing_fn():
    if envs.AR_ENABLE_COMPILE_PACKING:
        logger.warning_once(
            "Compiled FP4 to UINT8 packing may be incompatible with multi-threading."
            " Disable it by setting AR_ENABLE_COMPILE_PACKING=0"
        )
        return torch.compile(fullgraph=True, dynamic=True)(_pack_fp4_to_uint8)
    else:
        return torch.compiler.disable()(_pack_fp4_to_uint8)


def pack_fp4_to_uint8_cuda(x: torch.Tensor) -> torch.Tensor:
    """
    Packs a tensor with values in the fp4 range into uint8.

    :param x: tensor to pack
    returns: a packed tensor in uint8
    """
    pack_fn = _get_packing_fn()
    return pack_fn(x)


def _pack_fp4_to_uint8(x: torch.Tensor) -> torch.Tensor:

    m, n = x.shape
    device = x.device

    # Create lookup table for FP4 values to indices
    # Map the absolute values to 0-7 indices
    kE2M1 = torch.tensor(FLOAT_TO_E2M1, device=device, dtype=x.dtype)

    # Find closest valid FP4 value index for each element
    abs_x = torch.abs(x)
    abs_diff_x = torch.abs(abs_x.unsqueeze(-1) - kE2M1)  # [m, n, 8]
    abs_indices = torch.argmin(abs_diff_x, dim=-1)  # [m, n]

    # Apply sign bit (bit 3) to get final 4-bit representation
    indices = abs_indices + (torch.signbit(x).to(torch.long) << 3)

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
