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
from auto_round.compressors.utils import BackendDataType
from auto_round.data_type.mxfp import FP32_EXPONENT_BIAS, FP32_MIN_NORMAL
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad
from auto_round.utils import get_packing_device, logger

# from auto_round.utils import get_weight_compress_dtype
E8M0_EXPONENT_BIAS = 127
E8M0_EXPONENT_NAN_VAL = 255

__all__ = ["QuantLinear"]

FLOAT_TO_E0M4 = [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]


class QuantLinear(nn.Module):
    """
    MXINT quantized linear layer.
    """

    QUANT_TYPE = "MXINT"

    def __init__(self, bits, group_size, infeatures, outfeatures, bias, trainable=False, data_type="mx_int4", **kwargs):
        super().__init__()
        if bits not in [4]:
            raise NotImplementedError("Only 4 bits are supported.")
        if group_size != 32:
            raise NotImplementedError(f"Only group_size 32 are supported for {BackendDataType.MX_INT} data type.")
        if infeatures % group_size != 0:
            raise NotImplementedError(
                f"in_feature must be divisible by {group_size} for {BackendDataType.MX_INT} data type."
            )
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.data_type = data_type
        self.sym = kwargs.get("sym", True)
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2**self.bits - 1
        self.act_bits = kwargs.get("act_bits", None)

        weight_name = "weight_packed"
        weight_infeatures = infeatures if self.bits == 8 else infeatures // 2
        weight_dtype = torch.uint8
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
        scaled_tensor = tensor / (2 ** scales.reshape(tensor.shape[0], -1))
        scaled_tensor = revert_tensor_by_pad(scaled_tensor, orig_shape=orig_shape, pad_len=pad_len)
        final_scale = (scales + E8M0_EXPONENT_BIAS).clamp(0, E8M0_EXPONENT_NAN_VAL).to(torch.uint8)

        self.weight_scale = final_scale
        compress_dtype = torch.uint8
        self.weight_packed = pack_int4_to_uint8(scaled_tensor)


def pack_int4_to_uint8(scaled_tensor: torch.Tensor):
    if scaled_tensor.device.type == "cuda":
        return pack_int4_to_uint8_cuda(scaled_tensor)
    else:
        return pack_int4_to_uint8_cpu(scaled_tensor)


# The torch.compile with dynamic=True is incompatible with multiple threads
# https://github.com/pytorch/pytorch/issues/126024
@torch.compiler.disable()
def pack_int4_to_uint8_cpu(x: torch.Tensor) -> torch.Tensor:
    return _pack_int4_to_uint8(x)


# Adapted from https://github.com/neuralmagic/compressed-tensors/pull/400


def _get_packing_fn():
    if envs.AR_ENABLE_COMPILE_PACKING:
        logger.warning_once(
            "Compiled INT4 to UINT8 packing may be incompatible with multi-threading."
            " Disable it by setting AR_ENABLE_COMPILE_PACKING=0"
        )
        return torch.compile(fullgraph=True, dynamic=True)(_pack_int4_to_uint8)
    else:
        return torch.compiler.disable()(_pack_int4_to_uint8)


def pack_int4_to_uint8_cuda(x: torch.Tensor) -> torch.Tensor:
    """
    Packs a tensor with values in the int4 range into uint8.

    :param x: tensor to pack
    returns: a packed tensor in uint8
    """
    pack_fn = _get_packing_fn()
    return pack_fn(x)


def _pack_int4_to_uint8(x: torch.Tensor) -> torch.Tensor:

    m, n = x.shape
    device = x.device

    # Create lookup table for INT4 values to indices
    # Map the absolute values to 0-7 indices
    kE0M4 = torch.tensor(FLOAT_TO_E0M4, device=device, dtype=x.dtype)

    # Find closest valid INT4 value index for each element
    abs_x = torch.abs(x)
    abs_diff_x = torch.abs(abs_x.unsqueeze(-1) - kE0M4)  # [m, n, 8]
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
