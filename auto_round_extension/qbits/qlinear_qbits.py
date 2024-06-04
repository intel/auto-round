
# Copyright (c) 2024 Intel Corporation
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

import math

import numpy as np
import torch
import torch.nn as nn
import transformers
from intel_extension_for_transformers import qbits  # with QBits kernels ()
from auto_round.utils import convert_dtype_torch2str, logging

logger = logging.getLogger(__name__)

BITS_DTYPE_MAPPING = {
    4: "int4_clip",
    8: "int8",
}


class QuantLinear(nn.Module):
    QUANT_TYPE = "qbits"

    def __init__(
        self,
        bits,
        group_size,
        infeatures,
        outfeatures,
        bias,
        kernel_switch_threshold=128,
        trainable=False,
        weight_dtype=torch.bfloat16,
        **kwargs,
    ):
        super().__init__()

        if bits not in [4, 8]:
            raise NotImplementedError("Only 4,8 bits are supported for QBits.")

        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2**self.bits - 1
        self.weight_dtype = weight_dtype
        self.asym = True
        self.need_repack = True

        self.register_buffer(
            "qweight",
            torch.zeros((infeatures // 32 * self.bits,
                        outfeatures), dtype=torch.int32),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (
                    math.ceil(infeatures / self.group_size),
                    outfeatures // 32 * self.bits,
                ),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (math.ceil(infeatures / self.group_size), outfeatures),
                dtype=weight_dtype,
            ),
        )
        if bias:
            self.register_buffer("bias", torch.zeros(
                (outfeatures), dtype=torch.float))
        else:
            self.bias = None

        self.kernel_switch_threshold = kernel_switch_threshold

        self.trainable = trainable

    def post_init(self):
        assert self.qweight.device.type == "cpu"
        if self.bias is not None:
            self.bias = self.bias.to(dtype=torch.float32)

        # intweight: k x n, zeros: k / group_size x n
        intweight, zeros = unpack_to_8bit_signed(
            self.qweight, self.qzeros, self.bits)
        if zeros is None:
            zeros = torch.empty(0, dtype=torch.int8)
            self.asym = False
        else:
            # change it to int8 with offset 128
            if self.bits == 8:
                zeros = (zeros.to(torch.int32) -
                         (2 ** (self.bits - 1))).to(torch.int8)
            else:
                zeros -= (2**(self.bits - 1))

        if not self.asym:
            intweight -= (2**(self.bits - 1))
        intweight = intweight.to(torch.uint8 if self.asym else torch.int8)
        # due to asym return torch.uint8 but backend request int8,
        # change it to int8 with offset 128
        if self.asym:
            intweight = (intweight.to(torch.int32) -
                         (2 ** (self.bits - 1))).to(torch.int8)

        scales = self.scales

        self.qweight = qbits.repack_quantized_weight(intweight.contiguous(), scales.float(), zeros, torch.empty(0),
                                                     # weight_dtype
                                                     BITS_DTYPE_MAPPING[self.bits],
                                                     # scale_dtype
                                                     "fp32",
                                                     # TODO(zhe): consider dynamic-set cmpt for better perf?
                                                     "fp32",
                                                     self.asym,
                                                     self.group_size)
        # free mem
        self.qzeros = torch.empty(0)
        self.scales = torch.empty(0)

    def pack(self, linear, scales, zeros, g_idx=None):
        W = linear.weight.data.clone()
        if isinstance(linear, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.t()

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone().to(dtype=linear.weight.dtype)
        if linear.bias is not None:
            self.bias = linear.bias.clone().to(dtype=linear.weight.dtype)

        intweight = []
        for idx in range(self.infeatures):
            intweight.append(
                torch.round(
                    (W[:, idx] + scale_zeros[idx//self.group_size]) / self.scales[idx//self.group_size]).to(torch.int)[:, None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)

        i = 0
        row = 0
        qweight = np.zeros(
            (intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32)
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= intweight[i] << 30
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= intweight[i] << 31
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

        zeros -= 1
        zeros = zeros.numpy().astype(np.uint32)
        qzeros = np.zeros(
            (zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32 // self.bits
                col += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i))
                i += 10
                qzeros[:, col] |= zeros[:, i] << 30
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 1)
                i += 10
                qzeros[:, col] |= zeros[:, i] << 31
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 2)
                i += 10
                col += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros)

    def forward(self, x: torch.Tensor):
        if self.need_repack:
            self.post_init()
            self.need_repack = False
        raw_input_dtype = x.dtype
        if raw_input_dtype != torch.float32:
            x = x.to(torch.float32)
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.view(-1, x.shape[-1])  # convert xd to 2d
        out_2d_shape = x.shape[:-1] + (self.outfeatures,)

        outputs = torch.zeros(out_2d_shape, device=x.device, dtype=torch.float)
        bias = self.bias if self.bias is not None else torch.empty(
            0, dtype=torch.float)

        qbits.woq_linear(x, self.qweight, bias, outputs,
                         convert_dtype_torch2str(torch.float),  # compute_dtype
                         BITS_DTYPE_MAPPING[self.bits],  # weight_dtype
                         "fp32",  # scale_dtype
                         self.asym)
        return outputs.to(raw_input_dtype).view(out_shape)


@torch.no_grad()
def unpack_to_8bit_signed(qweight, qzeros, bits):
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
    zeros = None
    if not torch.all(torch.eq(qzeros, 2004318071 if bits == 4 else 0b01111111011111110111111101111111)):
        zp_shape = list(qzeros.shape)
        zp_shape[1] = zp_shape[1] * (32 // bits)

        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(qzeros, 2).expand(-1, -1,
                                              32 // bits), wf.unsqueeze(0)
        ).to(torch.int16 if bits == 8 else torch.int8)
        torch.bitwise_and(zeros, (2**bits) - 1, out=zeros)
        if bits == 8:
            zeros = zeros.to(torch.uint8)
        zeros = zeros + 1
        try:
            zeros = zeros.reshape(zp_shape)
        except:
            # zeros and scales have different iteam numbers.
            # remove 1 (due to 0 + 1 in line 252)
            zeros = zeros[zeros != 1]
            zeros = zeros.reshape(zp_shape)

    weight = torch.bitwise_right_shift(
        torch.unsqueeze(qweight, 1).expand(-1, 32 //
                                           bits, -1), wf.unsqueeze(-1)
    ).to(torch.int16 if bits == 8 else torch.int8)
    weight.bitwise_and_((2**bits) - 1)
    weight = weight.view(-1, weight.shape[-1])

    return weight, zeros


# Copied from qlinear_marlin.py
@torch.no_grad()
def dequantize_weight(qweight, qzeros, scales, bits):
    unpacked_qweight, unpacked_qzeros = unpack_to_8bit_signed(
        qweight, qzeros, bits)
    group_size = unpacked_qweight.shape[0] // scales.shape[0]
    scales = scales.repeat_interleave(group_size, dim=0)
    if unpacked_qzeros is not None:
        unpacked_qzeros = unpacked_qzeros.repeat_interleave(group_size, dim=0)
    else:
        unpacked_qzeros = torch.full_like(
            scales, 8 if bits == 4 else 128, dtype=torch.int32)
    unpacked_qweight = (unpacked_qweight - unpacked_qzeros) * scales

    return unpacked_qweight, unpacked_qzeros


__all__ = ["QuantLinear"]
