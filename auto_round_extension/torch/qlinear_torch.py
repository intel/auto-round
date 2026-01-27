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

import math
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import transformers

from auto_round.utils import get_packing_device

logger = getLogger(__name__)


_DEVICE_WF_3BITS_TENSORS = {}
# Constants for FP4 values (E2M1 format)
_wf_3bits = [
            [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
            [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
            [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],
        ]

def get_wf_3bits_tensor(device):
    """Get device-specific wf_3bits tensor, creating it if needed."""
    device_str = str(device)
    if device_str not in _DEVICE_WF_3BITS_TENSORS:
        _DEVICE_WF_3BITS_TENSORS[device_str] = torch.tensor(_wf_3bits, dtype=torch.int32, device=device).reshape(1, 3, 12)
    return _DEVICE_WF_3BITS_TENSORS[device_str]

class QuantLinear(nn.Module):
    """
    Torch quantized linear layer.
    """

    QUANT_TYPE = "torch"

    def __init__(self, bits, group_size, infeatures, outfeatures, bias, trainable=False, **kwargs):
        super().__init__()
        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2**self.bits - 1

        self.register_buffer(
            "qweight",
            torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32),
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
                dtype=torch.float16,
            ),
        )
        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None

        self.trainable = trainable

        # is performed by unpacking the weights and using torch.matmul
        if self.bits in [2, 4, 8]:
            self.wf = torch.tensor(list(range(0, 32, self.bits)), dtype=torch.int32, device=self.qweight.device).unsqueeze(0)
        elif self.bits == 3:
            self.wf = get_wf_3bits_tensor(device=self.qweight.device)
        self.dequant_dtype = torch.int16 if self.bits == 8 else torch.int8

    def post_init(self):
        pass

    # @torch.compile() ## cpu side has bug
    def pack_248_bits(self, linear, scales, zeros, g_idx=None, device=None):
        device = get_packing_device(device)
        scales_t = scales.t().contiguous()
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()
        self.scales = scales_t.clone().half()

        W = linear.weight.data.to(device).clone()
        if type(linear) == nn.Conv2d:
            W = W.flatten(1)
        if type(linear) == transformers.pytorch_utils.Conv1D:
            W = W.t()

        repeat_scales = scales.to(device).repeat_interleave(self.group_size, 1)
        if isinstance(zeros, torch.Tensor):
            repeat_zeros = zeros.to(device).repeat_interleave(self.group_size, 1)
            intweight = torch.round(W.to(device) / repeat_scales[:, : W.shape[1]] + repeat_zeros[:, : W.shape[1]]).to(
                torch.int32
            )
        else:
            repeat_zeros = zeros
            intweight = torch.round(W.to(device) / repeat_scales[:, : W.shape[1]] + repeat_zeros).to(torch.int32)

        del repeat_scales

        intweight = intweight.reshape(-1, intweight.shape[1] // 32 * self.bits, 32 // self.bits)
        order_map = torch.arange(0, 32 // self.bits, device=device) * self.bits
        intweight = intweight.to(torch.int32)
        intweight = intweight << order_map
        intweight = torch.sum(intweight, dim=-1)

        intweight = intweight.t().contiguous().to(torch.int32)
        self.qweight = intweight.to("cpu")

        if isinstance(zeros, torch.Tensor):
            zeros = zeros.t().contiguous().to(self.device)
            # zeros = zeros.numpy().astype(np.uint32)
            qzeros = torch.zeros(
                (zeros.shape[0], zeros.shape[1] // 32 * self.bits), device=self.device, dtype=torch.int32
            )
            i = 0
            col = 0
            shifts = torch.arange(0, (32 // self.bits), device=zeros.device) * self.bits
            while col < qzeros.shape[1]:
                packed_zeros = zeros[:, i : i + (32 // self.bits)].clone().to(dtype=torch.int32)

                shifted = packed_zeros << shifts
                qzeros[:, col] |= shifted.sum(dim=-1)
                i += 32 // self.bits
                col += 1
            self.qzeros = qzeros.cpu()
        else:
            shape = scales_t.shape
            value = 0
            for j in range(0, (32 // self.bits)):
                value |= zeros << (self.bits * j)
            qzeros = torch.ones((shape[0], shape[1] // 32 * self.bits), dtype=torch.int32) * value
            self.qzeros = qzeros.cpu()

    # @torch.compile()
    def pack_3bits(self, linear, scales, zeros, g_idx=None, device=None):
        device = get_packing_device(device)
        scales_t = scales.t().contiguous()
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()
        self.scales = scales_t.clone().half()

        W = linear.weight.data.to(device).clone()
        if type(linear) == nn.Conv2d:
            W = W.flatten(1)
        if type(linear) == transformers.pytorch_utils.Conv1D:
            W = W.t()

        repeat_scales = scales.to(device).repeat_interleave(self.group_size, 1)
        if isinstance(zeros, torch.Tensor):
            repeat_zeros = zeros.to(device).repeat_interleave(self.group_size, 1)
            intweight = torch.round(W.to(device) / repeat_scales[:, : W.shape[1]] + repeat_zeros[:, : W.shape[1]]).to(
                torch.int32
            )
        else:
            repeat_zeros = zeros
            intweight = torch.round(W.to(device) / repeat_scales[:, : W.shape[1]] + repeat_zeros).to(torch.int32)

        del repeat_scales

        intweight = intweight.t().contiguous().to(torch.int32)
        i = 0
        row = 0
        qweight = torch.zeros(
            (intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=torch.int32, device=device
        )
        while row < qweight.shape[0]:
            packed_weight = (intweight[i : i + 10]).to(dtype=torch.int32).t()
            shifts = torch.arange(0, 10).to(device) * self.bits
            shifted = packed_weight << shifts
            qweight[row] |= shifted.sum(dim=-1)
            i += 10
            qweight[row] |= intweight[i] << 30
            row += 1
            qweight[row] |= (intweight[i] >> 2) & 1
            i += 1
            packed_weight = (intweight[i : i + 10]).to(dtype=torch.int32).t()
            shifts = torch.arange(0, 10).to(device) * self.bits + 1
            shifted = packed_weight << shifts
            qweight[row] |= shifted.sum(dim=-1)
            i += 10
            qweight[row] |= intweight[i] << 31
            row += 1
            qweight[row] |= (intweight[i] >> 1) & 0x3
            i += 1
            packed_weight = (intweight[i : i + 10]).to(dtype=torch.int32).t()
            shifts = torch.arange(0, 10).to(device) * self.bits + 2
            shifted = packed_weight << shifts
            qweight[row] |= shifted.sum(dim=-1)

            i += 10
            row += 1

        self.qweight = qweight.cpu()

        if isinstance(zeros, torch.Tensor):
            zeros = zeros.t().contiguous().to(device).to(torch.int32)
            # zeros = zeros.numpy().astype(np.uint32)
            qzeros = torch.zeros((zeros.shape[0], zeros.shape[1] // 32 * self.bits), device=device, dtype=torch.int32)
            i = 0
            col = 0
            while col < qzeros.shape[1]:
                packed_zeros = (zeros[:, i : i + 10]).to(dtype=torch.int32)
                shifts = torch.arange(0, 10).to(device) * self.bits
                shifted = packed_zeros << shifts
                qzeros[:, col] = shifted.sum(dim=-1)
                i += 10
                qzeros[:, col] |= zeros[:, i] << 30
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 2) & 1
                i += 1
                packed_zeros = (zeros[:, i : i + 10]).to(dtype=torch.int32)
                shifts = torch.arange(0, 10).to(device) * self.bits + 1
                shifted = packed_zeros << shifts
                qzeros[:, col] |= shifted.sum(dim=-1)
                i += 10
                qzeros[:, col] |= zeros[:, i] << 31
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 1) & 0x3
                i += 1
                packed_zeros = (zeros[:, i : i + 10]).to(dtype=torch.int32)
                shifts = torch.arange(0, 10).to(device) * self.bits + 2
                shifted = packed_zeros << shifts
                qzeros[:, col] |= shifted.sum(dim=-1)
                i += 10
                col += 1
            self.qzeros = qzeros.cpu()
        else:
            shape = scales_t.shape[0], scales_t.shape[1] // 32 * self.bits
            qzeros = torch.zeros(shape, dtype=torch.int32)
            zero_val = zeros
            total_cols = shape[1]
            # Precompute shifts for 3-bit packing
            shifts0 = torch.arange(0, 10, dtype=torch.int32) * self.bits
            shifts1 = shifts0 + 1
            shifts2 = shifts0 + 2
            # Compute packed pattern parts
            part0 = (zero_val << shifts0).sum().to(torch.int32) | (zero_val << 30)
            part1 = (zero_val << shifts1).sum().to(torch.int32) | (zero_val << 31) | ((zero_val >> 2) & 1)
            part2 = (zero_val << shifts2).sum().to(torch.int32) | ((zero_val >> 1) & 0x3)
            pattern = torch.tensor([part0, part1, part2], dtype=torch.int32)
            # Tile pattern across all columns
            repeats = (total_cols + 2) // 3
            full_row = pattern.repeat(repeats)[:total_cols]
            # Broadcast across rows
            qzeros[:] = full_row.unsqueeze(0)
            self.qzeros = qzeros.cpu()

    def pack(self, linear, scales, zeros, g_idx=None, device=None):
        if self.bits in [2, 4, 8]:
            return self.pack_248_bits(linear, scales, zeros, g_idx, device)
        elif self.bits in [3]:
            return self.pack_3bits(linear, scales, zeros, g_idx, device)
        else:
            raise ValueError("Only 2,3,4,8 bits are supported.")

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.reshape(-1, x.shape[-1])
        x_dtype = x.dtype

        if self.bits in [2, 4, 8]:
            if self.wf.device != self.qzeros.device:
                self.wf = torch.tensor(list(range(0, 32, self.bits)), dtype=torch.int32, device=self.qzeros.device).unsqueeze(0)
            zeros = torch.bitwise_right_shift(
                torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits),
                self.wf.unsqueeze(0),
            ).to(self.dequant_dtype)
            zeros = torch.bitwise_and(zeros, self.maxq).reshape(self.scales.shape)

            weight = torch.bitwise_and(
                torch.bitwise_right_shift(
                    torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
                    self.wf.unsqueeze(-1),
                ).to(self.dequant_dtype),
                self.maxq,
            )
        elif self.bits == 3:
            if self.wf.device != self.qzeros.device:
                self.wf = get_wf_3bits_tensor(self.qzeros.device)
            zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1).expand(-1, -1, -1, 12)
            zeros = zeros >> self.wf.unsqueeze(0)
            zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
            zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
            zeros = zeros & 0x7
            zeros = torch.cat(
                [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
                dim=2,
            )

            zeros = zeros.reshape(self.scales.shape)

            weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(
                -1, -1, 12, -1
            )
            weight = (weight >> self.wf.unsqueeze(-1)) & 0x7
            weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
            weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
            weight = weight & 0x7
            weight = torch.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)

        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
        if hasattr(self, "g_idx"):
            num_itr = self.g_idx.shape[0] // x.shape[-1]
            num_dim = self.g_idx.shape[0] // num_itr
            weights = []
            for i in range(num_itr):
                scale_i = self.scales[:, i * num_dim : (i + 1) * num_dim]
                weight_i = weight[:, i * num_dim : (i + 1) * num_dim]
                zeros_i = zeros[:, i * num_dim : (i + 1) * num_dim]
                g_idx_i = self.g_idx[i * num_dim : (i + 1) * num_dim]
                weights.append(scale_i[g_idx_i.long()] * (weight_i - zeros_i[g_idx_i.long()]))
            weights = torch.cat(weights, dim=1)
        else:
            repeat_scales = self.scales.repeat_interleave(self.group_size, dim=0)
            repeat_zeros = zeros.repeat_interleave(self.group_size, dim=0)
            weights = repeat_scales * (weight - repeat_zeros)
        weights = weights.to(x_dtype)
        out = torch.matmul(x, weights)
        out = out.to(x_dtype)
        out = out.reshape(out_shape)
        out = out + self.bias if self.bias is not None else out
        return out


__all__ = ["QuantLinear"]

