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
import numba


##TODO different bits
# @numba.jit(nopython=True, parallel=True)
# def pack_array_with_numba_b4_c32(
#         raw_array: np.ndarray, packed_array: np.ndarray
# ) -> np.ndarray:
#     """Pack the array with numba when bits=4 and compress_bits=32."""
#     bits = 4
#     n_pack = 32 // bits
#
#     for row in range(packed_array.shape[0]):
#         packed_array[row] = ((((raw_array[row * n_pack + 7]) << 28)
#                               | ((raw_array[row * n_pack + 6]) << 24)
#                               | ((raw_array[row * n_pack + 5]) << 20)
#                               | ((raw_array[row * n_pack + 4]) << 16)
#                               | ((raw_array[row * n_pack + 3]) << 12)
#                               | (raw_array[row * n_pack + 2]) << 8)
#                              | ((raw_array[row * n_pack + 1]) << 4)
#                              | ((raw_array[row * n_pack]) << 0))
#
#     return packed_array


class TritonModuleMixin:
    @classmethod
    def warmup(cls, model, transpose=False, seqlen=2048):
        pass


class QuantLinear(nn.Module, TritonModuleMixin):
    QUANT_TYPE = "triton"

    def __init__(self, bits, group_size, infeatures, outfeatures, bias, trainable=False, **kwargs):
        super().__init__()
        if bits not in [2, 4, 8]:
            raise NotImplementedError("Only 2,4,8 bits are supported.")
        if infeatures % 32 != 0 or outfeatures % 32 != 0:
            raise NotImplementedError("in_feature and out_feature must be divisible by 32.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2 ** self.bits - 1

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
        self.register_buffer(
            "g_idx",
            torch.tensor([i // self.group_size for i in range(infeatures)], dtype=torch.int32),
        )
        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None

        self.trainable = trainable

    def post_init(self):
        pass

    def pack_cpu(self, linear, scales, zeros, g_idx=None):
        W = linear.weight.data.clone()
        if isinstance(linear, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.t()

        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx

        scales = scales.t().contiguous()
        if isinstance(zeros, torch.Tensor):
            zeros = zeros.t().contiguous()
            scale_zeros = zeros * scales
        else:
            scale_zeros = scales * zeros
        self.scales = scales.clone().half()
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

        intweight = []
        for idx in range(self.infeatures):
            intweight.append(
                torch.round((W[:, idx] + scale_zeros[self.g_idx[idx]]) / self.scales[self.g_idx[idx]]).to(torch.int)[
                :, None
                ]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)

        i = 0
        row = 0
        qweight = np.zeros((intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32)
        while row < qweight.shape[0]:
            for j in range(i, i + (32 // self.bits)):
                qweight[row] |= intweight[j] << (self.bits * (j - i))
            i += 32 // self.bits
            row += 1

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

        if isinstance(zeros, torch.Tensor):
            zeros -= 1
            zeros = zeros.numpy().astype(np.uint32)
            qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32)
            i = 0
            col = 0
            while col < qzeros.shape[1]:
                for j in range(i, i + (32 // self.bits)):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32 // self.bits
                col += 1

            qzeros = qzeros.astype(np.int32)
            self.qzeros = torch.from_numpy(qzeros)
        else:
            zeros -= 1
            shape = scales.shape
            value = 0
            for j in range(0, (32 // self.bits)):
                value |= zeros << (self.bits * j)
            qzeros = np.ones((shape[0], shape[1] // 32 * self.bits), dtype=np.uint32) * value
            qzeros = qzeros.astype(np.int32)
            self.qzeros = torch.from_numpy(qzeros)

    def pack(self, linear, scales, zeros, g_idx):
        if torch.cuda.is_available():
            return self.pack_cuda(linear, scales, zeros, g_idx)
        else:
            return self.pack_cpu(linear, scales, zeros, g_idx)

    def pack_cuda(self, linear, scales, zeros, g_idx=None):
        W = linear.weight.data.clone()
        if isinstance(linear, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.t()
        scales_t = scales.t().contiguous()
        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()
        self.scales = scales_t.clone().half()

        repeat_scales = scales.to("cuda:0").repeat_interleave(self.group_size, 1)
        if isinstance(zeros, torch.Tensor):
            repeat_zeros = zeros.to("cuda:0").repeat_interleave(self.group_size, 1)
        else:
            repeat_zeros = zeros

        intweight = torch.round(W.to("cuda:0") / repeat_scales + repeat_zeros).to(torch.int).t().contiguous().to("cpu")
        intweight = intweight.numpy().astype(np.uint32)
        del repeat_scales

        i = 0
        row = 0
        qweight = np.zeros((intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32)
        # pack_array_with_numba_b4_c32(intweight, qweight)
        while row < qweight.shape[0]:
            for j in range(i, i + (32 // self.bits)):
                qweight[row] |= intweight[j] << (self.bits * (j - i))
            i += 32 // self.bits
            row += 1

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

        if isinstance(zeros, torch.Tensor):
            zeros = zeros.t().contiguous()
            zeros -= 1
            zeros = zeros.numpy().astype(np.uint32)
            qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32)
            i = 0
            col = 0
            while col < qzeros.shape[1]:
                for j in range(i, i + (32 // self.bits)):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32 // self.bits
                col += 1

            qzeros = qzeros.astype(np.int32)
            self.qzeros = torch.from_numpy(qzeros)
        else:
            zeros -= 1
            shape = scales_t.shape
            value = 0
            for j in range(0, (32 // self.bits)):
                value |= zeros << (self.bits * j)
            qzeros = np.ones((shape[0], shape[1] // 32 * self.bits), dtype=np.uint32) * value
            qzeros = qzeros.astype(np.int32)
            self.qzeros = torch.from_numpy(qzeros)


__all__ = ["QuantLinear"]
