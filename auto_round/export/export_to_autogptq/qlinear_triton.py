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

    def pack(self, linear, scales, zeros, g_idx=None, device=None):
        scales_t = scales.t().contiguous()
        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()
        self.scales = scales_t.clone().half()

        W = linear.weight.data.to(device).clone()
        if isinstance(linear, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
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
        intweight = intweight << order_map
        intweight = torch.sum(intweight, dim=-1)

        intweight = intweight.t().contiguous().to(torch.int32)
        self.qweight = intweight.to("cpu")

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
