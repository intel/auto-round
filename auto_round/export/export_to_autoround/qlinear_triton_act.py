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

# MIT License
#
# Copyright (c) 2023 潘其威(William)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import math
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import transformers

from auto_round.utils import _get_packing_device

logger = getLogger(__name__)


class QuantLinear(nn.Module):
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
                dtype=torch.bfloat16,
            ),
        )
        use_pc = kwargs.pop("use_pc", False)
        w_bf16_to_fp8_scale_shape = (1, self.outfeatures) if use_pc else (1,)
        self.register_buffer(
            "w_bf16_to_fp8_scale",
            torch.zeros(
                w_bf16_to_fp8_scale_shape,
                dtype=torch.bfloat16,
            ),
        )
        self.register_buffer(
            "act_scales",
            torch.zeros(
                (1),
                dtype=torch.bfloat16,
            ),
        )

        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None

        self.trainable = trainable

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.infeatures}, {self.outfeatures}, "
            f"bits={self.bits}, group_size={self.group_size},"
            f"scales shape: {self.scales.shape}, "
            f"act_scales shape: {self.act_scales.shape}, w_bf16_to_fp8_scale shape: {self.w_bf16_to_fp8_scale.shape}"
            f")"
        )

    def post_init(self):
        pass

    def pack(self, linear, scales, zeros, act_scales, w_bf16_to_fp8_scale, g_idx=None, device=None):
        device = _get_packing_device(device)
        scales_t = scales.t().contiguous()

        self.act_scales.data.copy_(act_scales.squeeze().clone())
        self.w_bf16_to_fp8_scale.data.copy_(w_bf16_to_fp8_scale.squeeze().clone())
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
        else:
            repeat_zeros = zeros

        intweight = torch.round(W.to(device) / repeat_scales[:, : W.shape[1]] + repeat_zeros[:, : W.shape[1]]).to(
            torch.int32
        )
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
            zeros = zeros.to(torch.float16).numpy().astype(np.uint32)
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

    @classmethod
    def warmup(cls, model, transpose=False, seqlen=2048):
        return


__all__ = ["QuantLinear"]
