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
try:
    import habana_frameworks.torch.core as htcore
    convert_from_uint4 = torch.ops.hpu.convert_from_uint4
except Exception as e:
    hpu_import_exception = e

    def error_raiser_hpu(*args, **kwargs):
        raise ValueError(
            f"Trying to use HPU, but could not import the HPU framework with the following error: {hpu_import_exception}"
        )

    convert_from_uint4 = error_raiser_hpu


logger = getLogger(__name__)

def pack_tensor(input, bits = 4):
    normal = input.to(torch.int32)
    q = torch.zeros((normal.shape[0], normal.shape[1] // 32 * bits), dtype=torch.int32)
    i = 0
    col = 0
    while col < q.shape[1]:
        for j in range(i, i + (32 // bits)):
            q[:, col] |= normal[:, j] << (bits * (j - i))
        i += 32 // bits
        col += 1
    q = q.to(torch.int32)
    return q

class QuantLinear(nn.Module):
    QUANT_TYPE = "hpu"

    def __init__(
        self,
        bits,
        group_size,
        infeatures,
        outfeatures,
        bias,
        use_cuda_fp16=True,
        kernel_switch_threshold=128,
        trainable=False,
        weight_dtype=torch.float16,
    ):
        logger.debug(f"qlinear_hpu QuantLinear::__init__ {bits=}, {group_size=}, {infeatures=}, {outfeatures=}, {bias=}, {use_cuda_fp16=}, {kernel_switch_threshold=}, {trainable=}, {weight_dtype=}")
        super().__init__()
        if bits != 4:
            raise NotImplementedError("Only 4 bits are supported.")

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
                dtype=weight_dtype,
            ),
        )
        self.register_buffer(
            "g_idx",
            torch.tensor([i // self.group_size for i in range(infeatures)], dtype=torch.int32),
        )

        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=weight_dtype))
        else:
            self.bias = None
        self.half_indim = self.infeatures // 2

        self.wf = torch.tensor(list(range(0, 32, self.bits)), dtype=torch.int32).unsqueeze(0)

    def _preprocessing(self):
        self.qweight = self.qweight.cpu()
        weight = self.unpack_weight_from_cuda_old_format()
        new_qweight = pack_tensor(weight)
        self.qweight = new_qweight.to('hpu')

        # TODO: Support group indexing and remove the check
        columns = self.qweight.shape[0]
        g_idx_trivial = [i // self.group_size for i in range(columns)]
        g_idx_trivial = torch.tensor(g_idx_trivial, dtype=torch.int32)
        assert torch.equal(self.g_idx, g_idx_trivial), "Non-trivial tensor g_idx is not supported"

        zeros = self.unpack_zeros_from_cuda_old_format().cpu()
        new_qzeros = pack_tensor(zeros)
        self.qzeros = new_qzeros.to('hpu')

    def post_init(self):
        self._preprocessing()

    def pack(self, linear, scales, zeros, g_idx):
        # #TODO: implement
        # raise NotImplementedError("QuantLinear HPU currently doesn't support packing")
        W = linear.weight.data.clone()
        if isinstance(linear, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.t()

        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
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
            if self.bits in [4]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            else:
                raise NotImplementedError("Only 4 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

        # zeros -= 1
        zeros = zeros.numpy().astype(np.uint32)
        qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [4]:
                for j in range(i, i + (32 // self.bits)):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32 // self.bits
                col += 1
            else:
                raise NotImplementedError("Only 4 bits are supported.")

        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros)

    def set_packed(self, qlinear_cls):
        self.qweight = qlinear_cls.qweight
        self.qzeros = qlinear_cls.qzeros
        self.scales = qlinear_cls.scales
        self.bias = qlinear_cls.bias

    def forward(self, x):
        x_dtype = x.dtype
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.reshape(-1, x.shape[-1])
        scales = self.scales
        qweight = self.qweight
        zeros = self.qzeros
        weight = convert_from_uint4(qweight, scales, zeros, x_dtype)
        out = torch.matmul(x, weight)
        out = out.reshape(out_shape)
        out = out + self.bias if self.bias is not None else out
        return out

    def unpack_zeros_from_cuda_old_format(self):
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits),
            self.wf.unsqueeze(0),
        ).to(torch.int16 if self.bits == 8 else torch.int8)

        zeros = zeros + 1
        zeros = torch.bitwise_and(
            zeros, (2**self.bits) - 1
        ).to(self.scales.dtype)  # NOTE: It appears that casting here after the `zeros = zeros + 1` is important.
        zeros = zeros.reshape(-1, zeros.shape[1] * zeros.shape[2])
        return zeros

    def unpack_weight_from_cuda_old_format(self):
        weight = torch.bitwise_right_shift(
                torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
                self.wf.unsqueeze(-1),
            ).to(torch.int16 if self.bits == 8 else torch.int8)
        weight = torch.bitwise_and(weight, (2**self.bits) - 1)
        weight = weight.reshape((weight.shape[0]*weight.shape[1], weight.shape[2]))
        return weight

__all__ = ["QuantLinear"]
