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
import os
# from auto_round_extension.cuda.triton_utils.mixin import TritonModuleMixin

logger = getLogger(__name__)



try:
    from auto_gptq.nn_modules.triton_utils.kernels import (
        QuantLinearFunction,
        QuantLinearInferenceOnlyFunction,
        quant_matmul_248,
        quant_matmul_inference_only_248,
        transpose_quant_matmul_248,
    )
except ImportError as e:
    triton_import_exception = e


    def error_raiser_triton(*args, **kwargs):
        raise ValueError(
            f'Trying to use the triton backend, but could not import triton '
            f'dependencies with the following error: {triton_import_exception}'
        )


    class FakeTriton:
        def __getattr__(self, name):
            raise ImportError(
                f"Trying to use the triton backend, but could not import triton "
                f"dependencies with the following error: {triton_import_exception}"
            )


    quant_matmul_248 = error_raiser_triton
    transpose_quant_matmul_248 = error_raiser_triton
    quant_matmul_inference_only_248 = error_raiser_triton
    QuantLinearFunction = FakeTriton
    QuantLinearInferenceOnlyFunction = FakeTriton


# class QuantLinear(nn.Module, TritonModuleMixin):


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
                dtype=torch.bfloat16,
            ),
        )
        use_pc = kwargs.pop("use_pc", False)
        w_bf16_to_fp8_scale_shape = (1, self.outfeatures) if use_pc else (1, )
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

        self.register_buffer(
            "g_idx",
            torch.tensor([i // self.group_size for i in range(infeatures)], dtype=torch.int32),
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
            f"scales shape: {self.scales.shape}, act_scales shape: {self.act_scales.shape}, w_bf16_to_fp8_scale shape: {self.w_bf16_to_fp8_scale.shape}"
            f")"
        )

    def post_init(self):
        pass

    def pack(self, linear, scales, zeros, act_scales, w_bf16_to_fp8_scale, g_idx=None):
        W = linear.weight.data.clone()
        if isinstance(linear, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.t()

        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales.data.copy_(scales.clone().contiguous())
        self.act_scales.data.copy_(act_scales.squeeze().clone())
        self.w_bf16_to_fp8_scale.data.copy_(w_bf16_to_fp8_scale.squeeze().clone())
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
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            else:
                raise NotImplementedError("Only 2,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

        zeros -= 1
        zeros = zeros.numpy().astype(np.uint32)
        qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32 // self.bits
                col += 1
            else:
                raise NotImplementedError("Only 2,4,8 bits are supported.")

        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros)

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.outfeatures,)
        quant_linear_fn = QuantLinearFunction if self.trainable else QuantLinearInferenceOnlyFunction
        out = quant_linear_fn.apply(
            x.reshape(-1, x.shape[-1]),
            self.qweight,
            self.scales,
            self.qzeros,
            self.g_idx,
            self.bits,
            self.maxq,
        )
        out = out.half().reshape(out_shape)
        out = out + self.bias if self.bias is not None else out
        return out

    @classmethod
    def warmup(cls, model, transpose=False, seqlen=2048):
        """
        Pre-tunes the quantized kernel
        """
        from tqdm import tqdm

        kn_values = {}

        for _, m in model.named_modules():
            if not isinstance(m, cls):
                continue

            k = m.infeatures
            n = m.outfeatures

            if (k, n) not in kn_values:
                kn_values[(k, n)] = (
                    m.qweight,
                    m.scales,
                    m.qzeros,
                    m.g_idx,
                    m.bits,
                    m.maxq,
                )

        logger.info(f"Found {len(kn_values)} unique KN Linear values.")
        logger.info("Warming up autotune cache ...")
        with torch.no_grad():
            for m in tqdm(range(0, math.ceil(math.log2(seqlen)) + 1)):
                m = 2 ** m
                for (k, n), (
                        qweight,
                        scales,
                        qzeros,
                        g_idx,
                        bits,
                        maxq,
                ) in kn_values.items():
                    if transpose:
                        a = torch.randn(m, k, dtype=torch.float16, device=model.device)
                        quant_matmul_248(a, qweight, scales, qzeros, g_idx, bits, maxq)
                        a = torch.randn(m, n, dtype=torch.float16, device=model.device)
                        transpose_quant_matmul_248(a, qweight, scales, qzeros, g_idx, bits, maxq)
                    else:
                        a = torch.randn(m, k, dtype=torch.float16, device=model.device)
                        quant_matmul_inference_only_248(a, qweight, scales, qzeros, g_idx, bits, maxq)
        del kn_values


__all__ = ["QuantLinear"]
