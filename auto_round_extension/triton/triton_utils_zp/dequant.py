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
import itertools
import threading

import torch
import triton
import triton.language as tl


def make_dequant_configs(block_sizes, num_warps):
    configs = []
    for bs, ws in itertools.product(block_sizes, num_warps):
        configs.append(triton.Config({"X_BLOCK": bs}, num_warps=ws))
    return configs


DEFAULT_DEQUANT_CONFIGS = make_dequant_configs([128, 256, 512, 1024], [4, 8])


@triton.jit
def dequant_kernel_248(
    g_idx_ptr,
    scales_ptr,
    qweight_ptr,
    qzeros_ptr,
    out_ptr,
    numels,
    maxq: tl.constexpr,
    bits: tl.constexpr,
    outfeatures: tl.constexpr,
    num_groups: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block = tl.program_id(1)

    elements_per_feature: tl.constexpr = 32 // bits
    col_offsets = col_block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    col_mask = col_offsets < outfeatures

    g_idx = tl.load(g_idx_ptr + row_idx, None, eviction_policy="evict_last")
    tmp1 = g_idx + num_groups
    tmp2 = g_idx < 0
    tl.device_assert(g_idx >= 0, "index out of bounds: 0 <= tmp0 < 0")
    groups = tl.where(tmp2, tmp1, g_idx)

    qweight_offsets = col_offsets + (outfeatures * (row_idx // elements_per_feature))
    qweights = tl.load(qweight_ptr + qweight_offsets, mask=col_mask, other=0)

    wf_weights = (row_idx % elements_per_feature) * bits
    weights = (qweights >> wf_weights) & maxq

    qzero_ncols: tl.constexpr = outfeatures // elements_per_feature
    qzero_idx = (qzero_ncols * groups) + col_block
    qzeros = tl.load(qzeros_ptr + qzero_idx, None, eviction_policy="evict_last")
    wf_zeros = (col_offsets % elements_per_feature) * bits
    zeros = (qzeros >> wf_zeros) & maxq

    scales = tl.load(scales_ptr + (col_offsets + (outfeatures * groups)), mask=col_mask, other=0).to(tl.float32)

    zeros = zeros + 1
    weights = weights - zeros
    weights = weights.to(tl.float32)
    weights = scales * weights

    tl.store(out_ptr + (row_idx * outfeatures + col_offsets), weights, mask=col_mask)


def dequant248_core(qweight, scales, qzeros, g_idx, bits, maxq=None, input_dtype=torch.float16):
    """
    Launcher for triton dequant kernel.  Only valid for bits = 2, 4, 8
    """
    num_groups = scales.shape[0]
    outfeatures = scales.shape[1]
    infeatures = g_idx.shape[0]

    out = torch.empty((infeatures, outfeatures), device=qweight.device, dtype=input_dtype)
    maxq = 2**bits - 1 if maxq is None else maxq
    block_cols = 32 // bits
    grid = (infeatures, triton.cdiv(outfeatures, block_cols))

    dequant_kernel_248[grid](
        g_idx,
        scales,
        qweight,
        qzeros,
        out,
        out.numel(),
        maxq=maxq,
        bits=bits,
        outfeatures=outfeatures,
        num_groups=num_groups,
        BLOCK_COLS=block_cols,
    )
    return out


def dequant248(qweight, scales, qzeros, g_idx, bits, maxq=None, input_dtype=torch.float16):
    """
    Launcher for triton dequant kernel. Only valid for bits = 2, 4, 8
    """
    device_type = qweight.device.type
    if device_type in {"cuda", "xpu"}:
        with getattr(torch, device_type).device(qweight.device):
            return dequant248_core(qweight, scales, qzeros, g_idx, bits, maxq=maxq, input_dtype=input_dtype)
    else:
        raise ValueError(f"Unsupported device type: {device_type}")


def quant_matmul_248(input, qweight, scales, qzeros, g_idx, bits, maxq=None, transpose=False):
    input_dtype = input.dtype
    W = dequant248(qweight, scales, qzeros, g_idx, bits, maxq=maxq, input_dtype=input_dtype)
    orig_device = input.device
    if transpose:
        return (input.to(W.device) @ W.t()).to(orig_device)

    return (input.to(W.device) @ W).to(orig_device)


class QuantLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, qweight, scales, qzeros, g_idx, bits, maxq):
        output = quant_matmul_248(input, qweight, scales, qzeros, g_idx, bits, maxq)
        ctx.save_for_backward(qweight, scales, qzeros, g_idx)
        ctx.bits, ctx.maxq = bits, maxq
        return output

    @staticmethod
    def backward(ctx, grad_output):
        qweight, scales, qzeros, g_idx = ctx.saved_tensors
        bits, maxq = ctx.bits, ctx.maxq
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = quant_matmul_248(grad_output, qweight, scales, qzeros, g_idx, bits, maxq, transpose=True)
        return grad_input, None, None, None, None, None, None
