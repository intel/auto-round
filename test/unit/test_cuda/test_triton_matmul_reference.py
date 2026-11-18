# Copyright (c) 2025 Intel Corporation
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

"""A reference implementation of the 2/4/8-bit packed matmul, in plain torch.

The parity test compares the fused kernel against the dequantize-then-matmul path, so a
mistake shared by both would go unnoticed. This unpacks the words by hand with shifts and
masks over the packed int32 buffers and multiplies the resulting dense matrix, which is
independent of every kernel in `triton_utils`.
"""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")


def _unpack_weights(qweight, bits, maxq):
    """(K // per_word, N) packed int32 -> (N, K) dense int values."""
    per_word = 32 // bits
    shifts = torch.arange(per_word, device=qweight.device, dtype=torch.int32) * bits
    words = qweight.to(torch.int64).unsqueeze(-1)  # (K // per, N, 1)
    unpacked = (words >> shifts.long()).bitwise_and(maxq)  # (K // per, N, per)
    return unpacked.permute(1, 0, 2).reshape(qweight.shape[1], -1)  # (N, K)


def _unpack_zeros(qzeros, bits, maxq, out_features):
    """(G, N // per_word) packed int32 -> (G, N) int values."""
    per_word = 32 // bits
    shifts = torch.arange(per_word, device=qzeros.device, dtype=torch.int64) * bits
    words = qzeros.to(torch.int64).unsqueeze(-1)  # (G, N // per, 1)
    unpacked = (words >> shifts).bitwise_and(maxq)  # (G, N // per, per)
    return unpacked.reshape(qzeros.shape[0], out_features)  # (G, N)


def reference_quant_matmul(input, qweight, scales, qzeros, g_idx, bits, maxq):
    out_features = qweight.shape[1]
    weights = _unpack_weights(qweight, bits, maxq)  # (N, K)
    zeros = _unpack_zeros(qzeros, bits, maxq, out_features)  # (G, N)

    groups = g_idx.to(torch.long)
    # GPTQ records the last group as -1; resolve it the way the dequant kernel does.
    groups = torch.where(groups < 0, groups + scales.shape[0], groups)

    scale_per_k = scales[groups].t().to(torch.float32)  # (G, N) -> (K, N) -> (N, K)
    zero_per_k = zeros[groups].t().to(torch.float32)  # (G, N) -> (K, N) -> (N, K)
    dequant = (weights.to(torch.float32) - zero_per_k) * scale_per_k  # (N, K)

    return (input.to(torch.float32) @ dequant.t()).to(input.dtype)


@pytest.mark.parametrize(
    ("N", "K", "bits", "group_size"),
    [(256, 512, 4, 128), (256, 512, 8, 128), (128, 256, 2, 64), (256, 1000, 4, 8)],
    ids=["w4", "w8", "w2", "w4-k-not-divisible"],
)
def test_fused_matmul_matches_plain_torch_reference(N, K, bits, group_size):
    from auto_round_extension.triton.triton_utils.kernels import quant_matmul_248 as fused

    gen = torch.Generator(device="cpu").manual_seed(7)
    maxq = 2**bits - 1
    weight = torch.randn((N, K), generator=gen) * 0.05
    scales = torch.rand((N, K // group_size), generator=gen) * 0.01 + 1e-3
    zeros = torch.randint(0, maxq + 1, (N, K // group_size), generator=gen).to(torch.float32)

    intweight = torch.round(weight / scales.repeat_interleave(group_size, 1) + zeros.repeat_interleave(group_size, 1))
    intweight = intweight.clamp(0, maxq).to(torch.int32)

    per_word = 32 // bits
    intweight = intweight.reshape(-1, intweight.shape[1] // per_word, per_word)
    order = torch.arange(0, per_word) * bits
    qweight = (intweight << order).sum(-1).t().contiguous().to(torch.int32).cuda()

    z = zeros.t().contiguous().numpy().astype(np.uint32)
    packed_zeros = np.zeros((z.shape[0], z.shape[1] // per_word), dtype=np.uint32)
    i = col = 0
    while col < packed_zeros.shape[1]:
        for j in range(i, i + per_word):
            packed_zeros[:, col] |= z[:, j] << (bits * (j - i))
        i += per_word
        col += 1
    qzeros = torch.from_numpy(packed_zeros.astype(np.int32)).cuda()

    g_idx = torch.tensor([i // group_size for i in range(K)], dtype=torch.int32).cuda()
    x = (torch.randn((32, K), generator=gen) * 0.1).cuda().half()

    with torch.no_grad():
        expected = reference_quant_matmul(x, qweight, scales.t().contiguous().half().cuda(), qzeros, g_idx, bits, maxq)
        actual = fused(x, qweight, scales.t().contiguous().half().cuda(), qzeros, g_idx, bits, maxq)

    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
