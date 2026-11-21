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

"""Numerical agreement between the two triton int2/4/8 matmul paths.

`triton_utils.kernels.quant_matmul_248` fuses unpacking into the matmul, while
`triton_utils.dequant.quant_matmul_248` dequantizes to a dense matrix first. Both take
the same arguments and must agree; nothing in the tree compared them, and the fused
kernel has no other caller.
"""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")


def _pack(N, K, bits, group_size, device, seed=0, negative_last_group=False):
    """Pack a weight the way ``qlinear_tritonv2.pack`` does, so the kernels read a legal layout."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    maxq = 2**bits - 1
    weight = torch.randn((N, K), generator=gen) * 0.05
    scales = torch.rand((N, K // group_size), generator=gen) * 0.01 + 1e-3
    zeros = torch.randint(0, maxq + 1, (N, K // group_size), generator=gen).to(torch.float32)

    intweight = torch.round(weight / scales.repeat_interleave(group_size, 1) + zeros.repeat_interleave(group_size, 1))
    intweight = intweight.clamp(0, maxq).to(torch.int32)

    per_word = 32 // bits
    intweight = intweight.reshape(-1, intweight.shape[1] // per_word, per_word)
    order = torch.arange(0, per_word) * bits
    qweight = (intweight << order).sum(-1).t().contiguous().to(torch.int32).to(device)

    z = zeros.t().contiguous().numpy().astype(np.uint32)
    packed_zeros = np.zeros((z.shape[0], z.shape[1] // per_word), dtype=np.uint32)
    i = col = 0
    while col < packed_zeros.shape[1]:
        for j in range(i, i + per_word):
            packed_zeros[:, col] |= z[:, j] << (bits * (j - i))
        i += per_word
        col += 1
    qzeros = torch.from_numpy(packed_zeros.astype(np.int32)).to(device)

    groups = [i // group_size for i in range(K)]
    if negative_last_group:
        # GPTQ may record the final group as -1 rather than num_groups - 1.
        last = (K // group_size) - 1
        groups = [-1 if g == last else g for g in groups]
    g_idx = torch.tensor(groups, dtype=torch.int32).to(device)
    x = (torch.randn((64, K), generator=gen) * 0.1).to(device).half()
    return x, qweight, scales.t().contiguous().half().to(device), qzeros, g_idx, bits, maxq


@pytest.mark.parametrize(
    ("N", "K", "bits"),
    [(512, 1024, 4), (512, 1024, 8), (256, 512, 2)],
    ids=["w4", "w8", "w2"],
)
def test_fused_matmul_matches_dequantized_matmul(N, K, bits):
    from auto_round_extension.triton.triton_utils.dequant import quant_matmul_248 as dequant_then_matmul
    from auto_round_extension.triton.triton_utils.kernels import quant_matmul_248 as fused

    args = _pack(N, K, bits, 128, "cuda")
    with torch.no_grad():
        expected = dequant_then_matmul(*args)
        actual = fused(*args)

    assert actual.shape == expected.shape
    # fp16 accumulate: the two orders of operation differ by rounding, not by a quantization step.
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


def test_fused_matmul_handles_negative_group_index():
    """A -1 group marker must reach the same scales and zeros the reference uses."""
    from auto_round_extension.triton.triton_utils.dequant import quant_matmul_248 as dequant_then_matmul
    from auto_round_extension.triton.triton_utils.kernels import quant_matmul_248 as fused

    args = _pack(512, 1024, 4, 128, "cuda", negative_last_group=True)
    with torch.no_grad():
        expected = dequant_then_matmul(*args)
        actual = fused(*args)

    # Before the fix this returned NaN: the fused kernel indexed scales at g_idx * stride
    # with g_idx == -1, reading before the buffer.
    assert not torch.isnan(actual).any()
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


def test_fused_matmul_masks_the_tail_when_k_is_not_divisible():
    """K that does not divide BLOCK_SIZE_K must not read past the packed buffers."""
    from auto_round_extension.triton.triton_utils.dequant import quant_matmul_248 as dequant_then_matmul
    from auto_round_extension.triton.triton_utils.kernels import quant_matmul_248 as fused

    # K=1000 with group 8 is legal packing (1000 % 8 == 0) but never divides the
    # autotuner's BLOCK_SIZE_K. Before the fix this raised an illegal memory access.
    args = _pack(512, 1000, 4, 8, "cuda")
    with torch.no_grad():
        expected = dequant_then_matmul(*args)
        actual = fused(*args)

    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
