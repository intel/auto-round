# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import auto_round_kernel

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="XPU not available",
)


@pytest.mark.parametrize("seq_q,seq_kv", [(64, 64), (80, 64), (64, 80), (226, 226)])
def test_ark_sdpa_matches_torch_for_kv_remainders(seq_q, seq_kv):
    torch.manual_seed(2026)
    batch, heads, head_dim = 1, 2, 64
    scale = 1 / math.sqrt(head_dim)
    q = torch.randn(batch, heads, seq_q, head_dim, device="xpu", dtype=torch.float16)
    k = torch.randn(batch, heads, seq_kv, head_dim, device="xpu", dtype=torch.float16)
    v = torch.randn(batch, heads, seq_kv, head_dim, device="xpu", dtype=torch.float16)

    expected = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        scale=scale,
        is_causal=False,
    )
    actual = auto_round_kernel.ARK().sdpa(q, k, v, scale=scale, is_causal=False)
    torch.xpu.synchronize()

    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


def test_ark_sagev1_matches_torch_for_kv_remainder_tile():
    """SAGEV1 uses the same final partial-K tile mask pattern after Q/K quantization.

    If the SAGEV1 remainder mask is built from the row-reduced fragment, this
    case produces a large error because the mask is broadcast along the K/column
    dimension.  The column-reduced mask keeps the error within the expected
    SAGEV1 quantization tolerance.
    """
    torch.manual_seed(2027)
    batch, heads, seq_q, seq_kv, head_dim = 1, 2, 80, 80, 64
    scale = 1 / math.sqrt(head_dim)
    q = torch.randn(batch, heads, seq_q, head_dim, device="xpu", dtype=torch.float16)
    k = torch.randn(batch, heads, seq_kv, head_dim, device="xpu", dtype=torch.float16)
    v = torch.randn(batch, heads, seq_kv, head_dim, device="xpu", dtype=torch.float16)

    expected = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        scale=scale,
        is_causal=False,
    )
    actual = auto_round_kernel.ARK().sagev1(
        q,
        k,
        v,
        scale=scale,
        is_causal=False,
        quant_block_size=64,
    )
    torch.xpu.synchronize()

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
