# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import auto_round_kernel

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="XPU not available",
)


def _make_attention_inputs(layout: str):
    """Create contiguous Q/K/V tensors matching the requested layout."""
    torch.manual_seed(3030)
    batch, heads, seq, head_dim = 1, 2, 64, 64
    if layout == "HND":
        return (
            torch.randn(batch, heads, seq, head_dim, device="xpu", dtype=torch.float16).contiguous(),
            torch.randn(batch, heads, seq, head_dim, device="xpu", dtype=torch.float16).contiguous(),
            torch.randn(batch, heads, seq, head_dim, device="xpu", dtype=torch.float16).contiguous(),
        )
    if layout == "NHD":
        return (
            torch.randn(batch, seq, heads, head_dim, device="xpu", dtype=torch.float16).contiguous(),
            torch.randn(batch, seq, heads, head_dim, device="xpu", dtype=torch.float16).contiguous(),
            torch.randn(batch, seq, heads, head_dim, device="xpu", dtype=torch.float16).contiguous(),
        )
    raise ValueError(f"Unsupported layout: {layout}")


@pytest.mark.parametrize(
    ("api_name", "atol", "rtol"),
    [("sdpa", 1e-2, 1e-2), ("sagev1", 2e-2, 2e-2)],
)
@pytest.mark.parametrize("layout", ["HND", "NHD"])
def test_ark_attention_supports_contiguous_inputs_for_layouts(api_name, atol, rtol, layout):
    q, k, v = _make_attention_inputs(layout)
    scale = 1 / math.sqrt(q.shape[-1])

    expected = torch.nn.functional.scaled_dot_product_attention(
        q if layout == "HND" else q.transpose(1, 2),
        k if layout == "HND" else k.transpose(1, 2),
        v if layout == "HND" else v.transpose(1, 2),
        scale=scale,
        is_causal=False,
    )
    if layout == "NHD":
        expected = expected.transpose(1, 2)

    ark = auto_round_kernel
    if api_name == "sdpa":
        actual = ark.sdpa(q, k, v, scale=scale, is_causal=False, tensor_layout=layout)
    else:
        actual = ark.sagev1(
            q,
            k,
            v,
            scale=scale,
            is_causal=False,
            quant_block_size=64,
            tensor_layout=layout,
        )
    torch.xpu.synchronize()

    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


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
    actual = auto_round_kernel.sdpa(q, k, v, scale=scale, is_causal=False)
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
    actual = auto_round_kernel.sagev1(
        q,
        k,
        v,
        scale=scale,
        is_causal=False,
        quant_block_size=64,
    )
    torch.xpu.synchronize()

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
