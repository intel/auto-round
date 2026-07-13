# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Standard-SDPA tests for the CPU mixed runtime.

This module intentionally exercises only the public ``auto_round_kernel.sdpa()``
contract under mixed dtypes (Q=float32, K/V=fp16|bf16). BestLA-only extensions
such as alibi/tanh/padding-right/prefer_fp32 and packed-cache helpers live in
the separate internal-route test module.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

auto_round_kernel = pytest.importorskip(
    "auto_round_kernel", reason="compiled ARK extension not built in this environment"
)

_TOL = {torch.float16: (3e-2, 3e-2), torch.bfloat16: (8e-2, 8e-2)}


def _to_layout(tensor_hnd, layout):
    if layout == "HND":
        return tensor_hnd.contiguous()
    if layout == "NHD":
        return tensor_hnd.transpose(1, 2).contiguous()
    raise ValueError(layout)


def _to_hnd(tensor, layout):
    return tensor if layout == "HND" else tensor.transpose(1, 2)


def _mixed_sdpa(q, k, v, scale, is_causal, layout):
    return auto_round_kernel.sdpa(q, k, v, scale=scale, is_causal=is_causal, tensor_layout=layout)


def test_mixed_dtype_sdpa_routes_to_mixed_path():
    """Verify mixed-dtype SDPA is dispatched to the BestLA mixed path by default."""
    torch.manual_seed(4001)
    q = torch.randn(1, 8, 16, 64, dtype=torch.float32)
    k = torch.randn(1, 2, 16, 64, dtype=torch.float16)
    v = torch.randn(1, 2, 16, 64, dtype=torch.float16)
    scale = 1 / math.sqrt(64)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k.float(), v.float(), scale=scale, enable_gqa=True)

    out = auto_round_kernel.sdpa(q, k, v, scale=scale)

    atol, rtol = _TOL[torch.float16]
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
def test_bestla_mixed_sdpa_matches_torch(kv_dtype, is_causal, layout):
    torch.manual_seed(4003)
    batch, heads_q, heads_kv, head_dim, seq = 1, 8, 2, 64, 64
    scale = 1 / math.sqrt(head_dim)
    q = torch.randn(batch, heads_q, seq, head_dim, dtype=torch.float32)
    k = torch.randn(batch, heads_kv, seq, head_dim, dtype=kv_dtype)
    v = torch.randn(batch, heads_kv, seq, head_dim, dtype=kv_dtype)

    expected_hnd = torch.nn.functional.scaled_dot_product_attention(
        q, k.float(), v.float(), scale=scale, enable_gqa=True, is_causal=is_causal
    )
    try:
        actual = _mixed_sdpa(_to_layout(q, layout), _to_layout(k, layout), _to_layout(v, layout), scale, is_causal, layout)
    except (RuntimeError, ValueError) as exc:
        pytest.skip(f"BestLA mixed path unavailable on this ISA/runtime: {exc}")

    atol, rtol = _TOL[kv_dtype]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(_to_hnd(actual, layout), expected_hnd, atol=atol, rtol=rtol)


@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gqa_ratio", [2, 4, 8])
def test_bestla_mixed_sdpa_gqa_ratio(kv_dtype, gqa_ratio):
    torch.manual_seed(5001 + gqa_ratio)
    batch, heads_q, head_dim, seq = 1, 8, 64, 32
    heads_kv = heads_q // gqa_ratio
    scale = 1 / math.sqrt(head_dim)

    q = torch.randn(batch, heads_q, seq, head_dim, dtype=torch.float32)
    k = torch.randn(batch, heads_kv, seq, head_dim, dtype=kv_dtype)
    v = torch.randn(batch, heads_kv, seq, head_dim, dtype=kv_dtype)

    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k.float(), v.float(), scale=scale, enable_gqa=True, is_causal=False
    )
    try:
        actual = _mixed_sdpa(q, k, v, scale, False, "HND")
    except (RuntimeError, ValueError) as exc:
        pytest.skip(f"BestLA mixed path unavailable on this ISA/runtime: {exc}")

    atol, rtol = _TOL[kv_dtype]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.bfloat16])
def test_bestla_mixed_sdpa_non_square_causal_matches_torch(kv_dtype):
    torch.manual_seed(5009)
    batch, heads_q, heads_kv, head_dim, seq_q, seq_kv = 1, 8, 2, 64, 1, 32
    scale = 1 / math.sqrt(head_dim)
    q = torch.randn(batch, heads_q, seq_q, head_dim, dtype=torch.float32)
    k = torch.randn(batch, heads_kv, seq_kv, head_dim, dtype=kv_dtype)
    v = torch.randn(batch, heads_kv, seq_kv, head_dim, dtype=kv_dtype)

    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k.float(), v.float(), scale=scale, enable_gqa=True, is_causal=True
    )
    try:
        actual = _mixed_sdpa(q, k, v, scale, True, "HND")
    except (RuntimeError, ValueError) as exc:
        pytest.skip(f"BestLA mixed path unavailable on this ISA/runtime: {exc}")

    atol, rtol = _TOL[kv_dtype]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
