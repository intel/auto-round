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


def _to_layout(tensor_hnd, layout):
    if layout == "HND":
        return tensor_hnd.contiguous()
    if layout == "NHD":
        return tensor_hnd.transpose(1, 2).contiguous()
    raise ValueError(layout)


def _to_hnd(tensor, layout):
    return tensor if layout == "HND" else tensor.transpose(1, 2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ark_xpu_kv_update_hnd_and_nhd_produce_same_cache(dtype):
    torch.manual_seed(9001)
    batch, heads_kv, capacity, head_dim = 1, 2, 16, 64
    k_hnd = torch.randn(batch, heads_kv, capacity, head_dim, device="xpu", dtype=dtype)
    v_hnd = torch.randn(batch, heads_kv, capacity, head_dim, device="xpu", dtype=dtype)

    cache_k_hnd, cache_v_hnd = auto_round_kernel.ark_xpu_kv_cache_alloc(
        batch, heads_kv, capacity, head_dim, dtype=dtype
    )
    cache_k_nhd, cache_v_nhd = auto_round_kernel.ark_xpu_kv_cache_alloc(
        batch, heads_kv, capacity, head_dim, dtype=dtype
    )

    auto_round_kernel.ark_xpu_kv_update(cache_k_hnd, cache_v_hnd, k_hnd, v_hnd, 0, tensor_layout="HND")
    auto_round_kernel.ark_xpu_kv_update(
        cache_k_nhd,
        cache_v_nhd,
        _to_layout(k_hnd, "NHD"),
        _to_layout(v_hnd, "NHD"),
        0,
        tensor_layout="NHD",
    )
    torch.xpu.synchronize()

    torch.testing.assert_close(cache_k_hnd, cache_k_nhd, atol=0, rtol=0)
    torch.testing.assert_close(cache_v_hnd, cache_v_nhd, atol=0, rtol=0)


@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("is_causal", [False, True])
def test_sdpa_with_kv_cache_matches_raw_sdpa(layout, is_causal):
    torch.manual_seed(9002 + int(is_causal))
    batch, heads_q, heads_kv, seq_q, seq_kv, head_dim = 1, 4, 2, 1, 33, 64
    dtype = torch.float16
    scale = 1 / math.sqrt(head_dim)

    q_hnd = torch.randn(batch, heads_q, seq_q, head_dim, device="xpu", dtype=dtype)
    k_hnd = torch.randn(batch, heads_kv, seq_kv, head_dim, device="xpu", dtype=dtype)
    v_hnd = torch.randn(batch, heads_kv, seq_kv, head_dim, device="xpu", dtype=dtype)

    cache_k, cache_v = auto_round_kernel.ark_xpu_kv_cache_alloc(batch, heads_kv, seq_kv, head_dim, dtype=dtype)
    auto_round_kernel.ark_xpu_kv_update(
        cache_k,
        cache_v,
        _to_layout(k_hnd, layout),
        _to_layout(v_hnd, layout),
        0,
        tensor_layout=layout,
    )

    actual = auto_round_kernel.sdpa_with_kv_cache(
        _to_layout(q_hnd, layout),
        cache_k,
        cache_v,
        seq_kv,
        scale=scale,
        is_causal=is_causal,
        tensor_layout=layout,
    )
    torch.xpu.synchronize()

    expected = torch.nn.functional.scaled_dot_product_attention(
        q_hnd, k_hnd, v_hnd, scale=scale, enable_gqa=True, is_causal=is_causal
    )
    torch.testing.assert_close(_to_hnd(actual, layout), expected, atol=1e-2, rtol=1e-2)


def test_ark_xpu_kv_update_repeated_appends_preserve_sequence_order():
    torch.manual_seed(9003)
    batch, heads_q, heads_kv, capacity, head_dim = 1, 4, 2, 15, 64
    dtype = torch.float16
    chunks = [4, 6, 5]
    scale = 1 / math.sqrt(head_dim)

    q = torch.randn(batch, heads_q, 1, head_dim, device="xpu", dtype=dtype)
    k_full = torch.randn(batch, heads_kv, capacity, head_dim, device="xpu", dtype=dtype)
    v_full = torch.randn(batch, heads_kv, capacity, head_dim, device="xpu", dtype=dtype)
    cache_k, cache_v = auto_round_kernel.ark_xpu_kv_cache_alloc(batch, heads_kv, capacity, head_dim, dtype=dtype)

    pos = 0
    for chunk in chunks:
        auto_round_kernel.ark_xpu_kv_update(
            cache_k,
            cache_v,
            k_full[:, :, pos : pos + chunk, :],
            v_full[:, :, pos : pos + chunk, :],
            pos,
            tensor_layout="HND",
        )
        pos += chunk

    actual = auto_round_kernel.sdpa_with_kv_cache(q, cache_k, cache_v, capacity, scale=scale, tensor_layout="HND")
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k_full, v_full, scale=scale, enable_gqa=True, is_causal=False
    )
    torch.xpu.synchronize()

    torch.testing.assert_close(cache_k, k_full, atol=0, rtol=0)
    torch.testing.assert_close(cache_v, v_full, atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


def test_sdpa_with_kv_cache_rejects_multi_token_causal_decode():
    batch, heads_q, heads_kv, seq_q, seq_kv, head_dim = 1, 4, 2, 2, 8, 64
    dtype = torch.float16
    q = torch.randn(batch, heads_q, seq_q, head_dim, device="xpu", dtype=dtype)
    k = torch.randn(batch, heads_kv, seq_kv, head_dim, device="xpu", dtype=dtype)
    v = torch.randn(batch, heads_kv, seq_kv, head_dim, device="xpu", dtype=dtype)
    cache_k, cache_v = auto_round_kernel.ark_xpu_kv_cache_alloc(batch, heads_kv, seq_kv, head_dim, dtype=dtype)
    auto_round_kernel.ark_xpu_kv_update(cache_k, cache_v, k, v, 0, tensor_layout="HND")

    with pytest.raises(NotImplementedError, match="single-token decode"):
        auto_round_kernel.sdpa_with_kv_cache(q, cache_k, cache_v, seq_kv, is_causal=True)


def test_ark_xpu_kv_update_rejects_capacity_overflow():
    cache_k, cache_v = auto_round_kernel.ark_xpu_kv_cache_alloc(1, 2, 8, 64, dtype=torch.float16)
    k = torch.randn(1, 2, 4, 64, device="xpu", dtype=torch.float16)
    v = torch.randn(1, 2, 4, 64, device="xpu", dtype=torch.float16)

    with pytest.raises(ValueError, match="capacity"):
        auto_round_kernel.ark_xpu_kv_update(cache_k, cache_v, k, v, 5, tensor_layout="HND")
