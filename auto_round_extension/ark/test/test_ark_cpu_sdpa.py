# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import auto_round_kernel


def _to_layout(tensor_hnd, layout):
    if layout == "HND":
        return tensor_hnd.contiguous()
    if layout == "NHD":
        return tensor_hnd.transpose(1, 2).contiguous()
    raise ValueError(layout)


def _to_hnd(tensor, layout):
    return tensor if layout == "HND" else tensor.transpose(1, 2)


@pytest.mark.parametrize("layout", ["HND", "NHD"])
def test_ark_cpu_sdpa_decode_matches_torch_for_layout(layout):
    torch.manual_seed(2026)
    batch, seq_q, seq_kv, heads_q, heads_kv, head_dim = 2, 1, 128, 32, 8, 16
    scale = 1 / math.sqrt(head_dim)
    q_hnd = torch.randn(batch, heads_q, seq_q, head_dim, dtype=torch.float32)
    k_hnd = torch.randn(batch, heads_kv, seq_kv, head_dim, dtype=torch.float32)
    v_hnd = torch.randn(batch, heads_kv, seq_kv, head_dim, dtype=torch.float32)

    expected_hnd = torch.nn.functional.scaled_dot_product_attention(
        q_hnd,
        k_hnd,
        v_hnd,
        scale=scale,
        enable_gqa=True,
        is_causal=False,
    )
    actual = auto_round_kernel.sdpa(
        _to_layout(q_hnd, layout),
        _to_layout(k_hnd, layout),
        _to_layout(v_hnd, layout),
        scale=scale,
        tensor_layout=layout,
    )

    torch.testing.assert_close(_to_hnd(actual, layout), expected_hnd, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("layout", ["HND", "NHD"])
def test_ark_cpu_sdpa_prefill_causal_matches_torch_for_layout(layout):
    torch.manual_seed(2027)
    batch, seq, heads, head_dim = 1, 64, 4, 16
    scale = 1 / math.sqrt(head_dim)
    q_hnd = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    k_hnd = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    v_hnd = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)

    expected_hnd = torch.nn.functional.scaled_dot_product_attention(
        q_hnd,
        k_hnd,
        v_hnd,
        scale=scale,
        is_causal=True,
    )
    actual = auto_round_kernel.sdpa(
        _to_layout(q_hnd, layout),
        _to_layout(k_hnd, layout),
        _to_layout(v_hnd, layout),
        scale=scale,
        is_causal=True,
        tensor_layout=layout,
    )

    torch.testing.assert_close(_to_hnd(actual, layout), expected_hnd, atol=1e-5, rtol=1e-5)


def test_ark_cpu_sdpa_nhd_and_hnd_are_equivalent():
    torch.manual_seed(2028)
    batch, seq_q, seq_kv, heads, head_dim = 2, 17, 23, 3, 8
    scale = 1 / math.sqrt(head_dim)
    q_hnd = torch.randn(batch, heads, seq_q, head_dim, dtype=torch.float32)
    k_hnd = torch.randn(batch, heads, seq_kv, head_dim, dtype=torch.float32)
    v_hnd = torch.randn(batch, heads, seq_kv, head_dim, dtype=torch.float32)

    out_hnd = auto_round_kernel.sdpa(q_hnd, k_hnd, v_hnd, scale=scale, tensor_layout="HND")
    out_nhd = auto_round_kernel.sdpa(
        _to_layout(q_hnd, "NHD"),
        _to_layout(k_hnd, "NHD"),
        _to_layout(v_hnd, "NHD"),
        scale=scale,
        tensor_layout="NHD",
    )

    torch.testing.assert_close(out_hnd, out_nhd.transpose(1, 2), atol=0, rtol=0)


def test_ark_cpu_kv_update_append_matches_full_attention():
    torch.manual_seed(2029)
    batch, heads_q, heads_kv, head_dim = 1, 4, 2, 8
    chunks = [5, 7, 3]
    capacity = sum(chunks)
    scale = 1 / math.sqrt(head_dim)
    q = torch.randn(batch, heads_q, 1, head_dim, dtype=torch.float32)
    k_full = torch.randn(batch, heads_kv, capacity, head_dim, dtype=torch.float32)
    v_full = torch.randn(batch, heads_kv, capacity, head_dim, dtype=torch.float32)
    k_cache, v_cache = auto_round_kernel.ark_cpu_kv_cache_alloc(batch, heads_kv, capacity, head_dim)

    pos = 0
    for chunk in chunks:
        auto_round_kernel.ark_cpu_kv_update(
            k_cache,
            v_cache,
            k_full[:, :, pos : pos + chunk, :],
            v_full[:, :, pos : pos + chunk, :],
            pos,
        )
        pos += chunk

    expected = torch.nn.functional.scaled_dot_product_attention(
        q,
        k_full,
        v_full,
        scale=scale,
        enable_gqa=True,
    )
    actual = auto_round_kernel.sdpa(q, k_cache, v_cache, scale=scale)

    torch.testing.assert_close(k_cache, k_full, atol=0, rtol=0)
    torch.testing.assert_close(v_cache, v_full, atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_ark_cpu_sdpa_rejects_mask_with_causal():
    q = torch.randn(1, 1, 2, 8, dtype=torch.float32)
    k = torch.randn(1, 1, 2, 8, dtype=torch.float32)
    v = torch.randn(1, 1, 2, 8, dtype=torch.float32)
    mask = torch.zeros(1, 1, 2, 2, dtype=torch.float32)

    with pytest.raises(ValueError, match="mask and is_causal"):
        auto_round_kernel.sdpa(q, k, v, attn_mask=mask, is_causal=True)


@pytest.mark.parametrize("seq_kv", [257, 600])
def test_ark_cpu_sdpa_decode_spans_multiple_kv_tiles(seq_kv):
    # seq_kv larger than the default flash-attention K/V tile (256) exercises the
    # online-softmax rescaling across multiple tiles.
    torch.manual_seed(3001)
    batch, heads_q, heads_kv, head_dim = 2, 8, 2, 16
    scale = 1 / math.sqrt(head_dim)
    q = torch.randn(batch, heads_q, 1, head_dim, dtype=torch.float32)
    k = torch.randn(batch, heads_kv, seq_kv, head_dim, dtype=torch.float32)
    v = torch.randn(batch, heads_kv, seq_kv, head_dim, dtype=torch.float32)

    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, scale=scale, enable_gqa=True, is_causal=False
    )
    actual = auto_round_kernel.sdpa(q, k, v, scale=scale)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("layout", ["HND", "NHD"])
def test_ark_cpu_sdpa_prefill_causal_multi_tile(layout):
    # seq longer than the default tile checks tiled online softmax under causal
    # masking, including tiles that are fully masked for early query rows.
    torch.manual_seed(3002)
    batch, heads, head_dim, seq = 1, 4, 16, 300
    scale = 1 / math.sqrt(head_dim)
    q_hnd = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    k_hnd = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    v_hnd = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)

    expected_hnd = torch.nn.functional.scaled_dot_product_attention(
        q_hnd, k_hnd, v_hnd, scale=scale, is_causal=True
    )
    actual = auto_round_kernel.sdpa(
        _to_layout(q_hnd, layout),
        _to_layout(k_hnd, layout),
        _to_layout(v_hnd, layout),
        scale=scale,
        is_causal=True,
        tensor_layout=layout,
    )

    torch.testing.assert_close(_to_hnd(actual, layout), expected_hnd, atol=1e-5, rtol=1e-5)


def test_ark_cpu_sdpa_additive_mask_multi_tile_matches_torch():
    # Additive float mask combined with multi-tile K/V.
    torch.manual_seed(3003)
    batch, heads, head_dim, seq_q, seq_kv = 2, 3, 16, 4, 400
    scale = 1 / math.sqrt(head_dim)
    q = torch.randn(batch, heads, seq_q, head_dim, dtype=torch.float32)
    k = torch.randn(batch, heads, seq_kv, head_dim, dtype=torch.float32)
    v = torch.randn(batch, heads, seq_kv, head_dim, dtype=torch.float32)
    mask = torch.randn(batch, 1, seq_q, seq_kv, dtype=torch.float32)

    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    actual = auto_round_kernel.sdpa(q, k, v, attn_mask=mask, scale=scale)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ark_cpu_sdpa_decode_half_dtypes_match_torch(dtype):
    torch.manual_seed(3004)
    batch, heads_q, heads_kv, head_dim, seq_kv = 1, 8, 2, 16, 300
    scale = 1 / math.sqrt(head_dim)
    q = torch.randn(batch, heads_q, 1, head_dim, dtype=dtype)
    k = torch.randn(batch, heads_kv, seq_kv, head_dim, dtype=dtype)
    v = torch.randn(batch, heads_kv, seq_kv, head_dim, dtype=dtype)

    # Reference uses the same (already quantized) inputs upcast to fp32 so the
    # comparison isolates kernel error from input rounding.
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), scale=scale, enable_gqa=True
    )
    actual = auto_round_kernel.sdpa(q, k, v, scale=scale)

    assert actual.dtype == dtype
    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)
