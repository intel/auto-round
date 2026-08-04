# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Standard public CPU sdpa() tests.

This module covers only the standard sdpa() contract: mask, causal behavior,
scale, dtype, GQA, prefill/decode behavior, and homogeneous route
hit/fallback without touching internal mixed-route-only features.
"""

import inspect
import math
import sys
from pathlib import Path

import pytest
import torch

cpuinfo = pytest.importorskip("cpuinfo")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

auto_round_kernel = pytest.importorskip(
    "auto_round_kernel", reason="compiled ARK extension not built in this environment"
)

if auto_round_kernel.cpu_lib is None:
    pytest.skip("ARK CPU extension not available", allow_module_level=True)

CPU_FLAGS = set(cpuinfo.get_cpu_info().get("flags", []))
HAS_AVX512_FP16 = "avx512_fp16" in CPU_FLAGS
HAS_AMX_BF16 = "amx_bf16" in CPU_FLAGS
_cpu_lib = auto_round_kernel.cpu_lib
BUILD_HAS_FP16_ROUTE = bool(getattr(_cpu_lib, "ARK_CPU_SDPA_BUILD_HAS_FP16_ROUTE", False))
BUILD_HAS_BF16_ROUTE = bool(getattr(_cpu_lib, "ARK_CPU_SDPA_BUILD_HAS_BF16_ROUTE", False))
ROUTE_SCALAR = getattr(_cpu_lib, "ARK_CPU_SDPA_ROUTE_SCALAR", 0)
ROUTE_HOMOGENEOUS_FP16 = getattr(_cpu_lib, "ARK_CPU_SDPA_ROUTE_HOMOGENEOUS_FP16", 3)
ROUTE_HOMOGENEOUS_BF16 = getattr(_cpu_lib, "ARK_CPU_SDPA_ROUTE_HOMOGENEOUS_BF16", 4)


def test_public_xpu_attention_api_keeps_return_lse_kwargs():
    for fn in (
        auto_round_kernel.sdpa,
        auto_round_kernel.sagev1,
        auto_round_kernel.sagev1_pvi8,
        auto_round_kernel.sageattn,
    ):
        param = inspect.signature(fn).parameters["return_lse"]
        assert param.default is False


def test_ark_cpu_sdpa_rejects_return_lse():
    torch.manual_seed(2036)
    q = torch.randn(1, 2, 4, 8, dtype=torch.float32)
    k = torch.randn(1, 2, 4, 8, dtype=torch.float32)
    v = torch.randn(1, 2, 4, 8, dtype=torch.float32)

    with pytest.raises(NotImplementedError, match="return_lse is not supported on CPU"):
        auto_round_kernel.sdpa(q, k, v, return_lse=True)


def _to_layout(tensor_hnd, layout):
    if layout == "HND":
        return tensor_hnd.contiguous()
    if layout == "NHD":
        tensor_nhd = tensor_hnd.transpose(1, 2)
        _, seq_len, num_heads, head_dim = tensor_nhd.shape
        canonical_nhd = torch.empty_strided(
            tensor_nhd.shape,
            (seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, 1),
            dtype=tensor_nhd.dtype,
            device=tensor_nhd.device,
        )
        canonical_nhd.copy_(tensor_nhd)
        return canonical_nhd
    raise ValueError(layout)


def _to_hnd(tensor, layout):
    return tensor if layout == "HND" else tensor.transpose(1, 2)


def _resolved_cpu_sdpa_route(query, key, value, **kwargs):
    return auto_round_kernel.internal.cpu.debug_resolve_sdpa_route(query, key, value, **kwargs)


def _public_fp16_hom_route_expected():
    return ROUTE_HOMOGENEOUS_FP16 if (HAS_AVX512_FP16 and BUILD_HAS_FP16_ROUTE) else ROUTE_SCALAR


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

    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale, enable_gqa=True, is_causal=False)
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

    expected_hnd = torch.nn.functional.scaled_dot_product_attention(q_hnd, k_hnd, v_hnd, scale=scale, is_causal=True)
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
def test_homogeneous_additive_mask_falls_back_to_scalar(dtype):
    torch.manual_seed(3005)
    q = torch.randn(1, 2, 4, 16, dtype=dtype)
    k = torch.randn(1, 2, 8, 16, dtype=dtype)
    v = torch.randn(1, 2, 8, 16, dtype=dtype)
    mask = torch.randn(1, 1, 4, 8, dtype=torch.float32)

    route = _resolved_cpu_sdpa_route(q, k, v, attn_mask=mask)
    expected = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=mask)
    actual = auto_round_kernel.sdpa(q, k, v, attn_mask=mask)

    assert route == ROUTE_SCALAR
    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)


def test_ark_cpu_sdpa_rejects_strided_hnd_inputs():
    torch.manual_seed(3006)
    q = torch.randn(1, 2, 16, 8, dtype=torch.float32)[:, :, ::2, :]
    k = torch.randn(1, 2, 24, 8, dtype=torch.float32)[:, :, ::2, :]
    v = torch.randn(1, 2, 24, 8, dtype=torch.float32)[:, :, ::2, :]

    with pytest.raises(ValueError, match="do not match canonical"):
        auto_round_kernel.sdpa(q, k, v)


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


# ---------------------------------------------------------------------------
# Module C: homogeneous runtime backend semantics.
#
# Runtime dispatch may now select the homogeneous fp16 backend (route 3) for
# eligible fp16 inputs and the homogeneous bf16 backend (route 4) for the narrow
# no-GQA bf16 contract. Unsupported requests still fall back to Tier-0 scalar.
# These tests assert semantic stability rather than a specific backend choice.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_homogeneous_half_preserves_sdpa_semantics(dtype):
    torch.manual_seed(4100)
    batch, heads, seq, head_dim = 1, 4, 32, 16
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(batch, heads, seq, head_dim, dtype=dtype)
    k = torch.randn(batch, heads, seq, head_dim, dtype=dtype)
    v = torch.randn(batch, heads, seq, head_dim, dtype=dtype)

    expected = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=scale)

    out = auto_round_kernel.sdpa(q, k, v, scale=scale)

    assert out.dtype == dtype
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("layout", ["HND", "NHD"])
def test_fp16_homogeneous_route_resolution_prefill_causal(layout):
    torch.manual_seed(4103)
    batch, heads_q, heads_kv, seq, head_dim = 1, 4, 2, 32, 16
    q_hnd = torch.randn(batch, heads_q, seq, head_dim, dtype=torch.float16)
    k_hnd = torch.randn(batch, heads_kv, seq, head_dim, dtype=torch.float16)
    v_hnd = torch.randn(batch, heads_kv, seq, head_dim, dtype=torch.float16)

    route = _resolved_cpu_sdpa_route(
        _to_layout(q_hnd, layout),
        _to_layout(k_hnd, layout),
        _to_layout(v_hnd, layout),
        is_causal=True,
        tensor_layout=layout,
    )
    assert route == _public_fp16_hom_route_expected()


@pytest.mark.parametrize("layout", ["HND", "NHD"])
def test_fp16_homogeneous_route_resolution_decode_gqa(layout):
    torch.manual_seed(4104)
    batch, heads_q, heads_kv, seq_q, seq_kv, head_dim = 1, 8, 2, 1, 48, 16
    q_hnd = torch.randn(batch, heads_q, seq_q, head_dim, dtype=torch.float16)
    k_hnd = torch.randn(batch, heads_kv, seq_kv, head_dim, dtype=torch.float16)
    v_hnd = torch.randn(batch, heads_kv, seq_kv, head_dim, dtype=torch.float16)

    route = _resolved_cpu_sdpa_route(
        _to_layout(q_hnd, layout),
        _to_layout(k_hnd, layout),
        _to_layout(v_hnd, layout),
        tensor_layout=layout,
    )
    assert route == _public_fp16_hom_route_expected()


@pytest.mark.parametrize("layout", ["HND", "NHD"])
def test_bf16_homogeneous_route_resolution_causal_no_gqa(layout):
    torch.manual_seed(4105)
    batch, heads, seq, head_dim = 1, 4, 32, 16
    q_hnd = torch.randn(batch, heads, seq, head_dim, dtype=torch.bfloat16)
    k_hnd = torch.randn(batch, heads, seq, head_dim, dtype=torch.bfloat16)
    v_hnd = torch.randn(batch, heads, seq, head_dim, dtype=torch.bfloat16)

    route = _resolved_cpu_sdpa_route(
        _to_layout(q_hnd, layout),
        _to_layout(k_hnd, layout),
        _to_layout(v_hnd, layout),
        is_causal=True,
        tensor_layout=layout,
    )
    assert route == (ROUTE_HOMOGENEOUS_BF16 if HAS_AMX_BF16 and BUILD_HAS_BF16_ROUTE else ROUTE_SCALAR)


def test_homogeneous_bf16_gqa_falls_back_without_changing_semantics():
    torch.manual_seed(4102)
    batch, heads_q, heads_kv, seq, head_dim = 1, 4, 2, 24, 16
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(batch, heads_q, seq, head_dim, dtype=torch.bfloat16)
    k = torch.randn(batch, heads_kv, seq, head_dim, dtype=torch.bfloat16)
    v = torch.randn(batch, heads_kv, seq, head_dim, dtype=torch.bfloat16)

    expected = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), scale=scale, enable_gqa=True
    )
    route = _resolved_cpu_sdpa_route(q, k, v, scale=scale)
    assert route == ROUTE_SCALAR
    actual = auto_round_kernel.sdpa(q, k, v, scale=scale)

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)


def test_public_mixed_sdpa_reuses_hidden_packed_kv_cache(monkeypatch):
    torch.manual_seed(4110)
    q = torch.randn(1, 8, 1, 16, dtype=torch.float32)
    k = torch.randn(1, 2, 64, 16, dtype=torch.float16)
    v = torch.randn(1, 2, 64, 16, dtype=torch.float16)

    call_count = 0
    original = auto_round_kernel.ark_cpu_update_packed_kv_from_descriptor

    def counted_update(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(auto_round_kernel, "ark_cpu_update_packed_kv_from_descriptor", counted_update)

    out1 = auto_round_kernel.sdpa(q, k, v)
    out2 = auto_round_kernel.sdpa(q, k, v)

    assert call_count == 1
    torch.testing.assert_close(out1, out2, atol=0, rtol=0)


def test_public_mixed_sdpa_refreshes_hidden_packed_kv_cache_after_mutation(monkeypatch):
    torch.manual_seed(4111)
    q = torch.randn(1, 8, 1, 16, dtype=torch.float32)
    k = torch.randn(1, 2, 64, 16, dtype=torch.float16)
    v = torch.randn(1, 2, 64, 16, dtype=torch.float16)

    call_count = 0
    original = auto_round_kernel.ark_cpu_update_packed_kv_from_descriptor

    def counted_update(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(auto_round_kernel, "ark_cpu_update_packed_kv_from_descriptor", counted_update)

    out1 = auto_round_kernel.sdpa(q, k, v)
    k.add_(0.25)
    out2 = auto_round_kernel.sdpa(q, k, v)

    assert call_count == 2
    assert not torch.equal(out1, out2)
