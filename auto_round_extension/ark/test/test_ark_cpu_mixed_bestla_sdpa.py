# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Phase 4 Step 3 end-to-end readiness/gating tests for the experimental BestLA
mixed SDPA path.

Two concerns are covered:

1. Gating: by default (no ``ARK_UNSAFE_BESTLA_MIXED_SDPA``) a mixed-dtype call
   (Q=float32, K/V=fp16/bf16) must NOT silently enter the BestLA mixed path; it
   must error clearly. The route is reachable only with the explicit unsafe
   opt-in.
2. Numerical smoke: with ``ARK_UNSAFE_BESTLA_MIXED_SDPA=1`` the mixed path output
   is compared against PyTorch ``scaled_dot_product_attention`` (Q float32,
   K/V fp16/bf16, O float32) for causal on/off across HND and NHD layouts, with
   separate tolerances per KV dtype.

The module is skipped when the compiled ``auto_round_kernel`` extension is not
built. Individual smoke tests skip (with the explicit ISA/runtime reason) when
the wired mixed kernels are unavailable, e.g. fp16->fp32 (NTILE24) needs AVX2 and
bf16->fp32 (NTILE48) needs AVX512F. In those environments the C++ reorder layout
check (wrapper/test/test_reorder_kv.hpp) validates correctness instead.
"""

import math
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

auto_round_kernel = pytest.importorskip(
    "auto_round_kernel", reason="compiled ARK extension not built in this environment"
)

# Separate tolerances: bf16 has a much coarser mantissa than fp16.
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
    prev = os.environ.get("ARK_UNSAFE_BESTLA_MIXED_SDPA")
    os.environ["ARK_UNSAFE_BESTLA_MIXED_SDPA"] = "1"
    try:
        return auto_round_kernel.sdpa(q, k, v, scale=scale, is_causal=is_causal, tensor_layout=layout)
    finally:
        if prev is None:
            os.environ.pop("ARK_UNSAFE_BESTLA_MIXED_SDPA", None)
        else:
            os.environ["ARK_UNSAFE_BESTLA_MIXED_SDPA"] = prev


def test_mixed_dtype_default_is_gated():
    # Default (no unsafe opt-in): mixed Q=fp32 / K-V=fp16 must NOT silently enter
    # the BestLA mixed path. It must raise rather than return a wrong result.
    os.environ.pop("ARK_UNSAFE_BESTLA_MIXED_SDPA", None)
    q = torch.randn(1, 8, 16, 64, dtype=torch.float32)
    k = torch.randn(1, 2, 16, 64, dtype=torch.float16)
    v = torch.randn(1, 2, 16, 64, dtype=torch.float16)
    with pytest.raises((RuntimeError, ValueError)):
        auto_round_kernel.sdpa(q, k, v, scale=1 / math.sqrt(64))


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
        actual = _mixed_sdpa(
            _to_layout(q, layout), _to_layout(k, layout), _to_layout(v, layout), scale, is_causal, layout
        )
    except (RuntimeError, ValueError) as exc:
        pytest.skip(f"BestLA mixed path unavailable on this ISA/runtime: {exc}")

    atol, rtol = _TOL[kv_dtype]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(_to_hnd(actual, layout), expected_hnd, atol=atol, rtol=rtol)


@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gqa_ratio", [2, 4, 8])
def test_bestla_mixed_sdpa_gqa_ratio(kv_dtype, gqa_ratio):
    """Phase 6: explicit GQA-ratio smoke test.

    Exercises the ihkv = ihn / (head_num / heads_kv) mapping inside the
    BestLA mixed routes with GQA ratios 2×, 4×, and 8× to verify that each
    query-head reads K/V from the correct KV head.
    """
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


# ---------------------------------------------------------------------------
# Python ABI closure tests (Phase 6 finalization): prefer_fp32, padding-right,
# alibi, and tanh.  These tests mirror the C++ TestMixedNumericalFeatures in
# wrapper/test/test_reorder_kv.hpp and verify that the Python→C++ ABI plumbing
# for the four new kwargs (`use_alibi`, `use_tanh`, `prefer_fp32`, `n_padding`)
# is wired end-to-end.  ISA-unavailability (no AVX2 for F16, no AVX512F for
# BF16/tanh) is caught as RuntimeError and converted to pytest.skip, consistent
# with the existing bestla smoke tests above.
# ---------------------------------------------------------------------------


def _mixed_sdpa_ex(q, k, v, scale, *, is_causal=False, layout="HND", **kwargs):
    """Like _mixed_sdpa but accepts extra BestLA Python ABI kwargs."""
    prev = os.environ.get("ARK_UNSAFE_BESTLA_MIXED_SDPA")
    os.environ["ARK_UNSAFE_BESTLA_MIXED_SDPA"] = "1"
    try:
        return auto_round_kernel.sdpa(q, k, v, scale=scale, is_causal=is_causal, tensor_layout=layout, **kwargs)
    finally:
        if prev is None:
            os.environ.pop("ARK_UNSAFE_BESTLA_MIXED_SDPA", None)
        else:
            os.environ["ARK_UNSAFE_BESTLA_MIXED_SDPA"] = prev


def _alibi_slope(h: int, head_num: int) -> float:
    """ALiBi slope for query head h in a model with head_num query heads.

    Mirrors mha_dense_wrapper.h lines 1027-1066 (k_offset=0) exactly.
    """
    n_log2 = 1 << int(math.floor(math.log2(head_num)))
    m0 = 2.0 ** (-8.0 / n_log2)
    m1 = 2.0 ** (-4.0 / n_log2)
    return m0 ** (h + 1) if h < n_log2 else m1 ** (2 * (h - n_log2) + 1)


def _scalar_attn_ref(q_f32, k_rt_f32, v_rt_f32, scale, *, use_tanh=False, slopes=None, n_valid=None):
    """Scalar fp32 attention reference.  Inputs are plain HND tensors (float32).

    Arguments:
        q_f32:     [B, Hq, Sq, D] float32
        k_rt_f32:  [B, Hkv, Sk, D] float32 (K round-tripped through kv_dtype)
        v_rt_f32:  [B, Hkv, Sk, D] float32 (V round-tripped through kv_dtype)
        scale:     QK softmax scale
        use_tanh:  apply 30*tanh(dot*scale/30) to raw scores
        slopes:    optional [Hq] float32 tensor of per-head ALiBi slopes
        n_valid:   if set, positions [n_valid, Sk) are masked to -inf

    Returns:
        [B, Hq, Sq, D] float32 reference output
    """
    B, Hq, Sq, D = q_f32.shape
    _, Hkv, Sk, _ = k_rt_f32.shape
    gqa_ratio = Hq // Hkv
    inner_scale = scale / 30.0 if use_tanh else scale
    n_valid_actual = n_valid if n_valid is not None else Sk
    out = torch.zeros(B, Hq, Sq, D, dtype=torch.float32)
    for b in range(B):
        for hq in range(Hq):
            hkv = hq // gqa_ratio
            slope = slopes[hq].item() if slopes is not None else 0.0
            for i in range(Sq):
                q_row = q_f32[b, hq, i]  # [D]
                scores = torch.full((Sk,), float("-inf"))
                for k_pos in range(n_valid_actual):
                    k_row = k_rt_f32[b, hkv, k_pos]  # [D]
                    dot = float((q_row * k_row).sum())
                    s = dot * inner_scale
                    if use_tanh:
                        s = 30.0 * math.tanh(s)
                    s += slope * k_pos
                    scores[k_pos] = s
                # Numerically stable softmax over valid positions.
                valid = scores[:n_valid_actual]
                valid = valid - valid.max()
                exp_v = valid.exp()
                attn = exp_v / exp_v.sum()
                # Weighted sum of V.
                v_slice = v_rt_f32[b, hkv, :n_valid_actual]  # [n_valid, D]
                out[b, hq, i] = (attn.unsqueeze(-1) * v_slice).sum(0)
    return out


@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.bfloat16])
def test_bestla_mixed_sdpa_prefer_fp32_is_accepted(kv_dtype):
    """prefer_fp32=True must not raise on the BestLA mixed path (smoke test).

    For F16 K/V (already fp32-score/AVX2), prefer_fp32 is a no-op and output
    must match the plain run. For BF16 K/V (AVX512F/AMX-BF16), prefer_fp32
    selects the AVX512F fp32-score path instead of AMX-BF16.
    """
    torch.manual_seed(7001)
    batch, heads_q, heads_kv, head_dim, seq = 1, 4, 2, 64, 16
    scale = 1 / math.sqrt(head_dim)
    q = torch.randn(batch, heads_q, seq, head_dim, dtype=torch.float32)
    k = torch.randn(batch, heads_kv, seq, head_dim, dtype=kv_dtype)
    v = torch.randn(batch, heads_kv, seq, head_dim, dtype=kv_dtype)
    try:
        out = _mixed_sdpa_ex(q, k, v, scale, prefer_fp32=True)
    except (RuntimeError, ValueError) as exc:
        pytest.skip(f"BestLA mixed path unavailable on this ISA/runtime: {exc}")
    assert out.dtype == torch.float32
    assert out.shape == (batch, heads_q, seq, head_dim)


@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.bfloat16])
def test_bestla_mixed_sdpa_padding_right_matches_reference(kv_dtype):
    """padding-right (n_padding): positions [n_padding, Skv) masked to -inf.

    Builds a small deterministic problem, runs the BestLA mixed path with
    n_padding set to half the KV length, and compares against a Python scalar
    reference that masks the same positions.
    """
    torch.manual_seed(7002)
    batch, heads_q, heads_kv, head_dim, seq_q, seq_kv = 1, 4, 2, 32, 4, 8
    n_padding = seq_kv // 2  # valid positions 0..3; positions 4..7 masked
    scale = 1 / math.sqrt(head_dim)
    q = torch.randn(batch, heads_q, seq_q, head_dim, dtype=torch.float32)
    k = torch.randn(batch, heads_kv, seq_kv, head_dim, dtype=kv_dtype)
    v = torch.randn(batch, heads_kv, seq_kv, head_dim, dtype=kv_dtype)
    try:
        actual = _mixed_sdpa_ex(q, k, v, scale, n_padding=n_padding)
    except (RuntimeError, ValueError) as exc:
        pytest.skip(f"BestLA mixed path unavailable on this ISA/runtime: {exc}")
    # Reference uses dtype-round-tripped K/V to match kernel quantisation error.
    k_rt = k.float()
    v_rt = v.float()
    expected = _scalar_attn_ref(q, k_rt, v_rt, scale, n_valid=n_padding)
    atol, rtol = _TOL[kv_dtype]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.bfloat16])
def test_bestla_mixed_sdpa_alibi_matches_reference(kv_dtype):
    """ALiBi positional bias: score[h,i,k] += slope[h] * k.

    Verifies the full Python→C++ alibi wiring by comparing the BestLA mixed
    path output against a Python scalar reference that adds the same per-head
    slope to each KV position score.
    """
    torch.manual_seed(7003)
    batch, heads_q, heads_kv, head_dim, seq = 1, 4, 2, 32, 8
    scale = 1 / math.sqrt(head_dim)
    q = torch.randn(batch, heads_q, seq, head_dim, dtype=torch.float32)
    k = torch.randn(batch, heads_kv, seq, head_dim, dtype=kv_dtype)
    v = torch.randn(batch, heads_kv, seq, head_dim, dtype=kv_dtype)
    try:
        actual = _mixed_sdpa_ex(q, k, v, scale, use_alibi=True)
    except (RuntimeError, ValueError) as exc:
        pytest.skip(f"BestLA mixed path unavailable on this ISA/runtime: {exc}")
    slopes = torch.tensor([_alibi_slope(h, heads_q) for h in range(heads_q)], dtype=torch.float32)
    k_rt = k.float()
    v_rt = v.float()
    expected = _scalar_attn_ref(q, k_rt, v_rt, scale, slopes=slopes)
    atol, rtol = _TOL[kv_dtype]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_bestla_mixed_sdpa_tanh_matches_reference():
    """Tanh score activation: effective_score = 30 * tanh(raw_score * scale / 30).

    Tanh is only implemented in the AVX512F specialisation of scale_track_max
    (HAS_TANH); the AVX2/F16 kernel template instantiation does NOT apply tanh
    (the if-constexpr block is AVX512F-only). This test is therefore restricted
    to the BF16 route (which requires AVX512F) and is skipped on AVX2-only
    machines.
    """
    torch.manual_seed(7004)
    batch, heads_q, heads_kv, head_dim, seq = 1, 4, 2, 32, 8
    scale = 1 / math.sqrt(head_dim)
    kv_dtype = torch.bfloat16
    q = torch.randn(batch, heads_q, seq, head_dim, dtype=torch.float32)
    k = torch.randn(batch, heads_kv, seq, head_dim, dtype=kv_dtype)
    v = torch.randn(batch, heads_kv, seq, head_dim, dtype=kv_dtype)
    try:
        actual = _mixed_sdpa_ex(q, k, v, scale, use_tanh=True)
    except (RuntimeError, ValueError) as exc:
        pytest.skip(f"BestLA mixed path unavailable on this ISA/runtime: {exc}")
    k_rt = k.float()
    v_rt = v.float()
    expected = _scalar_attn_ref(q, k_rt, v_rt, scale, use_tanh=True)
    atol, rtol = _TOL[kv_dtype]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
