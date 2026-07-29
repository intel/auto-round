# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Internal/experimental CPU SDPA route tests.

This module covers behavior that is intentionally outside the public
``auto_round_kernel.sdpa()`` contract:

1. Public/internal API boundary checks for non-standard kwargs.
2. BestLA mixed-route-only features (`prefer_fp32`, `n_padding`, `use_alibi`,
   `use_tanh`) exercised through the private CPU binding.
3. Packed KV descriptor/cache helpers and route-specific validators.
"""

import math
import sys
from pathlib import Path
import pytest
cpuinfo = pytest.importorskip("cpuinfo")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import auto_round_kernel


_TOL = {torch.float16: (3e-2, 3e-2), torch.bfloat16: (8e-2, 8e-2)}
INTERNAL_CPU = auto_round_kernel.internal.cpu
CPU_FLAGS = set(cpuinfo.get_cpu_info().get("flags", []))
HAS_AVX2 = "avx2" in CPU_FLAGS
HAS_AVX512F = "avx512f" in CPU_FLAGS
HAS_AMX_BF16 = "amx_bf16" in CPU_FLAGS
BUILD_HAS_BF16_ROUTE = bool(auto_round_kernel.cpu_lib.ARK_CPU_SDPA_BUILD_HAS_BF16_ROUTE)
INTERNAL_FEATURES_ENABLED = bool(auto_round_kernel.cpu_lib.ARK_CPU_SDPA_INTERNAL_FEATURES_ENABLED)
pytestmark = pytest.mark.skipif(
    not INTERNAL_FEATURES_ENABLED,
    reason="internal SDPA features are disabled; rebuild with -DARK_ENABLE_INTERNAL_SDPA_FEATURES=ON",
)


def _resolved_cpu_sdpa_route(query, key, value, **kwargs):
    return INTERNAL_CPU.debug_resolve_sdpa_route(query, key, value, **kwargs)


def _mixed_sdpa_ex(q, k, v, scale, *, is_causal=False, layout="HND", **kwargs):
    """Exercise BestLA-only CPU ABI knobs outside the public sdpa() contract."""
    batch, num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim = auto_round_kernel._validate_attention_geometry(
        q, k, v, layout, key_dtype=k.dtype, value_dtype=v.dtype
    )
    out = auto_round_kernel._empty_attention_output(
        batch,
        num_heads_q,
        seq_len_q,
        head_dim,
        dtype=torch.float32,
        device=q.device,
        tensor_layout=layout,
    )
    q_strides = auto_round_kernel._attention_strides_qko(q, layout)
    k_strides = auto_round_kernel._attention_strides_qko(k, layout)
    v_strides = auto_round_kernel._attention_strides_v(v, layout)
    o_strides = auto_round_kernel._attention_strides_qko(out, layout)
    normalized_n_padding = auto_round_kernel._normalize_batch_padding(kwargs.get("n_padding"), batch)
    auto_round_kernel.cpu_lib.sdpa(
        0,
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        out.data_ptr(),
        0,
        *q_strides,
        *k_strides,
        *v_strides,
        *o_strides,
        auto_round_kernel.cvt_dtype(q.dtype),
        auto_round_kernel.cvt_dtype(k.dtype),
        auto_round_kernel.cvt_dtype(out.dtype),
        batch,
        num_heads_q,
        num_heads_kv,
        seq_len_q,
        seq_len_kv,
        head_dim,
        float(scale),
        bool(is_causal),
        bool(kwargs.get("use_alibi", False)),
        bool(kwargs.get("use_tanh", False)),
        bool(kwargs.get("prefer_fp32", False)),
        normalized_n_padding,
    )
    return out


def _alibi_slope(h: int, head_num: int) -> float:
    n_log2 = 1 << int(math.floor(math.log2(head_num)))
    m0 = 2.0 ** (-8.0 / n_log2)
    m1 = 2.0 ** (-4.0 / n_log2)
    return m0 ** (h + 1) if h < n_log2 else m1 ** (2 * (h - n_log2) + 1)


def _scalar_attn_ref(q_f32, k_rt_f32, v_rt_f32, scale, *, use_tanh=False, slopes=None, n_valid=None):
    B, Hq, Sq, D = q_f32.shape
    _, Hkv, Sk, _ = k_rt_f32.shape
    gqa_ratio = Hq // Hkv
    inner_scale = scale / 30.0 if use_tanh else scale
    if n_valid is None:
        n_valid_actual = [Sk] * B
    elif isinstance(n_valid, int):
        n_valid_actual = [n_valid] * B
    else:
        n_valid_actual = list(n_valid)
        if len(n_valid_actual) != B:
            raise ValueError(f"n_valid must have one entry per batch item, got {len(n_valid_actual)} for batch {B}")
    out = torch.zeros(B, Hq, Sq, D, dtype=torch.float32)
    for b in range(B):
        n_valid_b = n_valid_actual[b]
        for hq in range(Hq):
            hkv = hq // gqa_ratio
            slope = slopes[hq].item() if slopes is not None else 0.0
            for i in range(Sq):
                q_row = q_f32[b, hq, i]
                scores = torch.full((Sk,), float("-inf"))
                for k_pos in range(n_valid_b):
                    k_row = k_rt_f32[b, hkv, k_pos]
                    dot = float((q_row * k_row).sum())
                    s = dot * inner_scale
                    if use_tanh:
                        s = 30.0 * math.tanh(s)
                    s += slope * k_pos
                    scores[k_pos] = s
                valid = scores[:n_valid_b]
                valid = valid - valid.max()
                exp_v = valid.exp()
                attn = exp_v / exp_v.sum()
                v_slice = v_rt_f32[b, hkv, :n_valid_b]
                out[b, hq, i] = (attn.unsqueeze(-1) * v_slice).sum(0)
    return out


def _packed_sdpa(q_f32, k, v, scale, *, is_causal=False, n_padding=None):
    batch, heads_kv, seq_kv, head_dim = k.shape
    handle = INTERNAL_CPU.PackedKVHandle.create(batch, heads_kv, seq_kv, head_dim, dtype=k.dtype)
    cache_k, cache_v = handle.alloc()
    handle.update(cache_k, cache_v, k, v, 0)
    return handle.forward(q_f32, cache_k, cache_v, seq_kv, is_causal=is_causal, scale=scale, n_padding=n_padding)


@pytest.mark.parametrize(
    ("dtype", "feature_kwargs", "kwarg_name"),
    [
        (torch.float16, {"use_alibi": True}, "use_alibi"),
        (torch.float16, {"use_tanh": True}, "use_tanh"),
        (torch.float16, {"n_padding": 12}, "n_padding"),
        (torch.bfloat16, {"prefer_fp32": True}, "prefer_fp32"),
    ],
)
def test_public_sdpa_rejects_nonstandard_kwargs(dtype, feature_kwargs, kwarg_name):
    torch.manual_seed(4106)
    batch, heads, seq, head_dim = 1, 4, 24, 16
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(batch, heads, seq, head_dim, dtype=dtype)
    k = torch.randn(batch, heads, seq, head_dim, dtype=dtype)
    v = torch.randn(batch, heads, seq, head_dim, dtype=dtype)

    baseline_route = _resolved_cpu_sdpa_route(q, k, v, scale=scale)
    route = _resolved_cpu_sdpa_route(q, k, v, scale=scale, **feature_kwargs)
    assert route == baseline_route

    with pytest.raises(TypeError, match=rf"unexpected keyword argument '{kwarg_name}'"):
        auto_round_kernel.sdpa(q, k, v, scale=scale, **feature_kwargs)


def test_internal_cpu_kv_update_append_matches_full_attention():
    torch.manual_seed(2029)
    batch, heads_q, heads_kv, head_dim = 1, 4, 2, 8
    chunks = [5, 7, 3]
    capacity = sum(chunks)
    scale = 1 / math.sqrt(head_dim)
    q = torch.randn(batch, heads_q, 1, head_dim, dtype=torch.float32)
    k_full = torch.randn(batch, heads_kv, capacity, head_dim, dtype=torch.float32)
    v_full = torch.randn(batch, heads_kv, capacity, head_dim, dtype=torch.float32)
    k_cache, v_cache = INTERNAL_CPU.kv_cache_alloc(batch, heads_kv, capacity, head_dim)

    pos = 0
    for chunk in chunks:
        INTERNAL_CPU.kv_update(
            k_cache,
            v_cache,
            k_full[:, :, pos : pos + chunk, :],
            v_full[:, :, pos : pos + chunk, :],
            pos,
        )
        pos += chunk

    expected = torch.nn.functional.scaled_dot_product_attention(q, k_full, v_full, scale=scale, enable_gqa=True)
    actual = auto_round_kernel.sdpa(q, k_cache, v_cache, scale=scale)

    torch.testing.assert_close(k_cache, k_full, atol=0, rtol=0)
    torch.testing.assert_close(v_cache, v_full, atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.bfloat16])
def test_bestla_mixed_sdpa_prefer_fp32_is_accepted(kv_dtype):
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


@pytest.mark.parametrize("feature_kwargs", [{}, {"prefer_fp32": True}])
def test_bf16_mixed_route_accepts_avx512f_or_amx_bf16(feature_kwargs):
    """The BF16 mixed entry gate accepts either supported ISA."""
    if not BUILD_HAS_BF16_ROUTE:
        pytest.skip("BF16 mixed route was not compiled")
    q = torch.randn(1, 4, 4, 32, dtype=torch.float32)
    k = torch.randn(1, 2, 8, 32, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 8, 32, dtype=torch.bfloat16)
    expected_route = _resolved_cpu_sdpa_route(q, k, v, **feature_kwargs)
    assert expected_route == 1
    if not (HAS_AVX512F or HAS_AMX_BF16):
        with pytest.raises(RuntimeError, match="AVX512F or AMX-BF16"):
            _mixed_sdpa_ex(q, k, v, 1 / math.sqrt(32), **feature_kwargs)
        return
    out = _mixed_sdpa_ex(q, k, v, 1 / math.sqrt(32), **feature_kwargs)
    assert out.shape == q.shape


def test_bf16_mixed_avx512f_only_route():
    if not (HAS_AVX512F and not HAS_AMX_BF16):
        pytest.skip("requires AVX512F without AMX-BF16")
    q = torch.randn(1, 4, 4, 32, dtype=torch.float32)
    k = torch.randn(1, 2, 8, 32, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 8, 32, dtype=torch.bfloat16)
    out = _mixed_sdpa_ex(q, k, v, 1 / math.sqrt(32))
    assert out.shape == q.shape


def test_bf16_mixed_amx_only_route():
    if not (HAS_AMX_BF16 and not HAS_AVX512F):
        pytest.skip("requires AMX-BF16 without AVX512F")
    q = torch.randn(1, 4, 4, 32, dtype=torch.float32)
    k = torch.randn(1, 2, 8, 32, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 8, 32, dtype=torch.bfloat16)
    out = _mixed_sdpa_ex(q, k, v, 1 / math.sqrt(32))
    assert out.shape == q.shape


def test_bf16_mixed_amx_preferred_without_features():
    """A machine with AMX-BF16 may use the AMX route when no feature is enabled."""
    if not HAS_AMX_BF16:
        pytest.skip("requires AMX-BF16")
    q = torch.randn(1, 4, 8, 64, dtype=torch.float32)
    k = torch.randn(1, 2, 32, 64, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 32, 64, dtype=torch.bfloat16)
    out = _mixed_sdpa_ex(q, k, v, 1 / math.sqrt(64))
    assert out.shape == q.shape


@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.bfloat16])
def test_bestla_mixed_sdpa_padding_right_matches_reference(kv_dtype):
    torch.manual_seed(7002)
    batch, heads_q, heads_kv, head_dim, seq_q, seq_kv = 1, 4, 2, 32, 4, 8
    n_padding = seq_kv // 2
    scale = 1 / math.sqrt(head_dim)
    q = torch.randn(batch, heads_q, seq_q, head_dim, dtype=torch.float32)
    k = torch.randn(batch, heads_kv, seq_kv, head_dim, dtype=kv_dtype)
    v = torch.randn(batch, heads_kv, seq_kv, head_dim, dtype=kv_dtype)
    if kv_dtype == torch.float16 and not HAS_AVX2:
        with pytest.raises(RuntimeError, match="mixed fp16 extended features require the AVX2"):
            _mixed_sdpa_ex(q, k, v, scale, n_padding=n_padding)
        return
    try:
        actual = _mixed_sdpa_ex(q, k, v, scale, n_padding=n_padding)
    except (RuntimeError, ValueError) as exc:
        pytest.skip(f"BestLA mixed path unavailable on this ISA/runtime: {exc}")
    expected = _scalar_attn_ref(q, k.float(), v.float(), scale, n_valid=n_padding)
    atol, rtol = _TOL[kv_dtype]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.bfloat16])
def test_bestla_mixed_sdpa_padding_right_batch_vector_matches_reference(kv_dtype):
    torch.manual_seed(70021)
    batch, heads_q, heads_kv, head_dim, seq_q, seq_kv = 2, 4, 2, 32, 3, 8
    n_padding = [3, 6]
    scale = 1 / math.sqrt(head_dim)
    q = torch.randn(batch, heads_q, seq_q, head_dim, dtype=torch.float32)
    k = torch.randn(batch, heads_kv, seq_kv, head_dim, dtype=kv_dtype)
    v = torch.randn(batch, heads_kv, seq_kv, head_dim, dtype=kv_dtype)
    try:
        actual = _mixed_sdpa_ex(q, k, v, scale, n_padding=n_padding)
    except (RuntimeError, ValueError) as exc:
        pytest.skip(f"BestLA mixed path unavailable on this ISA/runtime: {exc}")
    expected = _scalar_attn_ref(q, k.float(), v.float(), scale, n_valid=n_padding)
    atol, rtol = _TOL[kv_dtype]
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.bfloat16])
def test_bestla_mixed_sdpa_alibi_matches_reference(kv_dtype):
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
    expected = _scalar_attn_ref(q, k.float(), v.float(), scale, slopes=slopes)
    atol, rtol = _TOL[kv_dtype]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_bestla_mixed_sdpa_tanh_matches_reference():
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
    expected = _scalar_attn_ref(q, k.float(), v.float(), scale, use_tanh=True)
    atol, rtol = _TOL[kv_dtype]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_causal", [False, True])
def test_bestla_packed_sdpa_numerical_parity(kv_dtype, is_causal):
    torch.manual_seed(8001)
    batch, heads_q, heads_kv, head_dim, seq_q, seq_kv = 1, 8, 2, 64, 1, 32
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(batch, heads_q, seq_q, head_dim, dtype=torch.float32)
    k = torch.randn(batch, heads_kv, seq_kv, head_dim, dtype=kv_dtype)
    v = torch.randn(batch, heads_kv, seq_kv, head_dim, dtype=kv_dtype)

    try:
        actual = _packed_sdpa(q, k, v, scale, is_causal=is_causal)
    except (RuntimeError, ValueError, NotImplementedError) as exc:
        pytest.skip(f"BestLA packed path unavailable on this ISA/runtime: {exc}")

    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k.float(), v.float(), scale=scale, enable_gqa=True, is_causal=is_causal
    )
    atol, rtol = _TOL[kv_dtype]
    assert actual.dtype == torch.float32
    assert actual.shape == (batch, heads_q, seq_q, head_dim)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.bfloat16])
def test_bestla_raw_vs_packed_output_consistency(kv_dtype):
    torch.manual_seed(8002)
    batch, heads_q, heads_kv, head_dim, seq_q, seq_kv = 1, 4, 2, 64, 1, 16
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(batch, heads_q, seq_q, head_dim, dtype=torch.float32)
    k = torch.randn(batch, heads_kv, seq_kv, head_dim, dtype=kv_dtype)
    v = torch.randn(batch, heads_kv, seq_kv, head_dim, dtype=kv_dtype)

    try:
        out_raw = _mixed_sdpa_ex(q, k, v, scale)
    except (RuntimeError, ValueError) as exc:
        pytest.skip(f"BestLA raw path unavailable on this ISA/runtime: {exc}")

    try:
        out_packed = _packed_sdpa(q, k, v, scale)
    except (RuntimeError, ValueError, NotImplementedError) as exc:
        pytest.skip(f"BestLA packed path unavailable on this ISA/runtime: {exc}")

    assert out_raw.dtype == torch.float32
    assert out_packed.dtype == torch.float32
    atol, rtol = _TOL[kv_dtype]
    torch.testing.assert_close(out_raw, out_packed, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    ("kv_dtype", "expected_layout", "expected_ntile", "expected_rowpack"),
    [(torch.float16, 3, 24, 1), (torch.bfloat16, 2, 48, 2)],
)
def test_packed_kv_info_reports_runtime_descriptor(kv_dtype, expected_layout, expected_ntile, expected_rowpack):
    descriptor = INTERNAL_CPU.packed_kv_descriptor(2, 3, 17, 33, dtype=kv_dtype)
    info = INTERNAL_CPU.packed_kv_info(descriptor=descriptor)
    assert info["batch_size"] == 2
    assert info["heads_kv"] == 3
    assert info["logical_capacity"] == 17
    assert info["head_dim"] == 33
    assert info["layout"] == expected_layout
    assert info["k_layout"] == expected_layout
    assert info["v_layout"] == expected_layout
    assert info["ntile"] == expected_ntile
    assert info["rowpack"] == expected_rowpack
    assert info["k_bytes"] == info["k_total_elems"] * info["elem_bytes"]
    assert info["v_bytes"] == info["v_total_elems"] * info["elem_bytes"]
    assert info["step_k_bs"] == info["step_k_head_num"] * info["heads_kv"]
    assert info["step_v_bs"] == info["step_v_head_num"] * info["heads_kv"]
    legacy = INTERNAL_CPU.packed_kv_info(2, 3, 17, 33, dtype=kv_dtype)
    assert info == legacy


@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.bfloat16])
def test_packed_kv_update_append_matches_one_shot(kv_dtype):
    torch.manual_seed(8100)
    batch, heads_kv, capacity, head_dim = 2, 2, 9, 17
    handle = INTERNAL_CPU.PackedKVHandle.create(batch, heads_kv, capacity, head_dim, dtype=kv_dtype)
    one_shot_k, one_shot_v = handle.alloc()
    append_k, append_v = handle.alloc()
    key = torch.randn(batch, heads_kv, capacity, head_dim, dtype=kv_dtype)
    value = torch.randn(batch, heads_kv, capacity, head_dim, dtype=kv_dtype)
    handle.update(one_shot_k, one_shot_v, key, value, 0)
    split = 4
    handle.update(append_k, append_v, key[:, :, :split], value[:, :, :split], 0)
    handle.update(append_k, append_v, key[:, :, split:], value[:, :, split:], split)
    assert torch.equal(one_shot_k, append_k)
    assert torch.equal(one_shot_v, append_v)


@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.bfloat16])
def test_packed_kv_update_no_zeroing_preserves_padding(kv_dtype):
    batch, heads_kv, capacity, head_dim = 1, 1, 5, 17
    key = torch.ones(batch, heads_kv, 1, head_dim, dtype=kv_dtype)
    value = torch.ones(batch, heads_kv, 1, head_dim, dtype=kv_dtype)
    cache_k_zero, cache_v_zero = INTERNAL_CPU.packed_kv_alloc(batch, heads_kv, capacity, head_dim, dtype=kv_dtype)
    cache_k_keep = torch.full_like(cache_k_zero, 3)
    cache_v_keep = torch.full_like(cache_v_zero, 3)
    cache_k_zero.fill_(3)
    cache_v_zero.fill_(3)
    INTERNAL_CPU.update_packed_kv(cache_k_zero, cache_v_zero, key, value, 0, capacity, no_zeroing=False)
    INTERNAL_CPU.update_packed_kv(cache_k_keep, cache_v_keep, key, value, 0, capacity, no_zeroing=True)
    assert torch.count_nonzero(cache_k_zero == 0) > 0 or torch.count_nonzero(cache_v_zero == 0) > 0
    assert torch.count_nonzero(cache_k_keep == 3) > 0 or torch.count_nonzero(cache_v_keep == 3) > 0
    assert not torch.equal(cache_k_zero, cache_k_keep) or not torch.equal(cache_v_zero, cache_v_keep)


@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.bfloat16])
def test_packed_kv_copy_replays_packed_region(kv_dtype):
    torch.manual_seed(8101)
    batch, heads_kv, capacity, head_dim = 2, 2, 9, 17
    handle = INTERNAL_CPU.PackedKVHandle.create(batch, heads_kv, capacity, head_dim, dtype=kv_dtype)
    src_k, src_v = handle.alloc()
    dst_k = torch.full_like(src_k, 5)
    dst_v = torch.full_like(src_v, 5)
    key = torch.randn(batch, heads_kv, capacity, head_dim, dtype=kv_dtype)
    value = torch.randn(batch, heads_kv, capacity, head_dim, dtype=kv_dtype)
    handle.update(src_k, src_v, key, value, 0, no_zeroing=False)
    handle.copy(dst_k, dst_v, src_k, src_v, 0, capacity, no_zeroing=False)
    assert torch.equal(dst_k, src_k)
    assert torch.equal(dst_v, src_v)


def test_packed_k_shift_rope_zero_scale_mutates_only_suffix():
    kv_dtype = torch.bfloat16
    batch, heads_kv, capacity, head_dim = 1, 1, 8, 32
    handle = INTERNAL_CPU.PackedKVHandle.create(batch, heads_kv, capacity, head_dim, dtype=kv_dtype)
    cache_k, cache_v = handle.alloc()
    key = torch.ones(batch, heads_kv, capacity, head_dim, dtype=kv_dtype)
    value = torch.ones(batch, heads_kv, capacity, head_dim, dtype=kv_dtype)
    handle.update(cache_k, cache_v, key, value, 0)
    before = cache_k.clone()
    cossin = torch.zeros(head_dim, dtype=torch.float16)
    handle.shift_k(cache_k, cossin, seq_keep=1)
    assert not torch.equal(cache_k, before)
    assert torch.count_nonzero(cache_k == 0) > torch.count_nonzero(before == 0)


@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.bfloat16])
def test_packed_sdpa_padding_right_batch_vector_matches_reference_and_raw(kv_dtype):
    torch.manual_seed(8102)
    batch, heads_q, heads_kv, head_dim, seq_q, seq_kv = 2, 4, 2, 32, 2, 8
    scale = 1.0 / math.sqrt(head_dim)
    n_padding = [2, 6]
    q = torch.randn(batch, heads_q, seq_q, head_dim, dtype=torch.float32)
    k = torch.randn(batch, heads_kv, seq_kv, head_dim, dtype=kv_dtype)
    v = torch.randn(batch, heads_kv, seq_kv, head_dim, dtype=kv_dtype)
    try:
        out_raw = _mixed_sdpa_ex(q, k, v, scale, n_padding=n_padding)
        out_packed = _packed_sdpa(q, k, v, scale, n_padding=n_padding)
    except (RuntimeError, ValueError, NotImplementedError) as exc:
        pytest.skip(f"BestLA packed path unavailable on this ISA/runtime: {exc}")
    expected = _scalar_attn_ref(q, k.float(), v.float(), scale, n_valid=n_padding)
    atol, rtol = _TOL[kv_dtype]
    torch.testing.assert_close(out_packed, expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(out_raw, out_packed, atol=atol, rtol=rtol)


def test_padding_and_causal_are_mutually_exclusive_raw_and_packed():
    q = torch.randn(1, 4, 2, 32, dtype=torch.float32)
    k = torch.randn(1, 2, 8, 32, dtype=torch.float16)
    v = torch.randn(1, 2, 8, 32, dtype=torch.float16)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _mixed_sdpa_ex(q, k, v, scale=1 / math.sqrt(32), is_causal=True, n_padding=[4])
    handle = INTERNAL_CPU.PackedKVHandle.create(1, 2, 8, 32, dtype=torch.float16)
    cache_k, cache_v = handle.alloc()
    handle.update(cache_k, cache_v, k, v, 0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        handle.forward(q, cache_k, cache_v, 8, is_causal=True, scale=1 / math.sqrt(32), n_padding=[4])


def test_debug_route_ignores_nonstandard_kwargs_for_homogeneous_paths():
    torch.manual_seed(4110)
    batch, heads, seq, head_dim = 1, 4, 24, 16
    q_fp16 = torch.randn(batch, heads, seq, head_dim, dtype=torch.float16)
    k_fp16 = torch.randn(batch, heads, seq, head_dim, dtype=torch.float16)
    v_fp16 = torch.randn(batch, heads, seq, head_dim, dtype=torch.float16)
    q_bf16 = torch.randn(batch, heads, seq, head_dim, dtype=torch.bfloat16)
    k_bf16 = torch.randn(batch, heads, seq, head_dim, dtype=torch.bfloat16)
    v_bf16 = torch.randn(batch, heads, seq, head_dim, dtype=torch.bfloat16)

    assert _resolved_cpu_sdpa_route(q_fp16, k_fp16, v_fp16, use_alibi=True) == _resolved_cpu_sdpa_route(q_fp16, k_fp16, v_fp16)
    assert _resolved_cpu_sdpa_route(q_bf16, k_bf16, v_bf16, prefer_fp32=True) == _resolved_cpu_sdpa_route(q_bf16, k_bf16, v_bf16)
    assert _resolved_cpu_sdpa_route(q_fp16, k_fp16, v_fp16, n_padding=[seq]) == _resolved_cpu_sdpa_route(q_fp16, k_fp16, v_fp16)
