#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parity tests for the unified ``ark.moe`` dispatcher.

``ark.moe`` is a thin Python-side dispatcher that picks between
``moe_gemm_decode`` (GEMV-tuned) and ``moe_gemm_prefill`` (GEMM-tuned) based
on the token distribution. The contract is that dispatching should never
change the numerical result: ``moe(...)`` must be **bit-identical** to the
underlying kernel that was dispatched.

This file checks:

  * Dispatch correctness: ``phase="auto"`` picks decode when every expert
    sees few tokens and prefill otherwise.
  * Bit-parity: ``moe(phase="auto")`` matches the kernel it dispatched to.
  * Explicit-phase parity: ``moe(phase="decode")`` matches
    ``moe_gemm_decode``, ``moe(phase="prefill")`` matches
    ``moe_gemm_prefill`` (for the same inputs).
  * Coverage across all supported quant schemes (fp / int8 / int4 / int2 /
    fp8) and both activation dtypes.
  * Argument validation (bad ``phase`` raises ``ValueError``).
"""

import auto_round_kernel
import pytest
import torch

# Reuse pack/dequant helpers from the correctness tests.
from test_moe import (  # noqa: E402
    _dequant_fp8,
    _dequant_int2_asym,
    _dequant_int2_sym,
    _dequant_int4_asym,
    _dequant_int4_sym,
    _dequant_int8_asym,
    _dequant_int8_sym,
    _pack_fp8,
    _pack_int2_asym,
    _pack_int2_sym,
    _pack_int4_asym,
    _pack_int4_sym,
    _pack_int8_asym,
    _pack_int8_sym,
)

ark = auto_round_kernel


# ---------------------------------------------------------------------------
# Skip reasons
# ---------------------------------------------------------------------------


def _xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _unified_skip_reason() -> str:
    if not _xpu_available():
        return "XPU not available"
    if ark.xpu_lib is None:
        return "ark.xpu_lib is None (XPU extension failed to import)"
    if not hasattr(ark.xpu_lib, "moe_gemm_decode"):
        return "ark.xpu_lib missing moe_gemm_decode (need ARK_SYCL_TLA=ON)"
    if not hasattr(ark.xpu_lib, "moe_gemm_prefill"):
        return "ark.xpu_lib missing moe_gemm_prefill (need ARK_SYCL_TLA=ON)"
    if not hasattr(ark, "moe"):
        return "ark.moe (unified entry point) not exported by auto_round_kernel"
    return ""


_UNIFIED_SKIP = _unified_skip_reason()


# ---------------------------------------------------------------------------
# Small shapes (one decode-shaped, one prefill-shaped) -- keep wall-clock low.
# ---------------------------------------------------------------------------

_DECODE_SHAPE = dict(num_experts=4, tokens_per_expert=[1, 2, 0, 2], N=128, K=256)
_PREFILL_SHAPE = dict(num_experts=4, tokens_per_expert=[16, 8, 0, 20], N=128, K=256)


def _make_int4_sym(E, N, K, group_size, dtype, total_tokens):
    activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
    w_float = (torch.randn(E, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
    scales = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
    packed = _pack_int4_sym(w_float, scales, group_size)
    return activations, packed, scales, None


def _make_int4_asym(E, N, K, group_size, dtype, total_tokens):
    activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
    w_float = (torch.randn(E, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
    scales = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
    zeros = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
    packed = _pack_int4_asym(w_float, scales, zeros, group_size)
    return activations, packed, scales, zeros


def _make_int8_sym(E, N, K, group_size, dtype, total_tokens):
    activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
    w_float = (torch.randn(E, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
    scales = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
    packed = _pack_int8_sym(w_float, scales, group_size)
    return activations, packed, scales, None


def _make_int8_asym(E, N, K, group_size, dtype, total_tokens):
    activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
    w_float = (torch.randn(E, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
    scales = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
    zeros = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
    packed = _pack_int8_asym(w_float, scales, zeros, group_size)
    return activations, packed, scales, zeros


def _make_int2_sym(E, N, K, group_size, dtype, total_tokens):
    activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
    w_float = (torch.randn(E, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
    scales = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
    packed = _pack_int2_sym(w_float, scales, group_size)
    return activations, packed, scales, None


def _make_int2_asym(E, N, K, group_size, dtype, total_tokens):
    activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
    w_float = (torch.randn(E, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
    scales = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
    zeros = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
    packed = _pack_int2_asym(w_float, scales, zeros, group_size)
    return activations, packed, scales, zeros


def _make_fp8(E, N, K, group_size, dtype, total_tokens, fp8_dtype):
    activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
    w_float = (torch.randn(E, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
    scales = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
    packed = _pack_fp8(w_float, scales, group_size, fp8_dtype)
    return activations, packed, scales, None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(bool(_UNIFIED_SKIP), reason=_UNIFIED_SKIP or "ok")
class TestMoeUnifiedDispatch:
    """Tests for the auto-dispatch logic itself."""

    def test_auto_picks_decode_for_small_tokens_per_expert(self):
        shape = _DECODE_SHAPE
        total_tokens = sum(shape["tokens_per_expert"])
        E, N, K = shape["num_experts"], shape["N"], shape["K"]
        group_size = 128
        dtype = torch.float16

        activations, packed, scales, _ = _make_int4_sym(E, N, K, group_size, dtype, total_tokens)
        ntpe = torch.tensor(shape["tokens_per_expert"], dtype=torch.int32, device="xpu")

        out_auto = ark.moe(
            activations,
            packed,
            ntpe,
            scales=scales,
            weight_bits=4,
            group_size=group_size,
            asym=False,
            phase="auto",
        )
        out_decode = ark.moe_gemm_decode(
            activations,
            packed,
            ntpe,
            scales=scales,
            weight_bits=4,
            group_size=group_size,
            asym=False,
        )
        # max tokens/expert = 2 (<= default threshold 4) -> dispatched to decode
        # -> output must be bit-identical to moe_gemm_decode.
        torch.testing.assert_close(out_auto, out_decode, rtol=0, atol=0)

    def test_auto_picks_prefill_for_large_tokens_per_expert(self):
        shape = _PREFILL_SHAPE
        total_tokens = sum(shape["tokens_per_expert"])
        E, N, K = shape["num_experts"], shape["N"], shape["K"]
        group_size = 128
        dtype = torch.float16

        activations, packed, scales, _ = _make_int4_sym(E, N, K, group_size, dtype, total_tokens)
        ntpe = torch.tensor(shape["tokens_per_expert"], dtype=torch.int32, device="xpu")

        out_auto = ark.moe(
            activations,
            packed,
            ntpe,
            scales=scales,
            weight_bits=4,
            group_size=group_size,
            asym=False,
            phase="auto",
        )
        out_prefill = ark.moe_gemm_prefill(
            activations,
            packed,
            ntpe,
            scales=scales,
            weight_bits=4,
            group_size=group_size,
            asym=False,
        )
        torch.testing.assert_close(out_auto, out_prefill, rtol=0, atol=0)

    def test_decode_threshold_override(self):
        # Same prefill-shaped input but bump the threshold above the max
        # tokens/expert -> auto must now pick decode.
        shape = _PREFILL_SHAPE
        total_tokens = sum(shape["tokens_per_expert"])
        E, N, K = shape["num_experts"], shape["N"], shape["K"]
        group_size = 128
        dtype = torch.float16

        activations, packed, scales, _ = _make_int4_sym(E, N, K, group_size, dtype, total_tokens)
        ntpe = torch.tensor(shape["tokens_per_expert"], dtype=torch.int32, device="xpu")

        max_tpe = max(shape["tokens_per_expert"])
        out_auto = ark.moe(
            activations,
            packed,
            ntpe,
            scales=scales,
            weight_bits=4,
            group_size=group_size,
            asym=False,
            phase="auto",
            decode_threshold=max_tpe + 1,
        )
        out_decode = ark.moe_gemm_decode(
            activations,
            packed,
            ntpe,
            scales=scales,
            weight_bits=4,
            group_size=group_size,
            asym=False,
        )
        torch.testing.assert_close(out_auto, out_decode, rtol=0, atol=0)

    def test_invalid_phase_raises(self):
        shape = _DECODE_SHAPE
        total_tokens = sum(shape["tokens_per_expert"])
        E, N, K = shape["num_experts"], shape["N"], shape["K"]
        group_size = 128
        dtype = torch.float16

        activations, packed, scales, _ = _make_int4_sym(E, N, K, group_size, dtype, total_tokens)
        ntpe = torch.tensor(shape["tokens_per_expert"], dtype=torch.int32, device="xpu")

        with pytest.raises(ValueError, match="phase"):
            ark.moe(
                activations,
                packed,
                ntpe,
                scales=scales,
                weight_bits=4,
                group_size=group_size,
                asym=False,
                phase="not_a_phase",
            )


@pytest.mark.skipif(bool(_UNIFIED_SKIP), reason=_UNIFIED_SKIP or "ok")
class TestMoeUnifiedBitParity:
    """``moe(phase=X)`` must be bit-identical to the dispatched kernel.

    Parametrised across all supported quant schemes; for each, we check both
    the decode-shaped and the prefill-shaped input so that both code paths
    are exercised regardless of which one ``"auto"`` would have picked.
    """

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "shape_name,shape",
        [
            ("decode-shape", _DECODE_SHAPE),
            ("prefill-shape", _PREFILL_SHAPE),
        ],
    )
    def test_fp_unquantized(self, dtype, shape_name, shape):
        E, N, K = shape["num_experts"], shape["N"], shape["K"]
        total_tokens = sum(shape["tokens_per_expert"])
        activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
        weights_NK = (torch.randn(E, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
        ntpe = torch.tensor(shape["tokens_per_expert"], dtype=torch.int32, device="xpu")

        # phase="decode" -> moe_gemm_decode
        out_unified = ark.moe(activations, weights_NK, ntpe, weight_bits=16, phase="decode")
        out_kernel = ark.moe_gemm_decode(activations, weights_NK, ntpe, weight_bits=16)
        torch.testing.assert_close(out_unified, out_kernel, rtol=0, atol=0)

        # phase="prefill" -> moe_gemm_prefill
        out_unified = ark.moe(activations, weights_NK, ntpe, weight_bits=16, phase="prefill")
        out_kernel = ark.moe_gemm_prefill(activations, weights_NK, ntpe, weight_bits=16)
        torch.testing.assert_close(out_unified, out_kernel, rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("asym", [False, True])
    @pytest.mark.parametrize(
        "shape_name,shape",
        [
            ("decode-shape", _DECODE_SHAPE),
            ("prefill-shape", _PREFILL_SHAPE),
        ],
    )
    def test_int4(self, dtype, asym, shape_name, shape):
        E, N, K = shape["num_experts"], shape["N"], shape["K"]
        total_tokens = sum(shape["tokens_per_expert"])
        group_size = 128
        if asym:
            activations, packed, scales, zeros = _make_int4_asym(E, N, K, group_size, dtype, total_tokens)
        else:
            activations, packed, scales, zeros = _make_int4_sym(E, N, K, group_size, dtype, total_tokens)
        ntpe = torch.tensor(shape["tokens_per_expert"], dtype=torch.int32, device="xpu")

        kwargs = dict(scales=scales, zeros=zeros, weight_bits=4, group_size=group_size, asym=asym)

        out_unified = ark.moe(activations, packed, ntpe, phase="decode", **kwargs)
        out_kernel = ark.moe_gemm_decode(activations, packed, ntpe, **kwargs)
        torch.testing.assert_close(out_unified, out_kernel, rtol=0, atol=0)

        out_unified = ark.moe(activations, packed, ntpe, phase="prefill", **kwargs)
        out_kernel = ark.moe_gemm_prefill(activations, packed, ntpe, **kwargs)
        torch.testing.assert_close(out_unified, out_kernel, rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("asym", [False, True])
    def test_int8(self, dtype, asym):
        # Single shape -- the quant path is the same on both shapes, so
        # iterating both would just slow the test suite down.
        shape = _PREFILL_SHAPE
        E, N, K = shape["num_experts"], shape["N"], shape["K"]
        total_tokens = sum(shape["tokens_per_expert"])
        group_size = 128
        if asym:
            activations, packed, scales, zeros = _make_int8_asym(E, N, K, group_size, dtype, total_tokens)
        else:
            activations, packed, scales, zeros = _make_int8_sym(E, N, K, group_size, dtype, total_tokens)
        ntpe = torch.tensor(shape["tokens_per_expert"], dtype=torch.int32, device="xpu")

        kwargs = dict(scales=scales, zeros=zeros, weight_bits=8, group_size=group_size, asym=asym)
        out_unified = ark.moe(activations, packed, ntpe, phase="prefill", **kwargs)
        out_kernel = ark.moe_gemm_prefill(activations, packed, ntpe, **kwargs)
        torch.testing.assert_close(out_unified, out_kernel, rtol=0, atol=0)

        out_unified = ark.moe(activations, packed, ntpe, phase="decode", **kwargs)
        out_kernel = ark.moe_gemm_decode(activations, packed, ntpe, **kwargs)
        torch.testing.assert_close(out_unified, out_kernel, rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("asym", [False, True])
    def test_int2(self, dtype, asym):
        shape = _PREFILL_SHAPE
        E, N, K = shape["num_experts"], shape["N"], shape["K"]
        total_tokens = sum(shape["tokens_per_expert"])
        group_size = 128
        if asym:
            activations, packed, scales, zeros = _make_int2_asym(E, N, K, group_size, dtype, total_tokens)
        else:
            activations, packed, scales, zeros = _make_int2_sym(E, N, K, group_size, dtype, total_tokens)
        ntpe = torch.tensor(shape["tokens_per_expert"], dtype=torch.int32, device="xpu")

        kwargs = dict(scales=scales, zeros=zeros, weight_bits=2, group_size=group_size, asym=asym)
        out_unified = ark.moe(activations, packed, ntpe, phase="prefill", **kwargs)
        out_kernel = ark.moe_gemm_prefill(activations, packed, ntpe, **kwargs)
        torch.testing.assert_close(out_unified, out_kernel, rtol=0, atol=0)

        out_unified = ark.moe(activations, packed, ntpe, phase="decode", **kwargs)
        out_kernel = ark.moe_gemm_decode(activations, packed, ntpe, **kwargs)
        torch.testing.assert_close(out_unified, out_kernel, rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_fp8(self, dtype, fp8_dtype):
        shape = _PREFILL_SHAPE
        E, N, K = shape["num_experts"], shape["N"], shape["K"]
        total_tokens = sum(shape["tokens_per_expert"])
        group_size = 128
        activations, packed, scales, _ = _make_fp8(E, N, K, group_size, dtype, total_tokens, fp8_dtype)
        ntpe = torch.tensor(shape["tokens_per_expert"], dtype=torch.int32, device="xpu")

        kwargs = dict(scales=scales, group_size=group_size, asym=False)
        out_unified = ark.moe(activations, packed, ntpe, phase="prefill", **kwargs)
        out_kernel = ark.moe_gemm_prefill(activations, packed, ntpe, **kwargs)
        torch.testing.assert_close(out_unified, out_kernel, rtol=0, atol=0)

        out_unified = ark.moe(activations, packed, ntpe, phase="decode", **kwargs)
        out_kernel = ark.moe_gemm_decode(activations, packed, ntpe, **kwargs)
        torch.testing.assert_close(out_unified, out_kernel, rtol=0, atol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
