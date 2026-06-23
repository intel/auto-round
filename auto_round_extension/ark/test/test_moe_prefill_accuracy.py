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

"""Accuracy (parity) tests for ``ark.moe_gemm`` / ``ark.moe_gemm_prefill``.

This complements ``test_moe_prefill_perf.py`` (which measures throughput) and
``test_moe.py`` (which exercises small toy shapes). The shape matrix here
mirrors the *production-scale* shapes used by the perf benchmark — large
hidden sizes (up to 14336), high expert counts (up to 64), and uneven
token-per-expert distributions typical of Mixtral/DeepSeek-style MoE models
during prefill.

For each (dtype, quant-scheme) combination the test:

  1. Packs / quantizes random weights.
  2. Dequantizes them back to the activation dtype.
  3. Runs the ark kernel on the *packed* weights.
  4. Runs a per-expert ``A @ W.T`` reference on the *dequantized* weights.
  5. Compares with ``torch.testing.assert_close`` at quant-appropriate
     tolerances.

This isolates kernel correctness (matmul + on-the-fly dequant) from
quantization noise: the reference shares the same dequantized weights as the
kernel, so tolerances reflect only accumulator/order-of-operations
differences, not quant error.

How to run::

    pytest -v -s auto_round_extension/ark/test/test_moe_prefill_accuracy.py
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
# Skip reasons (mirror test_moe_prefill_perf.py)
# ---------------------------------------------------------------------------


def _xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _xpu_skip_reason() -> str:
    if not hasattr(torch, "xpu"):
        return "torch has no xpu submodule (need an Intel XPU build of torch)"
    if not torch.xpu.is_available():
        return "torch.xpu.is_available() == False (no XPU device or driver visible)"
    return ""


def _prefill_skip_reason() -> str:
    reason = _xpu_skip_reason()
    if reason:
        return reason
    if ark.xpu_lib is None:
        return (
            "ark.xpu_lib is None -- the XPU extension module "
            "(auto_round_kernel_xpu) failed to import; check that auto_round_kernel "
            "was installed for THIS Python env with XPU support enabled"
        )
    if not hasattr(ark.xpu_lib, "moe_gemm"):
        return (
            "ark.xpu_lib loaded but has no moe_gemm symbol -- "
            "rebuild with ARK_SYCL_TLA=ON to compile the MoE GEMM kernel"
        )
    return ""


def _quantized_prefill_skip_reason() -> str:
    reason = _prefill_skip_reason()
    if reason:
        return reason
    if not hasattr(ark.xpu_lib, "moe_gemm_prefill"):
        return (
            "ark.xpu_lib loaded but has no moe_gemm_prefill symbol -- "
            "rebuild with ARK_SYCL_TLA=ON to compile the quantized MoE prefill kernel"
        )
    return ""


_PREFILL_SKIP = _prefill_skip_reason()
_QUANT_PREFILL_SKIP = _quantized_prefill_skip_reason()


# ---------------------------------------------------------------------------
# Shape matrix
#
# Subset of the perf benchmark's PREFILL_SHAPES. We keep the production-scale
# hidden sizes (up to N/K = 14336) and high expert counts (up to E = 64) so
# the accuracy check exercises the same code paths as the perf benchmark,
# but skip duplicate up-proj/down-proj rows to keep wall-clock reasonable.
# ---------------------------------------------------------------------------

PREFILL_SHAPES = [
    # (label, num_experts, tokens_per_expert_list, N, K)
    ("small  E=8 ", 8, [32, 28, 30, 35, 33, 31, 29, 34], 4096, 4096),
    ("medium E=8 ", 8, [64, 60, 68, 72, 65, 63, 70, 66], 4096, 14336),
    ("large  E=16", 16, [16] * 16, 2048, 2048),
    ("large  E=32", 32, [8] * 32, 2048, 2048),
    ("large  E=64", 64, [4] * 64, 2048, 2048),
    ("uneven E=8 ", 8, [100, 50, 75, 80, 60, 90, 70, 85], 4096, 4096),
]


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def _reference_moe_prefill(activations, dequant_weights_NK, num_tokens_per_expert):
    """Per-expert ``A @ W.T`` over ``[E, N, K]`` dequantized weights.

    This is the same reference used by ``TestMoEGemmPrefill`` in ``test_moe.py``
    and matches what a model would compute when no fused kernel is available.
    """
    total_tokens, _ = activations.shape
    E, N, _ = dequant_weights_NK.shape
    out = torch.empty(total_tokens, N, dtype=activations.dtype, device=activations.device)
    offset = 0
    for e in range(E):
        n_tokens = int(num_tokens_per_expert[e].item())
        if n_tokens == 0:
            continue
        a = activations[offset : offset + n_tokens]
        out[offset : offset + n_tokens] = a @ dequant_weights_NK[e].T
        offset += n_tokens
    return out


# ---------------------------------------------------------------------------
# Tolerances per quant scheme
#
# These mirror the tolerances used in ``test_moe.py`` for the small-shape
# parity tests. We loosen slightly for the large-shape cases because longer
# K-reduction accumulates more rounding noise (FP16/BF16 accumulators).
# ---------------------------------------------------------------------------

_TOL_FP = dict(rtol=3e-2, atol=3e-2)
_TOL_INT8 = dict(rtol=7e-2, atol=7e-2)
_TOL_INT4 = dict(rtol=7e-2, atol=7e-2)
_TOL_INT2 = dict(rtol=1.5e-1, atol=1.5e-1)
_TOL_FP8 = dict(rtol=7e-2, atol=7e-2)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(bool(_PREFILL_SKIP), reason=_PREFILL_SKIP or "ok")
class TestMoEGemmPrefillAccuracy:
    """Parity tests for ``moe_gemm`` / ``moe_gemm_prefill`` at production shapes.

    Each test iterates the production-scale shape matrix and asserts the ark
    kernel output matches a per-expert dequant + ``A @ W.T`` reference.
    """

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_accuracy_fp(self, dtype):
        for label, E, tpe, N, K in PREFILL_SHAPES:
            total_tokens = sum(tpe)
            activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
            # ark.moe_gemm wants weights in [E, K, N] layout.
            weights_KN = (torch.randn(E, K, N, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
            ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")

            out = ark.moe_gemm(activations, weights_KN, ntpe)

            # Reference uses [E, N, K] layout.
            weights_NK = weights_KN.transpose(1, 2).contiguous()
            ref = _reference_moe_prefill(activations, weights_NK, ntpe)

            assert out.shape == (total_tokens, N), f"{label}: bad shape {out.shape}"
            assert out.dtype == dtype, f"{label}: bad dtype {out.dtype}"
            torch.testing.assert_close(out, ref, msg=lambda m, lbl=label: f"[{lbl}] {m}", **_TOL_FP)

    @pytest.mark.skipif(bool(_QUANT_PREFILL_SKIP), reason=_QUANT_PREFILL_SKIP or "ok")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("asym", [False, True])
    def test_accuracy_int4(self, dtype, asym):
        group_size = 128
        for label, E, tpe, N, K in PREFILL_SHAPES:
            if K % group_size != 0:
                continue
            total_tokens = sum(tpe)
            activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
            w_float = (torch.randn(E, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
            scales = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
            if asym:
                zeros = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
                packed = _pack_int4_asym(w_float, scales, zeros, group_size)
                dequant = _dequant_int4_asym(packed, scales, zeros, group_size).to(dtype)
            else:
                zeros = None
                packed = _pack_int4_sym(w_float, scales, group_size)
                dequant = _dequant_int4_sym(packed, scales, group_size).to(dtype)
            ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")

            out = ark.moe_gemm_prefill(
                activations,
                packed,
                ntpe,
                scales=scales,
                zeros=zeros,
                weight_bits=4,
                group_size=group_size,
                asym=asym,
            )

            ref = _reference_moe_prefill(activations, dequant, ntpe)
            assert out.shape == (total_tokens, N), f"{label}: bad shape {out.shape}"
            torch.testing.assert_close(out, ref, msg=lambda m, lbl=label: f"[{lbl}] {m}", **_TOL_INT4)

    @pytest.mark.skipif(bool(_QUANT_PREFILL_SKIP), reason=_QUANT_PREFILL_SKIP or "ok")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("asym", [False, True])
    def test_accuracy_int8(self, dtype, asym):
        group_size = 128
        for label, E, tpe, N, K in PREFILL_SHAPES:
            if K % group_size != 0:
                continue
            total_tokens = sum(tpe)
            activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
            w_float = (torch.randn(E, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
            scales = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
            if asym:
                zeros = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
                packed = _pack_int8_asym(w_float, scales, zeros, group_size)
                dequant = _dequant_int8_asym(packed, scales, zeros, group_size).to(dtype)
            else:
                zeros = None
                packed = _pack_int8_sym(w_float, scales, group_size)
                dequant = _dequant_int8_sym(packed, scales, group_size).to(dtype)
            ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")

            out = ark.moe_gemm_prefill(
                activations,
                packed,
                ntpe,
                scales=scales,
                zeros=zeros,
                weight_bits=8,
                group_size=group_size,
                asym=asym,
            )

            ref = _reference_moe_prefill(activations, dequant, ntpe)
            assert out.shape == (total_tokens, N), f"{label}: bad shape {out.shape}"
            torch.testing.assert_close(out, ref, msg=lambda m, lbl=label: f"[{lbl}] {m}", **_TOL_INT8)

    @pytest.mark.skipif(bool(_QUANT_PREFILL_SKIP), reason=_QUANT_PREFILL_SKIP or "ok")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("asym", [False, True])
    def test_accuracy_int2(self, dtype, asym):
        group_size = 128
        for label, E, tpe, N, K in PREFILL_SHAPES:
            if K % group_size != 0 or K % 4 != 0:
                continue
            total_tokens = sum(tpe)
            activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
            w_float = (torch.randn(E, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
            scales = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
            if asym:
                zeros = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
                packed = _pack_int2_asym(w_float, scales, zeros, group_size)
                dequant = _dequant_int2_asym(packed, scales, zeros, group_size).to(dtype)
            else:
                zeros = None
                packed = _pack_int2_sym(w_float, scales, group_size)
                dequant = _dequant_int2_sym(packed, scales, group_size).to(dtype)
            ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")

            out = ark.moe_gemm_prefill(
                activations,
                packed,
                ntpe,
                scales=scales,
                zeros=zeros,
                weight_bits=2,
                group_size=group_size,
                asym=asym,
            )

            ref = _reference_moe_prefill(activations, dequant, ntpe)
            assert out.shape == (total_tokens, N), f"{label}: bad shape {out.shape}"
            torch.testing.assert_close(out, ref, msg=lambda m, lbl=label: f"[{lbl}] {m}", **_TOL_INT2)

    @pytest.mark.skipif(bool(_QUANT_PREFILL_SKIP), reason=_QUANT_PREFILL_SKIP or "ok")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_accuracy_fp8(self, dtype, fp8_dtype):
        group_size = 128
        for label, E, tpe, N, K in PREFILL_SHAPES:
            if K % group_size != 0:
                continue
            total_tokens = sum(tpe)
            activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
            w_float = (torch.randn(E, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
            scales = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
            packed = _pack_fp8(w_float, scales, group_size, fp8_dtype)
            dequant = _dequant_fp8(packed, scales, group_size, dtype)
            ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")

            out = ark.moe_gemm_prefill(
                activations,
                packed,
                ntpe,
                scales=scales,
                group_size=group_size,
                asym=False,
            )

            ref = _reference_moe_prefill(activations, dequant, ntpe)
            assert out.shape == (total_tokens, N), f"{label}: bad shape {out.shape}"
            torch.testing.assert_close(out, ref, msg=lambda m, lbl=label: f"[{lbl}] {m}", **_TOL_FP8)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
