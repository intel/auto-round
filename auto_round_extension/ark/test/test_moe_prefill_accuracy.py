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

import os

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

    The matmuls are executed on CPU (inputs are moved off the accelerator) so
    the reference is independent of the device kernels under test; the result is
    moved back to the original device so it stays comparable to the kernel output.
    """
    orig_device = activations.device
    activations = activations.cpu()
    dequant_weights_NK = dequant_weights_NK.cpu()
    num_tokens_per_expert = num_tokens_per_expert.cpu()
    total_tokens, _ = activations.shape
    E, N, _ = dequant_weights_NK.shape
    # Zero-initialise (not ``torch.empty``): any row not covered by the
    # per-expert loop below -- e.g. a zero-token expert, or a
    # ``num_tokens_per_expert`` sum that is short of ``total_tokens`` -- would
    # otherwise expose stale allocator memory (frequently read back as
    # all-zeros), silently corrupting the reference the kernel is compared to.
    out = torch.zeros(total_tokens, N, dtype=activations.dtype, device="cpu")
    offset = 0
    for e in range(E):
        n_tokens = int(num_tokens_per_expert[e].item())
        if n_tokens == 0:
            continue
        a = activations[offset : offset + n_tokens]
        out[offset : offset + n_tokens] = a @ dequant_weights_NK[e].T
        offset += n_tokens
    return out.to(orig_device)


# ---------------------------------------------------------------------------
# Tolerances per quant scheme
#
# These mirror the tolerances used in ``test_moe.py`` for the small-shape
# parity tests. We loosen slightly for the large-shape cases because longer
# K-reduction accumulates more rounding noise (FP16/BF16 accumulators).
# ---------------------------------------------------------------------------

_TOL_FP = dict(rtol=3e-2, atol=3e-2)
_TOL_INT8 = dict(rtol=1e-1, atol=1e-1)
_TOL_INT4 = dict(rtol=3e-1, atol=3e-1)
_TOL_INT2 = dict(rtol=1.5e-1, atol=1.5e-1)
_TOL_FP8 = dict(rtol=1e-1, atol=1e-1)


def _tol_for_dtype(base, dtype):
    """Return tolerances loosened for bf16 accumulator noise at large K.

    bf16 has 7 mantissa bits vs fp16's 10, so K-reductions of ~14K elements
    (the ``medium E=8`` prefill shape uses K=14336) can produce a handful of
    outliers per ~2M outputs that exceed the fp16-calibrated bound by up to
    ~1.3x (observed: max abs diff ~0.090). This is genuine accumulator-order
    noise, so we widen only for bf16 quant callers and leave fp16 (whose wider
    mantissa absorbs the noise) at the tight, quant-appropriate bound.

    fp16 is deliberately NOT loosened here. A previous change widened fp16 to a
    ``1e-1`` floor to silence an "intermittent" int4-sym + fp16 failure, but the
    observed failure (max abs diff ~70 at a single index) is orders of magnitude
    beyond any accumulator noise and beyond the ``1e-1`` floor itself -- i.e. the
    loosened tolerance never even made the assertion pass. Such a single wildly
    wrong element is a kernel bug (boundary / uninitialized-memory / predication
    in the S4 DPAS prefill path), not RNG luck, and must be fixed in the kernel
    rather than masked here. Keeping fp16 strict ensures that bug stays visible.
    """
    if dtype is torch.bfloat16:
        return dict(rtol=max(base["rtol"], 1e-1), atol=max(base["atol"], 1e-1))
    return base


def _assert_close_report(out, ref, tol, label):
    """``torch.testing.assert_close`` with a localized outlier report on failure.

    When the assertion fails this augments the error with a breakdown that makes
    the failure *attributable* to a root cause instead of just a single "greatest
    absolute difference" number:

      * the greatest abs diff and its index,
      * how many elements exceed a coarse ``1.0`` threshold and their (first few)
        indices -- a *single* / handful of huge outliers points at a kernel
        boundary or uninitialized-memory bug, whereas a large *fraction* of
        elements being off points at a global layout / dequant / scale bug.

    This directly supports triaging the MoE prefill kernels: accumulator-order
    noise is many tiny sub-``0.1`` diffs, so any element past ``1.0`` is a real
    correctness defect regardless of the configured tolerance.
    """
    try:
        torch.testing.assert_close(out, ref, msg=lambda m: f"[{label}] {m}", **tol)
    except AssertionError as err:
        diff = (out.detach().float() - ref.detach().float()).abs()
        total = diff.numel()
        gross_mask = diff > 1.0
        n_gross = int(gross_mask.sum().item())
        max_diff, max_flat = diff.reshape(-1).max(dim=0)
        max_idx = tuple(int(i) for i in torch.unravel_index(max_flat, diff.shape))
        gross_idx = [tuple(int(i) for i in idx) for idx in torch.nonzero(gross_mask)[:8].tolist()] if n_gross else []
        report = (
            f"\n[{label}] outlier report:"
            f"\n  shape={tuple(diff.shape)} total={total}"
            f"\n  max abs diff={float(max_diff):.6g} at index {max_idx}"
            f"\n  elements with |diff|>1.0: {n_gross} ({100.0 * n_gross / total:.4f}%)"
            f"\n  first gross indices: {gross_idx}"
            f"\n  -> a single / handful of huge outliers indicates a kernel"
            f" boundary or uninitialized-memory bug (fix the kernel, not the tol);"
            f"\n     a large fraction indicates a global layout / dequant / scale bug."
        )
        raise AssertionError(str(err) + report) from None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(bool(_PREFILL_SKIP), reason=_PREFILL_SKIP or "ok")
class TestMoEGemmPrefillAccuracy:
    """Parity tests for ``moe_gemm`` / ``moe_gemm_prefill`` at production shapes.

    Each test iterates the production-scale shape matrix and asserts the ark
    kernel output matches a per-expert dequant + ``A @ W.T`` reference.
    """

    @pytest.fixture(autouse=True)
    def _fix_seed(self):
        """Deterministic RNG per test so tolerance assertions are reproducible.

        Without a fixed seed, long-K reductions (K up to 14336) can produce a
        handful of stochastic outliers per ~2M outputs that occasionally cross
        the tolerance bound on some quant paths (notably int4-sym + fp16).
        Pinning the seed makes any failure repeatable so it can be attributed
        to the kernel implementation rather than RNG luck.
        """
        torch.manual_seed(0)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.manual_seed_all(0)

    @pytest.fixture(autouse=True)
    def _isolate_ark_moe_env(self):
        """Give every test a clean ``ARK_MOE_PREFILL_*`` env baseline.

        Several tests (here and in ``test_moe_prefill_perf.py``) opt into an
        experimental dispatch path by exporting an ``ARK_MOE_PREFILL_*`` flag
        -- some of which route through kernels with known numeric defects
        (e.g. the S4 DPAS large-M outliers). If such a flag leaks -- because a
        preceding test aborted before restoring it, or it was exported in the
        shell -- the *default-path* parity tests below silently run the wrong
        kernel and fail. That is the classic "passes in isolation, fails only
        in the full suite" symptom.

        Snapshot every ``ARK_MOE_PREFILL_*`` variable, clear them so each test
        starts from the documented default (all opt-in paths OFF), then
        restore the original process environment afterwards. Tests that need a
        flag set it explicitly within their own body, so clearing the baseline
        never changes their behaviour.
        """
        prefix = "ARK_MOE_PREFILL_"
        saved = {k: v for k, v in os.environ.items() if k.startswith(prefix)}
        for k in saved:
            del os.environ[k]
        try:
            yield
        finally:
            for k in [k for k in os.environ if k.startswith(prefix)]:
                del os.environ[k]
            os.environ.update(saved)

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
            _assert_close_report(out, ref, _TOL_FP, label)

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
            _assert_close_report(out, ref, _tol_for_dtype(_TOL_INT4, dtype), label)

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
            _assert_close_report(out, ref, _tol_for_dtype(_TOL_INT8, dtype), label)

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
            _assert_close_report(out, ref, _tol_for_dtype(_TOL_INT2, dtype), label)

    @pytest.mark.skipif(bool(_QUANT_PREFILL_SKIP), reason=_QUANT_PREFILL_SKIP or "ok")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    @pytest.mark.parametrize(
        "native",
        [
            pytest.param(False, id="dequant"),
            pytest.param(True, id="native"),
        ],
    )
    def test_accuracy_fp8(self, dtype, fp8_dtype, native):
        """Parity of the FP8 prefill kernel against a dequant + ``A @ W.T`` reference.

        Parametrised over the ``ARK_MOE_PREFILL_NATIVE_FP8`` opt-in so both
        the two-stage (dequant -> bf16 GEMM) path and the native fused path
        are covered by the same shape matrix. The two implementations share
        the ``moe_dequant::decode_fp8`` primitive, so tolerances are the
        same; only accumulator-order differences remain.
        """
        group_size = 128
        prev = os.environ.get("ARK_MOE_PREFILL_NATIVE_FP8")
        os.environ["ARK_MOE_PREFILL_NATIVE_FP8"] = "1" if native else "0"
        try:
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
                _assert_close_report(out, ref, _tol_for_dtype(_TOL_FP8, dtype), label)
        finally:
            if prev is None:
                os.environ.pop("ARK_MOE_PREFILL_NATIVE_FP8", None)
            else:
                os.environ["ARK_MOE_PREFILL_NATIVE_FP8"] = prev

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_accuracy_fp8_per_tensor_dpas(self, dtype, fp8_dtype):
        """Parity of the Variant A per-tensor FP8 DPAS entry point.

        Packs each expert's weight with a single per-tensor scale
        (max-abs of the tile) and dispatches through
        ``moe_gemm_prefill(..., scale_scheme="per_tensor")``. The
        weight layout is `[E, K, N]` row-major FP8 -- vllm-xpu-kernels
        convention, distinct from the per-group `[E, N, K]` layout the
        default `moe_gemm_prefill` path uses.

        Skips silently on builds without the `moe_gemm_prefill_fp8_dpas`
        pybind symbol (i.e. builds that were not linked against the
        DPAS kernel).

        STATUS: NEEDS-HARDWARE-VALIDATION.
        """
        if not hasattr(ark.xpu_lib, "moe_gemm_prefill_fp8_dpas"):
            pytest.skip("build lacks moe_gemm_prefill_fp8_dpas (Variant A) symbol")

        # Per-tensor FP8 max-abs quantisation matching the kernel's expected
        # semantics: q = round_to_fp8(w / scale), where scale = amax(w) / fmax.
        fp8_finfo_max = torch.finfo(fp8_dtype).max

        for label, E, tpe, N, K in PREFILL_SHAPES:
            total_tokens = sum(tpe)
            activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
            # Weights in the vllm layout: [E, K, N] row-major.
            w_float = torch.randn(E, K, N, dtype=torch.float32, device="xpu") * 0.1
            # One scalar scale per expert -- max-abs of the tile.
            amax = w_float.reshape(E, -1).abs().amax(dim=1).clamp_min(1e-8)
            scales = (amax / fp8_finfo_max).to(torch.float32)  # [E] fp32
            packed = (w_float / scales.reshape(E, 1, 1)).to(fp8_dtype)
            # Reference dequant: cast fp8 -> fp32 -> apply per-tensor scale ->
            # cast to act dtype, then GEMM as [E, N, K] via a transpose so
            # the shared `_reference_moe_prefill` (which wants [E, N, K]) can
            # consume it uniformly.
            dequant_KN = packed.to(torch.float32) * scales.reshape(E, 1, 1)
            dequant_NK = dequant_KN.transpose(1, 2).contiguous().to(dtype)
            ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")

            out = ark.moe_gemm_prefill(
                activations,
                packed,
                ntpe,
                scales=scales,
                scale_scheme="per_tensor",
            )

            ref = _reference_moe_prefill(activations, dequant_NK, ntpe)
            assert out.shape == (total_tokens, N), f"{label}: bad shape {out.shape}"
            _assert_close_report(out, ref, _tol_for_dtype(_TOL_FP8, dtype), label)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_accuracy_int8_per_tensor_dpas(self, dtype):
        """Parity of the Variant A per-tensor INT8 DPAS entry point.

        Sibling of :func:`test_accuracy_fp8_per_tensor_dpas`. Weights are
        stored as one signed byte per element in ``[E, K, N]`` row-major
        (vllm-xpu-kernels convention modulo dtype); scales are one FP32
        scalar per expert. The kernel upcasts int8 -> activation dtype
        in register before feeding DPAS, so numerics match a plain
        "dequantize to bf16/fp16 then torch.bmm" reference within FP8-
        style tolerances.

        Skips silently on builds without the ``moe_gemm_prefill_int_dpas``
        pybind symbol.

        STATUS: NEEDS-HARDWARE-VALIDATION.
        """
        if not hasattr(ark.xpu_lib, "moe_gemm_prefill_int_dpas"):
            pytest.skip("build lacks moe_gemm_prefill_int_dpas (Variant A) symbol")

        # Sym per-tensor INT8: q = round_to_int8(w / scale), scale = amax(w) / 127.
        int8_max = 127.0

        for label, E, tpe, N, K in PREFILL_SHAPES:
            total_tokens = sum(tpe)
            activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
            # Weights in the vllm layout: [E, K, N] row-major.
            w_float = torch.randn(E, K, N, dtype=torch.float32, device="xpu") * 0.1
            amax = w_float.reshape(E, -1).abs().amax(dim=1).clamp_min(1e-8)
            scales = (amax / int8_max).to(torch.float32)  # [E] fp32
            # Round-half-to-even then clamp; matches the kernel's implicit
            # round-nearest-then-saturate semantics on the int upcast.
            packed = (w_float / scales.reshape(E, 1, 1)).round().clamp(-128, 127).to(torch.int8)
            # Reference dequant: cast int8 -> fp32 -> apply per-tensor scale ->
            # cast to act dtype, then transpose to [E, N, K] for the shared
            # `_reference_moe_prefill` helper.
            dequant_KN = packed.to(torch.float32) * scales.reshape(E, 1, 1)
            dequant_NK = dequant_KN.transpose(1, 2).contiguous().to(dtype)
            ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")

            out = ark.moe_gemm_prefill(
                activations,
                packed,
                ntpe,
                scales=scales,
                scale_scheme="per_tensor",
            )

            ref = _reference_moe_prefill(activations, dequant_NK, ntpe)
            assert out.shape == (total_tokens, N), f"{label}: bad shape {out.shape}"
            _assert_close_report(out, ref, _tol_for_dtype(_TOL_INT8, dtype), label)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_accuracy_fp8_dpas_per_group(self, dtype, fp8_dtype):
        """Parity of the Variant B per-K-group FP8 DPAS branch.

        Uses the standard ``moe_gemm_prefill`` call path with
        ``ARK_MOE_PREFILL_DPAS_FP8=1`` (default). The C++ dispatcher
        should pick the DPAS branch for shapes that satisfy
        ``N%64==0 && K%32==0 && K%group_size==0`` and silently fall
        back to the native/dequant paths otherwise -- so this test is
        checking parity, not that the DPAS branch is exercised.

        STATUS: NEEDS-HARDWARE-VALIDATION.
        """
        group_size = 128
        prev = os.environ.get("ARK_MOE_PREFILL_DPAS_FP8")
        os.environ["ARK_MOE_PREFILL_DPAS_FP8"] = "1"
        try:
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
                _assert_close_report(out, ref, _tol_for_dtype(_TOL_FP8, dtype), label)
        finally:
            if prev is None:
                os.environ.pop("ARK_MOE_PREFILL_DPAS_FP8", None)
            else:
                os.environ["ARK_MOE_PREFILL_DPAS_FP8"] = prev

    @pytest.mark.skipif(bool(_QUANT_PREFILL_SKIP), reason=_QUANT_PREFILL_SKIP or "ok")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_accuracy_int8_dpas_per_group(self, dtype):
        """Parity of the Variant B per-K-group INT8 DPAS branch.

        Sibling of :meth:`test_accuracy_fp8_dpas_per_group` -- uses the
        standard ``moe_gemm_prefill`` call path with
        ``ARK_MOE_PREFILL_DPAS_INT8=1`` (default). The C++ dispatcher
        should pick the INT8 DPAS branch for shapes that satisfy
        ``N%64==0 && K%32==0 && K%group_size==0 && group_size in
        {32,64,128,256}`` and silently fall back to the dequant path
        otherwise, so this test is checking parity, not that the DPAS
        branch is exercised.

        STATUS: NEEDS-HARDWARE-VALIDATION.
        """
        group_size = 128
        prev = os.environ.get("ARK_MOE_PREFILL_DPAS_INT8")
        os.environ["ARK_MOE_PREFILL_DPAS_INT8"] = "1"
        try:
            for label, E, tpe, N, K in PREFILL_SHAPES:
                if K % group_size != 0:
                    continue
                total_tokens = sum(tpe)
                activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
                w_float = (torch.randn(E, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
                scales = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
                packed = _pack_int8_sym(w_float, scales, group_size)
                dequant = _dequant_int8_sym(packed, scales, group_size).to(dtype)
                ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")

                out = ark.moe_gemm_prefill(
                    activations,
                    packed,
                    ntpe,
                    scales=scales,
                    weight_bits=8,
                    group_size=group_size,
                    asym=False,
                )

                ref = _reference_moe_prefill(activations, dequant, ntpe)
                assert out.shape == (total_tokens, N), f"{label}: bad shape {out.shape}"
                _assert_close_report(out, ref, _tol_for_dtype(_TOL_INT8, dtype), label)
        finally:
            if prev is None:
                os.environ.pop("ARK_MOE_PREFILL_DPAS_INT8", None)
            else:
                os.environ["ARK_MOE_PREFILL_DPAS_INT8"] = prev

    @pytest.mark.skipif(bool(_QUANT_PREFILL_SKIP), reason=_QUANT_PREFILL_SKIP or "ok")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_accuracy_int4_dpas_per_group(self, dtype):
        """Parity of the S4-sym single-pass DPAS mixed-input mainloop.

        Sibling of :meth:`test_accuracy_int8_dpas_per_group` -- uses the
        standard ``moe_gemm_prefill`` call path with
        ``ARK_MOE_PREFILL_DPAS_S4=1`` (explicitly opting in; the path is
        default OFF) and ``ARK_MOE_PREFILL_DPAS_INT8=0`` so the two-pass
        S4->S8 upcast fallback is disabled and we exercise the single-pass
        mainloop exclusively. The C++ dispatcher should pick the S4 DPAS
        branch for shapes that satisfy ``N%64==0 && K%32==0 && K%group_size==0
        && group_size%2==0 && group_size in {32,64,128,256}`` and
        silently fall back to the dequant path otherwise, so this test
        is checking parity, not that the DPAS branch is exercised.

        STATUS: NEEDS-HARDWARE-VALIDATION.
        """
        group_size = 128
        prev_s4 = os.environ.get("ARK_MOE_PREFILL_DPAS_S4")
        prev_int8 = os.environ.get("ARK_MOE_PREFILL_DPAS_INT8")
        os.environ["ARK_MOE_PREFILL_DPAS_S4"] = "1"
        os.environ["ARK_MOE_PREFILL_DPAS_INT8"] = "0"
        try:
            for label, E, tpe, N, K in PREFILL_SHAPES:
                if K % group_size != 0:
                    continue
                total_tokens = sum(tpe)
                activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
                w_float = (torch.randn(E, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
                scales = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
                packed = _pack_int4_sym(w_float, scales, group_size)
                dequant = _dequant_int4_sym(packed, scales, group_size).to(dtype)
                ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")

                out = ark.moe_gemm_prefill(
                    activations,
                    packed,
                    ntpe,
                    scales=scales,
                    weight_bits=4,
                    group_size=group_size,
                    asym=False,
                )

                ref = _reference_moe_prefill(activations, dequant, ntpe)
                assert out.shape == (total_tokens, N), f"{label}: bad shape {out.shape}"
                _assert_close_report(out, ref, _tol_for_dtype(_TOL_INT4, dtype), label)
        finally:
            if prev_s4 is None:
                os.environ.pop("ARK_MOE_PREFILL_DPAS_S4", None)
            else:
                os.environ["ARK_MOE_PREFILL_DPAS_S4"] = prev_s4
            if prev_int8 is None:
                os.environ.pop("ARK_MOE_PREFILL_DPAS_INT8", None)
            else:
                os.environ["ARK_MOE_PREFILL_DPAS_INT8"] = prev_int8


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
