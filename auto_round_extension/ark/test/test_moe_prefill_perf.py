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

"""Performance benchmark: ``ark.moe_gemm`` for MoE prefill workloads.

MoE prefill is the matrix-matrix multiplication phase where many tokens (e.g.,
the entire prompt or a batch of sequences) are routed to different experts.
Unlike decode (one token per expert), prefill has multiple tokens per expert,
making it a batched GEMM problem.

This benchmark measures throughput (TFLOPS) and compares against a baseline
PyTorch implementation. The baseline uses per-expert matrix multiplication
with already-dequantized weights (for quantized tests), representing the
standard fallback path when no fused MoE kernel is available.

How to run::

    pytest -v -s auto_round_extension/ark/test/test_moe_prefill_perf.py

The ``-s`` flag is required to see the printed timing tables and TFLOPS.
"""

import auto_round_kernel
import pytest
import torch

# Reuse pack/dequant helpers from the correctness tests
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


def _xpu_skip_reason() -> str:
    if not hasattr(torch, "xpu"):
        return "torch has no xpu submodule (need an Intel XPU build of torch)"
    if not torch.xpu.is_available():
        return "torch.xpu.is_available() == False (no XPU device or driver visible)"
    return ""


def _prefill_skip_reason() -> str:
    """Return non-empty string if the MoE prefill kernel can't be exercised."""
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
    """Return non-empty string if the quantized MoE prefill kernel can't be exercised."""
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

# Surface diagnostics on collection
print(
    "[moe-prefill-perf] xpu_available=%s  xpu_lib=%s  has_moe_gemm=%s"
    % (
        _xpu_available(),
        "loaded" if ark.xpu_lib is not None else "None",
        hasattr(ark.xpu_lib, "moe_gemm") if ark.xpu_lib is not None else False,
    )
)
if _PREFILL_SKIP:
    print("[moe-prefill-perf] suite will SKIP. reason: %s" % _PREFILL_SKIP)


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

WARMUP = 5
ITERS = 30


def _xpu_time_ms(fn, warmup: int = WARMUP, iters: int = ITERS) -> float:
    """Time ``fn`` on XPU using device events; returns median ms per call."""
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()

    timings = []
    for _ in range(iters):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    timings.sort()
    return timings[len(timings) // 2]


def _default_moe_prefill(activations, dequant_weights, num_tokens_per_expert):
    """Default XPU MoE prefill baseline: per-expert torch matmul.

    This mirrors the path a model would take when no fused MoE kernel is
    available: iterate over experts and do ``A @ W.T`` on each token slice.
    For prefill, each expert may have many tokens (unlike decode where each
    expert typically has one token).
    """
    total_tokens, K = activations.shape
    E, N, _ = dequant_weights.shape
    out = torch.empty(total_tokens, N, dtype=activations.dtype, device=activations.device)
    offset = 0
    for e in range(E):
        n_tokens = int(num_tokens_per_expert[e].item())
        if n_tokens == 0:
            continue
        a = activations[offset : offset + n_tokens]
        # Weights are [N, K], activations are [n_tokens, K]
        # Output is [n_tokens, N]
        out[offset : offset + n_tokens] = a @ dequant_weights[e].T
        offset += n_tokens
    return out


# ---------------------------------------------------------------------------
# TFLOPS calculation
# ---------------------------------------------------------------------------


def _compute_moe_flops(total_tokens, K, N, num_experts_active):
    """Compute FLOPs for MoE GEMM: sum over active experts of (tokens * K * N * 2).

    For a typical prefill, all experts may be active with varying token counts.
    As a simplification, we use the total_tokens across all experts.
    """
    # Each GEMM is [tokens, K] @ [K, N] -> [tokens, N]
    # FLOPs = tokens * K * N * 2 (multiply-add counts as 2 ops)
    return total_tokens * K * N * 2


# ---------------------------------------------------------------------------
# Shape matrix for prefill
#
# Shapes follow MiniMax-Text-01 / MiniMax-M1 MoE config:
#   hidden_size       = 6144   (K for up/gate-proj, N for down-proj)
#   intermediate_size = 9216   (N for up/gate-proj, K for down-proj)
#   num_local_experts = 32
#   num_experts_per_tok = 2    (top-2 routing)
#
# Total expert-token count per row = seq_len * top_k. Rows are labelled by
# the originating sequence length (1K/2K/4K). Tokens are distributed
# evenly across the 32 experts, except for the "uneven" rows which keep a
# skewed distribution to exercise load imbalance.
# ---------------------------------------------------------------------------


def _minimax_uneven_tpe(total: int) -> list[int]:
    """Return a skewed token-per-expert list of length 32 summing to ``total``.

    Mimics the load imbalance commonly observed with top-2 routing on 32
    experts: a handful of experts get ~2x the mean, a handful get ~0.5x,
    the rest stay near the mean.
    """
    E = 32
    mean = total // E
    tpe = [mean] * E
    # Skew: indices 0..5 are "hot", 6..11 are "cold", rest are mean.
    bumps = [(0, +mean), (1, +mean), (2, +mean // 2), (3, +mean // 2), (4, +mean // 2), (5, +mean // 2)]
    drops = [(6, -mean), (7, -mean), (8, -mean // 2), (9, -mean // 2), (10, -mean // 2), (11, -mean // 2)]
    for i, d in bumps + drops:
        tpe[i] += d
    # Fix any rounding drift so the list sums exactly to ``total``.
    diff = total - sum(tpe)
    tpe[-1] += diff
    return tpe


# top-2 routing means total expert tokens = seq_len * 2.
_MINIMAX_E = 32
_MINIMAX_N = 9216  # intermediate_size (gate/up output)
_MINIMAX_K = 6144  # hidden_size

PREFILL_SHAPES = [
    # (label, num_experts, tokens_per_expert_list, N, K)
    # -- seq_len = 1K -> 2048 expert tokens, ~64/expert ----------------------
    ("minimax up  1K", _MINIMAX_E, [64] * _MINIMAX_E, _MINIMAX_N, _MINIMAX_K),
    ("minimax down 1K", _MINIMAX_E, [64] * _MINIMAX_E, _MINIMAX_K, _MINIMAX_N),
    # -- seq_len = 2K -> 4096 expert tokens, ~128/expert ---------------------
    ("minimax up  2K", _MINIMAX_E, [128] * _MINIMAX_E, _MINIMAX_N, _MINIMAX_K),
    ("minimax down 2K", _MINIMAX_E, [128] * _MINIMAX_E, _MINIMAX_K, _MINIMAX_N),
    ("minimax skew up  2K", _MINIMAX_E, _minimax_uneven_tpe(4096), _MINIMAX_N, _MINIMAX_K),
    ("minimax skew down 2K", _MINIMAX_E, _minimax_uneven_tpe(4096), _MINIMAX_K, _MINIMAX_N),
    # -- seq_len = 4K -> 8192 expert tokens, ~256/expert ---------------------
    ("minimax up  4K", _MINIMAX_E, [256] * _MINIMAX_E, _MINIMAX_N, _MINIMAX_K),
    ("minimax down 4K", _MINIMAX_E, [256] * _MINIMAX_E, _MINIMAX_K, _MINIMAX_N),
    ("minimax skew up  4K", _MINIMAX_E, _minimax_uneven_tpe(8192), _MINIMAX_N, _MINIMAX_K),
    ("minimax skew down 4K", _MINIMAX_E, _minimax_uneven_tpe(8192), _MINIMAX_K, _MINIMAX_N),
]


def _print_header(title: str, *, with_dequant_baseline: bool = False) -> None:
    """Print a benchmark header.

    When ``with_dequant_baseline`` is True, an extra ``base+deq(ms)`` column is
    printed alongside the matmul-only baseline and ``speedup`` is reported
    against the dequant-inclusive baseline (this is the apples-to-apples
    comparison for the current Stage-1 ``moe_gemm_prefill`` which dequants
    into a workspace before dispatching to the FP GEMM).
    """
    print()
    width = 130 if with_dequant_baseline else 110
    print("=" * width)
    print(title)
    if with_dequant_baseline:
        print(
            f"{'shape':<14}{'E':>4}{'N':>7}{'K':>7}{'tokens':>8}"
            f"{'base mm(ms)':>14}{'base+deq(ms)':>16}{'ark(ms)':>12}"
            f"{'speedup':>12}{'TFLOPS':>10}"
        )
    else:
        print(
            f"{'shape':<14}{'E':>4}{'N':>7}{'K':>7}{'tokens':>8}"
            f"{'baseline(ms)':>16}{'ark(ms)':>14}{'speedup':>12}{'TFLOPS':>10}"
        )
    print("-" * width)


def _print_row(label, E, N, K, total_tokens, base_ms, ark_ms, tflops, *, base_with_deq_ms=None):
    """Print a benchmark row.

    If ``base_with_deq_ms`` is provided, the dequant-inclusive baseline is
    shown and ``speedup`` is computed against it (apples-to-apples with the
    Stage-1 quantized path). Otherwise the row reverts to the original
    matmul-only layout.
    """
    if base_with_deq_ms is None:
        speedup = base_ms / ark_ms if ark_ms > 0 else float("nan")
        print(
            f"{label:<14}{E:>4}{N:>7}{K:>7}{total_tokens:>8}"
            f"{base_ms:>16.4f}{ark_ms:>14.4f}{speedup:>11.2f}x{tflops:>9.1f}"
        )
    else:
        speedup = base_with_deq_ms / ark_ms if ark_ms > 0 else float("nan")
        print(
            f"{label:<14}{E:>4}{N:>7}{K:>7}{total_tokens:>8}"
            f"{base_ms:>14.4f}{base_with_deq_ms:>16.4f}{ark_ms:>12.4f}"
            f"{speedup:>11.2f}x{tflops:>9.1f}"
        )


# ---------------------------------------------------------------------------
# Benchmark cases
# ---------------------------------------------------------------------------


@pytest.mark.skipif(bool(_PREFILL_SKIP), reason=_PREFILL_SKIP or "ok")
class TestMoEGemmPrefillPerf:
    """Median XPU-event timings of ``moe_gemm`` vs per-expert matrix multiply.

    The baseline uses *already-dequantized* weights for quantized tests, so
    the timed region only measures matmul cost. This is the most favorable
    apples-to-apples comparison for the baseline.
    """

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_perf_fp(self, dtype):
        _print_header(f"FP weights ({str(dtype).split('.')[-1]})  -- ark.moe_gemm (prefill) vs per-expert A @ W.T")
        for label, E, tpe, N, K in PREFILL_SHAPES:
            total_tokens = sum(tpe)
            activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
            # Weights layout: [E, K, N] for moe_gemm
            weights = (torch.randn(E, K, N, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
            ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")

            # Baseline: per-expert matmul. Weights need to be [E, N, K] for the baseline.
            weights_baseline = weights.transpose(1, 2)  # [E, N, K]

            base_ms = _xpu_time_ms(lambda: _default_moe_prefill(activations, weights_baseline, ntpe))
            ark_ms = _xpu_time_ms(lambda: ark.moe_gemm(activations, weights, ntpe))

            # Compute TFLOPS
            flops = _compute_moe_flops(total_tokens, K, N, E)
            tflops = flops / (ark_ms * 1e-3) / 1e12

            _print_row(label, E, N, K, total_tokens, base_ms, ark_ms, tflops)

    @pytest.mark.skipif(bool(_QUANT_PREFILL_SKIP), reason=_QUANT_PREFILL_SKIP or "ok")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("asym", [False, True])
    def test_perf_int4(self, dtype, asym):
        group_size = 128
        kind = "asym" if asym else "sym"
        _print_header(
            f"INT4 {kind} (group_size={group_size}, act={str(dtype).split('.')[-1]}) "
            f"-- ark.moe_gemm_prefill (prefill) vs dequant + per-expert A @ W.T",
            with_dequant_baseline=True,
        )
        for label, E, tpe, N, K in PREFILL_SHAPES:
            if K % group_size != 0:
                continue
            total_tokens = sum(tpe)
            activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
            # Pack helpers expect weights in [E, N, K] layout.
            w_float = (torch.randn(E, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
            scales = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
            if asym:
                zeros = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
                packed = _pack_int4_asym(w_float, scales, zeros, group_size)
                dequant = _dequant_int4_asym(packed, scales, zeros, group_size).to(dtype)

                def _baseline_with_dequant():
                    d = _dequant_int4_asym(packed, scales, zeros, group_size).to(dtype)
                    return _default_moe_prefill(activations, d, ntpe)
            else:
                zeros = None
                packed = _pack_int4_sym(w_float, scales, group_size)
                dequant = _dequant_int4_sym(packed, scales, group_size).to(dtype)

                def _baseline_with_dequant():
                    d = _dequant_int4_sym(packed, scales, group_size).to(dtype)
                    return _default_moe_prefill(activations, d, ntpe)

            ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")

            # ``dequant`` is already [E, N, K] -- matches the baseline contract.
            base_ms = _xpu_time_ms(lambda: _default_moe_prefill(activations, dequant, ntpe))
            base_deq_ms = _xpu_time_ms(_baseline_with_dequant)
            ark_ms = _xpu_time_ms(
                lambda: ark.moe_gemm_prefill(
                    activations,
                    packed,
                    ntpe,
                    scales=scales,
                    zeros=zeros,
                    weight_bits=4,
                    group_size=group_size,
                    asym=asym,
                )
            )

            flops = _compute_moe_flops(total_tokens, K, N, E)
            tflops = flops / (ark_ms * 1e-3) / 1e12

            _print_row(label, E, N, K, total_tokens, base_ms, ark_ms, tflops, base_with_deq_ms=base_deq_ms)

    @pytest.mark.skipif(bool(_QUANT_PREFILL_SKIP), reason=_QUANT_PREFILL_SKIP or "ok")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("asym", [False, True])
    def test_perf_int8(self, dtype, asym):
        group_size = 128
        kind = "asym" if asym else "sym"
        _print_header(
            f"INT8 {kind} (group_size={group_size}, act={str(dtype).split('.')[-1]}) "
            f"-- ark.moe_gemm_prefill (prefill) vs dequant + per-expert A @ W.T",
            with_dequant_baseline=True,
        )
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

                def _baseline_with_dequant():
                    d = _dequant_int8_asym(packed, scales, zeros, group_size).to(dtype)
                    return _default_moe_prefill(activations, d, ntpe)
            else:
                zeros = None
                packed = _pack_int8_sym(w_float, scales, group_size)
                dequant = _dequant_int8_sym(packed, scales, group_size).to(dtype)

                def _baseline_with_dequant():
                    d = _dequant_int8_sym(packed, scales, group_size).to(dtype)
                    return _default_moe_prefill(activations, d, ntpe)

            ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")

            base_ms = _xpu_time_ms(lambda: _default_moe_prefill(activations, dequant, ntpe))
            base_deq_ms = _xpu_time_ms(_baseline_with_dequant)
            ark_ms = _xpu_time_ms(
                lambda: ark.moe_gemm_prefill(
                    activations,
                    packed,
                    ntpe,
                    scales=scales,
                    zeros=zeros,
                    weight_bits=8,
                    group_size=group_size,
                    asym=asym,
                )
            )

            flops = _compute_moe_flops(total_tokens, K, N, E)
            tflops = flops / (ark_ms * 1e-3) / 1e12

            _print_row(label, E, N, K, total_tokens, base_ms, ark_ms, tflops, base_with_deq_ms=base_deq_ms)

    @pytest.mark.skipif(bool(_QUANT_PREFILL_SKIP), reason=_QUANT_PREFILL_SKIP or "ok")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("asym", [False, True])
    def test_perf_int2(self, dtype, asym):
        group_size = 128
        kind = "asym" if asym else "sym"
        _print_header(
            f"INT2 {kind} (group_size={group_size}, act={str(dtype).split('.')[-1]}) "
            f"-- ark.moe_gemm_prefill (prefill) vs dequant + per-expert A @ W.T",
            with_dequant_baseline=True,
        )
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

                def _baseline_with_dequant():
                    d = _dequant_int2_asym(packed, scales, zeros, group_size).to(dtype)
                    return _default_moe_prefill(activations, d, ntpe)
            else:
                zeros = None
                packed = _pack_int2_sym(w_float, scales, group_size)
                dequant = _dequant_int2_sym(packed, scales, group_size).to(dtype)

                def _baseline_with_dequant():
                    d = _dequant_int2_sym(packed, scales, group_size).to(dtype)
                    return _default_moe_prefill(activations, d, ntpe)

            ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")

            base_ms = _xpu_time_ms(lambda: _default_moe_prefill(activations, dequant, ntpe))
            base_deq_ms = _xpu_time_ms(_baseline_with_dequant)
            ark_ms = _xpu_time_ms(
                lambda: ark.moe_gemm_prefill(
                    activations,
                    packed,
                    ntpe,
                    scales=scales,
                    zeros=zeros,
                    weight_bits=2,
                    group_size=group_size,
                    asym=asym,
                )
            )

            flops = _compute_moe_flops(total_tokens, K, N, E)
            tflops = flops / (ark_ms * 1e-3) / 1e12

            _print_row(label, E, N, K, total_tokens, base_ms, ark_ms, tflops, base_with_deq_ms=base_deq_ms)

    @pytest.mark.skipif(bool(_QUANT_PREFILL_SKIP), reason=_QUANT_PREFILL_SKIP or "ok")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_perf_fp8(self, dtype, fp8_dtype):
        group_size = 128
        _print_header(
            f"FP8 {str(fp8_dtype).split('.')[-1]} (group_size={group_size}, "
            f"act={str(dtype).split('.')[-1]}) -- ark.moe_gemm_prefill (prefill) vs dequant + per-expert A @ W.T",
            with_dequant_baseline=True,
        )
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

            def _baseline_with_dequant():
                d = _dequant_fp8(packed, scales, group_size, dtype)
                return _default_moe_prefill(activations, d, ntpe)

            base_ms = _xpu_time_ms(lambda: _default_moe_prefill(activations, dequant, ntpe))
            base_deq_ms = _xpu_time_ms(_baseline_with_dequant)
            ark_ms = _xpu_time_ms(
                lambda: ark.moe_gemm_prefill(
                    activations,
                    packed,
                    ntpe,
                    scales=scales,
                    group_size=group_size,
                    asym=False,
                )
            )

            flops = _compute_moe_flops(total_tokens, K, N, E)
            tflops = flops / (ark_ms * 1e-3) / 1e12

            _print_row(label, E, N, K, total_tokens, base_ms, ark_ms, tflops, base_with_deq_ms=base_deq_ms)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
