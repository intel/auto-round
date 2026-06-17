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
# Prefill has many tokens per expert (e.g., batch size, prompt length).
# We test small to large expert counts and token distributions typical of
# MoE models during prefill (Mixtral, DeepSeek, etc.).
# ---------------------------------------------------------------------------

PREFILL_SHAPES = [
    # (label, num_experts, tokens_per_expert_list, N, K)
    # Small models (e.g., Mixtral 8x7B style)
    ("small  E=8 ", 8, [32, 28, 30, 35, 33, 31, 29, 34], 4096, 4096),
    ("medium E=8 ", 8, [64, 60, 68, 72, 65, 63, 70, 66], 4096, 14336),  # up-proj
    ("medium E=8 ", 8, [64, 60, 68, 72, 65, 63, 70, 66], 14336, 4096),  # down-proj
    # Larger models (e.g., DeepSeek style with more experts)
    ("large  E=16", 16, [16] * 16, 2048, 2048),
    ("large  E=32", 32, [8] * 32, 2048, 2048),
    ("large  E=64", 64, [4] * 64, 2048, 2048),
    # Uneven distribution (some experts get more tokens)
    ("uneven E=8 ", 8, [100, 50, 75, 80, 60, 90, 70, 85], 4096, 4096),
]


def _print_header(title: str) -> None:
    print()
    print("=" * 110)
    print(title)
    print(
        f"{'shape':<14}{'E':>4}{'N':>7}{'K':>7}{'tokens':>8}"
        f"{'baseline(ms)':>16}{'ark(ms)':>14}{'speedup':>12}{'TFLOPS':>10}"
    )
    print("-" * 110)


def _print_row(label, E, N, K, total_tokens, base_ms, ark_ms, tflops):
    speedup = base_ms / ark_ms if ark_ms > 0 else float("nan")
    print(
        f"{label:<14}{E:>4}{N:>7}{K:>7}{total_tokens:>8}"
        f"{base_ms:>16.4f}{ark_ms:>14.4f}{speedup:>11.2f}x{tflops:>9.1f}"
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
            f"-- ark.moe_gemm_prefill (prefill) vs dequant + per-expert A @ W.T"
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
            else:
                zeros = None
                packed = _pack_int4_sym(w_float, scales, group_size)
                dequant = _dequant_int4_sym(packed, scales, group_size).to(dtype)
            ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")

            # ``dequant`` is already [E, N, K] -- matches the baseline contract.
            base_ms = _xpu_time_ms(lambda: _default_moe_prefill(activations, dequant, ntpe))
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

            _print_row(label, E, N, K, total_tokens, base_ms, ark_ms, tflops)

    @pytest.mark.skipif(bool(_QUANT_PREFILL_SKIP), reason=_QUANT_PREFILL_SKIP or "ok")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("asym", [False, True])
    def test_perf_int8(self, dtype, asym):
        group_size = 128
        kind = "asym" if asym else "sym"
        _print_header(
            f"INT8 {kind} (group_size={group_size}, act={str(dtype).split('.')[-1]}) "
            f"-- ark.moe_gemm_prefill (prefill) vs dequant + per-expert A @ W.T"
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
            else:
                zeros = None
                packed = _pack_int8_sym(w_float, scales, group_size)
                dequant = _dequant_int8_sym(packed, scales, group_size).to(dtype)
            ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")

            base_ms = _xpu_time_ms(lambda: _default_moe_prefill(activations, dequant, ntpe))
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

            _print_row(label, E, N, K, total_tokens, base_ms, ark_ms, tflops)

    @pytest.mark.skipif(bool(_QUANT_PREFILL_SKIP), reason=_QUANT_PREFILL_SKIP or "ok")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("asym", [False, True])
    def test_perf_int2(self, dtype, asym):
        group_size = 128
        kind = "asym" if asym else "sym"
        _print_header(
            f"INT2 {kind} (group_size={group_size}, act={str(dtype).split('.')[-1]}) "
            f"-- ark.moe_gemm_prefill (prefill) vs dequant + per-expert A @ W.T"
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
            else:
                zeros = None
                packed = _pack_int2_sym(w_float, scales, group_size)
                dequant = _dequant_int2_sym(packed, scales, group_size).to(dtype)
            ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")

            base_ms = _xpu_time_ms(lambda: _default_moe_prefill(activations, dequant, ntpe))
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

            _print_row(label, E, N, K, total_tokens, base_ms, ark_ms, tflops)

    @pytest.mark.skipif(bool(_QUANT_PREFILL_SKIP), reason=_QUANT_PREFILL_SKIP or "ok")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_perf_fp8(self, dtype, fp8_dtype):
        group_size = 128
        _print_header(
            f"FP8 {str(fp8_dtype).split('.')[-1]} (group_size={group_size}, "
            f"act={str(dtype).split('.')[-1]}) -- ark.moe_gemm_prefill (prefill) vs dequant + per-expert A @ W.T"
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

            base_ms = _xpu_time_ms(lambda: _default_moe_prefill(activations, dequant, ntpe))
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

            _print_row(label, E, N, K, total_tokens, base_ms, ark_ms, tflops)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
