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
PyTorch implementation. The baseline is a single ``torch.bmm`` call over a
padded ``[E, M_max, K]`` activations buffer (with already-dequantized
weights for quantized tests), representing the most favorable non-fused
PyTorch fallback: it pays extra FLOPs on padding rows but collapses the
192-per-expert kernel launches into one.

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


def _build_bmm_pad_layout(activations, num_tokens_per_expert, num_experts):
    """Pack ``[total_tokens, K]`` activations into a ``[E, M_max, K]`` buffer.

    Each expert's token slice ``activations[offset:offset+n_e]`` is copied
    into ``act_padded[e, :n_e]``; rows ``act_padded[e, n_e:]`` stay zero.
    This rectangular layout is what the single-``torch.bmm`` baseline
    consumes (see ``_default_moe_prefill``). Building it requires one
    device->host sync to read ``M_max`` and is meant to be done once per
    shape, outside the timed region.
    """
    total_tokens, K = activations.shape
    tpe = num_tokens_per_expert.to(torch.int64)
    M_max = int(tpe.max().item())
    expert_id = torch.repeat_interleave(
        torch.arange(num_experts, device=tpe.device, dtype=torch.int64), tpe
    )
    # Cumulative start offset of each expert's slice in the flat activations.
    offsets = torch.cat([tpe.new_zeros(1), torch.cumsum(tpe, dim=0)])[:-1]
    local_pos = torch.arange(total_tokens, device=tpe.device, dtype=torch.int64) - torch.repeat_interleave(
        offsets, tpe
    )
    act_padded = activations.new_zeros((num_experts, M_max, K))
    act_padded[expert_id, local_pos] = activations
    return act_padded


def _default_moe_prefill(act_padded, dequant_weights):
    """Single-``torch.bmm`` MoE prefill baseline.

    Computes ``[E, M_max, K] @ [E, K, N] -> [E, M_max, N]`` in one batched
    GEMM call. ``act_padded`` (built by ``_build_bmm_pad_layout``) packs
    each expert's token slice into a uniform ``M_max`` rectangle, so the
    192 per-expert ``A @ W.T`` calls of the original Python-loop baseline
    collapse into a single kernel launch -- the launch / dispatch cost
    that previously dominated small-token cases is amortised away. The
    trade-off is that experts with fewer than ``M_max`` tokens do extra
    FLOPs on padding rows; for the heavily skewed shapes this can roughly
    double the bmm's nominal work versus the ragged total, which is the
    intended apples-to-apples ceiling for a non-fused PyTorch baseline.

    Args:
        act_padded: ``[E, M_max, K]`` activations (see ``_build_bmm_pad_layout``).
        dequant_weights: ``[E, N, K]`` dequantized per-expert weights.

    Returns:
        ``[E, M_max, N]`` padded output. Rows corresponding to the padding
        are not meaningful and are intentionally not gathered back into a
        ``[total_tokens, N]`` tensor (this is a perf-only path).
    """
    return torch.bmm(act_padded, dequant_weights.transpose(1, 2))


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
# Shapes follow MiniMax-M2 MoE config:
#   hidden_size         = 3072   (K for up/gate-proj, N for down-proj)
#   intermediate_size   = 1536   (N for up/gate-proj, K for down-proj)
#   num_local_experts   = 192
#   num_experts_per_tok = 8      (top-8 routing)
#
# Total expert-token count per row = seq_len * top_k. Rows are labelled by
# the originating sequence length (2K/4K/8K). Tokens are distributed
# evenly across the 192 experts, except for the "real" rows which replay
# an empirical 256-expert load distribution (resampled to 192) to
# exercise the realistic heavy-tailed routing imbalance.
# ---------------------------------------------------------------------------


# top-8 routing means total expert tokens = seq_len * 8.
_MINIMAX_E = 192
_MINIMAX_N = 1536  # intermediate_size (gate/up output)
_MINIMAX_K = 3072  # hidden_size


# Empirical per-expert token counts captured from a 256-expert MoE run.
# Used as a "shape template" -- ``_minimax_real_tpe`` resamples it down to
# ``_MINIMAX_E = 192`` experts and rescales to the requested total so the
# heavy-tailed real-world load distribution (many near-zero experts, a few
# hot experts with thousands of tokens) drives the benchmark instead of a
# synthetic Gaussian / bimodal one.
_REAL_TPE_TEMPLATE_256 = [
    6, 33, 106, 281, 10, 729, 837, 15, 42, 332, 6, 48, 27, 33, 0, 74,
    178, 86, 0, 50, 38, 26, 21, 10, 7, 2, 163, 0, 7, 1109, 6, 15,
    66, 49, 1, 4, 131, 3, 345, 0, 482, 10, 5, 17, 5, 224, 189, 425,
    0, 20, 1, 23, 26, 1, 0, 95, 12, 4, 74, 1, 326, 72, 0, 9,
    28, 13, 4, 112, 6, 77, 1, 13, 93, 215, 362, 4, 4, 28, 0, 12,
    62, 59, 59, 12, 2, 0, 9, 538, 69, 17, 309, 38, 43, 842, 361, 100,
    209, 27, 0, 1, 16, 196, 5, 710, 55, 4, 8, 1205, 1205, 10, 23, 111,
    60, 0, 20, 18, 43, 9, 247, 7, 9, 7, 5, 45, 67, 49, 48, 345,
    351, 672, 51, 1, 16, 10, 1, 4, 1, 0, 156, 15, 65, 174, 149, 67,
    45, 46, 35, 10, 8, 55, 18, 357, 7, 331, 464, 128, 2, 175, 546, 218,
    631, 12, 100, 62, 5, 167, 4, 13, 455, 10, 19, 168, 109, 164, 6, 6,
    52, 9, 1, 3, 0, 7, 9, 164, 14, 27, 229, 57, 212, 2, 68, 30,
    166, 89, 359, 1, 51, 0, 6, 42, 11, 4, 752, 3, 84, 0, 565, 177,
    0, 0, 2491, 6, 30, 13, 6, 5, 1, 84, 8, 2, 211, 35, 19, 1,
    49, 335, 12, 26, 4, 10, 24, 247, 2, 57, 3, 368, 373, 14, 300, 210,
    300, 644, 118, 0, 277, 124, 32, 3, 2, 25, 23, 95, 184, 3, 178, 10,
]
assert len(_REAL_TPE_TEMPLATE_256) == 256


def _minimax_even_tpe(total: int) -> list[int]:
    """Distribute ``total`` tokens as evenly as possible across all experts.

    Uses ``floor(total/E)`` per expert and spreads the remainder
    (``total % E`` tokens) one-per-expert across the first experts so the
    list sums exactly to ``total`` instead of falling short by up to E-1.
    """
    base = total // _MINIMAX_E
    extra = total % _MINIMAX_E
    return [base + 1 if i < extra else base for i in range(_MINIMAX_E)]


def _minimax_real_tpe(total: int) -> list[int]:
    """Resample the empirical 256-expert distribution down to 192 experts.

    The template in ``_REAL_TPE_TEMPLATE_256`` is collapsed to
    ``_MINIMAX_E`` buckets by summing every source expert into
    ``floor(src_idx * E / 256)``, then proportionally rescaled so the
    resulting list sums exactly to ``total`` (largest-remainder rounding).
    This preserves the heavy-tailed shape (zeros + a few hot experts)
    observed in real MoE routing, keeping the as-measured ordering.
    """
    E = _MINIMAX_E
    src_len = len(_REAL_TPE_TEMPLATE_256)
    # Aggregate 256 source experts into 192 buckets, preserving total mass.
    buckets = [0] * E
    for j, v in enumerate(_REAL_TPE_TEMPLATE_256):
        buckets[j * E // src_len] += v
    s = sum(buckets)
    if s == 0:
        return [0] * E
    scaled = [v * total / s for v in buckets]
    floored = [int(x) for x in scaled]
    diff = total - sum(floored)
    # Distribute the remaining ``diff`` tokens to the buckets with the
    # largest fractional remainders (standard largest-remainder rounding).
    remainders = sorted(range(E), key=lambda i: scaled[i] - floored[i], reverse=True)
    for k in range(diff):
        floored[remainders[k % E]] += 1
    return floored


PREFILL_SHAPES = [
    # (label, num_experts, tokens_per_expert_list, N, K)
    # -- seq_len = 2K -> 16384 expert tokens, ~85/expert ---------------------
    ("minimax up  2K", _MINIMAX_E, _minimax_even_tpe(16384), _MINIMAX_N, _MINIMAX_K),
    ("minimax down 2K", _MINIMAX_E, _minimax_even_tpe(16384), _MINIMAX_K, _MINIMAX_N),
    ("minimax real up  2K", _MINIMAX_E, _minimax_real_tpe(16384), _MINIMAX_N, _MINIMAX_K),
    ("minimax real down 2K", _MINIMAX_E, _minimax_real_tpe(16384), _MINIMAX_K, _MINIMAX_N),
    # -- seq_len = 4K -> 32768 expert tokens, ~171/expert --------------------
    ("minimax up  4K", _MINIMAX_E, _minimax_even_tpe(32768), _MINIMAX_N, _MINIMAX_K),
    ("minimax down 4K", _MINIMAX_E, _minimax_even_tpe(32768), _MINIMAX_K, _MINIMAX_N),
    ("minimax real up  4K", _MINIMAX_E, _minimax_real_tpe(32768), _MINIMAX_N, _MINIMAX_K),
    ("minimax real down 4K", _MINIMAX_E, _minimax_real_tpe(32768), _MINIMAX_K, _MINIMAX_N),
    # -- seq_len = 8K -> 65536 expert tokens, ~341/expert --------------------
    ("minimax up  8K", _MINIMAX_E, _minimax_even_tpe(65536), _MINIMAX_N, _MINIMAX_K),
    ("minimax down 8K", _MINIMAX_E, _minimax_even_tpe(65536), _MINIMAX_K, _MINIMAX_N),
    ("minimax real up  8K", _MINIMAX_E, _minimax_real_tpe(65536), _MINIMAX_N, _MINIMAX_K),
    ("minimax real down 8K", _MINIMAX_E, _minimax_real_tpe(65536), _MINIMAX_K, _MINIMAX_N),
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
    width = 138 if with_dequant_baseline else 118
    print("=" * width)
    print(title)
    if with_dequant_baseline:
        print(
            f"{'shape':<22}{'E':>4}{'N':>7}{'K':>7}{'tokens':>8}"
            f"{'base mm(ms)':>14}{'base+deq(ms)':>16}{'ark(ms)':>12}"
            f"{'speedup':>12}{'TFLOPS':>10}"
        )
    else:
        print(
            f"{'shape':<22}{'E':>4}{'N':>7}{'K':>7}{'tokens':>8}"
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
            f"{label:<22}{E:>4}{N:>7}{K:>7}{total_tokens:>8}"
            f"{base_ms:>16.4f}{ark_ms:>14.4f}{speedup:>11.2f}x{tflops:>9.1f}"
        )
    else:
        speedup = base_with_deq_ms / ark_ms if ark_ms > 0 else float("nan")
        print(
            f"{label:<22}{E:>4}{N:>7}{K:>7}{total_tokens:>8}"
            f"{base_ms:>14.4f}{base_with_deq_ms:>16.4f}{ark_ms:>12.4f}"
            f"{speedup:>11.2f}x{tflops:>9.1f}"
        )


# ---------------------------------------------------------------------------
# Benchmark cases
# ---------------------------------------------------------------------------


@pytest.mark.skipif(bool(_PREFILL_SKIP), reason=_PREFILL_SKIP or "ok")
class TestMoEGemmPrefillPerf:
    """Median XPU-event timings of ``moe_gemm`` vs a single-``torch.bmm`` baseline.

    The baseline pads each expert's token slice to ``M_max`` and runs one
    batched GEMM (see ``_default_moe_prefill``). For quantized tests the
    baseline operates on *already-dequantized* weights so the timed region
    only measures matmul cost. This is the most favorable apples-to-apples
    comparison for a non-fused PyTorch path: it removes the 192-iteration
    Python loop / per-expert launch overhead at the cost of doing extra
    FLOPs on padding rows.
    """

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_perf_fp(self, dtype):
        _print_header(f"FP weights ({str(dtype).split('.')[-1]})  -- ark.moe_gemm (prefill) vs single torch.bmm")
        for label, E, tpe, N, K in PREFILL_SHAPES:
            total_tokens = sum(tpe)
            activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
            # Weights layout: [E, K, N] for moe_gemm
            weights = (torch.randn(E, K, N, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
            ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")

            # Baseline: padded single-bmm. Weights need to be [E, N, K] for the baseline.
            weights_baseline = weights.transpose(1, 2)  # [E, N, K]
            act_padded = _build_bmm_pad_layout(activations, ntpe, E)

            base_ms = _xpu_time_ms(lambda: _default_moe_prefill(act_padded, weights_baseline))
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
            f"-- ark.moe_gemm_prefill (prefill) vs dequant + single torch.bmm",
            with_dequant_baseline=True,
        )
        for label, E, tpe, N, K in PREFILL_SHAPES:
            if K % group_size != 0:
                continue
            total_tokens = sum(tpe)
            activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
            ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")
            act_padded = _build_bmm_pad_layout(activations, ntpe, E)
            # Pack helpers expect weights in [E, N, K] layout.
            w_float = (torch.randn(E, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
            scales = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
            if asym:
                zeros = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
                packed = _pack_int4_asym(w_float, scales, zeros, group_size)
                dequant = _dequant_int4_asym(packed, scales, zeros, group_size).to(dtype)

                def _baseline_with_dequant():
                    d = _dequant_int4_asym(packed, scales, zeros, group_size).to(dtype)
                    return _default_moe_prefill(act_padded, d)
            else:
                zeros = None
                packed = _pack_int4_sym(w_float, scales, group_size)
                dequant = _dequant_int4_sym(packed, scales, group_size).to(dtype)

                def _baseline_with_dequant():
                    d = _dequant_int4_sym(packed, scales, group_size).to(dtype)
                    return _default_moe_prefill(act_padded, d)

            # ``dequant`` is already [E, N, K] -- matches the baseline contract.
            base_ms = _xpu_time_ms(lambda: _default_moe_prefill(act_padded, dequant))
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
            f"-- ark.moe_gemm_prefill (prefill) vs dequant + single torch.bmm",
            with_dequant_baseline=True,
        )
        for label, E, tpe, N, K in PREFILL_SHAPES:
            if K % group_size != 0:
                continue
            total_tokens = sum(tpe)
            activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
            ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")
            act_padded = _build_bmm_pad_layout(activations, ntpe, E)
            w_float = (torch.randn(E, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
            scales = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
            if asym:
                zeros = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
                packed = _pack_int8_asym(w_float, scales, zeros, group_size)
                dequant = _dequant_int8_asym(packed, scales, zeros, group_size).to(dtype)

                def _baseline_with_dequant():
                    d = _dequant_int8_asym(packed, scales, zeros, group_size).to(dtype)
                    return _default_moe_prefill(act_padded, d)
            else:
                zeros = None
                packed = _pack_int8_sym(w_float, scales, group_size)
                dequant = _dequant_int8_sym(packed, scales, group_size).to(dtype)

                def _baseline_with_dequant():
                    d = _dequant_int8_sym(packed, scales, group_size).to(dtype)
                    return _default_moe_prefill(act_padded, d)

            base_ms = _xpu_time_ms(lambda: _default_moe_prefill(act_padded, dequant))
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
            f"-- ark.moe_gemm_prefill (prefill) vs dequant + single torch.bmm",
            with_dequant_baseline=True,
        )
        for label, E, tpe, N, K in PREFILL_SHAPES:
            if K % group_size != 0 or K % 4 != 0:
                continue
            total_tokens = sum(tpe)
            activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
            ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")
            act_padded = _build_bmm_pad_layout(activations, ntpe, E)
            w_float = (torch.randn(E, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
            scales = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
            if asym:
                zeros = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
                packed = _pack_int2_asym(w_float, scales, zeros, group_size)
                dequant = _dequant_int2_asym(packed, scales, zeros, group_size).to(dtype)

                def _baseline_with_dequant():
                    d = _dequant_int2_asym(packed, scales, zeros, group_size).to(dtype)
                    return _default_moe_prefill(act_padded, d)
            else:
                zeros = None
                packed = _pack_int2_sym(w_float, scales, group_size)
                dequant = _dequant_int2_sym(packed, scales, group_size).to(dtype)

                def _baseline_with_dequant():
                    d = _dequant_int2_sym(packed, scales, group_size).to(dtype)
                    return _default_moe_prefill(act_padded, d)

            base_ms = _xpu_time_ms(lambda: _default_moe_prefill(act_padded, dequant))
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
            f"act={str(dtype).split('.')[-1]}) -- ark.moe_gemm_prefill (prefill) vs dequant + single torch.bmm",
            with_dequant_baseline=True,
        )
        for label, E, tpe, N, K in PREFILL_SHAPES:
            if K % group_size != 0:
                continue
            total_tokens = sum(tpe)
            activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
            ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")
            act_padded = _build_bmm_pad_layout(activations, ntpe, E)
            w_float = (torch.randn(E, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
            scales = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
            packed = _pack_fp8(w_float, scales, group_size, fp8_dtype)
            dequant = _dequant_fp8(packed, scales, group_size, dtype)

            def _baseline_with_dequant():
                d = _dequant_fp8(packed, scales, group_size, dtype)
                return _default_moe_prefill(act_padded, d)

            base_ms = _xpu_time_ms(lambda: _default_moe_prefill(act_padded, dequant))
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
