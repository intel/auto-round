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
padded ``[E, M_max, K]`` activations buffer with **already-dequantized**
weights for quantized tests, representing the most favorable non-fused
PyTorch fallback: it pays extra FLOPs on padding rows but collapses the
192-per-expert kernel launches into one.

Two baseline columns are reported per row:

* ``baseline(ms)`` -- matmul-only cost (weights pre-dequantized). This is
  the apples-to-apples GEMM ceiling for a non-fused PyTorch path.
* ``base+deq(ms)`` -- ``baseline + deq``, where ``deq`` is the cost of
  dequantizing the quantized weight buffer once per step (timed
  separately from the matmul). This reflects what a pipeline that keeps
  weights in quantized storage but reuses a stock matmul baseline would
  pay end-to-end, and is the denominator of the reported ``speedup``
  against our fused kernel. For FP rows ``deq`` is zero and the two
  columns are identical.

Reporting both keeps the comparison transparent: the matmul-only column
shows the raw GEMM ceiling, while the ``base+deq`` column shows the
realistic cost our fused kernel actually has to beat to be worth using.

How to run::

    pytest -v -s auto_round_extension/ark/test/test_moe_prefill_perf.py

The ``-s`` flag is required to see the printed timing tables and TFLOPS.
"""

import os

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


def _release_xpu_memory() -> None:
    """Free cached XPU memory and synchronize.

    Called between shapes to keep allocator fragmentation from one shape
    (especially heavy-tailed padded buffers in the multi-GB range) from
    bleeding into the next shape's timings.
    """
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()
        if hasattr(torch.xpu, "empty_cache"):
            torch.xpu.empty_cache()


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
    expert_id = torch.repeat_interleave(torch.arange(num_experts, device=tpe.device, dtype=torch.int64), tpe)
    # Cumulative start offset of each expert's slice in the flat activations.
    offsets = torch.cat([tpe.new_zeros(1), torch.cumsum(tpe, dim=0)])[:-1]
    local_pos = torch.arange(total_tokens, device=tpe.device, dtype=torch.int64) - torch.repeat_interleave(offsets, tpe)
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
# fmt: off
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
# fmt: on
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


def _minimax_real_tpe(total: int, max_ratio: float | None = None) -> list[int]:
    """Resample the empirical 256-expert distribution down to 192 experts.

    The template in ``_REAL_TPE_TEMPLATE_256`` is collapsed to
    ``_MINIMAX_E`` buckets by summing every source expert into
    ``floor(src_idx * E / 256)``, then proportionally rescaled so the
    resulting list sums exactly to ``total`` (largest-remainder rounding).
    This preserves the heavy-tailed shape (zeros + a few hot experts)
    observed in real MoE routing, keeping the as-measured ordering.

    Args:
        total: Total token count the returned list must sum to exactly.
        max_ratio: Optional cap expressed as a multiple of the mean
            tokens-per-expert (``total / E``). When set, any bucket
            exceeding ``max_ratio * mean`` is clipped down to the cap
            and the excess mass is redistributed proportionally to the
            uncapped buckets. This bounds ``M_max`` of the padded bmm
            baseline (which otherwise blows up to multi-GB buffers on
            heavy-tailed shapes and dominates measurement noise) while
            preserving the qualitative skew of the distribution. The
            default ``None`` performs no clipping and reproduces the
            raw empirical distribution.
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
    if max_ratio is not None and E > 0:
        cap = max(1, int(max_ratio * total / E))
        # Iterate: clipping may push redistributed mass back over the cap.
        for _ in range(E):  # bounded fixed-point loop
            over = [i for i, v in enumerate(floored) if v > cap]
            if not over:
                break
            excess = sum(floored[i] - cap for i in over)
            for i in over:
                floored[i] = cap
            recipients = [i for i in range(E) if floored[i] < cap]
            if not recipients:
                break  # infeasible cap; leave the lost mass (shouldn't happen for reasonable max_ratio)
            w_sum = sum(floored[i] for i in recipients)
            if w_sum == 0:
                # All recipients are empty -- spread uniformly.
                per = excess // len(recipients)
                rem = excess - per * len(recipients)
                for j, i in enumerate(recipients):
                    floored[i] += per + (1 if j < rem else 0)
            else:
                added = 0
                for i in recipients:
                    share = excess * floored[i] // w_sum
                    floored[i] += share
                    added += share
                # Place any rounding leftover into the first recipients with room.
                leftover = excess - added
                for i in recipients:
                    if leftover == 0:
                        break
                    room = cap - floored[i]
                    take = min(room, leftover)
                    floored[i] += take
                    leftover -= take
    return floored


PREFILL_SHAPES = [
    # (label, num_experts, tokens_per_expert_list, N, K)
    # The ``real`` rows use ``max_ratio=8`` to clip the heaviest experts to
    # at most 8x the mean tokens-per-expert. This keeps the qualitative
    # heavy-tail (many near-zero experts, a few hot ones) but bounds
    # ``M_max`` of the padded bmm baseline so its allocator pressure does
    # not dominate the measurement.
    # -- seq_len = 2K -> 16384 expert tokens, ~85/expert ---------------------
    ("minimax up  2K", _MINIMAX_E, _minimax_even_tpe(16384), _MINIMAX_N, _MINIMAX_K),
    ("minimax down 2K", _MINIMAX_E, _minimax_even_tpe(16384), _MINIMAX_K, _MINIMAX_N),
    ("minimax real up  2K", _MINIMAX_E, _minimax_real_tpe(16384, max_ratio=8), _MINIMAX_N, _MINIMAX_K),
    ("minimax real down 2K", _MINIMAX_E, _minimax_real_tpe(16384, max_ratio=8), _MINIMAX_K, _MINIMAX_N),
    # -- seq_len = 4K -> 32768 expert tokens, ~171/expert --------------------
    ("minimax up  4K", _MINIMAX_E, _minimax_even_tpe(32768), _MINIMAX_N, _MINIMAX_K),
    ("minimax down 4K", _MINIMAX_E, _minimax_even_tpe(32768), _MINIMAX_K, _MINIMAX_N),
    ("minimax real up  4K", _MINIMAX_E, _minimax_real_tpe(32768, max_ratio=8), _MINIMAX_N, _MINIMAX_K),
    ("minimax real down 4K", _MINIMAX_E, _minimax_real_tpe(32768, max_ratio=8), _MINIMAX_K, _MINIMAX_N),
    # -- seq_len = 8K -> 65536 expert tokens, ~341/expert --------------------
    ("minimax up  8K", _MINIMAX_E, _minimax_even_tpe(65536), _MINIMAX_N, _MINIMAX_K),
    ("minimax down 8K", _MINIMAX_E, _minimax_even_tpe(65536), _MINIMAX_K, _MINIMAX_N),
    ("minimax real up  8K", _MINIMAX_E, _minimax_real_tpe(65536, max_ratio=8), _MINIMAX_N, _MINIMAX_K),
    ("minimax real down 8K", _MINIMAX_E, _minimax_real_tpe(65536, max_ratio=8), _MINIMAX_K, _MINIMAX_N),
]


def _print_header(title: str) -> None:
    """Print a benchmark header.

    Columns:

    * ``baseline(ms)``: matmul-only baseline (single ``torch.bmm`` over a
      padded ``[E, M_max, K]`` buffer, with weights already dequantized
      for quantized tests). This is the most favorable apples-to-apples
      matmul comparison.
    * ``base+deq(ms)``: matmul baseline plus the per-iteration weight
      dequantization cost (``baseline + deq``). For FP rows this equals
      ``baseline`` (deq is a no-op). This is the comparison point against
      our fused kernel: any pipeline that wants to keep weights in
      quantized storage but reuse a stock matmul baseline must pay the
      dequant cost on every step.
    * ``ark(ms)``: default ARK path (dequant workspace + grouped GEMM,
      or fused-dequant if ``ARK_MOE_PREFILL_FUSED_FP8=1`` is set in the
      env for FP8 rows).
    * ``native(ms)`` / ``native TFLOPS``: FP8 rows only. ARK path with
      ``ARK_MOE_PREFILL_NATIVE_FP8=1`` — the fused scalar native-FP8 GEMM
      that skips the ``[E, K, N]`` bf16/fp16 workspace and folds the
      per-K-group scale into the accumulator. ``--`` for non-FP8 rows.
    * ``dpas(ms)`` / ``dpas TFLOPS``: FP8 rows only. Variant B mixed-input
      DPAS grouped GEMM (default-on branch behind
      ``ARK_MOE_PREFILL_DPAS_FP8``). Prints ``--`` for non-FP8 rows and
      for builds where ``moe_gemm_prefill_fp8_dpas`` is not linked in.
    * ``speedup``: ``(base+deq) / ark``.
    """
    print()
    width = 186
    print("=" * width)
    print(title)
    print(
        f"{'shape':<22}{'E':>4}{'N':>7}{'K':>7}{'tokens':>8}"
        f"{'baseline(ms)':>16}{'base+deq(ms)':>16}{'ark(ms)':>14}{'speedup':>12}{'TFLOPS':>10}"
        f"{'native(ms)':>14}{'native TFLOPS':>16}"
        f"{'dpas(ms)':>14}{'dpas TFLOPS':>16}"
    )
    print("-" * width)


def _print_row(label, E, N, K, total_tokens, base_ms, deq_ms, ark_ms, tflops, native_ms=None, native_tflops=None,
               dpas_ms=None, dpas_tflops=None):
    """Print a benchmark row.

    ``speedup`` is ``(base+deq) / ark`` -- a fair comparison against any
    pipeline that keeps weights quantized and pays per-step dequant cost
    on top of a stock matmul baseline. For FP rows ``deq_ms == 0`` so
    ``base+deq == baseline``.

    ``native_ms`` / ``native_tflops`` are printed for FP8 rows where the
    native fused kernel was benchmarked, and left blank otherwise.
    ``dpas_ms`` / ``dpas_tflops`` similarly for the Variant B DPAS path.
    """
    base_plus_deq_ms = base_ms + deq_ms
    speedup = base_plus_deq_ms / ark_ms if ark_ms > 0 else float("nan")
    if native_ms is None:
        native_col = f"{'--':>14}"
        native_tflops_col = f"{'--':>16}"
    else:
        native_col = f"{native_ms:>14.4f}"
        native_tflops_col = f"{native_tflops:>15.1f} "
    if dpas_ms is None:
        dpas_col = f"{'--':>14}"
        dpas_tflops_col = f"{'--':>16}"
    else:
        dpas_col = f"{dpas_ms:>14.4f}"
        dpas_tflops_col = f"{dpas_tflops:>15.1f} "
    print(
        f"{label:<22}{E:>4}{N:>7}{K:>7}{total_tokens:>8}"
        f"{base_ms:>16.4f}{base_plus_deq_ms:>16.4f}{ark_ms:>14.4f}{speedup:>11.2f}x{tflops:>9.1f}"
        f"{native_col}{native_tflops_col}"
        f"{dpas_col}{dpas_tflops_col}"
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

            # FP path: no dequant cost.
            _print_row(label, E, N, K, total_tokens, base_ms, 0.0, ark_ms, tflops)

            # Drop references to large per-shape tensors before releasing the
            # XPU allocator cache so peak memory stays close to one shape's
            # working set even on heavy-tailed buffers.
            activations = weights = weights_baseline = act_padded = ntpe = None
            _release_xpu_memory()

    @pytest.mark.skipif(bool(_QUANT_PREFILL_SKIP), reason=_QUANT_PREFILL_SKIP or "ok")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("asym", [False, True])
    def test_perf_int4(self, dtype, asym):
        group_size = 128
        kind = "asym" if asym else "sym"
        _print_header(
            f"INT4 {kind} (group_size={group_size}, act={str(dtype).split('.')[-1]}) "
            f"-- ark.moe_gemm_prefill (prefill) vs single torch.bmm (weights pre-dequantized)",
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
            else:
                zeros = None
                packed = _pack_int4_sym(w_float, scales, group_size)
                dequant = _dequant_int4_sym(packed, scales, group_size).to(dtype)

            # ``dequant`` is already [E, N, K] -- matches the baseline contract.
            # We time dequant separately so the report can show both the
            # matmul-only baseline (apples-to-apples GEMM cost) and the
            # ``base+deq`` total that a stock pipeline keeping weights in
            # quantized storage would actually pay per step.
            if asym:
                deq_ms = _xpu_time_ms(lambda: _dequant_int4_asym(packed, scales, zeros, group_size).to(dtype))
            else:
                deq_ms = _xpu_time_ms(lambda: _dequant_int4_sym(packed, scales, group_size).to(dtype))
            base_ms = _xpu_time_ms(lambda: _default_moe_prefill(act_padded, dequant))
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

            _print_row(label, E, N, K, total_tokens, base_ms, deq_ms, ark_ms, tflops)

            activations = ntpe = act_padded = w_float = scales = zeros = packed = dequant = None
            _release_xpu_memory()

    @pytest.mark.skipif(bool(_QUANT_PREFILL_SKIP), reason=_QUANT_PREFILL_SKIP or "ok")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("asym", [False, True])
    def test_perf_int8(self, dtype, asym):
        group_size = 128
        kind = "asym" if asym else "sym"
        _print_header(
            f"INT8 {kind} (group_size={group_size}, act={str(dtype).split('.')[-1]}) "
            f"-- ark.moe_gemm_prefill (prefill) vs single torch.bmm (weights pre-dequantized)",
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
            else:
                zeros = None
                packed = _pack_int8_sym(w_float, scales, group_size)
                dequant = _dequant_int8_sym(packed, scales, group_size).to(dtype)

            if asym:
                deq_ms = _xpu_time_ms(lambda: _dequant_int8_asym(packed, scales, zeros, group_size).to(dtype))
            else:
                deq_ms = _xpu_time_ms(lambda: _dequant_int8_sym(packed, scales, group_size).to(dtype))
            base_ms = _xpu_time_ms(lambda: _default_moe_prefill(act_padded, dequant))

            # Default ARK path (dequant + GEMM). Force
            # `ARK_MOE_PREFILL_DPAS_INT8=0` for this measurement so the
            # `ark(ms)` column measures the legacy dequant path
            # independently of the `dpas(ms)` column below.
            prev_dpas = os.environ.get("ARK_MOE_PREFILL_DPAS_INT8")
            # Force the legacy dequant + GEMM path for the `ark(ms)` column.
            os.environ["ARK_MOE_PREFILL_DPAS_INT8"] = "0"
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

            # Variant B DPAS INT8 (default-on branch). Both sym and asym are
            # supported; asym uses a per-M-row per-K-group activation-sum
            # precompute to fold the zero-point correction.
            dpas_ms = None
            dpas_tflops = None
            os.environ["ARK_MOE_PREFILL_DPAS_INT8"] = "1"
            dpas_ms = _xpu_time_ms(
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
            dpas_tflops = flops / (dpas_ms * 1e-3) / 1e12

            # Restore prior env state.
            if prev_dpas is None:
                os.environ.pop("ARK_MOE_PREFILL_DPAS_INT8", None)
            else:
                os.environ["ARK_MOE_PREFILL_DPAS_INT8"] = prev_dpas

            _print_row(label, E, N, K, total_tokens, base_ms, deq_ms, ark_ms, tflops,
                       dpas_ms=dpas_ms, dpas_tflops=dpas_tflops)

            activations = ntpe = act_padded = w_float = scales = zeros = packed = dequant = None
            _release_xpu_memory()

    @pytest.mark.skipif(bool(_QUANT_PREFILL_SKIP), reason=_QUANT_PREFILL_SKIP or "ok")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("asym", [False, True])
    def test_perf_int2(self, dtype, asym):
        group_size = 128
        kind = "asym" if asym else "sym"
        _print_header(
            f"INT2 {kind} (group_size={group_size}, act={str(dtype).split('.')[-1]}) "
            f"-- ark.moe_gemm_prefill (prefill) vs single torch.bmm (weights pre-dequantized)",
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
            else:
                zeros = None
                packed = _pack_int2_sym(w_float, scales, group_size)
                dequant = _dequant_int2_sym(packed, scales, group_size).to(dtype)

            if asym:
                deq_ms = _xpu_time_ms(lambda: _dequant_int2_asym(packed, scales, zeros, group_size).to(dtype))
            else:
                deq_ms = _xpu_time_ms(lambda: _dequant_int2_sym(packed, scales, group_size).to(dtype))
            base_ms = _xpu_time_ms(lambda: _default_moe_prefill(act_padded, dequant))
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

            _print_row(label, E, N, K, total_tokens, base_ms, deq_ms, ark_ms, tflops)

            activations = ntpe = act_padded = w_float = scales = zeros = packed = dequant = None
            _release_xpu_memory()

    @pytest.mark.skipif(bool(_QUANT_PREFILL_SKIP), reason=_QUANT_PREFILL_SKIP or "ok")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_perf_fp8(self, dtype, fp8_dtype):
        group_size = 128
        _print_header(
            f"FP8 {str(fp8_dtype).split('.')[-1]} (group_size={group_size}, "
            f"act={str(dtype).split('.')[-1]}) -- ark.moe_gemm_prefill (prefill) "
            f"vs single torch.bmm (weights pre-dequantized)",
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

            deq_ms = _xpu_time_ms(lambda: _dequant_fp8(packed, scales, group_size, dtype))
            base_ms = _xpu_time_ms(lambda: _default_moe_prefill(act_padded, dequant))

            # Default ARK path (respects the caller's env, i.e. dequant or
            # fused-dequant depending on `ARK_MOE_PREFILL_FUSED_FP8`). We
            # force `ARK_MOE_PREFILL_NATIVE_FP8=0` and
            # `ARK_MOE_PREFILL_DPAS_FP8=0` for this measurement so the
            # native / dpas columns are independently attributable.
            prev_native = os.environ.get("ARK_MOE_PREFILL_NATIVE_FP8")
            prev_dpas = os.environ.get("ARK_MOE_PREFILL_DPAS_FP8")
            os.environ["ARK_MOE_PREFILL_NATIVE_FP8"] = "0"
            os.environ["ARK_MOE_PREFILL_DPAS_FP8"] = "0"
            try:
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
            finally:
                pass  # keep DPAS off for the following native measurement

            flops = _compute_moe_flops(total_tokens, K, N, E)
            tflops = flops / (ark_ms * 1e-3) / 1e12

            # Native FP8 fused GEMM (scalar path, no [E, K, N] workspace).
            # Keep DPAS disabled so the C++ dispatcher does not shadow the
            # native branch.
            os.environ["ARK_MOE_PREFILL_NATIVE_FP8"] = "1"
            os.environ["ARK_MOE_PREFILL_DPAS_FP8"] = "0"
            try:
                native_ms = _xpu_time_ms(
                    lambda: ark.moe_gemm_prefill(
                        activations,
                        packed,
                        ntpe,
                        scales=scales,
                        group_size=group_size,
                        asym=False,
                    )
                )
            finally:
                pass
            native_tflops = flops / (native_ms * 1e-3) / 1e12

            # Variant B DPAS FP8 (default-on branch). Guarded on the
            # presence of the pybind symbol so builds without the DPAS
            # kernel print `--` for this column instead of failing.
            dpas_ms = None
            dpas_tflops = None
            if hasattr(ark.xpu_lib, "moe_gemm_prefill_fp8_dpas"):
                os.environ["ARK_MOE_PREFILL_NATIVE_FP8"] = "0"
                os.environ["ARK_MOE_PREFILL_DPAS_FP8"] = "1"
                try:
                    dpas_ms = _xpu_time_ms(
                        lambda: ark.moe_gemm_prefill(
                            activations,
                            packed,
                            ntpe,
                            scales=scales,
                            group_size=group_size,
                            asym=False,
                        )
                    )
                finally:
                    pass
                dpas_tflops = flops / (dpas_ms * 1e-3) / 1e12

            # Restore prior env state.
            if prev_native is None:
                os.environ.pop("ARK_MOE_PREFILL_NATIVE_FP8", None)
            else:
                os.environ["ARK_MOE_PREFILL_NATIVE_FP8"] = prev_native
            if prev_dpas is None:
                os.environ.pop("ARK_MOE_PREFILL_DPAS_FP8", None)
            else:
                os.environ["ARK_MOE_PREFILL_DPAS_FP8"] = prev_dpas

            _print_row(label, E, N, K, total_tokens, base_ms, deq_ms, ark_ms, tflops, native_ms, native_tflops,
                       dpas_ms, dpas_tflops)

            activations = ntpe = act_padded = w_float = scales = packed = dequant = None
            _release_xpu_memory()

    @pytest.mark.skipif(bool(_QUANT_PREFILL_SKIP), reason=_QUANT_PREFILL_SKIP or "ok")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_perf_fp8_per_tensor(self, dtype, fp8_dtype):
        """Perf: FP8 per-expert (per-tensor) scale -- Variant A DPAS prefill path.

        One FP32 scale per expert (``scales.shape == [E]``), weights laid
        out ``[E, K, N]`` row-major FP8 (vllm-xpu-kernels convention -- the
        transpose of the per-group ``[E, N, K]`` layout). Dispatches via
        ``moe_gemm_prefill(..., scale_scheme="per_tensor")`` which routes
        to ``moe_gemm_prefill_fp8_dpas`` (Variant A). Skipped silently if
        the build was not linked against that pybind symbol.

        The baseline (``base+deq``) pre-dequantizes weights back to
        ``[E, N, K]`` in the activation dtype for a fair single-``torch.bmm``
        comparison; the ``deq_ms`` column reports the cost of that
        per-tensor dequant so ``(base+deq) / ark`` reflects the end-to-end
        cost of a stock pipeline that keeps weights quantized.

        The ``native(ms)`` and ``dpas(ms)`` columns are ``--`` here because
        the Variant A DPAS entry point IS the ARK column for this scheme
        (there is no separate scalar-native / per-group fallback with
        ``[E]`` scales).
        """
        if not hasattr(ark.xpu_lib, "moe_gemm_prefill_fp8_dpas"):
            pytest.skip("build lacks moe_gemm_prefill_fp8_dpas (Variant A) symbol")

        fp8_finfo_max = torch.finfo(fp8_dtype).max
        _print_header(
            f"FP8 per-expert scale {str(fp8_dtype).split('.')[-1]} "
            f"(scales=[E] fp32, act={str(dtype).split('.')[-1]}) -- "
            f"ark.moe_gemm_prefill(scale_scheme='per_tensor') vs single torch.bmm "
            f"(weights pre-dequantized)",
        )
        for label, E, tpe, N, K in PREFILL_SHAPES:
            total_tokens = sum(tpe)
            activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
            ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")
            act_padded = _build_bmm_pad_layout(activations, ntpe, E)
            # Weights in the vllm layout: [E, K, N] row-major.
            w_float = (torch.randn(E, K, N, dtype=torch.float32, device="xpu") * 0.1)
            # One scalar scale per expert -- max-abs of the tile, matches
            # the semantics of test_accuracy_fp8_per_tensor_dpas.
            amax = w_float.reshape(E, -1).abs().amax(dim=1).clamp_min(1e-8)
            scales = (amax / fp8_finfo_max).to(torch.float32)  # [E] fp32
            packed = (w_float / scales.reshape(E, 1, 1)).to(fp8_dtype)

            # Baseline dequant: cast fp8 -> fp32 -> apply per-tensor scale ->
            # cast to act dtype, then transpose to [E, N, K] which is what
            # `_default_moe_prefill` (single torch.bmm) consumes.
            def _do_dequant():
                dequant_KN = packed.to(torch.float32) * scales.reshape(E, 1, 1)
                return dequant_KN.transpose(1, 2).contiguous().to(dtype)

            dequant_NK = _do_dequant()
            deq_ms = _xpu_time_ms(_do_dequant)
            base_ms = _xpu_time_ms(lambda: _default_moe_prefill(act_padded, dequant_NK))

            ark_ms = _xpu_time_ms(
                lambda: ark.moe_gemm_prefill(
                    activations,
                    packed,
                    ntpe,
                    scales=scales,
                    scale_scheme="per_tensor",
                )
            )

            flops = _compute_moe_flops(total_tokens, K, N, E)
            tflops = flops / (ark_ms * 1e-3) / 1e12

            _print_row(label, E, N, K, total_tokens, base_ms, deq_ms, ark_ms, tflops)

            activations = ntpe = act_padded = w_float = scales = packed = dequant_NK = None
            _release_xpu_memory()

    @pytest.mark.skipif(bool(_QUANT_PREFILL_SKIP), reason=_QUANT_PREFILL_SKIP or "ok")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_perf_int8_per_tensor(self, dtype):
        """Perf: INT8 per-expert (per-tensor) scale -- Variant A DPAS prefill path.

        Sibling of :func:`test_perf_fp8_per_tensor`. Weights are stored as
        one signed byte per element in ``[E, K, N]`` row-major
        ``torch.int8`` (vllm-xpu-kernels layout modulo dtype); scales are
        one FP32 scalar per expert. Dispatches via
        ``moe_gemm_prefill(..., scale_scheme="per_tensor")`` which routes
        to ``moe_gemm_prefill_int_dpas`` (Variant A INT8) when the
        weight dtype is ``torch.int8``. Skipped silently if the build
        was not linked against that pybind symbol.

        The ``native(ms)`` and ``dpas(ms)`` columns are ``--`` here for
        the same reason as the FP8 per-tensor case: the Variant A DPAS
        entry point IS the ARK column for this scheme (no separate
        scalar-native / per-group fallback with ``[E]`` scales).

        STATUS: NEEDS-HARDWARE-VALIDATION.
        """
        if not hasattr(ark.xpu_lib, "moe_gemm_prefill_int_dpas"):
            pytest.skip("build lacks moe_gemm_prefill_int_dpas (Variant A) symbol")

        int8_max = 127.0
        _print_header(
            f"INT8 per-expert scale "
            f"(scales=[E] fp32, act={str(dtype).split('.')[-1]}) -- "
            f"ark.moe_gemm_prefill(scale_scheme='per_tensor') vs single torch.bmm "
            f"(weights pre-dequantized)",
        )
        for label, E, tpe, N, K in PREFILL_SHAPES:
            total_tokens = sum(tpe)
            activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
            ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")
            act_padded = _build_bmm_pad_layout(activations, ntpe, E)
            # Weights in the vllm layout: [E, K, N] row-major.
            w_float = (torch.randn(E, K, N, dtype=torch.float32, device="xpu") * 0.1)
            amax = w_float.reshape(E, -1).abs().amax(dim=1).clamp_min(1e-8)
            scales = (amax / int8_max).to(torch.float32)  # [E] fp32
            packed = (
                (w_float / scales.reshape(E, 1, 1))
                .round()
                .clamp(-128, 127)
                .to(torch.int8)
            )

            def _do_dequant():
                dequant_KN = packed.to(torch.float32) * scales.reshape(E, 1, 1)
                return dequant_KN.transpose(1, 2).contiguous().to(dtype)

            dequant_NK = _do_dequant()
            deq_ms = _xpu_time_ms(_do_dequant)
            base_ms = _xpu_time_ms(lambda: _default_moe_prefill(act_padded, dequant_NK))

            ark_ms = _xpu_time_ms(
                lambda: ark.moe_gemm_prefill(
                    activations,
                    packed,
                    ntpe,
                    scales=scales,
                    scale_scheme="per_tensor",
                )
            )

            flops = _compute_moe_flops(total_tokens, K, N, E)
            tflops = flops / (ark_ms * 1e-3) / 1e12

            _print_row(label, E, N, K, total_tokens, base_ms, deq_ms, ark_ms, tflops)

            activations = ntpe = act_padded = w_float = scales = packed = dequant_NK = None
            _release_xpu_memory()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
