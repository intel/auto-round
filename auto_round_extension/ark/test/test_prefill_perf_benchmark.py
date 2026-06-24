#!/usr/bin/env python3
# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""Kernel-only MoE prefill threshold tuning benchmark.

The benchmark compares two strategies for one MoE FFN projection using a
hard-wired MiniMax-M2-style prefill distribution:

1. baseline: one kernel launch per active expert.
2. hybrid: one all-expert padded/clipped launch, followed by one extra launch
   for each hot expert's remaining tokens.

All routing, padding, token-count tensors, and hot-expert metadata are prepared
outside the timed region. The reported time is therefore kernel/wrapper-only.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Callable, List

import torch

try:
    import auto_round_kernel as ark
    from test_moe import (
        _pack_fp8,
        _pack_int2_asym,
        _pack_int2_sym,
        _pack_int4_asym,
        _pack_int4_sym,
        _pack_int8_asym,
        _pack_int8_sym,
    )
except ImportError as exc:
    print(f"Import error: {exc}")
    print("Run this script from auto_round_extension/ark/test after installing auto_round_kernel.")
    raise SystemExit(1) from exc


NUM_EXPERTS = 256
N = 3072
K = 1536
GROUP_SIZE = 128
WARMUP = 5
ITERS = 20
THRESHOLDS = [64, 128, 256, 512, 1024]
DEFAULT_THRESHOLD = 256

REAL_TOKEN_COUNTS = [
    6,
    33,
    106,
    281,
    10,
    729,
    837,
    15,
    42,
    332,
    6,
    48,
    27,
    33,
    0,
    74,
    178,
    86,
    0,
    50,
    38,
    26,
    21,
    10,
    7,
    2,
    163,
    0,
    7,
    1109,
    6,
    15,
    66,
    49,
    1,
    4,
    131,
    3,
    345,
    0,
    482,
    10,
    5,
    17,
    5,
    224,
    189,
    425,
    0,
    20,
    1,
    23,
    26,
    1,
    0,
    95,
    12,
    4,
    74,
    1,
    326,
    72,
    0,
    9,
    28,
    13,
    4,
    112,
    6,
    77,
    1,
    13,
    93,
    215,
    362,
    4,
    4,
    28,
    0,
    12,
    62,
    59,
    59,
    12,
    2,
    0,
    9,
    538,
    69,
    17,
    309,
    38,
    43,
    842,
    361,
    100,
    209,
    27,
    0,
    1,
    16,
    196,
    5,
    710,
    55,
    4,
    8,
    1205,
    1205,
    10,
    23,
    111,
    60,
    0,
    20,
    18,
    43,
    9,
    247,
    7,
    9,
    7,
    5,
    45,
    67,
    49,
    48,
    345,
    351,
    672,
    51,
    1,
    16,
    10,
    1,
    4,
    1,
    0,
    156,
    15,
    65,
    174,
    149,
    67,
    45,
    46,
    35,
    10,
    8,
    55,
    18,
    357,
    7,
    331,
    464,
    128,
    2,
    175,
    546,
    218,
    631,
    12,
    100,
    62,
    5,
    167,
    4,
    13,
    455,
    10,
    19,
    168,
    109,
    164,
    6,
    6,
    52,
    9,
    1,
    3,
    0,
    7,
    9,
    164,
    14,
    27,
    229,
    57,
    212,
    2,
    68,
    30,
    166,
    89,
    359,
    1,
    51,
    0,
    6,
    42,
    11,
    4,
    752,
    3,
    84,
    0,
    565,
    177,
    0,
    0,
    2491,
    6,
    30,
    13,
    6,
    5,
    1,
    84,
    8,
    2,
    211,
    35,
    19,
    1,
    49,
    335,
    12,
    26,
    4,
    10,
    24,
    247,
    2,
    57,
    3,
    368,
    373,
    14,
    300,
    210,
    300,
    644,
    118,
    0,
    277,
    124,
    32,
    3,
    2,
    25,
    23,
    95,
    184,
    3,
    178,
    10,
]

FP_DTYPES = [torch.float16, torch.bfloat16]
QUANT_FORMATS = [
    "int8_sym",
    "int8_asym",
    "int4_sym",
    "int4_asym",
    "int2_sym",
    "int2_asym",
    "fp8_e4m3",
    "fp8_e5m2",
]


@dataclass(frozen=True)
class QuantWeights:
    weights: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor | None
    weight_bits: int | None
    asym: bool


@dataclass(frozen=True)
class BenchmarkRow:
    format_name: str
    dtype_name: str
    threshold: int
    baseline_ms: float
    hybrid_ms: float
    baseline_tflops: float
    hybrid_tflops: float


@dataclass(frozen=True)
class BaselinePlan:
    activations: torch.Tensor
    weights: torch.Tensor
    num_tokens: List[torch.Tensor | None]
    activation_slices: List[torch.Tensor | None]
    weight_slices: List[torch.Tensor | None]
    scale_slices: List[torch.Tensor | None] | None = None
    zero_slices: List[torch.Tensor | None] | None = None


@dataclass(frozen=True)
class HybridPlan:
    padded_activations: torch.Tensor
    all_num_tokens: torch.Tensor
    full_weights: torch.Tensor
    remainder_activations: List[torch.Tensor]
    remainder_weights: List[torch.Tensor]
    remainder_num_tokens: List[torch.Tensor]
    full_scales: torch.Tensor | None = None
    full_zeros: torch.Tensor | None = None
    remainder_scales: List[torch.Tensor | None] | None = None
    remainder_zeros: List[torch.Tensor | None] | None = None


def _expert_offsets(counts: List[int]) -> List[int]:
    offsets = []
    offset = 0
    for count in counts:
        offsets.append(offset)
        offset += count
    return offsets


def _time_ms(fn: Callable[[], None]) -> float:
    for _ in range(WARMUP):
        fn()
    torch.xpu.synchronize()

    times = []
    for _ in range(ITERS):
        start = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _tflops(tokens: int, ms: float) -> float:
    return (2.0 * tokens * N * K) / (ms * 1e9) if ms > 0 else float("nan")


def _hybrid_compute_tokens(counts: List[int], threshold: int) -> int:
    effective_threshold = min(threshold, max(counts))
    return NUM_EXPERTS * effective_threshold + sum(max(count - effective_threshold, 0) for count in counts)


def _make_fp_weights(dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(NUM_EXPERTS, K, N, dtype=dtype, device="xpu") * 0.1


def _make_quant_weights(weight_format: str) -> QuantWeights:
    w_float = (torch.randn(NUM_EXPERTS, N, K, dtype=torch.float32, device="xpu") * 0.1).to(torch.bfloat16)
    scales = torch.empty(NUM_EXPERTS, N, K // GROUP_SIZE, dtype=torch.bfloat16, device="xpu")
    zeros = None
    weight_bits = None
    asym = False

    if weight_format == "int8_sym":
        weight_bits = 8
        weights = _pack_int8_sym(w_float, scales, GROUP_SIZE)
    elif weight_format == "int8_asym":
        weight_bits = 8
        zeros = torch.empty_like(scales)
        weights = _pack_int8_asym(w_float, scales, zeros, GROUP_SIZE)
        asym = True
    elif weight_format == "int4_sym":
        weight_bits = 4
        weights = _pack_int4_sym(w_float, scales, GROUP_SIZE)
    elif weight_format == "int4_asym":
        weight_bits = 4
        zeros = torch.empty_like(scales)
        weights = _pack_int4_asym(w_float, scales, zeros, GROUP_SIZE)
        asym = True
    elif weight_format == "int2_sym":
        weight_bits = 2
        weights = _pack_int2_sym(w_float, scales, GROUP_SIZE)
    elif weight_format == "int2_asym":
        weight_bits = 2
        zeros = torch.empty_like(scales)
        weights = _pack_int2_asym(w_float, scales, zeros, GROUP_SIZE)
        asym = True
    elif weight_format == "fp8_e4m3":
        weights = _pack_fp8(w_float, scales, GROUP_SIZE, torch.float8_e4m3fn)
    elif weight_format == "fp8_e5m2":
        weights = _pack_fp8(w_float, scales, GROUP_SIZE, torch.float8_e5m2)
    else:
        raise ValueError(f"Unknown weight format: {weight_format}")

    return QuantWeights(weights=weights, scales=scales, zeros=zeros, weight_bits=weight_bits, asym=asym)


def _call_fp(activations: torch.Tensor, weights: torch.Tensor, num_tokens: torch.Tensor) -> None:
    ark.moe_gemm(activations, weights, num_tokens)


def _call_quant(
    activations: torch.Tensor,
    weights: torch.Tensor,
    num_tokens: torch.Tensor,
    scales: torch.Tensor | None,
    zeros: torch.Tensor | None,
    weight_bits: int | None,
    asym: bool,
) -> None:
    kwargs = {"scales": scales, "zeros": zeros, "group_size": GROUP_SIZE, "asym": asym}
    if weight_bits is not None:
        kwargs["weight_bits"] = weight_bits
    ark.moe_gemm_prefill(activations, weights, num_tokens, **kwargs)


def _build_baseline_plan(
    activations: torch.Tensor,
    weights: torch.Tensor,
    counts: List[int],
    scales: torch.Tensor | None = None,
    zeros: torch.Tensor | None = None,
) -> BaselinePlan:
    offsets = _expert_offsets(counts)
    num_tokens = []
    activation_slices = []
    weight_slices = []
    scale_slices = [] if scales is not None else None
    zero_slices = [] if zeros is not None else None

    for expert, count in enumerate(counts):
        if count == 0:
            num_tokens.append(None)
            activation_slices.append(None)
            weight_slices.append(None)
            if scale_slices is not None:
                scale_slices.append(None)
            if zero_slices is not None:
                zero_slices.append(None)
            continue

        offset = offsets[expert]
        num_tokens.append(torch.tensor([count], dtype=torch.int32, device="xpu"))
        activation_slices.append(activations[offset : offset + count])
        weight_slices.append(weights[expert : expert + 1])
        if scale_slices is not None:
            scale_slices.append(scales[expert : expert + 1])
        if zero_slices is not None:
            zero_slices.append(zeros[expert : expert + 1])

    return BaselinePlan(
        activations=activations,
        weights=weights,
        num_tokens=num_tokens,
        activation_slices=activation_slices,
        weight_slices=weight_slices,
        scale_slices=scale_slices,
        zero_slices=zero_slices,
    )


def _build_hybrid_plan(
    activations: torch.Tensor,
    weights: torch.Tensor,
    counts: List[int],
    threshold: int,
    scales: torch.Tensor | None = None,
    zeros: torch.Tensor | None = None,
) -> HybridPlan:
    offsets = _expert_offsets(counts)
    effective_threshold = min(threshold, max(counts))
    padded_activations = torch.zeros((NUM_EXPERTS * effective_threshold, K), dtype=activations.dtype, device="xpu")
    all_num_tokens = torch.full((NUM_EXPERTS,), effective_threshold, dtype=torch.int32, device="xpu")
    remainder_activations = []
    remainder_weights = []
    remainder_num_tokens = []
    remainder_scales = [] if scales is not None else None
    remainder_zeros = [] if zeros is not None else None

    for expert, count in enumerate(counts):
        if count > 0:
            take = min(count, effective_threshold)
            src_offset = offsets[expert]
            dst_offset = expert * effective_threshold
            padded_activations[dst_offset : dst_offset + take].copy_(activations[src_offset : src_offset + take])

        if count > effective_threshold:
            remaining = count - effective_threshold
            src_offset = offsets[expert] + effective_threshold
            remainder_activations.append(activations[src_offset : src_offset + remaining])
            remainder_weights.append(weights[expert : expert + 1])
            remainder_num_tokens.append(torch.tensor([remaining], dtype=torch.int32, device="xpu"))
            if remainder_scales is not None:
                remainder_scales.append(scales[expert : expert + 1])
            if remainder_zeros is not None:
                remainder_zeros.append(zeros[expert : expert + 1])

    return HybridPlan(
        padded_activations=padded_activations,
        all_num_tokens=all_num_tokens,
        full_weights=weights,
        full_scales=scales,
        full_zeros=zeros,
        remainder_activations=remainder_activations,
        remainder_weights=remainder_weights,
        remainder_num_tokens=remainder_num_tokens,
        remainder_scales=remainder_scales,
        remainder_zeros=remainder_zeros,
    )


def _run_baseline_fp(plan: BaselinePlan) -> None:
    for activations, weights, num_tokens in zip(plan.activation_slices, plan.weight_slices, plan.num_tokens):
        if num_tokens is not None:
            _call_fp(activations, weights, num_tokens)


def _run_baseline_quant(plan: BaselinePlan, weight_bits: int | None, asym: bool) -> None:
    for expert, num_tokens in enumerate(plan.num_tokens):
        if num_tokens is None:
            continue
        scales = plan.scale_slices[expert] if plan.scale_slices is not None else None
        zeros = plan.zero_slices[expert] if plan.zero_slices is not None else None
        _call_quant(
            plan.activation_slices[expert],
            plan.weight_slices[expert],
            num_tokens,
            scales,
            zeros,
            weight_bits,
            asym,
        )


def _run_hybrid_fp(plan: HybridPlan) -> None:
    _call_fp(plan.padded_activations, plan.full_weights, plan.all_num_tokens)
    for activations, weights, num_tokens in zip(
        plan.remainder_activations, plan.remainder_weights, plan.remainder_num_tokens
    ):
        _call_fp(activations, weights, num_tokens)


def _run_hybrid_quant(plan: HybridPlan, weight_bits: int | None, asym: bool) -> None:
    _call_quant(
        plan.padded_activations,
        plan.full_weights,
        plan.all_num_tokens,
        plan.full_scales,
        plan.full_zeros,
        weight_bits,
        asym,
    )
    for idx, num_tokens in enumerate(plan.remainder_num_tokens):
        scales = plan.remainder_scales[idx] if plan.remainder_scales is not None else None
        zeros = plan.remainder_zeros[idx] if plan.remainder_zeros is not None else None
        _call_quant(
            plan.remainder_activations[idx],
            plan.remainder_weights[idx],
            num_tokens,
            scales,
            zeros,
            weight_bits,
            asym,
        )


def _print_table_header(format_name: str, dtype_name: str, baseline_ms: float, baseline_tflops: float) -> None:
    print()
    print(f"Format/dtype: {format_name}/{dtype_name}")
    print(f"Baseline: {baseline_ms:.4f} ms, {baseline_tflops:.2f} TFLOPs")
    print("-" * 104)
    print(
        f"{'format':<12} {'dtype':<10} {'threshold':>9} {'baseline_ms':>12} {'hybrid_ms':>11} "
        f"{'speedup':>8} {'base_TFLOPs':>12} {'hybrid_TFLOPs':>14}"
    )
    print("-" * 104)


def _print_summary_header() -> None:
    print()
    print(f"Selected-threshold strategy summary (threshold={DEFAULT_THRESHOLD})")
    print("-" * 104)
    print(
        f"{'format':<12} {'dtype':<10} {'threshold':>9} {'baseline_ms':>12} {'hybrid_ms':>11} "
        f"{'speedup':>8} {'base_TFLOPs':>12} {'hybrid_TFLOPs':>14}"
    )
    print("-" * 104)


def _print_row(row: BenchmarkRow) -> None:
    speedup = row.baseline_ms / row.hybrid_ms if row.hybrid_ms > 0 else float("nan")
    print(
        f"{row.format_name:<12} {row.dtype_name:<10} {row.threshold:>9} "
        f"{row.baseline_ms:>12.4f} {row.hybrid_ms:>11.4f} {speedup:>7.2f}x "
        f"{row.baseline_tflops:>12.2f} {row.hybrid_tflops:>14.2f}"
    )


def _benchmark_fp(
    dtype: torch.dtype, counts: List[int], thresholds: List[int], print_table: bool
) -> List[BenchmarkRow]:
    dtype_name = _dtype_name(dtype)
    total_tokens = sum(counts)
    activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
    weights = _make_fp_weights(dtype)
    baseline_plan = _build_baseline_plan(activations, weights, counts)
    baseline_ms = _time_ms(lambda: _run_baseline_fp(baseline_plan))
    baseline_tflops = _tflops(total_tokens, baseline_ms)

    rows = []
    if print_table:
        _print_table_header(dtype_name, dtype_name, baseline_ms, baseline_tflops)
    for threshold in thresholds:
        hybrid_plan = _build_hybrid_plan(activations, weights, counts, threshold)
        hybrid_ms = _time_ms(lambda plan=hybrid_plan: _run_hybrid_fp(plan))
        hybrid_tflops = _tflops(_hybrid_compute_tokens(counts, threshold), hybrid_ms)
        row = BenchmarkRow(dtype_name, dtype_name, threshold, baseline_ms, hybrid_ms, baseline_tflops, hybrid_tflops)
        rows.append(row)
        if print_table:
            _print_row(row)
        del hybrid_plan
    torch.xpu.empty_cache()
    return rows


def _benchmark_quant(
    format_name: str, counts: List[int], thresholds: List[int], print_table: bool
) -> List[BenchmarkRow]:
    dtype_name = "bfloat16"
    total_tokens = sum(counts)
    quant = _make_quant_weights(format_name)
    activations = torch.randn(total_tokens, K, dtype=torch.bfloat16, device="xpu")
    baseline_plan = _build_baseline_plan(activations, quant.weights, counts, quant.scales, quant.zeros)
    baseline_ms = _time_ms(lambda: _run_baseline_quant(baseline_plan, quant.weight_bits, quant.asym))
    baseline_tflops = _tflops(total_tokens, baseline_ms)

    rows = []
    if print_table:
        _print_table_header(format_name, dtype_name, baseline_ms, baseline_tflops)
    for threshold in thresholds:
        hybrid_plan = _build_hybrid_plan(activations, quant.weights, counts, threshold, quant.scales, quant.zeros)
        hybrid_ms = _time_ms(lambda plan=hybrid_plan: _run_hybrid_quant(plan, quant.weight_bits, quant.asym))
        hybrid_tflops = _tflops(_hybrid_compute_tokens(counts, threshold), hybrid_ms)
        row = BenchmarkRow(format_name, dtype_name, threshold, baseline_ms, hybrid_ms, baseline_tflops, hybrid_tflops)
        rows.append(row)
        if print_table:
            _print_row(row)
        del hybrid_plan
    torch.xpu.empty_cache()
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("summary", "full"),
        default="summary",
        help="summary: only threshold=512 in one table; full: print per-dtype threshold tuning tables",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("XPU is required for this benchmark")

    torch.manual_seed(42)
    counts = REAL_TOKEN_COUNTS[:NUM_EXPERTS]
    total_tokens = sum(counts)
    active_experts = sum(count > 0 for count in counts)

    print("=" * 104)
    print("MoE prefill kernel-only threshold tuning benchmark")
    print(f"Configuration: experts={NUM_EXPERTS}, N={N}, K={K}, group_size={GROUP_SIZE}")
    print(f"Distribution: total_tokens={total_tokens}, active_experts={active_experts}")
    thresholds = THRESHOLDS if args.mode == "full" else [DEFAULT_THRESHOLD]
    print(f"Mode: {args.mode}")
    print(f"Thresholds: {thresholds}")
    print("=" * 104)

    rows = []
    for dtype in FP_DTYPES:
        rows.extend(_benchmark_fp(dtype, counts, thresholds, args.mode == "full"))

    for format_name in QUANT_FORMATS:
        try:
            rows.extend(_benchmark_quant(format_name, counts, thresholds, args.mode == "full"))
        except Exception as exc:
            print()
            print(f"Format/dtype: {format_name}/bfloat16")
            print(f"SKIP: {exc}")

    if args.mode == "summary":
        _print_summary_header()
        for row in rows:
            _print_row(row)

    print("=" * 104)


if __name__ == "__main__":
    main()
