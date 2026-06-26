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

The script supports two API modes:

* ``wrapper`` calls the public Python ARK APIs. Its results include wrapper-side
  validation/normalization work such as shape checks, token-count consistency
  checks, and output/workspace allocation.
* ``raw`` calls ``ark.xpu_lib`` directly with preallocated buffers. This does
  not reimplement the kernel math; it bypasses the Python wrapper so the result
  more closely reflects the underlying kernel plus launch/runtime cost.

Use ``raw`` when comparing baseline vs. hybrid as execution strategies and when
you want to minimize machine-specific wrapper overhead (for example,
``num_tokens.sum().item()`` scalar-sync cost). Treat ``raw`` results as
kernel/runtime-facing measurements, and ``wrapper`` results as end-to-end public
API measurements.
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
    raw_outputs: List[torch.Tensor | None] | None = None
    raw_workspaces: List[torch.Tensor | None] | None = None


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
    raw_full_output: torch.Tensor | None = None
    raw_full_workspace: torch.Tensor | None = None
    raw_remainder_outputs: List[torch.Tensor] | None = None
    raw_remainder_workspaces: List[torch.Tensor | None] | None = None


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


def _call_fp_raw(
    activations: torch.Tensor,
    weights: torch.Tensor,
    num_tokens: torch.Tensor,
    output: torch.Tensor,
) -> None:
    ark.xpu_lib.moe_gemm(
        ark.get_stream(activations),
        activations.data_ptr(),
        weights.data_ptr(),
        0,
        output.data_ptr(),
        ark.cvt_dtype(activations.dtype),
        N,
        K,
        num_tokens.data_ptr(),
        weights.shape[0],
    )


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


def _quant_weight_dtype(weights: torch.Tensor, weight_bits: int | None) -> int:
    if weights.dtype == torch.float8_e4m3fn:
        return ark.ARK_DT.float8_e4m3
    if weights.dtype == torch.float8_e5m2:
        return ark.ARK_DT.float8_e5m2
    if weight_bits == 8:
        return ark.ARK_DT.int8
    if weight_bits == 4:
        return ark.ARK_DT.int4
    if weight_bits == 2:
        return ark.ARK_DT.int2
    raise ValueError(f"Unsupported raw quant weight dtype: weights={weights.dtype}, weight_bits={weight_bits}")


def _call_quant_raw(
    activations: torch.Tensor,
    weights: torch.Tensor,
    num_tokens: torch.Tensor,
    scales: torch.Tensor | None,
    zeros: torch.Tensor | None,
    weight_bits: int | None,
    asym: bool,
    output: torch.Tensor,
    workspace: torch.Tensor,
) -> None:
    ark.xpu_lib.moe_gemm_prefill(
        ark.get_stream(activations),
        activations.data_ptr(),
        weights.data_ptr(),
        scales.data_ptr() if scales is not None else 0,
        zeros.data_ptr() if zeros is not None else 0,
        output.data_ptr(),
        workspace.data_ptr(),
        ark.cvt_dtype(activations.dtype),
        _quant_weight_dtype(weights, weight_bits),
        N,
        K,
        GROUP_SIZE,
        num_tokens.data_ptr(),
        weights.shape[0],
        activations.shape[0],
        asym,
    )


def _build_baseline_plan(
    activations: torch.Tensor,
    weights: torch.Tensor,
    counts: List[int],
    scales: torch.Tensor | None = None,
    zeros: torch.Tensor | None = None,
    raw: bool = False,
) -> BaselinePlan:
    offsets = _expert_offsets(counts)
    num_tokens = []
    activation_slices = []
    weight_slices = []
    scale_slices = [] if scales is not None else None
    zero_slices = [] if zeros is not None else None
    raw_outputs = [] if raw else None
    raw_workspaces = [] if raw and scales is not None else None
    max_count = max(counts) if counts else 0
    shared_raw_output = (
        torch.empty((max_count, N), dtype=activations.dtype, device="xpu") if raw and max_count > 0 else None
    )
    shared_raw_workspace = (
        torch.empty((1, K, N), dtype=activations.dtype, device="xpu")
        if raw and scales is not None and max_count > 0
        else None
    )

    for expert, count in enumerate(counts):
        if count == 0:
            num_tokens.append(None)
            activation_slices.append(None)
            weight_slices.append(None)
            if scale_slices is not None:
                scale_slices.append(None)
            if zero_slices is not None:
                zero_slices.append(None)
            if raw_outputs is not None:
                raw_outputs.append(None)
            if raw_workspaces is not None:
                raw_workspaces.append(None)
            continue

        offset = offsets[expert]
        num_tokens.append(torch.tensor([count], dtype=torch.int32, device="xpu"))
        activation_slices.append(activations[offset : offset + count])
        weight_slices.append(weights[expert : expert + 1])
        if scale_slices is not None:
            scale_slices.append(scales[expert : expert + 1])
        if zero_slices is not None:
            zero_slices.append(zeros[expert: expert + 1])
        if raw_outputs is not None:
            raw_outputs.append(shared_raw_output)
        if raw_workspaces is not None:
            raw_workspaces.append(shared_raw_workspace)

    return BaselinePlan(
        activations=activations,
        weights=weights,
        num_tokens=num_tokens,
        activation_slices=activation_slices,
        weight_slices=weight_slices,
        scale_slices=scale_slices,
        zero_slices=zero_slices,
        raw_outputs=raw_outputs,
        raw_workspaces=raw_workspaces,
    )


def _build_hybrid_plan(
    activations: torch.Tensor,
    weights: torch.Tensor,
    counts: List[int],
    threshold: int,
    scales: torch.Tensor | None = None,
    zeros: torch.Tensor | None = None,
    raw: bool = False,
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
    raw_remainder_outputs = [] if raw else None
    raw_remainder_workspaces = [] if raw and scales is not None else None
    max_remainder = max(max(count - effective_threshold, 0) for count in counts) if counts else 0
    shared_remainder_output = (
        torch.empty((max_remainder, N), dtype=activations.dtype, device="xpu") if raw and max_remainder > 0 else None
    )
    shared_remainder_workspace = (
        torch.empty((1, K, N), dtype=activations.dtype, device="xpu")
        if raw and scales is not None and max_remainder > 0
        else None
    )

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
                remainder_zeros.append(zeros[expert: expert + 1])
            if raw_remainder_outputs is not None:
                raw_remainder_outputs.append(shared_remainder_output)
            if raw_remainder_workspaces is not None:
                raw_remainder_workspaces.append(shared_remainder_workspace)

    raw_full_output = None
    raw_full_workspace = None
    if raw:
        raw_full_output = torch.empty((padded_activations.shape[0], N), dtype=activations.dtype, device="xpu")
        if scales is not None:
            raw_full_workspace = torch.empty((weights.shape[0], K, N), dtype=activations.dtype, device="xpu")

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
        raw_full_output=raw_full_output,
        raw_full_workspace=raw_full_workspace,
        raw_remainder_outputs=raw_remainder_outputs,
        raw_remainder_workspaces=raw_remainder_workspaces,
    )


def _run_baseline_fp(plan: BaselinePlan, api: str) -> None:
    for expert, num_tokens in enumerate(plan.num_tokens):
        if num_tokens is None:
            continue
        activations = plan.activation_slices[expert]
        weights = plan.weight_slices[expert]
        if api == "raw":
            _call_fp_raw(activations, weights, num_tokens, plan.raw_outputs[expert])
        else:
            _call_fp(activations, weights, num_tokens)


def _run_baseline_quant(plan: BaselinePlan, weight_bits: int | None, asym: bool, api: str) -> None:
    for expert, num_tokens in enumerate(plan.num_tokens):
        if num_tokens is None:
            continue
        scales = plan.scale_slices[expert] if plan.scale_slices is not None else None
        zeros = plan.zero_slices[expert] if plan.zero_slices is not None else None
        if api == "raw":
            _call_quant_raw(
                plan.activation_slices[expert],
                plan.weight_slices[expert],
                num_tokens,
                scales,
                zeros,
                weight_bits,
                asym,
                plan.raw_outputs[expert],
                plan.raw_workspaces[expert],
            )
        else:
            _call_quant(
                plan.activation_slices[expert],
                plan.weight_slices[expert],
                num_tokens,
                scales,
                zeros,
                weight_bits,
                asym,
            )


def _run_hybrid_fp(plan: HybridPlan, api: str) -> None:
    if api == "raw":
        _call_fp_raw(plan.padded_activations, plan.full_weights, plan.all_num_tokens, plan.raw_full_output)
        for idx, num_tokens in enumerate(plan.remainder_num_tokens):
            _call_fp_raw(
                plan.remainder_activations[idx],
                plan.remainder_weights[idx],
                num_tokens,
                plan.raw_remainder_outputs[idx],
            )
    else:
        _call_fp(plan.padded_activations, plan.full_weights, plan.all_num_tokens)
        for activations, weights, num_tokens in zip(
            plan.remainder_activations, plan.remainder_weights, plan.remainder_num_tokens
        ):
            _call_fp(activations, weights, num_tokens)


def _run_hybrid_quant(plan: HybridPlan, weight_bits: int | None, asym: bool, api: str) -> None:
    if api == "raw":
        _call_quant_raw(
            plan.padded_activations,
            plan.full_weights,
            plan.all_num_tokens,
            plan.full_scales,
            plan.full_zeros,
            weight_bits,
            asym,
            plan.raw_full_output,
            plan.raw_full_workspace,
        )
    else:
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
        if api == "raw":
            _call_quant_raw(
                plan.remainder_activations[idx],
                plan.remainder_weights[idx],
                num_tokens,
                scales,
                zeros,
                weight_bits,
                asym,
                plan.raw_remainder_outputs[idx],
                plan.raw_remainder_workspaces[idx],
            )
        else:
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
    dtype: torch.dtype, counts: List[int], thresholds: List[int], print_table: bool, api: str
) -> List[BenchmarkRow]:
    dtype_name = _dtype_name(dtype)
    total_tokens = sum(counts)
    activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
    weights = _make_fp_weights(dtype)
    baseline_plan = _build_baseline_plan(activations, weights, counts, raw=api == "raw")
    baseline_ms = _time_ms(lambda: _run_baseline_fp(baseline_plan, api))
    baseline_tflops = _tflops(total_tokens, baseline_ms)

    rows = []
    if print_table:
        _print_table_header(dtype_name, dtype_name, baseline_ms, baseline_tflops)
    for threshold in thresholds:
        hybrid_plan = _build_hybrid_plan(activations, weights, counts, threshold, raw=api == "raw")
        hybrid_ms = _time_ms(lambda plan=hybrid_plan: _run_hybrid_fp(plan, api))
        hybrid_tflops = _tflops(_hybrid_compute_tokens(counts, threshold), hybrid_ms)
        row = BenchmarkRow(dtype_name, dtype_name, threshold, baseline_ms, hybrid_ms, baseline_tflops, hybrid_tflops)
        rows.append(row)
        if print_table:
            _print_row(row)
        del hybrid_plan
    torch.xpu.empty_cache()
    return rows


def _benchmark_quant(
    format_name: str, counts: List[int], thresholds: List[int], print_table: bool, api: str
) -> List[BenchmarkRow]:
    dtype_name = "bfloat16"
    total_tokens = sum(counts)
    quant = _make_quant_weights(format_name)
    activations = torch.randn(total_tokens, K, dtype=torch.bfloat16, device="xpu")
    baseline_plan = _build_baseline_plan(
        activations,
        quant.weights,
        counts,
        quant.scales,
        quant.zeros,
        raw=api == "raw",
    )
    baseline_ms = _time_ms(lambda: _run_baseline_quant(baseline_plan, quant.weight_bits, quant.asym, api))
    baseline_tflops = _tflops(total_tokens, baseline_ms)

    rows = []
    if print_table:
        _print_table_header(format_name, dtype_name, baseline_ms, baseline_tflops)
    for threshold in thresholds:
        hybrid_plan = _build_hybrid_plan(
            activations,
            quant.weights,
            counts,
            threshold,
            quant.scales,
            quant.zeros,
            raw=api == "raw",
        )
        hybrid_ms = _time_ms(lambda plan=hybrid_plan: _run_hybrid_quant(plan, quant.weight_bits, quant.asym, api))
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
    parser.add_argument(
        "--api",
        choices=("wrapper", "raw"),
        default="wrapper",
        help=(
            "wrapper: measure the public Python API including validation/allocation overhead; "
            "raw: call xpu_lib directly with preallocated buffers to better isolate kernel/runtime cost"
        ),
    )
    return parser.parse_args()


def _cleanup_prefill_runtime() -> None:
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        return
    if hasattr(ark, "clear_moe_prefill_workspace_cache"):
        ark.clear_moe_prefill_workspace_cache()
    torch.xpu.synchronize()
    torch.xpu.empty_cache()


def main() -> None:
    try:
        args = _parse_args()
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("XPU is required for this benchmark")
        if args.api == "raw" and ark.xpu_lib is None:
            raise RuntimeError("raw API requires auto_round_kernel XPU library")

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
        print(f"API: {args.api}")
        print(f"Thresholds: {thresholds}")
        print("=" * 104)

        rows = []
        for dtype in FP_DTYPES:
            rows.extend(_benchmark_fp(dtype, counts, thresholds, args.mode == "full", args.api))

        for format_name in QUANT_FORMATS:
            try:
                rows.extend(_benchmark_quant(format_name, counts, thresholds, args.mode == "full", args.api))
            except Exception as exc:
                print()
                print(f"Format/dtype: {format_name}/bfloat16")
                print(f"SKIP: {exc}")

        if args.mode == "summary":
            _print_summary_header()
            for row in rows:
                _print_row(row)

        print("=" * 104)
    finally:
        _cleanup_prefill_runtime()


if __name__ == "__main__":
    main()
