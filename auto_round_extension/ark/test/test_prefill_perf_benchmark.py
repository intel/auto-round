#!/usr/bin/env python3
# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""
MoE prefill strategy benchmark: baseline vs full pad-and-clip.

This script measures kernel-only wall-clock time of two strategies for a prefill MoE
layer using realistic token distribution from MiniMax-M2 traces.

Strategies compared:
 1. per-expert baseline : loop over active experts, launch one GEMM per expert.
                         This is the current standard approach with minimal padding.

 2. full pad-and-clip  : **all** experts (including inactive ones) are
                         clipped/padded to a fixed threshold, processed in one
                         big GEMM launch, then remaining tokens of hot experts
                         (count > threshold) are handled with individual launches.

The full pad-and-clip strategy trades memory bandwidth (more padding) for reduced
kernel launch overhead. Performance depends on the threshold choice and token
distribution characteristics.

Configuration:
- Model dimensions: N=3072 (output), K=1536 (input), matching typical MoE layers
- Thresholds tested: 128, 256, 512 tokens (power-of-2 for efficient padding)
- Timing: 5 warmup + 20 iterations, median reported
- All parameters are hard-wired (no command line arguments needed)

Example usage:
    python test_moe_prefill_perf_benchmark.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

import torch

# Import quantization helpers
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
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure auto_round_kernel and test_moe are available")
    exit(1)

# ---------------------------------------------------------------------------
# Hard-wired configuration (no command line args)
# ---------------------------------------------------------------------------
NUM_EXPERTS = 256
N = 3072  # output features
K = 1536  # input features
GROUP_SIZE = 128
PAD_MULTIPLE = 1  # alignment for padding
WARMUP = 5
ITERS = 20
THRESHOLDS = [128, 256, 512]  # candidate thresholds to benchmark

# Real per-expert token counts from MiniMax-M2 trace
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

# Test formats: FP16/BF16 + all quantized formats
ACTIVATION_DTYPES = [torch.float16, torch.bfloat16]
WEIGHT_FORMATS = ["int8_sym", "int8_asym", "int4_sym", "int4_asym", "int2_sym", "int2_asym", "fp8_e4m3", "fp8_e5m2"]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def _round_up(value: int, multiple: int) -> int:
    """Round value up to nearest multiple."""
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _expert_offsets(counts: List[int]) -> List[int]:
    """Calculate cumulative offsets for expert token ranges."""
    offsets = []
    offset = 0
    for count in counts:
        offsets.append(offset)
        offset += count
    return offsets


def _time_ms(fn, warmup: int, iters: int) -> float:
    """Time a function with warmup, return median time in ms."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()

    # Actual timing
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)

    times.sort()
    return times[len(times) // 2]  # median


# ---------------------------------------------------------------------------
# Weight preparation functions
# ---------------------------------------------------------------------------
def _make_fp_weights(dtype: torch.dtype):
    """Create FP16/BF16 weights."""
    return torch.randn(NUM_EXPERTS, K, N, dtype=dtype, device="xpu") * 0.1


def _make_quant_weights(weight_format: str):
    """Create quantized weights with scales/zeros."""
    # Start with FP32 weights, convert to BF16 for quantization
    w_float = (torch.randn(NUM_EXPERTS, N, K, dtype=torch.float32, device="xpu") * 0.1).to(torch.bfloat16)
    scales = torch.empty(NUM_EXPERTS, N, K // GROUP_SIZE, dtype=torch.bfloat16, device="xpu")
    zeros = None
    weight_bits = None

    if weight_format == "int8_sym":
        weight_bits = 8
        packed = _pack_int8_sym(w_float, scales, GROUP_SIZE)
        asym = False
    elif weight_format == "int8_asym":
        weight_bits = 8
        zeros = torch.empty_like(scales)
        packed = _pack_int8_asym(w_float, scales, zeros, GROUP_SIZE)
        asym = True
    elif weight_format == "int4_sym":
        weight_bits = 4
        packed = _pack_int4_sym(w_float, scales, GROUP_SIZE)
        asym = False
    elif weight_format == "int4_asym":
        weight_bits = 4
        zeros = torch.empty_like(scales)
        packed = _pack_int4_asym(w_float, scales, zeros, GROUP_SIZE)
        asym = True
    elif weight_format == "int2_sym":
        weight_bits = 2
        packed = _pack_int2_sym(w_float, scales, GROUP_SIZE)
        asym = False
    elif weight_format == "int2_asym":
        weight_bits = 2
        zeros = torch.empty_like(scales)
        packed = _pack_int2_asym(w_float, scales, zeros, GROUP_SIZE)
        asym = True
    elif weight_format == "fp8_e4m3":
        packed = _pack_fp8(w_float, scales, GROUP_SIZE, torch.float8_e4m3fn)
        asym = False
        weight_bits = None
    elif weight_format == "fp8_e5m2":
        packed = _pack_fp8(w_float, scales, GROUP_SIZE, torch.float8_e5m2)
        asym = False
        weight_bits = None
    else:
        raise ValueError(f"Unknown weight format: {weight_format}")

    return packed, scales, zeros, weight_bits, asym


# ---------------------------------------------------------------------------
# Kernel call wrappers
# ---------------------------------------------------------------------------
def _call_fp_moe_gemm(activations, weights, num_tokens):
    """Call FP16/BF16 MoE GEMM."""
    return ark.moe_gemm(activations, weights, num_tokens)


def _call_quant_moe_gemm(activations, weights, num_tokens, scales, zeros, weight_bits, asym):
    """Call quantized MoE GEMM."""
    kwargs = {"scales": scales, "zeros": zeros, "group_size": GROUP_SIZE, "asym": asym}
    if weight_bits is not None:
        kwargs["weight_bits"] = weight_bits
    return ark.moe_gemm_prefill(activations, weights, num_tokens, **kwargs)


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------
def _make_full_pad_clip_activations(activations, tokens_per_expert, offsets, threshold, pad_len):
    """Create padded activations for full pad+clip strategy."""
    padded_act = torch.zeros((NUM_EXPERTS * pad_len, activations.shape[1]), dtype=activations.dtype, device="xpu")

    for expert_id, count in enumerate(tokens_per_expert):
        if count == 0:
            continue
        take = min(count, threshold)
        src_start = offsets[expert_id]
        dst_start = expert_id * pad_len
        padded_act[dst_start : dst_start + take].copy_(activations[src_start : src_start + take])

    return padded_act


def run_baseline_fp(activations, weights, tokens_per_expert):
    """Baseline: per-expert launches for FP weights."""
    offsets = _expert_offsets(tokens_per_expert)

    def _run():
        for expert_id, count in enumerate(tokens_per_expert):
            if count == 0:
                continue
            act = activations[offsets[expert_id] : offsets[expert_id] + count]
            w = weights[expert_id : expert_id + 1]
            nt = torch.tensor([count], dtype=torch.int32, device="xpu")
            _call_fp_moe_gemm(act, w, nt)

    return _time_ms(_run, WARMUP, ITERS)


def run_baseline_quant(activations, weights, scales, zeros, weight_bits, asym, tokens_per_expert):
    """Baseline: per-expert launches for quantized weights."""
    offsets = _expert_offsets(tokens_per_expert)

    def _run():
        for expert_id, count in enumerate(tokens_per_expert):
            if count == 0:
                continue
            act = activations[offsets[expert_id] : offsets[expert_id] + count]
            w = weights[expert_id : expert_id + 1]
            s = scales[expert_id : expert_id + 1] if scales is not None else None
            z = zeros[expert_id : expert_id + 1] if zeros is not None else None
            nt = torch.tensor([count], dtype=torch.int32, device="xpu")
            _call_quant_moe_gemm(act, w, nt, s, z, weight_bits, asym)

    return _time_ms(_run, WARMUP, ITERS)


def run_full_pad_clip_fp(activations, weights, tokens_per_expert, threshold):
    """Full pad+clip strategy for FP weights."""
    offsets = _expert_offsets(tokens_per_expert)
    effective_threshold = min(threshold, max(tokens_per_expert))
    pad_len = _round_up(effective_threshold, PAD_MULTIPLE)

    # Pre-build padded activations (kernel-only timing)
    padded_activations = _make_full_pad_clip_activations(
        activations, tokens_per_expert, offsets, effective_threshold, pad_len
    )
    num_tokens_per_expert = torch.full((NUM_EXPERTS,), pad_len, dtype=torch.int32, device="xpu")

    def _run():
        # Big GEMM: all experts padded/clipped
        _call_fp_moe_gemm(padded_activations, weights, num_tokens_per_expert)

        # Handle remaining tokens for hot experts
        for expert_id, count in enumerate(tokens_per_expert):
            if count > effective_threshold:
                remaining = count - effective_threshold
                src_offset = offsets[expert_id] + effective_threshold
                remaining_acts = activations[src_offset : src_offset + remaining]
                remaining_w = weights[expert_id : expert_id + 1]
                remaining_nt = torch.tensor([remaining], dtype=torch.int32, device="xpu")
                _call_fp_moe_gemm(remaining_acts, remaining_w, remaining_nt)

    return _time_ms(_run, WARMUP, ITERS)


def run_full_pad_clip_quant(activations, weights, scales, zeros, weight_bits, asym, tokens_per_expert, threshold):
    """Full pad+clip strategy for quantized weights."""
    offsets = _expert_offsets(tokens_per_expert)
    effective_threshold = min(threshold, max(tokens_per_expert))
    pad_len = _round_up(effective_threshold, PAD_MULTIPLE)

    # Pre-build padded activations (kernel-only timing)
    padded_activations = _make_full_pad_clip_activations(
        activations, tokens_per_expert, offsets, effective_threshold, pad_len
    )
    num_tokens_per_expert = torch.full((NUM_EXPERTS,), pad_len, dtype=torch.int32, device="xpu")

    def _run():
        # Big GEMM: all experts padded/clipped
        _call_quant_moe_gemm(padded_activations, weights, num_tokens_per_expert, scales, zeros, weight_bits, asym)

        # Handle remaining tokens for hot experts
        for expert_id, count in enumerate(tokens_per_expert):
            if count > effective_threshold:
                remaining = count - effective_threshold
                src_offset = offsets[expert_id] + effective_threshold
                remaining_acts = activations[src_offset : src_offset + remaining]
                remaining_w = weights[expert_id : expert_id + 1]
                remaining_s = scales[expert_id : expert_id + 1] if scales is not None else None
                remaining_z = zeros[expert_id : expert_id + 1] if zeros is not None else None
                remaining_nt = torch.tensor([remaining], dtype=torch.int32, device="xpu")
                _call_quant_moe_gemm(
                    remaining_acts, remaining_w, remaining_nt, remaining_s, remaining_z, weight_bits, asym
                )

    return _time_ms(_run, WARMUP, ITERS)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def main():
    """Run the complete benchmark suite."""
    print("MoE Prefill Benchmark: Baseline vs Full Pad+Clip (kernel-only timing)")
    print("=" * 100)

    # Check XPU availability
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        print("ERROR: XPU not available")
        return

    torch.manual_seed(42)
    tokens_per_expert = REAL_TOKEN_COUNTS[:NUM_EXPERTS]
    total_routed = sum(tokens_per_expert)
    active_experts = sum(1 for c in tokens_per_expert if c > 0)

    print(f"Configuration: N={N}, K={K}, experts={NUM_EXPERTS}, group_size={GROUP_SIZE}")
    print(f"Token distribution: total={total_routed}")
    print(f"Thresholds to test: {THRESHOLDS}")
    print()

    # Results table header
    print(f"{'Format':<12} {'Threshold':<9} {'Baseline_ms':<12} {'PadClip_ms':<12} {'Speedup':<8}")
    print("-" * 100)

    # Test FP16/BF16 formats
    for act_dtype in ACTIVATION_DTYPES:
        activations = torch.randn(total_routed, K, dtype=act_dtype, device="xpu")
        weights = _make_fp_weights(act_dtype)

        # Baseline timing
        baseline_ms = run_baseline_fp(activations, weights, tokens_per_expert)

        # Test each threshold
        for threshold in THRESHOLDS:
            padclip_ms = run_full_pad_clip_fp(activations, weights, tokens_per_expert, threshold)
            speedup = baseline_ms / padclip_ms if padclip_ms > 0 else float("inf")

            dtype_name = str(act_dtype).split(".")[-1]  # torch.float16 -> float16
            print(f"{dtype_name:<12} {threshold:<9} {baseline_ms:<12.4f} {padclip_ms:<12.4f} {speedup:<8.2f}x")

        torch.xpu.empty_cache()

    # Test quantized formats
    for weight_format in WEIGHT_FORMATS:
        try:
            packed, scales, zeros, weight_bits, asym = _make_quant_weights(weight_format)
            activations = torch.randn(total_routed, K, dtype=torch.bfloat16, device="xpu")

            # Baseline timing
            baseline_ms = run_baseline_quant(activations, packed, scales, zeros, weight_bits, asym, tokens_per_expert)

            # Test each threshold
            for threshold in THRESHOLDS:
                padclip_ms = run_full_pad_clip_quant(
                    activations, packed, scales, zeros, weight_bits, asym, tokens_per_expert, threshold
                )
                speedup = baseline_ms / padclip_ms if padclip_ms > 0 else float("inf")

                print(f"{weight_format:<12} {threshold:<9} {baseline_ms:<12.4f} {padclip_ms:<12.4f} {speedup:<8.2f}x")

            torch.xpu.empty_cache()

        except Exception as e:
            print(f"{weight_format:<12} SKIP      Error: {e}")

    print("=" * 100)


if __name__ == "__main__":
    main()
