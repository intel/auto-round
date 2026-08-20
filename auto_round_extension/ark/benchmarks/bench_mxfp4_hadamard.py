#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""Bandwidth benchmark for the fused Hadamard + MXFP4 quantization XPU kernel.

Implements the Phase 3 acceptance criterion of
``xpu_mxfp4_hadamard_design_revised.md`` section 7:

    bytes = M*K*sizeof(input) + M*K/2 + M*K/32
    BW    = bytes / latency
    ratio = BW_fused / BW_measured_copy

The kernel is purely memory bound: it reads the activation once and writes
``K/2 + K/32`` bytes per row, so the only meaningful upper bound is the
bandwidth the device actually sustains on a streaming copy -- never a
theoretical peak. The baseline is therefore *measured* on the same device, in
the same dtype, at the same problem size and under the same warmup/timing
protocol as the kernel itself.

The baseline ``copy_same_shape`` is ``dst.copy_(src)`` on an ``[M, K]`` tensor
of the input dtype -- the baseline named in the design doc ("same dtype, same
scale"). It moves ``2*M*K*sizeof(dtype)`` bytes at a 1:1 read:write ratio, and
``ratio = BW_fused / BW_copy`` is the acceptance metric.

Cache residency
---------------

The ratio only means something when both the kernel and its baseline are limited
by DRAM. At small ``M*K`` the whole working set fits in the device cache and
``dst.copy_(src)`` reports several times the part's DRAM bandwidth -- on Arc Pro
B60 the 4 MB configurations measure over 1 TB/s, which no memory controller on
this device can deliver. Dividing by such a number says nothing about the
kernel.

A sustained DRAM copy is therefore measured once on a buffer far larger than the
cache, and any configuration whose own copy baseline beats it by more than
``CACHE_TOLERANCE`` is marked cache-resident. Those rows are still printed, but
they cannot pass or fail the bandwidth gate, because their denominator is not a
bandwidth the kernel could ever reach.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Import the *installed* auto_round_kernel: the benchmark must exercise the
# compiled XPU extension, and the in-tree source directory has no .so beside it.
# Only fall back to the source tree if the package is not installed at all.
try:
    import auto_round_kernel  # noqa: F401
except ImportError:  # pragma: no cover - developer convenience
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auto_round_kernel.mxfp4_hadamard import (  # noqa: E402
    GROUP_SIZE,
    HADAMARD_DIM,
    get_hadamard_matrix,
    mxfp4_hadamard_quant,
    mxfp4_hadamard_quant_reference,
)

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

DEFAULT_M = (1024, 4096, 16384)
DEFAULT_K = (2048, 4096, 8192)

# Phase 3 performance gate.
TARGET_RATIO = 0.90

# A configuration's copy baseline is treated as cache-resident, and therefore
# unusable as a DRAM-bandwidth denominator, once it exceeds the sustained DRAM
# copy by this factor. The margin absorbs run-to-run noise and the fact that a
# partially resident working set still gets some cache benefit.
CACHE_TOLERANCE = 1.15

# Buffer size for the sustained DRAM copy baseline. Far larger than any cache on
# a current Intel discrete GPU, so the copy has to reach memory.
DRAM_PROBE_BYTES = 1 << 29  # 512 MiB per buffer


def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def bench(fn, warmup: int, iters: int) -> float:
    """Return the mean latency of ``fn`` in milliseconds.

    ``torch.xpu.synchronize()`` is called on both timing boundaries so the
    measured window contains exactly ``iters`` completed kernel executions.
    """
    for _ in range(warmup):
        out = fn()
        del out
    torch.xpu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
        del out
    torch.xpu.synchronize()
    return (time.perf_counter() - start) * 1000.0 / float(iters)


def fused_bytes(m: int, k: int, dtype: torch.dtype) -> int:
    """Bytes moved by the fused kernel: read activation, write codes + scales."""
    itemsize = torch.empty((), dtype=dtype).element_size()
    return m * k * itemsize + m * k // 2 + m * k // GROUP_SIZE


def to_gbps(nbytes: int, latency_ms: float) -> float:
    return nbytes / (latency_ms * 1.0e-3) / 1.0e9


def measure_copy_same_shape(m: int, k: int, dtype: torch.dtype, warmup: int, iters: int) -> tuple[float, int]:
    src = torch.randn((m, k), dtype=dtype, device="xpu")
    dst = torch.empty_like(src)
    latency = bench(lambda: dst.copy_(src), warmup, iters)
    return latency, 2 * src.numel() * src.element_size()


def measure_sustained_dram_copy(dtype: torch.dtype, warmup: int, iters: int) -> float:
    """Copy bandwidth on a buffer too large to cache, in GB/s.

    This is the highest bandwidth the device can actually sustain from memory,
    and therefore the ceiling any DRAM-bound kernel is measured against. It is
    used only to decide which per-configuration baselines are cache-resident.
    """
    itemsize = torch.empty((), dtype=dtype).element_size()
    numel = DRAM_PROBE_BYTES // itemsize
    src = torch.randn(numel, dtype=dtype, device="xpu")
    dst = torch.empty_like(src)
    latency = bench(lambda: dst.copy_(src), warmup, iters)
    gbps = to_gbps(2 * src.numel() * itemsize, latency)
    del src, dst
    torch.xpu.empty_cache()
    return gbps


def verify_once(x: torch.Tensor, hadamard: torch.Tensor, rows: int) -> bool:
    """Spot-check the first ``rows`` rows against the CPU reference.

    A benchmark that measures an incorrect kernel is worthless, so every
    configuration is validated before it is timed. Only a slice is checked
    because the reference is a slow elementwise implementation.
    """
    sub = x[:rows].contiguous()
    codes, scale = mxfp4_hadamard_quant(sub, hadamard)
    ref_codes, ref_scale = mxfp4_hadamard_quant_reference(sub.cpu(), hadamard.cpu())
    return torch.equal(codes.cpu(), ref_codes) and torch.equal(scale.cpu(), ref_scale)


def run_case(m: int, k: int, dtype: torch.dtype, args: argparse.Namespace, dram_gbps: float) -> dict:
    torch.manual_seed(20260611)
    x = torch.randn((m, k), dtype=dtype, device="xpu")
    hadamard = get_hadamard_matrix(HADAMARD_DIM, x.device)

    correct = verify_once(x, hadamard, min(args.verify_rows, m)) if not args.no_verify else None

    latency = bench(lambda: mxfp4_hadamard_quant(x, hadamard), args.warmup, args.iters)
    nbytes = fused_bytes(m, k, dtype)
    bw_fused = to_gbps(nbytes, latency)

    copy_latency, copy_bytes = measure_copy_same_shape(m, k, dtype, args.warmup, args.iters)
    bw_copy = to_gbps(copy_bytes, copy_latency)

    return {
        "M": m,
        "K": k,
        "dtype": str(dtype).replace("torch.", ""),
        "correct": "" if correct is None else ("pass" if correct else "FAIL"),
        "bytes": nbytes,
        "latency_ms": latency,
        "BW_fused_GBps": bw_fused,
        "BW_copy_GBps": bw_copy,
        "ratio": bw_fused / bw_copy if bw_copy > 0 else float("nan"),
        # The baseline outran a sustained DRAM copy, so it came from cache and
        # is not a bandwidth the kernel could reach. Reported, but not gated on.
        "cached": bw_copy > dram_gbps * CACHE_TOLERANCE,
    }


def format_table(rows: list[dict]) -> str:
    header = (
        f"{'M':>6} {'K':>6} {'dtype':>8} {'ok':>4} {'lat(ms)':>9} "
        f"{'BW_fused':>9} {'BW_copy':>9} {'ratio':>7} {'note':>7}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['M']:>6} {r['K']:>6} {r['dtype']:>8} {r['correct']:>4} {r['latency_ms']:>9.4f} "
            f"{r['BW_fused_GBps']:>9.1f} {r['BW_copy_GBps']:>9.1f} {r['ratio']:>7.3f} "
            f"{'cached' if r['cached'] else '':>7}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m", type=int, nargs="+", default=list(DEFAULT_M))
    p.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_K))
    p.add_argument("--dtype", nargs="+", choices=sorted(DTYPES), default=["fp16", "bf16"])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--verify-rows", type=int, default=64, help="rows spot-checked against the CPU reference")
    p.add_argument("--no-verify", action="store_true", help="skip the correctness spot check")
    p.add_argument("--target-ratio", type=float, default=TARGET_RATIO)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not is_xpu_available():
        print("XPU is not available; nothing to benchmark.")
        return 1

    print(f"device: {torch.xpu.get_device_name(0)}")
    print(f"warmup={args.warmup} iters={args.iters} target_ratio={args.target_ratio}")
    print(f"bytes = M*K*sizeof(input) + M*K/{2} + M*K/{GROUP_SIZE}")

    dram_gbps = {name: measure_sustained_dram_copy(DTYPES[name], args.warmup, args.iters) for name in args.dtype}
    for name, gbps in dram_gbps.items():
        print(f"sustained DRAM copy ({name}): {gbps:.1f} GB/s")
    print()

    rows: list[dict] = []
    for name in args.dtype:
        dtype = DTYPES[name]
        for m in args.m:
            for k in args.k:
                if k % GROUP_SIZE != 0:
                    print(f"skipping K={k}: not a multiple of {GROUP_SIZE}")
                    continue
                rows.append(run_case(m, k, dtype, args, dram_gbps[name]))
                torch.xpu.empty_cache()

    print(format_table(rows))

    failed_correctness = [r for r in rows if r["correct"] == "FAIL"]
    if failed_correctness:
        print(f"\nCORRECTNESS FAILED for {len(failed_correctness)} configuration(s); timings are meaningless.")
        return 1

    # Cache-resident configurations are excluded: their denominator is a cache
    # copy, not a bandwidth the kernel could ever reach, so they can neither
    # pass nor fail the gate.
    gated = [r for r in rows if not r["cached"]]
    cached = len(rows) - len(gated)
    if cached:
        print(f"\n{cached} of {len(rows)} configuration(s) marked 'cached' and excluded from the gate.")
    if not gated:
        print("No DRAM-bound configuration was measured; increase M/K.")
        return 1

    below = [r for r in gated if r["ratio"] < args.target_ratio]
    worst = min(r["ratio"] for r in gated)
    mean_ratio = sum(r["ratio"] for r in gated) / len(gated)
    print(f"mean ratio over DRAM-bound configurations = {mean_ratio:.3f}")
    print(f"min  ratio over DRAM-bound configurations = {worst:.3f} (target {args.target_ratio})")
    if below:
        print(f"{len(below)} of {len(gated)} configuration(s) below target:")
        for r in below:
            print(f"  M={r['M']} K={r['K']} {r['dtype']}: ratio={r['ratio']:.3f}")
        return 1
    print("PASS: all DRAM-bound configurations meet the bandwidth target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
