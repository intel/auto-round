#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""Bandwidth benchmark for the fused Hadamard + MXFP4 quantization XPU kernel.

    bytes = M*K*sizeof(input) + M*K/2 + M*K/32
    BW    = bytes / latency

Choosing the denominator
------------------------

A bandwidth number only means something relative to what the device could
actually deliver *for this kernel*, and picking that ceiling is the whole
difficulty. Three candidates, in increasing order of usefulness:

``copy_same_shape`` (``dst.copy_(src)``) was the original acceptance baseline
and is now reported for context only. It is not an attainable target, for two
independent reasons. First, the traffic ratio is wrong: a copy reads and writes
in a 1:1 ratio, while the fused kernel reads 64 B and writes 17 B per group
(W/R = 0.266), and the device's cost per byte read is strongly non-linear in
that ratio -- measured on Arc Pro B60, 0.144 ns per 64 B read at W/R = 0, 0.187
at W/R = 0.25 and 0.414 at W/R = 1.0. Second, the instruction mix is wrong:
``memcpy`` reaches 407 GB/s where a hand-written 1:1 SYCL kernel reaches only
310, so even at a matched ratio the two are not comparable. Gating on this
number understates the kernel by roughly 8%.

``quant_only`` strips the Hadamard transform but keeps the quantization, with
byte-identical traffic. It is a clean *ablation* and answers "what does the
transform cost?" -- but it is not a roofline, because it is itself subject to
whatever limits the fused kernel.

``stream_only`` is the baseline this benchmark gates on. It keeps the fused
kernel's loads, its packing shape and its stores, and removes only the
transform and the quantization arithmetic. Same bytes, same access pattern,
same item mapping, no math -- so it *is* the traffic-matched roofline, and
``BW_fused / BW_stream`` is a true utilization figure. It also re-derives itself
automatically when the dtype, the shape or the Hadamard dimension changes,
which none of the alternatives do.

Cache residency
---------------

At small ``M*K`` the working set fits in device cache and absolute bandwidths
stop reflecting DRAM at all -- on Arc Pro B60 the 8 MB configurations measure
over 1 TB/s, which no memory controller on this part can deliver.

Residency is detected from ``stream_only`` itself, not from the copy. The two
have different footprints for the same ``[M, K]`` (the copy touches
``4 B/element``, the kernel ``2.53 B/element``), so the kernel stays partly
resident at sizes where the copy no longer is, and a copy-based test misses it.
Any configuration whose stream baseline beats a sustained DRAM copy by more
than ``CACHE_TOLERANCE`` is flagged.

Flagged configurations are reported but excluded from the gate, and the reason
is worth stating precisely, because it is not merely "the numbers are big".
Once the data is resident the roofline rises to cache bandwidth, and the
transform and quantization arithmetic no longer fits underneath it -- the
kernel stops being memory bound and becomes math bound. Measured: at
``[2048, 2048]`` bf16 the stream baseline reaches 1065 GB/s while the fused
kernel reaches 280, an ``f/s`` of 0.26, whereas the same kernel sits at
0.93-0.97 once the working set spills to DRAM. Both numbers are real; they
answer different questions, and only the DRAM-bound one answers "does this
kernel saturate memory?", which is what the gate is for.
"""

from __future__ import annotations

import argparse
import statistics
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
    HADAMARD_DIM_128,
    SUPPORTED_HADAMARD_DIMS,
    get_hadamard_matrix,
    mxfp4_hadamard_quant,
    mxfp4_hadamard_quant_reference,
    mxfp4_quant_reference,
    mxfp4_stream_reference,
)

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

# Typical prefill shapes: token counts 2K/4K/8K/16K (M=2048 is the smallest
# DRAM-bound prefill config; 1024 is too small to be representative).
DEFAULT_M = (2048, 4096, 8192, 16384)
DEFAULT_K = (2048, 4096, 8192)

# Primary gate: BW(fused) / BW(stream-only). The stream-only baseline is the
# traffic-matched roofline (same loads/stores, no math), so this is the
# kernel's utilization of what the device can actually deliver for this access
# pattern. Measured 0.95-1.00 on Arc Pro B60 for D = 32 and ~0.91 for D = 128.
TARGET_STREAM_RATIO = 0.95

# Secondary gate: BW(HMT+quant) / BW(quant-only) -- the Hadamard ablation.
# Quant-only moves byte-identical traffic and only drops the transform, so this
# isolates the transform's cost: it runs in-register and is expected to be
# hidden behind memory traffic, i.e. near 1.0.
TARGET_QUANT_RATIO = 0.95

# [# CRI-WAN-HMT] WAN per-op activation shapes [M, K] = the input activation of
# each listed GEMM, i.e. the tensor the HMT+quant kernel processes. Confirmed
# scope: bf16 + FWHT + the full (global) M, since bandwidth is M-invariant
# while the kernel is DRAM-bound. ``--wan`` benchmarks exactly these pairs
# (text projection / text-encoder FFN included per the colleague's update).
WAN_SHAPES = [(75600, 5120), (75600, 13824), (512, 5120), (512, 4096)]
WAN_DTYPE = "bf16"

# A configuration is treated as cache-resident, and therefore excluded from the
# DRAM-bandwidth gate, once its stream-only baseline exceeds the sustained DRAM
# copy by this factor. The margin absorbs run-to-run noise and the fact that a
# partially resident working set still gets some cache benefit.
CACHE_TOLERANCE = 1.15

# Buffer size for the sustained DRAM copy baseline. Far larger than any cache on
# a current Intel discrete GPU, so the copy has to reach memory.
DRAM_PROBE_BYTES = 1 << 29  # 512 MiB per buffer


def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def bench(fn, warmup: int, iters: int, reps: int = 5) -> float:
    """Return the median latency of ``fn`` in milliseconds.

    ``torch.xpu.synchronize()`` is called on both boundaries of each rep, so a
    measured window contains exactly ``iters`` completed kernel executions.

    The median over ``reps`` windows, rather than a single mean, is what makes
    the reported ratios stable: the device is often shared, and a single
    interfering process inflates one window by several percent. A mean absorbs
    that spike into the result; a median discards it as long as most windows
    are clean.
    """
    for _ in range(warmup):
        out = fn()
        del out
    samples = []
    for _ in range(reps):
        torch.xpu.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            out = fn()
            del out
        torch.xpu.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0 / float(iters))
    return statistics.median(samples)


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
    # Release both buffers (~2x DRAM_PROBE_BYTES) before the caller allocates
    # again. Rebinding is equivalent to ``del`` here but does not trip ruff's
    # F821 for a name captured by the lambda above.
    src = dst = None
    torch.xpu.empty_cache()
    return gbps


def _dequantize(codes: torch.Tensor, scale: torch.Tensor, k: int) -> torch.Tensor:
    """Unpack (codes, e8m0) back to FP32, for the tolerance check."""
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=codes.device)
    flat = codes.reshape(-1, k // 2).to(torch.int32)
    low = flat & 0x0F
    high = (flat >> 4) & 0x0F
    nibbles = torch.stack((low, high), dim=-1).reshape(-1, k)
    values = levels[nibbles & 0x07] * torch.where((nibbles & 0x08) != 0, -1.0, 1.0)
    exp = scale.reshape(-1, k // GROUP_SIZE).to(torch.int32) - 127
    return torch.ldexp(values.reshape(-1, GROUP_SIZE), exp.reshape(-1, 1)).reshape(-1, k)


def verify_once(
    x: torch.Tensor,
    hadamard: torch.Tensor,
    rows: int,
    *,
    use_xmx: bool = False,
    mode: str = "fused",
) -> bool:
    """Spot-check the first ``rows`` rows against the CPU reference.

    A benchmark that measures an incorrect kernel is worthless, so every
    configuration is validated before it is timed. Only a slice is checked
    because the reference is a slow elementwise implementation.

    ``mode`` selects which contract is checked:

    * ``fused``  -- the default FWHT/Path A path (bit-exact) or, with
      ``use_xmx``, the relaxed XMX path (SQNR >= 15 dB).
    * ``quant``  -- the quant-only ablation, bit-exact against
      :func:`mxfp4_quant_reference`.
    * ``stream`` -- the stream-only roofline baseline, bit-exact against
      :func:`mxfp4_stream_reference`. Verifying this one is not optional: if
      the compiler eliminated its loads, the baseline would report an
      unreachable bandwidth and every ratio computed against it would be wrong.
    """
    sub = x[:rows].contiguous()
    if mode == "quant":
        codes, scale = mxfp4_hadamard_quant(sub, _quant_only=True)
        ref_codes, ref_scale = mxfp4_quant_reference(sub.cpu())
        return torch.equal(codes.cpu(), ref_codes) and torch.equal(scale.cpu(), ref_scale)
    if mode == "stream":
        codes, scale = mxfp4_hadamard_quant(sub, _stream_only=True)
        ref_codes, ref_scale = mxfp4_stream_reference(sub.cpu())
        return torch.equal(codes.cpu(), ref_codes) and torch.equal(scale.cpu(), ref_scale)

    codes, scale = mxfp4_hadamard_quant(sub, hadamard, _force_xmx=use_xmx or None)
    ref_codes, ref_scale = mxfp4_hadamard_quant_reference(sub.cpu(), hadamard.cpu())
    if not use_xmx:
        return torch.equal(codes.cpu(), ref_codes) and torch.equal(scale.cpu(), ref_scale)
    k = sub.shape[-1]
    deq = _dequantize(codes.cpu(), scale.cpu(), k).double()
    ref = _dequantize(ref_codes, ref_scale, k).double()
    err = deq - ref
    sqnr = float(10.0 * torch.log10((ref * ref).sum() / (err * err).sum().clamp_min(1e-30)))
    return sqnr >= 15.0


def run_case(m: int, k: int, dtype: torch.dtype, args: argparse.Namespace, dram_gbps: float) -> dict:
    torch.manual_seed(20260611)
    x = torch.randn((m, k), dtype=dtype, device="xpu")
    hadamard = get_hadamard_matrix(args.hadamard_dim, x.device)

    verify_rows = min(args.verify_rows, m)
    if args.no_verify:
        correct = correct_qo = correct_so = None
    else:
        correct = verify_once(x, hadamard, verify_rows, use_xmx=args.xmx)
        correct_qo = verify_once(x, hadamard, verify_rows, mode="quant")
        correct_so = verify_once(x, hadamard, verify_rows, mode="stream")

    # All three modes move exactly the same bytes (fused_bytes), so their
    # bandwidths are directly comparable and the ratios below are meaningful.
    latency = bench(lambda: mxfp4_hadamard_quant(x, hadamard, _force_xmx=args.xmx or None), args.warmup, args.iters)
    latency_qo = bench(lambda: mxfp4_hadamard_quant(x, _quant_only=True), args.warmup, args.iters)
    latency_so = bench(lambda: mxfp4_hadamard_quant(x, _stream_only=True), args.warmup, args.iters)
    nbytes = fused_bytes(m, k, dtype)
    bw_fused = to_gbps(nbytes, latency)
    bw_qo = to_gbps(nbytes, latency_qo)
    bw_so = to_gbps(nbytes, latency_so)

    copy_latency, copy_bytes = measure_copy_same_shape(m, k, dtype, args.warmup, args.iters)
    bw_copy = to_gbps(copy_bytes, copy_latency)

    def _ok(v: bool | None) -> str:
        return "" if v is None else ("pass" if v else "FAIL")

    return {
        "M": m,
        "K": k,
        "dtype": str(dtype).replace("torch.", ""),
        "correct": _ok(correct),
        "correct_qo": _ok(correct_qo),
        "correct_so": _ok(correct_so),
        "bytes": nbytes,
        "latency_ms": latency,
        "latency_qo_ms": latency_qo,
        "latency_so_ms": latency_so,
        "BW_fused_GBps": bw_fused,
        "BW_quant_GBps": bw_qo,
        "BW_stream_GBps": bw_so,
        # Primary metric: utilization of the traffic-matched roofline.
        "ratio_fused_stream": bw_fused / bw_so if bw_so > 0 else float("nan"),
        # Secondary metric: cost of the Hadamard transform alone.
        "ratio_fused_quant": bw_fused / bw_qo if bw_qo > 0 else float("nan"),
        # Reported for context only; a copy is neither traffic-matched nor
        # instruction-matched to this kernel, so it is not gated on.
        "BW_copy_GBps": bw_copy,
        "ratio_copy": bw_fused / bw_copy if bw_copy > 0 else float("nan"),
        # Detected on the stream baseline, which is both the gate's denominator
        # and a closer proxy for the fused kernel's footprint than the copy is.
        "cached": bw_so > dram_gbps * CACHE_TOLERANCE,
    }


def format_table(rows: list[dict]) -> str:
    header = (
        f"{'M':>7} {'K':>7} {'dtype':>5} {'ok':>4} {'okQ':>4} {'okS':>4} "
        f"{'BW_fused':>9} {'BW_stream':>9} {'f/s':>6} {'BW_quant':>9} {'f/q':>6} "
        f"{'BW_copy':>9} {'f/cp':>6} {'note':>7}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['M']:>7} {r['K']:>7} {r['dtype']:>5} {r['correct']:>4} {r['correct_qo']:>4} "
            f"{r['correct_so']:>4} {r['BW_fused_GBps']:>9.1f} {r['BW_stream_GBps']:>9.1f} "
            f"{r['ratio_fused_stream']:>6.3f} {r['BW_quant_GBps']:>9.1f} {r['ratio_fused_quant']:>6.3f} "
            f"{r['BW_copy_GBps']:>9.1f} {r['ratio_copy']:>6.3f} "
            f"{'cached' if r['cached'] else '':>7}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--wan",
        action="store_true",
        help="benchmark the [# CRI-WAN-HMT] activation shapes (bf16, full M), overriding --m/--k defaults",
    )
    p.add_argument("--m", type=int, nargs="+", default=None, help="M values (default prefill set)")
    p.add_argument("--k", type=int, nargs="+", default=None, help="K values (default prefill set)")
    p.add_argument("--dtype", nargs="+", choices=sorted(DTYPES), default=None, help="default: fp16 bf16")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--verify-rows", type=int, default=64, help="rows spot-checked against the CPU reference")
    p.add_argument("--no-verify", action="store_true", help="skip the correctness spot check")
    p.add_argument(
        "--hadamard-dim",
        type=int,
        default=HADAMARD_DIM,
        choices=SUPPORTED_HADAMARD_DIMS,
        help="transform size: 32 (GEMM activations, one work-item per group) or "
        "32*L for L in {2,4,8,16} (cooperative L-lane FWHT; 128 is the attention head dim)",
    )
    p.add_argument("--target-stream-ratio", type=float, default=TARGET_STREAM_RATIO, help="min BW_fused/BW_stream")
    p.add_argument("--target-quant-ratio", type=float, default=TARGET_QUANT_RATIO, help="min BW_fused/BW_quant-only")
    p.add_argument("--xmx", action="store_true", help="force the XMX path (auto-routed otherwise)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not is_xpu_available():
        print("XPU is not available; nothing to benchmark.")
        return 1
    if args.xmx and args.hadamard_dim != HADAMARD_DIM:
        print(f"--xmx is not supported for hadamard_dim {args.hadamard_dim}.")
        return 1

    if args.wan:
        # WAN deliverable: exact per-op (M, K) pairs, bf16, full global M.
        dtypes = [WAN_DTYPE] if not args.dtype else args.dtype
        combos = list(WAN_SHAPES)
    else:
        dtypes = args.dtype or list(DTYPES)
        combos = [(m, k) for m in (args.m or list(DEFAULT_M)) for k in (args.k or list(DEFAULT_K))]

    print(f"device: {torch.xpu.get_device_name(0)}")
    print(f"warmup={args.warmup} iters={args.iters} (median of 5 windows)")
    print(f"hadamard_dim={args.hadamard_dim}  path={'XMX (forced)' if args.xmx else 'default (FWHT/Path A)'}")
    print(f"bytes = M*K*sizeof(input) + M*K/{2} + M*K/{GROUP_SIZE}")
    print(f"primary gate: f/s = BW_fused / BW_stream-only >= {args.target_stream_ratio}")
    print(f"secondary gate: f/q = BW_fused / BW_quant-only >= {args.target_quant_ratio}")
    print("f/cp (vs dst.copy_) is reported for context only and is NOT gated on.")

    dram_gbps = {name: measure_sustained_dram_copy(DTYPES[name], args.warmup, args.iters) for name in dtypes}
    for name, gbps in dram_gbps.items():
        print(f"sustained DRAM copy ({name}): {gbps:.1f} GB/s")
    print()

    rows: list[dict] = []
    for name in dtypes:
        dtype = DTYPES[name]
        for m, k in combos:
            if k % args.hadamard_dim != 0:
                print(f"skipping K={k}: not a multiple of {args.hadamard_dim}")
                continue
            rows.append(run_case(m, k, dtype, args, dram_gbps[name]))
            torch.xpu.empty_cache()

    if not rows:
        print("No configuration was benchmarked.")
        return 1

    print(format_table(rows))

    failed_correctness = [r for r in rows if "FAIL" in (r["correct"], r["correct_qo"], r["correct_so"])]
    if failed_correctness:
        print(f"\nCORRECTNESS FAILED for {len(failed_correctness)} configuration(s); timings are meaningless.")
        return 1

    # Cache-resident configurations are reported but not gated: with the data in
    # cache the roofline rises and the kernel becomes math bound rather than
    # memory bound, so a DRAM-derived target does not apply to them.
    gated = [r for r in rows if not r["cached"]]
    cached = len(rows) - len(gated)
    if cached:
        print(
            f"\n{cached} of {len(rows)} configuration(s) marked 'cached' and excluded from the gate: "
            "with the working set resident the roofline is cache bandwidth, not DRAM, and the kernel "
            "becomes math bound (see the module docstring)."
        )
    if not gated:
        print("No DRAM-bound configuration was measured; increase M/K.")
        return 1

    ok = True
    for key, target, label in (
        ("ratio_fused_stream", args.target_stream_ratio, "fused/stream-only"),
        ("ratio_fused_quant", args.target_quant_ratio, "fused/quant-only"),
    ):
        values = [r[key] for r in gated]
        below = [r for r in gated if r[key] < target]
        print(f"\nmean {label} ratio over DRAM-bound configurations = {sum(values) / len(values):.3f}")
        print(f"min  {label} ratio over DRAM-bound configurations = {min(values):.3f} (target {target})")
        if below:
            worst = min(below, key=lambda r: r[key])
            print(
                f"FAIL: {len(below)} of {len(gated)} DRAM-bound configuration(s) below the {label} target "
                f"(worst: M={worst['M']} K={worst['K']} {worst['dtype']} at {worst[key]:.3f})."
            )
            ok = False

    if not ok:
        return 1
    print("\nPASS: all configurations meet the bandwidth targets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
