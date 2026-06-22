# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CPU-only micro-benchmark for the ARK flash-attention (tiled online softmax) SDPA kernel.

The script mirrors the style of Neural Speed's CPU ``mha_dense`` benchmarks: it sweeps a
handful of representative decode (``seq_q == 1``) and prefill shapes, times the ARK CPU
kernel against PyTorch's reference ``scaled_dot_product_attention`` and reports per-call
latency plus the resulting speed-up. Correctness is checked first so a reported speed-up is
only ever counted for a kernel that matches the reference within tolerance.

This is intentionally CPU-only: it never touches ``torch.xpu``/``torch.cuda`` and forces the
reference SDPA onto the math backend so both sides run on the CPU.

Usage::

    # default sweep
    python auto_round_extension/ark/test/bench_ark_cpu_sdpa.py

    # custom run, e.g. single shape with CSV output
    OMP_NUM_THREADS=8 python auto_round_extension/ark/test/bench_ark_cpu_sdpa.py \
        --shape decode --batch 1 --heads-q 32 --heads-kv 8 --head-dim 128 \
        --seq-kv 4096 --runs 50 --csv results.csv
"""

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import torch

# Allow running the file directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import auto_round_kernel  # noqa: E402

# Default sweep loosely modelled on Neural Speed's CPU attention benchmarks: a decode
# (single-query) regime with growing KV cache, and a prefill (self-attention) regime.
DEFAULT_DECODE_SHAPES = [
    # (batch, heads_q, heads_kv, head_dim, seq_kv)
    (1, 32, 8, 128, 1024),
    (1, 32, 8, 128, 4096),
    (1, 32, 8, 128, 8192),
    (1, 32, 32, 64, 4096),
]

DEFAULT_PREFILL_SHAPES = [
    # (batch, heads_q, heads_kv, head_dim, seq)
    (1, 32, 8, 128, 512),
    (1, 32, 8, 128, 1024),
    (1, 16, 16, 64, 1024),
]


def _dtype_from_str(name):
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _make_qkv(batch, heads_q, heads_kv, head_dim, seq_q, seq_kv, dtype, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, heads_q, seq_q, head_dim, dtype=torch.float32, generator=gen).to(dtype)
    k = torch.randn(batch, heads_kv, seq_kv, head_dim, dtype=torch.float32, generator=gen).to(dtype)
    v = torch.randn(batch, heads_kv, seq_kv, head_dim, dtype=torch.float32, generator=gen).to(dtype)
    return q, k, v


def _reference_sdpa(q, k, v, scale, is_causal):
    # Force the math backend so the reference also runs on CPU and upcast to fp32 so the
    # comparison isolates kernel error from input rounding.
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        return torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), scale=scale, is_causal=is_causal, enable_gqa=True
        )


def _time_call(fn, warmup, runs):
    for _ in range(warmup):
        fn()
    best = math.inf
    total = 0.0
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        total += elapsed
        best = min(best, elapsed)
    return total / runs, best


def run_case(shape_kind, batch, heads_q, heads_kv, head_dim, seq, dtype, warmup, runs, atol, rtol):
    is_causal = shape_kind == "prefill"
    seq_q = 1 if shape_kind == "decode" else seq
    seq_kv = seq
    scale = 1.0 / math.sqrt(head_dim)

    q, k, v = _make_qkv(batch, heads_q, heads_kv, head_dim, seq_q, seq_kv, dtype)

    def ark_call():
        return auto_round_kernel.sdpa(q, k, v, scale=scale, is_causal=is_causal, tensor_layout="HND")

    actual = ark_call()
    expected = _reference_sdpa(q, k, v, scale, is_causal)
    max_err = (actual.float() - expected).abs().max().item()
    passed = torch.allclose(actual.float(), expected, atol=atol, rtol=rtol)

    ark_mean, ark_best = _time_call(ark_call, warmup, runs)
    ref_mean, ref_best = _time_call(lambda: _reference_sdpa(q, k, v, scale, is_causal), warmup, runs)

    return {
        "shape": shape_kind,
        "batch": batch,
        "heads_q": heads_q,
        "heads_kv": heads_kv,
        "head_dim": head_dim,
        "seq_q": seq_q,
        "seq_kv": seq_kv,
        "dtype": str(dtype).replace("torch.", ""),
        "ark_ms": ark_mean * 1e3,
        "ark_best_ms": ark_best * 1e3,
        "ref_ms": ref_mean * 1e3,
        "speedup": ref_mean / ark_mean if ark_mean > 0 else float("nan"),
        "max_abs_err": max_err,
        "passed": passed,
    }


def _build_cases(args):
    if args.shape == "decode" or args.shape == "all":
        decode = (
            [(args.batch, args.heads_q, args.heads_kv, args.head_dim, args.seq_kv)]
            if args.seq_kv
            else DEFAULT_DECODE_SHAPES
        )
        for batch, hq, hkv, hd, seq in decode:
            yield ("decode", batch, hq, hkv, hd, seq)
    if args.shape == "prefill" or args.shape == "all":
        prefill = (
            [(args.batch, args.heads_q, args.heads_kv, args.head_dim, args.seq_kv)]
            if args.seq_kv
            else DEFAULT_PREFILL_SHAPES
        )
        for batch, hq, hkv, hd, seq in prefill:
            yield ("prefill", batch, hq, hkv, hd, seq)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape", choices=["decode", "prefill", "all"], default="all")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads-q", type=int, default=32)
    parser.add_argument("--heads-kv", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seq-kv", type=int, default=0, help="Override the swept seq length (0 = use default sweep)")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--csv", type=str, default="", help="Optional path to write per-case results as CSV")
    args = parser.parse_args(argv)

    dtype = _dtype_from_str(args.dtype)
    threads = os.environ.get("OMP_NUM_THREADS", str(torch.get_num_threads()))
    print(f"CPU-only ARK SDPA benchmark | torch_threads={torch.get_num_threads()} OMP_NUM_THREADS={threads}")
    header = (
        f"{'shape':<8}{'B':>3}{'Hq':>4}{'Hkv':>4}{'D':>5}{'q':>6}{'kv':>7}"
        f"{'dtype':>10}{'ark(ms)':>11}{'ref(ms)':>11}{'speedup':>9}{'max_err':>11}{'ok':>4}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for shape_kind, batch, hq, hkv, hd, seq in _build_cases(args):
        row = run_case(shape_kind, batch, hq, hkv, hd, seq, dtype, args.warmup, args.runs, args.atol, args.rtol)
        rows.append(row)
        print(
            f"{row['shape']:<8}{row['batch']:>3}{row['heads_q']:>4}{row['heads_kv']:>4}{row['head_dim']:>5}"
            f"{row['seq_q']:>6}{row['seq_kv']:>7}{row['dtype']:>10}{row['ark_ms']:>11.3f}{row['ref_ms']:>11.3f}"
            f"{row['speedup']:>9.2f}{row['max_abs_err']:>11.2e}{('yes' if row['passed'] else 'NO'):>4}"
        )

    if rows:
        geomean = math.exp(sum(math.log(r["speedup"]) for r in rows) / len(rows))
        all_passed = all(r["passed"] for r in rows)
        print("-" * len(header))
        print(f"geomean speedup vs torch math SDPA: {geomean:.2f}x | parity: {'PASS' if all_passed else 'FAIL'}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.csv}")

    return 0 if all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
