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

The ``--mode`` flag selects which paths to benchmark:
  raw    — Tier 0 scalar (default Python path) vs PyTorch math SDPA (default).
  packed — Tier 1 BestLA packed KV cache path vs PyTorch math SDPA.
           Requires ARK_UNSAFE_BESTLA_MIXED_SDPA=1 and the BestLA extension build.
           Only valid for mixed dtypes (float16 or bfloat16 KV).
  both   — Side-by-side: raw mixed path vs packed mixed path vs PyTorch reference.
           Shows the packed-vs-raw latency ratio to quantify the reorder overhead.

Usage::

    # default sweep (Tier 0 raw path)
    python auto_round_extension/ark/test/bench_ark_cpu_sdpa.py

    # packed KV cache path benchmark (Route 1, decode only)
    ARK_UNSAFE_BESTLA_MIXED_SDPA=1 \\
    python auto_round_extension/ark/test/bench_ark_cpu_sdpa.py \\
        --dtype float16 --shape decode --mode packed

    # raw vs packed comparison for regression tracking (Route 2, decode)
    ARK_UNSAFE_BESTLA_MIXED_SDPA=1 \\
    python auto_round_extension/ark/test/bench_ark_cpu_sdpa.py \\
        --dtype bfloat16 --shape decode --mode both

    # custom run with CSV output
    OMP_NUM_THREADS=8 python auto_round_extension/ark/test/bench_ark_cpu_sdpa.py \\
        --shape decode --batch 1 --heads-q 32 --heads-kv 8 --head-dim 128 \\
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


def run_case_packed(shape_kind, batch, heads_q, heads_kv, head_dim, seq, dtype, warmup, runs, atol, rtol):
    """Benchmark the Tier 1 packed KV cache path (ark_cpu_bestla_sdpa_packed).

    Only meaningful for mixed dtypes (float16 or bfloat16 KV).  Requires
    ARK_UNSAFE_BESTLA_MIXED_SDPA=1 and the BestLA extension build.  Returns None
    when the packed path is unavailable (no extension or ISA not present).
    """
    if dtype not in (torch.float16, torch.bfloat16):
        return None
    if os.environ.get("ARK_UNSAFE_BESTLA_MIXED_SDPA", "0") != "1":
        return None
    if not hasattr(auto_round_kernel, "ark_cpu_packed_kv_alloc"):
        return None

    is_causal = shape_kind == "prefill"
    seq_q = 1 if shape_kind == "decode" else seq
    seq_kv = seq
    scale = 1.0 / math.sqrt(head_dim)

    q_f32, k, v = _make_qkv(batch, heads_q, heads_kv, head_dim, seq_q, seq_kv, dtype)
    q_f32 = q_f32.float()

    try:
        cache_k, cache_v = auto_round_kernel.ark_cpu_packed_kv_alloc(
            batch, heads_kv, seq_kv, head_dim, dtype=dtype
        )
        auto_round_kernel.ark_cpu_update_packed_kv(cache_k, cache_v, k, v, 0, seq_kv)
    except (RuntimeError, ValueError, NotImplementedError):
        return None

    def packed_call():
        return auto_round_kernel.ark_cpu_bestla_sdpa_packed(
            q_f32, cache_k, cache_v, seq_kv, seq_kv, heads_kv,
            is_causal=is_causal, scale=scale, tensor_layout="HND",
        )

    try:
        actual = packed_call()
    except (RuntimeError, ValueError, NotImplementedError):
        return None

    expected = _reference_sdpa(q_f32, k, v, scale, is_causal)
    max_err = (actual.float() - expected).abs().max().item()
    passed = torch.allclose(actual.float(), expected, atol=atol, rtol=rtol)

    packed_mean, packed_best = _time_call(packed_call, warmup, runs)
    ref_mean, _ = _time_call(lambda: _reference_sdpa(q_f32, k, v, scale, is_causal), warmup, runs)

    return {
        "shape": shape_kind,
        "batch": batch,
        "heads_q": heads_q,
        "heads_kv": heads_kv,
        "head_dim": head_dim,
        "seq_q": seq_q,
        "seq_kv": seq_kv,
        "dtype": str(dtype).replace("torch.", ""),
        "packed_ms": packed_mean * 1e3,
        "packed_best_ms": packed_best * 1e3,
        "ref_ms": ref_mean * 1e3,
        "speedup": ref_mean / packed_mean if packed_mean > 0 else float("nan"),
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
    parser.add_argument(
        "--mode",
        choices=["raw", "packed", "both"],
        default="raw",
        help=(
            "raw: Tier 0 scalar vs PyTorch ref (default); "
            "packed: Tier 1 packed KV vs PyTorch ref (requires ARK_UNSAFE_BESTLA_MIXED_SDPA=1 and mixed dtype); "
            "both: raw mixed path + packed mixed path side-by-side vs PyTorch ref"
        ),
    )
    args = parser.parse_args(argv)

    dtype = _dtype_from_str(args.dtype)
    threads = os.environ.get("OMP_NUM_THREADS", str(torch.get_num_threads()))
    print(f"CPU-only ARK SDPA benchmark | torch_threads={torch.get_num_threads()} OMP_NUM_THREADS={threads}")
    print(f"mode={args.mode} dtype={args.dtype}")

    run_raw = args.mode in ("raw", "both")
    run_packed = args.mode in ("packed", "both")

    all_passed = True

    if run_raw:
        header = (
            f"{'shape':<8}{'B':>3}{'Hq':>4}{'Hkv':>4}{'D':>5}{'q':>6}{'kv':>7}"
            f"{'dtype':>10}{'ark(ms)':>11}{'ref(ms)':>11}{'speedup':>9}{'max_err':>11}{'ok':>4}"
        )
        print("\n[raw path]")
        print(header)
        print("-" * len(header))

        raw_rows = []
        for shape_kind, batch, hq, hkv, hd, seq in _build_cases(args):
            row = run_case(shape_kind, batch, hq, hkv, hd, seq, dtype, args.warmup, args.runs, args.atol, args.rtol)
            raw_rows.append(row)
            print(
                f"{row['shape']:<8}{row['batch']:>3}{row['heads_q']:>4}{row['heads_kv']:>4}{row['head_dim']:>5}"
                f"{row['seq_q']:>6}{row['seq_kv']:>7}{row['dtype']:>10}{row['ark_ms']:>11.3f}{row['ref_ms']:>11.3f}"
                f"{row['speedup']:>9.2f}{row['max_abs_err']:>11.2e}{('yes' if row['passed'] else 'NO'):>4}"
            )

        if raw_rows:
            geomean = math.exp(sum(math.log(r["speedup"]) for r in raw_rows) / len(raw_rows))
            raw_passed = all(r["passed"] for r in raw_rows)
            all_passed = all_passed and raw_passed
            print("-" * len(header))
            print(f"geomean speedup vs torch math SDPA: {geomean:.2f}x | parity: {'PASS' if raw_passed else 'FAIL'}")

        if args.csv and run_raw and not run_packed:
            with open(args.csv, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(raw_rows[0].keys()))
                writer.writeheader()
                writer.writerows(raw_rows)
            print(f"wrote {len(raw_rows)} rows to {args.csv}")

    if run_packed:
        pack_header = (
            f"{'shape':<8}{'B':>3}{'Hq':>4}{'Hkv':>4}{'D':>5}{'q':>6}{'kv':>7}"
            f"{'dtype':>10}{'packed(ms)':>12}{'ref(ms)':>11}{'speedup':>9}{'max_err':>11}{'ok':>4}"
        )
        print("\n[packed KV path — ARK_UNSAFE_BESTLA_MIXED_SDPA=1 required]")
        print(pack_header)
        print("-" * len(pack_header))

        packed_rows = []
        skipped = 0
        for shape_kind, batch, hq, hkv, hd, seq in _build_cases(args):
            row = run_case_packed(
                shape_kind, batch, hq, hkv, hd, seq, dtype, args.warmup, args.runs, args.atol, args.rtol
            )
            if row is None:
                skipped += 1
                print(f"  {'SKIP':<8} shape={shape_kind} seq_kv={seq} (unavailable on this ISA/build)")
                continue
            packed_rows.append(row)
            print(
                f"{row['shape']:<8}{row['batch']:>3}{row['heads_q']:>4}{row['heads_kv']:>4}{row['head_dim']:>5}"
                f"{row['seq_q']:>6}{row['seq_kv']:>7}{row['dtype']:>10}{row['packed_ms']:>12.3f}{row['ref_ms']:>11.3f}"
                f"{row['speedup']:>9.2f}{row['max_abs_err']:>11.2e}{('yes' if row['passed'] else 'NO'):>4}"
            )

        if packed_rows:
            geomean = math.exp(sum(math.log(r["speedup"]) for r in packed_rows) / len(packed_rows))
            packed_passed = all(r["passed"] for r in packed_rows)
            all_passed = all_passed and packed_passed
            print("-" * len(pack_header))
            print(
                f"geomean speedup (packed) vs torch math SDPA: {geomean:.2f}x | "
                f"parity: {'PASS' if packed_passed else 'FAIL'}"
            )
        elif skipped:
            print(f"  All {skipped} cases skipped — BestLA extension not built or ISA unavailable.")

        # Raw-vs-packed ratio when running both.
        if run_raw and packed_rows and raw_rows:
            print("\n[raw vs packed comparison]")
            cmp_header = (
                f"{'shape':<8}{'B':>3}{'Hq':>4}{'Hkv':>4}{'D':>5}{'q':>6}{'kv':>7}"
                f"{'dtype':>10}{'raw(ms)':>10}{'packed(ms)':>12}{'ratio':>8}"
            )
            print(cmp_header)
            print("-" * len(cmp_header))
            for raw, packed in zip(raw_rows, packed_rows):
                ratio = raw["ark_ms"] / packed["packed_ms"] if packed["packed_ms"] > 0 else float("nan")
                print(
                    f"{raw['shape']:<8}{raw['batch']:>3}{raw['heads_q']:>4}{raw['heads_kv']:>4}{raw['head_dim']:>5}"
                    f"{raw['seq_q']:>6}{raw['seq_kv']:>7}{raw['dtype']:>10}{raw['ark_ms']:>10.3f}"
                    f"{packed['packed_ms']:>12.3f}{ratio:>8.2f}x"
                )

        if args.csv and packed_rows:
            csv_path = args.csv
            if run_raw and not csv_path.endswith("_packed.csv"):
                csv_path = csv_path.replace(".csv", "_packed.csv") if args.csv.endswith(".csv") else args.csv + "_packed"
            with open(csv_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(packed_rows[0].keys()))
                writer.writeheader()
                writer.writerows(packed_rows)
            print(f"wrote {len(packed_rows)} packed-path rows to {csv_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
