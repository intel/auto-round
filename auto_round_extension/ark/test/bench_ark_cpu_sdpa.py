# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CPU-only ARK SDPA benchmark with fixed 32-processor runtime.

This script benchmarks the CPU paths that are actually part of the current
product surface:

1. **Public standard SDPA** (same input dtype on Q/K/V):
   - float32
   - float16
   - bfloat16

2. **Mixed decode path** (`Q=float32`, `KV=float16|bfloat16`):
   - public raw mixed SDPA (BestLA route, env-enabled inside the script)
   - internal packed-KV decode path

The packed path is benchmarked only for decode because it is a persistent KV
cache optimization, not a generic prefill/public-SDPA mode.

The script intentionally does not expose arbitrary dtype/mode combinations that
are not part of the supported benchmark matrix.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

TARGET_PROCESSORS = 32

# Fix CPU thread env before importing torch / native extensions.
os.environ["OMP_NUM_THREADS"] = str(TARGET_PROCESSORS)
os.environ["MKL_NUM_THREADS"] = str(TARGET_PROCESSORS)
os.environ["OPENBLAS_NUM_THREADS"] = str(TARGET_PROCESSORS)
os.environ["NUMEXPR_NUM_THREADS"] = str(TARGET_PROCESSORS)

import torch

# Allow running the file directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import auto_round_kernel  # noqa: E402

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

PUBLIC_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
MIXED_KV_DTYPES = (torch.float16, torch.bfloat16)
ROUTE_NAMES = {}
if getattr(auto_round_kernel, "cpu_lib", None) is not None:
    ROUTE_NAMES = {
        auto_round_kernel.cpu_lib.ARK_CPU_SDPA_ROUTE_SCALAR: "scalar",
        auto_round_kernel.cpu_lib.ARK_CPU_SDPA_ROUTE_MIXED_RAW: "mixed-raw",
        auto_round_kernel.cpu_lib.ARK_CPU_SDPA_ROUTE_HOMOGENEOUS_FP16: "hom-f16",
        auto_round_kernel.cpu_lib.ARK_CPU_SDPA_ROUTE_HOMOGENEOUS_BF16: "hom-bf16",
    }


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _route_name(route: int) -> str:
    return ROUTE_NAMES.get(route, str(route))


def _configure_runtime() -> int:
    """Pin to 32 CPUs; fall back gracefully if affinity is restricted."""
    n_online = os.cpu_count() or TARGET_PROCESSORS
    desired = min(TARGET_PROCESSORS, n_online)

    # Try to expand to the first 32 online CPUs — succeeds when the calling
    # process's cgroup / parent affinity allows it.
    try:
        os.sched_setaffinity(0, set(range(desired)))
    except (OSError, PermissionError):
        pass

    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
        pinned = min(desired, len(affinity))
        os.sched_setaffinity(0, set(affinity[:pinned]))
    else:
        pinned = min(desired, os.cpu_count() or TARGET_PROCESSORS)

    torch.set_num_threads(pinned)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if pinned < TARGET_PROCESSORS:
        print(
            f"WARNING: Only {pinned} CPU(s) available (system has {n_online}). "
            f"Run with:  taskset -c 0-{TARGET_PROCESSORS - 1} python test/bench_ark_cpu_sdpa_old.py",
            file=sys.stderr,
        )
    return pinned


@contextmanager
def _force_math_sdpa():
    if hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "sdpa_kernel"):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel([SDPBackend.MATH]):
            yield
        return
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        yield


def _make_homogeneous_qkv(batch, heads_q, heads_kv, head_dim, seq_q, seq_kv, dtype, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, heads_q, seq_q, head_dim, dtype=torch.float32, generator=gen).to(dtype)
    k = torch.randn(batch, heads_kv, seq_kv, head_dim, dtype=torch.float32, generator=gen).to(dtype)
    v = torch.randn(batch, heads_kv, seq_kv, head_dim, dtype=torch.float32, generator=gen).to(dtype)
    return q, k, v


def _make_mixed_qkv(batch, heads_q, heads_kv, head_dim, seq_q, seq_kv, kv_dtype, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, heads_q, seq_q, head_dim, dtype=torch.float32, generator=gen)
    k = torch.randn(batch, heads_kv, seq_kv, head_dim, dtype=torch.float32, generator=gen).to(kv_dtype)
    v = torch.randn(batch, heads_kv, seq_kv, head_dim, dtype=torch.float32, generator=gen).to(kv_dtype)
    return q, k, v


def _reference_sdpa(q, k, v, scale, is_causal):
    with _force_math_sdpa():
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


def _compute_tflops(batch, heads_q, seq_q, seq_kv, head_dim, time_s):
    """Compute TFLOPS for SDPA: Q@K^T + softmax + P@V = 4*B*Hq*Sq*Skv*D MACs."""
    flops = 4.0 * batch * heads_q * seq_q * seq_kv * head_dim
    return flops / time_s / 1e12 if time_s > 0 else float("nan")


def _build_cases(shape):
    if shape in ("decode", "all"):
        for batch, hq, hkv, hd, seq in DEFAULT_DECODE_SHAPES:
            yield ("decode", batch, hq, hkv, hd, seq)
    if shape in ("prefill", "all"):
        for batch, hq, hkv, hd, seq in DEFAULT_PREFILL_SHAPES:
            yield ("prefill", batch, hq, hkv, hd, seq)


def _decode_cases(shape):
    for shape_kind, batch, hq, hkv, hd, seq in _build_cases(shape):
        if shape_kind == "decode":
            yield (shape_kind, batch, hq, hkv, hd, seq)


def run_public_case(shape_kind, batch, heads_q, heads_kv, head_dim, seq, dtype, warmup, runs, atol, rtol):
    is_causal = shape_kind == "prefill"
    seq_q = 1 if shape_kind == "decode" else seq
    seq_kv = seq
    scale = 1.0 / math.sqrt(head_dim)
    q, k, v = _make_homogeneous_qkv(batch, heads_q, heads_kv, head_dim, seq_q, seq_kv, dtype)
    route = auto_round_kernel.debug_cpu_sdpa_route(q, k, v, scale=scale, is_causal=is_causal, tensor_layout="HND")

    def ark_call():
        return auto_round_kernel.sdpa(q, k, v, scale=scale, is_causal=is_causal, tensor_layout="HND")

    actual = ark_call()
    expected = _reference_sdpa(q, k, v, scale, is_causal)
    max_err = (actual.float() - expected).abs().max().item()
    passed = torch.allclose(actual.float(), expected, atol=atol, rtol=rtol)
    ark_mean, ark_best = _time_call(ark_call, warmup, runs)
    ref_mean, ref_best = _time_call(lambda: _reference_sdpa(q, k, v, scale, is_causal), warmup, runs)
    return {
        "section": "public",
        "shape": shape_kind,
        "batch": batch,
        "heads_q": heads_q,
        "heads_kv": heads_kv,
        "head_dim": head_dim,
        "seq_q": seq_q,
        "seq_kv": seq_kv,
        "dtype": _dtype_name(dtype),
        "route": _route_name(route),
        "ark_ms": ark_mean * 1e3,
        "ark_best_ms": ark_best * 1e3,
        "ref_ms": ref_mean * 1e3,
        "ref_best_ms": ref_best * 1e3,
        "ark_tflops": _compute_tflops(batch, heads_q, seq_q, seq_kv, head_dim, ark_mean),
        "speedup": ref_mean / ark_mean if ark_mean > 0 else float("nan"),
        "max_abs_err": max_err,
        "passed": passed,
    }


def run_mixed_raw_case(batch, heads_q, heads_kv, head_dim, seq_kv, kv_dtype, warmup, runs, atol, rtol):
    seq_q = 1
    scale = 1.0 / math.sqrt(head_dim)
    q, k, v = _make_mixed_qkv(batch, heads_q, heads_kv, head_dim, seq_q, seq_kv, kv_dtype)
    route = auto_round_kernel.debug_cpu_sdpa_route(q, k, v, scale=scale, tensor_layout="HND")

    def ark_call():
        return auto_round_kernel.sdpa(q, k, v, scale=scale, tensor_layout="HND")

    actual = ark_call()
    expected = _reference_sdpa(q, k, v, scale, is_causal=False)
    max_err = (actual.float() - expected).abs().max().item()
    passed = torch.allclose(actual.float(), expected, atol=atol, rtol=rtol)
    ark_mean, ark_best = _time_call(ark_call, warmup, runs)
    ref_mean, ref_best = _time_call(lambda: _reference_sdpa(q, k, v, scale, is_causal=False), warmup, runs)
    return {
        "section": "mixed_raw",
        "shape": "decode",
        "batch": batch,
        "heads_q": heads_q,
        "heads_kv": heads_kv,
        "head_dim": head_dim,
        "seq_q": seq_q,
        "seq_kv": seq_kv,
        "q_dtype": "float32",
        "kv_dtype": _dtype_name(kv_dtype),
        "route": _route_name(route),
        "ark_ms": ark_mean * 1e3,
        "ark_best_ms": ark_best * 1e3,
        "ref_ms": ref_mean * 1e3,
        "ref_best_ms": ref_best * 1e3,
        "ark_tflops": _compute_tflops(batch, heads_q, seq_q, seq_kv, head_dim, ark_mean),
        "speedup": ref_mean / ark_mean if ark_mean > 0 else float("nan"),
        "max_abs_err": max_err,
        "passed": passed,
    }


def run_packed_case(batch, heads_q, heads_kv, head_dim, seq_kv, kv_dtype, warmup, runs, atol, rtol):
    seq_q = 1
    scale = 1.0 / math.sqrt(head_dim)
    q, k, v = _make_mixed_qkv(batch, heads_q, heads_kv, head_dim, seq_q, seq_kv, kv_dtype)

    if not hasattr(auto_round_kernel, "internal") or not hasattr(auto_round_kernel.internal, "cpu"):
        raise NotImplementedError("internal.cpu namespace is unavailable")
    cache_k, cache_v = auto_round_kernel.internal.cpu.packed_kv_alloc(batch, heads_kv, seq_kv, head_dim, dtype=kv_dtype)
    auto_round_kernel.internal.cpu.update_packed_kv(cache_k, cache_v, k, v, 0, seq_kv)

    def packed_call():
        return auto_round_kernel.internal.cpu.bestla_sdpa_packed(
            q,
            cache_k,
            cache_v,
            seq_kv,
            seq_kv,
            heads_kv,
            is_causal=False,
            scale=scale,
            tensor_layout="HND",
        )

    actual = packed_call()
    expected = _reference_sdpa(q, k, v, scale, is_causal=False)
    max_err = (actual.float() - expected).abs().max().item()
    passed = torch.allclose(actual.float(), expected, atol=atol, rtol=rtol)
    packed_mean, packed_best = _time_call(packed_call, warmup, runs)
    ref_mean, ref_best = _time_call(lambda: _reference_sdpa(q, k, v, scale, is_causal=False), warmup, runs)
    return {
        "section": "packed",
        "shape": "decode",
        "batch": batch,
        "heads_q": heads_q,
        "heads_kv": heads_kv,
        "head_dim": head_dim,
        "seq_q": seq_q,
        "seq_kv": seq_kv,
        "q_dtype": "float32",
        "kv_dtype": _dtype_name(kv_dtype),
        "route": "packed",
        "packed_ms": packed_mean * 1e3,
        "packed_best_ms": packed_best * 1e3,
        "ref_ms": ref_mean * 1e3,
        "ref_best_ms": ref_best * 1e3,
        "ark_tflops": _compute_tflops(batch, heads_q, seq_q, seq_kv, head_dim, packed_mean),
        "speedup": ref_mean / packed_mean if packed_mean > 0 else float("nan"),
        "max_abs_err": max_err,
        "passed": passed,
    }


def _print_public_rows(rows):
    header = (
        f"{'shape':<8}{'B':>3}{'Hq':>4}{'Hkv':>4}{'D':>5}{'q':>6}{'kv':>7}"
        f"{'dtype':>10}{'route':>22}{'ark(ms)':>11}{'tflops':>9}{'ref(ms)':>11}{'speedup':>9}{'max_err':>11}{'ok':>4}"
    )
    print("\n[public sdpa — homogeneous/input-matched dtypes]")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['shape']:<8}{row['batch']:>3}{row['heads_q']:>4}{row['heads_kv']:>4}{row['head_dim']:>5}"
            f"{row['seq_q']:>6}{row['seq_kv']:>7}{row['dtype']:>10}{row['route']:>22}"
            f"{row['ark_ms']:>11.3f}{row['ark_tflops']:>9.3f}{row['ref_ms']:>11.3f}"
            f"{row['speedup']:>9.2f}{row['max_abs_err']:>11.2e}{('yes' if row['passed'] else 'NO'):>4}"
        )
    if rows:
        geomean = math.exp(sum(math.log(r["speedup"]) for r in rows) / len(rows))
        passed = all(r["passed"] for r in rows)
        print("-" * len(header))
        print(f"geomean speedup vs torch math SDPA: {geomean:.2f}x | parity: {'PASS' if passed else 'FAIL'}")
    return all(r["passed"] for r in rows) if rows else True


def _print_mixed_rows(rows, title, latency_key, latency_label):
    header = (
        f"{'shape':<8}{'B':>3}{'Hq':>4}{'Hkv':>4}{'D':>5}{'q':>6}{'kv':>7}"
        f"{'q_dtype':>10}{'kv_dtype':>10}{'route':>22}{latency_label:>12}{'tflops':>9}{'ref(ms)':>11}{'speedup':>9}{'max_err':>11}{'ok':>4}"
    )
    print(f"\n[{title}]")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['shape']:<8}{row['batch']:>3}{row['heads_q']:>4}{row['heads_kv']:>4}{row['head_dim']:>5}"
            f"{row['seq_q']:>6}{row['seq_kv']:>7}{row['q_dtype']:>10}{row['kv_dtype']:>10}{row['route']:>22}"
            f"{row[latency_key]:>12.3f}{row['ark_tflops']:>9.3f}{row['ref_ms']:>11.3f}{row['speedup']:>9.2f}"
            f"{row['max_abs_err']:>11.2e}{('yes' if row['passed'] else 'NO'):>4}"
        )
    if rows:
        geomean = math.exp(sum(math.log(r["speedup"]) for r in rows) / len(rows))
        passed = all(r["passed"] for r in rows)
        print("-" * len(header))
        print(f"geomean speedup vs torch math SDPA: {geomean:.2f}x | parity: {'PASS' if passed else 'FAIL'}")
    return all(r["passed"] for r in rows) if rows else True


def _print_raw_vs_packed(raw_rows, packed_rows):
    header = (
        f"{'shape':<8}{'B':>3}{'Hq':>4}{'Hkv':>4}{'D':>5}{'q':>6}{'kv':>7}"
        f"{'kv_dtype':>10}{'raw(ms)':>10}{'packed(ms)':>12}{'ratio':>8}"
    )
    print("\n[mixed raw vs packed decode]")
    print(header)
    print("-" * len(header))
    packed_index = {
        (r["batch"], r["heads_q"], r["heads_kv"], r["head_dim"], r["seq_kv"], r["kv_dtype"]): r for r in packed_rows
    }
    for raw in raw_rows:
        key = (raw["batch"], raw["heads_q"], raw["heads_kv"], raw["head_dim"], raw["seq_kv"], raw["kv_dtype"])
        packed = packed_index.get(key)
        if packed is None:
            continue
        ratio = raw["ark_ms"] / packed["packed_ms"] if packed["packed_ms"] > 0 else float("nan")
        print(
            f"{raw['shape']:<8}{raw['batch']:>3}{raw['heads_q']:>4}{raw['heads_kv']:>4}{raw['head_dim']:>5}"
            f"{raw['seq_q']:>6}{raw['seq_kv']:>7}{raw['kv_dtype']:>10}{raw['ark_ms']:>10.3f}"
            f"{packed['packed_ms']:>12.3f}{ratio:>8.2f}x"
        )


def _write_csv(path, rows):
    if not path or not rows:
        return
    fieldnames = [
        "section",
        "shape",
        "batch",
        "heads_q",
        "heads_kv",
        "head_dim",
        "seq_q",
        "seq_kv",
        "dtype",
        "q_dtype",
        "kv_dtype",
        "route",
        "ark_ms",
        "packed_ms",
        "ref_ms",
        "ark_best_ms",
        "packed_best_ms",
        "ref_best_ms",
        "speedup",
        "ark_tflops",
        "max_abs_err",
        "passed",
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {len(rows)} rows to {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape", choices=["decode", "prefill", "all"], default="all")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--csv", type=str, default="", help="Optional path to write combined results as CSV")
    args = parser.parse_args(argv)

    pinned = _configure_runtime()
    print(
        "CPU-only ARK SDPA benchmark | "
        f"target_processors={TARGET_PROCESSORS} pinned_processors={pinned} "
        f"torch_threads={torch.get_num_threads()} OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}"
    )
    print(f"shape={args.shape}")

    public_rows = []
    for dtype in PUBLIC_DTYPES:
        for shape_kind, batch, hq, hkv, hd, seq in _build_cases(args.shape):
            public_rows.append(
                run_public_case(
                    shape_kind, batch, hq, hkv, hd, seq, dtype, args.warmup, args.runs, args.atol, args.rtol
                )
            )
    all_passed = _print_public_rows(public_rows)

    mixed_raw_rows = []
    packed_rows = []
    packed_error = None
    decode_cases = list(_decode_cases(args.shape))
    if decode_cases:
        for kv_dtype in MIXED_KV_DTYPES:
            for _, batch, hq, hkv, hd, seq in decode_cases:
                mixed_raw_rows.append(
                    run_mixed_raw_case(batch, hq, hkv, hd, seq, kv_dtype, args.warmup, args.runs, args.atol, args.rtol)
                )
        all_passed = (
            _print_mixed_rows(
                mixed_raw_rows,
                "mixed raw decode — q=float32, kv=fp16/bf16",
                "ark_ms",
                "raw(ms)",
            )
            and all_passed
        )

        for kv_dtype in MIXED_KV_DTYPES:
            for _, batch, hq, hkv, hd, seq in decode_cases:
                try:
                    packed_rows.append(
                        run_packed_case(batch, hq, hkv, hd, seq, kv_dtype, args.warmup, args.runs, args.atol, args.rtol)
                    )
                except (RuntimeError, ValueError, NotImplementedError) as exc:
                    packed_error = str(exc)
                    packed_rows = []
                    break
            if packed_error is not None:
                break
        if packed_rows:
            all_passed = (
                _print_mixed_rows(
                    packed_rows,
                    "packed kv decode — q=float32, kv=fp16/bf16",
                    "packed_ms",
                    "packed(ms)",
                )
                and all_passed
            )
            _print_raw_vs_packed(mixed_raw_rows, packed_rows)
        else:
            print("\n[packed kv decode — q=float32, kv=fp16/bf16]")
            print(f"unavailable: {packed_error or 'packed path is not available on this ISA/build'}")

    combined_rows = public_rows + mixed_raw_rows + packed_rows
    _write_csv(args.csv, combined_rows)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
