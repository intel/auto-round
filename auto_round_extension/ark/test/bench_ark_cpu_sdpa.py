#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Show the speed of every non-scalar ARK CPU SDPA route against Torch FP32 SDPA.

The Torch baseline always receives the original FP32 Q/K/V tensors. ARK receives
the route's production input dtype, so the reported speedup answers the practical
question: how much faster is this route than Torch FP32 SDPA?

The packed rows time only a prepared packed-KV cache forward. They model
steady-state decode and intentionally exclude one-time K/V packing.

Run from ``auto_round_extension/ark``:

    python test/benchmark_sdpa.py
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

THREADS = 32
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = str(THREADS)

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import auto_round_kernel as ark  # noqa: E402


@dataclass(frozen=True)
class Shape:
    name: str
    batch: int
    heads_q: int
    heads_kv: int
    head_dim: int
    seq_q: int
    seq_kv: int
    is_causal: bool

    @property
    def is_gqa(self) -> bool:
        return self.heads_q != self.heads_kv

    @property
    def label(self) -> str:
        heads = str(self.heads_q) if not self.is_gqa else f"{self.heads_q}/{self.heads_kv}"
        return f"B{self.batch} H{heads} D{self.head_dim} S{self.seq_q}/{self.seq_kv}"


@dataclass(frozen=True)
class Route:
    name: str
    expected: str
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    supports_gqa: bool = True
    decode_only: bool = False
    packed_kv: bool = False


# Shapes are taken from real transformer attention layers.
# B=4–8 represents continuous batching; it is a serving workload dimension,
# not a model setting.  Decode shapes are memory-bandwidth-bound (ARK excels
# with half-precision K/V); prefill shapes are compute-bound (representative
# minimum).  GQA ratios span 2:1 through 16:1.
#
# Model attention configs:
#   Llama 3.1 8B:   Hq/Hkv/D = 32/8/128   (GQA 4:1)
#   Qwen 2.5 7B:    Hq/Hkv/D = 28/4/128   (GQA 7:1)
#   Gemma 2 27B:    Hq/Hkv/D = 32/16/128  (GQA 2:1)
#   Llama 3.1 70B:  Hq/Hkv/D = 64/8/128   (GQA 8:1)
#   DeepSeek-V3:    Hq/Hkv/D = 128/8/128  (GQA 16:1, MLA-style)
SHAPES = (
    # Decode — memory-bandwidth-bound, ARK's strength with half-precision KV.
    Shape("llama3.1-8b-decode-b1-8k", 1, 32, 8, 128, 1, 8192, False),
    Shape("qwen2.5-7b-decode-b4-8k", 4, 28, 4, 128, 1, 8192, False),
    Shape("gemma2-27b-decode-b4-8k", 4, 32, 16, 128, 1, 8192, False),
    Shape("llama3.1-70b-decode-b4-8k", 4, 64, 8, 128, 1, 8192, False),
    Shape("deepseek-decode-b4-8k", 4, 128, 8, 128, 1, 8192, False),
    # Prefill — compute-bound, representative minimum.
    Shape("llama3.1-8b-prefill-b1-1k", 1, 32, 8, 128, 1024, 1024, True),
    Shape("llama3.1-70b-prefill-b1-512", 1, 64, 8, 128, 512, 512, True),
)
ROUTES = (
    Route("mixed-raw-fp16", "mixed-raw", torch.float32, torch.float16),
    Route("mixed-raw-bf16", "mixed-raw", torch.float32, torch.bfloat16),
    Route("hom-fp16", "hom-f16", torch.float16, torch.float16),
    Route("hom-bf16", "hom-bf16", torch.bfloat16, torch.bfloat16, supports_gqa=False),
    Route("packed-fp16", "packed", torch.float32, torch.float16, decode_only=True, packed_kv=True),
    Route("packed-bf16", "packed", torch.float32, torch.bfloat16, decode_only=True, packed_kv=True),
)


def _configure_runtime() -> int:
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        # A process may inherit a narrow affinity mask. Try the first 32 online
        # CPUs before deciding the surrounding allocation cannot provide 32 CPUs.
        try:
            os.sched_setaffinity(0, set(range(THREADS)))
        except OSError:
            pass
        available = sorted(os.sched_getaffinity(0))
        if len(available) < THREADS:
            raise RuntimeError(
                f"requires {THREADS} CPUs, but the process affinity permits only {len(available)} CPU(s): {available}"
            )
        os.sched_setaffinity(0, set(available[:THREADS]))
    else:
        if (os.cpu_count() or 0) < THREADS:
            raise RuntimeError(f"requires {THREADS} CPUs, but only {os.cpu_count()} online CPU(s) are available")
    torch.set_num_threads(THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return THREADS


def _route_names() -> dict[int, str]:
    return {
        getattr(ark.cpu_lib, "ARK_CPU_SDPA_ROUTE_SCALAR", -1): "scalar",
        getattr(ark.cpu_lib, "ARK_CPU_SDPA_ROUTE_MIXED_RAW", -2): "mixed-raw",
        getattr(ark.cpu_lib, "ARK_CPU_SDPA_ROUTE_HOMOGENEOUS_FP16", -3): "hom-f16",
        getattr(ark.cpu_lib, "ARK_CPU_SDPA_ROUTE_HOMOGENEOUS_BF16", -4): "hom-bf16",
    }


def _make_qkv(shape: Shape, route: Route, seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q_fp32 = torch.randn(shape.batch, shape.heads_q, shape.seq_q, shape.head_dim, generator=generator)
    k_fp32 = torch.randn(shape.batch, shape.heads_kv, shape.seq_kv, shape.head_dim, generator=generator)
    v_fp32 = torch.randn(shape.batch, shape.heads_kv, shape.seq_kv, shape.head_dim, generator=generator)
    return q_fp32, k_fp32, v_fp32, q_fp32.to(route.q_dtype), k_fp32.to(route.kv_dtype), v_fp32.to(route.kv_dtype)


def _measure_pair(ark_call, torch_call, warmup: int, runs: int) -> tuple[float, float]:
    for _ in range(warmup):
        ark_call()
        torch_call()

    ark_samples = []
    torch_samples = []
    for index in range(runs):
        first, second = (ark_call, torch_call) if index % 2 == 0 else (torch_call, ark_call)
        start = time.perf_counter()
        first()
        middle = time.perf_counter()
        second()
        end = time.perf_counter()
        if index % 2 == 0:
            ark_samples.append(middle - start)
            torch_samples.append(end - middle)
        else:
            torch_samples.append(middle - start)
            ark_samples.append(end - middle)
    return statistics.median(ark_samples) * 1e3, statistics.median(torch_samples) * 1e3


def _run_case(shape: Shape, route: Route, route_names: dict[int, str], warmup: int, runs: int, seed: int):
    q_fp32, k_fp32, v_fp32, q, k, v = _make_qkv(shape, route, seed)
    scale = shape.head_dim**-0.5
    if route.packed_kv:
        try:
            handle = ark.internal.cpu.PackedKVHandle.create(
                shape.batch, shape.heads_kv, shape.seq_kv, shape.head_dim, dtype=route.kv_dtype
            )
            cache_k, cache_v = handle.alloc()
            handle.update(cache_k, cache_v, k, v, 0, tensor_layout="HND")
        except (AttributeError, NotImplementedError, RuntimeError, ValueError) as error:
            return None, f"{shape.name:<22}{route.name:<18}{shape.label:<24}{'unavailable':<12} {error}"

        def ark_call():
            return handle.forward(q, cache_k, cache_v, shape.seq_kv, scale=scale, tensor_layout="HND")

        resolved_name = route.expected
    else:
        resolved = ark.debug_cpu_sdpa_route(q, k, v, scale=scale, is_causal=shape.is_causal, tensor_layout="HND")
        resolved_name = route_names.get(resolved, str(resolved))
        if resolved_name != route.expected:
            return None, f"{shape.name:<22}{route.name:<18}{shape.label:<24}{resolved_name:<12} skipped"

        def ark_call():
            return ark.sdpa(q, k, v, scale=scale, is_causal=shape.is_causal, tensor_layout="HND")

    def torch_call():
        return torch.nn.functional.scaled_dot_product_attention(
            q_fp32, k_fp32, v_fp32, scale=scale, is_causal=shape.is_causal, enable_gqa=shape.is_gqa
        )

    ark_ms, torch_ms = _measure_pair(ark_call, torch_call, warmup, runs)
    return (
        torch_ms / ark_ms,
        f"{shape.name:<22}{route.name:<18}{shape.label:<24}{resolved_name:<12}"
        f"{ark_ms:>10.3f}{torch_ms:>14.3f}{torch_ms / ark_ms:>10.2f}x",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.runs <= 0:
        parser.error("--warmup must be non-negative and --runs must be positive")
    if ark.cpu_lib is None:
        print(
            "ARK CPU extension is unavailable; build auto_round_kernel_cpu before running this benchmark.",
            file=sys.stderr,
        )
        return 2

    try:
        _configure_runtime()
    except RuntimeError as error:
        print(f"benchmark configuration error: {error}", file=sys.stderr)
        return 2
    print(
        f"ARK non-scalar SDPA vs Torch FP32 SDPA | threads={THREADS} | "
        f"torch_threads={torch.get_num_threads()} | affinity={len(os.sched_getaffinity(0))} | "
        f"warmup={args.warmup} | runs={args.runs}"
    )
    print("Torch always uses the original FP32 Q/K/V; scalar fallbacks are excluded; packed K/V is prebuilt.")
    header = (
        f"{'shape':<22}{'requested route':<18}{'B/H/D/Sq/Skv':<24}{'route':<12}"
        f"{'ARK ms':>10}{'Torch FP32 ms':>14}{'speedup':>10}"
    )
    print(header)
    print("-" * len(header))

    speedups = []
    route_names = _route_names()
    for shape_index, shape in enumerate(SHAPES):
        for route_index, route in enumerate(ROUTES):
            if shape.is_gqa and not route.supports_gqa:
                continue
            if shape.is_causal and route.decode_only:
                continue
            speedup, row = _run_case(
                shape, route, route_names, args.warmup, args.runs, seed=1000 + shape_index * len(ROUTES) + route_index
            )
            print(row)
            if speedup is not None:
                speedups.append(speedup)

    if speedups:
        print("-" * len(header))
        print(f"geomean speedup: {math.exp(statistics.fmean(math.log(value) for value in speedups)):.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
