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

"""Benchmark MXFP8 activation x MXFP4 weight MoE on XPU."""

import argparse
import statistics
from dataclasses import dataclass

import auto_round_kernel as ark
import torch


@dataclass(frozen=True)
class Shape:
    phase: str
    label: str
    experts: int
    tokens_per_expert: tuple[int, ...]
    n: int
    k: int
    iterations: int


SHAPES = [
    Shape("prefill", "prefill_E8_M512_N4096_K4096", 8, (64,) * 8, 4096, 4096, 20),
    Shape("prefill", "prefill_E8_M512_N4096_K14336", 8, (64,) * 8, 4096, 14336, 10),
    Shape("decode", "decode_top2_E8_M2_N4096_K4096", 8, (1, 1, 0, 0, 0, 0, 0, 0), 4096, 4096, 160),
    Shape("decode", "decode_top2_E8_M2_N4096_K14336", 8, (1, 1, 0, 0, 0, 0, 0, 0), 4096, 14336, 80),
    Shape("decode", "decode_shape_E8_M8_N4096_K4096", 8, (1,) * 8, 4096, 4096, 80),
    Shape("decode", "decode_shape_E8_M8_N4096_K14336", 8, (1,) * 8, 4096, 14336, 40),
]


def _event_time_ms(fn, iterations: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()

    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.xpu.synchronize()
    return start.elapsed_time(end) / iterations


def _stats_time_ms(fn, iterations: int, warmup: int, repeats: int) -> tuple[float, float]:
    samples = [_event_time_ms(fn, iterations, warmup) for _ in range(repeats)]
    return statistics.median(samples), min(samples)


def _make_inputs(shape: Shape, fp8_dtype: torch.dtype, output_dtype: torch.dtype):
    total_tokens = sum(shape.tokens_per_expert)
    activations_fp = (torch.randn(total_tokens, shape.k, device="xpu", dtype=torch.float16) * 0.25).clamp(-2, 2)
    activations = activations_fp.to(fp8_dtype)
    activation_scales = torch.randint(126, 129, (total_tokens, shape.k // 32), device="xpu", dtype=torch.uint8)
    weights = torch.randint(0, 256, (shape.experts, shape.n, shape.k // 2), device="xpu", dtype=torch.uint8)
    weight_scales = torch.randint(126, 129, (shape.experts, shape.n, shape.k // 32), device="xpu", dtype=torch.uint8)
    ntpe = torch.tensor(shape.tokens_per_expert, device="xpu", dtype=torch.int32)

    def run():
        return ark.moe_gemm_prefill_mxfp8_mxfp4(
            activations,
            activation_scales,
            weights,
            weight_scales,
            ntpe,
            output_dtype=output_dtype,
            group_size=32,
        )

    return run


def _packed_bytes(shape: Shape, output_dtype: torch.dtype) -> int:
    total_tokens = sum(shape.tokens_per_expert)
    active_experts = sum(1 for value in shape.tokens_per_expert if value > 0)
    output_bytes = torch.empty((), dtype=output_dtype).element_size()
    return (
        total_tokens * shape.k
        + total_tokens * (shape.k // 32)
        + active_experts * shape.n * (shape.k // 2)
        + active_experts * shape.n * (shape.k // 32)
        + total_tokens * shape.n * output_bytes
    )


def _workspace_bytes(shape: Shape, output_dtype: torch.dtype) -> int:
    total_tokens = sum(shape.tokens_per_expert)
    active_experts = sum(1 for value in shape.tokens_per_expert if value > 0)
    output_bytes = torch.empty((), dtype=output_dtype).element_size()
    return (
        _packed_bytes(shape, output_dtype)
        + total_tokens * shape.k * output_bytes
        + active_experts * shape.k * shape.n * output_bytes
        + total_tokens * shape.k * output_bytes
        + active_experts * shape.k * shape.n * output_bytes
    )


def _flops(shape: Shape) -> int:
    return 2 * sum(shape.tokens_per_expert) * shape.n * shape.k


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("torch.xpu is not available")
    if not hasattr(ark, "moe_gemm_prefill_mxfp8_mxfp4"):
        raise RuntimeError("auto_round_kernel Python API lacks moe_gemm_prefill_mxfp8_mxfp4")
    if ark.xpu_lib is None or not hasattr(ark.xpu_lib, "moe_gemm_prefill_mxfp8_mxfp4"):
        raise RuntimeError("auto_round_kernel XPU extension lacks moe_gemm_prefill_mxfp8_mxfp4")

    print(f"device={torch.xpu.get_device_name(0)}")
    print(f"ark={ark.__file__}")
    print("phase,label,fp8_dtype,output_dtype,M,E,N,K,iters,median_ms,best_ms,tflops_median,tflops_best,packed_GBps_median,packed_GBps_best,workspace_GBps_median,workspace_GBps_best")

    for shape in SHAPES:
        if shape.k % 32 != 0 or shape.n % 16 != 0:
            continue
        for fp8_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            for output_dtype in (torch.float16, torch.bfloat16):
                torch.manual_seed(123)
                torch.xpu.manual_seed_all(123)
                run = _make_inputs(shape, fp8_dtype, output_dtype)
                median_ms, best_ms = _stats_time_ms(run, shape.iterations, args.warmup, args.repeats)
                flops = _flops(shape)
                packed_bytes = _packed_bytes(shape, output_dtype)
                workspace_bytes = _workspace_bytes(shape, output_dtype)
                median_s = median_ms / 1000.0
                best_s = best_ms / 1000.0
                print(
                    f"{shape.phase},{shape.label},{str(fp8_dtype).split('.')[-1]},{str(output_dtype).split('.')[-1]},"
                    f"{sum(shape.tokens_per_expert)},{shape.experts},{shape.n},{shape.k},{shape.iterations},"
                    f"{median_ms:.4f},{best_ms:.4f},"
                    f"{flops / median_s / 1e12:.4f},{flops / best_s / 1e12:.4f},"
                    f"{packed_bytes / median_s / 1e9:.2f},{packed_bytes / best_s / 1e9:.2f},"
                    f"{workspace_bytes / median_s / 1e9:.2f},{workspace_bytes / best_s / 1e9:.2f}"
                )


if __name__ == "__main__":
    main()
