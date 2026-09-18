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

"""Accuracy and benchmark comparison for MXFP4-weight MoE activation formats."""

import argparse
import statistics
from dataclasses import dataclass

import auto_round_kernel as ark
import torch
from auto_round_kernel.mxfp4_hadamard import mxfp4_hadamard_quant_reference


FP4_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


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
    Shape("decode", "decode_all_E8_M8_N4096_K4096", 8, (1,) * 8, 4096, 4096, 80),
    Shape("decode", "decode_all_E8_M8_N4096_K14336", 8, (1,) * 8, 4096, 14336, 40),
]


ACCURACY_CASES = [
    Shape("accuracy", "accuracy_E3_M6_N64_K64", 3, (1, 3, 2), 64, 64, 1),
    Shape("accuracy", "accuracy_E4_M16_N128_K128", 4, (4, 2, 6, 4), 128, 128, 1),
]


def unpack_mxfp4_to_float(packed: torch.Tensor, scales: torch.Tensor, k: int) -> torch.Tensor:
    experts, n, _ = packed.shape
    raw = packed.cpu().to(torch.int64)
    low = raw & 0x0F
    high = (raw >> 4) & 0x0F
    nibbles = torch.empty((experts, n, k), dtype=torch.int64)
    nibbles[..., 0::2] = low
    nibbles[..., 1::2] = high
    mag = nibbles & 0x7
    sign = torch.where((nibbles & 0x8) != 0, -1.0, 1.0)
    values = FP4_E2M1[mag] * sign
    scale_values = torch.ldexp(torch.ones_like(scales.cpu(), dtype=torch.float32), scales.cpu().to(torch.int32) - 127)
    return values * scale_values.repeat_interleave(32, dim=2)


def unpack_mxfp4_activation_to_float(packed: torch.Tensor, scales: torch.Tensor, k: int) -> torch.Tensor:
    total_tokens, _ = packed.shape
    raw = packed.cpu().to(torch.int64)
    low = raw & 0x0F
    high = (raw >> 4) & 0x0F
    nibbles = torch.empty((total_tokens, k), dtype=torch.int64)
    nibbles[:, 0::2] = low
    nibbles[:, 1::2] = high
    mag = nibbles & 0x7
    sign = torch.where((nibbles & 0x8) != 0, -1.0, 1.0)
    values = FP4_E2M1[mag] * sign
    scale_values = torch.ldexp(torch.ones_like(scales.cpu(), dtype=torch.float32), scales.cpu().to(torch.int32) - 127)
    return values * scale_values.repeat_interleave(32, dim=1)


def reference_moe(activations: torch.Tensor, dequant_weights_nk: torch.Tensor, ntpe: torch.Tensor) -> torch.Tensor:
    total_tokens, _ = activations.shape
    experts, n, _ = dequant_weights_nk.shape
    output = torch.empty((total_tokens, n), dtype=torch.float32)
    act_cpu = activations.float().cpu()
    weights_cpu = dequant_weights_nk.float().cpu()
    offset = 0
    for expert in range(experts):
        count = int(ntpe.cpu()[expert].item())
        if count:
            output[offset : offset + count] = act_cpu[offset : offset + count] @ weights_cpu[expert].t()
        offset += count
    return output


def reference_mxfp8_mxfp4(
    activations: torch.Tensor,
    activation_scales: torch.Tensor,
    weights: torch.Tensor,
    weight_scales: torch.Tensor,
    ntpe: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    _, k = activations.shape
    act = activations.float().cpu()
    scale_values = torch.ldexp(
        torch.ones_like(activation_scales.cpu(), dtype=torch.float32), activation_scales.cpu().to(torch.int32) - 127
    )
    act = act * scale_values.repeat_interleave(32, dim=1)
    weights_fp = unpack_mxfp4_to_float(weights, weight_scales, k)
    return reference_moe(act.to(output_dtype), weights_fp.to(output_dtype), ntpe).to(output_dtype).float()


def reference_mxfp4_mxfp4(
    activations: torch.Tensor,
    activation_scales: torch.Tensor,
    weights: torch.Tensor,
    weight_scales: torch.Tensor,
    ntpe: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    _, k_packed = activations.shape
    k = k_packed * 2
    act = unpack_mxfp4_activation_to_float(activations, activation_scales, k)
    weights_fp = unpack_mxfp4_to_float(weights, weight_scales, k)
    return reference_moe(act.to(output_dtype), weights_fp.to(output_dtype), ntpe).to(output_dtype).float()


def reference_hmt_mxfp4_mxfp4(
    activations: torch.Tensor,
    weights: torch.Tensor,
    weight_scales: torch.Tensor,
    ntpe: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    activation_codes, activation_scales = mxfp4_hadamard_quant_reference(activations.cpu())
    return reference_mxfp4_mxfp4(activation_codes, activation_scales, weights, weight_scales, ntpe, output_dtype)


def reference_bf16_mxfp4(
    activations: torch.Tensor,
    weights: torch.Tensor,
    weight_scales: torch.Tensor,
    ntpe: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    _, k = activations.shape
    weights_fp = unpack_mxfp4_to_float(weights, weight_scales, k)
    return reference_moe(activations.to(output_dtype), weights_fp.to(output_dtype), ntpe).to(output_dtype).float()


def event_time_ms(fn, iterations: int, warmup: int) -> float:
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


def time_stats(fn, iterations: int, warmup: int, repeats: int) -> tuple[float, float]:
    samples = [event_time_ms(fn, iterations, warmup) for _ in range(repeats)]
    return statistics.median(samples), min(samples)


def make_common_inputs(shape: Shape, fp8_dtype: torch.dtype):
    total_tokens = sum(shape.tokens_per_expert)
    base = (torch.randn(total_tokens, shape.k, device="xpu", dtype=torch.float16) * 0.25).clamp(-2, 2)
    activations_bf16 = base.to(torch.bfloat16)
    activations_mxfp8 = base.to(fp8_dtype)
    activations_mxfp4 = torch.randint(0, 256, (total_tokens, shape.k // 2), device="xpu", dtype=torch.uint8)
    activation_scales = torch.randint(126, 129, (total_tokens, shape.k // 32), device="xpu", dtype=torch.uint8)
    weights = torch.randint(0, 256, (shape.experts, shape.n, shape.k // 2), device="xpu", dtype=torch.uint8)
    weight_scales = torch.randint(126, 129, (shape.experts, shape.n, shape.k // 32), device="xpu", dtype=torch.uint8)
    ntpe = torch.tensor(shape.tokens_per_expert, device="xpu", dtype=torch.int32)
    return activations_bf16, activations_mxfp8, activations_mxfp4, activation_scales, weights, weight_scales, ntpe


def run_accuracy(fp8_dtype: torch.dtype, output_dtype: torch.dtype) -> None:
    print("accuracy,label,kernel,fp8_dtype,output_dtype,max_abs_diff,mean_abs_diff")
    for shape in ACCURACY_CASES:
        torch.manual_seed(2026)
        torch.xpu.manual_seed_all(2026)
        act_bf16, act_mxfp8, act_mxfp4, act_scales, weights, weight_scales, ntpe = make_common_inputs(shape, fp8_dtype)

        out_mxfp8 = ark.moe_gemm_prefill_mxfp8_mxfp4(
            act_mxfp8, act_scales, weights, weight_scales, ntpe, output_dtype=output_dtype, group_size=32
        )
        ref_mxfp8 = reference_mxfp8_mxfp4(act_mxfp8, act_scales, weights, weight_scales, ntpe, output_dtype)
        diff_mxfp8 = (out_mxfp8.float().cpu() - ref_mxfp8).abs()
        print(
            f"accuracy,{shape.label},mxfp8_mxfp4,{str(fp8_dtype).split('.')[-1]},{str(output_dtype).split('.')[-1]},"
            f"{diff_mxfp8.max().item():.6f},{diff_mxfp8.mean().item():.6f}"
        )

        out_mxfp4 = ark.moe_gemm_prefill_mxfp4_mxfp4(
            act_mxfp4, act_scales, weights, weight_scales, ntpe, output_dtype=output_dtype, group_size=32
        )
        ref_mxfp4 = reference_mxfp4_mxfp4(act_mxfp4, act_scales, weights, weight_scales, ntpe, output_dtype)
        diff_mxfp4 = (out_mxfp4.float().cpu() - ref_mxfp4).abs()
        print(
            f"accuracy,{shape.label},mxfp4_mxfp4,{str(fp8_dtype).split('.')[-1]},{str(output_dtype).split('.')[-1]},"
            f"{diff_mxfp4.max().item():.6f},{diff_mxfp4.mean().item():.6f}"
        )

        if output_dtype == torch.bfloat16:
            out_hmt = ark.moe_gemm_prefill_hmt_mxfp4_mxfp4(
                act_bf16, weights, weight_scales, ntpe, output_dtype=torch.bfloat16, group_size=32
            )
            ref_hmt = reference_hmt_mxfp4_mxfp4(act_bf16, weights, weight_scales, ntpe, torch.bfloat16)
            diff_hmt = (out_hmt.float().cpu() - ref_hmt).abs()
            print(
                f"accuracy,{shape.label},hmt_mxfp4_mxfp4,{str(fp8_dtype).split('.')[-1]},bfloat16,"
                f"{diff_hmt.max().item():.6f},{diff_hmt.mean().item():.6f}"
            )

        out_bf16 = ark.moe_gemm_prefill(
            act_bf16,
            weights,
            ntpe,
            scales=weight_scales,
            weight_bits=4,
            group_size=32,
            asym=False,
            scale_dtype="fp8_e8m0",
        )
        ref_bf16 = reference_bf16_mxfp4(act_bf16, weights, weight_scales, ntpe, torch.bfloat16)
        diff_bf16 = (out_bf16.float().cpu() - ref_bf16).abs()
        print(
            f"accuracy,{shape.label},bf16_mxfp4,{str(fp8_dtype).split('.')[-1]},bfloat16,"
            f"{diff_bf16.max().item():.6f},{diff_bf16.mean().item():.6f}"
        )


def flops(shape: Shape) -> int:
    return 2 * sum(shape.tokens_per_expert) * shape.n * shape.k


def active_experts(shape: Shape) -> int:
    return sum(1 for value in shape.tokens_per_expert if value > 0)


def mxfp8_packed_bytes(shape: Shape, output_dtype: torch.dtype) -> int:
    total_tokens = sum(shape.tokens_per_expert)
    out_bytes = torch.empty((), dtype=output_dtype).element_size()
    return (
        total_tokens * shape.k
        + total_tokens * (shape.k // 32)
        + active_experts(shape) * shape.n * (shape.k // 2)
        + active_experts(shape) * shape.n * (shape.k // 32)
        + total_tokens * shape.n * out_bytes
    )


def mxfp4_packed_bytes(shape: Shape, output_dtype: torch.dtype) -> int:
    total_tokens = sum(shape.tokens_per_expert)
    out_bytes = torch.empty((), dtype=output_dtype).element_size()
    return (
        total_tokens * (shape.k // 2)
        + total_tokens * (shape.k // 32)
        + active_experts(shape) * shape.n * (shape.k // 2)
        + active_experts(shape) * shape.n * (shape.k // 32)
        + total_tokens * shape.n * out_bytes
    )


def bf16_packed_bytes(shape: Shape) -> int:
    total_tokens = sum(shape.tokens_per_expert)
    out_bytes = torch.empty((), dtype=torch.bfloat16).element_size()
    return (
        total_tokens * shape.k * out_bytes
        + active_experts(shape) * shape.n * (shape.k // 2)
        + active_experts(shape) * shape.n * (shape.k // 32)
        + total_tokens * shape.n * out_bytes
    )


def mxfp8_workspace_bytes(shape: Shape, output_dtype: torch.dtype) -> int:
    total_tokens = sum(shape.tokens_per_expert)
    out_bytes = torch.empty((), dtype=output_dtype).element_size()
    return (
        mxfp8_packed_bytes(shape, output_dtype)
        + 2 * total_tokens * shape.k * out_bytes
        + 2 * active_experts(shape) * shape.k * shape.n * out_bytes
    )


def mxfp4_workspace_bytes(shape: Shape, output_dtype: torch.dtype) -> int:
    total_tokens = sum(shape.tokens_per_expert)
    scale_groups = shape.k // 32
    return (
        mxfp4_packed_bytes(shape, output_dtype)
        + (total_tokens + 3 * shape.experts) * scale_groups
        + active_experts(shape) * shape.n * scale_groups
    )


def hmt_mxfp4_packed_bytes(shape: Shape) -> int:
    total_tokens = sum(shape.tokens_per_expert)
    out_bytes = torch.empty((), dtype=torch.bfloat16).element_size()
    return (
        total_tokens * shape.k * torch.empty((), dtype=torch.bfloat16).element_size()
        + active_experts(shape) * shape.n * (shape.k // 2)
        + active_experts(shape) * shape.n * (shape.k // 32)
        + total_tokens * shape.n * out_bytes
    )


def hmt_mxfp4_workspace_bytes(shape: Shape) -> int:
    total_tokens = sum(shape.tokens_per_expert)
    scale_groups = shape.k // 32
    return (
        hmt_mxfp4_packed_bytes(shape)
        + 2 * total_tokens * (shape.k // 2)
        + 2 * total_tokens * scale_groups
        + (total_tokens + 3 * shape.experts) * scale_groups
        + active_experts(shape) * shape.n * scale_groups
    )


def bf16_workspace_bytes(shape: Shape) -> int:
    out_bytes = torch.empty((), dtype=torch.bfloat16).element_size()
    return bf16_packed_bytes(shape) + 2 * active_experts(shape) * shape.k * shape.n * out_bytes


def benchmark(args: argparse.Namespace, fp8_dtype: torch.dtype, output_dtype: torch.dtype) -> None:
    print(
        "perf,phase,label,kernel,fp8_dtype,output_dtype,M,E,active_E,N,K,iters,median_ms,best_ms,"
        "tflops_median,tflops_best,packed_GBps_median,packed_GBps_best,workspace_GBps_median,workspace_GBps_best"
    )
    for shape in SHAPES:
        torch.manual_seed(123)
        torch.xpu.manual_seed_all(123)
        act_bf16, act_mxfp8, act_mxfp4, act_scales, weights, weight_scales, ntpe = make_common_inputs(shape, fp8_dtype)

        def run_mxfp8():
            return ark.moe_gemm_prefill_mxfp8_mxfp4(
                act_mxfp8, act_scales, weights, weight_scales, ntpe, output_dtype=output_dtype, group_size=32
            )

        def run_bf16():
            return ark.moe_gemm_prefill(
                act_bf16,
                weights,
                ntpe,
                scales=weight_scales,
                weight_bits=4,
                group_size=32,
                asym=False,
                scale_dtype="fp8_e8m0",
            )

        def run_mxfp4():
            return ark.moe_gemm_prefill_mxfp4_mxfp4(
                act_mxfp4, act_scales, weights, weight_scales, ntpe, output_dtype=output_dtype, group_size=32
            )

        kernels = [
            (
                "mxfp8_mxfp4",
                run_mxfp8,
                lambda current_shape: mxfp8_packed_bytes(current_shape, output_dtype),
                lambda current_shape: mxfp8_workspace_bytes(current_shape, output_dtype),
                str(output_dtype).split(".")[-1],
            ),
            (
                "mxfp4_mxfp4",
                run_mxfp4,
                lambda current_shape: mxfp4_packed_bytes(current_shape, output_dtype),
                lambda current_shape: mxfp4_workspace_bytes(current_shape, output_dtype),
                str(output_dtype).split(".")[-1],
            ),
        ]
        if output_dtype == torch.bfloat16:
            kernels.append(
                (
                    "hmt_mxfp4_mxfp4",
                    lambda: ark.moe_gemm_prefill_hmt_mxfp4_mxfp4(
                        act_bf16, weights, weight_scales, ntpe, output_dtype=torch.bfloat16, group_size=32
                    ),
                    hmt_mxfp4_packed_bytes,
                    hmt_mxfp4_workspace_bytes,
                    "bfloat16",
                )
            )
        kernels.append(
            (
                "bf16_mxfp4",
                run_bf16,
                bf16_packed_bytes,
                bf16_workspace_bytes,
                "bfloat16",
            ),
        )
        for kernel_name, fn, byte_fn, workspace_fn, dtype_name in kernels:
            median_ms, best_ms = time_stats(fn, shape.iterations, args.warmup, args.repeats)
            median_s = median_ms / 1000.0
            best_s = best_ms / 1000.0
            op_flops = flops(shape)
            packed_bytes = byte_fn(shape)
            workspace_bytes = workspace_fn(shape)
            print(
                f"perf,{shape.phase},{shape.label},{kernel_name},{str(fp8_dtype).split('.')[-1]},{dtype_name},"
                f"{sum(shape.tokens_per_expert)},{shape.experts},{active_experts(shape)},{shape.n},{shape.k},"
                f"{shape.iterations},{median_ms:.4f},{best_ms:.4f},"
                f"{op_flops / median_s / 1e12:.4f},{op_flops / best_s / 1e12:.4f},"
                f"{packed_bytes / median_s / 1e9:.2f},{packed_bytes / best_s / 1e9:.2f},"
                f"{workspace_bytes / median_s / 1e9:.2f},{workspace_bytes / best_s / 1e9:.2f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--fp8", choices=("e4m3", "e5m2"), default="e4m3")
    parser.add_argument("--output", choices=("bf16", "fp16"), default="bf16")
    args = parser.parse_args()

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("torch.xpu is not available")
    if not hasattr(ark, "moe_gemm_prefill_mxfp8_mxfp4"):
        raise RuntimeError("auto_round_kernel lacks moe_gemm_prefill_mxfp8_mxfp4")
    if ark.xpu_lib is None or not hasattr(ark.xpu_lib, "moe_gemm_prefill_mxfp8_mxfp4"):
        raise RuntimeError("auto_round_kernel XPU extension lacks moe_gemm_prefill_mxfp8_mxfp4")
    if not hasattr(ark, "moe_gemm_prefill_mxfp4_mxfp4"):
        raise RuntimeError("auto_round_kernel lacks moe_gemm_prefill_mxfp4_mxfp4")
    if ark.xpu_lib is None or not hasattr(ark.xpu_lib, "moe_gemm_prefill_mxfp4_mxfp4"):
        raise RuntimeError("auto_round_kernel XPU extension lacks moe_gemm_prefill_mxfp4_mxfp4")
    if not hasattr(ark, "moe_gemm_prefill_hmt_mxfp4_mxfp4"):
        raise RuntimeError("auto_round_kernel lacks moe_gemm_prefill_hmt_mxfp4_mxfp4")
    if ark.xpu_lib is None or not hasattr(ark.xpu_lib, "moe_gemm_prefill_hmt_mxfp4_mxfp4"):
        raise RuntimeError("auto_round_kernel XPU extension lacks moe_gemm_prefill_hmt_mxfp4_mxfp4")

    fp8_dtype = torch.float8_e4m3fn if args.fp8 == "e4m3" else torch.float8_e5m2
    output_dtype = torch.bfloat16 if args.output == "bf16" else torch.float16
    print(f"device={torch.xpu.get_device_name(0)}")
    print(f"ark={ark.__file__}")
    print(f"fp8_dtype={fp8_dtype} output_dtype={output_dtype}")
    run_accuracy(fp8_dtype, output_dtype)
    benchmark(args, fp8_dtype, output_dtype)


if __name__ == "__main__":
    main()
