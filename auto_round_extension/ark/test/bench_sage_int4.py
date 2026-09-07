# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import auto_round_kernel


def _pack_signed_int4(values: torch.Tensor) -> torch.Tensor:
    encoded = values.to(torch.int16) & 0xF
    return (encoded[..., 0::2] | (encoded[..., 1::2] << 4)).to(torch.uint8).contiguous()


def _benchmark(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - start) * 1e3 / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pre-packed INT4 Q/K SAGE with INT8 Q/K SAGE")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if not torch.xpu.is_available():
        raise RuntimeError("XPU is not available")
    minimum_seq = 128 if args.head_dim == 64 else 256
    if args.seq < minimum_seq:
        raise ValueError(f"seq must be at least {minimum_seq} for head_dim={args.head_dim}")
    if args.seq <= 0 or args.block_size <= 0:
        raise ValueError("seq and block-size must be positive")

    torch.manual_seed(4104)
    batch, heads, seq, head_dim, block_size = args.batch, args.heads, args.seq, args.head_dim, args.block_size
    blocks = (seq + block_size - 1) // block_size
    q_logical = torch.randint(-8, 8, (batch, heads, seq, head_dim), dtype=torch.int8)
    k_logical = torch.randint(-8, 8, (batch, heads, seq, head_dim), dtype=torch.int8)
    q_packed = _pack_signed_int4(q_logical).to("xpu")
    k_packed = _pack_signed_int4(k_logical).to("xpu")
    q_int8 = q_logical.to("xpu")
    k_int8 = k_logical.to("xpu")
    value = torch.randn(batch, heads, seq, head_dim, dtype=torch.float16, device="xpu")
    qscale = torch.full((batch, heads, blocks, 1), 0.125, dtype=torch.float32, device="xpu")
    kscale = torch.full((batch, heads, blocks, 1), 0.125, dtype=torch.float32, device="xpu")
    scale = 1.0 / math.sqrt(head_dim)

    sage_s4 = lambda: auto_round_kernel.sage_s4(
        q_packed,
        k_packed,
        value,
        scale=scale,
        quant_block_size=block_size,
        qscale=qscale,
        kscale=kscale,
    )
    sage_s8 = lambda: auto_round_kernel.sage(
        q_int8,
        k_int8,
        value,
        scale=scale,
        quant_block_size=block_size,
        qscale=qscale,
        kscale=kscale,
    )

    s4_output = sage_s4()
    s8_output = sage_s8()
    torch.xpu.synchronize()
    max_diff = (s4_output.float() - s8_output.float()).abs().max().item()
    mean_diff = (s4_output.float() - s8_output.float()).abs().mean().item()
    s4_ms = _benchmark(sage_s4, args.warmup, args.iterations)
    s8_ms = _benchmark(sage_s8, args.warmup, args.iterations)

    print(
        f"shape=[{batch},{heads},{seq},{head_dim}] block_size={block_size} "
        f"s4_ms={s4_ms:.4f} s8_ms={s8_ms:.4f} "
        f"s4_vs_s8={(s4_ms / s8_ms - 1.0) * 100.0:+.2f}% "
        f"max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}"
    )


if __name__ == "__main__":
    main()
