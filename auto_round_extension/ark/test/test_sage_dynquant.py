# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""Benchmark for DynQuant SAGE with block-wise INT8 quantization.

Tests sage_dynquant with different quant_block_size values (per-token, 32, 64, 128, 256).
Compares accuracy and latency against SAGE v1 and FP16 SDPA references.
"""

import math
import time

import auto_round_kernel
import torch

ark = None


def get_ark():
    global ark
    if ark is None:
        ark = auto_round_kernel
    return ark


def bench(fn, n_warmup=50, n_runs=100):
    """Benchmark helper: returns average time in ms."""
    for _ in range(n_warmup):
        fn()
    torch.xpu.synchronize()
    st = time.time()
    for _ in range(n_runs):
        fn()
    torch.xpu.synchronize()
    return (time.time() - st) / n_runs * 1e3


def run_benchmark(batch=1, seq=4096, h_q=30, h_kv=30, H=128, H_v=128, dt=torch.float16, is_causal=False, dev="xpu"):
    # Align seq to largest block_size for divisibility
    block_sizes = [1, 32, 64, 128, 256]
    max_bs = max(block_sizes)
    seq = seq // max_bs * max_bs
    q = torch.randn(batch, h_q, seq, H, dtype=dt, device=dev)
    k = torch.randn(batch, h_kv, seq, H, dtype=dt, device=dev)
    v = torch.randn(batch, h_kv, seq, H_v, dtype=dt, device=dev)
    scale = 1 / math.sqrt(H)

    print("=== SageDynQuant Block-wise Benchmark ===")
    print(f"Batch:{batch} Seq:{seq} HeadQ:{h_q} HeadKV:{h_kv} HDim:{H} Causal:{is_causal}")
    print()

    # --- FP16 reference ---
    ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, scale=scale, is_causal=is_causal, enable_gqa=(h_q // h_kv) > 1
    )

    # --- Test different block sizes ---
    results = {}

    for bs in block_sizes:
        out = get_ark().sage_dynquant(q, k, v, scale=scale, is_causal=is_causal, quant_block_size=bs)
        dff = (ref - out).abs()
        t = bench(lambda bs=bs: get_ark().sage_dynquant(q, k, v, scale=scale, is_causal=is_causal, quant_block_size=bs))
        results[bs] = (t, dff.max().item(), dff.mean().item())
        print(
            f"  SageDynQuan  block_size={bs:3d}: {t:8.1f} ms  diff max={dff.max().item():.6f} mean={dff.mean().item():.6f}"
        )

    # --- SAGE v1 high-level API comparison: PV half vs PV int8/PVS8 ---
    sagev1_pvhalf_results = {}
    sagev1_pvs8_results = {}
    for bs in block_sizes:
        pvhalf_out = get_ark().sagev1(q, k, v, scale=scale, is_causal=is_causal, quant_block_size=bs)
        pvhalf_diff = (ref - pvhalf_out).abs()
        pvhalf_t = bench(lambda bs=bs: get_ark().sagev1(q, k, v, scale=scale, is_causal=is_causal, quant_block_size=bs))
        sagev1_pvhalf_results[bs] = (pvhalf_t, pvhalf_diff.max().item(), pvhalf_diff.mean().item())
        print(
            f"  SAGE v1 pvhalf block_size={bs:3d}: {pvhalf_t:8.1f} ms  "
            f"diff max={pvhalf_diff.max().item():.6f} mean={pvhalf_diff.mean().item():.6f}"
        )

        pvs8_out = get_ark().sagev1_pvi8(q, k, v, scale=scale, is_causal=is_causal, quant_block_size=bs)
        pvs8_diff = (ref - pvs8_out).abs()
        pvs8_t = bench(
            lambda bs=bs: get_ark().sagev1_pvi8(q, k, v, scale=scale, is_causal=is_causal, quant_block_size=bs)
        )
        sagev1_pvs8_results[bs] = (pvs8_t, pvs8_diff.max().item(), pvs8_diff.mean().item())
        print(
            f"  SAGE v1 pvs8   block_size={bs:3d}: {pvs8_t:8.1f} ms  "
            f"diff max={pvs8_diff.max().item():.6f} mean={pvs8_diff.mean().item():.6f}"
        )

    # --- SAGE v1 reference for each block_size (pre-quantized INT8) ---
    sage_results = {}
    for bs in block_sizes:
        q_i8 = torch.randint(-128, 127, (batch, h_q, seq, H), dtype=torch.int8, device=dev)
        k_i8 = torch.randint(-128, 127, (batch, h_kv, seq, H), dtype=torch.int8, device=dev)
        qs = torch.randn(batch, h_q, seq // bs, 1, dtype=torch.float32, device=dev) / 100 + 0.001
        ks = torch.randn(batch, h_kv, seq // bs, 1, dtype=torch.float32, device=dev) / 100 + 0.001
        sage_t = bench(
            lambda bs=bs, q_i8=q_i8, k_i8=k_i8, qs=qs, ks=ks: get_ark().sage(
                q_i8, k_i8, v, quant_block_size=bs, qscale=qs, kscale=ks, scale=scale, is_causal=is_causal
            )
        )
        sage_results[bs] = sage_t
        print(f"  SAGE v1 bs={bs:3d}:  {sage_t:8.1f} ms")

    print()
    print("--- Summary (dynquant / sagev1 pvhalf / sagev1 pvs8 / pre-quantized SAGE at same block_size) ---")
    for bs in block_sizes:
        t, mx, mn = results[bs]
        pvhalf_t, pvhalf_mx, pvhalf_mn = sagev1_pvhalf_results[bs]
        pvs8_t, pvs8_mx, pvs8_mn = sagev1_pvs8_results[bs]
        sage_t = sage_results[bs]
        overhead = (t / sage_t - 1) * 100
        pvs8_over_pvhalf = (pvs8_t / pvhalf_t - 1) * 100
        print(
            f"  block_size={bs:3d}: dynquant {t:8.1f} ms  "
            f"pvhalf {pvhalf_t:8.1f} ms  pvs8 {pvs8_t:8.1f} ms  "
            f"prequant_sage {sage_t:8.1f} ms  "
            f"dynquant_vs_prequant={overhead:+.0f}%  pvs8_vs_pvhalf={pvs8_over_pvhalf:+.0f}%"
        )


if __name__ == "__main__":
    run_benchmark()
    run_benchmark(is_causal=True)
