# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""Benchmark our FP16 SDPA and INT8 sage_dynquant against reference sage_attention_page on BMG.

Configurations extracted from the reference performance table:
  batch=1, q_seq=4096, k_seq in {8192, 16384, 32768},
  q_head in {16, 32, 64, 80, 96, 128}, kv_head=8,
  head_dim in {64, 128}, block_size=512, dtype=float16, prefill (non-causal)
"""

import csv
import math
import sys
import time

import auto_round_kernel
import torch

ark = None


def get_ark():
    global ark
    if ark is None:
        ark = auto_round_kernel.ARK()
    return ark


def bench(fn, n_warmup=50, n_runs=100):
    for _ in range(n_warmup):
        fn()
    torch.xpu.synchronize()
    st = time.time()
    for _ in range(n_runs):
        fn()
    torch.xpu.synchronize()
    return (time.time() - st) / n_runs


# Reference numbers from the other team's sage_attention_page benchmark (latency in us)
REF_LATENCY = {
    # (q_head, head_dim, k_seq): latency_us
    (96, 128, 8192): 15959.507,
    (96, 128, 16384): 37002.037,
    (96, 128, 32768): 81393.724,
    (96, 64, 8192): 12055.134,
    (96, 64, 16384): 28907.408,
    (96, 64, 32768): 63472.798,
    (80, 128, 8192): 13501.044,
    (80, 128, 16384): 34071.083,
    (80, 128, 32768): 68894.799,
    (80, 64, 8192): 10320.806,
    (80, 64, 16384): 24004.720,
    (80, 64, 32768): 52323.563,
    (128, 128, 8192): 20964.512,
    (128, 128, 16384): 49097.406,
    (128, 128, 32768): 105522.442,
    (128, 64, 8192): 15976.130,
    (128, 64, 16384): 38390.276,
    (128, 64, 32768): 85014.886,
    (64, 128, 8192): 10555.244,
    (64, 128, 16384): 24920.573,
    (64, 128, 32768): 53484.178,
    (64, 64, 8192): 8117.160,
    (64, 64, 16384): 19467.069,
    (64, 64, 32768): 42681.761,
    (32, 128, 8192): 5704.147,
    (32, 128, 16384): 12386.491,
    (32, 128, 32768): 27239.501,
    (32, 64, 8192): 4081.240,
    (32, 64, 16384): 9875.475,
    (32, 64, 32768): 22412.057,
    (16, 128, 8192): 2874.092,
    (16, 128, 16384): 6634.709,
    (16, 128, 32768): 14326.504,
    (16, 64, 8192): 2246.905,
    (16, 64, 16384): 5303.178,
    (16, 64, 32768): 11535.505,
}


# Arc Pro B60: 160 XVEs × 512 FP16 OPs/XVE/clock × 2.4 GHz ≈ 196.6 TFLOPS
# Cross-validated with reference table: 78 TFLOPS / 40% MFU ≈ 195 TFLOPS
BMG_PEAK_TFLOPS_INT8 = 196.6
BMG_PEAK_TFLOPS_FP16 = BMG_PEAK_TFLOPS_INT8 / 2

# All configurations from the table
CONFIGS = [
    # (batch, q_seq, k_seq, q_head, kv_head, head_dim, block_size)
    (1, 4096, 8192, 96, 8, 128, 128),
    (1, 4096, 16384, 96, 8, 128, 128),
    (1, 4096, 32768, 96, 8, 128, 128),
    (1, 4096, 8192, 96, 8, 64, 128),
    (1, 4096, 16384, 96, 8, 64, 128),
    (1, 4096, 32768, 96, 8, 64, 128),
    (1, 4096, 8192, 80, 8, 128, 128),
    (1, 4096, 16384, 80, 8, 128, 128),
    (1, 4096, 32768, 80, 8, 128, 128),
    (1, 4096, 8192, 80, 8, 64, 128),
    (1, 4096, 16384, 80, 8, 64, 128),
    (1, 4096, 32768, 80, 8, 64, 128),
    (1, 4096, 8192, 128, 8, 128, 128),
    (1, 4096, 16384, 128, 8, 128, 128),
    (1, 4096, 32768, 128, 8, 128, 128),
    (1, 4096, 8192, 128, 8, 64, 128),
    (1, 4096, 16384, 128, 8, 64, 128),
    (1, 4096, 32768, 128, 8, 64, 128),
    (1, 4096, 8192, 64, 8, 128, 128),
    (1, 4096, 16384, 64, 8, 128, 128),
    (1, 4096, 32768, 64, 8, 128, 128),
    (1, 4096, 8192, 64, 8, 64, 128),
    (1, 4096, 16384, 64, 8, 64, 128),
    (1, 4096, 32768, 64, 8, 64, 128),
    (1, 4096, 8192, 32, 8, 128, 128),
    (1, 4096, 16384, 32, 8, 128, 128),
    (1, 4096, 32768, 32, 8, 128, 128),
    (1, 4096, 8192, 32, 8, 64, 128),
    (1, 4096, 16384, 32, 8, 64, 128),
    (1, 4096, 32768, 32, 8, 64, 128),
    (1, 4096, 8192, 16, 8, 128, 128),
    (1, 4096, 16384, 16, 8, 128, 128),
    (1, 4096, 32768, 16, 8, 128, 128),
    (1, 4096, 8192, 16, 8, 64, 128),
    (1, 4096, 16384, 16, 8, 64, 128),
    (1, 4096, 32768, 16, 8, 64, 128),
]


def compute_flops(batch, q_seq, k_seq, q_head, kv_head, head_dim):
    """Compute total FLOPs for attention (Q*K + softmax + S*V)."""
    group = q_head // kv_head
    flops = group * kv_head * q_seq * head_dim * k_seq * 2  # Q*K^T
    flops += group * kv_head * q_seq * k_seq * 2  # softmax
    flops += group * kv_head * q_seq * head_dim * k_seq * 2  # S*V
    flops *= batch

    return flops


def compute_mem_bytes(batch, q_seq, k_seq, q_head, kv_head, head_dim, elem_size=2):
    """Compute read+write bytes (Q + K + V + O)."""
    read_bytes = q_head * q_seq * head_dim * elem_size  # Q
    read_bytes += kv_head * k_seq * head_dim * elem_size  # K
    read_bytes += kv_head * k_seq * head_dim * elem_size  # V
    write_bytes = q_head * q_seq * head_dim * elem_size  # O
    read_bytes *= batch
    write_bytes *= batch
    return read_bytes, write_bytes


def run_one(batch, q_seq, k_seq, q_head, kv_head, head_dim, block_size, dt=torch.float16, is_causal=False, dev="xpu"):
    """Run both FP16 SDPA and INT8 sage_dynquant for one configuration."""
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(batch, q_head, q_seq, head_dim, dtype=dt, device=dev)
    k = torch.randn(batch, kv_head, k_seq, head_dim, dtype=dt, device=dev)
    v = torch.randn(batch, kv_head, k_seq, head_dim, dtype=dt, device=dev)

    # --- FP16 SDPA (our kernel) ---
    dur_fp16 = bench(lambda: get_ark().sdpa(q, k, v, scale=scale, is_causal=is_causal))
    fp16_us = dur_fp16 * 1e6

    # --- INT8 sage_dynquant (our kernel) ---
    dur_i8 = bench(lambda: get_ark().sagev1(q, k, v, scale=scale, is_causal=is_causal, quant_block_size=block_size))
    i8_us = dur_i8 * 1e6

    # Correctness: INT8 vs PyTorch FP16
    out_i8 = get_ark().sagev1(q, k, v, scale=scale, is_causal=is_causal, quant_block_size=block_size)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, scale=scale, is_causal=is_causal, enable_gqa=(q_head // kv_head) > 1
    )
    dff = (ref - out_i8).abs()

    # Metrics
    flops = compute_flops(batch, q_seq, k_seq, q_head, kv_head, head_dim)
    ratio = (
        1 if not is_causal else ((k_seq - q_seq) + q_seq / 2) / k_seq
    )  # Approximate ratio of effective k_seq for causal attention
    flops = flops * ratio  # Adjust FLOPs for causal case where effective k_seq is smaller
    fp16_tflops = flops / dur_fp16 / 1e12 if dur_fp16 > 0 else 0
    i8_tflops = flops / dur_i8 / 1e12 if dur_i8 > 0 else 0
    fp16_mfu = fp16_tflops / BMG_PEAK_TFLOPS_FP16
    i8_mfu = i8_tflops / BMG_PEAK_TFLOPS_INT8

    # Reference latency from the other team
    ref_us = REF_LATENCY.get((q_head, head_dim, k_seq), None)

    return {
        "q_head_num": q_head,
        "kv_head_num": kv_head,
        "head_dim": head_dim,
        "batch_size": batch,
        "q_seq_len": q_seq,
        "k_seq_len": k_seq,
        "block_size": block_size,
        "ref_us": ref_us,
        "fp16_us": fp16_us,
        "fp16_tflops": fp16_tflops,
        "fp16_mfu": fp16_mfu,
        "fp16_vs_ref": fp16_us / ref_us if ref_us else None,
        "i8_us": i8_us,
        "i8_tflops": i8_tflops,
        "i8_mfu": i8_mfu,
        "i8_vs_ref": i8_us / ref_us if ref_us else None,
        "i8_vs_fp16": i8_us / fp16_us if fp16_us > 0 else None,
        "diff_max": dff.max().item(),
        "diff_mean": dff.mean().item(),
    }


def main():
    dev = "xpu"
    dt = torch.float16
    is_causal = True

    print("=" * 160)
    print("BMG Benchmark: Our FP16 SDPA vs Our INT8 sage_dynquant vs Reference sage_attention_page")
    print(f"dtype={dt}, causal={is_causal}, device={dev}")
    print("=" * 160)

    header = (
        f"{'Hq':>4s} {'Hkv':>3s} {'D':>4s} {'Sq':>5s} {'Skv':>6s} "
        f"{'Ref(us)':>10s} "
        f"{'FP16(us)':>10s} {'TFLOPS':>7s} {'MFU':>5s} {'vs Ref':>7s} "
        f"{'INT8(us)':>10s} {'TFLOPS':>7s} {'MFU':>5s} {'vs Ref':>7s} {'vs FP16':>7s} "
        f"{'DiffMax':>9s} {'DiffMean':>9s}"
    )
    print(header)
    print("-" * 160)

    results = []
    for batch, q_seq, k_seq, q_head, kv_head, head_dim, block_size in CONFIGS:
        r = run_one(batch, q_seq, k_seq, q_head, kv_head, head_dim, block_size, dt=dt, is_causal=is_causal, dev=dev)
        results.append(r)

        ref_str = f"{r['ref_us']:.0f}" if r["ref_us"] else "N/A"
        fp16_vs_ref = f"{r['fp16_vs_ref']:.2f}x" if r["fp16_vs_ref"] else "N/A"
        i8_vs_ref = f"{r['i8_vs_ref']:.2f}x" if r["i8_vs_ref"] else "N/A"
        i8_vs_fp16 = f"{r['i8_vs_fp16']:.2f}x" if r["i8_vs_fp16"] else "N/A"

        print(
            f"{r['q_head_num']:4d} {r['kv_head_num']:3d} {r['head_dim']:4d} {r['q_seq_len']:5d} {r['k_seq_len']:6d} "
            f"{ref_str:>10s} "
            f"{r['fp16_us']:10.0f} {r['fp16_tflops']:7.1f} {r['fp16_mfu']:4.0%} {fp16_vs_ref:>7s} "
            f"{r['i8_us']:10.0f} {r['i8_tflops']:7.1f} {r['i8_mfu']:4.0%} {i8_vs_ref:>7s} {i8_vs_fp16:>7s} "
            f"{r['diff_max']:9.4f} {r['diff_mean']:9.6f}"
        )

    # Write CSV
    csv_path = f"bench_bmg_comparison_{'causal' if is_causal else 'non_causal'}.csv"
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")

    # Summary
    print("\n" + "=" * 100)
    print("Summary")
    print("=" * 100)

    fp16_ratios = [r["fp16_vs_ref"] for r in results if r["fp16_vs_ref"]]
    i8_ratios = [r["i8_vs_ref"] for r in results if r["i8_vs_ref"]]
    i8_fp16_ratios = [r["i8_vs_fp16"] for r in results if r["i8_vs_fp16"]]

    if fp16_ratios:
        faster_fp16 = sum(1 for x in fp16_ratios if x < 1.0)
        print("\n  Our FP16 SDPA vs Reference sage_attention_page:")
        print(f"    Faster configs: {faster_fp16}/{len(fp16_ratios)}")
        print(f"    Avg ratio:      {sum(fp16_ratios)/len(fp16_ratios):.3f}x")
        print(f"    Best:           {min(fp16_ratios):.3f}x    Worst: {max(fp16_ratios):.3f}x")

    if i8_ratios:
        faster_i8 = sum(1 for x in i8_ratios if x < 1.0)
        print("\n  Our INT8 sage_dynquant vs Reference sage_attention_page:")
        print(f"    Faster configs: {faster_i8}/{len(i8_ratios)}")
        print(f"    Avg ratio:      {sum(i8_ratios)/len(i8_ratios):.3f}x")
        print(f"    Best:           {min(i8_ratios):.3f}x    Worst: {max(i8_ratios):.3f}x")

    if i8_fp16_ratios:
        faster_i8_fp16 = sum(1 for x in i8_fp16_ratios if x < 1.0)
        print("\n  Our INT8 vs Our FP16:")
        print(f"    INT8 faster configs: {faster_i8_fp16}/{len(i8_fp16_ratios)}")
        print(f"    Avg ratio:           {sum(i8_fp16_ratios)/len(i8_fp16_ratios):.3f}x")
        print(f"    Best:                {min(i8_fp16_ratios):.3f}x    Worst: {max(i8_fp16_ratios):.3f}x")


if __name__ == "__main__":
    main()
