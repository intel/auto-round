"""Benchmark for DynQuant SAGE with block-wise INT8 quantization.

Tests sage_dynquant with different quant_block_size values (per-token, 32, 64, 128, 256).
Compares accuracy and latency against SAGE v1 and FP16 SDPA references.
"""

import math
import time
import torch
import auto_round_kernel

ark = None

def get_ark():
    global ark
    if ark is None:
        ark = auto_round_kernel.ARK()
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


def run_benchmark(batch=2, seq=17776, h_q=30, h_kv=30, H=128, H_v=128,
                  dt=torch.float16, is_causal=False, dev="xpu"):
    # Align seq to largest block_size for divisibility
    block_sizes = [1, 32, 64, 128, 256]
    max_bs = max(block_sizes)
    seq = seq // max_bs * max_bs
    q = torch.randn(batch, h_q, seq, H, dtype=dt, device=dev)
    k = torch.randn(batch, h_kv, seq, H, dtype=dt, device=dev)
    v = torch.randn(batch, h_kv, seq, H_v, dtype=dt, device=dev)
    scale = 1 / math.sqrt(H)

    print(f"=== SageDynQuant Block-wise Benchmark ===")
    print(f"Batch:{batch} Seq:{seq} HeadQ:{h_q} HeadKV:{h_kv} HDim:{H} Causal:{is_causal}")
    print()

    # --- FP16 reference ---
    ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, scale=scale, is_causal=is_causal, enable_gqa=(h_q // h_kv) > 1)

    # --- Test different block sizes ---
    results = {}

    for bs in block_sizes:
        out = get_ark().sage_dynquant(q, k, v, scale=scale, is_causal=is_causal,
                                       quant_block_size=bs)
        dff = (ref - out).abs()
        t = bench(lambda bs=bs: get_ark().sage_dynquant(
            q, k, v, scale=scale, is_causal=is_causal, quant_block_size=bs))
        results[bs] = (t, dff.max().item(), dff.mean().item())
        print(f"  SageDynQuan  block_size={bs:3d}: {t:8.1f} ms  diff max={dff.max().item():.6f} mean={dff.mean().item():.6f}")

    # --- SAGE v1 reference for each block_size (pre-quantized INT8) ---
    sage_results = {}
    for bs in block_sizes:
        q_i8 = torch.randint(-128, 127, (batch, h_q, seq, H), dtype=torch.int8, device=dev)
        k_i8 = torch.randint(-128, 127, (batch, h_kv, seq, H), dtype=torch.int8, device=dev)
        qs = torch.randn(batch, h_q, seq // bs, 1, dtype=torch.float32, device=dev) / 100 + 0.001
        ks = torch.randn(batch, h_kv, seq // bs, 1, dtype=torch.float32, device=dev) / 100 + 0.001
        sage_t = bench(lambda bs=bs, q_i8=q_i8, k_i8=k_i8, qs=qs, ks=ks: get_ark().sage(
            q_i8, k_i8, v, scale_block_size=bs,
            qscale=qs, kscale=ks, scale=scale, is_causal=is_causal))
        sage_results[bs] = sage_t
        print(f"  SAGE v1 bs={bs:3d}:  {sage_t:8.1f} ms")

    print()
    print(f"--- Summary (dynquant vs SAGE v1 at same block_size) ---")
    for bs in block_sizes:
        t, mx, mn = results[bs]
        sage_t = sage_results[bs]
        overhead = (t / sage_t - 1) * 100
        print(f"  block_size={bs:3d}: dynquant {t:8.1f} ms  vs  sage_v1 {sage_t:8.1f} ms  (+{overhead:.0f}%)")


if __name__ == "__main__":
    run_benchmark()
    run_benchmark(is_causal=True)
