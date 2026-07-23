# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import time

import pytest
import torch


ark = pytest.importorskip("auto_round_kernel", reason="ARK extension is not built")


# CRI hardware spec (per-device peak)
PEAK_TFLOPS_BF16 = 300.0
PEAK_TFLOPS_FP8 = 600.0
PEAK_BW_GBPS = 1370.0  # LPDDR5X
PEAK_MEM_GB = 256


def has_sage_fp8():
    return hasattr(torch, "xpu") and torch.xpu.is_available() and ark.xpu_lib is not None and hasattr(ark.xpu_lib, "sage_fp8")


def get_ark():
    return ark


def bench(fn, n_warmup=5, n_runs=10, dev="xpu"):
    """Benchmark helper: returns average time in ms using XPU events."""
    for _ in range(n_warmup):
        fn()
    if hasattr(torch, "xpu") and dev == "xpu":
        start_event = torch.xpu.Event(enable_timing=True)
        end_event = torch.xpu.Event(enable_timing=True)
        torch.xpu.synchronize()
        start_event.record()
        for _ in range(n_runs):
            fn()
        end_event.record()
        torch.xpu.synchronize()
        return start_event.elapsed_time(end_event) / n_runs
    else:
        st = time.time()
        for _ in range(n_runs):
            fn()
        return (time.time() - st) / n_runs * 1e3


def _perf_metrics(batch, num_heads_q, num_heads_kv, seq_q, seq_kv, head_dim,
                  is_causal, time_ms, in_bytes=2, out_bytes=2):
    """Compute cutlass-style Performance metrics.

    Args:
        time_ms: kernel execution time in milliseconds
        in_bytes: bytes per element for inputs (Q, K, V). BF16/FP16=2, FP8=1
        out_bytes: bytes per element for output. BF16/FP16=2, FP8=1

    Returns:
        (gbps, tflops)
    """
    B, Hq, Hkv = batch, num_heads_q, num_heads_kv
    Sq, Skv, D = seq_q, seq_kv, head_dim
    time_s = time_ms * 1e-3
    # GB/s = (bytes_read_Q + bytes_read_K + bytes_read_V + bytes_written_O) / time
    bytes_q = B * Hq * Sq * D * in_bytes
    bytes_k = B * Hkv * Skv * D * in_bytes
    bytes_v = B * Hkv * Skv * D * in_bytes
    bytes_o = B * Hq * Sq * D * out_bytes
    total_bytes = bytes_q + bytes_k + bytes_v + bytes_o
    gbps = (total_bytes * 1e-9) / time_s
    # effective seq lens: causal adjusts the active Skv; non-causal is full Skv
    if is_causal:
        offset = min(Sq, Skv)
        discard = Sq - offset
        effective_Sq = Sq - discard
        effective_Skv = Skv - offset + (offset + 1) / 2.0
    else:
        effective_Sq = Sq
        effective_Skv = float(Skv)
    batched_eff_Sq = B * effective_Sq
    batched_eff_Skv = B * effective_Skv
    # 2 * B * Hq * Sq * Skv * D for QK + 2 * B * Hq * Sq * Skv * D for PV
    flops_qk = 2.0 * Hq * batched_eff_Sq * batched_eff_Skv * D
    flops_pv = 2.0 * Hq * batched_eff_Sq * batched_eff_Skv * D
    tflops = ((flops_qk + flops_pv) * 1e-12) / time_s
    return gbps, tflops


def _print_perf(label, time_ms, batch, num_heads_q, num_heads_kv, seq_q, seq_kv,
                head_dim, is_causal, in_bytes=2, out_bytes=2, peak_tflops=PEAK_TFLOPS_BF16):
    gbps, tflops = _perf_metrics(
        batch, num_heads_q, num_heads_kv, seq_q, seq_kv, head_dim,
        is_causal, time_ms, in_bytes=in_bytes, out_bytes=out_bytes,
    )
    mfu = (tflops / peak_tflops) * 100.0
    mbu = (gbps / PEAK_BW_GBPS) * 100.0
    bound = "compute" if mfu >= mbu else "memory"
    print(
        f"  {label:<14} {time_ms:8.4f} ms  |  {gbps:8.3f} GB/s  |  {tflops:7.3f} TFlop/s"
        f"  |  MFU={mfu:5.1f}%  MBU={mbu:5.1f}%  ({bound}-bound)"
    )
    return gbps, tflops


def run_benchmark(
    batch=1,
    seq_q=256,
    seq_kv=256,
    num_heads_q=4,
    num_heads_kv=4,
    head_dim=64,
    dt=torch.bfloat16,
    is_causal=False,
    tensor_layout="HND",
    dev="xpu",
):
    """Run sage_fp8 accuracy and latency benchmark against FP16 SDPA reference."""
    if not has_sage_fp8():
        print("sage_fp8 not available, skipping benchmark")
        return

    enable_gqa = num_heads_q != num_heads_kv
    scale = 1.0 / math.sqrt(head_dim)

    shape_q = (batch, num_heads_q, seq_q, head_dim) if tensor_layout == "HND" else (batch, seq_q, num_heads_q, head_dim)
    shape_kv = (batch, num_heads_kv, seq_kv, head_dim) if tensor_layout == "HND" else (batch, seq_kv, num_heads_kv, head_dim)

    print("=== Sage FP8 Benchmark ===")
    print(f"Layout:{tensor_layout} Batch:{batch} SeqQ:{seq_q} SeqKV:{seq_kv} HeadQ:{num_heads_q} HeadKV:{num_heads_kv} HDim:{head_dim} Causal:{is_causal} GQA:{enable_gqa}")
    print()

    query = torch.randn(shape_q, dtype=dt, device=dev)
    key = torch.randn(shape_kv, dtype=dt, device=dev)
    value = torch.randn_like(key)

    ref = torch.nn.functional.scaled_dot_product_attention(
        query.to("cpu"), key.to("cpu"), value.to("cpu"), scale=scale, is_causal=is_causal
    )

    t_ref = bench(lambda: torch.nn.functional.scaled_dot_product_attention(
        query.to("cpu"), key.to("cpu"), value.to("cpu"), scale=scale, is_causal=is_causal
    ), dev="cpu")
    print(f"  FP16 SDPA ref (CPU):  {t_ref:8.2f} ms ")

    out_fp8 = get_ark().sage_fp8(query, key, value, scale=scale, is_causal=is_causal, enable_gqa=enable_gqa, tensor_layout=tensor_layout)
    dff_fp8 = (ref.to(dev) - out_fp8).abs()
    t_fp8 = bench(lambda: get_ark().sage_fp8(query, key, value, scale=scale, is_causal=is_causal, enable_gqa=enable_gqa, tensor_layout=tensor_layout), dev="xpu")
    print(f"  Sage FP8:       {t_fp8:8.4f} ms  diff max={dff_fp8.max().item():.6f} mean={dff_fp8.mean().item():.6f}")
    _print_perf("Sage FP8", t_fp8, batch, num_heads_q, num_heads_kv, seq_q, seq_kv,
                head_dim, is_causal, in_bytes=1, out_bytes=2, peak_tflops=PEAK_TFLOPS_FP8)

    if hasattr(get_ark().xpu_lib, "sagev1") and head_dim in (64, 128):
        out_v1 = get_ark().sagev1(query, key, value, scale=scale, is_causal=is_causal, enable_gqa=enable_gqa, smooth_k=True)
        dff_v1 = (ref.to(dev) - out_v1).abs()
        t_v1 = bench(lambda: get_ark().sagev1(query, key, value, scale=scale, is_causal=is_causal, enable_gqa=enable_gqa, smooth_k=True), dev="xpu")
        print(f"  SageV1:         {t_v1:8.4f} ms  diff max={dff_v1.max().item():.6f} mean={dff_v1.mean().item():.6f}")
        _print_perf("SageV1", t_v1, batch, num_heads_q, num_heads_kv, seq_q, seq_kv,
                    head_dim, is_causal)
        print(f"  SageFP8 vs V1:  {t_fp8/t_v1:.2f}x  max_err ratio={dff_fp8.max().item()/max(dff_v1.max().item(),1e-9):.2f}")

    print()


if __name__ == "__main__":
    run_benchmark(batch=1, seq_q=4096, seq_kv=4096, num_heads_q=4, num_heads_kv=4, head_dim=128)
    run_benchmark(batch=1, seq_q=4096, seq_kv=4096, num_heads_q=4, num_heads_kv=4, head_dim=128, is_causal=True)
    run_benchmark(batch=1, seq_q=4096, seq_kv=4096, num_heads_q=4, num_heads_kv=4, head_dim=128, is_causal=True, tensor_layout="NHD")
