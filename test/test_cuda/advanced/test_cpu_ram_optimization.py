# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Quantize Qwen/Qwen3-4B-Instruct-2507 with AutoRound (4-bit)
and compare CPU RAM peak usage with different optimization options.

Optimization options:
1. cpu_stream_offload_blocks: Offload block weights to disk, load on demand
2. cpu_stream_loss: Compute loss on-the-fly using frozen block copy
"""

import gc
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.utils.device import memory_monitor


def get_rss_gb() -> float:
    """Return process RSS in GB (Linux)."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    kb = int(parts[1])
                    return kb / 1024 / 1024
    except Exception:
        return -1.0
    return -1.0


def log_rss(tag: str) -> None:
    rss_gb = get_rss_gb()
    if rss_gb >= 0:
        print(f"[RAM] {tag}: {rss_gb:.2f} GB")
    else:
        print(f"[RAM] {tag}: N/A")


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_quantization(
    label: str,
    cpu_stream_offload_blocks: bool = False,
    cpu_stream_loss: bool = False,
) -> tuple[float, float]:
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)
    print(f"  cpu_stream_offload_blocks={cpu_stream_offload_blocks}")
    print(f"  cpu_stream_loss={cpu_stream_loss}")

    memory_monitor.reset()
    log_rss("before model load")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    log_rss("after model load")

    # Determine if any optimization is enabled
    any_optimization = cpu_stream_offload_blocks or cpu_stream_loss

    autoround = AutoRound(
        model,
        tokenizer,
        bits=4,
        group_size=128,
        low_gpu_mem_usage=True,
        low_cpu_mem_usage=any_optimization,
        cpu_stream_offload_blocks=cpu_stream_offload_blocks,
        cpu_stream_loss=cpu_stream_loss,
        iters=200,
        nsamples=512,
        seqlen=2048,
    )

    print("Start 4-bit quantization...")
    t0 = time.time()
    quantized_model, _ = autoround.quantize()
    t1 = time.time()
    elapsed = t1 - t0
    print(f"Quantization finished in {elapsed:.1f}s")

    print(f"[PEAK] {memory_monitor.get_summary()}")
    log_rss("after quantization")

    del quantized_model
    del autoround
    del model
    del tokenizer
    cleanup()

    return memory_monitor.peak_ram, elapsed


def main():
    print("=" * 60)
    print("AutoRound 4-bit Quantization - CPU RAM Optimization Test")
    print("=" * 60)

    results = []

    # Test 1: Baseline (no optimization)
    peak, elapsed = run_quantization(
        "Test 1: Baseline (no optimization)",
        cpu_stream_offload_blocks=False,
        cpu_stream_loss=False,
    )
    results.append(("Baseline", peak, elapsed))

    # Test 2: cpu_stream_offload_blocks only
    peak, elapsed = run_quantization(
        "Test 2: cpu_stream_offload_blocks only",
        cpu_stream_offload_blocks=True,
        cpu_stream_loss=False,
    )
    results.append(("+ offload_blocks", peak, elapsed))

    # Test 3: cpu_stream_loss only
    peak, elapsed = run_quantization(
        "Test 3: cpu_stream_loss only",
        cpu_stream_offload_blocks=False,
        cpu_stream_loss=True,
    )
    results.append(("+ stream_loss", peak, elapsed))

    # Test 4: offload_blocks + stream_loss (All optimizations)
    peak, elapsed = run_quantization(
        "Test 4: All optimizations (offload_blocks + stream_loss)",
        cpu_stream_offload_blocks=True,
        cpu_stream_loss=True,
    )
    results.append(("All optimizations", peak, elapsed))

    # Summary
    print("\n" + "=" * 60)
    print("Summary: Peak RAM Comparison")
    print("=" * 60)
    print(f"{'Configuration':<25} {'Peak RAM (GB)':<15} {'Time (s)':<10} {'RAM Saved':<12}")
    print("-" * 62)
    baseline_ram = results[0][1]
    for name, peak, elapsed in results:
        saved = baseline_ram - peak
        saved_str = f"-{saved:.2f} GB" if saved > 0 else "baseline"
        print(f"{name:<25} {peak:<15.2f} {elapsed:<10.1f} {saved_str:<12}")


if __name__ == "__main__":
    main()
