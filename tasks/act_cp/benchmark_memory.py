#!/usr/bin/env python
# Copyright (c) 2025 Intel Corporation
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
"""E2E benchmark: per-block CUDA memory tracking with/without activation checkpointing.

Uses torch.cuda memory snapshot hooks to track memory at block boundaries
without invasive monkey-patching.

Usage:
    /home/yiliu7/workspace/ar-local/bin/python tasks/act_cp/benchmark_memory.py
    /home/yiliu7/workspace/ar-local/bin/python tasks/act_cp/benchmark_memory.py --mode baseline
    /home/yiliu7/workspace/ar-local/bin/python tasks/act_cp/benchmark_memory.py --mode checkpointed
"""

import argparse
import gc
import sys
import threading
import time

import torch

from auto_round import AutoRound


def gpu_mb(device=0):
    return torch.cuda.memory_allocated(device) / (1024**2)


def gpu_peak_mb(device=0):
    return torch.cuda.max_memory_allocated(device) / (1024**2)


class BlockMemoryTracker:
    """Track per-block memory via forward hooks on the block modules."""

    def __init__(self, device=0):
        self.device = device
        self.records = []  # list of {block_name, stage, allocated_mb, peak_mb}
        self.handles = []
        self._lock = threading.Lock()

    def _snap(self, block_name, stage):
        torch.cuda.synchronize(self.device)
        with self._lock:
            self.records.append(
                {
                    "block_name": block_name,
                    "stage": stage,
                    "allocated_mb": gpu_mb(self.device),
                    "peak_mb": gpu_peak_mb(self.device),
                }
            )

    def attach(self, model, block_names):
        """Attach pre/post forward hooks on each block to record memory."""
        for name in block_names:
            parts = name.split(".")
            mod = model
            for p in parts:
                mod = getattr(mod, p)
            block_name = name

            def make_pre_hook(bname):
                def hook(module, inputs):
                    self._snap(bname, "pre_forward")

                return hook

            def make_post_hook(bname):
                def hook(module, inputs, outputs):
                    self._snap(bname, "post_forward")

                return hook

            h1 = mod.register_forward_pre_hook(make_pre_hook(block_name))
            h2 = mod.register_forward_hook(make_post_hook(block_name))
            self.handles.extend([h1, h2])

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def print_report(self, tag):
        print(f"\n{'='*80}")
        print(f"  PER-BLOCK FORWARD MEMORY: {tag}")
        print(f"{'='*80}")
        print(f"  {'Block':<40} {'Stage':<16} {'Alloc MB':>12} {'Peak MB':>12}")
        print(f"  {'-'*80}")
        for r in self.records:
            print(f"  {r['block_name']:<40} {r['stage']:<16} {r['allocated_mb']:>12.1f} {r['peak_mb']:>12.1f}")
        print(f"{'='*80}")


class PeriodicMemSampler:
    """Sample GPU memory in a background thread to capture the true peak during tuning."""

    def __init__(self, device=0, interval=0.05):
        self.device = device
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self.samples.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            alloc = torch.cuda.memory_allocated(self.device) / (1024**2)
            peak = torch.cuda.max_memory_allocated(self.device) / (1024**2)
            self.samples.append({"time": time.time(), "allocated_mb": alloc, "peak_mb": peak})
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def max_allocated(self):
        if not self.samples:
            return 0
        return max(s["allocated_mb"] for s in self.samples)

    def max_peak(self):
        if not self.samples:
            return 0
        return max(s["peak_mb"] for s in self.samples)


def get_block_names(model):
    """Extract block names from the model (same logic AutoRound uses)."""
    from auto_round.utils.model import get_block_names as _get_block_names

    all_blocks = _get_block_names(model)
    # Flatten
    names = []
    for group in all_blocks:
        names.extend(group)
    return names


def run_benchmark(model_name, scheme, enable_activation_checkpointing, iters=10, device=0):
    """Run quantization with memory tracking."""
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    gc.collect()

    tag = "CHECKPOINTED" if enable_activation_checkpointing else "BASELINE"
    print(f"\n{'#'*80}")
    print(f"  MODE: {tag}")
    print(f"  Model: {model_name}")
    print(f"  Scheme: {scheme}, iters={iters}")
    print(f"  enable_activation_checkpointing={enable_activation_checkpointing}")
    print(f"{'#'*80}\n")

    t0 = time.time()
    autoround = AutoRound(
        model_name,
        scheme=scheme,
        iters=iters,
        low_gpu_mem_usage=True,
        enable_activation_checkpointing=enable_activation_checkpointing,
        device_map=device,
    )

    # Get block names for hook attachment
    block_names = []
    for group in autoround.quant_block_list:
        block_names.extend(group)

    # Attach hooks
    tracker = BlockMemoryTracker(device=device)
    tracker.attach(autoround.model, block_names)

    # Start background sampler
    sampler = PeriodicMemSampler(device=device, interval=0.1)

    print(f"  Block forward function: {autoround.block_forward.__name__}")
    print(f"  Blocks to quantize: {block_names}")
    print("  Starting quantization...\n")

    torch.cuda.reset_peak_memory_stats(device)
    sampler.start()

    autoround.quantize()

    sampler.stop()
    tracker.detach()

    elapsed = time.time() - t0
    final_peak = gpu_peak_mb(device)

    # Print reports
    tracker.print_report(tag)

    print(f"\n{'='*80}")
    print(f"  SUMMARY: {tag}")
    print(f"{'='*80}")
    print(f"  CUDA peak (torch stats):   {final_peak:.1f} MB = {final_peak/1024:.2f} GB")
    print(f"  Sampler max allocated:     {sampler.max_allocated():.1f} MB = {sampler.max_allocated()/1024:.2f} GB")
    print(f"  Sampler max peak:          {sampler.max_peak():.1f} MB = {sampler.max_peak()/1024:.2f} GB")
    print(f"  Num samples:               {len(sampler.samples)}")
    print(f"  Wall time:                 {elapsed:.1f} s")
    print(f"{'='*80}")

    # Cleanup
    del autoround
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "tag": tag,
        "peak_mb": final_peak,
        "sampler_peak_mb": sampler.max_peak(),
        "elapsed_s": elapsed,
        "tracker": tracker,
        "sampler": sampler,
    }


def main():
    parser = argparse.ArgumentParser(description="Per-block CUDA memory benchmark")
    parser.add_argument("--model", default="/storage/yiliu7/Qwen/Qwen3-30B-A3B-L2")
    parser.add_argument("--scheme", default="MXFP8")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--mode", choices=["baseline", "checkpointed", "both"], default="both")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(args.device)
    gpu_mem = torch.cuda.get_device_properties(args.device).total_memory / (1024**3)
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"PyTorch: {torch.__version__}")

    results = []

    if args.mode in ("baseline", "both"):
        r = run_benchmark(args.model, args.scheme, False, args.iters, args.device)
        results.append(r)

    if args.mode in ("checkpointed", "both"):
        r = run_benchmark(args.model, args.scheme, True, args.iters, args.device)
        results.append(r)

    if len(results) == 2:
        bl, ck = results
        saved = bl["peak_mb"] - ck["peak_mb"]
        pct = (saved / bl["peak_mb"]) * 100 if bl["peak_mb"] > 0 else 0
        slowdown = ((ck["elapsed_s"] - bl["elapsed_s"]) / bl["elapsed_s"]) * 100

        print(f"\n{'='*80}")
        print("  COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"  Baseline peak:       {bl['peak_mb']:.1f} MB = {bl['peak_mb']/1024:.2f} GB")
        print(f"  Checkpointed peak:   {ck['peak_mb']:.1f} MB = {ck['peak_mb']/1024:.2f} GB")
        print(f"  Memory saved:        {saved:.1f} MB = {saved/1024:.2f} GB ({pct:.1f}%)")
        print(f"  Baseline time:       {bl['elapsed_s']:.1f} s")
        print(f"  Checkpointed time:   {ck['elapsed_s']:.1f} s")
        print(f"  Time overhead:       {slowdown:+.1f}%")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
