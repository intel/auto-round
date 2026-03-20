"""Shared quantization + memory tracking template.

Usage: each launcher script imports run_quant() and calls it with model-specific args.
CUDA_VISIBLE_DEVICES is set externally, so device is always 0 inside the script.
AR_ENABLE_ACTIVATION_CHECKPOINTING env var controls checkpointing (set externally).
"""

import gc
import os
import time

import torch

from auto_round import AutoRound


def run_quant(model, scheme, iters, save_dir):
    act_ckpt = os.environ.get("AR_ENABLE_ACTIVATION_CHECKPOINTING", "0").lower() in ("1", "true", "yes")

    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    t0 = time.time()
    autoround = AutoRound(
        model,
        scheme=scheme,
        iters=iters,
        low_gpu_mem_usage=True,
        enable_activation_checkpointing=act_ckpt,
    )
    res = autoround.quantize_and_save(format="auto_round", output_dir=save_dir)
    elapsed = time.time() - t0

    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)

    print(f"\n{'=' * 80}")
    print(f"  RESULTS: {model.split('/')[-1]} | {scheme} | act_ckpt={act_ckpt}")
    print(f"  Peak GPU allocated:  {peak_gb:.2f} GB")
    print(f"  Peak GPU reserved:   {peak_reserved_gb:.2f} GB")
    print(f"  Total quant time:    {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    print(f"  Output dir:          {save_dir}")
    print(f"{'=' * 80}")
