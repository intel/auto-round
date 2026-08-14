# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import contextlib
import csv
import os
import sys
import time
from pathlib import Path

import torch
from diffusers import FluxPipeline
from flux_sparse_patch import patch_flux_sparse_attention_from_env

dtype = torch.bfloat16

MODEL_DEFAULT = "/mnt/disk3/models/black-forest-labs/FLUX.1-dev"
PROMPT_DEFAULT = "A cat holding a sign that says hello world"


def env_flag(name, default="1"):
    return os.getenv(name, default).lower() not in {"0", "false", "no", "off"}


def sync_xpu():
    if hasattr(torch, "xpu") and hasattr(torch.xpu, "synchronize"):
        torch.xpu.synchronize()


def main():
    model_id = os.getenv("FLUX_MODEL", MODEL_DEFAULT)
    topks = os.getenv("FLUX_SPARSE_TOPKS", "0.5 0.25 0.125").split()
    run_dense = env_flag("FLUX_RUN_DENSE", "1")
    prompt = os.getenv("FLUX_PROMPT", PROMPT_DEFAULT)
    height = int(os.getenv("FLUX_HEIGHT", "512"))
    width = int(os.getenv("FLUX_WIDTH", "512"))
    steps = int(os.getenv("FLUX_STEPS", "50"))
    seed = int(os.getenv("FLUX_SEED", "0"))
    guidance_scale = float(os.getenv("FLUX_GUIDANCE_SCALE", "3.5"))
    max_sequence_length = int(os.getenv("FLUX_MAX_SEQUENCE_LENGTH", "512"))

    default_out = Path("benchmarks/results") / f"flux_bf16_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(os.getenv("FLUX_OUTPUT_DIR", str(default_out)))
    out_dir.mkdir(parents=True, exist_ok=True)

    device_id = os.getenv("ZE_AFFINITY_MASK_VALUE", "")
    dev_suffix = f"_dev{device_id}" if device_id else ""

    print(f"[flux_sweep] model={model_id}", flush=True)
    print(f"[flux_sweep] topks={topks} run_dense={run_dense} size={height}x{width} steps={steps} seed={seed}", flush=True)
    print(f"[flux_sweep] out_dir={out_dir}", flush=True)
    print(f"[flux_sweep] kernel={os.getenv('FLUX_SPARSE_KERNEL', '?')} "
          f"q_tile={os.getenv('FLUX_SPARSE_Q_TILE_OVERRIDE', '0')} "
          f"q_block={os.getenv('FLUX_SPARSE_Q_BLOCK_TOKENS', 'default')} "
          f"k_block={os.getenv('FLUX_SPARSE_K_BLOCK_TOKENS', 'default')}", flush=True)

    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
    # Offload-only: pipe.to(device) would load the whole ~54 GB model onto the
    # 24.4 GB device and OOM.
    #
    # Use sequential (block-level) offload, NOT enable_model_cpu_offload():
    # model-level offload pulls the whole ~46 GB transformer onto the device,
    # peaking at ~24 GB (the full 24.4 GB device) during every transformer
    # forward. The sparse BF16 preprocess (triton_xpu) then has <0.4 GB of
    # headroom, and its kernel launch flakily OOMs / resets the GPU
    # (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY / OUT_OF_RESOURCES / DEVICE_LOST).
    # Sequential offload keeps only a few blocks resident (peak ~0.16 GB), so
    # the triton_xpu sparse path always has room.
    pipe.enable_sequential_cpu_offload()

    common_kwargs = dict(
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        max_sequence_length=max_sequence_length,
        generator=torch.Generator("cpu").manual_seed(seed),
    )

    rows = []

    def do_run(tag, topk, png_name):
        if topk is None:
            ctx = contextlib.nullcontext(None)
            os.environ.pop("FLUX_SPARSE_TOPK", None)
        else:
            os.environ["FLUX_SPARSE_TOPK"] = str(topk)
            ctx = patch_flux_sparse_attention_from_env(pipe.transformer)

        sync_xpu()
        t0 = time.perf_counter()
        with ctx as stats:
            image = pipe(prompt, output_type="pil", **common_kwargs).images[0]
        sync_xpu()
        wall_s = time.perf_counter() - t0

        png_path = out_dir / png_name
        image.save(png_path)

        row = {
            "tag": tag,
            "topk": "" if topk is None else topk,
            "wall_s": round(wall_s, 3),
            "sparsity": "",
            "calls": "",
            "sparse_calls": "",
            "runtime_fallbacks": "",
            "unsupported_fallbacks": "",
            "png": str(png_path),
        }
        if topk is not None:
            if stats is None or stats.sparse_calls == 0:
                raise RuntimeError(
                    f"topk={topk}: sparse path did not run (sparse_calls=0) — check FLUX_SPARSE_KERNEL / qtile256 config"
                )
            if stats.runtime_fallbacks or stats.unsupported_fallbacks:
                raise RuntimeError(
                    f"topk={topk}: silent fallback detected — runtime={stats.runtime_fallbacks}, "
                    f"unsupported={stats.unsupported_fallbacks}"
                )
            row["sparsity"] = round(float(stats.avg_sparsity), 4)
            row["calls"] = stats.total_calls
            row["sparse_calls"] = stats.sparse_calls
            row["runtime_fallbacks"] = stats.runtime_fallbacks
            row["unsupported_fallbacks"] = stats.unsupported_fallbacks
            print(
                f"[flux_sweep] DONE tag={tag} topk={topk} wall={wall_s:.3f}s "
                f"sparsity={stats.avg_sparsity:.4f} sparse_calls={stats.sparse_calls} "
                f"runtime_fallbacks={stats.runtime_fallbacks}", flush=True
            )
        else:
            print(f"[flux_sweep] DONE tag={tag} wall={wall_s:.3f}s", flush=True)
        rows.append(row)
        return row

    if run_dense:
        do_run("dense", None, f"flux_dense_512{dev_suffix}.png")

    for topk in topks:
        do_run("sparse_bf16_qtile256", topk, f"flux_bf16_qtile256_topk{topk}_512{dev_suffix}.png")

    # CSV + markdown summary (device-suffixed so parallel instances sharing
    # FLUX_OUTPUT_DIR do not clobber each other).
    csv_path = out_dir / f"sweep_summary{dev_suffix}.csv"
    fieldnames = ["tag", "topk", "wall_s", "sparsity", "calls", "sparse_calls",
                  "runtime_fallbacks", "unsupported_fallbacks", "png"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_dir / f"SWEEP_SUMMARY{dev_suffix}.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("| tag | topk | wall_s | sparsity | calls | sparse_calls | runtime_fallbacks | png |\n")
        fh.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            fh.write(
                f"| {r['tag']} | {r['topk']} | {r['wall_s']} | {r['sparsity']} | {r['calls']} "
                f"| {r['sparse_calls']} | {r['runtime_fallbacks']} | `{r['png']}` |\n"
            )

    print(f"[flux_sweep] summary: {csv_path}", flush=True)
    print(f"[flux_sweep] summary: {md_path}", flush=True)


if __name__ == "__main__":
    main()
