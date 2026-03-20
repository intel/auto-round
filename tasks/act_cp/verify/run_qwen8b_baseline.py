"""Qwen3-8B, W4A16, 200 iters, baseline (no activation checkpointing)."""

from run_quant import run_quant

run_quant(
    model="/storage/yiliu7/Qwen/Qwen3-8B",
    scheme="W4A16",
    iters=200,
    save_dir="/storage/yiliu7/act_cp_verify/Qwen3-8B-W4A16-baseline",
)
