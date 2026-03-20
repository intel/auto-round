"""Qwen3-30B-A3B, MXFP8, 200 iters, with activation checkpointing."""

from run_quant import run_quant

run_quant(
    model="/storage/yiliu7/Qwen/Qwen3-30B-A3B",
    scheme="MXFP8",
    iters=200,
    save_dir="/storage/yiliu7/act_cp_verify/Qwen3-30B-A3B-MXFP8-actcp",
)
