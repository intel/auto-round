# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""Argparse parser construction for all CLI subcommands.

All static flags are declared directly via add_argument() — no intermediate
dataclass wrappers. To add a new flag, find the right argument group below
and add a line.

Public API:
    build_quantize_parser() — the main quantize parser
    build_list_parser()     — the `list` subcommand parser
    build_eval_parser()     — the `eval` subcommand parser
    build_root_parser()     — top-level router
"""

from __future__ import annotations

import argparse

from auto_round.cli.algorithms import AlgorithmHandler
from auto_round.eval.eval_cli import EvalArgumentParser


def _parse_group_size(s: str):
    """Parse group_size: a plain int, or comma-separated ints for block-wise fp8."""
    if s.lstrip("-").isdigit():
        return int(s)
    return tuple(int(x.strip()) for x in s.split(","))


def add_common_quantization_arguments(group) -> None:
    """Add common quantization flags to an argparse group.

    To add a new flag, append an add_argument() call here and mirror it in
    _extract_common_quantization_kwargs() in main.py.
    """
    group.add_argument("--scheme", default="W4A16", type=str, help="Quantization scheme preset, e.g. W4A16, W8A16.")
    group.add_argument("--bits", default=None, type=int, help="Weight quantization bit width.")
    group.add_argument(
        "--group_size",
        default=None,
        type=_parse_group_size,
        help="Weight group size: positive int, -1 per-channel, 0 per-tensor.",
    )
    group.add_argument(
        "--asym", default=False, action="store_true", help="Use asymmetric weight quantization instead of symmetric."
    )
    group.add_argument(
        "--data_type", "--dtype", default=None, type=str, help="Weight quantization data type, e.g. int, fp8."
    )
    group.add_argument("--act_bits", default=None, type=int, help="Activation quantization bit width.")
    group.add_argument("--act_group_size", default=None, type=int, help="Activation quantization group size.")
    group.add_argument(
        "--act_asym",
        default=False,
        action="store_true",
        help="Use asymmetric activation quantization instead of symmetric.",
    )
    group.add_argument(
        "--act_data_type", "--act_dtype", default=None, type=str, help="Activation quantization data type."
    )
    group.add_argument(
        "--disable_act_dynamic",
        default=False,
        action="store_true",
        help="Use static activation quantization instead of dynamic.",
    )
    group.add_argument("--super_bits", default=None, type=int, help="Bit width for double quantization metadata.")
    group.add_argument(
        "--super_group_size", default=None, type=int, help="Group size for double quantization metadata."
    )
    group.add_argument(
        "--scale_dtype",
        default=None,
        choices=["fp16", "float16", "bf16", "bfloat16", "fp32", "float32"],
        help="Data type used to store quantization scales.",
    )
    group.add_argument(
        "--ignore_layers",
        "--fp_layers",
        default="",
        type=str,
        help="Comma-separated layer names to keep in higher precision.",
    )
    group.add_argument(
        "--quant_lm_head", default=False, action=argparse.BooleanOptionalAction, help="Quantize the lm_head module."
    )
    group.add_argument(
        "--to_quant_block_names", default=None, type=str, help="Comma-separated subset of block names to quantize."
    )


def build_quantize_parser(*, prog: str = "auto_round quantize") -> argparse.ArgumentParser:
    """Build the quantize parser with all argument groups."""
    parser = argparse.ArgumentParser(prog=prog)

    # ---- Model / Runtime ----
    rt = parser.add_argument_group("Runtime Arguments")
    rt.add_argument("model", default=None, nargs="?", help="Path to the pre-trained model or Hugging Face model id.")
    rt.add_argument(
        "--model_name",
        "--model",
        "--model_name_or_path",
        default="facebook/opt-125m",
        help="Path to the pre-trained model or Hugging Face model id.",
    )
    rt.add_argument("--model_dtype", default=None, help="Model dtype used when loading the model.")
    rt.add_argument("--platform", default="hf", help="Model loading platform. Options: hf or model_scope.")
    rt.add_argument(
        "--batch_size", "--train_bs", "--bs", default=None, type=int, help="Batch size for calibration and tuning."
    )
    rt.add_argument("--seqlen", "--seq_len", default=None, type=int, help="Sequence length of the calibration samples.")
    rt.add_argument("--nsamples", "--nsample", default=None, type=int, help="Number of calibration samples to use.")
    rt.add_argument(
        "--device_map", "--device", "--devices", default="0", type=str, help="Device mapping used for quantization."
    )
    rt.add_argument(
        "--dataset", default="NeelNanda/pile-10k", type=str, help="Calibration dataset or local dataset path."
    )
    rt.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
    rt.add_argument(
        "--format", "--formats", default="auto_round", type=str, help="Output format for the quantized model."
    )
    rt.add_argument(
        "--algorithm", default=None, type=str, help="Comma-separated algorithms such as 'awq' or 'awq,auto_round'."
    )
    rt.add_argument("--output_dir", default="./tmp_autoround", type=str, help="Directory to save quantized artifacts.")
    rt.add_argument("--avg_bits", "--target_bits", default=None, type=float, help="Average target bits for AutoScheme.")
    rt.add_argument("--options", default=None, type=str, help="AutoScheme options, for example 'W4A16,W8A16'.")
    rt.add_argument(
        "--low_gpu_mem_usage", action="store_true", help="Enable memory-efficient mode by offloading features to CPU."
    )
    rt.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help="Deprecated compatibility flag. Low CPU memory mode is enabled by default.",
    )
    rt.add_argument("--disable_low_cpu_mem_usage", action="store_true", help="Disable low CPU memory mode.")
    rt.add_argument("--enable_torch_compile", action="store_true", help="Enable torch.compile during quantization.")
    rt.add_argument(
        "--disable_trust_remote_code", action="store_true", help="Disable trust_remote_code when loading models."
    )
    rt.add_argument(
        "--layer_config", default=None, type=str, help="Per-layer quantization overrides encoded as JSON-like text."
    )
    rt.add_argument(
        "--shared_layers",
        type=str,
        nargs="+",
        action="append",
        default=None,
        help="Ensure listed layers share the same quantization data type.",
    )
    rt.add_argument(
        "--static_kv_dtype",
        default=None,
        type=str,
        choices=["fp8", "float8_e4m3fn"],
        help="Static KV-cache quantization data type.",
    )
    rt.add_argument(
        "--static_attention_dtype",
        default=None,
        type=str,
        choices=["fp8", "float8_e4m3fn"],
        help="Static attention quantization data type.",
    )

    # ---- Evaluation ----
    ev = parser.add_argument_group("Evaluation Arguments")
    ev.add_argument(
        "--tasks",
        "--task",
        nargs="?",
        const="lambada_openai,hellaswag,winogrande,piqa,mmlu,wikitext,"
        "truthfulqa_mc1,openbookqa,boolq,arc_easy,arc_challenge",
        default=None,
        help="LM-Eval tasks to run after quantization.",
    )
    ev.add_argument("--eval_bs", default=None, type=int, help="Batch size for evaluation.")
    ev.add_argument(
        "--limit", type=float, default=None, metavar="N|0<N<1", help="Evaluation example limit as a count or fraction."
    )
    ev.add_argument("--eval_task_by_task", action="store_true", help="Evaluate tasks sequentially instead of batching.")
    ev.add_argument(
        "--eval_backend", default="hf", type=str, choices=["hf", "vllm"], help="Backend to use for evaluation."
    )
    ev.add_argument("--vllm_args", default=None, type=str, help="Comma-separated custom vLLM arguments.")
    ev.add_argument(
        "--eval_model_dtype", default=None, type=str, help="Torch dtype used when loading the evaluation model."
    )
    ev.add_argument("--add_bos_token", action="store_true", help="Add a BOS token during evaluation.")

    # ---- Compatibility ----
    compat = parser.add_argument_group("Compatibility Arguments")
    compat.add_argument(
        "--ignore_scale_zp_bits",
        action="store_true",
        help="Ignore scale and zero-point overhead when computing AutoScheme target bits.",
    )
    compat.add_argument("--disable_amp", action="store_true", help="Disable AMP during tuning.")
    compat.add_argument(
        "--disable_deterministic_algorithms",
        action="store_true",
        help="Deprecated flag to disable deterministic algorithms.",
    )
    compat.add_argument(
        "--enable_deterministic_algorithms",
        action="store_true",
        help="Enable deterministic algorithms for reproducible runs.",
    )
    compat.add_argument("--model_free", action="store_true", help="Force model-free quantization mode.")
    compat.add_argument("--disable_model_free", action="store_true", help="Disable automatic model-free routing.")

    # ---- Multimodal ----
    mllm = parser.add_argument_group("Multimodal Arguments")
    mllm.add_argument("--mllm", action="store_true", help="Deprecated compatibility flag for multimodal mode.")
    mllm.add_argument(
        "--quant_nontext_module", action="store_true", help="Quantize non-text modules such as vision or audio towers."
    )
    mllm.add_argument(
        "--extra_data_dir", default=None, type=str, help="Directory containing multimodal calibration assets."
    )
    mllm.add_argument("--template", default=None, type=str, help="Custom template for dataset construction.")

    # ---- Diffusion ----
    diff = parser.add_argument_group("Diffusion Arguments")
    diff.add_argument("--prompt_file", default=None, type=str, help="File containing prompts for diffusion evaluation.")
    diff.add_argument("--prompt", default=None, type=str, help="Single prompt used for quick diffusion testing.")
    diff.add_argument("--metrics", "--metric", default="clip", help="Diffusion evaluation metrics.")
    diff.add_argument(
        "--image_save_dir", default="./tmp_image_save", type=str, help="Directory to save generated images."
    )
    diff.add_argument("--guidance_scale", default=7.5, type=float, help="Classifier-free guidance scale.")
    diff.add_argument(
        "--num_inference_steps", default=50, type=int, help="Number of denoising steps for diffusion evaluation."
    )
    diff.add_argument("--generator_seed", default=None, type=int, help="Random seed used for diffusion generation.")

    # ---- Common Quantization Arguments ----
    quant_group = parser.add_argument_group("Common Quantization Arguments")
    add_common_quantization_arguments(quant_group)

    # ---- Algorithm-specific groups ----
    AlgorithmHandler.add_groups(parser)

    return parser


def build_list_parser(*, prog: str = "auto_round list") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("item", type=str, help="item to list, e.g., format, alg")
    parser.add_argument("name", nargs="?", default=None, help="optional specific format/algorithm name")
    return parser


def build_eval_parser(*, prog: str = "auto_round eval") -> argparse.ArgumentParser:
    return EvalArgumentParser(prog=prog)


def build_root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto_round",
        description="AutoRound command line interface.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("quantize", help="Quantize a model.", add_help=False)
    subparsers.add_parser("list", help="List supported algorithms or formats.", add_help=False)
    subparsers.add_parser("eval", help="Evaluate a model.", add_help=False)
    help_parser = subparsers.add_parser("help", help="Show help for the CLI or a subcommand.")
    help_parser.add_argument("topic", nargs="?", choices=["quantize", "list", "eval"], default=None)
    return parser
