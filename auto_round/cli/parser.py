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

ALGORITHM_FLAGS = ("--alg_configs", "--alg_config", "--algorithm", "--algorithms", "--alg", "--algs")


class _ParallelSizeAction(argparse.Action):
    """Allow repeated aliases only when they request the same degree."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: int,
        option_string: str | None = None,
    ) -> None:
        """Store a parallel degree while rejecting conflicting aliases.

        Args:
            parser (argparse.ArgumentParser): Parser reporting conflicting-size errors.
            namespace (argparse.Namespace): Parsed arguments updated in place.
            values (int): Parsed parallel degree requested by this option.
            option_string (str | None): Option spelling used for diagnostic messages.
        """
        seen = dict(getattr(namespace, "_parallel_sizes", {}))
        if self.dest in seen and seen[self.dest] != values:
            parser.error(f"conflicting parallel sizes for {option_string}")
        seen[self.dest] = values
        namespace._parallel_sizes = seen
        setattr(namespace, self.dest, values)


class _LegacyAliasAction(argparse.Action):
    """Parse aliases into their canonical argument, rejecting conflicting values.

    e.g. ``--avg_bits``/``--target_bits`` store into ``bits`` and
    ``--options``/``--option`` store into ``scheme``, so the legacy CLI usage
    keeps working while the old names stay out of ``--help``.
    """

    _CANONICAL_FLAGS = {"bits": "--bits", "scheme": "--scheme"}

    def __call__(self, parser, namespace, values, option_string=None):
        seen = dict(getattr(namespace, "_legacy_alias_values", {}))
        current_value = getattr(namespace, self.dest, None)
        if self.dest == "scheme":
            values = ",".join(part.strip() for value in values for part in value.split(",") if part.strip())
        if (
            current_value is not None
            and (self.dest in seen or current_value != self.default)
            and current_value != values
        ):
            parser.error(
                f"conflicting values for {self._CANONICAL_FLAGS[self.dest]}: "
                f"{current_value!r} was already provided, but {option_string!r} set {values!r}"
            )
        seen[self.dest] = values
        namespace._legacy_alias_values = seen
        setattr(namespace, self.dest, values)


def _parse_group_size(s: str):
    """Parse group_size: a plain int, or comma-separated ints for block-wise fp8."""
    if s.lstrip("-").isdigit():
        return int(s)
    return tuple(int(x.strip()) for x in s.split(","))


class _QuantizeArgumentParser(argparse.ArgumentParser):
    """Retain explicit options even when parsed args are passed directly to tune()."""

    def parse_known_args(self, args=None, namespace=None):
        import sys

        argv = list(sys.argv[1:] if args is None else args)
        parsed, extras = super().parse_known_args(argv, namespace)
        parsed._explicit_fields = self._explicit_dests(argv)
        return parsed, extras

    def _explicit_dests(self, argv) -> set[str]:
        """Return destinations set on the command line, including abbreviated options."""
        defaults = [(action, action.default) for action in self._actions]
        parser_defaults = dict(self._defaults)
        try:
            for action, _ in defaults:
                action.default = argparse.SUPPRESS
            self._defaults.clear()
            seen, _ = super().parse_known_args(argv, argparse.Namespace())
        except SystemExit:
            return set()
        finally:
            for action, default in defaults:
                action.default = default
            self._defaults.update(parser_defaults)
        return {dest for dest in vars(seen) if dest not in {"model", "_legacy_alias_values"}}


def add_common_quantization_arguments(group) -> None:
    """Add common quantization flags to an argparse group.

    To add a new flag, append an add_argument() call here and mirror it in
    _extract_common_quantization_kwargs() in main.py.
    """
    group.add_argument(
        "--scheme",
        "--schemes",
        default="W4A16",
        type=str,
        nargs="+",
        action=_LegacyAliasAction,
        help="Quantization preset, or comma/space-separated AutoScheme candidates. "
        "With multiple candidates, --bits sets the average target bits.",
    )
    group.add_argument(
        "--bits",
        "--bit",
        default=None,
        type=float,
        help="Weight quantization bit width. With multiple --scheme candidates, the AutoScheme average target bits.",
    )
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
    parser = _QuantizeArgumentParser(prog=prog)

    # ---- Model / Runtime ----
    rt = parser.add_argument_group("Runtime Arguments")
    rt.add_argument("--recipe", default=None, help="Recommended quantization plan (default: auto-round).")
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
        "--num_hidden_layers",
        "--debug_layer_num",
        default=None,
        type=int,
        help="Debug only: load only the first N decoder layers of the model. Useful for isolating and "
        "debugging issues on very large models with a fraction of the load time and memory. The resulting "
        "model is partial and must not be used for a real quantization run.",
    )
    rt.add_argument(
        "--batch_size", "--train_bs", "--bs", default=None, type=int, help="Batch size for calibration and tuning."
    )
    rt.add_argument("--seqlen", "--seq_len", default=None, type=int, help="Sequence length of the calibration samples.")
    rt.add_argument("--nsamples", "--nsample", default=None, type=int, help="Number of calibration samples to use.")
    rt.add_argument(
        "--device_map", "--device", "--devices", default="0", type=str, help="Device mapping used for quantization."
    )
    rt.add_argument(
        "--tensor_parallel",
        "--tp_size",
        dest="tp_size",
        action=_ParallelSizeAction,
        type=int,
        default=None,
        help="Native TP degree; starts local workers automatically, or reuses torchrun WORLD_SIZE.",
    )
    rt.add_argument(
        "--pipeline_parallel",
        "--pp_size",
        dest="pp_size",
        type=int,
        default=1,
        action=_ParallelSizeAction,
        help="GPUs for naive pipeline parallelism (single process, no overlapping microbatch schedule).",
    )
    rt.add_argument(
        "--data_parallel",
        "--dp",
        "--dp_size",
        default="off",
        type=str,
        help="Single-process data-parallel SignRound tuning: 'off' (default), 'auto' (all visible CUDA "
        "devices at --iters 0, otherwise the largest power of two that fits them), or an explicit "
        "world size N (at least 2, at most the visible device count; a power of two when "
        "--iters > 0). Replicas hold full block mirrors on the plan "
        "devices and draw disjoint calibration shards; free-VRAM estimates are advisory.",
    )
    rt.add_argument(
        "--parallel_batch",
        default="split",
        type=str,
        choices=["split", "per_replica"],
        help="Batch semantics for --data_parallel tuning. 'split' (default): the requested "
        "--batch_size is the GLOBAL per-iteration batch and every replica draws its world-split "
        "share (serial-equivalent updates). 'per_replica': every replica draws the FULL "
        "--batch_size from its own calibration shard each iteration, so the effective global "
        "batch is batch_size x world (torchrun-like); the per-replica draw clamps to the pool shard "
        "when it is smaller.",
    )
    rt.add_argument("--dataset", default=None, type=str, help="Calibration dataset or local dataset path.")
    rt.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
    rt.add_argument(
        "--format", "--formats", default="auto_round", type=str, help="Output format for the quantized model."
    )
    rt.add_argument(
        "--max_shard_size", default=None, type=str, help="Maximum size of each safetensors shard. Defaults to 5GB."
    )
    # TODO wenhuach need to add choice or verify the correctness
    rt.add_argument(
        *ALGORITHM_FLAGS,
        dest="alg_configs",
        default=None,
        type=str,
        help="Comma-separated algorithms such as 'auto_round', 'auto_round_v2', 'rtn', 'opt-rtn' "
        "or 'awq'. Run `auto_round list alg` for the full list.",
    )
    rt.add_argument("--output_dir", default="./tmp_autoround", type=str, help="Directory to save quantized artifacts.")
    rt.add_argument(
        "--avg_bits",
        "--target_bits",
        dest="bits",
        default=None,
        type=float,
        help=argparse.SUPPRESS,
        action=_LegacyAliasAction,
    )
    rt.add_argument(
        "--options",
        "--option",
        dest="scheme",
        default="W4A16",
        type=str,
        nargs="+",
        help=argparse.SUPPRESS,
        action=_LegacyAliasAction,
    )
    rt.add_argument(
        "--low_gpu_mem_usage", action="store_true", help="Enable memory-efficient mode by offloading features to CPU."
    )
    rt.add_argument(
        "--calibration_data_device",
        default="auto",
        type=str,
        help="Device placement for block calibration data (inputs/outputs between blocks): "
        "'auto' (default) keeps the pools on the primary cache device when they fit and spreads "
        "chunks over free GPUs otherwise; 'cpu' parks them on host RAM; 'off' keeps the "
        "single-device behavior; or an explicit device list like 'cuda:1,cuda:2' "
        "(bare indices '1,2' also accepted).",
    )
    rt.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help="Deprecated compatibility flag. Low CPU memory mode is enabled by default.",
    )
    rt.add_argument("--disable_low_cpu_mem_usage", action="store_true", help="Disable low CPU memory mode.")
    torch_compile_group = rt.add_mutually_exclusive_group()
    torch_compile_group.add_argument(
        "--enable_torch_compile",
        dest="enable_torch_compile",
        action="store_true",
        help="Enable torch.compile during quantization (force enable on Windows).",
    )
    torch_compile_group.add_argument(
        "--disable_torch_compile",
        dest="enable_torch_compile",
        action="store_false",
        help="Disable torch.compile during quantization.",
    )
    rt.set_defaults(enable_torch_compile=None)
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
        help="Ensure listed layers share the same quantization scheme (bits, group_size, data type, sym), "
        "as required by fused serving kernels. Added on top of the default vLLM fusion groups ("
        "q/k/v, MLA q_a/kv_a, gate/up); disable those with --disable_default_shared_layers. "
        "Accepts space-separated (--shared_layers l1 l2) or comma-separated ('l1,l2') per group.",
    )
    rt.add_argument(
        "--disable_default_shared_layers",
        action="store_true",
        help="Do not automatically share one quantization scheme across vLLM-fused projections "
        "(q_proj/k_proj/v_proj, MLA q_a_proj/kv_a_proj_with_mqa, gate_proj/up_proj).",
    )
    rt.add_argument(
        "--weight_score_mode",
        default="w*g",
        type=str,
        choices=["w*g", "weight", "grad"],
        help="AutoScheme delta-loss weight score: 'w*g' (default) uses |grad*(w-qdq_w)|, "
        "'weight' uses |w-qdq_w|, and 'grad' uses |grad|. Ignored for fixed schemes.",
    )
    rt.add_argument(
        "--static_kv_dtype",
        default=None,
        type=str,
        choices=["fp8", "float8_e4m3fn", "nvfp4"],
        help="Static KV-cache quantization data type.",
    )
    rt.add_argument(
        "--static_kv_granularity",
        default="tensor",
        type=str,
        choices=["tensor", "head"],
        help="Static KV-cache FP8 calibration granularity.",
    )
    rt.add_argument(
        "--static_attention_dtype",
        default=None,
        type=str,
        choices=["fp8", "float8_e4m3fn"],
        help="Static attention quantization data type.",
    )
    rt.add_argument(
        "--static_attention_granularity",
        default="tensor",
        type=str,
        choices=["tensor", "head"],
        help="Static attention FP8 calibration granularity.",
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
    ev.add_argument("--num_fewshot", "--num-fewshot", default=None, type=int, help="Number of few-shot examples.")
    ev.add_argument(
        "--eval_gen_kwargs", "--eval-gen-kwargs", default=None, type=str, help="Generation kwargs for LM-Eval."
    )
    ev.add_argument(
        "--fewshot_as_multiturn",
        "--fewshot-as-multiturn",
        action="store_true",
        help="Use multi-turn format for few-shot examples in LM-Eval.",
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
        "--enable_deterministic_algorithms",
        action="store_true",
        default=None,
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
        "--calib_num_inference_steps",
        default=8,
        type=int,
        help="Number of denoising steps used to collect diffusion quantization calibration inputs.",
    )
    diff.add_argument(
        "--num_inference_steps",
        default=50,
        type=int,
        help="Number of denoising steps for diffusion generation/evaluation.",
    )
    diff.add_argument("--generator_seed", default=None, type=int, help="Random seed used for diffusion generation.")
    diff.add_argument(
        "--diffusion_tuning_cache_size",
        default=0,
        type=lambda value: value if value == "auto" else float(value),
        help="Extra GPU buffer budget in GiB for single-CUDA diffusion SignRound prefetch with low_gpu_mem_usage. "
        "Use 'auto' to select a budget after the first tuning iteration; 0 preserves the existing path. "
        "This is not a limit on total GPU memory.",
    )

    # ---- Common Quantization Arguments ----
    quant_group = parser.add_argument_group("Common Quantization Arguments")
    add_common_quantization_arguments(quant_group)

    # ---- Algorithm-specific groups ----
    AlgorithmHandler.add_groups(parser)

    return parser


def build_list_parser(*, prog: str = "auto_round list") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("item", type=str, help="item to list, e.g., format, alg, recipe")
    parser.add_argument("name", nargs="?", default=None, help="optional specific format/algorithm/recipe name")
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
