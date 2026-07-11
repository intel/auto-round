# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""CLI entry points: command routing, RECIPES, tune, eval, list.

This module is the single place that wires together:
    parser.py     - argparse construction (all flag declarations here)
    algorithms.py - algorithm config building

All console_scripts (auto_round, auto-round-best, etc.) point here.
"""

from __future__ import annotations

import argparse
import difflib
import sys

from auto_round.cli.algorithms import AlgorithmHandler
from auto_round.cli.parser import (
    add_common_quantization_arguments,
    build_eval_parser,
    build_list_parser,
    build_quantize_parser,
    build_root_parser,
)


def _extract_common_quantization_kwargs(args) -> dict:
    """Map parsed CLI args back to QuantizationConfig constructor kwargs.

    Handles inverted flags: --asym -> sym, --act_asym -> act_sym,
    --disable_act_dynamic -> act_dynamic.
    When the flag was not set (False), the value is None (defer to scheme).
    """
    return {
        "bits": args.bits,
        "group_size": args.group_size,
        "sym": None if not args.asym else False,
        "data_type": args.data_type,
        "act_bits": args.act_bits,
        "act_group_size": args.act_group_size,
        "act_sym": None if not args.act_asym else False,
        "act_data_type": args.act_data_type,
        "act_dynamic": None if not args.disable_act_dynamic else False,
        "super_bits": args.super_bits,
        "super_group_size": args.super_group_size,
    }


def _build_entry_base_kwargs(args, *, low_cpu_mem_usage, enable_torch_compile, layer_config) -> dict:
    return {
        "platform": args.platform,
        "format": args.format,
        "dataset": args.dataset,
        "seqlen": args.seqlen,
        "nsamples": args.nsamples,
        "batch_size": args.batch_size,
        "gradient_accumulate_steps": getattr(args, "gradient_accumulate_steps", 1),
        "low_gpu_mem_usage": args.low_gpu_mem_usage,
        "low_cpu_mem_usage": low_cpu_mem_usage,
        "device_map": args.device_map,
        "enable_torch_compile": enable_torch_compile,
        "seed": args.seed,
        "layer_config": layer_config,
        "model_dtype": args.model_dtype,
        "trust_remote_code": not args.disable_trust_remote_code,
    }


def _build_entry_route_kwargs(args) -> dict:
    return {
        "model_free": args.model_free,
        "disable_model_free": args.disable_model_free,
    }


def _build_entry_compressor_kwargs(args) -> dict:
    return {
        "scale_dtype": args.scale_dtype,
        "ignore_layers": args.ignore_layers,
        "quant_lm_head": args.quant_lm_head,
        "to_quant_block_names": args.to_quant_block_names,
    }


def _build_entry_model_type_kwargs(args) -> dict:
    return {
        "quant_nontext_module": args.quant_nontext_module,
        "extra_data_dir": args.extra_data_dir,
        "template": args.template,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "generator_seed": args.generator_seed,
    }


def _to_autoround_kwargs(args, *, low_cpu_mem_usage, enable_torch_compile, layer_config) -> dict:
    """Collect only the kwargs accepted by the new AutoRound entry API."""
    kwargs = _build_entry_base_kwargs(
        args,
        low_cpu_mem_usage=low_cpu_mem_usage,
        enable_torch_compile=enable_torch_compile,
        layer_config=layer_config,
    )
    kwargs.update(_build_entry_route_kwargs(args))
    kwargs.update(_build_entry_compressor_kwargs(args))
    kwargs.update(_build_entry_model_type_kwargs(args))
    return kwargs


RECIPES = {
    "default": {
        "batch_size": 8,
        "iters": 200,
        "seqlen": 2048,
        "nsamples": 128,
        "lr": None,
    },
    "best": {
        "batch_size": 8,
        "iters": 1000,
        "seqlen": 2048,
        "nsamples": 512,
        "lr": None,
    },
    "light": {
        "batch_size": 8,
        "iters": 50,
        "seqlen": 2048,
        "nsamples": 128,
        "lr": 5e-3,
    },
    "rtn": {"batch_size": 8, "iters": 0, "seqlen": 2048, "nsamples": 1, "lr": None, "disable_opt_rtn": True},
    "opt_rtn": {"batch_size": 8, "iters": 0, "seqlen": 2048, "nsamples": 128, "lr": None, "disable_opt_rtn": False},
}

# ============================================================================
# list subcommand
# ============================================================================


def list_item(argv=None):
    args = build_list_parser().parse_args(argv)
    if args.item in {"format", "formats"}:
        from auto_round.formats import OutputFormat

        print("AutoRound supported output formats and quantization scheme:")
        print(OutputFormat.get_support_matrix())
    elif args.item in {"alg", "algs", "algorithm", "algorithms"}:
        if args.name:
            print(AlgorithmHandler.format_detail(args.name))
        else:
            print("AutoRound supported algorithms:")
            print(AlgorithmHandler.format_listing())
            print("\nUse `auto_round list alg <name>` or `auto_round --algorithm <name> --help` for details.")
    else:
        raise ValueError(f"Unsupported list target: {args.item}")


# ============================================================================
# quantize subcommand
# ============================================================================


def _print_algorithm_help(argv: list[str]) -> bool:
    """If --algorithm <name> --help is present, print algorithm-focused help and return True."""
    if not any(flag in argv for flag in ("-h", "--help")):
        return False

    # Pre-parse just --algorithm

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--algorithm", default=None)
    known, _ = pre.parse_known_args(argv)
    names = [n.strip().lower() for n in (known.algorithm or "").split(",") if n.strip()]
    if not names:
        return False

    # Resolve aliases to canonical names, silently ignore unknowns
    canonical_names: list[str] = []
    unknown_names: list[str] = []
    for name in names:
        canon = AlgorithmHandler.resolve_alias(name)
        if canon and canon not in canonical_names:
            canonical_names.append(canon)
            continue
        if canon is None:
            unknown_names.append(name)
    if unknown_names:
        from auto_round.algorithms.registry import list_registered_algorithms

        suggestions = []
        supported = list_registered_algorithms()
        for name in unknown_names:
            match = difflib.get_close_matches(name, supported, n=1)
            if match:
                suggestions.append(f"Unknown algorithm '{name}'. Did you mean '{match[0]}'?")
            else:
                suggestions.append(f"Unknown algorithm '{name}'.")
        raise SystemExit(" ".join(suggestions) + " Use `auto_round list alg` to see supported algorithms.")
    if not canonical_names:
        return False

    mini = argparse.ArgumentParser(
        prog=f"auto_round --algorithm {','.join(canonical_names)}",
        description=f"Flags for algorithm(s): {', '.join(canonical_names)}. "
        "Use `auto_round --help` for the full argument list.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    quant_group = mini.add_argument_group("Common Quantization Arguments")
    add_common_quantization_arguments(quant_group)
    for name in canonical_names:
        alg_group = mini.add_argument_group(f"Algorithm: {name}")
        AlgorithmHandler.get(name).register(alg_group)
    mini.print_help()
    return True


def start(recipe="default", argv=None):
    recipe_defaults = RECIPES[recipe]
    argv = list(sys.argv[1:] if argv is None else argv)

    if _print_algorithm_help(argv):
        return

    parser = build_quantize_parser(prog="auto_round quantize")
    args = parser.parse_args(argv)

    # Apply recipe defaults for fields the user didn't set
    for key, value in recipe_defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    tune(args)


def tune(args):
    assert args.model or args.model_name, "[model] or --model MODEL_NAME should be set."
    if args.model is None:
        args.model = args.model_name
    if args.eval_bs is None:
        args.eval_bs = "auto"

    from transformers.utils.versions import require_version

    if args.tasks is not None:
        require_version(
            "lm_eval>=0.4.2",
            "lm-eval is required for evaluation, please install it with `pip install 'lm-eval>=0.4.2'`",
        )

    from auto_round.utils import logger

    if args.low_cpu_mem_usage:
        logger.warning(
            "`low_cpu_mem_usage` is deprecated and is now enabled by default. "
            "To disable it, use `--disable_low_cpu_mem_usage`."
        )

    if args.format is None:
        args.format = "auto_round"

    formats = args.format.lower().replace(" ", "").split(",")
    from auto_round.utils import SUPPORTED_FORMATS

    for fmt in formats:
        if fmt not in SUPPORTED_FORMATS:
            raise ValueError(f"{fmt} is not supported, we only support {SUPPORTED_FORMATS}")

    if "auto_gptq" in args.format and args.asym is True:
        logger.warning(
            "the auto_gptq kernel has issues with asymmetric quantization. "
            "It is recommended to use sym quantization or --format='auto_round'"
        )

    if "marlin" in args.format and args.asym is True:
        raise RuntimeError("marlin backend only supports sym quantization, please remove --asym")

    from auto_round.utils import get_device_and_parallelism

    device_str, use_auto_mapping = get_device_and_parallelism(args.device_map)

    if args.enable_torch_compile:
        logger.info(
            "`torch.compile` is enabled to reduce tuning costs. "
            "If it causes issues, you can disable it by removing `--enable_torch_compile` argument."
        )

    model_name = args.model
    if model_name[-1] == "/":
        model_name = model_name[:-1]
    logger.info(f"start to quantize {model_name}")

    from auto_round.compressors.base import BaseCompressor
    from auto_round.compressors.entry import AutoRound as PipelineAutoRound

    if "bloom" in model_name:
        args.low_gpu_mem_usage = False

    if args.quant_lm_head:
        for fmt in formats:
            if "auto_round" not in fmt and "fake" not in fmt and "mlx" not in fmt:
                auto_round_formats = [s for s in SUPPORTED_FORMATS if s.startswith("auto_round") or s == "mlx"]
                raise ValueError(
                    f"{fmt} is not supported for lm-head quantization, please change to {auto_round_formats}"
                )

    enable_torch_compile = True if "--enable_torch_compile" in sys.argv else False
    scheme = args.scheme.upper()

    from auto_round.schemes import PRESET_SCHEMES

    if scheme not in PRESET_SCHEMES:
        raise ValueError(f"{scheme} is not supported. only {PRESET_SCHEMES.keys()} are supported ")

    if args.disable_deterministic_algorithms:
        logger.warning(
            "default not use deterministic_algorithms. disable_deterministic_algorithms is deprecated,"
            " please use enable_deterministic_algorithms instead. "
        )

    from auto_round.utils import parse_layer_config_arg

    layer_config = {}
    if args.layer_config:
        layer_config = parse_layer_config_arg(args.layer_config)
        args.layer_config = layer_config

    low_cpu_mem_usage = True
    if args.disable_low_cpu_mem_usage:
        low_cpu_mem_usage = False

    from auto_round.auto_scheme import AutoScheme

    if args.avg_bits is not None:
        if args.options is None:
            raise ValueError("please set --options for auto scheme")
        if enable_torch_compile:
            logger.warning(
                "`enable_torch_compile=True` with AutoScheme may cause compile errors "
                "on some models. If so, try removing `--enable_torch_compile`."
            )
        scheme = AutoScheme(
            options=args.options,
            avg_bits=args.avg_bits,
            shared_layers=args.shared_layers,
            ignore_scale_zp_bits=args.ignore_scale_zp_bits,
            low_gpu_mem_usage=True,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

    common_kwargs = _extract_common_quantization_kwargs(args)
    alg_configs = AlgorithmHandler.build_configs(args, common_kwargs)

    from auto_round.utils import clear_memory

    autoround: BaseCompressor = PipelineAutoRound(
        model_name,
        scheme,
        alg_configs if len(alg_configs) > 1 else alg_configs[0],
        **_to_autoround_kwargs(
            args,
            low_cpu_mem_usage=low_cpu_mem_usage,
            enable_torch_compile=enable_torch_compile,
            layer_config=layer_config,
        ),
    )

    model, folders = autoround.quantize_and_save(args.output_dir, format=args.format)  # pylint: disable=no-member
    tokenizer = autoround.tokenizer  # pylint: disable=no-member
    clear_memory()

    from auto_round.eval.evaluation import run_model_evaluation

    run_model_evaluation(model, tokenizer, autoround, folders, formats, args)


# ============================================================================
# eval subcommand
# ============================================================================


def setup_eval_parser(argv=None):
    parser = build_eval_parser(prog="auto_round eval")
    return parser.parse_args(argv)


def run_eval(argv=None):
    from auto_round.eval.eval_cli import eval, eval_task_by_task
    from auto_round.logger import logger
    from auto_round.utils import is_gguf_model, is_mllm_model

    args = setup_eval_parser(argv)
    assert args.model or args.model_name, "[model] or --model MODEL_NAME should be set."

    if args.model is None:
        args.model = args.model_name
    if "llama" in args.model.lower() and not args.add_bos_token:
        logger.warning("set add_bos_token=True for llama model.")
        args.add_bos_token = True
    if not is_gguf_model(args.model) and is_mllm_model(args.model):
        args.mllm = True

    if args.eval_task_by_task:
        eval_task_by_task(
            model=args.model,
            device=args.device_map,
            limit=args.limit,
            tasks=args.tasks,
            batch_size=args.eval_bs,
            trust_remote_code=not args.disable_trust_remote_code,
            eval_model_dtype=args.eval_model_dtype,
            add_bos_token=args.add_bos_token,
        )
    else:
        eval(args)


# ============================================================================
# Command routing
# ============================================================================


def _normalize_cli_invocation(argv):
    """Normalize legacy invocation styles to (command, rest_argv)."""
    if argv and argv[0] in {"quantize", "list", "eval", "help"}:
        return argv[0], argv[1:]
    if "--list" in argv:
        normalized = list(argv)
        normalized.remove("--list")
        return "list", normalized
    if "--eval" in argv:
        normalized = list(argv)
        normalized.remove("--eval")
        return "eval", normalized
    return "quantize", list(argv)


def _print_help(topic=None):
    if topic == "quantize":
        build_quantize_parser(prog="auto_round quantize").print_help()
        return
    if topic == "list":
        build_list_parser(prog="auto_round list").print_help()
        return
    if topic == "eval":
        build_eval_parser(prog="auto_round eval").print_help()
        return
    build_root_parser().print_help()


def run():
    argv = list(sys.argv[1:])
    command, command_argv = _normalize_cli_invocation(argv)

    if command == "help":
        root_args = build_root_parser().parse_args(argv)
        _print_help(root_args.topic)
        return
    if command == "list":
        list_item(command_argv)
        return
    if command == "eval":
        run_eval(command_argv)
        return
    start(argv=command_argv)


def run_best():
    start("best")


def run_light():
    start("light")


def run_rtn():
    start("rtn")


def run_opt_rtn():
    start("opt_rtn")


def run_mllm():
    run()
