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
from auto_round.compressors.config_resolution import ConfigResolutionError
from auto_round.logger import logger

# Flags surfaced by `auto_round --help`. Entries are (dest, shown option strings):
# the overview is a curated subset, so a few flags list only their primary name
# instead of every alias. The full list stays behind `auto_round quantize --help`.
_OVERVIEW_ARGUMENTS: tuple[tuple[str, int], ...] = (
    ("model", 0),
    ("model_name", 1),
    ("scheme", 1),
    ("algorithm", 1),
    ("bits", 1),
    ("group_size", 1),
    ("iters", 1),
    ("nsamples", 1),
    ("seqlen", 1),
    ("batch_size", 1),
    ("device_map", 1),
    ("format", 1),
    ("output_dir", 1),
    ("asym", 1),
    ("ignore_layers", 1),
    ("tasks", 1),
)

_OVERVIEW_DESCRIPTION = """\
AutoRound command line interface: quantize, evaluate and inspect models.

commands:
  quantize      Quantize a model. This is the default command, so
                `auto_round --model <path> ...` works without naming it.
  list          List the supported quantization schemes/formats or algorithms.
  eval          Evaluate a quantized or original model.
  help [topic]  Show this overview, or the help of `quantize`/`list`/`eval`.

examples:
  auto_round --model facebook/opt-125m --bits 4 --group_size 128
  auto_round --model Qwen/Qwen3-0.6B --algorithm auto_round_v2 --iters 200
  auto_round --model Qwen/Qwen3-0.6B --algorithm awq,auto_round_v2
  auto_round --model <model> --schemes W2A16,W4A16 --bits 3 --ignore_scale_zp_bits
  auto_round eval --model <quantized_model> --tasks lambada_openai
  auto_round list alg

This is only the commonly used subset of the quantize arguments.
"""

_OVERVIEW_EPILOG = """\
see also:
  auto_round quantize --help            every quantize argument
  auto_round --algorithm <name> --help  the flags of one algorithm
  auto_round list alg                   the supported algorithms
  auto_round list format                the supported output formats
"""


def _overview_parser(prog: str = "auto_round") -> argparse.ArgumentParser:
    """Build the display-only parser behind the `auto_round --help` overview.

    The curated flags are redeclared from the real quantize parser rather than
    hard-coded, so the overview cannot drift from the actual help strings and
    defaults. Only rendering matters here: this parser never parses anything.
    """
    full = build_quantize_parser(prog=prog)
    by_dest: dict[str, argparse.Action] = {}
    for action in full._actions:
        if action.help is argparse.SUPPRESS:
            # Hidden legacy aliases (e.g. --avg_bits, which also has dest="bits")
            # must not shadow the documented flag they alias.
            continue
        by_dest.setdefault(action.dest, action)

    parser = argparse.ArgumentParser(
        prog=prog,
        add_help=False,
        usage="auto_round [<command>] [options]",
        description=_OVERVIEW_DESCRIPTION,
        epilog=_OVERVIEW_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_argument_group("common arguments")
    for dest, shown_options in _OVERVIEW_ARGUMENTS:
        action = by_dest.get(dest)
        if action is None:
            continue
        if not action.option_strings:
            group.add_argument(dest, nargs="?", help=action.help)
            continue
        options = list(action.option_strings[:shown_options]) if shown_options else list(action.option_strings)
        kwargs = {"help": action.help}
        if action.nargs == 0:
            # A boolean switch: render it as one instead of as a value-taking flag.
            if isinstance(action, argparse.BooleanOptionalAction):
                group.add_argument(*options[:1], action=argparse.BooleanOptionalAction, **kwargs)
            else:
                group.add_argument(*options, action="store_true", default=argparse.SUPPRESS, **kwargs)
        else:
            kwargs["default"] = argparse.SUPPRESS
            if action.metavar is not None:
                kwargs["metavar"] = action.metavar
            if action.nargs is not None:
                kwargs["nargs"] = action.nargs
            if action.choices is not None:
                kwargs["choices"] = action.choices
            group.add_argument(*options, **kwargs)
    return parser


def _print_overview_help() -> None:
    """Print the concise top-level `auto_round --help` overview."""
    _overview_parser().print_help()


def _requests_overview_help(argv: list[str]) -> bool:
    """True for a bare `auto_round --help`/`-h` that names no subcommand or topic."""
    return bool(argv) and all(token in ("-h", "--help") for token in argv)


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


def _normalize_scheme_list(raw) -> str | None:
    """Normalize --schemes/--options into a comma-joined scheme list.

    Accepts both space-separated (nargs="+" list) and comma-separated forms,
    e.g. 'W4A16 W8A16' or 'W4A16,W8A16' -> 'W4A16,W8A16'.
    """
    if raw is None:
        return None
    flat = ",".join(raw)  # handles list; each element may itself contain commas
    return ",".join(p.strip() for p in flat.split(",") if p.strip())


def _build_entry_base_kwargs(args, *, low_cpu_mem_usage, enable_torch_compile, layer_config) -> dict:
    return {
        "platform": args.platform,
        "format": getattr(args, "_api_format", args.format),
        "dataset": args.dataset,
        "seqlen": args.seqlen,
        "nsamples": args.nsamples,
        "batch_size": args.batch_size,
        "low_gpu_mem_usage": args.low_gpu_mem_usage,
        "low_cpu_mem_usage": low_cpu_mem_usage,
        "device_map": args.device_map,
        "enable_torch_compile": enable_torch_compile,
        "enable_deterministic_algorithms": args.enable_deterministic_algorithms,
        "seed": args.seed,
        "layer_config": layer_config,
        "model_dtype": args.model_dtype,
        "trust_remote_code": not args.disable_trust_remote_code,
        "static_kv_dtype": args.static_kv_dtype,
        "static_kv_granularity": args.static_kv_granularity,
        "static_attention_dtype": args.static_attention_dtype,
        "static_attention_granularity": args.static_attention_granularity,
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
        "calib_num_inference_steps": args.calib_num_inference_steps,
        "generator_seed": args.generator_seed,
        "diffusion_tuning_cache_size": args.diffusion_tuning_cache_size,
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
        from auto_round.export.formats import OutputFormat

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
        AlgorithmHandler.add_group(name, alg_group)
    mini.print_help()
    return True


def start(recipe="default", argv=None):
    recipe_defaults = RECIPES[recipe]
    argv = list(sys.argv[1:] if argv is None else argv)
    format_was_explicit = any(
        arg in {"--format", "--formats"} or arg.startswith(("--format=", "--formats=")) for arg in argv
    )

    if _print_algorithm_help(argv):
        return

    parser = build_quantize_parser(prog="auto_round quantize")
    args = parser.parse_args(argv)
    args._api_format = args.format if format_was_explicit or args.model_free else None

    # Apply recipe defaults for fields the user didn't set
    for key, value in recipe_defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    if args.model_free:
        args.iters = 0

    tune(args)


def tune(args):
    assert args.model or args.model_name, "[model] or --model MODEL_NAME should be set."
    if args.model is None:
        args.model = args.model_name
    if args.eval_bs is None:
        args.eval_bs = "auto"

    if getattr(args, "num_hidden_layers", None) is not None:
        # Debug helper: load only the first N decoder layers. Propagated to the
        # model loader via an env var so every load path picks it up.
        from auto_round import envs

        envs.set_config(AR_DEBUG_LAYER_NUM=args.num_hidden_layers)

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

    if getattr(args, "enable_alg_ext", False):
        logger.warning(
            "`--enable_alg_ext` is deprecated; use `--algorithm auto_round_v2` instead. "
            "Both select the same algorithm and remain equivalent."
        )

    if args.format is None:
        args.format = "auto_round"

    formats = args.format.lower().replace(" ", "").split(",")
    from auto_round.utils import SUPPORTED_FORMATS

    for fmt in formats:
        if fmt not in SUPPORTED_FORMATS:
            raise ValueError(f"{fmt} is not supported, we only support {SUPPORTED_FORMATS}")

    if any("llm_compressor" in fmt for fmt in formats):
        from auto_round.export.export_to_llmcompressor import check_compressed_tensors_supported

        try:
            check_compressed_tensors_supported(raise_error=True)
        except ImportError as error:
            logger.error(str(error))
            raise SystemExit(1) from None

    if "auto_gptq" in args.format and args.asym is True:
        logger.warning(
            "the auto_gptq kernel has issues with asymmetric quantization. "
            "It is recommended to use sym quantization or --format='auto_round'"
        )

    if "marlin" in args.format and args.asym is True:
        raise RuntimeError("marlin backend only supports sym quantization, please remove --asym")

    from auto_round.utils import get_device_and_parallelism

    device_str, use_auto_mapping = get_device_and_parallelism(args.device_map)

    if args.enable_torch_compile is False:
        logger.info("`torch.compile` is explicitly disabled with `--disable_torch_compile`.")

    model_name = args.model
    if model_name[-1] == "/":
        model_name = model_name[:-1]
    logger.info(f"start to quantize {model_name}")

    from auto_round import AutoRound
    from auto_round.compressors.base import BaseOrchestrator as BaseCompressor

    if "bloom" in model_name:
        args.low_gpu_mem_usage = False

    if args.quant_lm_head:
        for fmt in formats:
            if "auto_round" not in fmt and "fake" not in fmt and "mlx" not in fmt:
                auto_round_formats = [s for s in SUPPORTED_FORMATS if s.startswith("auto_round") or s == "mlx"]
                raise ValueError(
                    f"{fmt} is not supported for lm-head quantization, please change to {auto_round_formats}"
                )

    enable_torch_compile = args.enable_torch_compile
    scheme = args.scheme.upper()

    from auto_round.schemes import PRESET_SCHEMES

    if scheme not in PRESET_SCHEMES:
        raise ValueError(f"{scheme} is not supported. only {PRESET_SCHEMES.keys()} are supported ")

    from auto_round.utils import parse_layer_config_arg

    layer_config = {}
    if args.layer_config:
        layer_config = parse_layer_config_arg(args.layer_config)
        args.layer_config = layer_config

    low_cpu_mem_usage = True
    if args.disable_low_cpu_mem_usage:
        low_cpu_mem_usage = False

    from auto_round.auto_scheme import AutoScheme

    # Supplying multiple schemes enables AutoScheme: the schemes are the
    # candidate options and --bits is the average target bits.
    # --options/--avg_bits/--target_bits are deprecated aliases that argparse
    # merges into --schemes/--bits (see _LegacyAliasAction in parser.py); they
    # stay functional but are hidden from --help.
    scheme_options = None
    if args.schemes is not None:
        if args.scheme.upper() != "W4A16":
            raise ValueError("`--scheme` and `--schemes` cannot be used together, please use only `--schemes`")
        scheme_options = _normalize_scheme_list(args.schemes)

    # Normalize --shared_layers: supports three forms per invocation:
    #   - all bare tokens (no commas): treated as one group
    #     e.g. --shared_layers l1 l2       → [['l1', 'l2']]
    #   - comma-containing tokens: each token is its own group
    #     e.g. --shared_layers l1,l2 l3,l4 → [['l1','l2'], ['l3','l4']]
    #   - single comma token: one group
    #     e.g. --shared_layers l1,l2       → [['l1', 'l2']]
    # Multiple --shared_layers flags always produce multiple groups (one per flag when each flag
    # yields exactly one group, or more if a flag contains comma-tokens).
    if args.shared_layers is not None:
        normalized_groups = []
        for invocation in args.shared_layers:
            # invocation is a list of tokens from one --shared_layers flag (nargs="+")
            if any("," in token for token in invocation):
                # at least one comma token → each token becomes its own group
                for token in invocation:
                    group = [p.strip() for p in token.split(",") if p.strip()]
                    if group:
                        normalized_groups.append(group)
            else:
                # all bare names → bundle into a single group
                group = [p.strip() for p in invocation if p.strip()]
                if group:
                    normalized_groups.append(group)
        args.shared_layers = normalized_groups or None

    if scheme_options is not None:
        if args.bits is None:
            raise ValueError("please set --bits for auto scheme")
        if enable_torch_compile is False:
            logger.warning(
                "`torch.compile` is disabled with AutoScheme. "
                "Enabling it (the default) is strongly recommended to save VRAM."
            )
        scheme = AutoScheme(
            options=scheme_options,
            avg_bits=args.bits,
            shared_layers=args.shared_layers,
            ignore_scale_zp_bits=args.ignore_scale_zp_bits,
            low_gpu_mem_usage=True,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

    common_kwargs = _extract_common_quantization_kwargs(args)
    if scheme_options is not None:
        # `bits` is the AutoScheme average target, not a scheme override:
        # the candidate options carry their own bit widths.
        common_kwargs.pop("bits", None)
    else:
        # Without --schemes, `--bits` is a plain weight bit width and must be an integer.
        bits = common_kwargs.get("bits")
        if bits is not None:
            if float(bits).is_integer():
                common_kwargs["bits"] = int(bits)
            else:
                raise ValueError(
                    "`--bits` must be an integer for weight quantization; "
                    "use `--schemes` to set a fractional AutoScheme average target"
                )
    alg_configs = AlgorithmHandler.build_configs(args, common_kwargs)

    from auto_round.utils import clear_memory

    autoround: BaseCompressor = AutoRound(
        model_name,
        scheme=scheme,
        alg_configs=alg_configs if len(alg_configs) > 1 else alg_configs[0],
        **_to_autoround_kwargs(
            args,
            low_cpu_mem_usage=low_cpu_mem_usage,
            enable_torch_compile=enable_torch_compile,
            layer_config=layer_config,
        ),
    )

    model, folders = autoround.quantize_and_save(  # pylint: disable=no-member
        args.output_dir,
        format=getattr(args, "_api_format", args.format),
        max_shard_size=args.max_shard_size,
    )
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
    if args.model_name is None:
        args.model_name = args.model
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
            num_fewshot=args.num_fewshot,
            gen_kwargs=args.eval_gen_kwargs,
            fewshot_as_multiturn=args.fewshot_as_multiturn,
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
    _print_overview_help()


def run():
    try:
        argv = list(sys.argv[1:])
        if _requests_overview_help(argv):
            _print_overview_help()
            return
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
    except ConfigResolutionError as error:
        logger.error(str(error))
        raise SystemExit(2) from error


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
