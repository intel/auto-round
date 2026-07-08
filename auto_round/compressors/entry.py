# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Callable, Optional, Union

import torch

from auto_round.algorithms.config_resolver import split_quantization_configs
from auto_round.algorithms.quantization.config import QuantizationConfig
from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig, RTNConfig
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.algorithms.registry import normalize_algorithm_config, resolve_alg_config
from auto_round.algorithms.transforms import normalize_rotation_config as _normalize_rotation_alg_config
from auto_round.algorithms.transforms.awq.config import AWQConfig
from auto_round.algorithms.transforms.hadamard.config import RotationConfig as _NewArchRotationConfig
from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
from auto_round.compressors.base import BaseCompressor
from auto_round.compressors.data_driven import CalibratedRTNCompressor, DataDrivenCompressor
from auto_round.compressors.utils import check_need_act_calibration
from auto_round.compressors.zero_shot import ZeroShotCompressor
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme, parse_scheme
from auto_round.utils.device_manager import normalize_default_device_map

_ENTRY_ROUTE_KWARGS = {"model_free", "disable_model_free", "disable_opt_rtn"}
_ENTRY_COMPRESSOR_KWARGS = {"scale_dtype", "ignore_layers", "quant_lm_head", "to_quant_block_names"}
_ENTRY_BASE_KWARGS = {
    "format",
    "dataset",
    "batch_size",
    "model_dtype",
    "trust_remote_code",
    "amp",
    "nblocks",
    "disable_deterministic_algorithms",
    "enable_deterministic_algorithms",
    "static_kv_dtype",
    "static_attention_dtype",
}
_ENTRY_MLLM_KWARGS = {"processor", "image_processor", "template", "extra_data_dir", "quant_nontext_module"}
_ENTRY_DIFFUSION_KWARGS = {"guidance_scale", "num_inference_steps", "generator_seed"}
_ENTRY_ALLOWED_KWARGS = (
    _ENTRY_ROUTE_KWARGS | _ENTRY_COMPRESSOR_KWARGS | _ENTRY_BASE_KWARGS | _ENTRY_MLLM_KWARGS | _ENTRY_DIFFUSION_KWARGS
)


def filter_supported_entry_kwargs(kwargs: dict[str, Any], *, context: str) -> dict[str, Any]:
    """Return only kwargs supported by the new entry API.

    Unsupported kwargs are ignored with a warning so callers can cleanly migrate
    without leaking old-API parameters into compressor constructors.
    """

    supported = {}
    unknown = []
    for key, value in kwargs.items():
        if key in _ENTRY_ALLOWED_KWARGS:
            supported[key] = value
        else:
            unknown.append(key)
    if unknown:
        logger.warning_once(
            "%s received unsupported kwargs %s. They will be ignored.",
            context,
            ", ".join(sorted(unknown)),
        )
    return supported


def _split_entry_kwargs(kwargs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Partition new-entry kwargs by ownership."""

    kwargs = filter_supported_entry_kwargs(kwargs, context="AutoRound entry")
    buckets = {
        "route": {},
        "compressor": {},
        "base": {},
        "mllm": {},
        "diffusion": {},
    }
    for key, value in kwargs.items():
        if key in _ENTRY_ROUTE_KWARGS:
            buckets["route"][key] = value
        elif key in _ENTRY_COMPRESSOR_KWARGS:
            buckets["compressor"][key] = value
        elif key in _ENTRY_BASE_KWARGS:
            buckets["base"][key] = value
        elif key in _ENTRY_MLLM_KWARGS:
            buckets["mllm"][key] = value
        elif key in _ENTRY_DIFFUSION_KWARGS:
            buckets["diffusion"][key] = value
    return buckets


def _preview_resolved_attrs(config, scheme=None) -> dict:
    """Resolve scheme attributes without mutating config, for routing decisions.

    Called in ``AutoRound.__new__`` before the concrete compressor class is
    chosen.  ``SchemeMixin.resolve_scheme()`` will do the authoritative
    resolution later; this is just a lightweight preview so routing logic
    (``enable_imatrix``, ``needs_act_calib``, etc.) can use the correct values
    even when the user specified only ``scheme=`` without explicit bit/dtype args.

    Returns:
        dict: resolved attributes (may be empty if scheme cannot be previewed).
    """
    if isinstance(scheme, AutoScheme):
        # AutoScheme needs model info — cannot preview, rely on raw config attrs
        return {}
    scheme_attr_names = tuple(config._scheme_fields)
    user_overrides = {k: getattr(config, k) for k in scheme_attr_names if getattr(config, k, None) is not None}
    try:
        _, _, final_attrs = parse_scheme(scheme, user_overrides)
        return final_attrs
    except Exception:
        return {}


def _eager_validate_scheme(config, scheme=None) -> None:
    """Eagerly validate scheme/config constraints at construction time.

    Mirrors the old-arch ``_check_configs()`` call in ``BaseCompressor.__init__``.
    Raises ``ValueError`` or ``NotImplementedError`` immediately if the scheme
    contains config-only invalid combinations (e.g. tuple group_size with non-fp8
    weight dtype) so that callers get a fast failure rather than a deferred error
    buried inside ``post_init()``.

    ``AutoScheme`` is skipped because it requires model information.
    """
    if isinstance(scheme, AutoScheme):
        return

    scheme_attr_names = tuple(config._scheme_fields)
    user_overrides = {k: getattr(config, k) for k in scheme_attr_names if getattr(config, k, None) is not None}
    try:
        _, _, final_attrs = parse_scheme(scheme, user_overrides)
    except (ValueError, NotImplementedError):
        raise
    except Exception:
        return  # Other parse errors are deferred to post_init

    import copy

    temp_config = copy.copy(config)
    if hasattr(config, "scheme"):
        temp_config.scheme = config.scheme.copy()
        temp_config._user_set_scheme_fields = set(getattr(config, "_user_set_scheme_fields", set()))
    for key, value in final_attrs.items():
        setattr(temp_config, key, value)
    temp_config.check_config()  # raises ValueError / NotImplementedError if invalid


# ---------------------------------------------------------------------------
# Compressor-class registry
# ---------------------------------------------------------------------------
# Maps (model_type, base_class_name) → combined class, created lazily.
_COMPRESSOR_REGISTRY: dict[tuple[str, str], type] = {}


def _get_compressor_class(model_type: str, base_cls: type) -> type:
    """Return the compressor class for *base_cls* wired with the right model-type Mixin.

    For ``model_type == "llm"`` the bare *base_cls* is returned unchanged.
    For ``"mllm"`` and ``"diffusion"`` the corresponding Mixin is prepended via
    :func:`type` and the result is cached in ``_COMPRESSOR_REGISTRY`` so that
    each ``(model_type, base_cls)`` pair is created at most once per process.
    """
    if model_type == "llm":
        return base_cls
    key = (model_type, base_cls.__name__)
    if key in _COMPRESSOR_REGISTRY:
        return _COMPRESSOR_REGISTRY[key]
    if model_type == "mllm":
        from auto_round.compressors.mllm_mixin import MLLMMixin

        mixin = MLLMMixin
    elif model_type == "diffusion":
        from auto_round.compressors.diffusion_mixin import DiffusionMixin

        mixin = DiffusionMixin
    else:
        return base_cls
    combined = type(f"{model_type.capitalize()}{base_cls.__name__}", (mixin, base_cls), {})
    _COMPRESSOR_REGISTRY[key] = combined
    return combined


def is_weight_scheme(scheme: Union[str, dict, AutoScheme, object]) -> bool:
    if isinstance(scheme, str):
        return scheme.upper().startswith("W")
    if isinstance(scheme, dict):
        return all(isinstance(s, str) and s.upper().startswith("W") for s in scheme.values())
    if isinstance(scheme, AutoScheme):
        opts = scheme.options
        if isinstance(opts, (list, tuple)):
            return all(isinstance(s, str) and s.upper().startswith("W") for s in opts)
        if isinstance(opts, str):
            return opts.upper().startswith("W")
    return False


def is_gguf_k_target(value: Union[str, AutoScheme, object]) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized.startswith("gguf:") and "_k" in normalized
    if isinstance(value, AutoScheme):
        opts = value.options
        if isinstance(opts, str):
            opts = [opts]
        if isinstance(opts, (list, tuple)):
            return any(isinstance(opt, str) and is_gguf_k_target(opt) for opt in opts)
    return False


def _resolve_quant_config_for_routing(alg_configs) -> tuple[list, list, QuantizationConfig]:
    preprocessor_configs, block_quant_configs = split_quantization_configs(alg_configs)
    if len(block_quant_configs) == 0 and preprocessor_configs:
        from auto_round.algorithms.quantization.rtn.config import RTNConfig as _RTNConfig

        return preprocessor_configs, block_quant_configs, _RTNConfig()
    if len(block_quant_configs) > 1:
        raise ValueError(
            f"Only one block-quantization config is allowed, but got {len(block_quant_configs)}: "
            f"{[type(c).__name__ for c in block_quant_configs]}"
        )
    if len(block_quant_configs) == 1:
        return preprocessor_configs, block_quant_configs, block_quant_configs[0]
    raise ValueError(
        "At least one quantization algorithm config is required. "
        "Pass a block quantizer such as RTNConfig or SignRoundConfig, "
        "or a quantization preprocessor such as AWQConfig."
    )


def _build_model_type_ctor_kwargs(model, base_kwargs, mllm_kwargs, diffusion_kwargs) -> tuple[str, dict[str, Any]]:
    from auto_round.utils.model import detect_model_type

    model_type = detect_model_type(model)
    has_multimodal_assets = mllm_kwargs.get("processor") is not None or mllm_kwargs.get("image_processor") is not None
    if has_multimodal_assets and model_type != "mllm":
        model_type = "mllm"

    ctor_kwargs = dict(base_kwargs)
    if model_type == "mllm":
        ctor_kwargs.update(mllm_kwargs)
    if model_type == "diffusion":
        ctor_kwargs.update(diffusion_kwargs)
    return model_type, ctor_kwargs


def _select_rtn_compressor_base_cls(quant_config: RTNConfig, scheme, format, base_kwargs) -> type:
    enable_imatrix = False
    disable_opt_rtn = getattr(quant_config, "disable_opt_rtn", False)

    # Preview resolved scheme attrs once (authoritative resolution happens later).
    resolved_attrs = _preview_resolved_attrs(quant_config, scheme)

    # Auto-disable rtn optimization for W8A16/W8A8-equivalent resolved schemes,
    # unless the user already set disable_opt_rtn explicitly.
    if getattr(quant_config, "orig_disable_opt_rtn", None) is None:
        bits = resolved_attrs.get("bits", getattr(quant_config, "bits", None))
        act_bits = resolved_attrs.get("act_bits", getattr(quant_config, "act_bits", None))
        data_type = resolved_attrs.get("data_type", getattr(quant_config, "data_type", None))
        if bits is not None and bits >= 8 and act_bits is not None and act_bits >= 8 and data_type == "int":
            logger.warning("`disable_opt_rtn` is turned on for W8A16/W8A8 quantization to improve efficiency.")
            disable_opt_rtn = True
            quant_config.disable_opt_rtn = True

    if not disable_opt_rtn:
        has_gguf_k = is_gguf_k_target(format) or is_gguf_k_target(scheme)
        if has_gguf_k:
            enable_imatrix = True
        else:
            sym = resolved_attrs.get("sym", getattr(quant_config, "sym", None))
            data_type = resolved_attrs.get("data_type", getattr(quant_config, "data_type", "") or "")
            bits = resolved_attrs.get("bits", getattr(quant_config, "bits", None))
            if sym is not None and sym is False:
                enable_imatrix = False
            elif data_type == "int" and (bits is None or bits < 8):
                enable_imatrix = True
            elif is_weight_scheme(scheme):
                enable_imatrix = True

    act_bits = resolved_attrs.get("act_bits", getattr(quant_config, "act_bits", None))
    act_data_type = resolved_attrs.get("act_data_type", getattr(quant_config, "act_data_type", None))
    act_dynamic = resolved_attrs.get("act_dynamic", getattr(quant_config, "act_dynamic", None))
    is_act_quantize = act_bits is not None and act_bits <= 8
    needs_act_calib = is_act_quantize and check_need_act_calibration(
        act_dynamic,
        act_data_type,
        act_bits if act_bits is not None else 16,
        static_kv_dtype=base_kwargs.get("static_kv_dtype"),
        static_attention_dtype=base_kwargs.get("static_attention_dtype"),
    )

    # AutoScheme always requires calibration data for delta-loss based scheme
    # selection, regardless of whether imatrix is needed.
    quant_config.enable_imatrix = enable_imatrix
    if enable_imatrix or needs_act_calib or isinstance(scheme, AutoScheme):
        if not isinstance(quant_config, OptimizedRTNConfig):
            quant_config.__class__ = OptimizedRTNConfig
        return CalibratedRTNCompressor

    if isinstance(quant_config, OptimizedRTNConfig):
        quant_config.__class__ = RTNConfig
    return ZeroShotCompressor


class AutoRound(object):

    @classmethod
    def _resolve_config(cls, config: Union[str, object, list]) -> Union[object, list[object]]:
        """Convert string alias(es) to the corresponding config instance(s) with default parameters."""
        if isinstance(config, str):
            return resolve_alg_config(config)
        if isinstance(config, list):
            return [cls._resolve_config(c) for c in config]
        return config

    def __new__(
        cls,
        model: Union[torch.nn.Module, str],
        scheme="W4A16",
        alg_configs: Union[str, object, list[Union[str, object]]] = None,
        tokenizer=None,
        platform="hf",
        format=None,
        low_gpu_mem_usage: bool = False,
        device_map: Union[str, torch.device, int, dict] = 0,
        iters: int = None,
        gradient_accumulate_steps: int = 1,
        enable_torch_compile: bool = False,
        seed: int = 42,
        low_cpu_mem_usage: bool = True,
        layer_config=None,
        nsamples: int = None,
        seqlen: int = None,
        **kwargs,
    ) -> "BaseCompressor":
        from auto_round.utils.model import is_model_free_route

        if alg_configs is None:
            alg_configs = "auto_round"

        device_map = normalize_default_device_map(device_map)
        split_kwargs = _split_entry_kwargs(kwargs)
        route_kwargs = dict(split_kwargs["route"])
        compressor_kwargs = dict(split_kwargs["compressor"])
        base_kwargs = dict(split_kwargs["base"])
        mllm_kwargs = dict(split_kwargs["mllm"])
        diffusion_kwargs = dict(split_kwargs["diffusion"])

        # Resolve string alias(es) to config instance(s) before routing.
        alg_configs = cls._resolve_config(alg_configs)
        if isinstance(alg_configs, list):
            alg_configs = [normalize_algorithm_config(cfg) for cfg in alg_configs]
        else:
            alg_configs = normalize_algorithm_config(alg_configs)
        configs_for_routing = alg_configs if isinstance(alg_configs, list) else [alg_configs]
        preprocessor_configs, _, quant_config = _resolve_quant_config_for_routing(configs_for_routing)

        # Model-free routing is now supported directly by the new entry path.
        model_free_iters = 0 if isinstance(quant_config, RTNConfig) else getattr(quant_config, "iters", None)
        model_free_disable_opt_rtn = getattr(quant_config, "disable_opt_rtn", None)
        route_decision_kwargs = dict(route_kwargs, format=format)
        if is_model_free_route(model, scheme, model_free_iters, model_free_disable_opt_rtn, route_decision_kwargs):
            from auto_round.compressors.model_free import ModelFreeCompressor

            if not isinstance(model, str):
                raise ValueError("model_free=True requires `model` to be a HuggingFace ID or local path string.")
            if not bool(route_kwargs.get("model_free", False)):
                logger.info(
                    "Auto-routing to model-free quantization "
                    "(iters=0, disable_opt_rtn=True, supported scheme). "
                    "Pass disable_model_free=True to use the regular flow."
                )
            return ModelFreeCompressor(
                model_name_or_path=model,
                scheme=scheme,
                layer_config=layer_config,
                tokenizer=tokenizer,
                device_map=device_map,
                **compressor_kwargs,
                **base_kwargs,
                **mllm_kwargs,
                **diffusion_kwargs,
                **route_kwargs,
            )

        # Eagerly validate scheme constraints that do not require model info.
        # This mirrors old-arch _check_configs() called at __init__ time so that
        # callers get ValueError/NotImplementedError on construction, not deferred.
        _eager_validate_scheme(quant_config, scheme)

        local_args = dict(
            model=model,
            tokenizer=tokenizer,
            platform=platform,
            format=format,
            scheme=scheme,
            low_gpu_mem_usage=low_gpu_mem_usage,
            device_map=device_map,
            iters=iters,
            gradient_accumulate_steps=gradient_accumulate_steps,
            enable_torch_compile=enable_torch_compile,
            seed=seed,
            low_cpu_mem_usage=low_cpu_mem_usage,
            layer_config=layer_config,
            nsamples=nsamples,
            seqlen=seqlen,
            **compressor_kwargs,
        )
        model_type, ctor_kwargs = _build_model_type_ctor_kwargs(model, base_kwargs, mllm_kwargs, diffusion_kwargs)

        # Preprocessor algorithms (AWQ, …) require a data-driven host so that
        # the per-block preprocessor lifecycle (prepare_block_group ->
        # block_forward_hooks -> pre_quantize_block -> pre_quantize_block ->
        # post_quantize_block) actually runs.  CalibratedRTNCompressor's
        # Preprocessor algorithms require DataDrivenCompressor for per-block lifecycle hooks.
        # The pipeline auto-appends RTN when no block_quantizer is supplied.
        if preprocessor_configs:
            return _get_compressor_class(model_type, DataDrivenCompressor)(alg_configs, **local_args, **ctor_kwargs)

        if isinstance(quant_config, SignRoundConfig):
            return _get_compressor_class(model_type, DataDrivenCompressor)(alg_configs, **local_args, **ctor_kwargs)

        elif isinstance(quant_config, RTNConfig):
            base_cls = _select_rtn_compressor_base_cls(quant_config, scheme, format, base_kwargs)
            return _get_compressor_class(model_type, base_cls)(alg_configs, **local_args, **ctor_kwargs)


class AutoRoundCompatible:
    """AutoRoundCompatible wrapper class for backward compatibility.

    This class provides the same API as the old AutoRoundCompatible class but internally
    uses the new AutoRound architecture with Mixin pattern.

    Args:
        model: Model object or model name to load
        tokenizer: Tokenizer for text processing
        platform: Platform to download model ("hf" or "model_scope")
        scheme: Quantization scheme (str, dict, or QuantizationScheme)
        layer_config: Layer-wise quantization config
        dataset: Calibration data
        iters: Optimization iterations
        seqlen: Calibration sequence length
        nsamples: Number of calibration samples
        batch_size: Calibration batch size
        gradient_accumulate_steps: Gradient accumulation steps
        low_gpu_mem_usage: Lower GPU memory mode
        device_map: Device map for each module
        enable_torch_compile: Enable torch.compile
        seed: Random seed
        low_cpu_mem_usage: Lower CPU memory mode
        **kwargs: Additional arguments (bits, group_size, sym, etc.)

    Example:
        >>> # Old API - still works
        >>> from auto_round.compressors.entry import AutoRoundCompatible
        >>> autoround = AutoRoundCompatible(
        ...     model="/models/opt-125m",
        ...     bits=4,
        ...     group_size=128,
        ...     iters=200,
        ... )
        >>> quantized_model, layer_config = autoround.quantize()
    """

    SKIP_ARGS = ("local_args", "kwargs", "cls", "config")

    bits: int | None
    group_size: int | None
    sym: bool | None
    data_type: str | None
    act_bits: int | None
    act_group_size: int | None
    act_sym: bool | None
    act_data_type: str | None
    act_dynamic: bool | None
    super_bits: int | None
    super_group_size: int | None

    @staticmethod
    def _pop_config_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Extract old-API config kwargs and split them by config type."""
        common_keys = ("super_bits", "super_group_size")
        auto_round_only_keys = (
            "nblocks",
            "enable_alg_ext",
            "lr_scheduler",
            "not_use_best_mse",
            "dynamic_max_gap",
            "optimizer",
            "enable_adam",
            "momentum",
        )
        common_kwargs = {}
        auto_round_kwargs = {}
        for key in common_keys:
            if key in kwargs:
                common_kwargs[key] = kwargs.pop(key)
        for key in auto_round_only_keys:
            if key in kwargs:
                auto_round_kwargs[key] = kwargs.pop(key)
        return common_kwargs, auto_round_kwargs

    @staticmethod
    def _pop_compressor_only_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            "scale_dtype": kwargs.pop("scale_dtype", None),
            "ignore_layers": kwargs.pop("ignore_layers", ""),
            "quant_lm_head": kwargs.pop("quant_lm_head", False),
            "to_quant_block_names": kwargs.pop("to_quant_block_names", None),
        }

    @staticmethod
    def _resolve_compat_algorithm(algorithm, iters) -> str:
        if algorithm and algorithm.lower() == "awq":
            return "awq"
        if (algorithm and algorithm.lower() == "rtn") or iters == 0:
            return "rtn"
        return "signround"

    @staticmethod
    def _pop_shared_quant_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            "bits": kwargs.pop("bits", None),
            "group_size": kwargs.pop("group_size", None),
            "sym": kwargs.pop("sym", None),
            "data_type": kwargs.pop("data_type", None),
            "act_bits": kwargs.pop("act_bits", None),
            "act_group_size": kwargs.pop("act_group_size", None),
            "act_sym": kwargs.pop("act_sym", None),
            "act_data_type": kwargs.pop("act_data_type", None),
            "act_dynamic": kwargs.pop("act_dynamic", None),
        }

    @staticmethod
    def _build_awq_config(
        shared_quant_kwargs: dict[str, Any],
        *,
        seqlen,
        nsamples,
        batch_size,
        kwargs,
        common_config_kwargs,
    ):
        return AWQConfig(
            **shared_quant_kwargs,
            duo_scaling=kwargs.pop("duo_scaling", True),
            n_grid=kwargs.pop("n_grid", 20),
            seqlen=seqlen,
            nsamples=nsamples,
            batch_size=batch_size,
            mappings=kwargs.pop("mappings", None),
            **common_config_kwargs,
        )

    @staticmethod
    def _build_rtn_config(shared_quant_kwargs: dict[str, Any], *, kwargs, common_config_kwargs):
        cfg = RTNConfig(
            **shared_quant_kwargs,
            disable_opt_rtn=kwargs.pop("disable_opt_rtn", None),
            enable_opt_rtn=kwargs.pop("enable_opt_rtn", None),
            **common_config_kwargs,
        )
        return normalize_algorithm_config(cfg)

    @staticmethod
    def _build_signround_config(
        shared_quant_kwargs: dict[str, Any],
        *,
        iters,
        gradient_accumulate_steps,
        kwargs,
        common_config_kwargs,
        auto_round_config_kwargs,
    ):
        cfg = SignRoundConfig(
            **shared_quant_kwargs,
            iters=iters,
            gradient_accumulate_steps=gradient_accumulate_steps,
            lr=kwargs.pop("lr", None),
            minmax_lr=kwargs.pop("minmax_lr", None),
            enable_minmax_tuning=kwargs.pop("enable_minmax_tuning", True),
            enable_norm_bias_tuning=kwargs.pop("enable_norm_bias_tuning", False),
            enable_quanted_input=kwargs.pop("enable_quanted_input", True),
            **common_config_kwargs,
            **auto_round_config_kwargs,
        )
        return normalize_algorithm_config(cfg)

    @classmethod
    def _build_alg_config(
        cls,
        *,
        algorithm,
        iters,
        gradient_accumulate_steps,
        seqlen,
        nsamples,
        batch_size,
        kwargs,
        common_config_kwargs,
        auto_round_config_kwargs,
    ):
        alg_name = cls._resolve_compat_algorithm(algorithm, iters)
        shared_quant_kwargs = cls._pop_shared_quant_kwargs(kwargs)

        if alg_name == "awq":
            return cls._build_awq_config(
                shared_quant_kwargs,
                seqlen=seqlen,
                nsamples=nsamples,
                batch_size=batch_size,
                kwargs=kwargs,
                common_config_kwargs=common_config_kwargs,
            )
        if alg_name == "rtn":
            return cls._build_rtn_config(
                shared_quant_kwargs,
                kwargs=kwargs,
                common_config_kwargs=common_config_kwargs,
            )
        return cls._build_signround_config(
            shared_quant_kwargs,
            iters=iters,
            gradient_accumulate_steps=gradient_accumulate_steps,
            kwargs=kwargs,
            common_config_kwargs=common_config_kwargs,
            auto_round_config_kwargs=auto_round_config_kwargs,
        )

    @staticmethod
    def _build_entry_forward_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        format_name = kwargs.pop("format", None)
        rotation_config = kwargs.pop("rotation_config", None)
        mllm_kwargs = {
            "processor": kwargs.pop("processor", None),
            "image_processor": kwargs.pop("image_processor", None),
            "template": kwargs.pop("template", None),
            "extra_data_dir": kwargs.pop("extra_data_dir", None),
            "quant_nontext_module": kwargs.pop("quant_nontext_module", False),
        }
        diffusion_kwargs = {
            "guidance_scale": kwargs.pop("guidance_scale", 7.5),
            "num_inference_steps": kwargs.pop("num_inference_steps", 50),
            "generator_seed": kwargs.pop("generator_seed", None),
        }
        return {
            "format": format_name,
            "rotation_config": rotation_config,
            **mllm_kwargs,
            **diffusion_kwargs,
            **kwargs,
        }

    def __new__(
        cls,
        model: Union[torch.nn.Module, str],
        tokenizer=None,
        platform: str = "hf",
        scheme: Union[str, dict, QuantizationScheme, AutoScheme] = "W4A16",
        layer_config: dict[str, Union[str, dict, QuantizationScheme]] = None,
        dataset: Union[str, list, tuple, torch.utils.data.DataLoader] = "NeelNanda/pile-10k",
        iters: int = 200,
        seqlen: int = 2048,
        nsamples: int = 128,
        batch_size: int = 8,
        gradient_accumulate_steps: int = 1,
        low_gpu_mem_usage: bool = False,
        device_map: Union[str, torch.device, int, dict] = 0,
        enable_torch_compile: bool = False,
        seed: int = 42,
        low_cpu_mem_usage: bool = True,
        algorithm: str = None,
        **kwargs,
    ) -> "BaseCompressor":
        """Create AutoRoundCompatible instance using new AutoRound architecture.

        This method translates old AutoRoundCompatible API to new AutoRound API.
        """
        from auto_round.utils import is_diffusion_model, is_mllm_model
        from auto_round.utils.model import is_model_free_route

        device = kwargs.pop("device", None)
        if device is not None:
            logger.warning_once("`device` is deprecated, please use `device_map` instead")
            if device_map in (None, 0):
                device_map = device

        # ---- Model-free fast-path detection --------------------------------
        if is_model_free_route(model, scheme, iters, kwargs.get("disable_opt_rtn"), kwargs):
            from auto_round.compressors.model_free import ModelFreeCompressor

            compressor_only_kwargs = cls._pop_compressor_only_kwargs(kwargs)

            if not isinstance(model, str):
                raise ValueError("model_free=True requires `model` to be a HuggingFace ID or local path string.")
            if not bool(kwargs.get("model_free", False)):
                logger.info(
                    "Auto-routing to model-free quantization "
                    "(iters=0, disable_opt_rtn=True, supported scheme). "
                    "Pass disable_model_free=True to use the regular flow."
                )
            return ModelFreeCompressor(
                model_name_or_path=model,
                scheme=scheme,
                layer_config=layer_config,
                tokenizer=tokenizer,
                device_map=device_map,
                **compressor_only_kwargs,
                **kwargs,
            )
        # --------------------------------------------------------------------

        compressor_only_kwargs = cls._pop_compressor_only_kwargs(kwargs)
        common_config_kwargs, auto_round_config_kwargs = cls._pop_config_kwargs(kwargs)

        config = cls._build_alg_config(
            algorithm=algorithm,
            iters=iters,
            gradient_accumulate_steps=gradient_accumulate_steps,
            seqlen=seqlen,
            nsamples=nsamples,
            batch_size=batch_size,
            kwargs=kwargs,
            common_config_kwargs=common_config_kwargs,
            auto_round_config_kwargs=auto_round_config_kwargs,
        )

        forward_kwargs = cls._build_entry_forward_kwargs(kwargs)
        format_name = forward_kwargs.pop("format", None)
        _rotation_config_raw = forward_kwargs.pop("rotation_config", None)
        if _rotation_config_raw is not None:
            _rc = _normalize_rotation_alg_config(_rotation_config_raw)
            if _rc is None:
                _rc = _NewArchRotationConfig()
            config = [config, _rc]

        # Check model type for logging (use warning_once to avoid repeating for every block
        # when called from LLM-Compressor which instantiates AutoRound per block)
        if is_mllm_model(model, platform=platform):
            logger.info("Using MLLM mode for multimodal model.")
        elif is_diffusion_model(model):
            logger.info("Using Diffusion mode for diffusion model.")
        else:
            logger.info("Using LLM mode.")

        # Create AutoRound instance using new architecture
        compressor = AutoRound(
            model,
            scheme,
            config,
            tokenizer=tokenizer,
            platform=platform,
            format=format_name,
            dataset=dataset,
            iters=iters,
            gradient_accumulate_steps=gradient_accumulate_steps,
            low_gpu_mem_usage=low_gpu_mem_usage,
            device_map=device_map,
            enable_torch_compile=enable_torch_compile,
            seed=seed,
            low_cpu_mem_usage=low_cpu_mem_usage,
            layer_config=layer_config,
            nsamples=nsamples,
            seqlen=seqlen,
            batch_size=batch_size,
            **compressor_only_kwargs,
            **forward_kwargs,
        )

        return compressor
