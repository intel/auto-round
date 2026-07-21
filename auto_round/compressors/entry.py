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
from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
from auto_round.compressors.base import BaseCompressor
from auto_round.compressors.data_driven import CalibratedRTNCompressor, DataDrivenCompressor
from auto_round.compressors.entry_contract import filter_supported_entry_kwargs, split_entry_kwargs
from auto_round.compressors.utils import check_need_act_calibration
from auto_round.compressors.zero_shot import ZeroShotCompressor
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme, parse_scheme
from auto_round.utils.device_manager import normalize_default_device_map


def _collect_config_scheme_overrides(config) -> dict:
    """Return the config's explicitly-set scheme fields as a ``{field: value}`` dict.

    These are exactly the per-field overrides layered on top of ``scheme=`` — the
    single mechanism through which ``bits`` / ``act_bits`` / ``data_type`` etc.
    reach the resolved scheme. Fields left as ``None`` are omitted so the scheme's
    own value wins.
    """
    return {k: getattr(config, k) for k in config._scheme_fields if getattr(config, k, None) is not None}


def _preview_resolved_attrs(config, scheme=None) -> dict:
    """Resolve scheme attributes without mutating config, for routing decisions.

    Called in ``AutoRound.__new__`` before the concrete compressor class is
    chosen.  ``SchemeMixin.resolve_scheme()`` will do the authoritative
    resolution later; this is just a lightweight preview so routing logic
    (``enable_imatrix``, ``needs_act_calib``, etc.) can use the correct values
    even when the user specified only ``scheme=`` without explicit bit/dtype args.

    This is the single source of resolved scheme fields for entry-level routing:
    callers read from the returned dict and never re-read raw ``config`` attrs.
    When the scheme cannot be previewed (``AutoScheme``, or a deferred parse
    error), the config's own explicitly-set scheme overrides are returned so the
    values still reflect what the user passed.

    Returns:
        dict: resolved scheme attributes (config overrides when preview is skipped).
    """
    config_overrides = _collect_config_scheme_overrides(config)
    if isinstance(scheme, AutoScheme):
        # AutoScheme needs model info — cannot preview; fall back to raw config attrs.
        return config_overrides
    try:
        _, _, final_attrs = parse_scheme(scheme, config_overrides)
        return final_attrs
    except Exception:
        return config_overrides


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

    user_overrides = _collect_config_scheme_overrides(config)
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


def _build_model_free_compressor(
    model,
    scheme,
    layer_config,
    tokenizer,
    device_map,
    *,
    announced_via_flag: bool,
    **model_free_kwargs,
):
    """Construct a ``ModelFreeCompressor`` for the model-free auto-route.

    Shared by both :class:`PipelineCompressor` and the compatibility entry
    (:func:`build_compatible_compressor`) so the
    string-model guard, the auto-routing info log, and the constructor call live in
    one place. ``announced_via_flag`` is truthy when the caller explicitly passed
    ``model_free=True`` (so the informational auto-routing message is suppressed).
    """
    from auto_round.compressors.model_free import ModelFreeCompressor

    if not isinstance(model, str):
        raise ValueError("model_free=True requires `model` to be a HuggingFace ID or local path string.")
    if not announced_via_flag:
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
        **model_free_kwargs,
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

    # Single resolved-scheme source for routing (SchemeMixin does the authoritative
    # resolution later; this preview only chooses the class). Computed once: neither
    # `quant_config`'s scheme fields nor `scheme` itself change within this function,
    # so the result is invariant across every use below — no need to recompute it.
    resolved_attrs = _preview_resolved_attrs(quant_config, scheme)

    # Auto-disable rtn optimization for W8A16/W8A8-equivalent resolved schemes,
    # unless the user already set disable_opt_rtn explicitly.
    if getattr(quant_config, "orig_disable_opt_rtn", None) is None:
        bits = resolved_attrs.get("bits")
        act_bits = resolved_attrs.get("act_bits")
        data_type = resolved_attrs.get("data_type")
        if bits is not None and bits >= 8 and act_bits is not None and act_bits >= 8 and data_type == "int":
            logger.warning("`disable_opt_rtn` is turned on for W8A16/W8A8 quantization to improve efficiency.")
            disable_opt_rtn = True
            quant_config.disable_opt_rtn = True

    if not disable_opt_rtn:
        has_gguf_k = is_gguf_k_target(format) or is_gguf_k_target(scheme)
        if has_gguf_k:
            enable_imatrix = True
        else:
            sym = resolved_attrs.get("sym")
            data_type = resolved_attrs.get("data_type") or ""
            bits = resolved_attrs.get("bits")
            if sym is not None and sym is False:
                enable_imatrix = False
            elif data_type == "int" and (bits is None or bits < 8):
                enable_imatrix = True
            elif is_weight_scheme(scheme):
                enable_imatrix = True

    act_bits = resolved_attrs.get("act_bits")
    act_data_type = resolved_attrs.get("act_data_type")
    act_dynamic = resolved_attrs.get("act_dynamic")
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


class PipelineCompressor(object):
    """Algorithm-config-driven entry point (``scheme`` + ``alg_configs``).

    This is the internal pipeline entry: it resolves the algorithm config(s),
    routes to the concrete :class:`BaseCompressor` subclass (ZeroShot / DataDriven
    / ModelFree / …) wired with the right model-type Mixin, and returns that
    compressor instance. It is distinct from the public dispatcher
    :class:`auto_round.AutoRound` (in ``auto_round/autoround.py``), which forwards
    here or through :func:`build_compatible_compressor`.
    """

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
        split_kwargs = split_entry_kwargs(kwargs)
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
            return _build_model_free_compressor(
                model,
                scheme,
                layer_config,
                tokenizer,
                device_map,
                announced_via_flag=bool(route_kwargs.get("model_free", False)),
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
        # block_forward_hooks -> pre_quantize_block -> post_quantize_block)
        # actually runs; the pipeline auto-appends RTN when no block_quantizer
        # is supplied. SignRound is itself data-driven and shares the same host.
        if preprocessor_configs or isinstance(quant_config, SignRoundConfig):
            return _get_compressor_class(model_type, DataDrivenCompressor)(alg_configs, **local_args, **ctor_kwargs)
        elif isinstance(quant_config, RTNConfig):
            base_cls = _select_rtn_compressor_base_cls(quant_config, scheme, format, base_kwargs)
            return _get_compressor_class(model_type, base_cls)(alg_configs, **local_args, **ctor_kwargs)
