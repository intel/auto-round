# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

import torch

from auto_round.logger import deprecated, logger
from auto_round.schemes import QuantizationScheme, parse_scheme
from auto_round.utils.device_manager import normalize_default_device_map

if TYPE_CHECKING:
    from auto_round.algorithms.quantization.config import QuantizationConfig
    from auto_round.algorithms.quantization.rtn.config import RTNConfig
    from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
    from auto_round.compressors.base import BaseOrchestrator as BaseCompressor


def _collect_config_scheme_overrides(config) -> dict:
    """Return the config's explicitly-set scheme fields as a ``{field: value}`` dict.

    These are exactly the per-field overrides layered on top of ``scheme=`` — the
    single mechanism through which ``bits`` / ``act_bits`` / ``data_type`` etc.
    reach the resolved scheme. Fields left as ``None`` are omitted so the scheme's
    own value wins.
    """
    return {k: getattr(config, k) for k in config._scheme_fields if getattr(config, k, None) is not None}


def _preview_resolved_attrs(config, scheme=None) -> dict:
    from auto_round.auto_scheme.gen_auto_scheme import AutoScheme

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
    from auto_round.auto_scheme.gen_auto_scheme import AutoScheme

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


def is_weight_scheme(scheme: Union[str, dict, object]) -> bool:
    if isinstance(scheme, str):
        return scheme.upper().startswith("W")
    if isinstance(scheme, dict):
        return all(isinstance(s, str) and s.upper().startswith("W") for s in scheme.values())
    from auto_round.auto_scheme.gen_auto_scheme import AutoScheme

    if isinstance(scheme, AutoScheme):
        opts = scheme.options
        if isinstance(opts, (list, tuple)):
            return all(isinstance(s, str) and s.upper().startswith("W") for s in opts)
        if isinstance(opts, str):
            return opts.upper().startswith("W")
    return False


def is_gguf_k_target(value: Union[str, "AutoScheme", object]) -> bool:
    from auto_round.auto_scheme.gen_auto_scheme import AutoScheme

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


def _resolve_quant_config_for_routing(alg_configs) -> tuple[list, list, "QuantizationConfig"]:
    from auto_round.algorithms.config_resolver import split_quantization_configs
    from auto_round.algorithms.quantization.config import QuantizationConfig
    from auto_round.algorithms.quantization.rtn.config import RTNConfig

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

    Shared by the unified entry and the model-free route so the string-model
    guard, the auto-routing info log, and the constructor call live in one place.
    ``announced_via_flag`` is truthy when the caller explicitly passed
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


def _select_rtn_compressor_base_cls(quant_config: "RTNConfig", scheme, format, base_kwargs) -> type:
    from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig, RTNConfig
    from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
    from auto_round.compressors.orchestrator import CompressionOrchestrator as Compressor
    from auto_round.compressors.utils import check_need_act_calibration

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
    needs_optimized_rtn = enable_imatrix or needs_act_calib or isinstance(scheme, AutoScheme)
    if needs_optimized_rtn:
        if not isinstance(quant_config, OptimizedRTNConfig):
            quant_config.__class__ = OptimizedRTNConfig
    else:
        # Pure zero-shot RTN: downgrade to basic RTNConfig
        if isinstance(quant_config, OptimizedRTNConfig):
            quant_config.__class__ = RTNConfig

    # Always use Compressor — it internally detects whether calibration
    # data is needed and falls back to the zero-shot (RTN) path when it is not.
    return Compressor


_ENTRY_KWARG_OWNERS = {
    "model_free": "route",
    "disable_model_free": "route",
    "scale_dtype": "compressor",
    "ignore_layers": "compressor",
    "quant_lm_head": "compressor",
    "to_quant_block_names": "compressor",
    "format": "base",
    "dataset": "base",
    "batch_size": "base",
    "model_dtype": "base",
    "trust_remote_code": "base",
    "amp": "base",
    "disable_deterministic_algorithms": "base",
    "enable_deterministic_algorithms": "base",
    "static_kv_dtype": "base",
    "static_attention_dtype": "base",
    "static_kv_granularity": "base",
    "static_attention_granularity": "base",
    "processor": "mllm",
    "image_processor": "mllm",
    "template": "mllm",
    "extra_data_dir": "mllm",
    "quant_nontext_module": "mllm",
    "guidance_scale": "diffusion",
    "num_inference_steps": "diffusion",
    "generator_seed": "diffusion",
}

_SCHEME_FIELDS = set(QuantizationScheme.get_attributes())
_SIGNROUND_FIELDS = {
    "iters",
    "lr",
    "minmax_lr",
    "lr_scheduler",
    "momentum",
    "nblocks",
    "enable_minmax_tuning",
    "enable_norm_bias_tuning",
    "gradient_accumulate_steps",
    "enable_alg_ext",
    "not_use_best_mse",
    "dynamic_max_gap",
    "enable_quanted_input",
    "optimizer",
    "enable_adam",
    "enable_lfq",
}
_RTN_FIELDS = {"disable_opt_rtn", "enable_opt_rtn"}
_AWQ_FIELDS = {
    "duo_scaling",
    "n_grid",
    "seqlen",
    "nsamples",
    "batch_size",
    "apply_smooth",
    "smooth_iters",
    "apply_clip",
    "clip_as_init",
    "clip_n_grid",
    "clip_max_shrink",
    "clip_n_sample_token",
    "awq_seqlen",
    "smooth_batch_size",
    "disable_opt_rtn",
    "enable_opt_rtn",
    "skip_moe",
    "mappings",
}
_ROTATION_FIELDS = {
    "hadamard_type",
    "block_size",
    "fuse_online_to_weight",
    "allow_online_rotation",
}


def _filter_supported_entry_kwargs(kwargs, *, context="AutoRound"):
    supported = {key: value for key, value in kwargs.items() if key in _ENTRY_KWARG_OWNERS}
    unknown = sorted(set(kwargs) - set(_ENTRY_KWARG_OWNERS))
    if unknown:
        logger.warning_once(
            "%s received unsupported runtime kwargs %s. They will be ignored.",
            context,
            ", ".join(unknown),
        )
    return supported


def _split_entry_kwargs(kwargs, *, context="AutoRound"):
    buckets = {"route": {}, "compressor": {}, "base": {}, "mllm": {}, "diffusion": {}}
    for key, value in _filter_supported_entry_kwargs(kwargs, context=context).items():
        buckets[_ENTRY_KWARG_OWNERS[key]][key] = value
    return buckets


def _config_fields(config):
    fields = set(_SCHEME_FIELDS)
    name = type(config).__name__.lower()
    if "awq" in name:
        fields.update(_AWQ_FIELDS)
    elif "rtn" in name:
        fields.update(_RTN_FIELDS)
    elif "signround" in name or "adamround" in name:
        fields.update(_SIGNROUND_FIELDS)
    elif "rotation" in name:
        fields.update(_ROTATION_FIELDS)
    return fields


def _normalize_alg_configs(alg_configs, direct_kwargs=None):
    from auto_round.algorithms.config_resolver import split_quantization_configs
    from auto_round.algorithms.quantization.config import QuantizationConfig
    from auto_round.algorithms.quantization.rtn.config import RTNConfig
    from auto_round.algorithms.registry import normalize_algorithm_config, resolve_alg_config, resolve_algorithm_names
    from auto_round.algorithms.transforms import normalize_rotation_config
    from auto_round.algorithms.transforms.awq.config import (
        awq_disable_opt_rtn,
        rtn_inherited_opt_kwargs,
        sync_rtn_opt_rtn_from_awq,
    )
    from auto_round.algorithms.transforms.base import BaseRotationConfig

    direct_kwargs = dict(direct_kwargs or {})
    legacy_algorithm = direct_kwargs.pop("algorithm", None)
    if legacy_algorithm is not None:
        if alg_configs is not None:
            raise ValueError("`algorithm` and `alg_configs` cannot be used together.")
        alg_configs = legacy_algorithm
        logger.warning_once("`algorithm` is deprecated; use `alg_configs` instead.")
    if "backend" in direct_kwargs:
        raise ValueError(
            "Rotation backend selection must be nested in `rotation_config`; "
            "do not pass it as AutoRound(..., backend=...)."
        )
    rotation_config = direct_kwargs.pop("rotation_config", None)
    config_kwargs = {key: value for key, value in direct_kwargs.items() if key not in _ENTRY_KWARG_OWNERS}
    if alg_configs is None:
        # Preserve the legacy entry semantics: zero iterations are RTN, while
        # positive iterations use SignRound.  RTN-only kwargs also select RTN
        # so they are not silently ignored by the default SignRound config.
        if direct_kwargs.get("iters") == 0:
            raw_configs = ["rtn"]
        else:
            raw_configs = ["signround"]
    elif isinstance(alg_configs, str):
        raw_configs = resolve_algorithm_names(alg_configs)
        if not raw_configs:
            raise ValueError("`algorithm`/`alg_configs` must contain at least one algorithm name.")
    elif isinstance(alg_configs, (list, tuple)):
        raw_configs = []
        seen_names = set()
        for raw_config in alg_configs:
            if not isinstance(raw_config, str):
                raw_configs.append(raw_config)
                continue
            for canonical_name in resolve_algorithm_names(raw_config):
                if canonical_name not in seen_names:
                    raw_configs.append(canonical_name)
                    seen_names.add(canonical_name)
    else:
        raw_configs = [alg_configs]

    configs = []
    pending_rtn_indices = []
    for raw_config in raw_configs:
        if isinstance(raw_config, str) and raw_config == "rtn":
            pending_rtn_indices.append(len(configs))
            configs.append(None)
            continue
        else:
            config = resolve_alg_config(raw_config) if isinstance(raw_config, str) else raw_config
        if not isinstance(config, (QuantizationConfig, BaseRotationConfig)):
            raise TypeError(
                f"alg_configs entries must be algorithm or QuantizationConfig instances, "
                f"got {type(config).__name__}."
            )
        configs.append(normalize_algorithm_config(config))

    awq_opt_rtn_policy = awq_disable_opt_rtn(configs)
    for index in pending_rtn_indices:
        config = RTNConfig(**rtn_inherited_opt_kwargs(config_kwargs, awq_opt_rtn_policy))
        configs[index] = normalize_algorithm_config(config)

    # ``iters=0`` has always selected RTN in the public entry and CLI. Apply
    # that rule after every input form has become a config so aliases, config
    # objects, and the deprecated ``algorithm`` argument cannot choose
    # different quantizer implementations.
    from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig

    direct_iters = config_kwargs.get("iters")
    for index, config in enumerate(configs):
        effective_iters = direct_iters if direct_iters is not None else getattr(config, "iters", None)
        if isinstance(config, SignRoundConfig) and effective_iters == 0:
            rtn_kwargs = rtn_inherited_opt_kwargs(config_kwargs, awq_disable_opt_rtn(configs))
            rtn_config = RTNConfig(scheme=config.scheme.copy(), **rtn_kwargs)
            rtn_config._user_set_scheme_fields = set(getattr(config, "_user_set_scheme_fields", set()))
            configs[index] = normalize_algorithm_config(rtn_config)

    if rotation_config is not None:
        normalized_rotation = normalize_rotation_config(rotation_config)
        if normalized_rotation is not None:
            configs.append(normalized_rotation)

    if not any(isinstance(config, QuantizationConfig) for config in configs):
        raise TypeError(
            "alg_configs entries must be algorithm aliases or QuantizationConfig instances, "
            "and must include at least one quantization algorithm config."
        )

    preprocessors, block_configs = split_quantization_configs(configs)
    if preprocessors and not block_configs:
        fallback_rtn_kwargs = rtn_inherited_opt_kwargs(config_kwargs, awq_disable_opt_rtn(configs))
        configs.append(normalize_algorithm_config(RTNConfig(**fallback_rtn_kwargs)))

    _, block_configs = split_quantization_configs(configs)
    for key, value in config_kwargs.items():
        if value is None:
            continue
        if key in ("disable_opt_rtn", "enable_opt_rtn"):
            if key == "enable_opt_rtn" and not value:
                continue
            opt_rtn_value = False if key == "enable_opt_rtn" else value
            targets = [config for config in configs if "disable_opt_rtn" in _config_fields(config)]
            if not targets:
                logger.warning_once(
                    "RTN-specific parameter '%s' was provided, but RTN/AWQ is not enabled by alg_configs. "
                    "The parameter is ignored.",
                    key,
                )
                continue
            for target in targets:
                target.disable_opt_rtn = opt_rtn_value
                if hasattr(target, "orig_disable_opt_rtn"):
                    target.orig_disable_opt_rtn = opt_rtn_value
            logger.warning(
                "Passing '%s' directly to AutoRound is supported, but the recommended usage is "
                "'alg_configs=\"awq\"' or 'alg_configs=\"rtn\"'.",
                key,
            )
            continue
        if key in _SCHEME_FIELDS:
            targets = block_configs
        else:
            targets = [config for config in configs if key in _config_fields(config)]
        if not targets:
            # ``iters`` is a legacy route selector.  RTN intentionally has no
            # iterative parameter, so ``iters=0`` must not be reported as an
            # ignored algorithm-specific error after selecting RTN.
            if key == "iters" and any(isinstance(config, RTNConfig) for config in configs):
                continue
            owner = "AWQ" if key in _AWQ_FIELDS else "the selected algorithm"
            logger.error(
                "%s-specific parameter '%s' was provided, but %s is not enabled by alg_configs. "
                "The parameter is ignored.",
                owner,
                key,
                owner,
            )
            continue
        if len(targets) > 1:
            logger.error(
                "Parameter '%s' matches multiple algorithm configs. Pass it through the matching "
                "config object in 'alg_configs'; the direct value is ignored.",
                key,
            )
            continue
        target = targets[0]
        setattr(target, key, value)
        recommended_config_name = type(target).__name__.replace("Config", "")
        logger.warning(
            "Passing '%s' directly to AutoRound is supported, but the recommended usage is "
            "'alg_configs=%sConfig(...)'.",
            key,
            recommended_config_name,
        )
    sync_rtn_opt_rtn_from_awq(configs)
    return [normalize_algorithm_config(config) for config in configs]


def _prepare_entry_kwargs(alg_configs, direct_kwargs):
    configs = _normalize_alg_configs(alg_configs, direct_kwargs)
    runtime_kwargs = {key: value for key, value in direct_kwargs.items() if key in _ENTRY_KWARG_OWNERS}
    return configs, runtime_kwargs


class _CompressorBuilder(object):
    """Algorithm-config-driven entry point (``scheme`` + ``alg_configs``).

    This is the internal pipeline entry: it resolves the algorithm config(s),
    routes to the concrete :class:`BaseCompressor` subclass (ZeroShot / DataDriven
    / ModelFree / …) wired with the right model-type Mixin, and returns that
    compressor instance. It is distinct from the public dispatcher
    :class:`auto_round.AutoRound` (in ``auto_round/autoround.py``).
    """

    @classmethod
    def _resolve_config(cls, config: Union[str, object, list]) -> Union[object, list[object]]:
        """Convert string alias(es) to the corresponding config instance(s) with default parameters."""
        from auto_round.algorithms.registry import resolve_alg_config

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
        dataset="NeelNanda/pile-10k",
        low_gpu_mem_usage: bool = False,
        device_map: Union[str, torch.device, int, dict] = 0,
        iters: int = None,
        enable_torch_compile: bool = False,
        seed: int = 42,
        low_cpu_mem_usage: bool = True,
        layer_config=None,
        nsamples: int = None,
        seqlen: int = None,
        **kwargs,
    ) -> "BaseCompressor":
        from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig, RTNConfig
        from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
        from auto_round.algorithms.registry import normalize_algorithm_config
        from auto_round.compressors.orchestrator import CompressionOrchestrator as Compressor
        from auto_round.compressors.utils import check_need_act_calibration
        from auto_round.utils.model import is_model_free_route

        if alg_configs is None:
            alg_configs = "signround"
        # TODO  wenhuach if key in kwargs could override scheme and alg_config, we should pop and override,
        #  e.g. gradient_accumulate_step
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
        is_svdquant = any(type(config).__name__ == "SVDQuantConfig" for config in preprocessor_configs)
        if is_svdquant:
            format = "svdquant_nunchaku"

        # Any preprocessor that requires calibration data (e.g. AWQ, SVDQuant
        # smoothing) must run on the regular model-loaded path; model-free RTN
        # cannot replay their calibration.
        calibration_preprocessors = [
            type(config).__name__ for config in preprocessor_configs if getattr(config, "need_calib", False)
        ]
        has_calibration_preprocessor = bool(calibration_preprocessors)
        if has_calibration_preprocessor and route_kwargs.get("model_free", False):
            raise ValueError(
                "model_free=True is not supported with calibration-based preprocessor algorithms "
                f"({', '.join(calibration_preprocessors)}). "
                "Use the regular flow so the model can be loaded for calibration."
            )

        # Model-free routing is now supported directly by the new entry path.
        model_free_iters = 0 if isinstance(quant_config, RTNConfig) else getattr(quant_config, "iters", None)
        model_free_disable_opt_rtn = getattr(quant_config, "disable_opt_rtn", None)
        # Model-free eligibility also depends on base-level options such as
        # static KV/attention quantization. Keep those options visible to the
        # route predicate; otherwise the fast path silently drops them and
        # cannot emit the required export metadata.
        route_decision_kwargs = dict(base_kwargs, **route_kwargs, format=format)
        is_svdquant_rtn = type(quant_config) is RTNConfig and is_svdquant
        if is_svdquant_rtn and not route_kwargs.get("model_free", False):
            # SVDQuant must run before plain RTN on the regular blockwise path.
            route_decision_kwargs["disable_model_free"] = True
        if has_calibration_preprocessor:
            # Calibration-based preprocessors need the model loaded for
            # calibration; model-free RTN cannot apply their transforms.
            route_decision_kwargs["disable_model_free"] = True
        route_scheme = (
            scheme
            if hasattr(scheme, "options") and hasattr(scheme, "avg_bits")
            else QuantizationScheme.from_dict(_preview_resolved_attrs(quant_config, scheme))
        )
        if is_model_free_route(
            model, route_scheme, model_free_iters, model_free_disable_opt_rtn, route_decision_kwargs
        ):
            # Direct scheme fields are consumed into ``quant_config`` during
            # entry normalization. Pass the fully resolved scheme onward so
            # model-free export does not silently fall back to preset defaults.
            model_free_scheme = scheme if hasattr(scheme, "options") and hasattr(scheme, "avg_bits") else route_scheme
            return _build_model_free_compressor(
                model,
                model_free_scheme,
                layer_config,
                tokenizer,
                device_map,
                announced_via_flag=bool(route_kwargs.get("model_free", False)),
                dataset=dataset,
                nsamples=nsamples,
                seqlen=seqlen,
                seed=seed,
                enable_torch_compile=enable_torch_compile,
                disable_opt_rtn=model_free_disable_opt_rtn,
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
            dataset=dataset,
            low_gpu_mem_usage=low_gpu_mem_usage,
            device_map=device_map,
            iters=iters,
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
            return _get_compressor_class(model_type, Compressor)(alg_configs, **local_args, **ctor_kwargs)
        elif isinstance(quant_config, RTNConfig):
            base_cls = _select_rtn_compressor_base_cls(quant_config, scheme, format, base_kwargs)
            return _get_compressor_class(model_type, base_cls)(alg_configs, **local_args, **ctor_kwargs)


class AutoRound:
    """Unified AutoRound entry point.

    alg_configs accepts an algorithm alias, one QuantizationConfig, or a
    sequence of either. When omitted, SignRound is selected. AWQ-only
    pipelines receive an RTN block quantizer by default.
    """

    SKIP_ARGS = ("local_args", "kwargs", "cls", "model_cls", "dynamic_compressor", "alg_configs")

    def __new__(
        cls,
        model: Union[torch.nn.Module, str],
        tokenizer=None,
        platform: str = "hf",
        scheme: Union[str, dict, QuantizationScheme, "AutoScheme"] = "W4A16",
        layer_config: dict[str, Union[str, dict, QuantizationScheme]] = None,
        dataset: Optional[Union[str, list, tuple, torch.utils.data.DataLoader]] = None,
        iters: int | None = None,
        seqlen: int = 2048,
        nsamples: int = 128,
        batch_size: int = 8,
        gradient_accumulate_steps: int | None = None,
        low_gpu_mem_usage: bool = False,
        device_map: Union[str, torch.device, int, dict] = 0,
        enable_torch_compile: Optional[bool] = None,
        seed: int = 42,
        low_cpu_mem_usage: bool = True,
        alg_configs=None,
        algorithm: str | None = None,
        **kwargs,
    ) -> "BaseCompressor":
        direct_kwargs = dict(kwargs)
        legacy_device = direct_kwargs.pop("device", None)
        if legacy_device is not None:
            logger.warning_once("`device` is deprecated, please use `device_map` instead")
            if device_map in (None, 0):
                device_map = legacy_device
        if iters is not None:
            direct_kwargs["iters"] = iters
        if gradient_accumulate_steps is not None:
            direct_kwargs["gradient_accumulate_steps"] = gradient_accumulate_steps
        if algorithm is not None:
            direct_kwargs["algorithm"] = algorithm

        configs, runtime_kwargs = _prepare_entry_kwargs(alg_configs, direct_kwargs)
        runtime_kwargs["batch_size"] = batch_size

        return _CompressorBuilder(
            model,
            scheme,
            configs,
            tokenizer=tokenizer,
            platform=platform,
            format=runtime_kwargs.pop("format", None),
            dataset=dataset,
            low_gpu_mem_usage=low_gpu_mem_usage,
            device_map=normalize_default_device_map(device_map),
            iters=None,
            enable_torch_compile=enable_torch_compile,
            seed=seed,
            low_cpu_mem_usage=low_cpu_mem_usage,
            layer_config=layer_config,
            nsamples=nsamples,
            seqlen=seqlen,
            **runtime_kwargs,
        )


# Keep legacy entry points available for downstream integrations such as
# Neural Compressor. They delegate to the unified entry point.
@deprecated("AutoRound")
class AutoRoundLLM:
    def __new__(cls, *args, **kwargs):
        return AutoRound(*args, **kwargs)


@deprecated("AutoRound")
class AutoRoundAdam:
    def __new__(cls, *args, **kwargs):
        kwargs.setdefault("enable_adam", True)
        return AutoRound(*args, **kwargs)


@deprecated("AutoRound")
class AutoRoundMLLM:
    def __new__(cls, *args, **kwargs):
        return AutoRound(*args, **kwargs)


@deprecated("AutoRound")
class AutoRoundDiffusion:
    def __new__(cls, *args, **kwargs):
        return AutoRound(*args, **kwargs)
