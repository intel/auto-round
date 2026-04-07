# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Optional, Union

import torch

from auto_round.algorithms.alg_config import AlgConfig
from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.algorithms.transforms.hadamard.config import HadamardConfig
from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
from auto_round.compressors_new.calib import CalibCompressor, CalibratedRTNCompressor
from auto_round.compressors_new.utils import check_need_act_calibration
from auto_round.compressors_new.zero_shot import ZeroShotCompressor
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme, _parse_scheme


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
    scheme_attr_names = QuantizationScheme.get_attributes()
    user_overrides = {k: getattr(config, k) for k in scheme_attr_names if getattr(config, k, None) is not None}
    try:
        _, _, final_attrs = _parse_scheme(scheme, user_overrides)
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

    scheme_attr_names = QuantizationScheme.get_attributes()
    user_overrides = {k: getattr(config, k) for k in scheme_attr_names if getattr(config, k, None) is not None}
    try:
        _, _, final_attrs = _parse_scheme(scheme, user_overrides)
    except (ValueError, NotImplementedError):
        raise
    except Exception:
        return  # Other parse errors are deferred to post_init

    import copy

    temp_config = copy.copy(config)
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
        from auto_round.compressors_new.mllm_mixin import MLLMMixin

        mixin = MLLMMixin
    elif model_type == "diffusion":
        from auto_round.compressors_new.diffusion_mixin import DiffusionMixin

        mixin = DiffusionMixin
    else:
        return base_cls
    combined = type(f"{model_type.capitalize()}{base_cls.__name__}", (mixin, base_cls), {})
    _COMPRESSOR_REGISTRY[key] = combined
    return combined


def is_weight_scheme(scheme):
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


def detect_model_type(model):
    """Detect the type of model (LLM, MLLM, or Diffusion).

    Args:
        model: Model instance or model path string

    Returns:
        str: "mllm", "diffusion", or "llm"
    """
    from auto_round.utils import is_diffusion_model, is_mllm_model

    # Check if it's a diffusion model first (more specific)
    if is_diffusion_model(model):
        return "diffusion"

    # Check if it's an MLLM
    if is_mllm_model(model):
        return "mllm"

    # Default to standard LLM
    return "llm"


class AutoRound(object):
    SKIP_ARGS = ("local_args", "kwargs", "cls", "alg_configs", "quant_config", "quant_configs")

    # Mapping from string alias to config class (and optional defaults override).
    _CONFIG_ALIASES: dict[str, type] = {
        "sign_round": SignRoundConfig,
        "signround": SignRoundConfig,
        "rtn": RTNConfig,
        "hadamard": HadamardConfig,
    }

    @classmethod
    def _resolve_config(cls, config: Union[str, AlgConfig, list]) -> Union[AlgConfig, list[AlgConfig]]:
        """Convert string alias(es) to the corresponding config instance(s) with default parameters."""
        if isinstance(config, str):
            key = config.strip().lower()
            if key not in cls._CONFIG_ALIASES:
                raise ValueError(f"Unknown config alias '{config}'. " f"Supported: {list(cls._CONFIG_ALIASES.keys())}")
            return cls._CONFIG_ALIASES[key]()
        if isinstance(config, list):
            return [cls._resolve_config(c) for c in config]
        return config

    def __new__(
        cls,
        alg_configs: Union[str, AlgConfig, list[Union[str, AlgConfig]]],
        model: Union[torch.nn.Module, str],
        tokenizer=None,
        platform="hf",
        format=None,
        scheme="W4A16",
        low_gpu_mem_usage: bool = False,
        device_map: Union[str, torch.device, int, dict] = 0,
        enable_torch_compile: bool = False,
        seed: int = 42,
        low_cpu_mem_usage: bool = True,
        layer_config=None,
        nsamples: int = None,
        seqlen: int = None,
        **kwargs,
    ):
        from auto_round.algorithms.quantization.config import QuantizationConfig

        # Resolve string alias(es) to config instance(s) before routing.
        alg_configs = cls._resolve_config(alg_configs)

        # Extract the single QuantizationConfig from a list; validate at most one exists.
        if isinstance(alg_configs, list):
            quant_configs = [c for c in alg_configs if isinstance(c, QuantizationConfig)]
            if len(quant_configs) == 0:
                raise ValueError("At least one QuantizationConfig (SignRoundConfig / RTNConfig) is required.")
            if len(quant_configs) > 1:
                raise ValueError(
                    f"Only one QuantizationConfig is allowed, but got {len(quant_configs)}: "
                    f"{[type(c).__name__ for c in quant_configs]}"
                )
            quant_config = quant_configs[0]
        else:
            quant_config = alg_configs

        # Eagerly validate scheme constraints that do not require model info.
        # This mirrors old-arch _check_configs() called at __init__ time so that
        # callers get ValueError/NotImplementedError on construction, not deferred.
        _eager_validate_scheme(quant_config, scheme)

        # using different compressor base on AlgConfigs
        local_args = {k: v for k, v in locals().items() if k not in cls.SKIP_ARGS}

        # Detect model type to determine if we need special compressor
        model_type = detect_model_type(model)

        # If the user explicitly passes processor/image_processor, treat as MLLM even if
        # auto-detection missed it (mirrors the has_multimodal_assets check in autoround.py).
        has_multimodal_assets = kwargs.get("processor") is not None or kwargs.get("image_processor") is not None
        if has_multimodal_assets and model_type != "mllm":
            model_type = "mllm"

        if isinstance(quant_config, SignRoundConfig):
            return _get_compressor_class(model_type, CalibCompressor)(alg_configs, **local_args, **kwargs)

        elif isinstance(quant_config, RTNConfig):
            enable_imatrix = False
            disable_opt_rtn = getattr(quant_config, "disable_opt_rtn", False)
            # If disable_opt_rtn was not explicitly set and scheme is W8A16/W8A8,
            # auto-disable optimization to improve efficiency.
            if getattr(quant_config, "orig_disable_opt_rtn", None) is None:
                if isinstance(scheme, str) and scheme.upper() in ["W8A16", "W8A8"]:
                    logger.warning("`disable_opt_rtn` is turned on for W8A16/W8A8 quantization to improve efficiency.")
                    disable_opt_rtn = True
                    quant_config.disable_opt_rtn = True
            if not disable_opt_rtn:
                has_gguf_k = "gguf" in format.lower() and "_k" in format.lower() if format else False
                if has_gguf_k:
                    enable_imatrix = True
                else:
                    # Resolve scheme attrs for routing (config hasn't been through
                    # SchemeMixin yet; user may have specified only scheme="W4A16").
                    _resolved = _preview_resolved_attrs(quant_config, scheme)
                    _sym = _resolved.get("sym", getattr(quant_config, "sym", None))
                    _data_type = _resolved.get("data_type", getattr(quant_config, "data_type", "") or "")
                    _bits = _resolved.get("bits", getattr(quant_config, "bits", None))
                    if _sym is not None and _sym is False:
                        enable_imatrix = False
                    elif _data_type == "int" and (_bits is None or _bits < 8):
                        enable_imatrix = True
                    elif is_weight_scheme(scheme):
                        enable_imatrix = True
            else:
                _resolved = {}

            _resolved = _resolved if not disable_opt_rtn else _preview_resolved_attrs(quant_config, scheme)
            _act_bits = _resolved.get("act_bits", getattr(quant_config, "act_bits", None))
            _act_data_type = _resolved.get("act_data_type", getattr(quant_config, "act_data_type", None))
            _act_dynamic = _resolved.get("act_dynamic", getattr(quant_config, "act_dynamic", None))
            _is_act_quantize = _act_bits is not None and _act_bits <= 8
            needs_act_calib = _is_act_quantize and check_need_act_calibration(
                _act_dynamic,
                _act_data_type,
                _act_bits if _act_bits is not None else 16,
                static_kv_dtype=kwargs.get("static_kv_dtype"),
                static_attention_dtype=kwargs.get("static_attention_dtype"),
            )

            # AutoScheme always requires calibration data for delta-loss based
            # scheme selection, regardless of whether imatrix is needed.
            from auto_round.auto_scheme.gen_auto_scheme import AutoScheme as _AutoScheme

            is_auto_scheme = isinstance(scheme, _AutoScheme)

            if enable_imatrix or needs_act_calib or is_auto_scheme:
                quant_config._alg_cls = "OptimizedRTNQuantizer"
                return _get_compressor_class(model_type, CalibratedRTNCompressor)(alg_configs, **local_args, **kwargs)
            else:
                quant_config._alg_cls = "RTNQuantizer"
                return _get_compressor_class(model_type, ZeroShotCompressor)(alg_configs, **local_args, **kwargs)


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
        >>> from auto_round.compressors_new.entry import AutoRoundCompatible
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
        common_keys = (
            "ignore_layers",
            "quant_lm_head",
            "scale_dtype",
            "super_bits",
            "super_group_size",
            "to_quant_block_names",
        )
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
        **kwargs,
    ):
        """Create AutoRoundCompatible instance using new AutoRound architecture.

        This method translates old AutoRoundCompatible API to new AutoRound API.
        """
        from auto_round.utils import is_diffusion_model, is_mllm_model

        common_config_kwargs, auto_round_config_kwargs = cls._pop_config_kwargs(kwargs)

        # Extract quantization parameters from kwargs or use defaults
        bits = kwargs.pop("bits", None)
        group_size = kwargs.pop("group_size", None)
        sym = kwargs.pop("sym", None)
        data_type = kwargs.pop("data_type", None)
        act_bits = kwargs.pop("act_bits", None)
        act_group_size = kwargs.pop("act_group_size", None)
        act_sym = kwargs.pop("act_sym", None)
        act_data_type = kwargs.pop("act_data_type", None)
        act_dynamic = kwargs.pop("act_dynamic", None)

        # Decide which algorithm to use
        if iters == 0:
            # RTN mode
            disable_opt_rtn = kwargs.pop("disable_opt_rtn", None)
            config = RTNConfig(
                bits=bits,
                group_size=group_size,
                sym=sym,
                data_type=data_type,
                act_bits=act_bits,
                act_group_size=act_group_size,
                act_sym=act_sym,
                act_data_type=act_data_type,
                act_dynamic=act_dynamic,
                disable_opt_rtn=disable_opt_rtn,
                # for optRTN
                batch_size=batch_size,
                **common_config_kwargs,
            )
        else:
            # AutoRoundCompatible mode
            lr = kwargs.pop("lr", None)
            minmax_lr = kwargs.pop("minmax_lr", None)
            enable_minmax_tuning = kwargs.pop("enable_minmax_tuning", True)
            enable_norm_bias_tuning = kwargs.pop("enable_norm_bias_tuning", False)
            enable_quanted_input = kwargs.pop("enable_quanted_input", True)

            config = SignRoundConfig(
                iters=iters,
                batch_size=batch_size,
                gradient_accumulate_steps=gradient_accumulate_steps,
                bits=bits,
                group_size=group_size,
                sym=sym,
                data_type=data_type,
                act_bits=act_bits,
                act_group_size=act_group_size,
                act_sym=act_sym,
                act_data_type=act_data_type,
                act_dynamic=act_dynamic,
                lr=lr,
                minmax_lr=minmax_lr,
                enable_minmax_tuning=enable_minmax_tuning,
                enable_norm_bias_tuning=enable_norm_bias_tuning,
                enable_quanted_input=enable_quanted_input,
                **common_config_kwargs,
                **auto_round_config_kwargs,
            )

        # Determine output format if specified
        format = kwargs.pop("format", None)

        # Extract MLLM-specific parameters
        processor = kwargs.pop("processor", None)
        image_processor = kwargs.pop("image_processor", None)
        template = kwargs.pop("template", None)
        extra_data_dir = kwargs.pop("extra_data_dir", None)
        quant_nontext_module = kwargs.pop("quant_nontext_module", False)

        # Extract Diffusion-specific parameters
        guidance_scale = kwargs.pop("guidance_scale", 7.5)
        num_inference_steps = kwargs.pop("num_inference_steps", 50)
        generator_seed = kwargs.pop("generator_seed", None)

        # Check model type for logging
        if is_mllm_model(model, platform=platform):
            logger.info("Using MLLM mode for multimodal model (new architecture).")
        elif is_diffusion_model(model):
            logger.info("Using Diffusion mode for diffusion model (new architecture).")
        else:
            logger.info("Using LLM mode (new architecture).")

        # Create AutoRound instance using new architecture
        compressor = AutoRound(
            alg_configs=config,
            model=model,
            tokenizer=tokenizer,
            platform=platform,
            format=format,
            scheme=scheme,
            dataset=dataset,
            low_gpu_mem_usage=low_gpu_mem_usage,
            device_map=device_map,
            enable_torch_compile=enable_torch_compile,
            seed=seed,
            low_cpu_mem_usage=low_cpu_mem_usage,
            layer_config=layer_config,
            nsamples=nsamples,
            seqlen=seqlen,
            # MLLM parameters
            processor=processor,
            image_processor=image_processor,
            template=template,
            extra_data_dir=extra_data_dir,
            quant_nontext_module=quant_nontext_module,
            # Diffusion parameters
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator_seed=generator_seed,
            # Pass remaining kwargs
            **kwargs,
        )

        return compressor
