# Copyright (c) 2026 Intel Corporation
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

from typing import TYPE_CHECKING, Any

import torch

from auto_round.algorithms.config_resolver import _split_entry_kwargs
from auto_round.logger import logger
from auto_round.parallel import ParallelConfig
from auto_round.scheme_entry import eager_validate_scheme, is_gguf_k_target, is_weight_scheme, preview_resolved_attrs
from auto_round.schemes import QuantizationScheme
from auto_round.utils.dataset_utils import CalibDataset, normalize_dataset_spec
from auto_round.utils.device_manager import normalize_default_device_map

if TYPE_CHECKING:
    from auto_round.algorithms.quantization.config import QuantizationConfig
    from auto_round.algorithms.quantization.rtn.config import RTNConfig
    from auto_round.compressors.base import BaseCompressor


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


def _resolve_quant_config_for_routing(alg_configs) -> tuple[list, list, QuantizationConfig]:
    from auto_round.algorithms.config_resolver import split_quantization_configs
    from auto_round.algorithms.quantization.rtn.config import RTNConfig

    preprocessor_configs, block_quant_configs = split_quantization_configs(alg_configs)
    if len(block_quant_configs) == 0 and preprocessor_configs:
        return preprocessor_configs, block_quant_configs, RTNConfig()
    if block_quant_configs:
        # Stage ordering belongs to the composer. The last quantizer drives
        # public scheme/format resolution, independently of algorithm identity.
        return preprocessor_configs, block_quant_configs, block_quant_configs[-1]
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


def _resolve_rtn_config(quant_config: RTNConfig, scheme, format, base_kwargs, scheme_overrides=None) -> RTNConfig:
    """Adjust RTN calibration requirements while preserving the existing config instance."""
    from auto_round.algorithms.quantization.rrq.config import RRQConfig
    from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig, RTNConfig
    from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
    from auto_round.compressors.utils import check_need_act_calibration

    # RRQConfig inherits RTNConfig but must not be coerced to OptimizedRTNConfig
    # or downgraded to plain RTNConfig: both would silently drop the residual
    # planes.  RRQ enforces disable_opt_rtn=True in its own check_config() and
    # runs the regular calibrated path (see the model-free guard in
    # _CompressorBuilder.__new__), so there is nothing to adjust here.
    if isinstance(quant_config, RRQConfig):
        return quant_config

    enable_imatrix = False
    disable_opt_rtn = getattr(quant_config, "disable_opt_rtn", False)

    # Single resolved-scheme source for routing (SchemeMixin does the authoritative
    # resolution later; this preview only chooses the class). Computed once: neither
    # the override mapping nor `scheme` itself change within this function,
    # so the result is invariant across every use below — no need to recompute it.
    resolved_attrs = preview_resolved_attrs(scheme_overrides, scheme, format=format)

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
                # enable_neuqi: the joint (scale, zero-point) search
                # consumes the activation imatrix as a per-element weighting,
                # so it must be collected even on the asymmetric path where
                # the plain min/max initialization ignores it
                if getattr(quant_config, "enable_neuqi", False):
                    enable_imatrix = True
            elif (
                (data_type == "int" and (bits is None or bits < 8))
                or (data_type in ("mx_fp", "mx_int") and bits is not None and bits <= 8)
                or (data_type == "nv_fp" and bits == 4)
                or is_weight_scheme(scheme)
            ):
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

    return quant_config


def _disable_model_free_for_parallel(route_kwargs: dict, mode: str) -> None:
    """Require model loading for parallel execution, rejecting an explicit conflict."""
    if route_kwargs.get("model_free", False):
        raise ValueError(f"{mode} cannot be combined with model_free=True.")
    route_kwargs["disable_model_free"] = True


class _CompressorBuilder:
    """Algorithm-config-driven entry point (``scheme`` + ``alg_configs``).

    This is the internal pipeline entry: it resolves the algorithm config(s),
    routes to the concrete :class:`BaseCompressor` subclass (ZeroShot / DataDriven
    / ModelFree / …) wired with the right model-type Mixin, and returns that
    compressor instance. It is distinct from the public dispatcher
    :class:`auto_round.AutoRound` (in ``auto_round/autoround.py``).
    """

    @classmethod
    def _resolve_config(cls, config: str | object | list) -> object | list[object]:
        """Convert string alias(es) to the corresponding config instance(s) with default parameters."""
        from auto_round.algorithms.registry import resolve_alg_config

        if isinstance(config, str):
            return resolve_alg_config(config)
        if isinstance(config, list):
            return [cls._resolve_config(c) for c in config]
        return config

    def __new__(
        cls,
        model: torch.nn.Module | str,
        scheme="W4A16",
        alg_configs: str | object | list[str | object] = None,
        tokenizer=None,
        platform="hf",
        format=None,
        dataset: str | CalibDataset | list | tuple | torch.utils.data.DataLoader | None = None,
        low_gpu_mem_usage: bool = False,
        calibration_data_device: str = "auto",
        device_map: str | torch.device | int | dict = 0,
        parallel_config: ParallelConfig | dict[str, int | str] | None = None,
        iters: int | None = None,
        enable_torch_compile: bool = False,
        seed: int = 42,
        low_cpu_mem_usage: bool = True,
        layer_config=None,
        nsamples: int | None = None,
        seqlen: int | None = None,
        scheme_overrides: dict | None = None,
        **kwargs,
    ) -> BaseCompressor:
        from auto_round.algorithms.quantization.rrq.config import RRQConfig
        from auto_round.algorithms.quantization.rtn.config import RTNConfig
        from auto_round.algorithms.registry import normalize_algorithm_config
        from auto_round.compressors.orchestrator import CompressionOrchestrator as Compressor
        from auto_round.utils.model import is_model_free_route

        parallel_config = ParallelConfig.from_value(parallel_config)
        device_map = parallel_config.resolve_device_map(device_map)

        # Normalize CalibDataset objects to spec strings
        if isinstance(dataset, CalibDataset):
            dataset = dataset.to_spec_string()
        elif isinstance(dataset, (list, tuple)):
            dataset = normalize_dataset_spec(dataset)

        if alg_configs is None:
            fmt_list = format if isinstance(format, list) else [format] if format else []
            if any(f == "auto_round:rrq" for f in fmt_list):
                # Auto-select RRQ when the output format is auto_round:rrq.
                alg_configs = "rrq"
            else:
                alg_configs = "signround"
        device_map = normalize_default_device_map(device_map)
        split_kwargs = _split_entry_kwargs(kwargs, strict=bool(kwargs.get("model_free", False)))
        route_kwargs = dict(split_kwargs["route"])
        compressor_kwargs = dict(split_kwargs["compressor"])
        base_kwargs = dict(split_kwargs["base"])
        mllm_kwargs = dict(split_kwargs["mllm"])
        diffusion_kwargs = dict(split_kwargs["diffusion"])

        if parallel_config.data_parallel != "off":
            _disable_model_free_for_parallel(route_kwargs, "data_parallel")
        if parallel_config.pp_size > 1:
            _disable_model_free_for_parallel(route_kwargs, "Pipeline parallelism")

        # Resolve string alias(es) to config instance(s) before routing.
        alg_configs = cls._resolve_config(alg_configs)
        if isinstance(alg_configs, list):
            alg_configs = [normalize_algorithm_config(cfg) for cfg in alg_configs]
        else:
            alg_configs = normalize_algorithm_config(alg_configs)
        configs_for_routing = alg_configs if isinstance(alg_configs, list) else [alg_configs]
        preprocessor_configs, quantizer_configs, quant_config = _resolve_quant_config_for_routing(configs_for_routing)
        if parallel_config.tp_size > 1 or parallel_config.pp_size > 1:
            from auto_round.algorithms.config_resolver import get_algorithm_class
            from auto_round.parallel.execution import ParallelExecution

            execution = ParallelExecution(parallel_config)
            parallel_quantizer_configs = quantizer_configs or [quant_config]
            for config in [*preprocessor_configs, *parallel_quantizer_configs]:
                algorithm_class = get_algorithm_class(config)
                if algorithm_class is None:
                    raise NotImplementedError(f"{type(config).__name__} has no parallel capability contract.")
                execution.for_algorithm(algorithm_class, config)
            if (
                parallel_config.tp_size > 1
                and preprocessor_configs
                and getattr(get_algorithm_class(parallel_quantizer_configs[0]), "whole_model_tuning", False)
            ):
                raise NotImplementedError("Whole-model TP tuning cannot run block preprocessors without a block stage.")
        if parallel_config.tp_size > 1:
            from auto_round.auto_scheme import AutoScheme

            if platform != "hf" or isinstance(scheme, AutoScheme):
                raise NotImplementedError("Native TP currently requires a Hugging Face model and a fixed scheme.")
            if enable_torch_compile:
                raise NotImplementedError("Native TP QDQ does not support torch.compile yet.")
            enable_torch_compile = False
            _disable_model_free_for_parallel(route_kwargs, "Native TP")
        is_svdquant = any(type(config).__name__ == "SVDQuantConfig" for config in preprocessor_configs)
        if is_svdquant and "svdquant_omni" not in {name.strip().lower() for name in (format or "").split(",")}:
            format = "svdquant_nunchaku"

        # Validate format/algorithm compatibility for RRQ.
        is_rrq = isinstance(quant_config, RRQConfig)
        fmt_list = format if isinstance(format, list) else [format] if format else []
        has_rrq_fmt = any(f == "auto_round:rrq" for f in fmt_list)
        if is_rrq and not has_rrq_fmt:
            if format is None:
                # Auto-set the format so the user doesn't have to specify it.
                format = "auto_round:rrq"
            else:
                raise ValueError(
                    "RRQ requires --format auto_round:rrq. " "Use: auto-round --model <model> --format auto_round:rrq"
                )
        elif not is_rrq and has_rrq_fmt:
            raise ValueError(
                "--format auto_round:rrq requires the RRQ algorithm. "
                "The algorithm is auto-selected when format is auto_round:rrq. "
                "Do not pass a conflicting --alg."
            )

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
        # RRQConfig inherits RTNConfig but must use the regular calibrated
        # path: the model-free RTN path only emits a single base plane and
        # would silently drop every residual plane.  Two guard cases:
        #
        # (a) explicit ``model_free=True``: raises ValueError because
        #     ``is_model_free_route`` checks explicit before
        #     ``disable_model_free`` — there is no way to "undo" an
        #     explicit request via ``disable_model_free``.
        # (b) auto-routing (no explicit flag): set ``disable_model_free``
        #     so the regular calibrated path is taken.  Without this,
        #     ``is_model_free_route`` would auto-route when model is a
        #     string + iters==0 + disable_opt_rtn=True.
        if isinstance(quant_config, RRQConfig):
            if bool(route_kwargs.get("model_free", base_kwargs.get("model_free", False))):
                # (a) explicit model_free=True — cannot be overridden
                raise ValueError(
                    "RRQ requires the regular calibrated path (model_free=False). "
                    "The model-free RTN path emits only a single base plane and would "
                    "silently drop the residual planes. Pass a loaded model (not a model "
                    "path) or set model_free=False / omit it. "
                    "Note: disable_model_free only suppresses the automatic route and "
                    "cannot override an explicit model_free=True."
                )
            else:
                # (b) auto-route guard: force regular path.
                # ``route_kwargs`` wins over ``base_kwargs`` in the
                # ``route_decision_kwargs`` merge, so this alone is enough
                # to make ``is_model_free_route`` return False.
                route_kwargs["disable_model_free"] = True
        # Model-free eligibility also depends on base-level options such as
        # static KV/attention quantization. Keep those options visible to the
        # route predicate; otherwise the fast path silently drops them and
        # cannot emit the required export metadata.
        route_decision_kwargs = dict(base_kwargs, **route_kwargs, format=format)
        if len(quantizer_configs) > 1:
            if route_kwargs.get("model_free", False):
                raise ValueError("model_free=True is not supported with multiple quantizer stages.")
            route_decision_kwargs["disable_model_free"] = True
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
            else QuantizationScheme.from_dict(preview_resolved_attrs(scheme_overrides, scheme, format=format))
        )
        # Eagerly validate scheme constraints that do not require model info.
        # This mirrors old-arch _check_configs() called at __init__ time so that
        # callers get ValueError/NotImplementedError on construction, not deferred.
        # Runs before the model-free early return so both routes enforce the
        # same config-level constraints (e.g. the format-scoped 8-bit asym rule).
        eager_validate_scheme(scheme_overrides, scheme, format=format)
        if parallel_config.tp_size > 1:
            from auto_round.parallel.tp.policy import validate_tp_quantization

            validate_tp_quantization(route_scheme)
            for config in [*preprocessor_configs, *parallel_quantizer_configs]:
                get_algorithm_class(config).validate_parallel_scheme(route_scheme, execution)
            if base_kwargs.get("static_kv_dtype") is not None or base_kwargs.get("static_attention_dtype") is not None:
                raise NotImplementedError("Native TP does not support static KV/attention quantization yet.")
        # NOTE: the W8-asym opt-in is the AR_ALLOW_W8_ASYM environment
        # variable, read directly by parse_scheme and the generation-time
        # gates; nothing is threaded through the compressor here.
        if is_model_free_route(
            model, route_scheme, model_free_iters, model_free_disable_opt_rtn, route_decision_kwargs
        ):
            # Pass the fully resolved scheme onward so
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
                format=format,
                **compressor_kwargs,
                **base_kwargs,
                **mllm_kwargs,
                **diffusion_kwargs,
                **route_kwargs,
            )

        local_args = dict(
            model=model,
            tokenizer=tokenizer,
            platform=platform,
            format=format,
            scheme=scheme,
            scheme_overrides=scheme_overrides,
            dataset=dataset,
            low_gpu_mem_usage=low_gpu_mem_usage,
            calibration_data_device=calibration_data_device,
            device_map=device_map,
            parallel_config=parallel_config,
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
        # actually runs; the pipeline auto-appends RTN when no quantizer
        # is supplied. SignRound is itself data-driven and shares the same host.
        if not preprocessor_configs and len(quantizer_configs) <= 1 and isinstance(quant_config, RTNConfig):
            _resolve_rtn_config(quant_config, scheme, format, base_kwargs, scheme_overrides)
        return _get_compressor_class(model_type, Compressor)(alg_configs, **local_args, **ctor_kwargs)
