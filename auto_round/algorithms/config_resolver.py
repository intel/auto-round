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

import functools
import inspect
from collections import defaultdict
from typing import Any

from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme


def get_algorithm_class(config: Any):
    """Return the registered implementation class for a quantization config."""
    from auto_round.algorithms.registry import normalize_algorithm_config, resolve_pipeline_member

    try:
        return resolve_pipeline_member(normalize_algorithm_config(config))
    except ValueError:
        return None


def is_preprocessor_config(config: Any) -> bool:
    """Whether *config* resolves to a BasePreprocessor implementation."""
    from auto_round.algorithms.transforms.base import BasePreprocessor

    alg_cls = get_algorithm_class(config)
    return alg_cls is not None and issubclass(alg_cls, BasePreprocessor)


def is_quantizer_config(config: Any) -> bool:
    """Check whether a configuration resolves to a BaseQuantizer implementation.

    Args:
        config (Any): Algorithm configuration to resolve through the implementation registry.

    Returns:
        bool: Whether a registered implementation exists and subclasses BaseQuantizer.
    """
    from auto_round.algorithms.quantization.base import BaseQuantizer

    alg_cls = get_algorithm_class(config)
    return alg_cls is not None and issubclass(alg_cls, BaseQuantizer)


def split_quantization_configs(configs: list[Any]) -> tuple[list[Any], list[Any]]:
    """Split configs into preprocessor and quantizer config lists."""
    preprocessors = []
    quantizers = []
    for config in configs:
        if is_preprocessor_config(config):
            preprocessors.append(config)
        elif is_quantizer_config(config):
            quantizers.append(config)
    return preprocessors, quantizers


def _quantization_configs(configs: list[Any]) -> list[Any]:
    """Return configs that participate in quantization shared-field resolution."""
    from auto_round.algorithms.quantization.config import QuantizationConfig

    return [config for config in configs if isinstance(config, QuantizationConfig)]


def _public_config_attrs(config: Any) -> dict[str, Any]:
    """Public, data-like attrs used for structural shared config resolution."""
    return {
        key: value
        for key, value in vars(config).items()
        if key != "scheme" and key not in config.stage_local_fields and not key.startswith("_") and not callable(value)
    }


def _format_shared_config_values(field: str, values: list[tuple[Any, Any]]) -> str:
    parts = [f"{type(config).__name__}.{field}={value!r}" for config, value in values]
    return ", ".join(parts)


def resolve_shared_config_values(configs: list[Any], reconcile_non_scheme: bool = True) -> list[Any]:
    """Merge shared public attrs across quantization configs without naming fields.

    A field is shared when at least two quantization configs define the same
    public attribute. ``None`` means "not set" and inherits from the single
    non-None value, while conflicting non-None values fail fast. Configs that do
    not define a field do not participate in that field. Fields declared in
    ``stage_local_fields`` are never merged or propagated to another stage.

    Set *reconcile_non_scheme* to ``False`` for independent quantizer stages.
    """
    quant_configs = _quantization_configs(configs)
    if not reconcile_non_scheme:
        return configs
    attrs_by_config = [(config, _public_config_attrs(config)) for config in quant_configs]
    field_to_configs: dict[str, list[Any]] = defaultdict(list)
    for config, attrs in attrs_by_config:
        for attr_name in attrs:
            field_to_configs[attr_name].append(config)

    for attr_name, field_configs in field_to_configs.items():
        if len(field_configs) < 2:
            continue

        field_attrs = [
            (config, attrs[attr_name])
            for config, attrs in attrs_by_config
            if any(config is field_config for field_config in field_configs)
        ]
        non_none_values = [(config, value) for config, value in field_attrs if value is not None]
        unique_values = []
        for _, value in non_none_values:
            if not any(value == existing for existing in unique_values):
                unique_values.append(value)
        if len(unique_values) > 1:
            raise ValueError(
                f"Conflicting shared config field {attr_name!r}: "
                f"{_format_shared_config_values(attr_name, non_none_values)}. "
                "Use the same value for shared fields or leave it unset on secondary algorithms."
            )
        if len(unique_values) == 1:
            shared_value = unique_values[0]
            for config in field_configs:
                if getattr(config, attr_name) is None:
                    setattr(config, attr_name, shared_value)
    return configs


def sync_shared_config_from(source_config: Any, target_configs: list[Any]) -> None:
    """Propagate resolved source values to targets that already define matching attrs."""
    source_attrs = _public_config_attrs(source_config)
    for target in _quantization_configs(target_configs):
        if target is source_config:
            continue
        target_attrs = _public_config_attrs(target)
        for attr_name, source_value in source_attrs.items():
            if attr_name in target_attrs and source_value is not None:
                setattr(target, attr_name, source_value)


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
    "calib_num_inference_steps": "diffusion",
    "generator_seed": "diffusion",
    "pipeline_call_kwargs": "diffusion",
    "diffusion_tuning_cache_size": "diffusion",
}

_SCHEME_FIELDS = set(QuantizationScheme.get_attributes())


def _iter_registered_alg_configs() -> list[tuple[str, type]]:
    """Return each registered algorithm's canonical name and config class."""
    from auto_round.algorithms.registry import iter_algorithm_entries

    result = []
    seen = set()
    for entry in iter_algorithm_entries():
        factory = entry.config_factory
        if factory is None:
            continue
        config_cls = factory if isinstance(factory, type) else type(factory())
        if config_cls not in seen:
            seen.add(config_cls)
            result.append((entry.name, config_cls))
    return result


@functools.cache
def _discover_alg_config_fields(config_cls: type) -> frozenset:
    """Discover accepted config fields without maintaining per-algorithm lists."""
    from pydantic import BaseModel

    if issubclass(config_cls, BaseModel):
        return frozenset(config_cls.model_fields.keys())

    from auto_round.algorithms.config import AlgorithmConfig
    from auto_round.algorithms.quantization.config import QuantizationConfig

    fields = set()
    for klass in config_cls.__mro__:
        if klass in (QuantizationConfig, AlgorithmConfig, object):
            break
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        for name, parameter in inspect.signature(init).parameters.items():
            if name != "self" and parameter.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                fields.add(name)
    return frozenset(fields)


def _filter_supported_entry_kwargs(kwargs, *, context="AutoRound", strict=False):
    supported = {key: value for key, value in kwargs.items() if key in _ENTRY_KWARG_OWNERS}
    unknown = sorted(set(kwargs) - set(_ENTRY_KWARG_OWNERS))
    if unknown:
        if strict:
            raise TypeError(f"{context} received unsupported runtime kwargs with model_free=True: {', '.join(unknown)}")
        logger.warning_once(
            "%s received unsupported runtime kwargs %s. They will be ignored.",
            context,
            ", ".join(unknown),
        )
    return supported


def _split_entry_kwargs(kwargs, *, context="AutoRound", strict=False):
    buckets = {"route": {}, "compressor": {}, "base": {}, "mllm": {}, "diffusion": {}}
    for key, value in _filter_supported_entry_kwargs(kwargs, context=context, strict=strict).items():
        buckets[_ENTRY_KWARG_OWNERS[key]][key] = value
    return buckets


def _config_fields(config):
    return set(_discover_alg_config_fields(type(config)))


def _owning_algorithm_names(field_name: str) -> list[str]:
    return [cls.__name__ for _, cls in _iter_registered_alg_configs() if field_name in _discover_alg_config_fields(cls)]


def _normalize_alg_configs(alg_configs, direct_kwargs=None, *, strict=False):
    from auto_round.algorithms.quantization.config import QuantizationConfig
    from auto_round.algorithms.quantization.rtn.config import (
        RTN_VARIANT_DISABLE_OPT_RTN,
        RTNConfig,
        validate_rtn_variant_names,
    )
    from auto_round.algorithms.registry import normalize_algorithm_config, resolve_alg_config, resolve_algorithm_names
    from auto_round.algorithms.transforms import normalize_rotation_config
    from auto_round.algorithms.transforms.awq.config import (
        awq_disable_opt_rtn,
        rtn_inherited_opt_kwargs,
        sync_rtn_opt_rtn_from_awq,
    )
    from auto_round.algorithms.transforms.base import BaseRotationConfig

    direct_kwargs = dict(direct_kwargs or {})
    if "algorithm" in direct_kwargs:
        raise TypeError("`algorithm` is not supported; pass names or config objects through `alg_configs` instead.")
    if "backend" in direct_kwargs:
        raise ValueError(
            "Rotation backend selection must be nested in `rotation_config`; "
            "do not pass it as AutoRound(..., backend=...)."
        )
    scheme_fields = sorted((set(direct_kwargs) & _SCHEME_FIELDS) - {"rotation_config"})
    if scheme_fields:
        raise TypeError(
            "Scheme fields belong to AutoRound/Compressor, not algorithm configs: " + ", ".join(scheme_fields)
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
            raise ValueError("`alg_configs` must contain at least one algorithm name.")
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

    validate_rtn_variant_names(raw_configs)

    # An `rtn` / `opt-rtn` the caller actually named selects the RTN variant, the
    # same way `--algorithm rtn` / `--algorithm opt-rtn` does on the CLI. When
    # `alg_configs` is None the RTN config below is an implicit fallback, so it
    # must keep the existing flag/heuristic behaviour.
    named_rtn_variants = (
        [name for name in raw_configs if isinstance(name, str) and name in RTN_VARIANT_DISABLE_OPT_RTN]
        if alg_configs is not None
        else []
    )

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
        configs.append(config)

    awq_opt_rtn_policy = awq_disable_opt_rtn(configs)
    for index in pending_rtn_indices:
        config = RTNConfig(**rtn_inherited_opt_kwargs(config_kwargs, awq_opt_rtn_policy))
        configs[index] = normalize_algorithm_config(config)

    # ``iters=0`` has always selected RTN in the public entry and CLI. Apply
    # that rule after every input form has become a config so aliases, config
    # objects cannot choose
    # different quantizer implementations.
    from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig

    direct_iters = config_kwargs.get("iters")
    for index, config in enumerate(configs):
        effective_iters = direct_iters if direct_iters is not None else getattr(config, "iters", None)
        if isinstance(config, SignRoundConfig) and effective_iters == 0:
            rtn_kwargs = rtn_inherited_opt_kwargs(config_kwargs, awq_disable_opt_rtn(configs))
            rtn_config = RTNConfig(**rtn_kwargs)
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

    # Apply existing fields before deriving defaults for a different algorithm
    # class. Only then resolve fields introduced by that conversion (e.g. QAD
    # topk/qad_lora_rank), independently of direct-kwargs insertion order.
    known_fields = set().union(*(_config_fields(config) for config in configs))
    before_conversion = {key: value for key, value in config_kwargs.items() if key in known_fields}
    after_conversion = {key: value for key, value in config_kwargs.items() if key not in known_fields}
    for overrides in (before_conversion, after_conversion):
        _, block_configs = split_quantization_configs(configs)
        for key, value in overrides.items():
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
                continue
            targets = [config for config in configs if key in _config_fields(config)]
            if not targets:
                # ``iters`` is a legacy route selector. RTN intentionally has no
                # iterative parameter after the zero-iteration route is selected.
                if key == "iters" and any(isinstance(config, RTNConfig) for config in configs):
                    continue
                if strict:
                    raise TypeError(f"Unsupported parameter '{key}' for the selected algorithm with model_free=True.")
                owners = _owning_algorithm_names(key)
                owner = "/".join(owners) if owners else "the selected algorithm"
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
        configs = [normalize_algorithm_config(config) for config in configs]
    sync_rtn_opt_rtn_from_awq(configs)
    if named_rtn_variants:
        # Applied last, on RTN configs only (AWQ keeps its own policy), so the
        # algorithm name is not overridden by the AWQ inheritance above or by a
        # contradicting `disable_opt_rtn` argument. `orig_disable_opt_rtn` marks
        # it as user policy, which is what stops the W8A16/W8A8 efficiency
        # heuristic from silently switching the variant back.
        forced_disable_opt_rtn = RTN_VARIANT_DISABLE_OPT_RTN[named_rtn_variants[0]]
        for config in configs:
            if isinstance(config, RTNConfig):
                config.disable_opt_rtn = forced_disable_opt_rtn
                if hasattr(config, "orig_disable_opt_rtn"):
                    config.orig_disable_opt_rtn = forced_disable_opt_rtn
    configs = [normalize_algorithm_config(config) for config in configs]
    for config in configs:
        if isinstance(config, SignRoundConfig):
            config.validate_lora_config()
    return configs


def _prepare_entry_kwargs(alg_configs, direct_kwargs, *, strict=False):
    scheme_overrides = {
        key: value for key, value in direct_kwargs.items() if key in _SCHEME_FIELDS and value is not None
    }
    algorithm_kwargs = {
        key: value for key, value in direct_kwargs.items() if key not in _SCHEME_FIELDS or key == "rotation_config"
    }
    configs = _normalize_alg_configs(alg_configs, algorithm_kwargs, strict=strict)
    runtime_kwargs = {key: value for key, value in direct_kwargs.items() if key in _ENTRY_KWARG_OWNERS}
    runtime_kwargs["scheme_overrides"] = scheme_overrides
    return configs, runtime_kwargs
