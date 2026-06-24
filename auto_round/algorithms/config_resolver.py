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

from collections import defaultdict
from dataclasses import fields
from typing import Any


def get_algorithm_class(config: Any):
    """Return the registered implementation class for a quantization config."""
    from auto_round.algorithms.registry import normalize_algorithm_config, resolve_pipeline_member

    try:
        return resolve_pipeline_member(normalize_algorithm_config(config))
    except ValueError:
        return None


def is_preprocessor_config(config: Any) -> bool:
    """Whether *config* resolves to a BaseWeightTransformer implementation."""
    from auto_round.algorithms.transforms.base import BaseWeightTransformer

    alg_cls = get_algorithm_class(config)
    return alg_cls is not None and issubclass(alg_cls, BaseWeightTransformer)


def is_block_quantizer_config(config: Any) -> bool:
    """Whether *config* resolves to a BaseQuantizer implementation."""
    from auto_round.algorithms.quantization.base import BaseQuantizer

    alg_cls = get_algorithm_class(config)
    return alg_cls is not None and issubclass(alg_cls, BaseQuantizer)


def split_quantization_configs(configs: list[Any]) -> tuple[list[Any], list[Any]]:
    """Split configs into preprocessor and block-quantizer config lists."""
    preprocessors = []
    block_quantizers = []
    for config in configs:
        if is_preprocessor_config(config):
            preprocessors.append(config)
        elif is_block_quantizer_config(config):
            block_quantizers.append(config)
    return preprocessors, block_quantizers


def _quantization_configs(configs: list[Any]) -> list[Any]:
    """Return configs that participate in quantization shared-field resolution."""
    from auto_round.algorithms.quantization.config import QuantizationConfig

    return [config for config in configs if isinstance(config, QuantizationConfig)]


def _public_config_attrs(config: Any) -> dict[str, Any]:
    """Public, data-like attrs used for structural shared config resolution."""
    return {
        key: value
        for key, value in vars(config).items()
        if key != "scheme" and not key.startswith("_") and not callable(value)
    }


def _format_shared_config_values(field: str, values: list[tuple[Any, Any]]) -> str:
    parts = [f"{type(config).__name__}.{field}={value!r}" for config, value in values]
    return ", ".join(parts)


def _resolve_shared_scheme_values(configs: list[Any]) -> None:
    from auto_round.schemes import QuantizationScheme

    scheme_fields = {field.name for field in fields(QuantizationScheme)}
    field_values: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
    for config in configs:
        for attr_name in getattr(config, "_user_set_scheme_fields", set()):
            if attr_name not in scheme_fields:
                continue
            value = getattr(config, attr_name, None)
            if value is not None:
                field_values[attr_name].append((config, value))

    for attr_name, values in field_values.items():
        unique_values = []
        for _, value in values:
            if not any(value == existing for existing in unique_values):
                unique_values.append(value)
        if len(unique_values) > 1:
            raise ValueError(
                f"Conflicting shared scheme field {attr_name!r}: "
                f"{_format_shared_config_values(attr_name, values)}. "
                "Use the same value for shared fields or pass scheme arguments through Compressor."
            )
        shared_value = unique_values[0]
        for config in configs:
            if hasattr(config, "scheme") and attr_name not in getattr(config, "_user_set_scheme_fields", set()):
                setattr(config.scheme, attr_name, shared_value)


def resolve_shared_config_values(configs: list[Any]) -> list[Any]:
    """Merge shared public attrs across quantization configs without naming fields.

    A field is shared when at least two quantization configs define the same
    public attribute. ``None`` means "not set" and inherits from the single
    non-None value, while conflicting non-None values fail fast. Configs that do
    not define a field do not participate in that field.
    """
    quant_configs = _quantization_configs(configs)
    _resolve_shared_scheme_values(quant_configs)
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
