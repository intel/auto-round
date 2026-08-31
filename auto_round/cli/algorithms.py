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

"""Generic CLI adapter for algorithm-owned parameter declarations."""

from __future__ import annotations

import argparse
import inspect
from dataclasses import replace
from typing import Any, get_args, get_origin, get_type_hints

from auto_round.algorithms.config import AlgorithmConfig, AlgorithmParameterRegistry
from auto_round.algorithms.registry import (
    get_algorithm_entry,
    iter_algorithm_entries,
    resolve_algorithm_alias,
    resolve_algorithm_names,
)


def _parameter_registry(config_cls: type) -> AlgorithmParameterRegistry:
    if _has_custom_register_args(config_cls):
        return config_cls.get_registered_args()
    return _fallback_parameter_registry(config_cls)


def _has_custom_register_args(config_cls: type) -> bool:
    if not issubclass(config_cls, AlgorithmConfig):
        return False
    for cls in config_cls.__mro__:
        if cls is AlgorithmConfig:
            break
        if "register_args" in cls.__dict__:
            return True
    return False


def _fallback_parameter_registry(config_cls: type) -> AlgorithmParameterRegistry:
    """Derive simple CLI arguments from a config that has no custom registration."""
    registry = AlgorithmParameterRegistry()
    init = config_cls.__init__
    try:
        type_hints = get_type_hints(init)
    except (NameError, TypeError):
        type_hints = {}

    for name, parameter in inspect.signature(init).parameters.items():
        if name in {"self", "algorithm"} or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = type_hints.get(name, parameter.annotation)
        default = parameter.default
        if annotation is inspect.Parameter.empty:
            annotation = type(default) if default is not inspect.Parameter.empty and default is not None else str
        if annotation is bool or isinstance(default, bool):
            action = argparse.BooleanOptionalAction
            kwargs = {"action": action, "default": argparse.SUPPRESS}
        else:
            origin = get_origin(annotation)
            args = [arg for arg in get_args(annotation) if arg is not type(None)]
            value_type = args[0] if origin is not None and args else annotation
            kwargs = {"type": value_type, "default": argparse.SUPPRESS}
        registry.add_argument(f"--{name}", field=name, **kwargs)
    return registry


def _argument_compatibility_key(parameter) -> tuple:
    kwargs = parameter.argparse_kwargs
    return (
        kwargs.get("action"),
        kwargs.get("type"),
        tuple(kwargs.get("choices", ())) if kwargs.get("choices") is not None else None,
        kwargs.get("nargs"),
        kwargs.get("const"),
        kwargs.get("dest", parameter.dest),
    )


def _existing_action_compatibility_key(action) -> tuple:
    return (
        type(action),
        action.type,
        tuple(action.choices) if action.choices is not None else None,
        action.nargs,
        action.const,
        action.dest,
    )


def _check_existing_parser_argument(parser, parameter, *, fallback: bool = False) -> bool:
    """Reuse an existing common argument when its parsing semantics match."""
    actions = [
        parser._option_string_actions[option]
        for option in parameter.option_strings
        if option in parser._option_string_actions
    ]
    if not actions:
        return False
    action = actions[0]
    parameter_key = _argument_compatibility_key(parameter)
    action_key = _existing_action_compatibility_key(action)
    if parameter_key[1:] != action_key[1:]:
        if fallback:
            return True
        option = next(option for option in parameter.option_strings if option in parser._option_string_actions)
        raise ValueError(f"incompatible shared CLI argument {option!r}")
    return True


def _merge_parameter(merged, parameter):
    """Merge one shared CLI parameter or raise for incompatible definitions."""
    overlapping_options = set(merged.option_strings) & set(parameter.option_strings)
    if not overlapping_options and merged.dest != parameter.dest:
        return None
    if merged.dest != parameter.dest or _argument_compatibility_key(merged) != _argument_compatibility_key(parameter):
        option = sorted(overlapping_options)[0] if overlapping_options else merged.dest
        raise ValueError(f"incompatible shared CLI argument {option!r}")
    kwargs = dict(merged.argparse_kwargs)
    kwargs["default"] = argparse.SUPPRESS
    return replace(
        merged,
        option_strings=tuple(dict.fromkeys(merged.option_strings + parameter.option_strings)),
        argparse_kwargs=kwargs,
    )


def _add_parameter(group, parameter) -> None:
    kwargs = dict(parameter.argparse_kwargs)
    kwargs.pop("fallback", None)
    if kwargs.get("action") == "boolean_optional":
        kwargs["action"] = argparse.BooleanOptionalAction
    group.add_argument(*parameter.option_strings, **kwargs)


def _add_registered_arguments(group, registry: AlgorithmParameterRegistry) -> None:
    mutex_groups = {}
    for parameter in registry.parameters:
        target = group
        if parameter.mutex_group is not None:
            if parameter.mutex_group not in mutex_groups:
                mutex_groups[parameter.mutex_group] = group.add_mutually_exclusive_group()
            target = mutex_groups[parameter.mutex_group]
        _add_parameter(target, parameter)


class AlgorithmHandler:
    """Generic discovery, argparse adaptation, and config construction."""

    @classmethod
    def get(cls, name: str) -> type:
        entry = get_algorithm_entry(name)
        if entry.config_factory is None:
            raise KeyError(f"No config class registered for algorithm '{name}'.")
        return entry.config_factory if isinstance(entry.config_factory, type) else type(entry.config_factory())

    @classmethod
    def resolve_alias(cls, user_name: str) -> str | None:
        return resolve_algorithm_alias(user_name)

    @classmethod
    def add_group(cls, name: str, group) -> None:
        _add_registered_arguments(group, _parameter_registry(cls.get(name)))

    @classmethod
    def add_groups(cls, parser) -> None:
        merged_parameters = []
        parameters_by_group = {}
        locations = []
        for entry in iter_algorithm_entries():
            if entry.config_factory is None:
                continue
            config_cls = (
                entry.config_factory if isinstance(entry.config_factory, type) else type(entry.config_factory())
            )
            registry = _parameter_registry(config_cls)
            fallback = not _has_custom_register_args(config_cls)
            if registry.parameters:
                group_name = f"Algorithm: {entry.name}"
                parameters_by_group.setdefault(group_name, [])
                for parameter in registry.parameters:
                    if _check_existing_parser_argument(parser, parameter, fallback=fallback):
                        continue
                    existing_index = next(
                        (
                            index
                            for index, item in enumerate(merged_parameters)
                            if locations[index][0] != group_name
                            and (
                                item.dest == parameter.dest or set(item.option_strings) & set(parameter.option_strings)
                            )
                        ),
                        None,
                    )
                    if existing_index is not None:
                        merged = _merge_parameter(merged_parameters[existing_index], parameter)
                        merged_parameters[existing_index] = merged
                        existing_group, existing_position = locations[existing_index]
                        parameters_by_group[existing_group][existing_position] = merged
                        continue
                    merged_parameters.append(parameter)
                    parameters_by_group[group_name].append(parameter)
                    locations.append((group_name, len(parameters_by_group[group_name]) - 1))

        for group_name, parameters in parameters_by_group.items():
            group = parser.add_argument_group(group_name)
            registry = AlgorithmParameterRegistry()
            registry.parameters = parameters
            _add_registered_arguments(group, registry)

    @classmethod
    def build_configs(cls, args, common_kwargs: dict[str, Any]) -> list:
        raw = getattr(args, "algorithm", None) or ""
        names = [name.strip().lower() for name in raw.split(",") if name.strip()]

        if getattr(args, "rotation_hadamard_type", None) and "hadamard" not in names:
            names.append("hadamard")

        canonical = resolve_algorithm_names(names, ignore_unknown=True)
        seen = set(canonical)
        if not ({"rtn", "auto_round"} & seen):
            canonical.append("rtn" if "awq" in seen or getattr(args, "iters", 0) == 0 else "auto_round")
        if getattr(args, "iters", None) == 0:
            canonical = ["rtn" if name == "auto_round" else name for name in canonical]

        configs = []
        built_configs = {}
        if "awq" in canonical:
            awq_entry = get_algorithm_entry("awq")
            awq_cls = cls.get("awq")
            awq_kwargs = _parameter_registry(awq_cls).config_kwargs(args)
            if getattr(awq_cls, "cli_include_common_args", True):
                awq_kwargs.update(common_kwargs)
            if isinstance(awq_entry.config_factory, type):
                awq_config = awq_entry.config_factory(**awq_kwargs)
            elif awq_kwargs:
                awq_config = awq_cls(**awq_kwargs)
            else:
                awq_config = awq_entry.config_factory()
            explicit_opt_rtn = getattr(args, "disable_opt_rtn", None)
            if explicit_opt_rtn is not None:
                awq_config.disable_opt_rtn = explicit_opt_rtn
            built_configs["awq"] = awq_config
        awq_disable_opt_rtn = getattr(built_configs.get("awq"), "disable_opt_rtn", None)
        for name in canonical:
            if name in built_configs:
                configs.append(built_configs[name])
                continue
            entry = get_algorithm_entry(name)
            config_cls = cls.get(name)
            kwargs = _parameter_registry(config_cls).config_kwargs(args)
            if getattr(config_cls, "cli_include_common_args", True):
                kwargs.update(common_kwargs)
            if name == "rtn" and getattr(args, "disable_opt_rtn", None) is None and awq_disable_opt_rtn is not None:
                kwargs["disable_opt_rtn"] = awq_disable_opt_rtn
            if isinstance(entry.config_factory, type):
                config = entry.config_factory(**kwargs)
            elif kwargs:
                config = config_cls(**kwargs)
            else:
                config = entry.config_factory()
            configs.append(config)
        return configs

    @classmethod
    def format_listing(cls) -> str:
        lines = []
        for entry in iter_algorithm_entries():
            if entry.config_factory is None:
                continue
            other = [alias for alias in entry.aliases if alias != entry.name]
            alias_str = f" (aliases: {', '.join(other)})" if other else ""
            lines.append(f"- {entry.name}{alias_str}: {entry.summary}")
        return "\n".join(lines)

    @classmethod
    def format_detail(cls, name: str) -> str:
        canonical = cls.resolve_alias(name)
        if canonical is None:
            supported = [entry.name for entry in iter_algorithm_entries() if entry.config_factory is not None]
            raise ValueError(f"Unknown algorithm '{name}'. Supported: {', '.join(supported)}.")
        entry = get_algorithm_entry(canonical)
        lines = [f"{entry.name}: {entry.summary}"]
        other = [alias for alias in entry.aliases if alias != entry.name]
        if other:
            lines.append(f"Aliases: {', '.join(other)}")
        temp = argparse.ArgumentParser(add_help=False)
        group = temp.add_argument_group(f"Flags for {entry.name}")
        cls.add_group(canonical, group)
        for action in group._group_actions:
            flags = ", ".join(action.option_strings)
            default = f" (default: {action.default})" if action.default is not None else ""
            lines.append(f"  {flags}: {action.help or ''}{default}")
        return "\n".join(lines)
