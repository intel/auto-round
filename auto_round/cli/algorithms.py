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
from typing import Any

from auto_round.algorithms.config import AlgorithmConfig, AlgorithmParameterRegistry
from auto_round.algorithms.registry import (
    get_algorithm_entry,
    iter_algorithm_entries,
    resolve_algorithm_alias,
    resolve_algorithm_names,
)


def _parameter_registry(config_cls: type) -> AlgorithmParameterRegistry:
    if not issubclass(config_cls, AlgorithmConfig):
        return AlgorithmParameterRegistry()
    return config_cls.get_registered_args()


def _add_registered_arguments(group, registry: AlgorithmParameterRegistry) -> None:
    mutex_groups = {}
    for parameter in registry.parameters:
        target = group
        if parameter.mutex_group is not None:
            if parameter.mutex_group not in mutex_groups:
                mutex_groups[parameter.mutex_group] = group.add_mutually_exclusive_group()
            target = mutex_groups[parameter.mutex_group]
        kwargs = dict(parameter.argparse_kwargs)
        kwargs.pop("fallback", None)
        if kwargs.get("action") == "boolean_optional":
            kwargs["action"] = argparse.BooleanOptionalAction
        target.add_argument(*parameter.option_strings, **kwargs)


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
        for entry in iter_algorithm_entries():
            if entry.config_factory is None:
                continue
            config_cls = (
                entry.config_factory if isinstance(entry.config_factory, type) else type(entry.config_factory())
            )
            registry = _parameter_registry(config_cls)
            if registry.parameters:
                group = parser.add_argument_group(f"Algorithm: {entry.name}")
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
