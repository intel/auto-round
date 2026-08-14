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
        if entry.config_factory is None or not isinstance(entry.config_factory, type):
            raise KeyError(f"No config class registered for algorithm '{name}'.")
        return entry.config_factory

    @classmethod
    def resolve_alias(cls, user_name: str) -> str | None:
        return resolve_algorithm_alias(user_name)

    @classmethod
    def add_group(cls, name: str, group) -> None:
        _add_registered_arguments(group, _parameter_registry(cls.get(name)))

    @classmethod
    def add_groups(cls, parser) -> None:
        for entry in iter_algorithm_entries():
            if entry.config_factory is None or not isinstance(entry.config_factory, type):
                continue
            registry = _parameter_registry(entry.config_factory)
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
        if not ({"awq", "rtn", "auto_round"} & seen):
            canonical.append("rtn" if getattr(args, "iters", 0) == 0 else "auto_round")
        if getattr(args, "iters", None) == 0:
            canonical = ["rtn" if name == "auto_round" else name for name in canonical]

        configs = []
        for name in canonical:
            config_cls = cls.get(name)
            kwargs = _parameter_registry(config_cls).config_kwargs(args)
            if getattr(config_cls, "cli_include_common_args", True):
                kwargs.update(common_kwargs)
            configs.append(config_cls(**kwargs))
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
