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

"""Algorithm configuration metadata used by generic front ends."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, ClassVar


@dataclass(frozen=True)
class AlgorithmParameter:
    option_strings: tuple[str, ...]
    field: str
    argparse_kwargs: dict[str, Any]
    mutex_group: int | None = None

    @property
    def dest(self) -> str:
        return self.argparse_kwargs.get("dest", self.option_strings[0].lstrip("-").replace("-", "_"))


class _MutuallyExclusiveParameterRegistry:
    def __init__(self, registry: "AlgorithmParameterRegistry", group_id: int) -> None:
        self._registry = registry
        self._group_id = group_id

    def add_argument(self, *option_strings: str, field: str, **kwargs) -> None:
        self._registry._add_argument(*option_strings, field=field, mutex_group=self._group_id, **kwargs)


class AlgorithmParameterRegistry:
    """Collect algorithm parameter declarations without exposing argparse to configs."""

    def __init__(self) -> None:
        self.parameters: list[AlgorithmParameter] = []
        self._fields: set[str] = set()
        self._next_mutex_group = 0

    def add_argument(self, *option_strings: str, field: str, **kwargs) -> None:
        self._add_argument(*option_strings, field=field, mutex_group=None, **kwargs)

    def _add_argument(self, *option_strings: str, field: str, mutex_group: int | None, **kwargs) -> None:
        if not option_strings or any(not option.startswith("-") for option in option_strings):
            raise ValueError("Algorithm arguments must define one or more option strings.")
        if not field or not field.isidentifier():
            raise ValueError(f"Invalid config field {field!r}.")
        duplicate = next((parameter for parameter in self.parameters if parameter.field == field), None)
        if duplicate is not None and not (
            mutex_group is not None
            and duplicate.mutex_group == mutex_group
            and duplicate.dest == kwargs.get("dest", option_strings[0].lstrip("-").replace("-", "_"))
        ):
            raise ValueError(f"Config field {field!r} is already registered.")
        self._fields.add(field)
        self.parameters.append(AlgorithmParameter(tuple(option_strings), field, dict(kwargs), mutex_group))

    def add_mutually_exclusive_group(self) -> _MutuallyExclusiveParameterRegistry:
        group_id = self._next_mutex_group
        self._next_mutex_group += 1
        return _MutuallyExclusiveParameterRegistry(self, group_id)

    def config_kwargs(self, args) -> dict[str, Any]:
        values = {}
        for parameter in self.parameters:
            if not hasattr(args, parameter.dest):
                continue
            value = getattr(args, parameter.dest)
            fallback = parameter.argparse_kwargs.get("fallback")
            values[parameter.field] = fallback if value is None and fallback is not None else value
        return values


class AlgorithmConfig:
    """Base mixin for config-owned command-line parameter declarations."""

    cli_include_common_args: ClassVar[bool] = True

    @classmethod
    def register_args(cls, registry: AlgorithmParameterRegistry) -> None:
        """Register algorithm-specific parameters. The default declares none."""

    @classmethod
    def get_registered_args(cls) -> AlgorithmParameterRegistry:
        registry = AlgorithmParameterRegistry()
        cls.register_args(registry)
        return registry
