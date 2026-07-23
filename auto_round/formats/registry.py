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

from typing import Any, Tuple


class FormatRegistry:
    """Explicit name-to-format registry with duplicate registration protection."""

    def __init__(self) -> None:
        self._entries: dict[str, Any] = {}

    def register(self, name: str, value: Any) -> None:
        if name in self._entries:
            raise ValueError(f"Format '{name}' is already registered")
        self._entries[name] = value

    def get(self, name: str) -> Any:
        try:
            return self._entries[name]
        except KeyError as error:
            supported = ", ".join(sorted(self._entries)) or "<none>"
            raise KeyError(f"Unknown format '{name}'. Supported formats: {supported}") from error

    def resolve(self, name: str) -> Any:
        return self.get(name)

    def items(self) -> Tuple[Tuple[str, Any], ...]:
        return tuple(self._entries.items())


_BUILTIN_REGISTRY: FormatRegistry | None = None


def get_format_registry() -> FormatRegistry:
    """Return the registry populated by built-in backend module decorators."""
    global _BUILTIN_REGISTRY
    if _BUILTIN_REGISTRY is None:
        from auto_round.formats.base import OutputFormat

        registry = FormatRegistry()
        for name, executor_class in OutputFormat._format_list.items():
            registry.register(name, executor_class)
        _BUILTIN_REGISTRY = registry
    return _BUILTIN_REGISTRY
