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

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

TargetPredicate = Callable[[str, torch.nn.Module], bool]


def normalize_tensors(output: Any, indices: tuple[int, ...] | None = None) -> tuple[torch.Tensor, ...]:
    """Return floating-point tensors used by the smooth-search output objective."""

    if indices is not None:
        if not isinstance(output, (tuple, list)):
            raise TypeError("indexed smooth-search output must be a tuple or list")
        output = tuple(output[index] for index in indices)

    tensors: list[torch.Tensor] = []

    def collect(value: Any) -> None:
        if torch.is_tensor(value):
            if value.is_floating_point():
                tensors.append(value)
            return
        if isinstance(value, Mapping):
            for item in value.values():
                collect(item)
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                collect(item)

    collect(output)
    if not tensors:
        raise ValueError("smooth-search evaluation produced no floating-point tensors")
    return tuple(tensors)


def filter_supported_kwargs(module: torch.nn.Module, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Keep keyword arguments accepted by a module's forward method."""

    parameters = inspect.signature(module.forward).parameters
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return dict(kwargs)
    return {name: value for name, value in kwargs.items() if name in parameters}


@dataclass(frozen=True)
class SmoothSearchGroup:
    """A projection group sharing one input scale and one output objective."""

    key: str
    projection_names: tuple[str, ...]
    projections: tuple[torch.nn.Linear, ...]
    projection_input_key: str
    projection_input_module: torch.nn.Module
    evaluation_input_key: str
    evaluation_module: torch.nn.Module
    output_indices: tuple[int, ...] | None = None
    output_splits: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.projections:
            raise ValueError(f"smooth-search group {self.key!r} has no projections")
        if len(self.projection_names) != len(self.projections):
            raise ValueError(f"smooth-search group {self.key!r} has mismatched names and projections")
        widths = {projection.in_features for projection in self.projections}
        if len(widths) != 1:
            raise ValueError(f"smooth-search group {self.key!r} projections must share an input width")
        if self.output_splits and sum(self.output_splits) != len(self.projections):
            raise ValueError(f"smooth-search group {self.key!r} output splits do not cover its projections")

    def filter_evaluation_kwargs(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        return filter_supported_kwargs(self.evaluation_module, kwargs)

    def normalize_output(self, output: Any) -> tuple[torch.Tensor, ...]:
        return normalize_tensors(output, self.output_indices)


def module_global_name(block: torch.nn.Module, local_name: str) -> str:
    prefix = str(getattr(block, "global_name", block.__class__.__name__))
    return f"{prefix}.{local_name}" if local_name else prefix


def resolve_module(root: torch.nn.Module, path: str) -> torch.nn.Module | None:
    module: torch.nn.Module = root
    for part in path.split("."):
        if part.isdigit() and isinstance(module, (torch.nn.ModuleList, torch.nn.Sequential)):
            index = int(part)
            if index >= len(module):
                return None
            module = module[index]
        else:
            candidate = getattr(module, part, None)
            if not isinstance(candidate, torch.nn.Module):
                return None
            module = candidate
    return module


def generic_linear_groups(
    block: torch.nn.Module,
    is_target: TargetPredicate,
    *,
    consumed: set[int] | None = None,
) -> list[SmoothSearchGroup]:
    consumed = consumed or set()
    groups = []
    for local_name, module in block.named_modules():
        if not local_name or id(module) in consumed or not is_target(local_name, module):
            continue
        if not isinstance(module, torch.nn.Linear):
            continue
        name = module_global_name(block, local_name)
        groups.append(
            SmoothSearchGroup(
                key=name,
                projection_names=(name,),
                projections=(module,),
                projection_input_key=name,
                projection_input_module=module,
                evaluation_input_key=name,
                evaluation_module=module,
                output_splits=(1,),
            )
        )
    return groups
