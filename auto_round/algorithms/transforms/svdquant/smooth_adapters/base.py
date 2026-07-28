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

from collections.abc import Callable
from dataclasses import dataclass

import torch

TargetPredicate = Callable[[str, torch.nn.Module], bool]


@dataclass(frozen=True)
class SmoothSearchGroup:
    """Projection group that shares an input scale and low-rank down factor."""

    key: str
    projection_names: tuple[str, ...]
    projections: tuple[torch.nn.Linear, ...]
    projection_input_module: torch.nn.Module
    evaluation_module: torch.nn.Module
    output_indices: tuple[int, ...] | None = None
    output_splits: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.projections:
            raise ValueError(f"SVDQuant group {self.key!r} has no projections.")
        if len(self.projection_names) != len(self.projections):
            raise ValueError(f"SVDQuant group {self.key!r} has mismatched names and projections.")
        if len({projection.in_features for projection in self.projections}) != 1:
            raise ValueError(f"SVDQuant group {self.key!r} projections must share an input width.")
        if self.output_splits and sum(self.output_splits) != len(self.projections):
            raise ValueError(f"SVDQuant group {self.key!r} output splits do not cover its projections.")


def module_global_name(block: torch.nn.Module, local_name: str) -> str:
    prefix = str(getattr(block, "global_name", block.__class__.__name__))
    return f"{prefix}.{local_name}" if local_name else prefix


def resolve_module(root: torch.nn.Module, path: str) -> torch.nn.Module | None:
    module = root
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
                projection_input_module=module,
                evaluation_module=module,
                output_splits=(1,),
            )
        )
    return groups
