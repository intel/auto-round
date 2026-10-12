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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from auto_round.algorithms.transforms.smoothing.replay import filter_supported_kwargs, normalize_tensors


@dataclass(frozen=True)
class SmoothGroup:
    """Targets sharing an input scale and an evaluation boundary.

    Model adapters own discovery. Shared scale does not imply shared low-rank
    factors, and strategies own target-specific dimension checks."""

    key: str
    projection_names: tuple[str, ...]
    projections: tuple[torch.nn.Module, ...]
    projection_input_key: str
    projection_input_module: torch.nn.Module
    evaluation_input_key: str
    evaluation_module: torch.nn.Module
    output_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.projections:
            raise ValueError(f"Smooth group {self.key!r} has no projections.")
        if len(self.projection_names) != len(self.projections):
            raise ValueError(f"Smooth group {self.key!r} has mismatched names and projections.")

    def filter_evaluation_kwargs(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        return filter_supported_kwargs(self.evaluation_module, kwargs)

    def normalize_output(self, output: Any) -> tuple[torch.Tensor, ...]:
        return normalize_tensors(output, self.output_indices)
