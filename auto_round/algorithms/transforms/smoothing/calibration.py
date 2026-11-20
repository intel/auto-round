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

import random
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import torch

from auto_round.logger import logger
from auto_round.utils.model import map_nested_tensors

_GroupT = TypeVar("_GroupT")


def detach_to_cpu(value: Any) -> Any:
    return map_nested_tensors(value, lambda tensor: tensor.detach().to("cpu", copy=True))


def move_to_device(value: Any, device: torch.device, dtype: torch.dtype | None = None) -> Any:
    def move(tensor: torch.Tensor) -> torch.Tensor:
        target_dtype = dtype if dtype is not None and tensor.is_floating_point() else tensor.dtype
        return tensor.to(device=device, dtype=target_dtype)

    return map_nested_tensors(value, move)


@dataclass
class CapturedEvaluation:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    output: Any


@dataclass
class SmoothGroupCalibration(Generic[_GroupT]):
    group: _GroupT
    limit: int
    projection_inputs: list[torch.Tensor] = field(default_factory=list)
    evaluation_calls: list[CapturedEvaluation] = field(default_factory=list)
    seen_calls: int = 0
    pending_slot: int | None = None
    pending_input: torch.Tensor | None = None
    random: random.Random = field(default_factory=lambda: random.Random(0))

    def begin_call(self, inputs: torch.Tensor) -> None:
        self.seen_calls += 1
        if len(self.projection_inputs) < self.limit:
            slot = len(self.projection_inputs)
        else:
            candidate = self.random.randrange(self.seen_calls)
            slot = candidate if candidate < self.limit else None
        self.pending_slot = slot
        self.pending_input = detach_to_cpu(inputs) if slot is not None else None

    def finish_call(self, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
        slot = self.pending_slot
        captured_input = self.pending_input
        self.pending_slot = None
        self.pending_input = None
        if slot is None or captured_input is None:
            return
        captured = CapturedEvaluation(detach_to_cpu(args), detach_to_cpu(kwargs), detach_to_cpu(output))
        if slot == len(self.projection_inputs):
            self.projection_inputs.append(captured_input)
            self.evaluation_calls.append(captured)
        else:
            self.projection_inputs[slot] = captured_input
            self.evaluation_calls[slot] = captured

    def clear(self) -> None:
        """Release retained and pending calls, including failed captures."""
        self.projection_inputs.clear()
        self.evaluation_calls.clear()
        self.pending_input = None
        self.pending_slot = None
        self.seen_calls = 0


def clear_caches(*caches: MutableMapping, scope: str) -> None:
    """Release algorithm-owned caches with common lifecycle logging."""
    entries = sum(len(cache) for cache in caches)
    for cache in caches:
        cache.clear()
    logger.debug("Smooth caches cleared for %s: entries=%d", scope, entries)
