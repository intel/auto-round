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

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import torch

from auto_round.algorithms.transforms.smoothing.groups import SmoothGroup
from auto_round.algorithms.transforms.smoothing.search import search_candidates

_CalibrationT_contra = TypeVar("_CalibrationT_contra", contravariant=True)
_CandidateT = TypeVar("_CandidateT")
_AppliedT_co = TypeVar("_AppliedT_co", covariant=True)


class SmoothStrategy(Protocol[_CalibrationT_contra, _CandidateT, _AppliedT_co]):
    """Extension contract; statistics, candidates, loss and deployment are policies.

    Each group uses a separate strategy instance. Scoring must restore temporary
    mutations before returning or raising. clear() releases only strategy-owned
    state, preserving calibration caches shared with other groups or passes.
    """

    def prepare(self, group: SmoothGroup, calibration: _CalibrationT_contra) -> None: ...

    def candidates(self) -> Iterable[_CandidateT]: ...

    def score(self, candidate: _CandidateT) -> float | torch.Tensor: ...

    def apply(self, candidate: _CandidateT) -> _AppliedT_co: ...

    def clear(self) -> None: ...


@dataclass(frozen=True)
class SmoothResult(Generic[_CandidateT]):
    group: SmoothGroup
    candidate: _CandidateT
    error: float


class SmoothEngine:
    """Algorithm-independent smoothing with separate search and apply phases."""

    @contextmanager
    def session(
        self, strategy: SmoothStrategy[_CalibrationT_contra, _CandidateT, _AppliedT_co]
    ) -> Iterator[SmoothStrategy[_CalibrationT_contra, _CandidateT, _AppliedT_co]]:
        try:
            yield strategy
        finally:
            strategy.clear()

    @torch.no_grad()
    def search(
        self,
        group: SmoothGroup,
        calibration: _CalibrationT_contra,
        strategy: SmoothStrategy[_CalibrationT_contra, _CandidateT, _AppliedT_co],
    ) -> SmoothResult[_CandidateT]:
        strategy.prepare(group, calibration)
        candidate, error = search_candidates(strategy.candidates(), strategy.score, module_name=group.key)
        return SmoothResult(group, candidate, error)

    @torch.no_grad()
    def apply(
        self,
        group: SmoothGroup,
        strategy: SmoothStrategy[_CalibrationT_contra, _CandidateT, _AppliedT_co],
        result: SmoothResult[_CandidateT],
    ) -> _AppliedT_co:
        if result.group is not group:
            raise ValueError("Smooth result belongs to a different group.")
        return strategy.apply(result.candidate)

    def run(
        self,
        group: SmoothGroup,
        calibration: _CalibrationT_contra,
        strategy: SmoothStrategy[_CalibrationT_contra, _CandidateT, _AppliedT_co],
    ) -> _AppliedT_co:
        with self.session(strategy):
            return self.apply(group, strategy, self.search(group, calibration, strategy))
