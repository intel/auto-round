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

import math
from collections.abc import Callable, Iterable
from typing import TypeVar

import torch

from auto_round.algorithms.transforms.smoothing.errors import InvalidSmoothCandidateError, NoFiniteCandidateError
from auto_round.logger import logger

_CandidateT = TypeVar("_CandidateT")


def _select_best_layer_result(
    candidates: Iterable[tuple[_CandidateT, float | torch.Tensor]],
    *,
    module_name: str,
    require_finite: bool = True,
) -> tuple[_CandidateT, float]:
    """Keep the candidate and its score together, including a valid None candidate."""
    best = None
    for candidate, error in candidates:
        error_value = error.item() if torch.is_tensor(error) else float(error)
        if (not require_finite or math.isfinite(error_value)) and error_value < (best[1] if best else float("inf")):
            best = (candidate, error_value)
    if best is None:
        raise NoFiniteCandidateError(f"Smooth search produced no finite candidate for {module_name!r}.")
    return best


def select_best_layer_candidate(
    candidates: Iterable[tuple[_CandidateT, float | torch.Tensor]],
    *,
    module_name: str,
) -> _CandidateT:
    """Select the lowest finite error, retaining the first exact tie."""
    return _select_best_layer_result(candidates, module_name=module_name)[0]


def search_candidates(
    candidates: Iterable[_CandidateT],
    score: Callable[[_CandidateT], float | torch.Tensor],
    *,
    module_name: str,
    on_invalid: Callable[[_CandidateT, Exception], None] | None = None,
    require_finite: bool = True,
    log_search: bool = True,
) -> tuple[_CandidateT, float]:
    """Score candidates incrementally and retain only the first lowest-error candidate.

    Candidate validity and diagnostics can preserve algorithm-specific behavior.
    Execution errors propagate; temporary mutation handling belongs to the strategy.
    """

    def scored_candidates():
        for index, candidate in enumerate(candidates):
            try:
                error = score(candidate)
                error = error.item() if torch.is_tensor(error) else float(error)
                if require_finite and not math.isfinite(error):
                    raise InvalidSmoothCandidateError(f"Nonfinite candidate error: {error}.")
            except InvalidSmoothCandidateError as exc:
                if log_search:
                    logger.debug("Smooth search skipped candidate %d for %s: %s", index, module_name, exc)
                if on_invalid is not None:
                    on_invalid(candidate, exc)
                error = float("inf")
            except Exception:
                if log_search:
                    logger.error("Smooth search failed at candidate %d for %s", index, module_name, exc_info=True)
                raise
            yield candidate, error

    try:
        selected, error = _select_best_layer_result(
            scored_candidates(), module_name=module_name, require_finite=require_finite
        )
    except NoFiniteCandidateError:
        if log_search:
            logger.warning("Smooth search produced no finite candidate for %s", module_name)
        raise
    if log_search:
        logger.debug("Smooth search selected a candidate for %s: error=%.6g", module_name, error)
    return selected, error
