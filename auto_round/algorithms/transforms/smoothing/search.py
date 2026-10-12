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


def select_best_layer_candidate(
    candidates: Iterable[tuple[_CandidateT, float | torch.Tensor]],
    *,
    module_name: str,
) -> _CandidateT:
    """Select the lowest finite error, retaining the first exact tie."""
    best_candidate = None
    best_error = float("inf")
    for candidate, error in candidates:
        error_value = error.item() if torch.is_tensor(error) else float(error)
        if math.isfinite(error_value) and error_value < best_error:
            best_candidate = candidate
            best_error = error_value
    if best_candidate is None:
        raise NoFiniteCandidateError(f"Smooth search produced no finite candidate for {module_name!r}.")
    return best_candidate


def search_candidates(
    candidates: Iterable[_CandidateT],
    score: Callable[[_CandidateT], float | torch.Tensor],
    *,
    module_name: str,
    on_invalid: Callable[[_CandidateT, Exception], None] | None = None,
) -> tuple[_CandidateT, float]:
    """Keep first ties and skip only explicitly invalid numerical candidates.

    Execution errors, including OOM, propagate after the scorer restores trial
    mutations. Candidate construction, output selection and loss remain owned
    by the algorithm. No usable trial raises NoFiniteCandidateError.
    """
    scored = []
    for index, candidate in enumerate(candidates):
        try:
            error = score(candidate)
            error = error.item() if torch.is_tensor(error) else float(error)
            if not math.isfinite(error):
                raise InvalidSmoothCandidateError(f"Nonfinite candidate error: {error}.")
        except InvalidSmoothCandidateError as exc:
            logger.debug("Smooth search skipped candidate %d for %s: %s", index, module_name, exc)
            if on_invalid is not None:
                on_invalid(candidate, exc)
            error = float("inf")
        except Exception:
            logger.error("Smooth search failed at candidate %d for %s", index, module_name, exc_info=True)
            raise
        scored.append((candidate, error))
    try:
        selected = select_best_layer_candidate(scored, module_name=module_name)
    except NoFiniteCandidateError:
        logger.warning("Smooth search produced no finite candidate for %s", module_name)
        raise
    error = next(error for candidate, error in reversed(scored) if candidate is selected)
    logger.debug("Smooth search selected a candidate for %s: error=%.6g", module_name, error)
    return selected, error
