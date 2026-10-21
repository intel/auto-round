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
) -> tuple[_CandidateT, float]:
    """Keep the candidate and its score together, including a valid None candidate."""
    best = None
    for candidate, error in candidates:
        error_value = error.item() if torch.is_tensor(error) else float(error)
        if math.isfinite(error_value) and (best is None or error_value < best[1]):
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
        selected, error = _select_best_layer_result(scored, module_name=module_name)
    except NoFiniteCandidateError:
        logger.warning("Smooth search produced no finite candidate for %s", module_name)
        raise
    logger.debug("Smooth search selected a candidate for %s: error=%.6g", module_name, error)
    return selected, error
