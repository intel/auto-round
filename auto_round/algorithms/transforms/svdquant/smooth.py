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
from dataclasses import dataclass
from typing import Iterable, TypeVar

import torch


@dataclass(frozen=True)
class SmoothCandidate:
    alpha: float
    beta: float
    scale: torch.Tensor


def build_alpha_beta_candidates(num_grids: int) -> list[tuple[float, float]]:
    if type(num_grids) is not int or num_grids < 2:
        raise ValueError(f"`num_grids` must be an integer greater than or equal to 2, got {num_grids!r}")
    choices = [index / num_grids for index in range(1, num_grids)]
    return [(0.0, 0.0), *[(alpha, 0.0) for alpha in choices], *[(alpha, 1.0 - alpha) for alpha in choices]]


def absmax_channel_span(tensor: torch.Tensor, channels_dim: int) -> torch.Tensor:
    if tensor.ndim == 0:
        raise ValueError("Cannot calculate a channel span for a scalar tensor.")
    channels_dim %= tensor.ndim
    moved = tensor.detach().movedim(channels_dim, -1)
    return moved.abs().reshape(-1, moved.shape[-1]).amax(dim=0).to(torch.float32)


def build_smooth_scale(
    x_span: torch.Tensor,
    w_span: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    if not 0.0 <= alpha <= 1.0 or not 0.0 <= beta <= 1.0:
        raise ValueError(f"Smooth alpha and beta must be in [0, 1], got alpha={alpha!r}, beta={beta!r}")
    if x_span.shape != w_span.shape:
        raise ValueError(f"Smooth spans must have matching shapes, got {x_span.shape} and {w_span.shape}")

    x_span = x_span.to(torch.float32)
    w_span = w_span.to(device=x_span.device, dtype=torch.float32)
    if alpha == 0.0 and beta == 0.0:
        return torch.ones_like(x_span)

    if alpha > 0.0:
        scale = x_span.pow(alpha)
        if beta > 0.0:
            scale = scale / w_span.pow(beta)
    else:
        scale = w_span.pow(-beta)

    scale = scale.clone()
    scale[scale == 0] = 1
    if not torch.isfinite(scale).all():
        scale.fill_(1)
    return scale


_CandidateT = TypeVar("_CandidateT")


def select_best_layer_candidate(
    candidates: Iterable[tuple[_CandidateT, float | torch.Tensor]],
    *,
    module_name: str,
) -> _CandidateT:
    best_candidate: _CandidateT | None = None
    best_error = float("inf")
    for candidate, error in candidates:
        error_value = error.item() if torch.is_tensor(error) else float(error)
        if math.isfinite(error_value) and error_value <= best_error:
            best_candidate = candidate
            best_error = error_value
    if best_candidate is None:
        raise ValueError(f"SVDQuant smooth search produced no finite candidate for {module_name!r}.")
    return best_candidate
