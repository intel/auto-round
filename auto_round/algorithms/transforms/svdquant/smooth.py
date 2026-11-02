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

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

# Compatibility exports for existing callers.
from auto_round.algorithms.transforms.smoothing.scale import (
    SmoothScaleStats,
    absmax_channel_span,
    summarize_smooth_scale,
    validate_smooth_scale_for_deployment,
)
from auto_round.algorithms.transforms.smoothing.search import select_best_layer_candidate

if TYPE_CHECKING:
    from auto_round.algorithms.transforms.smoothing.calibration import SmoothGroupCalibration
    from auto_round.algorithms.transforms.smoothing.groups import SmoothGroup
    from auto_round.algorithms.transforms.svdquant.apply import SVDQuantTransform

__all__ = [
    "SVDQuantSmoothStrategy",
    "SmoothCandidate",
    "SmoothScaleStats",
    "absmax_channel_span",
    "build_alpha_beta_candidates",
    "build_smooth_scale",
    "select_best_layer_candidate",
    "summarize_smooth_scale",
    "validate_smooth_scale_for_deployment",
]


@dataclass(frozen=True)
class SmoothCandidate:
    alpha: float
    beta: float
    scale: torch.Tensor


def build_alpha_beta_candidates(num_grids: int) -> list[tuple[float, float]]:
    """Build identity, activation-only, and activation/weight balanced candidates."""
    if type(num_grids) is not int or num_grids < 2:
        raise ValueError(f"`num_grids` must be an integer greater than or equal to 2, got {num_grids!r}")
    choices = [index / num_grids for index in range(1, num_grids)]
    return [(0.0, 0.0), *[(alpha, 0.0) for alpha in choices], *[(alpha, 1.0 - alpha) for alpha in choices]]


def build_smooth_scale(
    x_span: torch.Tensor,
    w_span: torch.Tensor,
    alpha: float,
    beta: float,
    eps: float | None = None,
) -> torch.Tensor:
    """Construct the factor ``x_span**alpha / w_span**beta`` with safe fallbacks."""
    if not 0.0 <= alpha <= 1.0 or not 0.0 <= beta <= 1.0:
        raise ValueError(f"Smooth alpha and beta must be in [0, 1], got alpha={alpha!r}, beta={beta!r}")
    if x_span.shape != w_span.shape:
        raise ValueError(f"Smooth spans must have matching shapes, got {x_span.shape} and {w_span.shape}")

    x_span = x_span.to(torch.float32)
    w_span = w_span.to(device=x_span.device, dtype=torch.float32)
    x_zero = x_span == 0
    w_zero = w_span == 0
    if eps is not None:
        if eps <= 0:
            raise ValueError(f"`eps` must be positive, got {eps!r}")
        x_span = torch.where(x_zero, eps, x_span)
        w_span = torch.where(w_zero, eps, w_span)
    if alpha == 0.0 and beta == 0.0:
        return torch.ones_like(x_span)

    if alpha > 0.0:
        scale = x_span.pow(alpha)
        if beta > 0.0:
            scale = scale / w_span.pow(beta)
    else:
        scale = w_span.pow(-beta)

    scale = scale.clone()
    if beta > 0.0 and bool(w_zero.any()):
        scale.fill_(1)
    elif alpha > 0.0:
        scale[x_zero] = 1
    scale[scale == 0] = 1
    if not torch.isfinite(scale).all():
        scale.fill_(1)
    return scale


class SVDQuantSmoothStrategy:
    """Absmax candidates and low-rank-aware reconstruction scoring policy."""

    def __init__(self, owner: SVDQuantTransform, block: torch.nn.Module, local_names: dict[int, str]) -> None:
        self.owner = owner
        self.block = block
        self.local_names = local_names
        self._candidates = None
        self._score = None
        self.calibration = None

    def prepare(self, group: SmoothGroup, calibration: SmoothGroupCalibration) -> None:
        self.calibration = calibration
        owner = self.owner
        block = self.block
        local_names = self.local_names
        group = calibration.group
        device = group.projections[0].weight.device
        x_span = torch.stack([absmax_channel_span(inputs, -1) for inputs in calibration.projection_inputs], dim=0).amax(
            dim=0
        )
        weights = [
            projection.weight.detach().to(device=device, dtype=torch.float32) for projection in group.projections
        ]
        w_span = absmax_channel_span(torch.cat(weights, dim=0), 1).cpu()
        candidates = (
            SmoothCandidate(alpha, beta, build_smooth_scale(x_span, w_span, alpha, beta, eps=owner.config.smooth_eps))
            for alpha, beta in build_alpha_beta_candidates(owner.config.smooth_num_grids)
        )

        def score(candidate):
            scale = validate_smooth_scale_for_deployment(
                candidate.scale, dtype=group.projections[0].weight.dtype, module_name=group.key
            ).to(torch.float32)
            # Keep the deployed precision in the selected candidate.
            candidate.scale.copy_(scale)
            return owner._score_group_wrappers(
                calibration, owner._candidate_group_wrappers(group, scale), block, local_names
            )

        self._candidates = candidates
        self._score = score

    def candidates(self):
        return self._candidates

    def score(self, candidate: SmoothCandidate) -> float:
        return self._score(candidate)

    def apply(self, candidate: SmoothCandidate):
        return self.owner._decompose_smoothed_group(self.calibration, candidate.scale, self.block, self.local_names)

    def clear(self) -> None:
        self._candidates = None
        self._score = None
        self.calibration = None
