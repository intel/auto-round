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

import torch

from auto_round.algorithms.transforms.smoothing.errors import InvalidSmoothCandidateError


@dataclass(frozen=True)
class SmoothScaleStats:
    minimum: float
    maximum: float
    ratio: float
    below_min_count: int
    above_max_count: int


def absmax_channel_span(tensor: torch.Tensor, channels_dim: int) -> torch.Tensor:
    """Return per-channel absolute maxima over every non-channel dimension."""
    if tensor.ndim == 0:
        raise ValueError("Cannot calculate a channel span for a scalar tensor.")
    channels_dim %= tensor.ndim
    moved = tensor.detach().movedim(channels_dim, -1)
    return moved.abs().reshape(-1, moved.shape[-1]).amax(dim=0).to(torch.float32)


def validate_smooth_scale_for_deployment(
    scale: torch.Tensor,
    *,
    dtype: torch.dtype,
    module_name: str,
) -> torch.Tensor:
    """Validate a smooth scale after materialization in its deployment dtype."""
    deployed = scale.to(dtype=dtype)
    reciprocal = deployed.reciprocal()
    if (
        not bool(torch.isfinite(deployed).all())
        or not bool((deployed > 0).all())
        or not bool(torch.isfinite(reciprocal).all())
        or not bool((reciprocal > 0).all())
    ):
        raise InvalidSmoothCandidateError(f"Smooth scale is not deployable for {module_name!r} in dtype {dtype}.")
    return deployed


def summarize_smooth_scale(
    scale: torch.Tensor,
    *,
    low_threshold: float = 1e-3,
    high_threshold: float = 20.0,
) -> SmoothScaleStats:
    """Summarize factor range and deployment-risk threshold counts."""
    values = scale.detach().to(device="cpu", dtype=torch.float32)
    minimum = values.amin().item()
    maximum = values.amax().item()
    return SmoothScaleStats(
        minimum=minimum,
        maximum=maximum,
        ratio=maximum / minimum,
        below_min_count=int((values < low_threshold).sum().item()),
        above_max_count=int((values > high_threshold).sum().item()),
    )
