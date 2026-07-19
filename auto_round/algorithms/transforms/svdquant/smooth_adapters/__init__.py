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

import torch

from auto_round.algorithms.transforms.svdquant.smooth_adapters.base import (
    SmoothSearchGroup,
    TargetPredicate,
    generic_linear_groups,
)
from auto_round.algorithms.transforms.svdquant.smooth_adapters.flux import discover_flux_groups, supports_flux_block


def discover_smooth_search_groups(
    block: torch.nn.Module,
    is_target: TargetPredicate,
) -> list[SmoothSearchGroup]:
    """Discover shared-input projection groups for one quantization block."""

    if supports_flux_block(block):
        return discover_flux_groups(block, is_target)
    return generic_linear_groups(block, is_target)


__all__ = ["SmoothSearchGroup", "discover_smooth_search_groups"]
