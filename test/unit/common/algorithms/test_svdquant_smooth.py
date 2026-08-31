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

import pytest
import torch

from auto_round.algorithms.transforms.svdquant.smooth import (
    absmax_channel_span,
    build_alpha_beta_candidates,
    build_smooth_scale,
    select_best_layer_candidate,
    validate_smooth_scale_for_deployment,
)


def test_alpha_beta_grid_matches_proven_candidate_order():
    assert build_alpha_beta_candidates(4) == [
        (0.0, 0.0),
        (0.25, 0.0),
        (0.5, 0.0),
        (0.75, 0.0),
        (0.25, 0.75),
        (0.5, 0.5),
        (0.75, 0.25),
    ]
    assert len(build_alpha_beta_candidates(20)) == 39


def test_smooth_scale_uses_activation_and_weight_channel_spans():
    activations = torch.tensor([[[1.0, -9.0, 4.0], [16.0, 2.0, -1.0]]])
    weights = torch.tensor([[1.0, 4.0, 16.0], [-0.5, 2.0, 8.0]])
    x_span = absmax_channel_span(activations, -1)
    w_span = absmax_channel_span(weights, 1)

    scale = build_smooth_scale(x_span, w_span, alpha=0.5, beta=0.5)

    torch.testing.assert_close(scale, x_span.sqrt() / w_span.sqrt())


def test_smooth_scale_zero_channels_follow_identity_fallback():
    scale = build_smooth_scale(
        torch.tensor([0.0, 4.0]),
        torch.tensor([0.0, 1.0]),
        alpha=0.5,
        beta=0.5,
        eps=1e-6,
    )

    torch.testing.assert_close(scale, torch.ones(2))


def test_deployment_validation_rejects_bfloat16_reciprocal_overflow():
    with pytest.raises(ValueError, match="deployable.*proj"):
        validate_smooth_scale_for_deployment(
            torch.tensor([1e-40, 1.0]),
            dtype=torch.bfloat16,
            module_name="transformer_blocks.0.proj",
        )


def test_candidate_selection_keeps_later_exact_tie_and_skips_nonfinite():
    candidates = [("nan", float("nan")), ("first", 1.0), ("later", 1.0), ("worse", 2.0)]

    assert select_best_layer_candidate(candidates, module_name="blocks.0.qkv") == "later"
