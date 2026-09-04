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

import torch
import torch.nn as nn

from auto_round.compressors.config_resolution import ResolvedQuantizationConfig, ResolvedScheme
from auto_round.compressors.layer_config_resolver import apply_plan_to_model
from auto_round.schemes import QuantizationScheme


def test_apply_plan_is_the_explicit_model_write_boundary():
    model = nn.Sequential(nn.Linear(32, 32))
    plan = ResolvedQuantizationConfig(
        scheme=ResolvedScheme.from_scheme(QuantizationScheme()),
        formats=(),
        layer_config={"0": {"bits": 4, "data_type": "int"}},
    )

    apply_plan_to_model(model, plan)

    assert model[0].bits == 4
    assert model[0].data_type == "int"


def test_apply_plan_preserves_non_target_module_own_attributes():
    """A non-quantized module's own attribute that happens to share a scheme-key
    name (e.g. a grouped RMSNorm's ``group_size``) must not be stripped.

    Regression: ``apply_plan_to_model`` used to ``delattr`` every scheme-key-named
    attribute from *every* module, deleting ``Qwen4ExpTextRMSNorm.group_size`` and
    crashing the norm's next forward with ``AttributeError``.
    """

    class GroupedRMSNorm(nn.Module):
        def __init__(self, dim, group_size):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.group_size = group_size  # legitimately owned by the model

    model = nn.Sequential(nn.Linear(32, 32), GroupedRMSNorm(32, group_size=8))
    plan = ResolvedQuantizationConfig(
        scheme=ResolvedScheme.from_scheme(QuantizationScheme()),
        formats=(),
        layer_config={"0": {"bits": 4, "group_size": 128, "data_type": "int"}},
    )

    apply_plan_to_model(model, plan)

    # Quantization target still receives the plan.
    assert model[0].bits == 4
    assert model[0].group_size == 128
    # The norm keeps its own group_size, untouched by the scheme-key reset.
    assert model[1].group_size == 8

