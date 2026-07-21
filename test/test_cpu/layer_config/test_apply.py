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

import torch.nn as nn

from auto_round.layer_config import apply_plan_to_model
from auto_round.planning import CompressionPlan, ResolvedScheme
from auto_round.schemes import QuantizationScheme


def test_apply_plan_is_the_explicit_model_write_boundary():
    model = nn.Sequential(nn.Linear(32, 32))
    plan = CompressionPlan(
        scheme=ResolvedScheme.from_scheme(QuantizationScheme()),
        formats=(),
        layer_config={"0": {"bits": 4, "data_type": "int"}},
    )

    apply_plan_to_model(model, plan)

    assert model[0].bits == 4
    assert model[0].data_type == "int"
