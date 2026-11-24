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

from auto_round.compressors.orchestrator import _summarize_quantized_layers
from auto_round.export.formats.autoround.qlinear_fp import QuantLinear


def test_summary_counts_packed_float_layers_and_keeps_bf16_unquantized():
    model = nn.Module()
    quantized_linear = nn.Linear(4, 4)
    model.add_module("quantized_linear", quantized_linear)
    model.add_module("bf16_linear", nn.Linear(4, 4))
    model.add_module("packed_float_linear", QuantLinear(4, 32, 32, 4, False))

    layer_config = {
        "quantized_linear": {"bits": 4},
        "bf16_linear": {"bits": 16},
        "packed_float_linear": {"bits": 4},
    }
    model.to("meta")
    quantized, unquantized = _summarize_quantized_layers(model, layer_config)

    assert quantized == ["quantized_linear", "packed_float_linear"]
    assert unquantized == ["bf16_linear"]
