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

from auto_round.layer_config import extract_regex_config, has_quantized_layer_outside_blocks, resolve_layer_config
from auto_round.planning import ResolvedScheme
from auto_round.schemes import QuantizationScheme


def test_resolver_does_not_write_quantization_attributes_to_modules():
    model = nn.Sequential(nn.Linear(32, 32))
    scheme = ResolvedScheme.from_scheme(QuantizationScheme(act_bits=16, act_data_type="float"))

    resolved = resolve_layer_config(
        model=model,
        scheme=scheme,
        layer_config={"0": {}},
        supported_types=(nn.Linear,),
        inner_supported_types=(),
    )

    assert "0" in resolved
    assert not hasattr(model[0], "bits")


def test_extract_regex_config_keeps_pattern_separate_from_expanded_layers():
    model = nn.Sequential(nn.Linear(32, 32))
    scheme = ResolvedScheme.from_scheme(QuantizationScheme(act_bits=16, act_data_type="float"))

    regex_config = extract_regex_config(
        model=model,
        scheme=scheme,
        layer_config={"0": {"bits": 4}, "missing.*": {"bits": 16}},
        supported_types=(nn.Linear,),
        inner_supported_types=(),
    )

    assert set(regex_config) == {"missing.*"}
    assert regex_config["missing.*"]["bits"] == 16


def test_has_quantized_layer_outside_blocks_is_derived_from_final_mapping():
    assert has_quantized_layer_outside_blocks({"layer": {"bits": 4, "in_blocks": False}})
    assert not has_quantized_layer_outside_blocks({"layer": {"bits": 16, "in_blocks": False}})
