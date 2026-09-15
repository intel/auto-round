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

"""Focused coverage for the datatype public lifecycle."""

import pytest
import torch

from auto_round.data_type.base import (
    ActivationQuantizationSpec,
    WeightExecutionMode,
    WeightQuantizationSpec,
    WeightTuningOptions,
    canonical_data_type,
    create_activation_quantizer,
    create_weight_quantizer,
    state_tunables,
)
from auto_round.data_type.fp8 import quant_fp8_e5m2
from auto_round.data_type.int import quant_tensor_sym
from auto_round.data_type.mxfp import quant_mx


def _weight_quantizer(data_type, bits, group_size, *, sym=True, mode=WeightExecutionMode.TUNED):
    spec = WeightQuantizationSpec(data_type, bits, group_size, sym, torch.float32)
    options = WeightTuningOptions(mode, False, False)
    return create_weight_quantizer(data_type, spec, options), options


def test_integer_lifecycle_matches_primitive_and_materializes_payload():
    weight = torch.tensor([[1.0, -0.75, 0.25, -0.5]], dtype=torch.float32)
    quantizer, options = _weight_quantizer("int", 4, 4)
    state = quantizer.create_state(weight, tuning_options=options)
    result = quantizer.qdq(weight, state, tunables=state_tunables(state), materialize=True)
    expected, scale, zero_point = quant_tensor_sym(weight, bits=4, group_size=4, scale_dtype=torch.float32)

    assert torch.equal(result.weight, expected)
    assert torch.equal(result.scale, scale)
    if isinstance(zero_point, torch.Tensor):
        assert torch.equal(result.zero_point, zero_point)
    else:
        assert result.zero_point == zero_point


def test_fp8_format_selection_does_not_turn_e5m2_into_block_e4m3():
    weight = torch.randn(16, 32)
    quantizer, options = _weight_quantizer("fp8_e5m2", 8, (8, 16))
    state = quantizer.create_state(weight, tuning_options=options)
    result = quantizer.qdq(weight, state, tunables=state_tunables(state), materialize=True)
    expected, _, _ = quant_fp8_e5m2(weight, bits=8, group_size=(8, 16))

    assert torch.equal(result.weight, expected)


def test_generic_mx_name_uses_requested_bit_width():
    weight = torch.randn(8, 32)
    quantizer, options = _weight_quantizer("mx_fp", 6, 32)
    state = quantizer.create_state(weight, tuning_options=options)
    result = quantizer.qdq(weight, state, tunables=state_tunables(state), materialize=True)
    expected, _, _ = quant_mx(weight, bits=6, group_size=32, data_type="mx_fp6")

    assert quantizer.data_type == "mx_fp6"
    assert torch.equal(result.weight, expected)


@pytest.mark.parametrize("dynamic", (True, False))
def test_activation_lifecycle_handles_dynamic_and_calibrated_integer_quantization(dynamic):
    activation = torch.tensor([[-1.0, 0.5, 0.25, 1.0]], dtype=torch.float32)
    spec = ActivationQuantizationSpec("int", 8, 4, True, torch.float32, dynamic)
    quantizer = create_activation_quantizer("int", spec)
    observed = None if dynamic else quantizer.observe(activation, None)

    result = quantizer.qdq(activation, observed_max=observed)

    assert result.shape == activation.shape
    assert torch.isfinite(result).all()


def test_aliases_keep_a_single_canonical_datatype_for_policy():
    assert canonical_data_type("int") == "int_sym"
    assert canonical_data_type("INT4_ASYM") == "int_asym"
