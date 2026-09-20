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
    activation_quantizer_for_layer,
    cache_activation_quantizer,
    canonical_data_type,
    create_quantizer,
)
from auto_round.data_type.fp8 import quant_fp8_e5m2
from auto_round.data_type.gguf import _GGUFWeightQuantizer
from auto_round.data_type.int import quant_tensor_sym
from auto_round.data_type.mxfp import quant_mx, quant_mx_rceil


def _weight_config(data_type, bits, group_size, *, sym=True):
    return dict(data_type=data_type, bits=bits, group_size=group_size, sym=sym, scale_dtype=torch.float32)


def test_integer_lifecycle_matches_primitive_and_materializes_payload():
    weight = torch.tensor([[1.0, -0.75, 0.25, -0.5]], dtype=torch.float32)
    quantizer = create_quantizer(_weight_config("int", 4, 4), disable_opt_rtn=True)
    quantizer.initialize(weight)
    actual = quantizer.quantize(weight)
    expected, _, _ = quant_tensor_sym(weight, bits=4, group_size=4, scale_dtype=torch.float32)

    assert torch.equal(actual, expected)


def test_qdq_returns_the_same_result_used_for_write_back():
    """Algorithms can execute QDQ once and hand its materialized result to the datatype."""
    weight = torch.tensor([[1.0, -0.75, 0.25, -0.5]], dtype=torch.float32)
    layer = torch.nn.Linear(4, 1, bias=False)
    quantizer = create_quantizer(_weight_config("int", 4, 4), disable_opt_rtn=True)
    quantizer.initialize(weight)

    result = quantizer.qdq(weight, materialize=True)
    quantizer.apply_result(layer, result)

    assert torch.equal(layer.weight, result.weight)
    assert layer.scale.shape == (1, 1)


def test_write_back_keeps_conv1d_scale_indexed_by_output_channels():
    """Conv1D stores a transposed weight, but its scale stays per output channel."""
    weight = torch.tensor([[1.0, -0.75, 0.25, -0.5], [-1.0, 0.75, -0.25, 0.5]], dtype=torch.float32)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.empty(4, 2))
    quantizer = create_quantizer(_weight_config("int", 4, 4), disable_opt_rtn=True)
    quantizer.initialize(weight)

    quantizer.write_back(layer, weight, transpose=True)

    assert layer.weight.shape == (4, 2)
    assert layer.scale.shape == (2, 1)


def test_fp8_format_selection_does_not_turn_e5m2_into_block_e4m3():
    weight = torch.randn(16, 32)
    quantizer = create_quantizer(_weight_config("fp8_e5m2", 8, (8, 16)), iters=1)
    quantizer.initialize(weight)
    result = quantizer.quantize(weight)
    expected, _, _ = quant_fp8_e5m2(weight, bits=8, group_size=(8, 16))

    assert torch.equal(result, expected)


def test_generic_mx_name_uses_requested_bit_width():
    weight = torch.randn(8, 32)
    quantizer = create_quantizer(_weight_config("mx_fp", 6, 32), iters=1)
    quantizer.initialize(weight)
    result = quantizer.quantize(weight)
    expected, _, _ = quant_mx(weight, bits=6, group_size=32, data_type="mx_fp6")

    assert torch.equal(result, expected)


@pytest.mark.parametrize("dynamic", (True, False))
def test_activation_lifecycle_handles_dynamic_and_calibrated_integer_quantization(dynamic):
    activation = torch.tensor([[-1.0, 0.5, 0.25, 1.0]], dtype=torch.float32)
    layer = type(
        "Layer",
        (),
        dict(act_data_type="int", act_bits=8, act_group_size=4, act_sym=True, act_dynamic=dynamic),
    )()
    quantizer = activation_quantizer_for_layer(layer, scale_dtype=torch.float32)
    observed = None if dynamic else quantizer.observe(activation, None)

    result = quantizer.qdq(activation, observed_max=observed)

    assert result.shape == activation.shape
    assert torch.isfinite(result).all()


def test_nvfp_static_global_scale_accepts_dynamic_calibration():
    """NVFP4 calibrates a static global scale while keeping dynamic QDQ enabled."""
    layer = type(
        "Layer",
        (),
        dict(
            act_data_type="nv_fp4_with_static_gs",
            act_bits=4,
            act_group_size=16,
            act_sym=True,
            act_dynamic=True,
        ),
    )()

    quantizer = activation_quantizer_for_layer(layer, scale_dtype=torch.float32)

    assert quantizer.requires_calibration is True


def test_aliases_keep_a_single_canonical_datatype_for_policy():
    assert canonical_data_type("int") == "int_sym"
    assert canonical_data_type("INT4_ASYM") == "int_asym"


def test_activation_quantizer_uses_the_spec_default_datatype():
    """Partial layer metadata keeps the public ``int_sym`` activation default."""
    layer = type("Layer", (), dict(act_bits=8, act_group_size=4, act_sym=True, act_dynamic=True))()

    assert type(cache_activation_quantizer(layer)).__name__ == "_IntActivationQuantizer"
    assert type(activation_quantizer_for_layer(layer)).__name__ == "_IntActivationQuantizer"


def test_format_aliases_keep_the_requested_quantization_format():
    weight = torch.randn(4, 32)
    fp8 = create_quantizer(_weight_config("fp8-e5m2", 8, -1), iters=1)
    fp8.initialize(weight)
    expected, _, _ = quant_fp8_e5m2(weight, bits=8, group_size=-1)

    gguf = create_quantizer(_weight_config("INT_SYM_DQ", 4, 32), disable_opt_rtn=True)
    legacy_gguf = create_quantizer(_weight_config("rtn_int_sym_dq", 4, 32), disable_opt_rtn=True)

    assert torch.equal(fp8.quantize(weight), expected)
    assert isinstance(gguf._implementation, _GGUFWeightQuantizer)
    assert gguf._implementation.kind == "sym"
    assert isinstance(legacy_gguf._implementation, _GGUFWeightQuantizer)


def test_mx_aliases_and_rceil_keep_the_requested_format():
    weight = torch.randn(4, 32)
    layer = type(
        "Layer", (), dict(act_data_type="mxfp4", act_bits=4, act_group_size=32, act_sym=True, act_dynamic=True)
    )()
    activation = activation_quantizer_for_layer(layer)
    quantizer = create_quantizer(_weight_config("mx_fp4e2m1_rceil", 4, 32), iters=1)
    quantizer.initialize(weight)
    expected, _, _ = quant_mx_rceil(weight, bits=4, group_size=32, data_type="mx_fp4e2m1")

    assert activation.data_type == "mx_fp4"
    assert torch.equal(quantizer.quantize(weight), expected)


def test_awq_clip_ranges_are_flattened_before_integer_qdq():
    """AWQ stores [output, group] clips while INT QDQ uses flattened group rows."""
    weight = torch.randn(4, 8)
    clip = weight.abs().reshape(4, 2, 4).amax(dim=-1)
    quantizer = create_quantizer(
        {
            **_weight_config("int", 4, 4),
            "awq_clip_min": -clip,
            "awq_clip_max": clip,
        },
        iters=1,
    )
    quantizer.initialize(weight)

    result = quantizer.quantize(weight)

    assert result.shape == weight.shape
    assert torch.isfinite(result).all()
