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

"""Check the public datatype API used by wrappers and algorithms."""

from types import SimpleNamespace

import torch

from auto_round.data_type.base import activation_quantizer_for_layer, create_quantizer
from auto_round.data_type.int import quant_tensor_sym


def test_weight_quantization_reuses_initialization_and_matches_int_math():
    weight = torch.tensor([[1.0, -0.75, 0.25, -0.5]], dtype=torch.float32)
    config = dict(data_type="int", bits=4, group_size=4, sym=True, scale_dtype=torch.float32)
    quantizer = create_quantizer(config, disable_opt_rtn=True)
    quantizer.initialize(weight)
    first = quantizer.quantize(weight)
    second = quantizer.quantize(weight)
    expected = quant_tensor_sym(weight, bits=4, group_size=4, scale_dtype=torch.float32)[0]
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    assert torch.equal(first, second)


def test_weight_training_and_writeback_share_parameters():
    layer = torch.nn.Linear(4, 2, bias=False)
    layer.data_type = "int"
    layer.bits = 4
    layer.group_size = 4
    layer.sym = True
    layer.scale_dtype = torch.float32
    quantizer = create_quantizer(layer, iters=10, tune_rounding=True, tune_minmax=True)
    quantizer.initialize(layer.weight.detach())
    output = quantizer.quantize(layer.weight.detach())
    output.square().sum().backward()
    assert all(parameter.grad is not None for parameter in quantizer.parameters.values())
    expected = output.detach().clone()
    with torch.no_grad():
        quantizer.write_back(layer, layer.weight.detach())
    torch.testing.assert_close(layer.weight, expected, rtol=0, atol=0)
    assert layer.scale.shape == (2, 1)
    assert layer.zp is not None


def test_reinitializing_search_quantizer_does_not_reuse_previous_weight_range():
    config = dict(data_type="int", bits=4, group_size=4, sym=False, scale_dtype=torch.float32)
    quantizer = create_quantizer(config, disable_opt_rtn=True)
    first = torch.tensor([[1.0, -0.75, 0.25, -0.5]])
    second = first * 5
    quantizer.initialize(first)
    quantizer.quantize(first)
    quantizer.initialize(second)
    actual = quantizer.quantize(second)
    fresh = create_quantizer(config, disable_opt_rtn=True)
    fresh.initialize(second)
    assert torch.equal(actual, fresh.quantize(second))


def test_activation_adapter_static_and_dynamic_contracts():
    activation = torch.tensor([[-1.0, 0.5, 0.25, 1.0]], dtype=torch.float32)
    for dynamic in (False, True):
        layer = SimpleNamespace(
            act_data_type="int",
            act_bits=8,
            act_group_size=4,
            act_sym=True,
            scale_dtype=torch.float32,
            act_dynamic=dynamic,
        )
        quantizer = activation_quantizer_for_layer(layer)
        observed = quantizer.observe(activation, None)
        qdq = quantizer.qdq(activation, observed_max=observed)
        assert qdq.shape == activation.shape
        assert torch.isfinite(qdq).all()
