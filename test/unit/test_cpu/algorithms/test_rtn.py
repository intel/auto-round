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

from auto_round.algorithms.quantization.base import BaseQuantizer
from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig
from auto_round.algorithms.quantization.rtn.quantizer import OptimizedRTNQuantizer
from auto_round.data_type.mxfp import quant_mx, quant_mx_opt_rtn
from auto_round.data_type.utils import get_quant_func


def _make_quantized_linear(name: str):
    layer = torch.nn.Linear(8, 8, bias=False)
    layer.global_name = name
    layer.bits = 4
    layer.act_bits = 4
    layer.data_type = "mx_fp"
    layer.group_size = 32
    layer.sym = True
    layer.scale_dtype = torch.float16
    layer.imatrix = torch.ones(8)
    layer.imatrix_cnt = 1
    return layer


def test_mxfp_opt_rtn_dispatches_to_mx_optimized_function():
    func, data_type = get_quant_func("mx_fp", bits=4, sym=True, disable_opt_rtn=False, group_size=32, iters=0)

    assert data_type == "opt_rtn_mx_fp"
    assert func is quant_mx_opt_rtn


def test_mxfp_plain_rtn_dispatches_to_plain_mx_function():
    func, data_type = get_quant_func("mx_fp", bits=4, sym=True, disable_opt_rtn=True, group_size=32, iters=0)

    assert data_type == "mx_fp4"
    assert func is quant_mx


def test_optimized_rtn_quantizes_each_block_layer_once(monkeypatch):
    block = torch.nn.Sequential(_make_quantized_linear("block.0"), _make_quantized_linear("block.1"))
    calls = []

    def fake_quantize_layer(self, layer, disable_opt_rtn=None):
        calls.append((layer.global_name, disable_opt_rtn, self.config.disable_opt_rtn))

    monkeypatch.setattr(BaseQuantizer, "_quantize_layer_via_rtn", fake_quantize_layer)

    quantizer = OptimizedRTNQuantizer(OptimizedRTNConfig(disable_opt_rtn=False))
    quantizer.quantize_block(block, None, None, None, None, None)

    assert calls == [
        ("block.0", None, False),
        ("block.1", None, False),
    ]
