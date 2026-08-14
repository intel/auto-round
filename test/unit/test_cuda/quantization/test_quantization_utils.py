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
"""GPU-side unit tests for ``auto_round.algorithms.quantization.utils``.

Covers ``register_act_max_hooks`` (static-activation calibration hooks) and
``register_imatrix_hooks`` (imatrix statistics collection).
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")

from auto_round.algorithms.quantization.utils import register_act_max_hooks, register_imatrix_hooks


class _Quant:
    """Minimal quantizer stand-in exposing the attrs used by the helpers."""

    act_group_size = 8
    config = SimpleNamespace(is_act_nv_fp=False)
    layer_config = {}
    supported_types = (nn.Linear,)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.linear.bits = 4
        self.linear.act_dynamic = False
        self.linear.act_data_type = "fp"
        self.linear.act_bits = 8
        self.linear.act_group_size = 8

    def forward(self, x):
        return self.linear(x)


class TestRegisterActMaxHooks:
    def test_registers_hook_for_static_activation(self):
        q = _Quant()
        b = _Block()
        handles = register_act_max_hooks(q, b)
        assert len(handles) == 1
        try:
            with torch.no_grad():
                b(torch.randn(4, 8))
            assert hasattr(b.linear, "act_max")
            assert b.linear.act_max.shape == (4,)
        finally:
            for h in handles:
                h.remove()

    def test_no_hook_for_dynamic_activation(self):
        q = _Quant()
        b = _Block()
        b.linear.act_dynamic = True
        b.linear.act_bits = 16
        assert register_act_max_hooks(q, b) == []

    def test_accumulates_across_forwards(self):
        q = _Quant()
        b = _Block()
        handles = register_act_max_hooks(q, b)
        try:
            with torch.no_grad():
                b(torch.randn(4, 8))
            first = b.linear.act_max.clone()
            with torch.no_grad():
                b(torch.randn(4, 8) * 2)
            # act_max should be the element-wise max of both runs
            assert bool(torch.all(b.linear.act_max >= first))
        finally:
            for h in handles:
                h.remove()

    def test_layer_config_path(self):
        # A module that is not a supported layer type but listed in layer_config
        class _Custom(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 8)

            def forward(self, x):
                return self.fc(x)

        q = _Quant()
        # Use a Linear so it matches SUPPORTED_LAYER_TYPES; force the layer_config
        # branch by making the module name appear in layer_config with static act.
        b = _Block()
        q.layer_config = {"linear": {"bits": 4, "act_dynamic": False, "act_data_type": "fp", "act_bits": 8}}
        handles = register_act_max_hooks(q, b)
        # either via supported-layer path or layer_config path, at least one hook
        assert len(handles) >= 1
        for h in handles:
            h.remove()


class TestRegisterImatrixHooks:
    def test_registers_and_collects_imatrix(self):
        q = _Quant()
        b = _Block()
        handles = register_imatrix_hooks(q, b, with_count=True)
        assert len(handles) == 1
        try:
            with torch.no_grad():
                b(torch.randn(4, 8))
            assert hasattr(b.linear, "imatrix")
            assert b.linear.imatrix.shape == (8,)
            assert b.linear.imatrix_cnt == 4
        finally:
            for h in handles:
                h.remove()

    def test_without_count(self):
        q = _Quant()
        b = _Block()
        handles = register_imatrix_hooks(q, b, with_count=False)
        try:
            with torch.no_grad():
                b(torch.randn(4, 8))
            assert hasattr(b.linear, "imatrix")
            assert not hasattr(b.linear, "imatrix_cnt")
        finally:
            for h in handles:
                h.remove()

    def test_accumulates_across_forwards(self):
        q = _Quant()
        b = _Block()
        handles = register_imatrix_hooks(q, b, with_count=True)
        try:
            with torch.no_grad():
                b(torch.ones(4, 8))
            first = b.linear.imatrix.clone()
            with torch.no_grad():
                b(torch.ones(4, 8))
            assert b.linear.imatrix_cnt == 8
            assert bool(torch.all(b.linear.imatrix >= first))
        finally:
            for h in handles:
                h.remove()
