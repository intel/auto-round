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

"""Tests for algorithms/quantization/utils.py."""

from unittest.mock import patch

import torch
import torch.nn as nn

from auto_round.algorithms.quantization.utils import (
    register_act_max_hooks,
    register_imatrix_hooks,
)


class MockConfig:
    """Mock config with is_act_nv_fp flag."""
    def __init__(self):
        self.is_act_nv_fp = False


class MockQuantizer:
    """Minimal mock quantizer for testing hooks."""

    def __init__(self, act_group_size=128, layer_config=None, supported_types=None):
        self.act_group_size = act_group_size
        self.layer_config = layer_config or {}
        self.supported_types = supported_types or (nn.Linear,)
        self.config = MockConfig()


def _mock_check_to_quantized(config):
    """Always return True for testing."""
    return True


class TestRegisterActMaxHooks:
    """Tests for register_act_max_hooks."""

    def test_single_linear_module_with_act_dynamic(self):
        """Test hook registered on a single linear with act_dynamic=False."""
        model = nn.Linear(256, 128)
        model.act_dynamic = False
        model.act_data_type = "int8"
        model.act_bits = 8

        quantizer = MockQuantizer(act_group_size=128)
        with patch("auto_round.algorithms.quantization.utils.check_to_quantized", _mock_check_to_quantized):
            handles = register_act_max_hooks(quantizer, model)

        assert len(handles) == 1
        x = torch.randn(4, 256)
        model(x)
        assert hasattr(model, "act_max")
        for h in handles:
            h.remove()

    def test_empty_input_tensor(self):
        """Test hook handles empty tensor gracefully."""
        model = nn.Linear(256, 128)
        model.act_dynamic = False
        model.act_data_type = "int8"
        model.act_bits = 8

        quantizer = MockQuantizer(act_group_size=128)
        with patch("auto_round.algorithms.quantization.utils.check_to_quantized", _mock_check_to_quantized):
            handles = register_act_max_hooks(quantizer, model)

        x = torch.randn(0, 256)
        model(x)
        for h in handles:
            h.remove()

    def test_act_max_accumulates(self):
        """Test act_max is updated on subsequent forward passes."""
        model = nn.Linear(256, 128)
        model.act_dynamic = False
        model.act_data_type = "int8"
        model.act_bits = 8

        quantizer = MockQuantizer(act_group_size=128)
        with patch("auto_round.algorithms.quantization.utils.check_to_quantized", _mock_check_to_quantized):
            handles = register_act_max_hooks(quantizer, model)

        x1 = torch.randn(4, 256) * 0.1
        model(x1)
        first_max = model.act_max.clone()

        x2 = torch.randn(4, 256) * 10.0
        model(x2)
        second_max = model.act_max

        assert (second_max >= first_max).all()
        for h in handles:
            h.remove()

    def test_act_dynamic_false_skips(self):
        """Test that module without act_dynamic attribute is skipped."""
        model = nn.Linear(256, 128)

        quantizer = MockQuantizer(act_group_size=128)
        handles = register_act_max_hooks(quantizer, model)

        assert len(handles) == 0

    def test_layer_config_matching(self):
        """Test hook registered via layer_config matching."""
        model = nn.Sequential(nn.Linear(256, 128), nn.Linear(128, 64))

        quantizer = MockQuantizer(
            act_group_size=128,
            layer_config={
                "0": {
                    "bits": 4,
                    "act_dynamic": False,
                    "act_data_type": "int8",
                    "act_bits": 8,
                }
            },
        )
        with patch("auto_round.algorithms.quantization.utils.check_to_quantized", _mock_check_to_quantized):
            handles = register_act_max_hooks(quantizer, model)

        assert len(handles) == 1
        x = torch.randn(4, 256)
        model(x)
        assert hasattr(model[0], "act_max")
        for h in handles:
            h.remove()

    def test_layer_config_bits_gt_8_skipped(self):
        """Test that layer_config entry with bits > 8 is skipped."""
        model = nn.Sequential(nn.Linear(256, 128))

        quantizer = MockQuantizer(
            act_group_size=128,
            layer_config={
                "0": {
                    "bits": 16,  # > 8
                    "act_dynamic": False,
                    "act_data_type": "int8",
                    "act_bits": 8,
                }
            },
        )
        with patch("auto_round.algorithms.quantization.utils.check_to_quantized", _mock_check_to_quantized):
            handles = register_act_max_hooks(quantizer, model)

        assert len(handles) == 0


class TestRegisterImatrixHooks:
    """Tests for register_imatrix_hooks."""

    def test_hooks_registered_on_supported_types(self):
        """Test imatrix hooks registered on Linear modules."""
        model = nn.Sequential(nn.Linear(256, 128), nn.Linear(128, 64))

        quantizer = MockQuantizer(supported_types=(nn.Linear,))
        with patch("auto_round.algorithms.quantization.utils.check_to_quantized", _mock_check_to_quantized):
            handles = register_imatrix_hooks(quantizer, model)

        assert len(handles) == 2

        x = torch.randn(4, 256)
        model(x)

        assert hasattr(model[0], "imatrix")
        assert hasattr(model[1], "imatrix")
        for h in handles:
            h.remove()

    def test_imatrix_accumulates(self):
        """Test imatrix accumulates squared inputs across forward passes."""
        model = nn.Linear(256, 128)
        quantizer = MockQuantizer(supported_types=(nn.Linear,))
        with patch("auto_round.algorithms.quantization.utils.check_to_quantized", _mock_check_to_quantized):
            handles = register_imatrix_hooks(quantizer, model)

        x1 = torch.randn(4, 256).float()
        model(x1)
        first = model.imatrix.clone()

        x2 = torch.randn(4, 256).float()
        model(x2)
        second = model.imatrix

        assert torch.allclose(second, first + (x2.reshape(-1, 256).pow(2).sum(0)), atol=1e-4)
        for h in handles:
            h.remove()

    def test_imatrix_with_count(self):
        """Test imatrix with with_count=True tracks sample count."""
        model = nn.Linear(256, 128)
        quantizer = MockQuantizer(supported_types=(nn.Linear,))
        with patch("auto_round.algorithms.quantization.utils.check_to_quantized", _mock_check_to_quantized):
            handles = register_imatrix_hooks(quantizer, model, with_count=True)

        x1 = torch.randn(4, 256).float()
        model(x1)
        assert model.imatrix_cnt == 4

        x2 = torch.randn(2, 256).float()
        model(x2)
        assert model.imatrix_cnt == 6
        for h in handles:
            h.remove()

    def test_imatrix_empty_input(self):
        """Test imatrix with empty batch (shape[0] == 0)."""
        model = nn.Linear(256, 128)
        quantizer = MockQuantizer(supported_types=(nn.Linear,))
        with patch("auto_round.algorithms.quantization.utils.check_to_quantized", _mock_check_to_quantized):
            handles = register_imatrix_hooks(quantizer, model, with_count=True)

        x = torch.randn(0, 256).float()
        model(x)

        assert model.imatrix_cnt == 0
        for h in handles:
            h.remove()

    def test_imatrix_skips_unsupported_types(self):
        """Test that non-Linear modules are skipped."""
        model = nn.Sequential(nn.Linear(256, 128), nn.ReLU())

        quantizer = MockQuantizer(supported_types=(nn.Linear,))
        with patch("auto_round.algorithms.quantization.utils.check_to_quantized", _mock_check_to_quantized):
            handles = register_imatrix_hooks(quantizer, model)

        assert len(handles) == 1
        for h in handles:
            h.remove()
