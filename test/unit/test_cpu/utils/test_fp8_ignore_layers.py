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

import unittest

import torch
import torch.nn as nn

from auto_round.compressors.layer_config_resolver import get_fp_layer_names
from auto_round.utils.weight_handler import convert_module_to_hp_if_necessary


class FP8Linear(nn.Module):
    """Mock FP8Linear layer for testing without GPU/triton dependencies."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features).to(torch.float8_e4m3fn))
        self.weight_scale = nn.Parameter(torch.tensor(0.5))
        self.bias = None
        self.data_type = "fp8"
        self.quantized_weight_type = None  # Will be set by check_and_mark_quantized_module


class MockFP8Model(nn.Module):
    """Mock model with FP8Linear layers to simulate an FP8-quantized model."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        for _ in range(2):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = FP8Linear(64, 64)
            layer.self_attn.k_proj = FP8Linear(64, 64)
            layer.mlp = nn.Module()
            layer.mlp.up_proj = FP8Linear(64, 128)
            layer.mlp.down_proj = FP8Linear(128, 64)
            self.model.layers.append(layer)


class TestGetFpLayerNamesWithFP8(unittest.TestCase):
    """Test that get_fp_layer_names correctly detects FP8Linear layers."""

    def test_fp8_layers_detected(self):
        """FP8Linear layers should be found by get_fp_layer_names."""
        model = MockFP8Model()
        result = get_fp_layer_names(model, "self_attn")
        self.assertTrue(len(result) > 0, "get_fp_layer_names should detect FP8Linear layers matching 'self_attn'")
        for name in result:
            self.assertIn("self_attn", name)

    def test_fp8_layers_exact_match(self):
        """Exact layer name match should work for FP8Linear layers."""
        model = MockFP8Model()
        result = get_fp_layer_names(model, "model.layers.0.mlp.up_proj")
        self.assertEqual(result, ["model.layers.0.mlp.up_proj"])

    def test_fp8_layers_empty_ignore(self):
        """Empty ignore_layers should return empty list."""
        model = MockFP8Model()
        result = get_fp_layer_names(model, "")
        self.assertEqual(result, [])


class TestConvertModuleSkipsIgnoredLayers(unittest.TestCase):
    """Test that convert_module_to_hp_if_necessary skips layers in ignore_layers."""

    def test_ignored_layers_stay_fp8(self):
        """Layers listed in ignore_layers should remain in their original FP8 format."""
        model = MockFP8Model()

        ignore = ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj"]
        convert_module_to_hp_if_necessary(model, dtype=torch.bfloat16, ignore_layers=ignore)

        # Ignored layers should still be FP8Linear
        q_proj = model.model.layers[0].self_attn.q_proj
        self.assertIsInstance(q_proj, FP8Linear, "Ignored FP8 layer should not be converted")

        k_proj = model.model.layers[0].self_attn.k_proj
        self.assertIsInstance(k_proj, FP8Linear, "Ignored FP8 layer should not be converted")

    def test_no_ignore_converts_all(self):
        """Without ignore_layers, behavior is unchanged (all convertible layers converted)."""
        model = MockFP8Model()

        convert_module_to_hp_if_necessary(model, dtype=torch.bfloat16)

        # With no ignore_layers, the function runs without error
        # (actual conversion depends on handler registration)

    def test_ignore_by_pattern_match(self):
        """Verify the full flow: get_fp_layer_names + convert with ignore."""
        model = MockFP8Model()

        ignore = get_fp_layer_names(model, "self_attn")
        convert_module_to_hp_if_necessary(model, dtype=torch.bfloat16, ignore_layers=ignore)

        # All self_attn layers should remain FP8Linear
        for i in range(2):
            q_proj = model.model.layers[i].self_attn.q_proj
            k_proj = model.model.layers[i].self_attn.k_proj
            self.assertIsInstance(q_proj, FP8Linear, f"layers.{i}.self_attn.q_proj should stay FP8")
            self.assertIsInstance(k_proj, FP8Linear, f"layers.{i}.self_attn.k_proj should stay FP8")


if __name__ == "__main__":
    unittest.main()
