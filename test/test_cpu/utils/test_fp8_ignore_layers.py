import unittest

import torch
import torch.nn as nn

from auto_round.compressors.utils import get_fp_layer_names
from auto_round.utils.model import (
    check_and_mark_quantized_module,
    convert_module_to_hp_if_necessary,
)


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
        check_and_mark_quantized_module(model)

        ignore = ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj"]
        convert_module_to_hp_if_necessary(model, dtype=torch.bfloat16, ignore_layers=ignore)

        # Ignored layers should still be FP8Linear
        q_proj = model.model.layers[0].self_attn.q_proj
        self.assertIsInstance(q_proj, FP8Linear, "Ignored FP8 layer should not be converted")

        k_proj = model.model.layers[0].self_attn.k_proj
        self.assertIsInstance(k_proj, FP8Linear, "Ignored FP8 layer should not be converted")

        # Non-ignored layers should be converted to nn.Linear
        up_proj = model.model.layers[0].mlp.up_proj
        self.assertIsInstance(up_proj, nn.Linear, "Non-ignored FP8 layer should be converted to Linear")
        self.assertEqual(up_proj.weight.dtype, torch.bfloat16)

    def test_no_ignore_converts_all(self):
        """Without ignore_layers, all FP8 layers should be converted."""
        model = MockFP8Model()
        check_and_mark_quantized_module(model)

        convert_module_to_hp_if_necessary(model, dtype=torch.bfloat16)

        q_proj = model.model.layers[0].self_attn.q_proj
        self.assertIsInstance(q_proj, nn.Linear, "All FP8 layers should be converted when no ignore_layers")

    def test_ignore_by_pattern_match(self):
        """Verify the full flow: get_fp_layer_names + convert with ignore."""
        model = MockFP8Model()
        check_and_mark_quantized_module(model)

        ignore = get_fp_layer_names(model, "self_attn")
        convert_module_to_hp_if_necessary(model, dtype=torch.bfloat16, ignore_layers=ignore)

        # All self_attn layers should remain FP8Linear
        for i in range(2):
            q_proj = model.model.layers[i].self_attn.q_proj
            k_proj = model.model.layers[i].self_attn.k_proj
            self.assertIsInstance(q_proj, FP8Linear, f"layers.{i}.self_attn.q_proj should stay FP8")
            self.assertIsInstance(k_proj, FP8Linear, f"layers.{i}.self_attn.k_proj should stay FP8")

        # All mlp layers should be converted
        for i in range(2):
            up_proj = model.model.layers[i].mlp.up_proj
            down_proj = model.model.layers[i].mlp.down_proj
            self.assertIsInstance(up_proj, nn.Linear, f"layers.{i}.mlp.up_proj should be converted")
            self.assertIsInstance(down_proj, nn.Linear, f"layers.{i}.mlp.down_proj should be converted")


if __name__ == "__main__":
    unittest.main()
