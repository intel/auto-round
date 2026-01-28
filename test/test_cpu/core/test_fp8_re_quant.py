import unittest

import torch
import torch.nn as nn
from transformers import AutoConfig

from auto_round.utils.model import convert_fp8_layer_to_linear, dequant_block_fp8_weight


class MockFP8Layer:
    def __init__(self, in_features, out_features, has_block_size=False):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.randn(out_features, in_features).to(torch.float8_e4m3fn)
        self.weight_scale = torch.tensor(0.5)
        self.bias = nn.Parameter(torch.zeros(out_features))
        if has_block_size:
            self.block_size = [1, 1]
            self.weight_scale = torch.full((out_features, in_features), 0.5)
        self.data_type = "fp8"

    def to(self, device):
        self.weight = self.weight.to(device)
        self.weight_scale = self.weight_scale.to(device)
        self.bias.data = self.bias.data.to(device)
        return self


class TestFP8ReQuant(unittest.TestCase):
    def test_per_tensor_dequant(self):
        """Test basic per-tensor dequantization logic."""
        in_features, out_features = 128, 64
        layer = MockFP8Layer(in_features, out_features)

        dq_weight = dequant_block_fp8_weight(layer.weight, layer.weight_scale, block_size=None)
        self.assertEqual(dq_weight.shape, (out_features, in_features))
        self.assertEqual(dq_weight.dtype, torch.bfloat16)

        expected = layer.weight.to(torch.bfloat16) * layer.weight_scale.to(torch.bfloat16)
        torch.testing.assert_close(dq_weight, expected)

    def test_per_channel_dequant(self):
        """Test per-channel (vector) dequantization logic."""
        in_features, out_features = 128, 64
        weight = torch.randn(out_features, in_features).to(torch.float8_e4m3fn)

        # Per-channel scale (out_features)
        scale = torch.randn(out_features)
        dq_weight = dequant_block_fp8_weight(weight, scale, block_size=None)
        expected = weight.to(torch.bfloat16) * scale.view(-1, 1).to(torch.bfloat16)
        torch.testing.assert_close(dq_weight, expected)

        # Per-channel scale (in_features)
        scale = torch.randn(in_features)
        dq_weight = dequant_block_fp8_weight(weight, scale, block_size=None)
        expected = weight.to(torch.bfloat16) * scale.view(1, -1).to(torch.bfloat16)
        torch.testing.assert_close(dq_weight, expected)

    def test_devstral_model_structure(self):
        """
        Specific test for Devstral-2-123B architecture.
        Uses actual configuration metadata from HF.
        """
        print("\nVerifying Devstral-2-123B layer structure conversion...")
        model_id = "mistralai/Devstral-2-123B-Instruct-2512"
        try:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            in_features = config.hidden_size
            out_features = config.intermediate_size
        except Exception as e:
            print(f"Skipping remote config fetch due to: {e}. Using hardcoded fallback.")
            in_features, out_features = 12288, 28672  # Devstral 2 parameters

        class DevstralLayerMock(nn.Module):
            def __init__(self, in_f, out_f):
                super().__init__()
                self.in_features = in_f
                self.out_features = out_f
                self.weight = nn.Parameter(torch.randn(out_f, in_f).to(torch.float8_e4m3fn))
                # Devstral uses per-tensor FP8 (no block_size)
                self.weight_scale = nn.Parameter(torch.tensor(0.001953125))
                self.bias = None
                self.data_type = "fp8"

        mock_layer = DevstralLayerMock(in_features, out_features)

        # This should now pass without AttributeError for 'block_size'
        new_layer = convert_fp8_layer_to_linear(mock_layer, dtype=torch.bfloat16)

        self.assertIsInstance(new_layer, nn.Linear)
        self.assertEqual(new_layer.in_features, in_features)
        self.assertEqual(new_layer.out_features, out_features)
        self.assertEqual(new_layer.weight.dtype, torch.bfloat16)

        expected = mock_layer.weight.to(torch.bfloat16) * mock_layer.weight_scale.to(torch.bfloat16)
        torch.testing.assert_close(new_layer.weight, expected)
        print("Devstral layer conversion successful!")
