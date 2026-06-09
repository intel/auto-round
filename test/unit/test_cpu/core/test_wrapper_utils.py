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

"""Unit tests for auto_round/wrapper.py to improve code coverage."""

import torch
import pytest
from unittest.mock import MagicMock, patch


class TestGetScaleShape:
    """Tests for get_scale_shape function."""

    def test_default_behavior_group_size_positive(self):
        from auto_round.wrapper import get_scale_shape
        weight = torch.randn(128, 64)
        shape = get_scale_shape(weight, group_size=32)
        assert shape == 128 * 2  # 64/32 = 2, so 128*2 = 256

    def test_group_size_zero(self):
        from auto_round.wrapper import get_scale_shape
        weight = torch.randn(128, 64)
        shape = get_scale_shape(weight, group_size=0)
        assert shape == 1

    def test_group_size_negative_one(self):
        from auto_round.wrapper import get_scale_shape
        weight = torch.randn(128, 64)
        shape = get_scale_shape(weight, group_size=-1)
        assert shape == 128  # Returns weight.shape[0]

    def test_group_size_larger_than_dim(self):
        from auto_round.wrapper import get_scale_shape
        weight = torch.randn(128, 64)
        shape = get_scale_shape(weight, group_size=128)
        assert shape == 128  # weight.shape[1] < group_size, returns weight.shape[0]

    def test_tuple_group_size(self):
        from auto_round.wrapper import get_scale_shape
        weight = torch.randn(128, 64)
        shape = get_scale_shape(weight, group_size=(8, 8))
        # (128//8, 64//8) = (16, 8)
        assert shape == (16, 8)

    def test_tuple_group_size_wrong_dim_raises(self):
        from auto_round.wrapper import get_scale_shape
        weight = torch.randn(128, 64)
        with pytest.raises(AssertionError):
            get_scale_shape(weight, group_size=(8,))  # 1D tuple but weight is 2D


class TestWrapperLayerNorm:
    """Tests for WrapperLayerNorm class."""

    def test_creation_and_forward(self):
        from auto_round.wrapper import WrapperLayerNorm
        import torch.nn as nn

        orig_layer = nn.LayerNorm(64)
        wrapper = WrapperLayerNorm(orig_layer, bit=4, group_size=-1, device="cpu")

        assert wrapper.orig_layer is orig_layer
        assert wrapper.bits == 4
        assert wrapper.group_size == -1

        # Test forward pass
        x = torch.randn(2, 10, 64)
        output = wrapper(x)
        assert output.shape == x.shape
        assert not torch.isnan(output).any()


class TestWrapperLlamaNorm:
    """Tests for WrapperLlamaNorm class."""

    def test_creation_and_forward(self):
        from auto_round.wrapper import WrapperLlamaNorm

        try:
            from transformers.models.llama.modeling_llama import LlamaRMSNorm
        except ImportError:
            pytest.skip("LlamaRMSNorm not available")

        orig_layer = LlamaRMSNorm(64)
        wrapper = WrapperLlamaNorm(orig_layer, bit=4, group_size=-1, device="cpu")

        assert wrapper.orig_layer is orig_layer
        assert wrapper.bits == 4
        assert wrapper.group_size == -1

        # Test forward pass
        x = torch.randn(2, 10, 64)
        output = wrapper(x)
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_unwrapper(self):
        from auto_round.wrapper import WrapperLlamaNorm

        try:
            from transformers.models.llama.modeling_llama import LlamaRMSNorm
        except ImportError:
            pytest.skip("LlamaRMSNorm not available")

        orig_layer = LlamaRMSNorm(64)
        wrapper = WrapperLlamaNorm(orig_layer, bit=4, group_size=-1, device="cpu")

        # Test unwrapper with None - returns orig_layer
        result = wrapper.unwrapper(None)
        assert result is orig_layer

    def test_unwrapper_with_best_params(self):
        from auto_round.wrapper import WrapperLlamaNorm

        try:
            from transformers.models.llama.modeling_llama import LlamaRMSNorm
        except ImportError:
            pytest.skip("LlamaRMSNorm not available")

        orig_layer = LlamaRMSNorm(64)
        wrapper = WrapperLlamaNorm(orig_layer, bit=4, group_size=-1, device="cpu")

        # Create mock best_params
        best_params = {"v": torch.zeros_like(wrapper.v)}
        result = wrapper.unwrapper(best_params)
        # Returns orig_layer after quantization
        assert result is orig_layer


class TestWrapperMultiblock:
    """Tests for WrapperMultiblock class."""

    def test_creation_and_forward(self):
        from auto_round.wrapper import WrapperMultiblock

        # Create simple mock layers instead of actual model
        class MockDecoderLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(64, 128)
                self.linear2 = torch.nn.Linear(128, 64)

            def forward(self, x, **kwargs):
                x = self.linear1(x)
                x = torch.nn.functional.relu(x)
                x = self.linear2(x)
                return x

        layer = MockDecoderLayer()
        wrapper = WrapperMultiblock([layer])

        # Test forward pass
        x = torch.randn(1, 4, 64)
        output = wrapper(x)
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_forward_with_kwargs(self):
        from auto_round.wrapper import WrapperMultiblock

        # Create simple mock layers that accept kwargs
        class MockDecoderLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(64, 64)

            def forward(self, x, attention_mask=None, **kwargs):
                return self.linear(x)

        layer = MockDecoderLayer()
        wrapper = WrapperMultiblock([layer])

        # Test forward with attention mask
        x = torch.randn(1, 4, 64)
        attention_mask = torch.ones(1, 4)
        output = wrapper(x, attention_mask=attention_mask)
        assert output.shape == x.shape

    def test_forward_returns_tuple(self):
        from auto_round.wrapper import WrapperMultiblock

        # Test wrapper that returns tuple
        class MockDecoderLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(64, 64)

            def forward(self, x, **kwargs):
                return (self.linear(x),)  # Return tuple

        layer = MockDecoderLayer()
        wrapper = WrapperMultiblock([layer])

        x = torch.randn(1, 4, 64)
        output = wrapper(x)
        assert output.shape == x.shape


class TestWrapperBlock:
    """Tests for wrapper_block function."""

    def test_wrapper_block_with_opt(self):
        from auto_round.wrapper import wrapper_block, WrapperLinear

        try:
            from transformers.models.opt.modeling_opt import OPTDecoderLayer
            from transformers.models.opt.configuration_opt import OPTConfig
            config = OPTConfig(
                d_model=64,
                ffn_dim=128,
                num_layers=1,
                num_attention_heads=2,
            )
            block = OPTDecoderLayer(config)
        except ImportError:
            pytest.skip("OPTDecoderLayer not available")

        quantized, unquantized = wrapper_block(
            block,
            enable_minmax_tuning=True,
            enable_norm_bias_tuning=False,
            device="cpu",
        )

        # Should have quantized some layers
        assert isinstance(quantized, list)
        assert isinstance(unquantized, list)

    def test_wrapper_block_with_enable_norm_bias(self):
        from auto_round.wrapper import wrapper_block, WrapperLinear, NORM_MAPPING

        try:
            from transformers.models.opt.modeling_opt import OPTDecoderLayer
            from transformers.models.opt.configuration_opt import OPTConfig
            config = OPTConfig(
                d_model=64,
                ffn_dim=128,
                num_layers=1,
                num_attention_heads=2,
            )
            block = OPTDecoderLayer(config)
        except ImportError:
            pytest.skip("OPTDecoderLayer not available")

        quantized, unquantized = wrapper_block(
            block,
            enable_minmax_tuning=True,
            enable_norm_bias_tuning=True,
            device="cpu",
        )

        # Should have quantized some layers
        assert isinstance(quantized, list)


class TestWrapperLinearQdqBias:
    """Tests for WrapperLinear._qdq_bias method."""

    def test_qdq_bias_fp16(self):
        from auto_round.wrapper import WrapperLinear

        # Create a real linear layer with all required attributes
        # Enable norm_bias_tuning so bias_quant_func is created
        orig_layer = torch.nn.Linear(128, 64, bias=True)
        orig_layer.bits = 4
        orig_layer.sym = True
        orig_layer.group_size = -1
        orig_layer.scale_dtype = torch.float32
        orig_layer.data_type = "int"
        orig_layer.act_bits = 16  # >= 16 disables act_quant
        orig_layer.act_data_type = "int"
        orig_layer.act_sym = True
        orig_layer.act_dynamic = True
        orig_layer.act_group_size = -1
        orig_layer.iters = 200
        orig_layer.tuning_device = "cpu"

        wrapper = WrapperLinear(orig_layer, device="cpu", enable_norm_bias_tuning=True, disable_opt_rtn=True)

        # Test _qdq_bias with fp16 bias
        bias = torch.randn(64, dtype=torch.float16)
        bias_v = torch.zeros(64, dtype=torch.float32, device="cpu")
        bias_v = torch.nn.Parameter(bias_v, requires_grad=True)

        quantized_bias, scale, zp = wrapper._qdq_bias(bias, bias_v)

        assert quantized_bias.shape == bias.shape


class TestWrapperLinearDeviceTransfer:
    """Tests for WrapperLinear device transfer."""

    def test_wrapper_linear_basic_creation(self):
        from auto_round.wrapper import WrapperLinear

        # Create a real linear layer with all required attributes
        orig_layer = torch.nn.Linear(128, 64, bias=True)
        orig_layer.bits = 16  # >= 16 no quantization
        orig_layer.sym = True
        orig_layer.group_size = -1
        orig_layer.scale_dtype = torch.float32
        orig_layer.data_type = "int"
        orig_layer.act_bits = 16  # >= 16 disables act_quant
        orig_layer.act_data_type = "int"
        orig_layer.act_sym = True
        orig_layer.act_dynamic = True
        orig_layer.act_group_size = -1
        orig_layer.iters = 200
        orig_layer.tuning_device = "cpu"

        wrapper = WrapperLinear(orig_layer, device="cpu", disable_opt_rtn=True)

        # Verify wrapper was created
        assert wrapper.device == "cpu"
        assert wrapper.orig_layer is orig_layer

    def test_wrapper_linear_forward(self):
        from auto_round.wrapper import WrapperLinear

        # Create a real linear layer
        orig_layer = torch.nn.Linear(128, 64, bias=True)
        orig_layer.bits = 16  # >= 16 no quantization
        orig_layer.sym = True
        orig_layer.group_size = -1
        orig_layer.scale_dtype = torch.float32
        orig_layer.data_type = "int"
        orig_layer.act_bits = 16  # >= 16 disables act_quant
        orig_layer.act_data_type = "int"
        orig_layer.act_sym = True
        orig_layer.act_dynamic = True
        orig_layer.act_group_size = -1
        orig_layer.iters = 200
        orig_layer.tuning_device = "cpu"

        wrapper = WrapperLinear(orig_layer, device="cpu", disable_opt_rtn=True)

        # Test forward pass
        x = torch.randn(2, 10, 128)
        output = wrapper(x)
        assert output.shape == (2, 10, 64)
        assert not torch.isnan(output).any()
