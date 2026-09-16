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

from unittest.mock import MagicMock, patch

import pytest
import torch


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


def test_wrapper_keeps_canonical_dtype_for_optimized_nvfp4_v2():
    from auto_round.data_type.nvfp import opt_rtn_nvfp4_v2
    from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear
    from auto_round.wrapper import WrapperLinear

    orig_layer = torch.nn.Linear(32, 4, bias=False)
    orig_layer.bits = 4
    orig_layer.sym = True
    orig_layer.group_size = 16
    orig_layer.data_type = "nvfp4_v2"
    orig_layer.act_bits = 4
    orig_layer.act_sym = True
    orig_layer.act_group_size = 16
    orig_layer.act_data_type = "nvfp4_v2"
    orig_layer.act_dynamic = True
    orig_layer.scale_dtype = torch.float32

    wrapper = WrapperLinear(orig_layer, device="cpu", disable_opt_rtn=False, enable_torch_compile=False, iters=0)

    assert wrapper.weight_quant_func is opt_rtn_nvfp4_v2
    assert wrapper.data_type == "nvfp4_v2"
    wrapper.unwrapper({})
    assert orig_layer.data_type == "nvfp4_v2"
    exported_layer = QuantLinear(4, 16, 32, 4, False, data_type=orig_layer.data_type, act_bits=4)
    assert exported_layer.is_nvfp4_e5m3


class TestWrapperLayerNorm:
    """Tests for WrapperLayerNorm class."""

    def test_creation_and_forward(self):
        import torch.nn as nn

        from auto_round.wrapper import WrapperLayerNorm

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
        from auto_round.wrapper import WrapperLinear, wrapper_block

        try:
            from transformers.models.opt.configuration_opt import OPTConfig
            from transformers.models.opt.modeling_opt import OPTDecoderLayer

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
        from auto_round.wrapper import NORM_MAPPING, WrapperLinear, wrapper_block

        try:
            from transformers.models.opt.configuration_opt import OPTConfig
            from transformers.models.opt.modeling_opt import OPTDecoderLayer

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

    def test_unwrapper_keeps_weight_for_activation_only_quantization(self):
        """W16A8 writes activation state only; its weight stays full precision."""
        from auto_round.wrapper import WrapperLinear

        layer = torch.nn.Linear(4, 2, bias=False)
        layer.bits = 16
        layer.sym = True
        layer.group_size = -1
        layer.scale_dtype = torch.float32
        layer.data_type = "int"
        layer.act_bits = 8
        layer.act_data_type = "int"
        layer.act_sym = True
        layer.act_dynamic = True
        layer.act_group_size = -1
        layer.iters = 0
        layer.tuning_device = "cpu"
        expected = layer.weight.detach().clone()

        WrapperLinear(layer, device="cpu", disable_opt_rtn=True).unwrapper({})

        assert torch.equal(layer.weight, expected)
        assert not hasattr(layer, "scale")

    def test_custom_weight_qdq_is_shared_by_forward_and_write_back(self):
        """A per-layer algorithm hook controls both tuning and final materialization."""
        from auto_round.wrapper import WrapperLinear

        orig_layer = torch.nn.Linear(4, 2, bias=False)
        orig_layer.bits = 4
        orig_layer.sym = True
        orig_layer.group_size = 4
        orig_layer.scale_dtype = torch.float32
        orig_layer.data_type = "int"
        orig_layer.act_bits = 16
        orig_layer.act_data_type = "float"
        orig_layer.act_sym = True
        orig_layer.act_dynamic = True
        orig_layer.act_group_size = -1
        orig_layer.tuning_device = "cpu"
        calls = []

        def build_weight_qdq(*, weight, default_qdq, parameters):
            assert weight.shape == (2, 4)
            parameters["offset"] = torch.nn.Parameter(weight.new_tensor(0.1))

            def custom_qdq(weight, *, materialize=False):
                calls.append(materialize)
                return default_qdq(weight + parameters["offset"], materialize=materialize, **parameters)

            return custom_qdq

        wrapper = WrapperLinear(
            orig_layer,
            device="cpu",
            disable_opt_rtn=True,
            iters=0,
            weight_qdq_builder=build_weight_qdq,
        )

        result, _, _ = wrapper._qdq_weight(wrapper.value, wrapper.min_scale, wrapper.max_scale)
        result.sum().backward()
        assert wrapper.params["offset"].grad is not None
        best = torch.tensor(0.5)
        wrapper.params["offset"] = best
        expected = wrapper.weight_qdq(orig_layer.weight, materialize=True).weight.detach().clone()
        wrapper.params["offset"] = torch.tensor(0.9)
        wrapper.unwrapper({"offset": best})

        assert calls == [False, True, True]
        assert torch.equal(orig_layer.weight, expected)

    def test_base_algorithm_uses_datatype_qdq_by_default(self):
        """Algorithms only override QDQ when they explicitly need custom behavior."""
        from auto_round.algorithms.quantization.base import BaseQuantizer
        from auto_round.data_type.base import create_quantizer

        weight = torch.tensor([[1.0, -0.5, 0.25, -0.75]])
        datatype = create_quantizer(dict(data_type="int", bits=4, group_size=4, sym=True), disable_opt_rtn=True)
        datatype.initialize(weight)
        parameters = {"value": torch.tensor(0.0)}
        qdq = BaseQuantizer.build_weight_qdq(object(), weight=weight, default_qdq=datatype.qdq, parameters=parameters)
        torch.testing.assert_close(qdq(weight).weight, datatype.qdq(weight, value=0.0).weight)
        parameters["value"] = torch.tensor(0.75)
        result = qdq(weight, materialize=True)
        expected = datatype.qdq(weight, value=0.75, materialize=True)
        torch.testing.assert_close(result.weight, expected.weight)
        torch.testing.assert_close(result.scale, expected.scale)

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
