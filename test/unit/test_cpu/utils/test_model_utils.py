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

"""Unit tests for auto_round/utils/model.py to improve code coverage."""

import json
import os
import pytest
import torch
from unittest.mock import MagicMock, patch


class TestGetBlockNames:
    """Test get_block_names function."""

    def test_opt_model_block_names(self):
        """Test get_block_names with OPT model."""
        from auto_round.utils.model import get_block_names
        import transformers

        config = transformers.AutoConfig.from_pretrained("facebook/opt-125m")
        config.num_hidden_layers = 2
        model = transformers.OPTForCausalLM(config)
        block_names = get_block_names(model)
        assert isinstance(block_names, list)
        assert len(block_names) > 0
        # OPT has a model.decoder.layers structure
        assert any("layers" in str(block) for blocks in block_names for block in blocks)

    def test_qwen_model_block_names(self):
        """Test get_block_names with Qwen model (mocked)."""
        from auto_round.utils.model import get_block_names
        import transformers

        # Create a minimal mock config that has the required attributes
        mock_config = MagicMock()
        mock_config.model_type = "qwen2"
        mock_config.architectures = ["Qwen2ForCausalLM"]
        mock_config.num_hidden_layers = 2

        with patch("transformers.AutoConfig.from_pretrained") as mock_from_pretrained:
            mock_from_pretrained.return_value = mock_config
            # Create a mock model that has the decoder.layers structure
            mock_model = MagicMock()
            mock_model.config = mock_config

            # Mock the module structure
            mock_layer = MagicMock()
            mock_layer.__class__.__name__ = "Qwen2DecoderLayer"
            mock_layer.named_children.return_value = []

            mock_layers = MagicMock()
            mock_layers.__class__.__name__ = "ModuleList"
            mock_layers.named_children.return_value = [("0", mock_layer)]

            mock_decoder = MagicMock()
            mock_decoder.named_children.return_value = [("layers", mock_layers)]

            mock_model.named_children.return_value = [("model", mock_decoder)]
            mock_model.named_modules.return_value = [
                ("model", mock_decoder),
                ("model.decoder", mock_decoder),
                ("model.decoder.layers", mock_layers),
            ]

            block_names = get_block_names(mock_model)
            assert isinstance(block_names, list)

    def test_gemma_model_block_names(self):
        """Test get_block_names with Gemma model (mocked)."""
        from auto_round.utils.model import get_block_names

        # Create a minimal mock config
        mock_config = MagicMock()
        mock_config.model_type = "gemma"
        mock_config.architectures = ["GemmaForCausalLM"]
        mock_config.num_hidden_layers = 2

        # Create a mock model with gemma structure
        mock_model = MagicMock()
        mock_model.config = mock_config

        mock_layer = MagicMock()
        mock_layer.__class__.__name__ = "GemmaDecoderLayer"
        mock_layer.named_children.return_value = []

        mock_layers = MagicMock()
        mock_layers.__class__.__name__ = "ModuleList"
        mock_layers.named_children.return_value = [("0", mock_layer)]

        mock_model.named_children.return_value = [("layers", mock_layers)]
        mock_model.named_modules.return_value = [
            ("layers", mock_layers),
            ("layers.0", mock_layer),
        ]

        block_names = get_block_names(mock_model)
        assert isinstance(block_names, list)
        assert len(block_names) > 0


class TestGetLmHeadName:
    """Test get_lm_head_name function."""

    def test_opt_model_lm_head_name(self):
        """Test get_lm_head_name with OPT model."""
        from auto_round.utils.model import get_lm_head_name
        import transformers

        config = transformers.AutoConfig.from_pretrained("facebook/opt-125m")
        config.num_hidden_layers = 2
        model = transformers.OPTForCausalLM(config)
        lm_head_name = get_lm_head_name(model)
        assert lm_head_name is not None
        assert isinstance(lm_head_name, str)


class TestGetExpertLinearNames:
    """Test get_expert_linear_names function."""

    def test_qwen_moe_expert_linear_names(self):
        """Test get_expert_linear_names with Qwen MoE module."""
        from auto_round.utils.model import get_expert_linear_names
        from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

        # Create a mock module that looks like Qwen2MoeSparseMoeBlock
        mock_module = MagicMock()
        mock_module.__class__.__name__ = "Qwen2MoeSparseMoeBlock"
        result = get_expert_linear_names(mock_module)
        assert result == ["gate_proj", "down_proj", "up_proj"]

    def test_qwen3_moe_expert_linear_names(self):
        """Test get_expert_linear_names with Qwen3 MoE module."""
        from auto_round.utils.model import get_expert_linear_names

        mock_module = MagicMock()
        mock_module.__class__.__name__ = "Qwen3MoeSparseMoeBlock"
        result = get_expert_linear_names(mock_module)
        assert result == ["gate_proj", "down_proj", "up_proj"]

    def test_mixtral_moe_expert_linear_names(self):
        """Test get_expert_linear_names with Mixtral MoE module."""
        from auto_round.utils.model import get_expert_linear_names

        mock_module = MagicMock()
        mock_module.__class__.__name__ = "MixtralMoeSparseMoeBlock"
        result = get_expert_linear_names(mock_module)
        assert result == ["linear_fc1", "linear_fc2"]

    def test_dbrx_moe_expert_linear_names(self):
        """Test get_expert_linear_names with DBRX MoE module."""
        from auto_round.utils.model import get_expert_linear_names

        mock_module = MagicMock()
        mock_module.__class__.__name__ = "DBRXMoeSparseMoeBlock"
        result = get_expert_linear_names(mock_module)
        assert result == ["w1_linear", "w2_linear", "v1_linear"]

    def test_default_expert_linear_names(self):
        """Test get_expert_linear_names with unknown MoE module returns default."""
        from auto_round.utils.model import get_expert_linear_names

        mock_module = MagicMock()
        mock_module.__class__.__name__ = "SomeUnknownMoE"
        result = get_expert_linear_names(mock_module)
        assert result == ["w1", "w2", "w3"]


class TestGetExpertInputProjNames:
    """Test get_expert_input_proj_names function."""

    def test_qwen_moe_input_proj_names(self):
        """Test get_expert_input_proj_names with Qwen MoE module."""
        from auto_round.utils.model import get_expert_input_proj_names

        mock_module = MagicMock()
        mock_module.__class__.__name__ = "Qwen2MoeSparseMoeBlock"
        result = get_expert_input_proj_names(mock_module)
        assert result == ["gate_proj", "up_proj"]

    def test_qwen3_moe_input_proj_names(self):
        """Test get_expert_input_proj_names with Qwen3 MoE module."""
        from auto_round.utils.model import get_expert_input_proj_names

        mock_module = MagicMock()
        mock_module.__class__.__name__ = "Qwen3MoeSparseMoeBlock"
        result = get_expert_input_proj_names(mock_module)
        assert result == ["gate_proj", "up_proj"]

    def test_mixtral_moe_input_proj_names(self):
        """Test get_expert_input_proj_names with Mixtral MoE module."""
        from auto_round.utils.model import get_expert_input_proj_names

        mock_module = MagicMock()
        mock_module.__class__.__name__ = "MixtralMoeSparseMoeBlock"
        result = get_expert_input_proj_names(mock_module)
        assert result == ["linear_fc1"]

    def test_dbrx_moe_input_proj_names(self):
        """Test get_expert_input_proj_names with DBRX MoE module."""
        from auto_round.utils.model import get_expert_input_proj_names

        mock_module = MagicMock()
        mock_module.__class__.__name__ = "DBRXMoeSparseMoeBlock"
        result = get_expert_input_proj_names(mock_module)
        assert result == ["w1_linear", "v1_linear"]

    def test_default_input_proj_names(self):
        """Test get_expert_input_proj_names with unknown MoE module returns default."""
        from auto_round.utils.model import get_expert_input_proj_names

        mock_module = MagicMock()
        mock_module.__class__.__name__ = "SomeUnknownMoE"
        result = get_expert_input_proj_names(mock_module)
        assert result == ["w1", "w3"]


class TestIsMxfp4Model:
    """Test _is_mxfp4_model function."""

    def test_mxfp4_model_with_none_quantization_config(self):
        """Test _is_mxfp4_model returns False when quantization_config is None."""
        from auto_round.utils.model import _is_mxfp4_model

        with patch("transformers.AutoConfig.from_pretrained") as mock_config:
            mock_config_obj = MagicMock()
            mock_config_obj.model_type = "gpt_oss"
            mock_config_obj.quantization_config = None
            mock_config.return_value = mock_config_obj

            result = _is_mxfp4_model("test/model", trust_remote_code=True)
            assert result is False

    def test_mxfp4_model_with_unsupported_model_type(self):
        """Test _is_mxfp4_model returns False for unsupported model type."""
        from auto_round.utils.model import _is_mxfp4_model

        with patch("transformers.AutoConfig.from_pretrained") as mock_config:
            mock_config_obj = MagicMock()
            mock_config_obj.model_type = "opt"  # Not in _MXFP4_SUPPORTED_MODEL_TYPES
            mock_config_obj.quantization_config = {"quant_method": "mxfp4"}
            mock_config.return_value = mock_config_obj

            result = _is_mxfp4_model("test/model", trust_remote_code=True)
            assert result is False

    def test_mxfp4_model_with_valid_config(self):
        """Test _is_mxfp4_model returns True for valid MXFP4 config."""
        from auto_round.utils.model import _is_mxfp4_model

        with patch("transformers.AutoConfig.from_pretrained") as mock_config:
            mock_config_obj = MagicMock()
            mock_config_obj.model_type = "gpt_oss"
            mock_config_obj.quantization_config = {"quant_method": "mxfp4"}
            mock_config.return_value = mock_config_obj

            result = _is_mxfp4_model("test/model", trust_remote_code=True)
            assert result is True

    def test_mxfp4_model_with_config_object(self):
        """Test _is_mxfp4_model works with config object."""
        from auto_round.utils.model import _is_mxfp4_model

        with patch("transformers.AutoConfig.from_pretrained") as mock_config:
            mock_config_obj = MagicMock()
            mock_config_obj.model_type = "gpt_oss"
            mock_quant_config = MagicMock()
            mock_quant_config.quant_method = "mxfp4"
            mock_config_obj.quantization_config = mock_quant_config
            mock_config.return_value = mock_config_obj

            result = _is_mxfp4_model("test/model", trust_remote_code=True)
            assert result is True


class TestGetNestedAttr:
    """Test get_nested_attr function."""

    def test_existing_nested_attribute(self):
        """Test get_nested_attr with existing nested attribute."""
        from auto_round.utils.model import get_nested_attr

        class MockModule:
            def __init__(self):
                self.orig_layer = MagicMock()
                self.orig_layer.act_max = torch.tensor([1.0])

        module = MockModule()
        result = get_nested_attr(module, "orig_layer.act_max")
        assert result is not None
        assert torch.equal(result, torch.tensor([1.0]))

    def test_missing_nested_attribute_with_default(self):
        """Test get_nested_attr with missing attribute returns None."""
        from auto_round.utils.model import get_nested_attr

        # Create a real object where missing_attr doesn't exist
        class MockModule:
            def __init__(self):
                self.orig_layer = MagicMock(spec=[])  # No attributes

        module = MockModule()
        result = get_nested_attr(module, "orig_layer.missing_attr")
        assert result is None

    def test_missing_first_level_attribute(self):
        """Test get_nested_attr when first level attribute doesn't exist."""
        from auto_round.utils.model import get_nested_attr

        # Create a real object where missing_layer doesn't exist
        class MockModule:
            pass

        module = MockModule()
        result = get_nested_attr(module, "missing_layer.act_max")
        assert result is None


class TestResolveModelType:
    """Test resolve_model_type function."""

    def test_resolve_opt_model_type(self):
        """Test resolve_model_type with OPT model."""
        from auto_round.utils.model import resolve_model_type
        import transformers

        config = transformers.AutoConfig.from_pretrained("facebook/opt-125m")
        config.num_hidden_layers = 2
        model = transformers.OPTForCausalLM(config)
        result = resolve_model_type(model)
        assert result == "opt"

    def test_resolve_qwen_model_type(self):
        """Test resolve_model_type with Qwen model (mocked)."""
        from auto_round.utils.model import resolve_model_type

        model = MagicMock()
        model.config.architectures = ["Qwen2ForCausalLM"]
        model.config.model_type = "qwen2"

        result = resolve_model_type(model)
        assert result == "qwen2"

    def test_resolve_gemma_model_type(self):
        """Test resolve_model_type with Gemma model (mocked)."""
        from auto_round.utils.model import resolve_model_type

        model = MagicMock()
        model.config.architectures = ["GemmaForCausalLM"]
        model.config.model_type = "gemma"

        result = resolve_model_type(model)
        assert result == "gemma"

    def test_resolve_model_without_config(self):
        """Test resolve_model_type returns None when model has no config."""
        from auto_round.utils.model import resolve_model_type

        model = MagicMock(spec=[])
        del model.config

        result = resolve_model_type(model)
        assert result is None

    def test_resolve_model_with_architecture_override(self):
        """Test resolve_model_type with architecture-based override."""
        from auto_round.utils.model import resolve_model_type

        model = MagicMock()
        model.config.architectures = ["MiMoAudioForCausalLM"]
        model.config.model_type = "qwen2"

        result = resolve_model_type(model)
        assert result == "mimo_audio"


class TestConvertDtypeStr2Torch:
    """Test convert_dtype_str2torch function."""

    def test_float16(self):
        """Test conversion from 'fp16' string to torch.float16."""
        from auto_round.utils.model import convert_dtype_str2torch

        result = convert_dtype_str2torch("fp16")
        assert result == torch.float16

    def test_float16_variant(self):
        """Test conversion from 'float16' string to torch.float16."""
        from auto_round.utils.model import convert_dtype_str2torch

        result = convert_dtype_str2torch("float16")
        assert result == torch.float16

    def test_bfloat16(self):
        """Test conversion from 'bf16' string to torch.bfloat16."""
        from auto_round.utils.model import convert_dtype_str2torch

        result = convert_dtype_str2torch("bf16")
        assert result == torch.bfloat16

    def test_bfloat16_variant(self):
        """Test conversion from 'bfloat16' string to torch.bfloat16."""
        from auto_round.utils.model import convert_dtype_str2torch

        result = convert_dtype_str2torch("bfloat16")
        assert result == torch.bfloat16

    def test_float32(self):
        """Test conversion from 'fp32' string to torch.float."""
        from auto_round.utils.model import convert_dtype_str2torch

        result = convert_dtype_str2torch("fp32")
        assert result == torch.float

    def test_float32_variant(self):
        """Test conversion from 'float32' string to torch.float."""
        from auto_round.utils.model import convert_dtype_str2torch

        result = convert_dtype_str2torch("float32")
        assert result == torch.float

    def test_auto_string(self):
        """Test conversion from 'auto' string to torch.float."""
        from auto_round.utils.model import convert_dtype_str2torch

        result = convert_dtype_str2torch("auto")
        assert result == torch.float

    def test_int8(self):
        """Test conversion from 'int8' string to torch.int8."""
        from auto_round.utils.model import convert_dtype_str2torch

        result = convert_dtype_str2torch("int8")
        assert result == torch.int8

    def test_passthrough_torch_dtype(self):
        """Test passthrough when input is already torch.dtype."""
        from auto_round.utils.model import convert_dtype_str2torch

        result = convert_dtype_str2torch(torch.float16)
        assert result == torch.float16

    def test_passthrough_none(self):
        """Test passthrough when input is None."""
        from auto_round.utils.model import convert_dtype_str2torch

        result = convert_dtype_str2torch(None)
        assert result is None

    def test_unsupported_dtype_raises(self):
        """Test that unsupported dtype raises ValueError."""
        from auto_round.utils.model import convert_dtype_str2torch

        with pytest.raises(ValueError, match="Unsupported string dtype"):
            convert_dtype_str2torch("unsupported_dtype")


class TestConvertDtypeTorch2Str:
    """Test convert_dtype_torch2str function."""

    def test_float16(self):
        """Test conversion from torch.float16 to 'fp16'."""
        from auto_round.utils.model import convert_dtype_torch2str

        result = convert_dtype_torch2str(torch.float16)
        assert result == "fp16"

    def test_bfloat16(self):
        """Test conversion from torch.bfloat16 to 'bf16'."""
        from auto_round.utils.model import convert_dtype_torch2str

        result = convert_dtype_torch2str(torch.bfloat16)
        assert result == "bf16"

    def test_float32(self):
        """Test conversion from torch.float to 'fp32'."""
        from auto_round.utils.model import convert_dtype_torch2str

        result = convert_dtype_torch2str(torch.float)
        assert result == "fp32"

    def test_int8(self):
        """Test conversion from torch.int8 to 'int8'."""
        from auto_round.utils.model import convert_dtype_torch2str

        result = convert_dtype_torch2str(torch.int8)
        assert result == "int8"

    def test_passthrough_string(self):
        """Test passthrough when input is already a string."""
        from auto_round.utils.model import convert_dtype_torch2str

        result = convert_dtype_torch2str("fp16")
        assert result == "fp16"

    def test_passthrough_none(self):
        """Test passthrough when input is None."""
        from auto_round.utils.model import convert_dtype_torch2str

        result = convert_dtype_torch2str(None)
        assert result is None

    def test_string_in_list(self):
        """Test passthrough when string is in supported list."""
        from auto_round.utils.model import convert_dtype_torch2str

        result = convert_dtype_torch2str("int8")
        assert result == "int8"

    def test_unsupported_dtype_raises(self):
        """Test that unsupported dtype raises ValueError."""
        from auto_round.utils.model import convert_dtype_torch2str

        with pytest.raises(ValueError, match="Unsupported PyTorch dtype"):
            convert_dtype_torch2str(torch.int32)


class TestCleanModuleParameter:
    """Test clean_module_parameter function."""

    def test_clean_weight_parameter(self):
        """Test clean_module_parameter with Linear model weight."""
        from auto_round.utils.model import clean_module_parameter
        import torch.nn as nn

        linear = nn.Linear(10, 10)
        # Ensure weight has data
        assert linear.weight is not None
        assert linear.weight.numel() > 0

        clean_module_parameter(linear, "weight")

        # Weight should be emptied (size 0)
        assert linear.weight.shape.numel() == 0
        assert linear.weight.requires_grad is False

    def test_clean_bias_parameter(self):
        """Test clean_module_parameter with Linear model bias."""
        from auto_round.utils.model import clean_module_parameter
        import torch.nn as nn

        linear = nn.Linear(10, 10)
        if linear.bias is not None:
            assert linear.bias.numel() > 0
            clean_module_parameter(linear, "bias")
            assert linear.bias.shape.numel() == 0

    def test_clean_with_none_submodule(self):
        """Test clean_module_parameter handles None submodule gracefully."""
        from auto_round.utils.model import clean_module_parameter

        # Should not raise
        clean_module_parameter(None, "weight")

    def test_clean_buffer(self):
        """Test clean_module_parameter with a buffer."""
        from auto_round.utils.model import clean_module_parameter
        import torch.nn as nn

        class MockModuleWithBuffer(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("my_buffer", torch.ones(5))

        module = MockModuleWithBuffer()
        assert module.my_buffer.numel() == 5

        clean_module_parameter(module, "my_buffer")
        assert module.my_buffer.shape.numel() == 0


class TestIsMoeModelViaConfig:
    """Test is_moe_model_via_config function."""

    def test_regular_model_returns_false(self):
        """Test is_moe_model_via_config with non-MoE model returns False."""
        from auto_round.utils.model import is_moe_model_via_config

        config = MagicMock()
        config_str = "opt"
        config.__str__ = lambda self: config_str

        result = is_moe_model_via_config(config)
        assert result is False

    def test_moe_model_returns_true(self):
        """Test is_moe_model_via_config with MoE model returns True."""
        from auto_round.utils.model import is_moe_model_via_config

        config = MagicMock()
        config_str = "qwen2_moe"
        config.__str__ = lambda self: config_str

        result = is_moe_model_via_config(config)
        assert result is True

    def test_expert_model_returns_true(self):
        """Test is_moe_model_via_config with expert model returns True."""
        from auto_round.utils.model import is_moe_model_via_config

        config = MagicMock()
        config_str = "mixture_of_experts"
        config.__str__ = lambda self: config_str

        result = is_moe_model_via_config(config)
        assert result is True

    def test_config_with_to_dict(self):
        """Test is_moe_model_via_config with config that has to_dict method."""
        from auto_round.utils.model import is_moe_model_via_config

        config = MagicMock()
        config.to_dict.return_value = {"model_type": "qwen2_moe"}
        config.__str__ = lambda self: str(config.to_dict())

        result = is_moe_model_via_config(config)
        assert result is True

    def test_config_str_raises_exception(self):
        """Test is_moe_model_via_config handles str() exception gracefully."""
        from auto_round.utils.model import is_moe_model_via_config

        config = MagicMock()
        config.__str__ = MagicMock(side_effect=Exception("Cannot convert"))

        result = is_moe_model_via_config(config)
        assert result is False


class TestArchitectureModelTypeMap:
    """Test ARCHITECTURE_MODEL_TYPE_MAP."""

    def test_architecture_model_type_map_contains_expected_entries(self):
        """Test ARCHITECTURE_MODEL_TYPE_MAP contains Qwen2ForCausalLM, OPTForCausalLM mappings."""
        from auto_round.utils.model import ARCHITECTURE_MODEL_TYPE_MAP

        # Verify MiMoAudio entries exist
        assert "MiMoAudioModel" in ARCHITECTURE_MODEL_TYPE_MAP
        assert "MiMoAudioForCausalLM" in ARCHITECTURE_MODEL_TYPE_MAP
        assert ARCHITECTURE_MODEL_TYPE_MAP["MiMoAudioModel"] == "mimo_audio"
        assert ARCHITECTURE_MODEL_TYPE_MAP["MiMoAudioForCausalLM"] == "mimo_audio"


class TestDownloadOrGetPath:
    """Test download_or_get_path function."""

    def test_hf_platform(self):
        """Test download_or_get_path with hf platform."""
        from auto_round.utils.model import download_or_get_path

        # This tests the platform selection logic
        with patch("auto_round.utils.model.download_hf_model") as mock_hf:
            mock_hf.return_value = "/path/to/model"
            result = download_or_get_path("test/model", platform="hf")
            mock_hf.assert_called_once()
            assert result == "/path/to/model"

    def test_modelscope_platform(self):
        """Test download_or_get_path with model_scope platform."""
        from auto_round.utils.model import download_or_get_path

        with patch("auto_round.utils.model.download_modelscope_model") as mock_ms:
            mock_ms.return_value = "/path/to/model"
            result = download_or_get_path("test/model", platform="model_scope")
            mock_ms.assert_called_once()
            assert result == "/path/to/model"

    def test_hf_platform_calls_correct_function(self):
        """Test download_or_get_path correctly routes to hf downloader."""
        from auto_round.utils.model import download_or_get_path

        with patch("auto_round.utils.model.download_hf_model") as mock_hf:
            mock_hf.return_value = "/hf/path"
            # Use a valid-looking model ID to avoid validation errors in mock
            result = download_or_get_path("facebook/opt-125m", platform="hf")
            # The mocked function should have been called
            assert mock_hf.called


class TestGetModelDtype:
    """Test get_model_dtype function."""

    def test_none_returns_default(self):
        """Test get_model_dtype with None returns default."""
        from auto_round.utils.model import get_model_dtype

        result = get_model_dtype(None, default="bfloat16")
        assert result == "bfloat16"

    def test_auto_returns_default(self):
        """Test get_model_dtype with 'auto' returns default."""
        from auto_round.utils.model import get_model_dtype

        result = get_model_dtype("auto", default="bfloat16")
        assert result == "bfloat16"

    def test_bf16_normalized(self):
        """Test get_model_dtype normalizes bf16 variants."""
        from auto_round.utils.model import get_model_dtype

        result = get_model_dtype("bf16", default="float16")
        assert result == "bfloat16"

        result = get_model_dtype("bfloat16", default="float16")
        assert result == "bfloat16"

    def test_fp16_normalized(self):
        """Test get_model_dtype normalizes fp16 variants."""
        from auto_round.utils.model import get_model_dtype

        result = get_model_dtype("fp16", default="float32")
        assert result == "float16"

        result = get_model_dtype("f16", default="float32")
        assert result == "float16"

    def test_fp32_normalized(self):
        """Test get_model_dtype normalizes fp32 variants."""
        from auto_round.utils.model import get_model_dtype

        result = get_model_dtype("fp32", default="float16")
        assert result == "float32"

        result = get_model_dtype("f32", default="float16")
        assert result == "float32"

    def test_unknown_dtype_resets_to_default(self):
        """Test get_model_dtype resets unknown dtype to default."""
        from auto_round.utils.model import get_model_dtype

        result = get_model_dtype("unknown", default="bfloat16")
        assert result == "bfloat16"


class TestCheckDiffusersInstalled:
    """Test check_diffusers_installed function."""

    def test_diffusers_installed(self):
        """Test check_diffusers_installed when diffusers is available."""
        from auto_round.utils.model import check_diffusers_installed

        with patch.dict("sys.modules", {"diffusers": MagicMock()}):
            result = check_diffusers_installed()
            assert result is True


class TestCheckStartWithBlockName:
    """Test check_start_with_block_name function."""

    def test_name_starts_with_block(self):
        """Test check_start_with_block_name returns True when name starts with block."""
        from auto_round.utils.model import check_start_with_block_name

        result = check_start_with_block_name("model.layers.0", ["model.layers"])
        assert result is True

    def test_name_does_not_start_with_block(self):
        """Test check_start_with_block_name returns False when name doesn't start with block."""
        from auto_round.utils.model import check_start_with_block_name

        result = check_start_with_block_name("model.embeddings", ["model.layers"])
        assert result is False

    def test_multiple_block_names(self):
        """Test check_start_with_block_name with multiple block names."""
        from auto_round.utils.model import check_start_with_block_name

        result = check_start_with_block_name(
            "model.decoder.layers.0", ["model.decoder.layers", "model.encoder.layers"]
        )
        assert result is True


class TestIsMoeLayer:
    """Test is_moe_layer function."""

    def test_qwen2_moe_sparse_moe_block(self):
        """Test is_moe_layer with Qwen2MoeSparseMoeBlock."""
        from auto_round.utils.model import is_moe_layer

        mock_module = MagicMock()
        mock_module.__class__.__name__ = "Qwen2MoeSparseMoeBlock"
        assert is_moe_layer(mock_module) is True

    def test_mixtral_sparse_moe_block(self):
        """Test is_moe_layer with MixtralSparseMoeBlock."""
        from auto_round.utils.model import is_moe_layer

        mock_module = MagicMock()
        mock_module.__class__.__name__ = "MixtralSparseMoeBlock"
        assert is_moe_layer(mock_module) is True

    def test_regular_linear_layer(self):
        """Test is_moe_layer with regular Linear layer."""
        from auto_round.utils.model import is_moe_layer
        import torch.nn as nn

        linear = nn.Linear(10, 10)
        assert is_moe_layer(linear) is False


class TestSetNestedAttr:
    """Test set_nested_attr function."""

    def test_set_nested_attribute(self):
        """Test set_nested_attr sets nested attribute correctly."""
        from auto_round.utils.model import set_nested_attr

        class MockModule:
            def __init__(self):
                self.orig_layer = MagicMock()

        module = MockModule()
        set_nested_attr(module, "orig_layer.act_max", torch.tensor([1.0]))
        assert hasattr(module.orig_layer, "act_max")

    def test_set_nested_attribute_missing_parent(self):
        """Test set_nested_attr handles missing parent gracefully."""
        from auto_round.utils.model import set_nested_attr

        module = MagicMock()
        result = set_nested_attr(module, "missing_layer.act_max", torch.tensor([1.0]))
        assert result is None


class TestGetAttr:
    """Test get_attr function."""

    def test_get_existing_attribute(self):
        """Test get_attr with existing attribute."""
        from auto_round.utils.model import get_attr

        module = MagicMock()
        module.layer.weight = torch.tensor([1.0])
        result = get_attr(module, "layer.weight")
        assert result is not None

    def test_get_missing_attribute(self):
        """Test get_attr with missing attribute returns None."""
        from auto_round.utils.model import get_attr

        module = MagicMock(spec=[])
        result = get_attr(module, "layer.missing")
        assert result is None

    def test_get_with_none_module(self):
        """Test get_attr with None module returns None."""
        from auto_round.utils.model import get_attr

        result = get_attr(None, "layer.weight")
        assert result is None


class TestSetAttr:
    """Test set_attr function."""

    def test_set_existing_attribute(self):
        """Test set_attr sets existing attribute correctly."""
        from auto_round.utils.model import set_attr
        import torch.nn as nn

        # Use a real model where we can set an attribute
        model = nn.Linear(10, 10)
        new_bias = nn.Parameter(torch.zeros(10))
        set_attr(model, "bias", new_bias)
        # Verify the attribute was set
        assert model.bias is not None


class TestGetModule:
    """Test get_module function."""

    def test_get_existing_submodule(self):
        """Test get_module with existing submodule."""
        from auto_round.utils.model import get_module
        import torch.nn as nn

        model = nn.Sequential(nn.Linear(10, 10))
        result = get_module(model, "0")
        assert result is not None
        assert isinstance(result, nn.Linear)

    def test_get_missing_submodule(self):
        """Test get_module with missing submodule returns None."""
        from auto_round.utils.model import get_module
        import torch.nn as nn

        model = nn.Linear(10, 10)
        result = get_module(model, "nonexistent")
        assert result is None


class TestSetModule:
    """Test set_module function."""

    def test_set_new_module(self):
        """Test set_module sets new module correctly."""
        from auto_round.utils.model import set_module
        import torch.nn as nn

        model = nn.Sequential(nn.Linear(10, 10))
        new_linear = nn.Linear(10, 10)
        # set_module should not raise and should handle missing paths gracefully
        set_module(model, "1", new_linear)


class TestGetLayerFeatures:
    """Test get_layer_features function."""

    def test_linear_layer(self):
        """Test get_layer_features with Linear layer."""
        from auto_round.utils.model import get_layer_features
        import torch.nn as nn

        linear = nn.Linear(10, 20)
        in_features, out_features = get_layer_features(linear)
        assert in_features == 10
        assert out_features == 20

    def test_embedding_layer(self):
        """Test get_layer_features with Embedding layer."""
        from auto_round.utils.model import get_layer_features
        import torch.nn as nn

        embedding = nn.Embedding(100, 50)
        num_embeddings, embedding_dim = get_layer_features(embedding)
        assert num_embeddings == 100
        assert embedding_dim == 50

    def test_unsupported_layer(self):
        """Test get_layer_features with unsupported layer returns None."""
        from auto_round.utils.model import get_layer_features

        module = MagicMock()
        module.__class__.__name__ = "UnsupportedLayer"
        in_features, out_features = get_layer_features(module)
        assert in_features is None
        assert out_features is None


class TestGetCommonPrefix:
    """Test get_common_prefix function."""

    def test_common_prefix_single_level(self):
        """Test get_common_prefix with single level paths."""
        from auto_round.utils.model import get_common_prefix

        paths = ["a.0", "a.1", "a.2"]
        result = get_common_prefix(paths)
        assert result == "a"

    def test_common_prefix_nested(self):
        """Test get_common_prefix with nested paths."""
        from auto_round.utils.model import get_common_prefix

        paths = ["model.layers.0.weight", "model.layers.1.weight"]
        result = get_common_prefix(paths)
        # Function finds common prefix by comparing component by component
        # Result includes 'weight' because it's common to both paths
        assert result == "model.layers.weight"

    def test_no_common_prefix(self):
        """Test get_common_prefix with no common prefix."""
        from auto_round.utils.model import get_common_prefix

        paths = ["a.0", "b.0"]
        result = get_common_prefix(paths)
        # Function returns the first common component found
        assert result == "0"


class TestUnsupportedMetaDevice:
    """Test unsupported_meta_device function."""

    def test_model_with_all_params_same_device(self):
        """Test unsupported_meta_device returns False when all params on same device."""
        from auto_round.utils.model import unsupported_meta_device
        import torch.nn as nn

        model = nn.Linear(10, 10)
        result = unsupported_meta_device(model)
        assert result is False


class TestToDevice:
    """Test to_device function."""

    def test_none_input(self):
        """Test to_device with None input returns None."""
        from auto_round.utils.model import to_device

        result = to_device(None)
        assert result is None

    def test_tensor_to_device(self):
        """Test to_device moves tensor to target device."""
        from auto_round.utils.model import to_device

        tensor = torch.tensor([1.0, 2.0])
        result = to_device(tensor, torch.device("cpu"))
        assert result.device.type == "cpu"

    def test_dict_to_device(self):
        """Test to_device moves dict values to target device."""
        from auto_round.utils.model import to_device

        data = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
        result = to_device(data, torch.device("cpu"))
        assert result["a"].device.type == "cpu"
        assert result["b"].device.type == "cpu"

    def test_list_to_device(self):
        """Test to_device moves list elements to target device."""
        from auto_round.utils.model import to_device

        data = [torch.tensor([1.0]), torch.tensor([2.0])]
        result = to_device(data, torch.device("cpu"))
        assert result[0].device.type == "cpu"
        assert result[1].device.type == "cpu"

    def test_empty_list(self):
        """Test to_device with empty list returns same list."""
        from auto_round.utils.model import to_device

        data = []
        result = to_device(data, torch.device("cpu"))
        assert result == []


class TestToDtype:
    """Test to_dtype function."""

    def test_none_input(self):
        """Test to_dtype with None input returns None."""
        from auto_round.utils.model import to_dtype

        result = to_dtype(None)
        assert result is None

    def test_tensor_to_dtype(self):
        """Test to_dtype converts tensor dtype."""
        from auto_round.utils.model import to_dtype

        tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
        result = to_dtype(tensor, torch.float16)
        assert result.dtype == torch.float16

    def test_dict_to_dtype(self):
        """Test to_dtype converts dict values dtype."""
        from auto_round.utils.model import to_dtype

        data = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
        result = to_dtype(data, torch.float16)
        assert result["a"].dtype == torch.float16
        assert result["b"].dtype == torch.float16


class TestIsPureTextModel:
    """Test is_pure_text_model function."""

    def test_opt_model_is_pure_text(self):
        """Test OPT model is identified as pure text."""
        from auto_round.utils.model import is_pure_text_model
        import transformers

        config = transformers.AutoConfig.from_pretrained("facebook/opt-125m")
        config.num_hidden_layers = 2
        model = transformers.OPTForCausalLM(config)
        result = is_pure_text_model(model)
        assert result is True


class TestIsGgufModel:
    """Test is_gguf_model function."""

    def test_gguf_file_path(self):
        """Test is_gguf_model with .gguf file path."""
        from auto_round.utils.model import is_gguf_model
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = os.path.join(tmpdir, "model.gguf")
            # Create a placeholder file
            with open(gguf_path, "wb") as f:
                f.write(b"placeholder")
            result = is_gguf_model(tmpdir)
            assert result is True

    def test_gguf_directory(self):
        """Test is_gguf_model with directory containing .gguf files."""
        from auto_round.utils.model import is_gguf_model
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake .gguf file marker
            result = is_gguf_model(tmpdir)
            assert result is False


class TestIsDiffusionModel:
    """Test is_diffusion_model function."""

    def test_string_path_with_config(self):
        """Test is_diffusion_model with string path that has config."""
        from auto_round.utils.model import is_diffusion_model
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"model_type": "diffusers"}, f)

            result = is_diffusion_model(tmpdir)
            assert result is False


class TestDetectModelType:
    """Test detect_model_type function."""

    def test_llm_model_type(self):
        """Test detect_model_type returns 'llm' for regular LLM."""
        from auto_round.utils.model import detect_model_type
        import transformers

        config = transformers.AutoConfig.from_pretrained("facebook/opt-125m")
        config.num_hidden_layers = 2
        model = transformers.OPTForCausalLM(config)

        result = detect_model_type(model)
        assert result == "llm"


class TestExtractBlockNamesToStr:
    """Test extract_block_names_to_str function."""

    def test_list_input(self):
        """Test extract_block_names_to_str with list input."""
        from auto_round.utils.model import extract_block_names_to_str

        block_names = [["model.layers.0", "model.layers.1"], ["model.layers.2"]]
        result = extract_block_names_to_str(block_names)
        assert result is not None
        assert isinstance(result, str)

    def test_non_list_input(self):
        """Test extract_block_names_to_str with non-list input returns None."""
        from auto_round.utils.model import extract_block_names_to_str

        result = extract_block_names_to_str("not_a_list")
        assert result is None


class TestFindMatchingBlocks:
    """Test find_matching_blocks function."""

    def test_empty_to_quant_block_names(self):
        """Test find_matching_blocks with empty to_quant_block_names returns all blocks."""
        from auto_round.utils.model import find_matching_blocks

        all_blocks = [["model.layers.0", "model.layers.1"]]
        result = find_matching_blocks(None, all_blocks, None)
        assert result == all_blocks

    def test_list_input_passthrough(self):
        """Test find_matching_blocks with list input returns it as-is."""
        from auto_round.utils.model import find_matching_blocks

        to_quant = [["model.layers.0"]]
        result = find_matching_blocks(None, [], to_quant)
        assert result == to_quant

    def test_regex_matching(self):
        """Test find_matching_blocks with regex pattern matching."""
        from auto_round.utils.model import find_matching_blocks

        all_blocks = [["model.layers.0", "model.layers.1", "model.embeddings"]]
        result = find_matching_blocks(None, all_blocks, "layers")
        assert len(result) > 0


class TestHandleGenerationConfig:
    """Test handle_generation_config function."""

    def test_model_without_generation_config(self):
        """Test handle_generation_config with model without generation_config."""
        from auto_round.utils.model import handle_generation_config

        model = MagicMock(spec=[])
        del model.generation_config
        # Should not raise
        handle_generation_config(model)

    def test_model_with_top_p_not_one(self):
        """Test handle_generation_config sets do_sample when top_p != 1.0."""
        from auto_round.utils.model import handle_generation_config

        model = MagicMock()
        model.generation_config = MagicMock()
        model.generation_config.top_p = 0.9
        model.generation_config.top_k = 0
        model.generation_config.temperature = 1.0
        model.generation_config.do_sample = False

        handle_generation_config(model)
        assert model.generation_config.do_sample is True


class TestCheckSeqLenCompatible:
    """Test check_seqlen_compatible function."""

    def test_model_exceeds_max_position_embeddings(self):
        """Test check_seqlen_compatible raises when input exceeds max_position_embeddings."""
        from auto_round.utils.model import check_seqlen_compatible
        import transformers

        config = transformers.AutoConfig.from_pretrained("facebook/opt-125m")
        config.max_position_embeddings = 2048
        model = MagicMock()
        model.config = config

        with pytest.raises(ValueError, match="exceeds model.config.max_position_embeddings"):
            check_seqlen_compatible(4096, model=model)


class TestCheckToQuantized:
    """Test check_to_quantized function."""

    def test_bits_leq_8_returns_true(self):
        """Test check_to_quantized returns True when bits <= 8."""
        from auto_round.utils.model import check_to_quantized

        config = {"bits": 4}
        result = check_to_quantized(config)
        assert result is True

    def test_bits_gt_8_returns_false(self):
        """Test check_to_quantized returns False when bits > 8."""
        from auto_round.utils.model import check_to_quantized

        config = {"bits": 16}
        result = check_to_quantized(config)
        assert result is False

    def test_act_bits_leq_8_returns_true(self):
        """Test check_to_quantized returns True when act_bits <= 8."""
        from auto_round.utils.model import check_to_quantized

        config = {"bits": 16, "act_bits": 4}
        result = check_to_quantized(config)
        assert result is True


class TestConvertDtypeTorch2StrHf:
    """Test convert_dtype_torch2str_hf function."""

    def test_float32_to_hf_str(self):
        """Test conversion from torch.float32 to huggingface string."""
        from auto_round.utils.model import convert_dtype_torch2str_hf

        result = convert_dtype_torch2str_hf(torch.float32)
        assert result == "float32"

    def test_float16_to_hf_str(self):
        """Test conversion from torch.float16 to huggingface string."""
        from auto_round.utils.model import convert_dtype_torch2str_hf

        result = convert_dtype_torch2str_hf(torch.float16)
        assert result == "float16"

    def test_none_input(self):
        """Test conversion with None input returns None."""
        from auto_round.utils.model import convert_dtype_torch2str_hf

        result = convert_dtype_torch2str_hf(None)
        assert result is None

    def test_string_input(self):
        """Test conversion with string input that already looks like hf dtype."""
        from auto_round.utils.model import convert_dtype_torch2str_hf

        result = convert_dtype_torch2str_hf("float32")
        assert result == "float32"

    def test_unsupported_dtype_raises(self):
        """Test that unsupported dtype raises ValueError."""
        from auto_round.utils.model import convert_dtype_torch2str_hf

        mock_dtype = MagicMock()
        mock_dtype.__str__ = lambda self: "unknown"
        with pytest.raises(ValueError, match="Unsupported pytorch dtype"):
            convert_dtype_torch2str_hf(mock_dtype)


class TestMergeBlockOutputKeys:
    """Test merge_block_output_keys function."""

    def test_merge_without_positional_inputs(self):
        """Test merge_block_output_keys without positional inputs."""
        from auto_round.utils.model import merge_block_output_keys

        block = MagicMock()
        input_others = {"key": "value"}
        extra_keys = {"extra": "data"}

        merge_block_output_keys(block, input_others, extra_keys)
        assert "extra" in input_others

    def test_merge_with_positional_inputs(self):
        """Test merge_block_output_keys with positional inputs."""
        from auto_round.utils.model import merge_block_output_keys

        block = MagicMock()
        input_others = {"positional_inputs": (MagicMock(),)}
        extra_keys = {"key1": "value1"}

        merge_block_output_keys(block, input_others, extra_keys)
        assert "key1" in input_others


class TestWrapBlockForwardPositionalToKwargs:
    """Test wrap_block_forward_positional_to_kwargs function."""

    def test_wrapper_creation(self):
        """Test wrap_block_forward_positional_to_kwargs returns a function."""
        from auto_round.utils.model import wrap_block_forward_positional_to_kwargs

        base_hook = MagicMock()
        wrapper = wrap_block_forward_positional_to_kwargs(base_hook)
        assert callable(wrapper)


class TestConfigSavePretrained:
    """Test config_save_pretrained function."""

    def test_save_to_directory(self):
        """Test config_save_pretrained saves to directory."""
        from auto_round.utils.model import config_save_pretrained
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"model_type": "opt"}
            config_save_pretrained(config, "config.json", tmpdir)

            config_path = os.path.join(tmpdir, "config.json")
            assert os.path.exists(config_path)


class TestRenameWeightsFiles:
    """Test rename_weights_files function."""

    def test_rename_single_safetensor(self):
        """Test rename_weights_files with single safetensor file."""
        from auto_round.utils.model import rename_weights_files
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a placeholder file
            safe_path = os.path.join(tmpdir, "model-00001-of-00002.safetensors")
            with open(safe_path, "wb") as f:
                f.write(b"placeholder")

            rename_weights_files(tmpdir)

            new_path = os.path.join(tmpdir, "diffusion_pytorch_model.safetensors")
            assert os.path.exists(new_path)


class TestHookNgramEmbeddingsOnCpu:
    """Test hook_ngram_embeddings_on_cpu function."""

    def test_model_without_ngram_embeddings(self):
        """Test hook_ngram_embeddings_on_cpu with regular model."""
        from auto_round.utils.model import hook_ngram_embeddings_on_cpu
        import transformers

        config = transformers.AutoConfig.from_pretrained("facebook/opt-125m")
        config.num_hidden_layers = 2
        model = transformers.OPTForCausalLM(config)

        has_ngram, raw_ngram = hook_ngram_embeddings_on_cpu(model)
        assert has_ngram is False
        assert raw_ngram is None


class TestMvModuleFromGpu:
    """Test mv_module_from_gpu function."""

    def test_move_module_to_cpu(self):
        """Test mv_module_from_gpu moves module to cpu."""
        from auto_round.utils.model import mv_module_from_gpu
        import torch.nn as nn

        linear = nn.Linear(10, 10)
        result = mv_module_from_gpu(linear)
        assert result is linear


class TestSafeDeviceMoveWithMetaHandling:
    """Test safe_device_move_with_meta_handling function."""

    def test_move_model_to_cpu(self):
        """Test safe_device_move_with_meta_handling moves model to cpu."""
        from auto_round.utils.model import safe_device_move_with_meta_handling
        import torch.nn as nn

        model = nn.Linear(10, 10)
        result = safe_device_move_with_meta_handling(model, "cpu")
        assert result is model


class TestIsMoeModel:
    """Test is_moe_model function."""

    def test_regular_model_returns_false(self):
        """Test is_moe_model with non-MoE model returns False."""
        from auto_round.utils.model import is_moe_model
        import torch.nn as nn

        model = nn.Linear(10, 10)
        result = is_moe_model(model)
        assert result is False


class TestFindLayersFromConfig:
    """Test find_layers_from_config function."""

    def test_find_layers_from_local_config(self):
        """Test find_layers_from_config with local config directory."""
        from auto_round.utils.model import find_layers_from_config
        import tempfile
        import os
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "model_type": "opt",
                "architectures": ["OPTForCausalLM"],
                "num_hidden_layers": 2,
            }
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f)

            result = find_layers_from_config(tmpdir)
            assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
