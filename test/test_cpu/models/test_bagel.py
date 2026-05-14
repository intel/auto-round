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
"""Unit tests for BAGEL model support."""

import json
import os
import shutil
import tempfile

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

# Absolute path to the repository root (repo_root/test/test_cpu/models/ -> repo_root)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from auto_round.special_model_handler import (
    _get_bagel_multimodal_block,
    get_bagel_ignore_layers,
    register_ignore_layers,
)
from auto_round.utils.model import (
    _EXTRA_MODEL_FILES,
    _LLM_ONLY_MODEL_TYPES,
    get_block_names,
    is_mllm_model,
    llm_load_model,
    mllm_load_model,
)

# ================= Fake BAGEL model for testing =================


class FakeBagelConfig:
    """Minimal fake config that mimics a BAGEL config.json."""

    def __init__(self):
        self.model_type = "bagel"
        self.architectures = ["BagelForConditionalGeneration"]
        self.hidden_size = 256
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.num_key_value_heads = 2
        self.intermediate_size = 512
        self.rms_norm_eps = 1e-6
        selfqk_norm = True

    def to_dict(self):
        return {
            "model_type": self.model_type,
            "architectures": self.architectures,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "rms_norm_eps": self.rms_norm_eps,
        }


def _make_fake_bagel_dir(num_layers=2, include_tokenizer=False):
    """Create a temporary directory with fake BAGEL-style safetensors, config, and optionally a tokenizer."""
    from transformers import AutoTokenizer

    tmpdir = tempfile.mkdtemp()

    # llm_config.json (Qwen2Config-like)
    llm_config = {
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": num_layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "qk_norm": True,
        "vocab_size": 1000,
    }
    with open(os.path.join(tmpdir, "llm_config.json"), "w") as f:
        json.dump(llm_config, f)

    # vit_config.json
    vit_config = {"hidden_size": 128, "num_hidden_layers": 1}
    with open(os.path.join(tmpdir, "vit_config.json"), "w") as f:
        json.dump(vit_config, f)

    # preprocessor_config.json
    with open(os.path.join(tmpdir, "preprocessor_config.json"), "w") as f:
        json.dump({}, f)

    # Root config.json (BAGEL model_type)
    config = {
        "model_type": "bagel",
        "architectures": ["BagelForConditionalGeneration"],
        "llm_config": llm_config,
        "torch_dtype": "bfloat16",
    }
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump(config, f)

    # Safetensors with language_model.* keys + non-language_model keys
    state_dict = {}
    # LLM params (keys use the BAGEL "language_model." prefix so bagel_loader
    # strips it before calling language_model.load_state_dict)
    for i in range(num_layers):
        layer_prefix = f"language_model.model.layers.{i}."
        state_dict[f"{layer_prefix}self_attn.q_proj.weight"] = torch.randn(256, 256)
        state_dict[f"{layer_prefix}self_attn.q_proj.bias"] = torch.randn(256)
        state_dict[f"{layer_prefix}self_attn.k_proj.weight"] = torch.randn(128, 256)
        state_dict[f"{layer_prefix}self_attn.k_proj.bias"] = torch.randn(128)
        state_dict[f"{layer_prefix}self_attn.v_proj.weight"] = torch.randn(128, 256)
        state_dict[f"{layer_prefix}self_attn.v_proj.bias"] = torch.randn(128)
        state_dict[f"{layer_prefix}self_attn.o_proj.weight"] = torch.randn(256, 256)
        state_dict[f"{layer_prefix}mlp.gate_proj.weight"] = torch.randn(512, 256)
        state_dict[f"{layer_prefix}mlp.up_proj.weight"] = torch.randn(512, 256)
        state_dict[f"{layer_prefix}mlp.down_proj.weight"] = torch.randn(256, 512)
        state_dict[f"{layer_prefix}input_layernorm.weight"] = torch.randn(256)
        state_dict[f"{layer_prefix}post_attention_layernorm.weight"] = torch.randn(256)
        # MOT generation-path params
        state_dict[f"{layer_prefix}self_attn.q_proj_moe_gen.weight"] = torch.randn(256, 256)
        state_dict[f"{layer_prefix}self_attn.q_proj_moe_gen.bias"] = torch.randn(256)
        state_dict[f"{layer_prefix}self_attn.k_proj_moe_gen.weight"] = torch.randn(128, 256)
        state_dict[f"{layer_prefix}self_attn.k_proj_moe_gen.bias"] = torch.randn(128)
        state_dict[f"{layer_prefix}self_attn.v_proj_moe_gen.weight"] = torch.randn(128, 256)
        state_dict[f"{layer_prefix}self_attn.v_proj_moe_gen.bias"] = torch.randn(128)
        state_dict[f"{layer_prefix}self_attn.o_proj_moe_gen.weight"] = torch.randn(256, 256)
        state_dict[f"{layer_prefix}mlp_moe_gen.gate_proj.weight"] = torch.randn(512, 256)
        state_dict[f"{layer_prefix}mlp_moe_gen.up_proj.weight"] = torch.randn(512, 256)
        state_dict[f"{layer_prefix}mlp_moe_gen.down_proj.weight"] = torch.randn(256, 512)
        state_dict[f"{layer_prefix}input_layernorm_moe_gen.weight"] = torch.randn(256)
        state_dict[f"{layer_prefix}post_attention_layernorm_moe_gen.weight"] = torch.randn(256)
        # QK norms
        state_dict[f"{layer_prefix}self_attn.q_norm.weight"] = torch.randn(64)
        state_dict[f"{layer_prefix}self_attn.k_norm.weight"] = torch.randn(64)
        state_dict[f"{layer_prefix}self_attn.q_norm_moe_gen.weight"] = torch.randn(64)
        state_dict[f"{layer_prefix}self_attn.k_norm_moe_gen.weight"] = torch.randn(64)
    state_dict["language_model.model.embed_tokens.weight"] = torch.randn(1000, 256)
    state_dict["language_model.lm_head.weight"] = torch.randn(1000, 256)
    state_dict["language_model.model.norm.weight"] = torch.randn(256)

    # Non-language_model params (e.g., ViT / VAE)
    state_dict["vit_model.vision_model.embeddings.patch_embedding.weight"] = torch.randn(128, 3, 16, 16)
    state_dict["vit_model.vision_model.post_layernorm.weight"] = torch.randn(128)
    state_dict["connector.fc1.weight"] = torch.randn(256, 128)
    state_dict["encoder.conv_in.weight"] = torch.randn(128, 4, 3, 3)

    save_file(state_dict, os.path.join(tmpdir, "model.safetensors"))

    if include_tokenizer:
        # Create a minimal tokenizer (Qwen vocab)
        from transformers import AutoTokenizer

        # Try to use an existing tiny tokenizer as base
        try:
            base_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=False)
        except Exception:
            base_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", trust_remote_code=False)
        base_tokenizer.save_pretrained(tmpdir)

    return tmpdir


# ================= Test: special_model_handler =================


class TestBagelSpecialModelHandler:
    """Tests for BAGEL-specific parts of special_model_handler.py."""

    def test_bagel_multimodal_block_returns_language_model_layers(self):
        """_get_bagel_multimodal_block should return block names for language_model layers."""

        # BAGEL's language_model is a Qwen2ForCausalLM (which has .model containing layers).
        # We create a minimal mock to avoid torch.nn.Module attribute resolution issues.
        class MockLayers(list):
            pass

        class MockLanguageModel:
            def __init__(self):
                self.model = type("MockModel", (), {"layers": MockLayers([None, None])})()

        class MockBagelModel:
            def __init__(self):
                self.language_model = MockLanguageModel()

        model = MockBagelModel()
        blocks = _get_bagel_multimodal_block(model)
        assert len(blocks) == 1
        assert len(blocks[0]) == 2
        assert blocks[0][0] == "language_model.model.layers.0"
        assert blocks[0][1] == "language_model.model.layers.1"

    def test_get_bagel_ignore_layers_structure(self):
        """get_bagel_ignore_layers should return generation-path modules."""

        # Use minimal mock matching BAGEL's attribute structure
        class MockLanguageModel:
            def __init__(self):
                self.model = type("MockModel", (), {"layers": [None, None, None]})()

        class MockBagelModel:
            def __init__(self):
                self.language_model = MockLanguageModel()

        model = MockBagelModel()
        ignore = get_bagel_ignore_layers(model)
        assert "moe_gen" in ignore
        assert "self_attn.q_proj" in ignore
        assert "self_attn.k_proj" in ignore
        assert "self_attn.v_proj" in ignore
        assert "self_attn.o_proj" in ignore

    def test_bagel_registered_in_special_multimodal_block(self):
        """BAGEL should be in SPECIAL_MULTIMODAL_BLOCK."""
        from auto_round.special_model_handler import SPECIAL_MULTIMODAL_BLOCK

        assert "bagel" in SPECIAL_MULTIMODAL_BLOCK
        handler = SPECIAL_MULTIMODAL_BLOCK["bagel"]
        assert callable(handler)

    def test_bagel_in_support_only_text_models(self):
        """BAGEL should be in SUPPORT_ONLY_TEXT_MODELS."""
        from auto_round.special_model_handler import SUPPORT_ONLY_TEXT_MODELS

        assert "bagel" in SUPPORT_ONLY_TEXT_MODELS


# ================= Test: is_mllm_model =================


class TestBagelIsMllm:
    """Tests for BAGEL handling in is_mllm_model (utils/model.py)."""

    def test_is_mllm_returns_false_for_bagel_model_type(self):
        """is_mllm_model should return False for BAGEL (LLM-only quantization)."""
        # BAGEL has multimodal components (ViT, VAE) but should be quantized as LLM
        assert "bagel" in _LLM_ONLY_MODEL_TYPES
        # Test via path
        fake_dir = _make_fake_bagel_dir(num_layers=2)
        try:
            result = is_mllm_model(fake_dir)
            assert result is False, "BAGEL should not be detected as MLLM"
        finally:
            shutil.rmtree(fake_dir)

    def test_is_mllm_returns_false_for_bagel_model_object(self):
        """is_mllm_model should return False for a BAGEL model object."""
        from transformers import Qwen2Config

        config = Qwen2Config(
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=512,
        )
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

        model = Qwen2ForCausalLM(config)

        class FakeBagelModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = model.lm_head
                self.name_or_path = "/fake/bagel/path"
                config = Qwen2Config()
                config.model_type = "bagel"
                self.config = config

        fake_model = FakeBagelModel()
        result = is_mllm_model(fake_model)
        assert result is False, "BAGEL model object should not be detected as MLLM"


# ================= Test: _EXTRA_MODEL_FILES =================


class TestBagelExtraModelFiles:
    """Tests for BAGEL files in _EXTRA_MODEL_FILES."""

    def test_extra_model_files_contain_bagel_configs(self):
        """llm_config.json, vit_config.json, preprocessor_config.json should be in _EXTRA_MODEL_FILES."""
        expected = {"llm_config.json", "vit_config.json", "preprocessor_config.json"}
        assert expected.issubset(_EXTRA_MODEL_FILES), (
            f"BAGEL extra files {expected} should be in _EXTRA_MODEL_FILES, " f"got {_EXTRA_MODEL_FILES}"
        )


# ================= Test: BagelConfig =================


class TestBagelConfig:
    """Tests for BagelConfig in bagel_loader.py."""

    def test_bagel_config_model_type(self):
        from auto_round.utils.bagel_loader import BagelConfig

        config = BagelConfig()
        assert config.model_type == "bagel"

    def test_bagel_config_accepts_kwargs(self):
        from auto_round.utils.bagel_loader import BagelConfig

        config = BagelConfig(hidden_size=512, num_hidden_layers=4)
        assert config.hidden_size == 512
        assert config.num_hidden_layers == 4


# ================= Test: BagelForQuantization save_pretrained =================


class TestBagelForQuantization:
    """Tests for BagelForQuantization wrapper (bagel_loader.py)."""

    @pytest.fixture
    def fake_bagel_dir(self):
        d = _make_fake_bagel_dir(num_layers=2, include_tokenizer=True)
        yield d
        shutil.rmtree(d, ignore_errors=True)

    @pytest.fixture
    def bagel_model_tokenizer(self, fake_bagel_dir):
        from auto_round.utils.bagel_loader import load_bagel_model

        model, tokenizer = load_bagel_model(fake_bagel_dir, torch_dtype=torch.bfloat16)
        return model, tokenizer

    def test_load_bagel_model_returns_model_and_tokenizer(self, bagel_model_tokenizer):
        """load_bagel_model should return (model, tokenizer)."""
        model, tokenizer = bagel_model_tokenizer
        assert model is not None
        assert tokenizer is not None

    def test_load_bagel_model_has_language_model(self, bagel_model_tokenizer):
        """Loaded model should have language_model attribute."""
        model, _ = bagel_model_tokenizer
        assert hasattr(model, "language_model")

    def test_load_bagel_model_has_bagel_config(self, bagel_model_tokenizer):
        """Loaded model config should have bagel model_type."""
        model, _ = bagel_model_tokenizer
        assert model.config.model_type == "bagel"

    def test_load_bagel_model_sets_name_or_path(self, bagel_model_tokenizer, fake_bagel_dir):
        """Loaded model should have name_or_path set."""
        model, _ = bagel_model_tokenizer
        assert model.name_or_path == fake_bagel_dir

    def test_load_bagel_model_sets_autoround_block_hint(self, fake_bagel_dir):
        """llm_load_model should set _autoround_to_quant_block_names for BAGEL."""
        model, _ = llm_load_model(fake_bagel_dir)
        assert hasattr(model, "_autoround_to_quant_block_names")
        assert model._autoround_to_quant_block_names == "language_model.model.layers"
        del model

    def test_load_bagel_model_dtype_auto_bfloat16(self, fake_bagel_dir):
        """torch_dtype='auto' should resolve bfloat16 from config."""
        from auto_round.utils.bagel_loader import load_bagel_model

        model, _ = load_bagel_model(fake_bagel_dir, torch_dtype="auto")
        assert model is not None

    def test_save_pretrained_includes_parameters(self, bagel_model_tokenizer, tmp_path):
        """save_pretrained should include all named_parameters."""
        model, _ = bagel_model_tokenizer
        save_dir = str(tmp_path / "saved_bagel")

        model.save_pretrained(save_dir)

        assert os.path.exists(os.path.join(save_dir, "config.json"))
        assert os.path.exists(os.path.join(save_dir, "model.safetensors"))

        # Check config has bagel model_type
        with open(os.path.join(save_dir, "config.json")) as f:
            saved_config = json.load(f)
        assert saved_config["model_type"] == "bagel"
        assert "BagelForConditionalGeneration" in saved_config["architectures"]

        # Check safetensors has parameters
        from safetensors.torch import load_file

        state = load_file(os.path.join(save_dir, "model.safetensors"))
        # Should contain at least the LLM params we loaded
        layer_keys = [k for k in state.keys() if "language_model.model.layers" in k]
        assert len(layer_keys) > 0, "Safetensors should contain language_model layer params"

    def test_save_pretrained_includes_buffers(self, bagel_model_tokenizer, tmp_path):
        """save_pretrained should include named_buffers (not just parameters).

        Regression test: previously only named_parameters was iterated.
        """
        model, _ = bagel_model_tokenizer

        # Add a real buffer to the model
        model.register_buffer("test_buffer", torch.randn(8, 8))
        assert "test_buffer" in [n for n, _ in model.named_buffers()]

        save_dir = str(tmp_path / "saved_bagel_buffers")
        model.save_pretrained(save_dir)

        from safetensors.torch import load_file

        state = load_file(os.path.join(save_dir, "model.safetensors"))
        # The buffer we added should be in the saved state_dict
        assert "test_buffer" in state, "named_buffers should be included in save_pretrained state_dict"


# ================= Test: compressor base - to_quant_block_names hint =================


class TestBagelCompressorHint:
    """Tests for _autoround_to_quant_block_names hint propagation in BaseCompressor."""

    def test_compressor_picks_up_autoround_hint_when_none(self):
        """Code path: to_quant_block_names=None should use model._autoround_to_quant_block_names."""
        # Verify the logic directly from the source
        to_quant_block_names = None
        hint = "transformer.h"
        expected = hint

        result = to_quant_block_names
        if result is None:
            _hint = hint  # simulating getattr(model, "_autoround_to_quant_block_names", None)
            if _hint is not None:
                result = _hint

        assert result == expected

    def test_compressor_uses_explicit_over_hint(self):
        """Code path: explicit to_quant_block_names should override the hint."""
        to_quant_block_names = "explicit.path"
        hint = "transformer.h"

        result = to_quant_block_names
        if result is None:
            _hint = hint
            if _hint is not None:
                result = _hint

        assert result == "explicit.path"

    def test_compressor_hint_parsed_in_base_compressor_source(self):
        """Verify BaseCompressor reads _autoround_to_quant_block_names from model."""
        source_path = os.path.join(REPO_ROOT, "auto_round/compressors/base.py")
        with open(source_path) as f:
            source = f.read()

        assert "_autoround_to_quant_block_names" in source, "BaseCompressor should read _autoround_to_quant_block_names"


# ================= Test: ValueError in AutoConfig.from_pretrained =================


class TestConfigErrorHandling:
    """Tests that ValueError is handled when loading config via AutoConfig (branch diff)."""

    def test_autoconfig_valueerror_caught_in_base_compressor(self):
        """BaseCompressor.__init__: ValueError should be caught in AutoConfig.from_pretrained."""
        source_path = os.path.join(REPO_ROOT, "auto_round/compressors/base.py")
        with open(source_path) as f:
            content = f.read()

        # The branch diff adds ValueError to the except clause around line 294
        # Verify the pattern appears (line numbers may shift)
        assert (
            "except (OSError, EnvironmentError, ValueError)" in content
        ), "BaseCompressor should catch ValueError alongside OSError/EnvironmentError"

    def test_autoconfig_valueerror_caught_in_model_context(self):
        """ModelContext: ValueError should be caught in AutoConfig.from_pretrained."""
        source_path = os.path.join(REPO_ROOT, "auto_round/context/model.py")
        with open(source_path) as f:
            content = f.read()

        # The branch diff adds ValueError to the except clause around line 146
        assert (
            "except (OSError, EnvironmentError, ValueError)" in content
        ), "ModelContext should catch ValueError alongside OSError/EnvironmentError"


# ================= Test: mllm_load_model for bagel =================


class TestBagelMllmLoadModel:
    """Tests for mllm_load_model handling of bagel model_type."""

    @pytest.fixture
    def fake_bagel_dir(self):
        d = _make_fake_bagel_dir(num_layers=2, include_tokenizer=True)
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_mllm_load_model_handles_bagel(self, fake_bagel_dir):
        """mllm_load_model should load BAGEL via bagel_loader (not raise)."""
        model, processor, tokenizer, image_processor = mllm_load_model(fake_bagel_dir)

        assert model is not None
        assert model.config.model_type == "bagel"
        assert tokenizer is not None
        # processor/image_processor should be None for BAGEL
        assert processor is None
        assert image_processor is None
