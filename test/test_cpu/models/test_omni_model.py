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

"""Unit tests for Qwen2.5-Omni and Qwen3-Omni-MoE model support.

Tests cover:
- Block name discovery (thinker + talker layers)
- MoE module replacement and weight fidelity (Qwen3-Omni)
- Forward function patching (_handle_special_model)
- Processor and template registration
- MoE utility functions (is_moe_layer, get_expert_linear_names, etc.)
- Ignore layer registration (mlp.gate for MoE)
"""

import copy
import shutil

import pytest
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration

from ...helpers import check_version, transformers_version

pytestmark = pytest.mark.skipif(
    not check_version("transformers>=5.1.0"),
    reason="Qwen-Omni models require transformers >= 5.1.0",
)


# ---------------------------------------------------------------------------
# Helper: create tiny Qwen3-Omni-MoE config (no real checkpoint needed for MoE)
# ---------------------------------------------------------------------------
def _make_tiny_qwen3_omni_moe_config():
    config = Qwen3OmniMoeConfig()
    # Thinker
    config.thinker_config.text_config.num_hidden_layers = 1
    config.thinker_config.text_config.hidden_size = 64
    config.thinker_config.text_config.intermediate_size = 128
    config.thinker_config.text_config.moe_intermediate_size = 32
    config.thinker_config.text_config.num_attention_heads = 4
    config.thinker_config.text_config.num_key_value_heads = 2
    config.thinker_config.text_config.num_experts = 4
    config.thinker_config.text_config.num_experts_per_tok = 2
    config.thinker_config.vision_config.depth = 1
    config.thinker_config.vision_config.embed_dim = 64
    config.thinker_config.vision_config.hidden_size = 64
    config.thinker_config.vision_config.num_heads = 4
    config.thinker_config.audio_config.num_hidden_layers = 1
    # Talker
    config.talker_config.text_config.num_hidden_layers = 1
    config.talker_config.text_config.hidden_size = 64
    config.talker_config.text_config.intermediate_size = 128
    config.talker_config.text_config.moe_intermediate_size = 32
    config.talker_config.text_config.num_attention_heads = 4
    config.talker_config.text_config.num_key_value_heads = 2
    config.talker_config.text_config.num_experts = 4
    config.talker_config.text_config.num_local_experts = 4
    config.talker_config.text_config.num_experts_per_tok = 2
    config.talker_config.text_config.shared_expert_intermediate_size = 64
    config.talker_config.thinker_hidden_size = 64
    config.talker_config.spatial_merge_size = 2
    # Code2wav (minimal)
    config.code2wav_config.hidden_size = 64
    config.code2wav_config.num_hidden_layers = 1
    config.code2wav_config.num_attention_heads = 4
    config.code2wav_config.num_key_value_heads = 4
    config.code2wav_config.intermediate_size = 128
    config.initializer_range = 0.02  # Default initializer range for weight initialization
    return config


# ========================= Qwen2.5-Omni Tests =============================
# NOTE: Tests use the `tiny_qwen2_5_omni` session-scoped fixture from fixtures.py
# (real config, reduced layers, random weights). Skipped if model not available.


class TestQwen2_5Omni:
    """Test block name discovery for Qwen2.5-Omni (dense, not MoE)."""

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self, tiny_qwen2_5_omni_model_path, request):
        request.cls.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            tiny_qwen2_5_omni_model_path, trust_remote_code=True
        )
        yield
        shutil.rmtree("runs", ignore_errors=True)

    def test_block_names_default(self):
        """Test that get_block_names returns thinker + talker layers."""
        from auto_round.utils import get_block_names

        block_names = get_block_names(self.model, quant_vision=False)
        # Should have thinker.model.layers and talker.model.layers
        assert any(
            "thinker.model.layers" in str(b) for b in block_names
        ), f"Expected thinker.model.layers in block_names, got: {block_names}"
        assert any(
            "talker.model.layers" in str(b) for b in block_names
        ), f"Expected talker.model.layers in block_names, got: {block_names}"

    def test_block_names_quant_vision(self):
        """Test that quant_vision adds visual and audio blocks."""
        from auto_round.utils import get_block_names

        blocks_no_vision = get_block_names(self.model, quant_vision=False)
        blocks_with_vision = get_block_names(self.model, quant_vision=True)

        assert len(blocks_with_vision) > len(blocks_no_vision), "quant_vision=True should add visual/audio blocks"

    def test_handle_special_model_forward(self):
        """Test that _handle_special_model patches the forward function."""
        from auto_round.special_model_handler import _handle_special_model

        # Deepcopy to avoid mutating the shared session-scoped fixture
        model = copy.deepcopy(self.model)
        original_forward = model.forward
        model = _handle_special_model(model)
        assert model.forward != original_forward, "Forward should be patched for qwen2_5_omni"

    def test_not_moe(self):
        from auto_round.utils.model import is_moe_layer

        # Thinker uses dense MLP, not MoE
        thinker_mlp = self.model.thinker.model.layers[0].mlp
        assert not is_moe_layer(thinker_mlp), "Qwen2.5-Omni should not be detected as MoE"

    def test_not_custom_model(self):
        from auto_round.modeling.fused_moe.replace_modules import is_custom_model

        assert not is_custom_model(self.model), "Qwen2.5-Omni should not be in BUILTIN_MODULES"


class TestQwen2_5OmniProcessor:
    """Test processor and template registration for Qwen2.5-Omni."""

    def test_processor_registered(self):
        from auto_round.compressors.mllm.processor import PROCESSORS

        assert "qwen2_5_omni" in PROCESSORS, "qwen2_5_omni processor not registered"

    def test_template_registered(self):
        from auto_round.compressors.mllm.template import TEMPLATES

        assert "qwen2_5_omni" in TEMPLATES, "qwen2_5_omni template not registered"

    def test_template_default_dataset(self):
        from auto_round.compressors.mllm.template import TEMPLATES

        template = TEMPLATES["qwen2_5_omni"]
        assert template.default_dataset is not None


# ========================= Qwen3-Omni-MoE Tests ===========================


class TestQwen3OmniMoeBlockNames:
    """Test block name discovery for Qwen3-Omni-MoE."""

    def test_block_names_default(self):
        """Test that get_block_names returns thinker + talker layers."""
        from auto_round.utils import get_block_names

        config = _make_tiny_qwen3_omni_moe_config()
        model = Qwen3OmniMoeForConditionalGeneration(config)

        block_names = get_block_names(model, quant_vision=False)
        assert any(
            "thinker.model.layers" in str(b) for b in block_names
        ), f"Expected thinker.model.layers, got: {block_names}"
        assert any(
            "talker.model.layers" in str(b) for b in block_names
        ), f"Expected talker.model.layers, got: {block_names}"

    def test_block_names_quant_vision(self):
        """Test that quant_vision adds visual and audio blocks."""
        from auto_round.utils import get_block_names

        config = _make_tiny_qwen3_omni_moe_config()
        model = Qwen3OmniMoeForConditionalGeneration(config)

        blocks_no_vision = get_block_names(model, quant_vision=False)
        blocks_with_vision = get_block_names(model, quant_vision=True)

        assert len(blocks_with_vision) > len(blocks_no_vision), "quant_vision=True should add visual/audio blocks"


class TestQwen3OmniMoeForward:
    """Test forward function patching for Qwen3-Omni-MoE."""

    def test_handle_special_model(self):
        """Test that _handle_special_model patches the forward function."""
        from auto_round.special_model_handler import _handle_special_model

        config = _make_tiny_qwen3_omni_moe_config()
        model = Qwen3OmniMoeForConditionalGeneration(config)

        original_forward = model.forward
        model = _handle_special_model(model)
        assert model.forward != original_forward, "Forward should be patched for qwen3_omni_moe"


class TestQwen3OmniMoeReplacement:
    """Test MoE module replacement for Qwen3-Omni-MoE."""

    def test_replacement_registered(self):
        """Test that both thinker and talker MoE blocks are registered."""
        from auto_round.modeling.fused_moe.qwen3_omni import (
            LinearQwen3OmniTalkerSparseMoeBlock,
            LinearQwen3OmniThinkerSparseMoeBlock,
        )
        from auto_round.modeling.fused_moe.replace_modules import ReplacementModuleBase

        assert ReplacementModuleBase.is_registered("Qwen3OmniMoeThinkerTextSparseMoeBlock")
        assert ReplacementModuleBase.is_registered("Qwen3OmniMoeTalkerTextSparseMoeBlock")

    def test_builtin_modules_entry(self):
        """Test that qwen3_omni_moe is in BUILTIN_MODULES."""
        from auto_round.modeling.fused_moe.replace_modules import BUILTIN_MODULES

        assert "qwen3_omni_moe" in BUILTIN_MODULES

    def test_is_custom_model(self):
        """Test that is_custom_model returns True for Qwen3-Omni-MoE."""
        from auto_round.modeling.fused_moe.replace_modules import is_custom_model

        config = _make_tiny_qwen3_omni_moe_config()
        model = Qwen3OmniMoeForConditionalGeneration(config)
        assert is_custom_model(model)

    def test_apply_replacements(self):
        """Test that MoE blocks are correctly replaced."""
        from auto_round.modeling.fused_moe.replace_modules import apply_replacements

        config = _make_tiny_qwen3_omni_moe_config()
        model = Qwen3OmniMoeForConditionalGeneration(config)

        model = apply_replacements(model)

        # Check that thinker MoE was replaced
        thinker_mlp = model.thinker.model.layers[0].mlp
        assert (
            "LinearQwen3OmniThinker" in thinker_mlp.__class__.__name__
        ), f"Expected LinearQwen3OmniThinker, got {thinker_mlp.__class__.__name__}"

        # Check that talker MoE was replaced
        talker_mlp = model.talker.model.layers[0].mlp
        assert (
            "LinearQwen3OmniTalker" in talker_mlp.__class__.__name__
        ), f"Expected LinearQwen3OmniTalker, got {talker_mlp.__class__.__name__}"

    def test_weight_fidelity(self):
        """Test that unfused weights match original fused weights."""
        from auto_round.modeling.fused_moe.replace_modules import apply_replacements, materialize_model_

        torch.manual_seed(42)
        config = _make_tiny_qwen3_omni_moe_config()
        model = Qwen3OmniMoeForConditionalGeneration(config)

        # Save original fused weights
        thinker_gate_up = model.thinker.model.layers[0].mlp.experts.gate_up_proj.data.clone()
        thinker_down = model.thinker.model.layers[0].mlp.experts.down_proj.data.clone()
        talker_gate_up = model.talker.model.layers[0].mlp.experts.gate_up_proj.data.clone()
        talker_down = model.talker.model.layers[0].mlp.experts.down_proj.data.clone()

        model = apply_replacements(model)
        materialize_model_(model)

        intermediate = 32  # moe_intermediate_size
        # Verify thinker expert weights
        for i in range(4):
            expert = model.thinker.model.layers[0].mlp.experts[i]
            assert torch.allclose(expert.gate_proj.weight.data, thinker_gate_up[i, :intermediate, :])
            assert torch.allclose(expert.up_proj.weight.data, thinker_gate_up[i, intermediate:, :])
            assert torch.allclose(expert.down_proj.weight.data, thinker_down[i])

        # Verify talker expert weights
        for i in range(4):
            expert = model.talker.model.layers[0].mlp.experts[i]
            assert torch.allclose(expert.gate_proj.weight.data, talker_gate_up[i, :intermediate, :])
            assert torch.allclose(expert.up_proj.weight.data, talker_gate_up[i, intermediate:, :])
            assert torch.allclose(expert.down_proj.weight.data, talker_down[i])

    def test_forward_output_match(self):
        """Test that replaced MoE forward output matches original."""
        from auto_round.modeling.fused_moe.replace_modules import apply_replacements, materialize_model_

        # Fix seed for deterministic model weights and use small-scale input to
        # prevent numerical overflow with random weights (talker has a larger
        # shared_expert MLP that can produce NaN otherwise).
        torch.manual_seed(0)
        config = _make_tiny_qwen3_omni_moe_config()
        model = Qwen3OmniMoeForConditionalGeneration(config)

        x = torch.randn(1, 4, 64) * 0.1
        with torch.no_grad():
            orig_thinker_out = model.thinker.model.layers[0].mlp(x)
            orig_talker_out = model.talker.model.layers[0].mlp(x)

        model = apply_replacements(model)
        materialize_model_(model)

        with torch.no_grad():
            new_thinker_out = model.thinker.model.layers[0].mlp(x)
            new_talker_out = model.talker.model.layers[0].mlp(x)

        # Use a NaN-safe comparison: check that NaN positions match, then
        # compare finite values.  This avoids false failures when random
        # weights cause some outputs to overflow to NaN.
        for name, orig, new in [
            ("Thinker", orig_thinker_out, new_thinker_out),
            ("Talker", orig_talker_out, new_talker_out),
        ]:
            assert orig.shape == new.shape, f"{name} shape mismatch: {orig.shape} vs {new.shape}"
            orig_nan = torch.isnan(orig)
            new_nan = torch.isnan(new)
            assert torch.equal(orig_nan, new_nan), f"{name} NaN positions differ"
            finite_mask = ~orig_nan
            if finite_mask.any():
                assert torch.allclose(
                    orig[finite_mask], new[finite_mask], atol=1e-5
                ), f"{name} MoE forward mismatch on finite values"


class TestQwen3OmniMoeProcessor:
    """Test processor and template registration for Qwen3-Omni-MoE."""

    def test_processor_registered(self):
        from auto_round.compressors.mllm.processor import PROCESSORS

        assert "qwen3_omni" in PROCESSORS, "qwen3_omni processor not registered"

    def test_template_registered(self):
        from auto_round.compressors.mllm.template import TEMPLATES

        assert "qwen3_omni_moe" in TEMPLATES, "qwen3_omni_moe template not registered"


class TestQwen3OmniMoeUtils:
    """Test MoE utility functions for Qwen3-Omni-MoE classes."""

    def test_is_moe_layer_thinker(self):
        from auto_round.utils.model import is_moe_layer

        config = _make_tiny_qwen3_omni_moe_config()
        model = Qwen3OmniMoeForConditionalGeneration(config)
        moe_block = model.thinker.model.layers[0].mlp
        assert is_moe_layer(moe_block), f"Thinker MoE block ({moe_block.__class__.__name__}) should be detected as MoE"

    def test_is_moe_layer_talker(self):
        from auto_round.utils.model import is_moe_layer

        config = _make_tiny_qwen3_omni_moe_config()
        model = Qwen3OmniMoeForConditionalGeneration(config)
        moe_block = model.talker.model.layers[0].mlp
        assert is_moe_layer(moe_block), f"Talker MoE block ({moe_block.__class__.__name__}) should be detected as MoE"

    def test_get_expert_linear_names(self):
        from auto_round.utils.model import get_expert_linear_names

        config = _make_tiny_qwen3_omni_moe_config()
        model = Qwen3OmniMoeForConditionalGeneration(config)

        thinker_mlp = model.thinker.model.layers[0].mlp
        names = get_expert_linear_names(thinker_mlp)
        assert set(names) == {"gate_proj", "down_proj", "up_proj"}

        talker_mlp = model.talker.model.layers[0].mlp
        names = get_expert_linear_names(talker_mlp)
        assert set(names) == {"gate_proj", "down_proj", "up_proj"}

    def test_get_expert_input_proj_names(self):
        from auto_round.utils.model import get_expert_input_proj_names

        config = _make_tiny_qwen3_omni_moe_config()
        model = Qwen3OmniMoeForConditionalGeneration(config)

        thinker_mlp = model.thinker.model.layers[0].mlp
        names = get_expert_input_proj_names(thinker_mlp)
        assert set(names) == {"gate_proj", "up_proj"}

    def test_ignore_layers_registered(self):
        from auto_round.special_model_handler import get_predefined_ignore_layers

        config = _make_tiny_qwen3_omni_moe_config()
        model = Qwen3OmniMoeForConditionalGeneration(config)

        ignore_layers = get_predefined_ignore_layers(model)
        assert ignore_layers == [
            "thinker.model.layers.0.mlp.gate",
            "talker.model.layers.0.mlp.gate",
        ], f"Expected mlp.gate in ignore_layers for qwen3_omni_moe, got: {ignore_layers}"


class TestVisualKeysExclusion:
    """Test that omni sub-modules are properly excluded from quantization."""

    def test_visual_keys_contain_omni_keys(self):
        from auto_round.compressors.mllm.utils import VISUAL_KEYS

        expected_keys = ["thinker", "talker", "audio", "token2wav", "code2wav", "audio_tower", "code_predictor"]
        for key in expected_keys:
            assert key in VISUAL_KEYS, f"'{key}' should be in VISUAL_KEYS"
