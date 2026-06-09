#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Comprehensive unit tests for GGUF conversion modules with low coverage.

Tests conversion modules that have 0% coverage in the test suite, covering
the actual logic paths in each module.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


# ==============================================================================
# Helper: build a minimal mock model that exercises specific methods
# ==============================================================================

def _make_mock_model(cls, hparams=None):
    """Create a bare-minimum mock of a conversion model class for testing."""
    if hparams is None:
        hparams = {}

    with patch.object(cls, "__init__", lambda self, *args, **kwargs: None):
        obj = cls.__new__(cls)

    obj.hparams = dict(hparams)
    obj.gguf_writer = MagicMock()
    obj.ftype = MagicMock()
    obj.dir_model = Path(tempfile.mkdtemp())
    obj.block_count = hparams.get("num_hidden_layers", 2)
    obj.model_tensors = {}
    obj._experts = None
    obj.lerp_weights = {}
    obj.lora_needs_transpose = True
    obj.rope_parameters = hparams.get("rope_parameters", {})
    obj.fuse_gate_up_exps = False
    obj.hparams_vision = hparams.get("hparams_vision")
    obj.global_config = hparams.get("global_config", {})
    obj.preprocessor_config = hparams.get("preprocessor_config", {})
    obj.is_mistral_format = False
    obj.origin_hf_arch = None
    obj.hf_arch = ""
    obj.undo_permute = True
    obj._is_nvfp4 = False
    obj._is_mxfp4 = False
    obj.head_dim = None
    obj.shared_token_embeddings_found = False
    obj.is_moe = False

    # tensor_map needs to handle both map_tensor_name calls (key=..., try_suffixes=...)
    # and format_tensor_name calls (key=..., bid=..., suffix=...)
    # It also needs a .mapping attribute that returns tuples of (name, formatted_name)
    def mock_get_name(key=None, try_suffixes=None):
        if key is not None and try_suffixes is not None:
            return key
        return key if key else "tensor"

    def mock_format_name(key, bid=None, suffix=".weight"):
        if hasattr(key, "name"):
            return key.name + suffix
        return f"tensor_{bid if bid is not None else '0'}{suffix}"

    def make_mapping():
        # tensor_map.mapping is a dict-like object. When iterating over .values(),
        # it yields (key, name) tuples. The unpacking is: for _, s in .values()
        # So we need a dict where values are tuples of (key, name_str)
        return {"tensor.weight": ("tensor_key", "tensor_name_weight")}

    mock_map = MagicMock()
    mock_map.get_name.side_effect = mock_get_name
    mock_map.mapping = make_mapping()

    obj.tensor_map = mock_map
    obj.format_tensor_name = mock_format_name

    return obj


# ==============================================================================
# mimo.py tests
# ==============================================================================

class TestMimoConversion:
    """Tests for MiMo conversion module."""

    def test_tp_aware_qkv_dequant_tp4(self):
        """Test _tp_aware_qkv_dequant with TP=4 configuration."""
        from auto_round.export.export_to_gguf.conversion.mimo import MimoV2Model

        # n_q=8, n_kv=2, hd=64, vhd=64
        # q_size=512, k_size=128, v_size=128, total=768
        n_q, n_kv, hd, vhd = 8, 2, 64, 64
        total_rows = n_q * hd + n_kv * hd + n_kv * vhd  # 768
        n_col = 1024
        bs = 128

        weight = torch.randn(total_rows, n_col)
        # TP=4: total_rows % 4 == 0, rows_per_rank = 192, bpr = ceil(192/128) = 2
        # scale_inv shape = tp * bpr x n_col_blocks = 4*2 x ceil(1024/128)=8
        # n_col_blocks = ceil(n_col/bs) for proper broadcasting
        n_col_blocks = (n_col + bs - 1) // bs  # = 8
        scale_inv = torch.randn(8, n_col_blocks)

        result = MimoV2Model._tp_aware_qkv_dequant(
            weight, scale_inv, n_q, n_kv, hd, vhd, bs=bs
        )
        assert result.shape == (total_rows, n_col)

    def test_tp_aware_qkv_dequant_tp8(self):
        """Test _tp_aware_qkv_dequant with TP=8 configuration."""
        from auto_round.export.export_to_gguf.conversion.mimo import MimoV2Model

        n_q, n_kv, hd, vhd = 8, 2, 64, 64
        total_rows = n_q * hd + n_kv * hd + n_kv * vhd  # 768
        n_col = 1024
        bs = 128

        weight = torch.randn(total_rows, n_col)
        # TP=8: total_rows % 8 == 0, rows_per_rank = 96, bpr = ceil(96/128) = 1
        # scale_inv shape = tp * bpr x n_col_blocks = 8*1 x 8
        n_col_blocks = (n_col + bs - 1) // bs  # = 8
        scale_inv = torch.randn(8, n_col_blocks)

        result = MimoV2Model._tp_aware_qkv_dequant(
            weight, scale_inv, n_q, n_kv, hd, vhd, bs=bs
        )
        assert result.shape == (total_rows, n_col)

    def test_tp_aware_qkv_dequant_invalid_rows(self):
        """Test that mismatched weight rows raise ValueError."""
        from auto_round.export.export_to_gguf.conversion.mimo import MimoV2Model

        weight = torch.randn(100, 64)
        scale_inv = torch.randn(2, 1)

        with pytest.raises(ValueError, match="qkv_proj weight rows"):
            MimoV2Model._tp_aware_qkv_dequant(weight, scale_inv, 4, 2, 32, 32)

    def test_tp_aware_qkv_dequant_cannot_detect_tp(self):
        """Test that undetectable TP raises ValueError."""
        from auto_round.export.export_to_gguf.conversion.mimo import MimoV2Model

        n_q, n_kv, hd, vhd = 8, 2, 64, 64
        total_rows = n_q * hd + n_kv * hd + n_kv * vhd
        weight = torch.randn(total_rows, 512)
        scale_inv = torch.randn(7, 1)  # no candidate TP matches

        with pytest.raises(ValueError, match="cannot detect TP"):
            MimoV2Model._tp_aware_qkv_dequant(weight, scale_inv, n_q, n_kv, hd, vhd)

    def test_filter_tensors_attention_sink_without_weight_suffix(self):
        """Test filter_tensors appends .weight to attention_sink tensor name."""
        from auto_round.export.export_to_gguf.conversion.mimo import MimoV2Model

        obj = _make_mock_model(MimoV2Model)
        obj.filter_tensors = MimoV2Model.filter_tensors.__get__(obj, MimoV2Model)

        name, gen = "model.layers.0.attention_sink", lambda: None
        result = obj.filter_tensors((name, gen))

        assert result is not None
        assert result[0] == "model.layers.0.attention_sink.weight"

    def test_filter_tensors_attention_sink_with_weight_suffix(self):
        """Test filter_tensors leaves attention_sink.weight unchanged."""
        from auto_round.export.export_to_gguf.conversion.mimo import MimoV2Model

        obj = _make_mock_model(MimoV2Model)
        obj.filter_tensors = MimoV2Model.filter_tensors.__get__(obj, MimoV2Model)

        name, gen = "model.layers.0.attention_sink.weight", lambda: None
        result = obj.filter_tensors((name, gen))

        assert result is not None
        assert result[0] == "model.layers.0.attention_sink.weight"

    def test_prepare_tensors_unprocessed_experts_error(self):
        """Test that unprocessed experts raise ValueError."""
        from auto_round.export.export_to_gguf.conversion.mimo import MimoV2Model

        obj = _make_mock_model(MimoV2Model, {
            "num_hidden_layers": 2,
            "n_routed_experts": 4,
        })
        obj._experts = [{"unprocessed.tensor.0": None}]
        obj.tensor_map.mapping = {"tensor": ("KEY", "tensor_name")}

        # Mock super().prepare_tensors() to skip the base class work
        with patch("auto_round.export.export_to_gguf.conversion.base.ModelBase.prepare_tensors"):
            with pytest.raises(ValueError, match="Unprocessed experts"):
                obj.prepare_tensors()


# ==============================================================================
# minicpm.py tests
# ==============================================================================

class TestMiniCPMConversion:
    """Tests for MiniCPM conversion module."""

    def test_generate_extra_tensors_with_rope_scaling(self):
        """Test generate_extra_tensors yields rope long/short factors."""
        from auto_round.export.export_to_gguf.conversion.minicpm import MiniCPMModel

        rope_dims = 64
        long_factors = [1.0] * (rope_dims // 2)
        short_factors = [1.0] * (rope_dims // 2)

        obj = _make_mock_model(MiniCPMModel, {
            "num_hidden_layers": 2,
            "hidden_size": 2048,
            "num_attention_heads": 32,
            "scale_emb": 1.0,
            "scale_depth": 4.0,
            "dim_model_base": 1024,
            "rope_scaling": {
                "long_factor": long_factors,
                "short_factor": short_factors,
            },
        })

        results = list(obj.generate_extra_tensors())

        assert len(results) == 2
        # The tensor name contains ROPE_FACTORS_LONG
        assert "ROPE_FACTORS_LONG" in results[0][0]
        assert "ROPE_FACTORS_SHORT" in results[1][0]
        assert results[0][1].shape == (rope_dims // 2,)

    def test_generate_extra_tensors_missing_long_factor_raises(self):
        """Test missing long_factor raises KeyError."""
        from auto_round.export.export_to_gguf.conversion.minicpm import MiniCPMModel

        obj = _make_mock_model(MiniCPMModel, {
            "num_hidden_layers": 2,
            "hidden_size": 2048,
            "num_attention_heads": 32,
            "scale_emb": 1.0,
            "scale_depth": 4.0,
            "dim_model_base": 1024,
            "rope_scaling": {
                "short_factor": [1.0] * 32,
            },
        })

        with pytest.raises(KeyError, match="long_factor"):
            list(obj.generate_extra_tensors())

    def test_generate_extra_tensors_length_mismatch_raises(self):
        """Test mismatched factor lengths raise ValueError."""
        from auto_round.export.export_to_gguf.conversion.minicpm import MiniCPMModel

        obj = _make_mock_model(MiniCPMModel, {
            "num_hidden_layers": 2,
            "hidden_size": 2048,
            "num_attention_heads": 32,
            "scale_emb": 1.0,
            "scale_depth": 4.0,
            "dim_model_base": 1024,
            "rope_scaling": {
                "long_factor": [1.0] * 64,
                "short_factor": [1.0] * 32,  # wrong length
            },
        })

        with pytest.raises(ValueError, match="length of rope long and short factors"):
            list(obj.generate_extra_tensors())

    def test_minicpm3_reverse_hf_permute(self):
        """Test MiniCPM3Model._reverse_hf_permute transforms tensor correctly."""
        from auto_round.export.export_to_gguf.conversion.minicpm import MiniCPM3Model

        obj = _make_mock_model(MiniCPM3Model, {
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "qk_nope_head_dim": 64,
            "qk_rope_head_dim": 32,
            "v_head_dim": 64,
            "kv_lora_rank": 128,
            "hidden_size": 512,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 4096,
        })

        tensor = torch.randn(256, 512)
        result = obj._reverse_hf_permute(tensor, 4, 2)
        assert result.shape == tensor.shape

    def test_minicpm3_reverse_hf_permute_same_head(self):
        """Test _reverse_hf_permute when n_kv_head equals n_head."""
        from auto_round.export.export_to_gguf.conversion.minicpm import MiniCPM3Model

        obj = _make_mock_model(MiniCPM3Model, {
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "qk_nope_head_dim": 64,
            "qk_rope_head_dim": 64,
            "v_head_dim": 64,
            "kv_lora_rank": 128,
            "hidden_size": 512,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 4096,
        })

        tensor = torch.randn(512, 512)
        result = obj._reverse_hf_permute(tensor, 8, 8)
        assert result.shape == tensor.shape


# ==============================================================================
# minimax.py tests
# ==============================================================================

class TestMiniMaxConversion:
    """Tests for MiniMax conversion module."""

    def test_set_gguf_parameters(self):
        """Test MiniMaxM2Model.set_gguf_parameters sets MoE parameters."""
        from auto_round.export.export_to_gguf.conversion.minimax import MiniMaxM2Model

        obj = _make_mock_model(MiniMaxM2Model, {
            "num_hidden_layers": 2,
            "num_local_experts": 8,
            "intermediate_size": 10944,
            "rotary_dim": 32,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "hidden_size": 2048,
        })

        obj.set_gguf_parameters()

        # Calls add_expert_feed_forward_length and add_rope_dimension_count
        obj.gguf_writer.add_expert_feed_forward_length.assert_called_with(10944)
        obj.gguf_writer.add_rope_dimension_count.assert_called_with(32)


# ==============================================================================
# mpt.py tests
# ==============================================================================

class TestMPTConversion:
    """Tests for MPT conversion module."""

    def test_modify_tensors_with_scales(self):
        """Test modify_tensors handles names containing 'scales'."""
        from auto_round.export.export_to_gguf.conversion.mpt import MPTModel

        obj = _make_mock_model(MPTModel, {
            "num_hidden_layers": 2,
            "max_seq_len": 2048,
            "d_model": 2048,
            "n_heads": 16,
            "attn_config": {"kv_n_heads": 4, "clip_qkv": None, "alibi": False, "alibi_bias_max": 8.0},
        })

        data = torch.randn(2048, 2048)
        # The MPT model replaces "scales" -> "act.scales" in map_tensor_name result
        # Our mock returns the key as-is, so we check it doesn't crash
        results = list(obj.modify_tensors(data, "blk.0.ffn.scales", bid=0))

        assert len(results) == 1

    def test_modify_tensors_regular_weight(self):
        """Test regular weight tensor is mapped normally."""
        from auto_round.export.export_to_gguf.conversion.mpt import MPTModel

        obj = _make_mock_model(MPTModel, {
            "num_hidden_layers": 2,
            "max_seq_len": 2048,
            "d_model": 2048,
            "n_heads": 16,
            "attn_config": {"kv_n_heads": 4, "clip_qkv": None, "alibi": False, "alibi_bias_max": 8.0},
        })

        data = torch.randn(2048, 2048)
        obj.tensor_map.get_name.return_value = "blk.0.attn_q.weight"
        results = list(obj.modify_tensors(data, "blk.0.attn_q.weight", bid=0))

        assert len(results) == 1


# ==============================================================================
# nemotron.py tests
# ==============================================================================

class TestNemotronConversion:
    """Tests for Nemotron conversion module."""

    def test_nemotron_modify_tensors_norm_weight_plus_one(self):
        """Test that norm.weight tensors get +1 added."""
        from auto_round.export.export_to_gguf.conversion.nemotron import NemotronModel

        obj = _make_mock_model(NemotronModel, {
            "num_hidden_layers": 2,
            "vocab_size": 32000,
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "partial_rotary_factor": 0.25,
            "layer_norm_eps": 1e-5,
            "rope_pct": 0.25,
        })

        data = torch.ones(2048) * 0.5
        results = list(obj.modify_tensors(data, "model.layers.0.input_layernorm.weight", bid=0))

        assert len(results) == 1
        assert torch.allclose(results[0][1], torch.ones(2048) * 1.5)

    def test_nemotron_modify_tensors_non_norm_unchanged(self):
        """Test non-norm tensors are passed through unchanged."""
        from auto_round.export.export_to_gguf.conversion.nemotron import NemotronModel

        obj = _make_mock_model(NemotronModel, {
            "num_hidden_layers": 2,
            "vocab_size": 32000,
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "partial_rotary_factor": 0.25,
            "layer_norm_eps": 1e-5,
            "rope_pct": 0.25,
        })

        data = torch.randn(2048, 2048)
        original = data.clone()
        results = list(obj.modify_tensors(data, "model.layers.0.self_attn.q_proj.weight", bid=0))

        assert torch.equal(results[0][1], original)

    def test_nemotron_set_gguf_parameters_rope_scaling_linear(self):
        """Test rope_scaling with LINEAR type."""
        from auto_round.export.export_to_gguf.conversion.nemotron import NemotronModel

        obj = _make_mock_model(NemotronModel, {
            "num_hidden_layers": 2,
            "vocab_size": 32000,
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "partial_rotary_factor": 0.25,
            "layer_norm_eps": 1e-5,
            "rope_pct": 0.25,
            "rope_scaling": {"type": "linear"},
            "factor": 4.0,
        })
        obj.rope_parameters = {"rope_type": "linear", "factor": 4.0}

        obj.set_gguf_parameters()

        obj.gguf_writer.add_rope_scaling_type.assert_called()
        obj.gguf_writer.add_rope_scaling_factor.assert_called_with(4.0)

    def test_nemotron_set_gguf_parameters_rope_scaling_none(self):
        """Test when rope_scaling is None."""
        from auto_round.export.export_to_gguf.conversion.nemotron import NemotronModel

        obj = _make_mock_model(NemotronModel, {
            "num_hidden_layers": 2,
            "vocab_size": 32000,
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "partial_rotary_factor": 0.25,
            "layer_norm_eps": 1e-5,
            "rope_pct": 0.25,
            "rope_scaling": None,
        })

        obj.set_gguf_parameters()

        obj.gguf_writer.add_rope_scaling_type.assert_called()

    def test_nemotron_nanov2_filter_tensors_input_conditioner(self):
        """Test NemotronNanoV2VLModel.filter_tensors skips input_conditioner."""
        from auto_round.export.export_to_gguf.conversion.nemotron import NemotronNanoV2VLModel

        obj = _make_mock_model(NemotronNanoV2VLModel, {
            "vision_config": {"ImageSize": 512},
        })
        obj.hparams_vision = {"patch_size": 14}
        obj.global_config = {"force_image_size": 512, "vision_config": {}}
        obj.preprocessor_config = {}

        # Should skip input_conditioner
        result = obj.filter_tensors(("input_conditioner.some_tensor", lambda: None))
        assert result is None

    def test_nemotron_nanov2_filter_tensors_video_skip(self):
        """Test NemotronNanoV2VLModel.filter_tensors skips video tensors."""
        from auto_round.export.export_to_gguf.conversion.nemotron import NemotronNanoV2VLModel

        obj = _make_mock_model(NemotronNanoV2VLModel, {
            "vision_config": {"ImageSize": 512},
        })
        obj.hparams_vision = {"patch_size": 14}
        obj.global_config = {"force_image_size": 512, "vision_config": {}}
        obj.preprocessor_config = {}

        # Should skip video tensors
        result = obj.filter_tensors((
            "vision_model.radio_model.model.patch_generator.video_embedder.tensor",
            lambda: None
        ))
        assert result is None

    def test_nemotron_nanov2_filter_tensors_passes_vision(self):
        """Test NemotronNanoV2VLModel.filter_tensors passes vision tensors."""
        from auto_round.export.export_to_gguf.conversion.nemotron import NemotronNanoV2VLModel

        obj = _make_mock_model(NemotronNanoV2VLModel, {
            "vision_config": {"ImageSize": 512},
        })
        obj.hparams_vision = {"patch_size": 14}
        obj.global_config = {"force_image_size": 512, "vision_config": {}}
        obj.preprocessor_config = {}

        result = obj.filter_tensors((
            "vision_model.radio_model.model.patch_generator.pos_embed",
            lambda: None
        ))
        assert result is not None


# ==============================================================================
# olmo.py tests
# ==============================================================================

class TestOlmoConversion:
    """Tests for Olmo conversion module."""

    def test_olmo_modify_tensors_q_proj_permute(self):
        """Test OlmoModel permutes q_proj tensor."""
        from auto_round.export.export_to_gguf.conversion.olmo import OlmoModel

        obj = _make_mock_model(OlmoModel, {
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "hidden_size": 2048,
        })

        n_head, hd, hidden = 8, 64, 2048
        tensor = torch.randn(n_head * hd, hidden)
        results = list(obj.modify_tensors(tensor, "blk.0.self_attn.q_proj.weight", bid=0))

        assert len(results) == 1
        assert results[0][1].shape == tensor.shape

    def test_olmo2_set_gguf_parameters_sliding_window(self):
        """Test Olmo2Model adds sliding window pattern."""
        from auto_round.export.export_to_gguf.conversion.olmo import Olmo2Model

        obj = _make_mock_model(Olmo2Model, {
            "num_hidden_layers": 8,
            "hidden_size": 2048,
            "sliding_window": 4096,
            "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        })

        obj.set_gguf_parameters()

        obj.gguf_writer.add_sliding_window.assert_called_with(4096)
        obj.gguf_writer.add_sliding_window_pattern.assert_called()

    def test_olmo2_set_gguf_parameters_no_layer_types(self):
        """Test Olmo2Model with no layer_types defaults to every-4th."""
        from auto_round.export.export_to_gguf.conversion.olmo import Olmo2Model

        obj = _make_mock_model(Olmo2Model, {
            "num_hidden_layers": 8,
            "hidden_size": 2048,
            "sliding_window": 4096,
        })

        obj.set_gguf_parameters()

        obj.gguf_writer.add_sliding_window_pattern.assert_called()
        call_args = obj.gguf_writer.add_sliding_window_pattern.call_args[0][0]
        assert len(call_args) == 8

    def test_olmoe_prepare_tensors_unprocessed_experts_error(self):
        """Test unprocessed experts raise ValueError."""
        from auto_round.export.export_to_gguf.conversion.olmo import OlmoeModel

        obj = _make_mock_model(OlmoeModel, {
            "num_hidden_layers": 2,
            "num_local_experts": 8,
            "hidden_size": 2048,
        })
        obj._experts = [{"unprocessed.tensor": None}]
        obj.tensor_map.mapping = {"tensor": ("KEY", "tensor_name")}

        with patch("auto_round.export.export_to_gguf.conversion.base.ModelBase.prepare_tensors"):
            with pytest.raises(ValueError, match="Unprocessed experts"):
                obj.prepare_tensors()


# ==============================================================================
# openelm.py tests
# ==============================================================================

class TestOpenELMConversion:
    """Tests for OpenELM conversion module."""

    def test_find_hparam_n_layers(self):
        """Test find_hparam returns num_transformer_layers for n_layers key."""
        from auto_round.export.export_to_gguf.conversion.openelm import OpenELMModel

        obj = _make_mock_model(OpenELMModel, {
            "num_transformer_layers": 12,
            "model_dim": 1024,
            "ffn_multipliers": [2.0] * 12,
            "ffn_dim_divisor": 64,
            "num_kv_heads": [2] * 12,
            "num_query_heads": [4] * 12,
            "head_dim": 128,
            "vocab_size": 32000,
            "max_context_length": 2048,
            "rope_freq_constant": 10000.0,
        })
        obj._n_embd = 1024
        obj._num_kv_heads = [2] * 12
        obj._num_query_heads = [4] * 12
        obj._ffn_dims = [2048] * 12

        result = obj.find_hparam(["n_layers"])
        assert result == 12

    def test_modify_tensors_ffn_split(self):
        """Test OpenELMModel splits ffn.proj_1.weight into gate and up."""
        from auto_round.export.export_to_gguf.conversion.openelm import OpenELMModel

        obj = _make_mock_model(OpenELMModel, {
            "num_transformer_layers": 2,
            "model_dim": 1024,
            "ffn_multipliers": [2.0, 2.0],
            "ffn_dim_divisor": 64,
            "num_kv_heads": [2, 2],
            "num_query_heads": [4, 4],
            "head_dim": 128,
            "vocab_size": 32000,
            "max_context_length": 2048,
            "rope_freq_constant": 10000.0,
        })
        obj._n_embd = 1024
        obj._num_kv_heads = [2, 2]
        obj._num_query_heads = [4, 4]
        obj._ffn_dims = [2048, 2048]

        # Use bid=1 to avoid index out of range in ffn_dims access
        tensor = torch.randn(2048, 1024)
        results = list(obj.modify_tensors(tensor, "transformer.layers.1.ffn.proj_1.weight", bid=1))

        # Should yield two tensors: FFN_GATE and FFN_UP
        assert len(results) == 2


# ==============================================================================
# orion.py tests
# ==============================================================================

class TestOrionConversion:
    """Tests for Orion conversion module."""

    def test_set_gguf_parameters_max_sequence_length(self):
        """Test context length from max_sequence_length."""
        from auto_round.export.export_to_gguf.conversion.orion import OrionModel

        obj = _make_mock_model(OrionModel, {
            "num_hidden_layers": 2,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "max_sequence_length": 8192,
            "rms_norm_eps": 1e-6,
        })

        obj.set_gguf_parameters()

        obj.gguf_writer.add_context_length.assert_called_with(8192)

    def test_set_gguf_parameters_max_position_embeddings(self):
        """Test context length from max_position_embeddings."""
        from auto_round.export.export_to_gguf.conversion.orion import OrionModel

        obj = _make_mock_model(OrionModel, {
            "num_hidden_layers": 2,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6,
        })

        obj.set_gguf_parameters()

        obj.gguf_writer.add_context_length.assert_called_with(4096)

    def test_set_gguf_parameters_model_max_length(self):
        """Test context length from model_max_length."""
        from auto_round.export.export_to_gguf.conversion.orion import OrionModel

        obj = _make_mock_model(OrionModel, {
            "num_hidden_layers": 2,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "model_max_length": 16384,
            "rms_norm_eps": 1e-6,
        })

        obj.set_gguf_parameters()

        obj.gguf_writer.add_context_length.assert_called_with(16384)

    def test_set_gguf_parameters_raises_without_ctx_length(self):
        """Test ValueError when no context length parameter is present."""
        from auto_round.export.export_to_gguf.conversion.orion import OrionModel

        obj = _make_mock_model(OrionModel, {
            "num_hidden_layers": 2,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "rms_norm_eps": 1e-6,
        })

        with pytest.raises(ValueError, match="can not find ctx length"):
            obj.set_gguf_parameters()


# ==============================================================================
# pangu.py tests
# ==============================================================================

class TestPanguConversion:
    """Tests for Pangu conversion module."""

    def test_modify_tensors_tied_lm_head(self):
        """Test that tied lm_head.weight is skipped."""
        from auto_round.export.export_to_gguf.conversion.pangu import PanguEmbeddedModel

        obj = _make_mock_model(PanguEmbeddedModel, {
            "num_hidden_layers": 2,
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "head_dim": 64,
            "vocab_size": 43008,
            "tie_word_embeddings": True,
        })

        data = torch.randn(43008, 2048)
        results = list(obj.modify_tensors(data, "lm_head.weight", bid=None))

        assert len(results) == 0

    def test_modify_tensors_untied_lm_head(self):
        """Test that untied lm_head.weight is passed through."""
        from auto_round.export.export_to_gguf.conversion.pangu import PanguEmbeddedModel

        obj = _make_mock_model(PanguEmbeddedModel, {
            "num_hidden_layers": 2,
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "head_dim": 64,
            "vocab_size": 43008,
            "tie_word_embeddings": False,
        })

        data = torch.randn(43008, 2048)
        results = list(obj.modify_tensors(data, "lm_head.weight", bid=None))

        assert len(results) == 1


# ==============================================================================
# plamo.py tests
# ==============================================================================

class TestPlamoConversion:
    """Tests for Plamo conversion module."""

    def test_plamo_shuffle_attn_q_weight(self):
        """Test shuffle_attn_q_weight reshapes and permutes correctly."""
        from auto_round.export.export_to_gguf.conversion.plamo import PlamoModel

        obj = _make_mock_model(PlamoModel, {
            "num_hidden_layers": 2,
            "hidden_size": 5120,
            "intermediate_size": 8192,
            "num_attention_heads": 40,
            "num_key_value_heads": 5,
            "rms_norm_eps": 1e-6,
        })

        data = torch.randn(5120, 5120)
        result = obj.shuffle_attn_q_weight(data)

        assert result.shape == (5120, 5120)
        assert not torch.equal(result, data)

    def test_plamo_shuffle_attn_output_weight(self):
        """Test shuffle_attn_output_weight reshapes and permutes correctly."""
        from auto_round.export.export_to_gguf.conversion.plamo import PlamoModel

        obj = _make_mock_model(PlamoModel, {
            "num_hidden_layers": 2,
            "hidden_size": 5120,
            "intermediate_size": 8192,
            "num_attention_heads": 40,
            "num_key_value_heads": 5,
            "rms_norm_eps": 1e-6,
        })

        data = torch.randn(5120, 5120)
        result = obj.shuffle_attn_output_weight(data)

        assert result.shape == (5120, 5120)
        assert not torch.equal(result, data)

    def test_plamo_modify_tensors_q_weight_shuffle(self):
        """Test attn_q.weight triggers shuffle."""
        from auto_round.export.export_to_gguf.conversion.plamo import PlamoModel

        obj = _make_mock_model(PlamoModel, {
            "num_hidden_layers": 2,
            "hidden_size": 5120,
            "intermediate_size": 8192,
            "num_attention_heads": 40,
            "num_key_value_heads": 5,
            "rms_norm_eps": 1e-6,
        })

        data = torch.randn(5120, 5120)
        results = list(obj.modify_tensors(data, "blk.0.attn_q.weight", bid=0))

        assert len(results) == 1
        assert results[0][1].shape == (5120, 5120)

    def test_plamo2_modify_tensors_A_log_transform(self):
        """Test A_log transformation (negate exp)."""
        from auto_round.export.export_to_gguf.conversion.plamo import Plamo2Model

        obj = _make_mock_model(Plamo2Model, {
            "num_hidden_layers": 2,
            "hidden_size": 4096,
            "intermediate_size": 8192,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "vocab_size": 32000,
            "rms_norm_eps": 1e-6,
            "mamba_step": 2,
            "mamba_enabled": True,
        })
        obj.rope_parameters = {"rope_theta": 10000}

        data = torch.tensor([0.0, 1.0, 2.0])
        results = list(obj.modify_tensors(data, "blk.0.mixer.A_log", bid=0))
        assert torch.equal(results[0][1], -torch.exp(data))

    def test_plamo2_modify_tensors_pre_mixer_norm_plus_one(self):
        """Test pre_mixer_norm.weight gets +1 added."""
        from auto_round.export.export_to_gguf.conversion.plamo import Plamo2Model

        obj = _make_mock_model(Plamo2Model, {
            "num_hidden_layers": 2,
            "hidden_size": 4096,
            "intermediate_size": 8192,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "vocab_size": 32000,
            "rms_norm_eps": 1e-6,
            "mamba_step": 2,
            "mamba_enabled": True,
        })
        obj.rope_parameters = {"rope_theta": 10000}

        data = torch.ones(4096) * 0.5
        results = list(obj.modify_tensors(data, "blk.0.mixer.pre_mixer_norm.weight", bid=0))
        assert torch.allclose(results[0][1], torch.ones(4096) * 1.5)

    def test_plamo2_set_gguf_parameters_mamba_layers(self):
        """Test Plamo2Model with mamba layers sets head counts correctly."""
        from auto_round.export.export_to_gguf.conversion.plamo import Plamo2Model

        obj = _make_mock_model(Plamo2Model, {
            "num_hidden_layers": 8,
            "hidden_size": 4096,
            "intermediate_size": 8192,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "vocab_size": 32000,
            "rms_norm_eps": 1e-6,
            "mamba_step": 2,
            "mamba_enabled": True,
            "hidden_size_per_head": 128,
        })
        obj.rope_parameters = {"rope_theta": 10000}

        obj.set_gguf_parameters()

        obj.gguf_writer.add_head_count_kv.assert_called()
        obj.gguf_writer.add_head_count.assert_called()

    def test_plamo3_modify_tensors_norm_plus_one(self):
        """Test norm.weight gets +1 in Plamo3Model."""
        from auto_round.export.export_to_gguf.conversion.plamo import Plamo3Model

        obj = _make_mock_model(Plamo3Model, {
            "num_hidden_layers": 2,
            "hidden_size": 4096,
            "intermediate_size": 8192,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "vocab_size": 32000,
        })

        data = torch.ones(4096)
        results = list(obj.modify_tensors(data, "blk.0.norm.weight", bid=0))
        assert torch.allclose(results[0][1], torch.ones(4096) * 2.0)


# ==============================================================================
# rwkv.py tests
# ==============================================================================

class TestRWKVConversion:
    """Tests for RWKV conversion module."""

    def test_rwkv6_set_gguf_parameters(self):
        """Test Rwkv6Model.set_gguf_parameters sets all expected parameters."""
        from auto_round.export.export_to_gguf.conversion.rwkv import Rwkv6Model

        obj = _make_mock_model(Rwkv6Model, {
            "num_hidden_layers": 2,
            "head_size": 64,
            "hidden_size": 4096,
            "intermediate_size": 28672,
            "layer_norm_epsilon": 1e-5,
            "rescale_every": 3,
        })

        obj.set_gguf_parameters()

        obj.gguf_writer.add_context_length.assert_called_with(1048576)
        obj.gguf_writer.add_embedding_length.assert_called_with(4096)
        obj.gguf_writer.add_block_count.assert_called()
        obj.gguf_writer.add_head_count.assert_called_with(0)  # RWKV-specific
        obj.gguf_writer.add_wkv_head_size.assert_called_with(64)

    def test_rwkv7_calc_lora_rank(self):
        """Test Rwkv7Model.calc_lora_rank computes correctly."""
        from auto_round.export.export_to_gguf.conversion.rwkv import Rwkv7Model

        obj = _make_mock_model(Rwkv7Model, {
            "num_hidden_layers": 2,
            "head_dim": 64,
            "hidden_size": 4096,
            "intermediate_size": 28672,
            "norm_eps": 1e-5,
        })

        # calc_lora_rank = max(1, round(hidden_size ** exponent * multiplier / 32)) * 32
        result = obj.calc_lora_rank(4096, 0.5, 1.8)
        # 4096 ** 0.5 = 64, 64 * 1.8 = 115.2, /32 = 3.6, round = 4, *32 = 128
        assert result == 128

    def test_rwkv7_filter_tensors_unifies_names(self):
        """Test Rwkv7Model.filter_tensors unifies tensor name patterns."""
        from auto_round.export.export_to_gguf.conversion.rwkv import Rwkv7Model

        obj = _make_mock_model(Rwkv7Model, {
            "num_hidden_layers": 2,
            "head_dim": 64,
            "hidden_size": 4096,
            "intermediate_size": 28672,
            "norm_eps": 1e-5,
        })

        # "blocks" -> "layers", "ffn" -> "feed_forward"
        name, gen = "model.blocks.0.ffn.linear.weight", lambda: None
        result = obj.filter_tensors((name, gen))
        assert result is not None
        assert "layers" in result[0]
        assert "feed_forward" in result[0]

    def test_rwkv7_filter_tensors_self_attn_renaming(self):
        """Test self_attn and attn are renamed to attention."""
        from auto_round.export.export_to_gguf.conversion.rwkv import Rwkv7Model

        obj = _make_mock_model(Rwkv7Model, {
            "num_hidden_layers": 2,
            "head_dim": 64,
            "hidden_size": 4096,
            "intermediate_size": 28672,
            "norm_eps": 1e-5,
        })

        name, gen = "model.layers.0.self_attn.q_proj.weight", lambda: None
        result = obj.filter_tensors((name, gen))
        assert result is not None
        assert "attention" in result[0]


# ==============================================================================
# smallthinker.py tests
# ==============================================================================

class TestSmallThinkerConversion:
    """Tests for SmallThinker conversion module."""

    def test_smallthinker_prepare_tensors_unprocessed_experts_error(self):
        """Test unprocessed experts raise ValueError."""
        from auto_round.export.export_to_gguf.conversion.smallthinker import SmallThinkerModel

        obj = _make_mock_model(SmallThinkerModel, {
            "num_hidden_layers": 2,
            "hidden_size": 2048,
            "moe_num_primary_experts": 8,
        })
        obj._experts = [{"unprocessed.tensor": None}]
        obj.tensor_map.mapping = {"tensor": ("KEY", "tensor_name")}

        with patch("auto_round.export.export_to_gguf.conversion.base.ModelBase.prepare_tensors"):
            with pytest.raises(ValueError, match="Unprocessed experts"):
                obj.prepare_tensors()

    def test_smallthinker_set_gguf_parameters_expert_gating(self):
        """Test expert gating function is set correctly."""
        from auto_round.export.export_to_gguf.conversion.smallthinker import SmallThinkerModel

        obj = _make_mock_model(SmallThinkerModel, {
            "num_hidden_layers": 2,
            "hidden_size": 2048,
            "moe_num_primary_experts": 8,
            "moe_num_active_primary_experts": 2,
            "moe_ffn_hidden_size": 8192,
            "moe_primary_router_apply_softmax": True,
        })

        obj.set_gguf_parameters()

        obj.gguf_writer.add_expert_gating_func.assert_called()


# ==============================================================================
# step3.py tests
# ==============================================================================

class TestStep3Conversion:
    """Tests for Step3 conversion module."""

    def test_step35_filter_tensors_router_bias(self):
        """Test router_bias tensor gets .bias suffix appended."""
        from auto_round.export.export_to_gguf.conversion.step3 import Step35Model

        obj = _make_mock_model(Step35Model, {
            "num_hidden_layers": 2,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_attention_groups": 32,
            "head_dim": 128,
            "sliding_window": 4096,
            "moe_num_experts": 8,
            "moe_top_k": 2,
            "moe_intermediate_size": 14336,
            "share_expert_dim": 2048,
            "rms_norm_eps": 1e-5,
            "layer_types": ["full_attention"] * 2,
            "partial_rotary_factors": [1.0] * 2,
        })

        name, gen = "model.layers.0.moe.router_bias", lambda: None
        result = obj.filter_tensors((name, gen))

        assert result is not None
        assert result[0] == "model.layers.0.moe.router_bias.bias"

    def test_step35_modify_tensors_norm_plus_one(self):
        """Test norm.weight gets +1 added."""
        from auto_round.export.export_to_gguf.conversion.step3 import Step35Model

        obj = _make_mock_model(Step35Model, {
            "num_hidden_layers": 2,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_attention_groups": 32,
            "head_dim": 128,
            "sliding_window": 4096,
            "moe_num_experts": 8,
            "moe_top_k": 2,
            "moe_intermediate_size": 14336,
            "share_expert_dim": 2048,
            "rms_norm_eps": 1e-5,
            "layer_types": ["full_attention"] * 2,
            "partial_rotary_factors": [1.0] * 2,
        })
        obj.rope_parameters = {}

        data = torch.ones(4096) * 0.5
        results = list(obj.modify_tensors(data, "model.layers.0.input_layernorm.weight", bid=0))

        assert torch.allclose(results[0][1], torch.ones(4096) * 1.5)

    def test_step35_generate_extra_tensors_llama3_rope(self):
        """Test generate_extra_tensors yields rope freqs for llama3 rope scaling."""
        from auto_round.export.export_to_gguf.conversion.step3 import Step35Model

        obj = _make_mock_model(Step35Model, {
            "num_hidden_layers": 2,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_attention_groups": 32,
            "head_dim": 128,
            "sliding_window": 4096,
            "moe_num_experts": 8,
            "moe_top_k": 2,
            "moe_intermediate_size": 14336,
            "share_expert_dim": 2048,
            "rms_norm_eps": 1e-5,
            "layer_types": ["full_attention"] * 2,
            "partial_rotary_factors": [1.0] * 2,
            "rope_theta": 10000.0,
        })
        obj.rope_parameters = {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        }

        results = list(obj.generate_extra_tensors())

        assert len(results) == 1
        # The tensor name contains ROPE_FREQS
        assert "ROPE_FREQS" in results[0][0]
        assert results[0][1].dtype == torch.float32

    def test_step35_generate_extra_tensors_non_llama3_returns_empty(self):
        """Test generate_extra_tensors returns empty for non-llama3 rope."""
        from auto_round.export.export_to_gguf.conversion.step3 import Step35Model

        obj = _make_mock_model(Step35Model, {
            "num_hidden_layers": 2,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_attention_groups": 32,
            "head_dim": 128,
            "sliding_window": 4096,
            "moe_num_experts": 8,
            "moe_top_k": 2,
            "moe_intermediate_size": 14336,
            "share_expert_dim": 2048,
            "rms_norm_eps": 1e-5,
            "layer_types": ["full_attention"] * 2,
            "partial_rotary_factors": [1.0] * 2,
            "rope_theta": 10000.0,
        })
        obj.rope_parameters = {
            "rope_type": "linear",
            "factor": 2.0,
        }

        results = list(obj.generate_extra_tensors())
        assert len(results) == 0


# ==============================================================================
# t5.py tests
# ==============================================================================

class TestT5Conversion:
    """Tests for T5 conversion module."""

    def test_modify_tensors_shared_token_first_occurrence(self):
        """Test first shared token embedding is used."""
        from auto_round.export.export_to_gguf.conversion.t5 import T5Model

        obj = _make_mock_model(T5Model, {
            "num_decoder_layers": 2,
            "d_model": 512,
            "d_ff": 2048,
            "num_heads": 8,
            "d_kv": 64,
            "layer_norm_epsilon": 1e-6,
            "relative_attention_num_buckets": 32,
            "decoder_start_token_id": 0,
        })
        obj.shared_token_embeddings_found = False

        data = torch.randn(32000, 512)
        results = list(obj.modify_tensors(data, "decoder.embed_tokens.weight", bid=None))

        assert len(results) == 1
        assert results[0][0] == "shared.weight"
        assert obj.shared_token_embeddings_found is True

    def test_modify_tensors_shared_token_second_occurrence_skipped(self):
        """Test second shared token embedding is skipped."""
        from auto_round.export.export_to_gguf.conversion.t5 import T5Model

        obj = _make_mock_model(T5Model, {
            "num_decoder_layers": 2,
            "d_model": 512,
            "d_ff": 2048,
            "num_heads": 8,
            "d_kv": 64,
            "layer_norm_epsilon": 1e-6,
            "relative_attention_num_buckets": 32,
            "decoder_start_token_id": 0,
        })
        obj.shared_token_embeddings_found = True  # Already found

        data = torch.randn(32000, 512)
        results = list(obj.modify_tensors(data, "encoder.embed_tokens.weight", bid=None))

        assert len(results) == 0


# ==============================================================================
# ultravox.py tests
# ==============================================================================

class TestUltravoxConversion:
    """Tests for Ultravox conversion module."""

    def test_glmasr_filter_tensors_audio_encoder(self):
        """Test audio_encoder.* tensors are renamed correctly."""
        from auto_round.export.export_to_gguf.conversion.ultravox import GlmASRWhisperEncoderModel

        obj = _make_mock_model(GlmASRWhisperEncoderModel, {
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "num_mel_bins": 80,
            "d_model": 1024,
            "encoder_ffn_dim": 4096,
            "encoder_attention_heads": 16,
            "merge_factor": 4,
        })
        obj.global_config = {"merge_factor": 4}
        obj.hparams_vision = {}

        # Whisper prefix should be stripped
        name, gen = "audio_encoder.whisper.layer1.weight", lambda: None
        result = obj.filter_tensors((name, gen))
        assert result is not None
        assert "audio_tower." in result[0]

    def test_glmasr_filter_tensors_skips_lm_tensors(self):
        """Test model.* and lm_head.* tensors are skipped."""
        from auto_round.export.export_to_gguf.conversion.ultravox import GlmASRWhisperEncoderModel

        obj = _make_mock_model(GlmASRWhisperEncoderModel, {
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "num_mel_bins": 80,
            "d_model": 1024,
            "encoder_ffn_dim": 4096,
            "encoder_attention_heads": 16,
            "merge_factor": 4,
        })
        obj.global_config = {"merge_factor": 4}
        obj.hparams_vision = {}

        result = obj.filter_tensors(("model.layers.0.weight", lambda: None))
        assert result is None

        result = obj.filter_tensors(("lm_head.weight", lambda: None))
        assert result is None

    def test_whisper_encoder_tensor_force_quant_conv(self):
        """Test conv weights are forced to F16."""
        from auto_round.export.export_to_gguf.conversion.ultravox import WhisperEncoderModel

        obj = _make_mock_model(WhisperEncoderModel, {
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "num_mel_bins": 80,
            "d_model": 1024,
            "encoder_ffn_dim": 4096,
            "encoder_attention_heads": 16,
        })
        obj.hparams_vision = {}

        result = obj.tensor_force_quant(
            "audio.conv1.weight", "audio.conv1.weight", 0, 4
        )
        from auto_round.export.export_to_gguf.conversion.base import gguf
        assert result == gguf.GGMLQuantizationType.F16


# ==============================================================================
# refact.py tests
# ==============================================================================

class TestRefactConversion:
    """Tests for Refact conversion module."""

    def test_modify_tensors_attn_q(self):
        """Test attn.q.weight is passed through."""
        from auto_round.export.export_to_gguf.conversion.refact import RefactModel

        obj = _make_mock_model(RefactModel, {
            "num_hidden_layers": 2,
            "n_embd": 2048,
            "n_positions": 2048,
            "n_head": 16,
            "layer_norm_epsilon": 1e-5,
        })

        data = torch.randn(2048, 2048)
        results = list(obj.modify_tensors(data, "transformer.h.0.attn.q.weight", bid=0))

        assert len(results) == 1

    def test_modify_tensors_attn_kv(self):
        """Test attn.kv.weight is split into k and v."""
        from auto_round.export.export_to_gguf.conversion.refact import RefactModel

        obj = _make_mock_model(RefactModel, {
            "num_hidden_layers": 2,
            "n_embd": 2048,
            "n_positions": 2048,
            "n_head": 16,
            "layer_norm_epsilon": 1e-5,
        })

        # k and v each have 1 head_dim = n_embd/n_head = 128
        head_dim = 128
        data = torch.randn(head_dim * 2, 2048)  # [k_head + v_head, hidden]
        results = list(obj.modify_tensors(data, "transformer.h.0.attn.kv.weight", bid=0))

        # Should yield 2 results: ATTN_K and ATTN_V
        assert len(results) == 2
        assert "ATTN_K" in results[0][0]
        assert "ATTN_V" in results[1][0]

    def test_modify_tensors_gate_up_proj(self):
        """Test gate_up_proj.weight is split into gate and up."""
        from auto_round.export.export_to_gguf.conversion.refact import RefactModel

        obj = _make_mock_model(RefactModel, {
            "num_hidden_layers": 2,
            "n_embd": 2048,
            "n_positions": 2048,
            "n_head": 16,
            "layer_norm_epsilon": 1e-5,
        })

        hidden_dim = 2048
        inner_dim = 4 * hidden_dim
        hidden_dim_calc = int(2 * inner_dim / 3)
        multiple_of = 256
        ff_dim = multiple_of * ((hidden_dim_calc + multiple_of - 1) // multiple_of)  # 2730

        # gate_up_proj is [ff_dim*2, hidden] = [5460, 2048]
        data = torch.randn(ff_dim * 2, hidden_dim)
        results = list(obj.modify_tensors(data, "transformer.h.0.mlp.gate_up_proj.weight", bid=0))

        # Should yield 2 results: FFN_GATE and FFN_UP
        assert len(results) == 2


# ==============================================================================
# wavtokenizer.py tests
# ==============================================================================

class TestWavTokenizerConversion:
    """Tests for WavTokenizer conversion module."""

    def test_filter_tensors_skips_codebook_tensors(self):
        """Test codebook.* tensors are skipped."""
        from auto_round.export.export_to_gguf.conversion.wavtokenizer import WavTokenizerDecModel

        obj = _make_mock_model(WavTokenizerDecModel, {
            "vocab_size": 1024,
            "n_embd_features": 128,
            "n_ff": 2048,
            "group_norm_epsilon": 1e-5,
            "group_norm_groups": 32,
            "posnet": {"n_embd": 512, "n_layer": 4},
            "convnext": {"n_embd": 256, "n_layer": 3},
        })

        # Should skip codebook tensors
        for suffix in ["codebook.cluster_size", "codebook.embed_avg", "codebook.inited"]:
            result = obj.filter_tensors((f"model.{suffix}", lambda: None))
            assert result is None

    def test_filter_tensors_keeps_other_tensors(self):
        """Test non-codebook tensors are kept."""
        from auto_round.export.export_to_gguf.conversion.wavtokenizer import WavTokenizerDecModel

        obj = _make_mock_model(WavTokenizerDecModel, {
            "vocab_size": 1024,
            "n_embd_features": 128,
            "n_ff": 2048,
            "group_norm_epsilon": 1e-5,
            "group_norm_groups": 32,
            "posnet": {"n_embd": 512, "n_layer": 4},
            "convnext": {"n_embd": 256, "n_layer": 3},
        })

        result = obj.filter_tensors(("model.encoder.conv.weight", lambda: None))
        assert result is not None
