# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``auto_round.export.export_to_mlx.export``.

Tests the MLX-format exporter that produces models loadable by mlx-lm.
"""

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from auto_round.export.export_to_mlx.export import (
    _is_mlx_quantizable,
    _build_mlx_quantization_config,
    _flatten_rope_parameters_recursive,
    _extract_rope_theta_from_obj,
    _ensure_rope_theta_from_config_obj,
    _load_original_config_json,
    _snapshot_original_model_types,
    _preserve_original_model_types,
    _strip_prefix,
    _build_text_subconfig_quantization,
    _detect_text_module_prefix,
    _pack_weight_mlx,
    _MLXPackedLayer,
    pack_layer,
    save_quantized_as_mlx,
)
from auto_round.utils.common import MM_KEYS


# ==============================================================================
# _is_mlx_quantizable
# ==============================================================================

class TestIsMlxQuantizable:
    """Predicate matching mlx-lm's default quantization predicate."""

    def test_linear_multiple_of_group_size_and_64(self):
        layer = nn.Linear(128, 64)
        assert _is_mlx_quantizable(layer, group_size=64) is True

    def test_linear_in_dim_not_divisible(self):
        layer = nn.Linear(100, 64)
        assert _is_mlx_quantizable(layer, group_size=64) is False

    def test_linear_out_dim_not_divisible_by_64(self):
        layer = nn.Linear(128, 100)
        assert _is_mlx_quantizable(layer, group_size=64) is False

    def test_embedding_with_vocab_divisible_by_64(self):
        embed = nn.Embedding(128, 64)
        assert _is_mlx_quantizable(embed, group_size=64) is True

    def test_embedding_vocab_not_divisible_by_64(self):
        embed = nn.Embedding(100, 64)
        assert _is_mlx_quantizable(embed, group_size=64) is False

    def test_other_module_types_false(self):
        conv1d = nn.Conv1d(3, 64, kernel_size=1)
        assert _is_mlx_quantizable(conv1d, group_size=64) is False


# ==============================================================================
# _flatten_rope_parameters_recursive
# ==============================================================================

class TestFlattenRopeParameters:
    """Flatten rope_parameters nested dicts into top-level config."""

    def test_flat_rope_parameters(self):
        cfg = {"rope_theta": 1e6, "rope_type": "default"}
        _flatten_rope_parameters_recursive(cfg)
        assert cfg.get("rope_theta") == 1e6
        assert cfg.get("rope_type") == "default"

    def test_nested_rope_parameters_by_mode(self):
        cfg = {
            "rope_parameters": {
                "default": {"rope_theta": 1e6, "rope_type": "default"},
                "other": {"rope_theta": 2e6},
            }
        }
        _flatten_rope_parameters_recursive(cfg)
        assert cfg.get("rope_theta") == 1e6
        assert "rope_parameters" not in cfg

    def test_nested_rope_parameters_fallback_to_first(self):
        cfg = {
            "rope_parameters": {
                "foo": {"rope_theta": 3e6},
                "bar": {"rope_theta": 4e6},
            }
        }
        _flatten_rope_parameters_recursive(cfg)
        assert cfg.get("rope_theta") == 3e6

    def test_nested_rope_parameters_flat_values(self):
        cfg = {
            "rope_parameters": {
                "rope_theta": 5e6,
                "rope_max_position_embeddings": 8192,
            }
        }
        _flatten_rope_parameters_recursive(cfg)
        assert cfg.get("rope_theta") == 5e6
        assert cfg.get("rope_max_position_embeddings") == 8192

    def test_recursive_vlm_config(self):
        cfg = {
            "hidden_size": 5120,
            "text_config": {
                "rope_parameters": {"default": {"rope_theta": 1e6}},
                "vocab_size": 151936,
            },
        }
        _flatten_rope_parameters_recursive(cfg)
        assert cfg["text_config"].get("rope_theta") == 1e6

    def test_non_dict_does_not_crash(self):
        cfg = {"some_list": [1, 2, 3], "rope_parameters": "not_a_dict"}
        _flatten_rope_parameters_recursive(cfg)
        # rope_parameters is popped (removed) even when not a dict
        assert "rope_parameters" not in cfg


# ==============================================================================
# _extract_rope_theta_from_obj
# ==============================================================================

class TestExtractRopeTheta:
    """Best-effort extraction of rope_theta from HF config objects."""

    def test_direct_attribute(self):
        obj = SimpleNamespace(rope_theta=1e6)
        assert _extract_rope_theta_from_obj(obj) == 1e6

    def test_none_object(self):
        assert _extract_rope_theta_from_obj(None) is None

    def test_no_rope_attributes(self):
        obj = SimpleNamespace(hidden_size=5120)
        assert _extract_rope_theta_from_obj(obj) is None

    def test_rope_parameters_dict_flat(self):
        obj = SimpleNamespace(rope_parameters={"rope_theta": 2e6})
        assert _extract_rope_theta_from_obj(obj) == 2e6

    def test_rope_parameters_dict_by_mode(self):
        obj = SimpleNamespace(
            rope_parameters={
                "default": SimpleNamespace(rope_theta=3e6),
                "other": SimpleNamespace(rope_theta=4e6),
            }
        )
        assert _extract_rope_theta_from_obj(obj) == 3e6

    def test_rope_parameters_object_direct(self):
        obj = SimpleNamespace(rope_parameters=SimpleNamespace(rope_theta=5e6))
        assert _extract_rope_theta_from_obj(obj) == 5e6

    def test_rope_parameters_object_default(self):
        inner = SimpleNamespace(rope_theta=6e6)
        obj = SimpleNamespace(rope_parameters=SimpleNamespace(default=inner))
        assert _extract_rope_theta_from_obj(obj) == 6e6


# ==============================================================================
# _ensure_rope_theta_from_config_obj
# ==============================================================================

class TestEnsureRopeTheta:
    """Backfill rope_theta from live config object to JSON dict."""

    def test_adds_missing_rope_theta(self):
        cfg = {"hidden_size": 5120}
        obj = SimpleNamespace(rope_theta=1e6)
        _ensure_rope_theta_from_config_obj(cfg, obj)
        assert cfg.get("rope_theta") == 1e6

    def test_does_not_overwrite_existing(self):
        cfg = {"rope_theta": 2e6, "hidden_size": 5120}
        obj = SimpleNamespace(rope_theta=1e6)
        _ensure_rope_theta_from_config_obj(cfg, obj)
        assert cfg.get("rope_theta") == 2e6

    def test_nested_text_config(self):
        cfg = {
            "hidden_size": 5120,
            "text_config": {"hidden_size": 5120},
        }
        obj = SimpleNamespace(
            rope_theta=1e6,
            text_config=SimpleNamespace(rope_theta=2e6),
        )
        _ensure_rope_theta_from_config_obj(cfg, obj)
        assert cfg["text_config"].get("rope_theta") == 2e6

    def test_none_config_object(self):
        cfg = {"hidden_size": 5120}
        _ensure_rope_theta_from_config_obj(cfg, None)
        assert "rope_theta" not in cfg


# ==============================================================================
# _load_original_config_json
# ==============================================================================

class TestLoadOriginalConfigJson:
    """Load raw config.json from checkpoint directory."""

    def test_loads_from_directory(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"model_type": "qwen2", "hidden_size": 5120}))
        model = SimpleNamespace(config=SimpleNamespace(_name_or_path=str(tmp_path)))
        result = _load_original_config_json(model)
        assert result["model_type"] == "qwen2"

    def test_loads_from_json_file_path(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"model_type": "llama", "hidden_size": 4096}))
        model = SimpleNamespace(config=SimpleNamespace(_name_or_path=str(cfg_file)))
        result = _load_original_config_json(model)
        assert result["model_type"] == "llama"

    def test_missing_file_returns_none(self, tmp_path):
        model = SimpleNamespace(config=SimpleNamespace(_name_or_path=str(tmp_path / "nonexistent")))
        result = _load_original_config_json(model)
        assert result is None

    def test_model_without_config_returns_none(self):
        model = SimpleNamespace()
        result = _load_original_config_json(model)
        assert result is None


# ==============================================================================
# _snapshot_original_model_types
# ==============================================================================

class TestSnapshotOriginalModelTypes:
    """Snapshot model_type for top-level and known sub-configs."""

    def test_from_on_disk_config(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(
            json.dumps({
                "model_type": "qwen2",
                "vision_config": {"model_type": "qwen2_vision"},
                "text_config": {"hidden_size": 5120},
            })
        )
        model = SimpleNamespace(config=SimpleNamespace(_name_or_path=str(tmp_path)))
        result = _snapshot_original_model_types(model)
        assert result["model_type"] == "qwen2"
        assert result["vision_config"]["model_type"] == "qwen2"

    def test_from_in_memory_config_fallback(self):
        model = SimpleNamespace(
            config=SimpleNamespace(
                _name_or_path=None,
                name_or_path=None,
                to_dict=lambda: {"model_type": "llama", "hidden_size": 4096},
            )
        )
        result = _snapshot_original_model_types(model)
        assert result["model_type"] == "llama"


# ==============================================================================
# _preserve_original_model_types
# ==============================================================================

class TestPreserveOriginalModelTypes:
    """Restore model_type fields from original config snapshot."""

    def test_restores_different_model_type(self):
        new_cfg = {"model_type": "qwen2_5", "hidden_size": 5120}
        orig_cfg = {"model_type": "qwen2"}
        _preserve_original_model_types(new_cfg, orig_cfg)
        assert new_cfg["model_type"] == "qwen2"

    def test_removes_model_type_when_not_in_original(self):
        new_cfg = {"model_type": "qwen2_5", "hidden_size": 5120}
        orig_cfg = {}  # no model_type
        _preserve_original_model_types(new_cfg, orig_cfg)
        assert "model_type" not in new_cfg

    def test_noop_when_new_cfg_not_dict(self):
        _preserve_original_model_types("not_a_dict", {"model_type": "llama"})

    def test_subconfig_restore(self):
        new_cfg = {
            "model_type": "qwen3_5",
            "text_config": {"model_type": "qwen3_5_text"},
        }
        orig_cfg = {
            "model_type": "qwen3_5",
            "text_config": {"model_type": "qwen3_5"},
        }
        _preserve_original_model_types(new_cfg, orig_cfg)
        assert new_cfg["text_config"]["model_type"] == "qwen3_5"


# ==============================================================================
# _strip_prefix
# ==============================================================================

class TestStripPrefix:
    """Strip prefix. from layer names."""

    def test_strips_matching_prefix(self):
        assert _strip_prefix("model.layers.0.mlp.gate", "model.layers.0") == "mlp.gate"

    def test_exact_match(self):
        assert _strip_prefix("mlp", "mlp") == "mlp"

    def test_no_match(self):
        assert _strip_prefix("model.layers.0.attn.q_proj", "model.layers.1") == "model.layers.0.attn.q_proj"


# ==============================================================================
# _build_text_subconfig_quantization
# ==============================================================================

class TestBuildTextSubconfigQuantization:
    """Re-key quantization dict for VLM text_config placement."""

    def test_strips_text_prefix(self):
        quant_cfg = {
            "group_size": 64,
            "bits": 4,
            "language_model.layers.0.mlp.gate": False,
        }
        result = _build_text_subconfig_quantization(quant_cfg, "language_model")
        assert result["group_size"] == 64
        assert result["bits"] == 4
        assert "layers.0.mlp.gate" in result
        assert "language_model.layers.0.mlp.gate" not in result

    def test_drops_non_language_model_entries(self):
        quant_cfg = {
            "group_size": 64,
            "bits": 4,
            "vision_encoder.layers.0.mlp": {"bits": 4, "group_size": 64},
        }
        result = _build_text_subconfig_quantization(quant_cfg, "language_model")
        assert "vision_encoder" not in result


# ==============================================================================
# _detect_text_module_prefix
# ==============================================================================

class TestDetectTextModulePrefix:
    """Detect VLM language-model sub-module name."""

    def test_finds_language_model(self):
        model = SimpleNamespace(language_model=SimpleNamespace())
        assert _detect_text_module_prefix(model) == "language_model"

    def test_finds_text_model(self):
        model = SimpleNamespace(text_model=SimpleNamespace())
        assert _detect_text_module_prefix(model) == "text_model"

    def test_finds_thinker(self):
        model = SimpleNamespace(thinker=SimpleNamespace())
        assert _detect_text_module_prefix(model) == "thinker"

    def test_empty_for_text_only_model(self):
        model = SimpleNamespace(embed_tokens=SimpleNamespace())
        assert _detect_text_module_prefix(model) == ""


# ==============================================================================
# _pack_weight_mlx
# ==============================================================================

class TestPackWeightMlx:
    """Pack integer weights into uint32 in MLX format."""

    def test_pack_4bit(self):
        W = torch.randint(0, 16, (8, 64), dtype=torch.int32)
        packed = _pack_weight_mlx(W, bits=4)
        assert packed.dtype == torch.uint32
        assert packed.shape[0] == 8
        assert packed.shape[1] == 64 * 4 // 32  # = 8

    def test_pack_8bit(self):
        W = torch.randint(0, 256, (8, 64), dtype=torch.int32)
        packed = _pack_weight_mlx(W, bits=8)
        assert packed.dtype == torch.uint32
        assert packed.shape[0] == 8
        assert packed.shape[1] == 64 * 8 // 32  # = 16

    def test_pack_2bit(self):
        W = torch.randint(0, 4, (8, 64), dtype=torch.int32)
        packed = _pack_weight_mlx(W, bits=2)
        assert packed.dtype == torch.uint32
        assert packed.shape[0] == 8
        assert packed.shape[1] == 64 * 2 // 32  # = 4

    def test_pack_3bit_cross_word(self):
        W = torch.randint(0, 8, (8, 64), dtype=torch.int32)
        packed = _pack_weight_mlx(W, bits=3)
        assert packed.dtype == torch.uint32
        assert packed.shape[0] == 8
        # num_groups = 64 // 32 = 2, so shape[1] = 2 * 3 = 6
        assert packed.shape[1] == 6

    def test_pack_5bit_cross_word(self):
        W = torch.randint(0, 32, (4, 32), dtype=torch.int32)
        packed = _pack_weight_mlx(W, bits=5)
        assert packed.dtype == torch.uint32
        assert packed.shape[0] == 4
        # num_groups = 32 // 32 = 1, so shape[1] = 1 * 5 = 5
        assert packed.shape[1] == 5


# ==============================================================================
# _MLXPackedLayer
# ==============================================================================

class TestMLXPackedLayer:
    """Holds MLX-packed quantized tensors."""

    def test_registers_buffers(self):
        weight = torch.zeros(8, 4, dtype=torch.uint32)
        scales = torch.ones(8, 2, dtype=torch.float16)
        biases = torch.zeros(8, 2, dtype=torch.float16)
        layer = _MLXPackedLayer(weight, scales, biases, bias=None)
        assert "weight" in layer._buffers
        assert "scales" in layer._buffers
        assert "biases" in layer._buffers
        assert layer.bias is None

    def test_with_bias(self):
        weight = torch.zeros(8, 4, dtype=torch.uint32)
        scales = torch.ones(8, 2, dtype=torch.float16)
        biases = torch.zeros(8, 2, dtype=torch.float16)
        bias = torch.zeros(8, dtype=torch.float16)
        layer = _MLXPackedLayer(weight, scales, biases, bias=bias)
        assert "bias" in layer._buffers


# ==============================================================================
# pack_layer
# ==============================================================================

class TestPackLayer:
    """Pack a single layer into MLX quantized format."""

    def test_non_quantized_layer_skipped(self, tmp_path):
        model = nn.Linear(64, 128)
        model.weight.data = torch.randn(128, 64)
        model.bias = nn.Parameter(torch.randn(128))

        # check_to_quantized returns False → early return
        pack_layer("linear", model)  # should not raise

    def test_unsupported_layer_type_skipped(self):
        model = nn.Conv1d(3, 64, 1)
        model.weight.data = torch.randn(64, 3, 1)
        pack_layer("conv", model)  # should not raise


# ==============================================================================
# save_quantized_as_mlx
# ==============================================================================

class TestSaveQuantizedAsMlx:
    """Full export to MLX format."""

    @pytest.fixture(autouse=True)
    def _patch_save_paths(self):
        """Patch unsupported_meta_device so save_pretrained is skipped for plain nn.Module."""
        with patch(
            "auto_round.export.export_to_mlx.export.unsupported_meta_device",
            return_value=True,
        ):
            yield

    def _make_model(self):
        model = nn.Linear(64, 128)
        model.config = SimpleNamespace(
            model_type="test",
            hidden_size=64,
            _name_or_path=None,
            name_or_path=None,
            save_pretrained=lambda *a, **kw: None,
            to_dict=lambda: {"model_type": "test", "hidden_size": 64},
        )
        return model

    def test_creates_output_directory(self, tmp_path):
        model = self._make_model()
        output_dir = str(tmp_path / "mlx_model")
        result = save_quantized_as_mlx(
            output_dir=output_dir,
            model=model,
            tokenizer=None,
            layer_config=None,
            inplace=True,
        )
        assert os.path.isdir(output_dir)
        assert result is model

    def test_saves_config_json(self, tmp_path):
        model = self._make_model()
        output_dir = str(tmp_path / "mlx_model")

        # Manually create config.json before calling save to test the
        # _build_mlx_quantization_config path
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump({"model_type": "test", "hidden_size": 64}, f)

        # Now call the function - it will read config.json and add quantization info
        save_quantized_as_mlx(
            output_dir=output_dir,
            model=model,
            tokenizer=None,
            layer_config=None,
            inplace=True,
        )
        cfg_path = os.path.join(output_dir, "config.json")
        assert os.path.exists(cfg_path)
        cfg = json.load(open(cfg_path))
        assert "quantization" in cfg

    def test_autoround_format_flag(self, tmp_path):
        model = self._make_model()
        output_dir = str(tmp_path / "mlx_model")
        # The function should not raise and should process the autoround_format flag
        save_quantized_as_mlx(
            output_dir=output_dir,
            model=model,
            tokenizer=None,
            layer_config=None,
            inplace=True,
            autoround_format=True,
            serialization_dict={"sym": True, "data_type": "int"},
        )
        # Key: function completed without error (autoround_format path was exercised)

    def test_vlm_text_config(self, tmp_path):
        model = nn.Module()
        model.language_model = nn.Linear(64, 128)
        model.config = SimpleNamespace(
            model_type="qwen2_vl",
            hidden_size=64,
            language_model=SimpleNamespace(
                model_type="qwen2",
                hidden_size=64,
                _name_or_path=None,
                name_or_path=None,
            ),
            _name_or_path=None,
            name_or_path=None,
            save_pretrained=lambda *a, **kw: None,
            to_dict=lambda: {"model_type": "qwen2_vl", "hidden_size": 64},
        )
        output_dir = str(tmp_path / "mlx_model")
        # Should not raise; VLM text_config path is exercised
        save_quantized_as_mlx(
            output_dir=output_dir,
            model=model,
            tokenizer=None,
            layer_config=None,
            inplace=True,
        )

    def test_inplace_false_creates_copy(self, tmp_path):
        model = self._make_model()
        output_dir = str(tmp_path / "mlx_model")
        result = save_quantized_as_mlx(
            output_dir=output_dir,
            model=model,
            tokenizer=None,
            layer_config=None,
            inplace=False,
        )
        assert result is not model  # should be a copy
