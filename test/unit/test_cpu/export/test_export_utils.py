# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``auto_round.export.utils``."""

import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from auto_round.export.utils import (
    _resolve_model_source_dir,
    _resolve_pipeline_source_dir,
    _save_model_configs,
    _state_dict_has_meta_tensor,
    filter_quantization_config,
    get_autogptq_packing_qlinear,
    is_immediate_saving_mode,
    is_local_pipeline_model_dir,
    is_pipeline_model_dir,
    is_remote_pipeline_model_dir,
    release_layer_safely,
    resolve_pipeline_export_layout,
    save_model,
    save_pretrained_artifact,
)

# ==============================================================================
# save_pretrained_artifact
# ==============================================================================


class TestSavePretrainedArtifact:
    """Test save_pretrained_artifact function."""

    def test_none_output_dir_returns_false(self):
        artifact = MagicMock()
        result = save_pretrained_artifact(artifact, None)
        assert result is False

    def test_none_artifact_returns_false(self):
        result = save_pretrained_artifact(None, "/tmp/test")
        assert result is False

    def test_no_save_pretrained_method_returns_false(self):
        artifact = "not_callable_object"
        result = save_pretrained_artifact(artifact, "/tmp/test")
        assert result is False

    def test_valid_artifact_saves(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = MagicMock()
            result = save_pretrained_artifact(artifact, tmpdir, "test_artifact")
            assert result is True
            artifact.save_pretrained.assert_called_once_with(tmpdir)


# ==============================================================================
# _save_model_configs
# ==============================================================================


class TestSaveModelConfigs:
    """Test _save_model_configs function."""

    def test_no_config_attribute_does_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = nn.Module()
            _save_model_configs(model, tmpdir)
            # No exception means success

    def test_config_is_none_does_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = nn.Module()
            model.config = None
            _save_model_configs(model, tmpdir)
            # No exception means success

    def test_saves_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MagicMock()
            model = nn.Module()
            model.config = config
            model.generation_config = None
            _save_model_configs(model, tmpdir)
            config.save_pretrained.assert_called_once_with(tmpdir)

    def test_saves_generation_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MagicMock()
            gen_config = MagicMock()
            model = nn.Module()
            model.config = config
            model.generation_config = gen_config
            _save_model_configs(model, tmpdir)
            config.save_pretrained.assert_called_once_with(tmpdir)
            gen_config.save_pretrained.assert_called_once_with(tmpdir)

    def test_fallback_on_save_pretrained_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MagicMock()
            config.save_pretrained.side_effect = KeyError("bad_key")
            config.to_json_string.return_value = '{"model_type": "test"}'
            model = nn.Module()
            model.config = config
            model.generation_config = None
            _save_model_configs(model, tmpdir)
            # Should fall back to writing json
            assert os.path.exists(os.path.join(tmpdir, "config.json"))


# ==============================================================================
# _state_dict_has_meta_tensor
# ==============================================================================


class TestStateDictHasMetaTensor:
    """Test _state_dict_has_meta_tensor function."""

    def test_no_meta_tensor(self):
        m = nn.Linear(4, 4)
        assert _state_dict_has_meta_tensor(m) is False

    def test_with_meta_tensor(self):
        # Create a model on meta device
        with torch.device("meta"):
            m = nn.Linear(4, 4)
        assert _state_dict_has_meta_tensor(m) is True


# ==============================================================================
# is_immediate_saving_mode
# ==============================================================================


class TestIsImmediateSavingMode:
    """Test is_immediate_saving_mode function."""

    def test_returns_false_for_normal_model(self):
        m = nn.Linear(4, 4)
        assert is_immediate_saving_mode(m) is False

    def test_meta_tensor_returns_true(self):
        with torch.device("meta"):
            m = nn.Linear(4, 4)
        assert is_immediate_saving_mode(m) is True

    def test_with_serialization_dict(self):
        m = nn.Linear(4, 4)
        assert is_immediate_saving_mode(m, {"some_key": "some_value"}) is False


# ==============================================================================
# is_local_pipeline_model_dir
# ==============================================================================


class TestIsLocalPipelineModelDir:
    """Test is_local_pipeline_model_dir function."""

    def test_empty_dir_returns_false(self):
        assert is_local_pipeline_model_dir("") is False

    def test_none_dir_returns_false(self):
        assert is_local_pipeline_model_dir(None) is False

    def test_non_existent_dir_returns_false(self):
        assert is_local_pipeline_model_dir("/non/existent/path") is False

    def test_dir_without_model_index_returns_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert is_local_pipeline_model_dir(tmpdir) is False

    def test_dir_with_model_index_returns_true(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "model_index.json"), "w").close()
            assert is_local_pipeline_model_dir(tmpdir) is True


# ==============================================================================
# is_remote_pipeline_model_dir
# ==============================================================================


class TestIsRemotePipelineModelDir:
    """Test is_remote_pipeline_model_dir function."""

    def test_local_dir_returns_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert is_remote_pipeline_model_dir(tmpdir) is False

    def test_non_string_returns_false(self):
        assert is_remote_pipeline_model_dir(None) is False

    def test_remote_dir_with_model_index(self):
        with patch("huggingface_hub.list_repo_files", return_value=["model_index.json", "config.json"]):
            assert is_remote_pipeline_model_dir("some/repo") is True

    def test_remote_dir_without_model_index(self):
        with patch("huggingface_hub.list_repo_files", return_value=["config.json"]):
            assert is_remote_pipeline_model_dir("some/repo") is False


# ==============================================================================
# is_pipeline_model_dir
# ==============================================================================


class TestIsPipelineModelDir:
    """Test is_pipeline_model_dir function."""

    def test_local_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "model_index.json"), "w").close()
            assert is_pipeline_model_dir(tmpdir) is True

    def test_empty(self):
        assert is_pipeline_model_dir("") is False


# ==============================================================================
# _resolve_pipeline_source_dir
# ==============================================================================


class TestResolvePipelineSourceDir:
    """Test _resolve_pipeline_source_dir function."""

    def test_no_source(self):
        model = nn.Module()
        assert _resolve_pipeline_source_dir(model) is None

    def test_with_local_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "model_index.json"), "w").close()
            model = nn.Module()
            model.name_or_path = tmpdir
            assert _resolve_pipeline_source_dir(model) == tmpdir

    def test_with_config_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "model_index.json"), "w").close()
            model = nn.Module()
            model.config = SimpleNamespace(_name_or_path=tmpdir)
            assert _resolve_pipeline_source_dir(model) == tmpdir


# ==============================================================================
# _resolve_model_source_dir
# ==============================================================================


class TestResolveModelSourceDir:
    """Test _resolve_model_source_dir function."""

    def test_no_source(self):
        model = nn.Module()
        assert _resolve_model_source_dir(model) is None

    def test_with_name_or_path(self):
        model = nn.Module()
        model.name_or_path = "/some/path"
        assert _resolve_model_source_dir(model) == "/some/path"

    def test_with_config_name_or_path(self):
        model = nn.Module()
        model.config = SimpleNamespace(_name_or_path="/from/config")
        assert _resolve_model_source_dir(model) == "/from/config"

    def test_with_config_name(self):
        model = nn.Module()
        model.config = SimpleNamespace(name_or_path="/from/config/name")
        assert _resolve_model_source_dir(model) == "/from/config/name"


# ==============================================================================
# resolve_pipeline_export_layout
# ==============================================================================


class TestResolvePipelineExportLayout:
    """Test resolve_pipeline_export_layout function."""

    def test_no_subfolder_returns_same_dir(self):
        model = nn.Module()
        out_dir = "/tmp/out"
        model_out, proc_out, is_pipeline = resolve_pipeline_export_layout(model, out_dir)
        assert model_out == out_dir
        assert proc_out == out_dir
        assert is_pipeline is False

    def test_with_subfolder_no_source(self):
        model = nn.Module()
        model._autoround_pipeline_subfolder = "transformer"
        out_dir = "/tmp/out"
        model_out, proc_out, is_pipeline = resolve_pipeline_export_layout(model, out_dir)
        assert model_out == os.path.join(out_dir, "transformer")
        assert proc_out == out_dir
        assert is_pipeline is True


# ==============================================================================
# save_model
# ==============================================================================


class TestSaveModel:
    """Test save_model function."""

    def test_immediate_saving(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = nn.Linear(4, 4)
            save_model(model, tmpdir, immediate_saving=True)
            # Should not raise

    def test_normal_saving(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MagicMock()
            model.dtype = torch.float32
            model.config = MagicMock()
            model.config.quantization_config = None
            with patch("auto_round.export.utils._resolve_model_source_dir", return_value=None):
                save_model(model, tmpdir, safe_serialization=False)
                # Should have called save_pretrained
                model.save_pretrained.assert_called()

    def test_dtype_change_updates_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MagicMock()
            model.dtype = torch.float32
            model.config = MagicMock()
            model.config.quantization_config = None
            # Pre-create config.json
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"torch_dtype": "float32", "dtype": "float32"}, f)
            with patch("auto_round.export.utils._resolve_model_source_dir", return_value=None):
                save_model(model, tmpdir, dtype=torch.bfloat16, safe_serialization=False)
            # Check dtype was updated
            with open(config_path, "r") as f:
                data = json.load(f)
            assert data["torch_dtype"] == "bfloat16"


# ==============================================================================
# get_autogptq_packing_qlinear
# ==============================================================================


class TestGetAutogptqPackingQlinear:
    """Test get_autogptq_packing_qlinear function."""

    def test_returns_quant_linear(self):
        from auto_round_extension.torch.qlinear_torch_zp import QuantLinear

        result = get_autogptq_packing_qlinear("cuda", bits=4)
        assert result is QuantLinear


# ==============================================================================
# filter_quantization_config
# ==============================================================================


class TestFilterQuantizationConfig:
    """Test filter_quantization_config function."""

    def test_basic_filtering(self):
        cfg = {"amp": True, "batch_size": 8, "data_type": int, "custom_key": "value"}
        result = filter_quantization_config(cfg)
        # Defaults should be removed
        assert "amp" not in result
        assert "custom_key" in result

    def test_none_values_removed(self):
        cfg = {"amp": None, "custom": "value"}
        filter_quantization_config(cfg)
        assert "amp" not in cfg

    def test_act_bits_handling(self):
        cfg = {"act_bits": 16, "act_data_type": "fp8", "custom": "value"}
        result = filter_quantization_config(cfg)
        assert "act_bits" not in result
        assert "act_data_type" not in result
        assert "custom" in result

    def test_empty_lists_removed(self):
        cfg = {"supported_types": [], "custom": "value"}
        result = filter_quantization_config(cfg)
        assert "supported_types" not in result

    def test_iters_based_lr(self):
        cfg = {"iters": 100, "custom": "value"}
        result = filter_quantization_config(cfg)
        # Iters with lr may or may not be in result based on defaults
        # The function modifies in place so check
        assert "custom" in result


# ==============================================================================
# release_layer_safely
# ==============================================================================


class TestReleaseLayerSafely:
    """Test release_layer_safely function."""

    def test_releases_weight_and_bias(self):
        layer = nn.Linear(4, 4)
        weight = layer.weight
        bias = layer.bias
        release_layer_safely(layer)
        assert layer.weight is None
        assert layer.bias is None

    def test_handles_missing_attrs(self):
        layer = nn.Module()
        # Should not raise
        release_layer_safely(layer)

    def test_handles_none_attrs(self):
        layer = nn.Module()
        layer.weight = None
        layer.bias = None
        release_layer_safely(layer)
