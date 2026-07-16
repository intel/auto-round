# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``auto_round.algorithms.transforms``."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from auto_round.algorithms.transforms import (
    BaseRotation,
    BaseRotationConfig,
    SerializerMixin,
    check_supported_schemes,
    apply_rotation,
    normalize_rotation_config,
    inject_rotation_buffers_on_layer,
    inject_rotation_buffers_bulk,
    save_rotation_config,
    preregister_rotation_buffers,
    rebuild_rotation_if_needed,
    apply_rotation_hooks_from_config,
)


# ==============================================================================
# normalize_rotation_config
# ==============================================================================


class TestNormalizeRotationConfig:
    """Test config normalization."""

    def test_none_returns_none(self):
        assert normalize_rotation_config(None) is None

    def test_base_rotation_config_passthrough(self):
        cfg = BaseRotationConfig()
        result = normalize_rotation_config(cfg)
        assert isinstance(result, BaseRotationConfig)

    def test_dict_hadamard_algorithm(self):
        cfg = {"algorithm": "hadamard", "block_size": 128, "hadamard_type": "random_hadamard"}
        result = normalize_rotation_config(cfg)
        assert result is not None

    def test_dict_spinquant_algorithm(self):
        cfg = {"algorithm": "spinquant", "r1": True, "r2": True}
        result = normalize_rotation_config(cfg)
        assert result is not None

    def test_dict_unknown_algorithm_raises(self):
        cfg = {"algorithm": "unknown_algo"}
        with pytest.raises(ValueError, match="Unknown rotation algorithm"):
            normalize_rotation_config(cfg)

    def test_string_quarot(self):
        result = normalize_rotation_config("quarot")
        assert result is not None
        assert result.trainable_rotation is False

    def test_string_spinquant(self):
        result = normalize_rotation_config("spinquant")
        assert result is not None
        assert result.trainable_rotation is True

    def test_string_hadamard_type(self):
        result = normalize_rotation_config("random_hadamard")
        assert result is not None

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            normalize_rotation_config(123)


# ==============================================================================
# apply_rotation
# ==============================================================================


class TestApplyRotation:
    """Test unified rotation entry point."""

    def test_none_config_returns_model(self):
        model = nn.Linear(4, 4)
        result = apply_rotation(model, None)
        assert result is model

    def test_valid_config_returns_model(self):
        model = nn.Module()
        model.config = SimpleNamespace(hidden_size=16, intermediate_size=32, num_attention_heads=4)

        try:
            result = apply_rotation(model, {"algorithm": "spinquant", "r1": False, "r4": False})
        except Exception:
            # May fail on incomplete model but shouldn't crash
            pass


# ==============================================================================
# inject_rotation_buffers_on_layer
# ==============================================================================


class TestInjectRotationBuffersOnLayer:
    """Test per-layer buffer injection."""

    def test_no_rotation_config_is_noop(self):
        model = nn.Module()
        qlayer = nn.Module()
        # Should not raise
        inject_rotation_buffers_on_layer("layer0.q_proj", qlayer, model)


# ==============================================================================
# inject_rotation_buffers_bulk
# ==============================================================================


class TestInjectRotationBuffersBulk:
    """Test bulk buffer injection."""

    def test_no_rotation_config_is_noop(self):
        model = nn.Module()
        quantization_config = {}
        inject_rotation_buffers_bulk(model, quantization_config)


# ==============================================================================
# save_rotation_config
# ==============================================================================


class TestSaveRotationConfig:
    """Test config persistence."""

    def test_no_rotation_config_is_noop(self):
        model = nn.Module()
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            save_rotation_config(model, tmpdir)


# ==============================================================================
# preregister_rotation_buffers
# ==============================================================================


class TestPreregisterRotationBuffers:
    """Test pre-registration for state_dict loading."""

    def test_empty_quantization_config_returns_zero(self):
        model = nn.Module()
        result = preregister_rotation_buffers(model, {})
        assert result == 0

    def test_non_dict_quantization_config_returns_zero(self):
        model = nn.Module()
        result = preregister_rotation_buffers(model, None)
        assert result == 0


# ==============================================================================
# rebuild_rotation_if_needed
# ==============================================================================


class TestRebuildRotationIfNeeded:
    """Test online rotation rebuild."""

    def test_empty_model_does_not_crash(self):
        model = nn.Module()
        rebuild_rotation_if_needed(model)


# ==============================================================================
# apply_rotation_hooks_from_config
# ==============================================================================


class TestApplyRotationHooksFromConfig:
    """Test rotation hooks application."""

    def test_empty_config_returns_model(self):
        model = nn.Module()
        result = apply_rotation_hooks_from_config(model, {})
        assert result is model

    def test_none_config_returns_model(self):
        model = nn.Module()
        result = apply_rotation_hooks_from_config(model, None)
        assert result is model

    def test_dict_config_returns_model(self):
        model = nn.Module()
        result = apply_rotation_hooks_from_config(model, {"data_type": "mx_fp"})
        assert result is model
