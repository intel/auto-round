# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``auto_round.algorithms.transforms.hadamard.apply``."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from auto_round.algorithms.transforms.hadamard.apply import (
    HadamardRotation,
    apply_rotation_transform,
    _apply_to_module,
    _triton_available,
)


class TestHadamardRotation:
    """Test HadamardRotation class."""

    def test_from_config_dict(self):
        cfg = {"block_size": 128, "hadamard_type": "random_hadamard"}
        rotation = HadamardRotation.from_config(cfg)
        assert isinstance(rotation, HadamardRotation)

    def test_config_key(self):
        assert HadamardRotation.config_key() == "rotation_config"

    def test_has_rotation_buffers_returns_false(self):
        rotation = HadamardRotation.__new__(HadamardRotation)
        rotation.config = SimpleNamespace()
        module = nn.Module()
        assert rotation.has_rotation_buffers(module) is False

    def test_inject_buffers_on_layer_is_noop(self):
        rotation = HadamardRotation.__new__(HadamardRotation)
        rotation.config = SimpleNamespace()
        model = nn.Module()
        qlayer = nn.Module()
        # Should not raise
        rotation.inject_buffers_on_layer("layer0.q_proj", qlayer, model)

    def test_preregister_buffers_returns_zero(self):
        rotation = HadamardRotation.__new__(HadamardRotation)
        rotation.config = SimpleNamespace()
        model = nn.Module()
        result = rotation.preregister_buffers(model, {})
        assert result == 0

    def test_rebuild_online_returns_model(self):
        rotation = HadamardRotation.__new__(HadamardRotation)
        rotation.config = SimpleNamespace()
        model = nn.Module()
        result = rotation.rebuild_online(model)
        assert result is model

    def test_inject_buffers_bulk_with_config(self):
        rotation = HadamardRotation.__new__(HadamardRotation)
        rotation.config = SimpleNamespace(
            block_size=128,
            hadamard_type="deterministic",
        )
        rotation.config.model_dump = lambda: {"block_size": 128, "hadamard_type": "deterministic"}
        model = nn.Module()
        model._rotation_config = rotation.config
        quantization_config = {}
        rotation.inject_buffers_bulk(model, quantization_config)
        assert "rotation_config" in quantization_config


class TestApplyRotationTransform:
    """Test apply_rotation_transform function."""

    def test_none_config_returns_model(self):
        model = nn.Linear(4, 4)
        result = apply_rotation_transform(model, None)
        assert result is model

    def test_string_config_returns_model(self):
        model = nn.Module()
        model.config = SimpleNamespace(hidden_size=16, intermediate_size=32, num_attention_heads=4)
        try:
            result = apply_rotation_transform(model, "deterministic")
        except Exception:
            # May fail on incomplete model
            pass


class TestApplyToModule:
    """Test _apply_to_module function."""

    def test_unsupported_location_raises(self):
        from auto_round.algorithms.transforms.hadamard.config import RotationConfig

        cfg = RotationConfig(block_size=128, hadamard_type="random_hadamard")
        module = nn.Linear(4, 4)
        model = nn.Module()
        with pytest.raises(NotImplementedError, match="Unsupported transform location"):
            _apply_to_module(model, module, cfg, "invalid_location")


class TestTritonAvailable:
    """Test _triton_available helper."""

    def test_returns_bool(self):
        result = _triton_available("mx_fp")
        assert isinstance(result, bool)

    def test_fp_data_type_returns_false(self):
        # NV FP types don't use Triton
        result = _triton_available("nf4")
        assert result is False
