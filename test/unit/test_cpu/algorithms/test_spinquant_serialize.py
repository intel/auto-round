# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``auto_round.algorithms.transforms.spinquant.serialize``."""

import json
import os
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from auto_round.algorithms.transforms.spinquant import SpinQuantConfig
from auto_round.algorithms.transforms.spinquant.serialize import (
    ROTATION_TYPE_HADAMARD,
    ROTATION_TYPE_RANDOM,
    ROTATION_TYPE_TRAINED,
    _apply_block_rotation_butterfly,
    _apply_rotation_from_buffer,
    _config_to_serializable,
    _get_head_dim,
    _get_hidden_size,
    _get_intermediate_size,
    _get_online_r1_target_names,
    _get_r4_target_names,
    _has_spinquant_buffers,
    _inject_rotation_buffers,
    _is_quantlinear,
    _load_config_from_model,
    _preregister_buffers_on_module,
    preregister_spinquant_buffers,
)

# ==============================================================================
# _is_quantlinear
# ==============================================================================


class TestIsQuantLinear:
    """Detect quantized linear layers."""

    def test_named_quantlinear(self):
        class FakeQuantLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(8, 4))

        assert _is_quantlinear(FakeQuantLinear()) is True

    def test_nvfp4_quantlinear(self):
        class NVFP4QuantLinear(nn.Module):
            pass

        assert _is_quantlinear(NVFP4QuantLinear()) is True

    def test_regular_linear_is_false(self):
        linear = nn.Linear(8, 4)
        assert _is_quantlinear(linear) is False

    def test_other_modules_false(self):
        assert _is_quantlinear(nn.Conv2d(3, 8, 3)) is False


# ==============================================================================
# _has_spinquant_buffers
# ==============================================================================


class TestHasSpinquantBuffers:
    """Detect spinquant buffers on modules."""

    def test_detects_r1_buffer(self):
        module = nn.Module()
        module.register_buffer("spinquant_r1_type", torch.tensor(0))
        assert _has_spinquant_buffers(module) is True

    def test_detects_r4_buffer(self):
        module = nn.Module()
        module.register_buffer("spinquant_r4_type", torch.tensor(0))
        assert _has_spinquant_buffers(module) is True

    def test_false_without_buffers(self):
        module = nn.Module()
        module.register_buffer("weight", torch.randn(8, 4))
        assert _has_spinquant_buffers(module) is False


# ==============================================================================
# _get_online_r1_target_names
# ==============================================================================


class TestGetOnlineR1TargetNames:
    """Find modules that need online R1 rotation."""

    def test_finds_qkv_proj(self):
        model = nn.Module()
        model.layer0 = nn.Module()
        model.layer0.attn = nn.Module()
        model.layer0.attn.q_proj = nn.Linear(16, 16)
        model.layer0.attn.k_proj = nn.Linear(16, 16)
        model.layer0.attn.v_proj = nn.Linear(16, 16)
        model.layer0.attn.o_proj = nn.Linear(16, 16)

        targets = _get_online_r1_target_names(model)
        assert "layer0.attn.q_proj" in targets
        assert "layer0.attn.k_proj" in targets
        assert "layer0.attn.v_proj" in targets
        assert "layer0.attn.o_proj" not in targets

    def test_finds_gate_up_proj(self):
        model = nn.Module()
        model.layer0 = nn.Module()
        model.layer0.mlp = nn.Module()
        model.layer0.mlp.gate_proj = nn.Linear(16, 32)
        model.layer0.mlp.up_proj = nn.Linear(16, 32)
        model.layer0.mlp.down_proj = nn.Linear(32, 16)

        targets = _get_online_r1_target_names(model)
        assert "layer0.mlp.gate_proj" in targets
        assert "layer0.mlp.up_proj" in targets
        assert "layer0.mlp.down_proj" not in targets


# ==============================================================================
# _get_r4_target_names
# ==============================================================================


class TestGetR4TargetNames:
    """Find down_proj layers for R4 rotation."""

    def test_finds_down_proj(self):
        model = nn.Module()
        model.layer0 = nn.Module()
        model.layer0.mlp = nn.Module()
        model.layer0.mlp.down_proj = nn.Linear(32, 16)
        model.layer0.mlp.gate_proj = nn.Linear(16, 32)

        targets = _get_r4_target_names(model)
        assert "layer0.mlp.down_proj" in targets
        assert "layer0.mlp.gate_proj" not in targets


# ==============================================================================
# Architecture extraction helpers
# ==============================================================================


class TestArchitectureExtraction:
    """Extract model architecture info from config."""

    def test_get_hidden_size(self):
        model = SimpleNamespace(config=SimpleNamespace(hidden_size=4096))
        assert _get_hidden_size(model) == 4096

    def test_get_hidden_size_missing(self):
        model = SimpleNamespace(config=SimpleNamespace())
        assert _get_hidden_size(model) == 0

    def test_get_head_dim_direct(self):
        model = SimpleNamespace(config=SimpleNamespace(head_dim=128))
        assert _get_head_dim(model) == 128

    def test_get_head_dim_computed(self):
        model = SimpleNamespace(config=SimpleNamespace(hidden_size=5120, num_attention_heads=40))
        assert _get_head_dim(model) == 128

    def test_get_head_dim_missing(self):
        model = SimpleNamespace(config=SimpleNamespace())
        assert _get_head_dim(model) == 0

    def test_get_intermediate_size(self):
        model = SimpleNamespace(config=SimpleNamespace(intermediate_size=11008))
        assert _get_intermediate_size(model) == 11008

    def test_get_intermediate_size_missing(self):
        model = SimpleNamespace(config=SimpleNamespace())
        assert _get_intermediate_size(model) == 0


# ==============================================================================
# Config serialization / deserialization
# ==============================================================================


class TestConfigSerialization:
    """SpinQuantConfig <-> dict roundtrip."""

    def test_config_to_serializable(self):
        model = SimpleNamespace(config=SimpleNamespace(hidden_size=4096, intermediate_size=11008))
        config = SpinQuantConfig(r1=True, r2=True, r3=False, r4=False)
        result = _config_to_serializable(config, model)
        assert result["r1"] is True
        assert result["r2"] is True
        assert result["r3"] is False
        assert result["r4"] is False
        assert result["hidden_size"] == 4096
        assert result["intermediate_size"] == 11008

    def test_load_config_from_model_dict(self):
        model = SimpleNamespace()
        model.config = SimpleNamespace(quantization_config={"spinquant_config": {"r1": True, "r2": False}})
        loaded = _load_config_from_model(model)
        assert loaded is not None
        assert loaded.r1 is True
        assert loaded.r2 is False

    def test_load_config_from_top_level(self):
        model = SimpleNamespace()
        model.config = SimpleNamespace(spinquant_config={"r1": False, "r2": True})
        loaded = _load_config_from_model(model)
        assert loaded is not None
        assert loaded.r1 is False
        assert loaded.r2 is True

    def test_load_config_missing(self):
        model = SimpleNamespace(config=SimpleNamespace())
        assert _load_config_from_model(model) is None


# ==============================================================================
# Buffer injection
# ==============================================================================


class TestInjectRotationBuffers:
    """Inject rotation buffers into QuantLinear modules."""

    def test_injects_hadamard_type_buffers(self):
        module = nn.Module()
        module.in_features = 16
        module.out_features = 32

        _inject_rotation_buffers(
            module,
            prefix="spinquant_r1",
            rotation_size=16,
            random=False,
            is_trained=False,
            rotation_matrix=None,
        )

        assert hasattr(module, "spinquant_r1_type")
        assert hasattr(module, "spinquant_r1_size")
        assert int(module.spinquant_r1_type) == ROTATION_TYPE_HADAMARD
        assert int(module.spinquant_r1_size) == 16

    def test_injects_random_type_buffers(self):
        module = nn.Module()
        matrix = torch.randint(0, 2, (16, 16)).float() * 2 - 1
        _inject_rotation_buffers(
            module,
            prefix="spinquant_r1",
            rotation_size=16,
            random=True,
            is_trained=False,
            rotation_matrix=matrix,
        )

        assert int(module.spinquant_r1_type) == ROTATION_TYPE_RANDOM
        assert hasattr(module, "spinquant_r1_matrix")
        assert module.spinquant_r1_matrix.dtype == torch.int8

    def test_injects_trained_type_buffers(self):
        module = nn.Module()
        matrix = torch.randn(16, 16)
        _inject_rotation_buffers(
            module,
            prefix="spinquant_r4",
            rotation_size=16,
            random=False,
            is_trained=True,
            rotation_matrix=matrix,
        )

        assert int(module.spinquant_r4_type) == ROTATION_TYPE_TRAINED
        assert module.spinquant_r4_matrix.dtype == torch.float32


# ==============================================================================
# Buffer pre-registration
# ==============================================================================


class TestPreregisterBuffers:
    """Pre-register empty buffers for state_dict loading."""

    def test_preregisters_hadamard_type(self):
        module = nn.Module()

        _preregister_buffers_on_module(
            module,
            prefix="spinquant_r1",
            rotation_size=16,
            needs_matrix=False,
            matrix_dtype=torch.int8,
        )

        assert hasattr(module, "spinquant_r1_type")
        assert hasattr(module, "spinquant_r1_size")
        assert not hasattr(module, "spinquant_r1_matrix")

    def test_preregisters_with_matrix(self):
        module = nn.Module()

        _preregister_buffers_on_module(
            module,
            prefix="spinquant_r4",
            rotation_size=16,
            needs_matrix=True,
            matrix_dtype=torch.int8,
        )

        assert hasattr(module, "spinquant_r4_matrix")
        assert module.spinquant_r4_matrix.shape == (16, 16)

    def test_preregister_spinquant_buffers_integration(self):
        """preregister_spinquant_buffers walks modules and pre-registers."""
        model = nn.Module()
        model.layer0 = nn.Module()
        model.layer0.mlp = nn.Module()
        model.layer0.mlp.down_proj = nn.Module()
        model.layer0.mlp.down_proj.in_features = 32
        model.layer0.mlp.down_proj.out_features = 16
        model.layer0.mlp.down_proj.weight = nn.Parameter(torch.randn(16, 32))

        # Simulate QuantLinear by patching type
        old_type = type(model.layer0.mlp.down_proj)

        class MockQuantLinear(nn.Module):
            pass

        model.layer0.mlp.down_proj.__class__ = MockQuantLinear

        spinquant_config = {
            "r4": True,
            "r1": False,
            "r2": False,
            "online_r1_rotation": False,
            "hidden_size": 16,
            "intermediate_size": 32,
            "rotation_size": None,
        }

        n = preregister_spinquant_buffers(model, spinquant_config)
        assert n >= 1


# ==============================================================================
# Rotation application from buffers
# ==============================================================================


class TestApplyRotationFromBuffer:
    """Apply rotation using buffers stored on QuantLinear."""

    def test_apply_hadamard_deterministic(self):
        module = nn.Module()
        module.register_buffer("spinquant_r1_type", torch.tensor(ROTATION_TYPE_HADAMARD))
        module.register_buffer("spinquant_r1_size", torch.tensor(8))

        x = torch.randn(4, 8)
        result = _apply_rotation_from_buffer(module, x, "spinquant_r1")

        assert result.shape == x.shape
        assert result.dtype == x.dtype

    def test_apply_random_rotation(self):
        module = nn.Module()
        module.register_buffer("spinquant_r4_type", torch.tensor(ROTATION_TYPE_RANDOM))
        module.register_buffer("spinquant_r4_size", torch.tensor(8))
        # ±1 matrix
        sign_matrix = (torch.randint(0, 2, (8, 8)).float() * 2 - 1).to(torch.int8)
        module.register_buffer("spinquant_r4_matrix", sign_matrix)

        x = torch.randn(4, 8)
        result = _apply_rotation_from_buffer(module, x, "spinquant_r4")
        assert result.shape == x.shape

    def test_apply_block_rotation_butterfly(self):
        """Block rotation with butterfly algorithm."""
        from auto_round.algorithms.transforms.spinquant.rotation_utils import deterministic_hadamard_matrix

        had_K = deterministic_hadamard_matrix(8)
        x = torch.randn(4, 8)
        result = _apply_block_rotation_butterfly(x, had_K, 1, 8)
        assert result.shape == x.shape

    def test_apply_block_rotation_with_block_size(self):
        """Block rotation with smaller block size."""
        from auto_round.algorithms.transforms.spinquant.rotation_utils import deterministic_hadamard_matrix

        x = torch.randn(4, 32)
        # Use a valid had_K (16x16 power-of-2 matrix)
        had_K = deterministic_hadamard_matrix(16)
        result = _apply_block_rotation_butterfly(x, had_K, 1, 16)
        assert result.shape == x.shape
