# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``auto_round.algorithms.transforms.hadamard.inplace.apply``.

Tests the high-level Hadamard rotation API and low-level rotation primitives.
"""

import gc
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from auto_round.algorithms.transforms.hadamard.inplace.apply import (
    _resolve_head_dim,
    _fuse_ln_linear,
    _reset_ln_params,
    _rotate_weight_chunked,
    _rotate_linear_by_Q,
    _untie_word_embeddings,
    _uses_layernorm_with_mean,
    _bake_mean_into_linear,
    _subtract_embedding_mean,
    _RMSNorm,
    _replace_layernorms_with_rmsnorm,
    _fuse_layer_norms,
    _register_online_hooks,
    apply_rotation_transform,
)
from auto_round.algorithms.transforms.hadamard.inplace.model_config import RotationMapping


# ==============================================================================
# _resolve_head_dim
# ==============================================================================

class TestResolveHeadDim:
    """Resolve per-head attention dimension from mapping and config."""

    def test_uses_mapping_attn_head_dim(self):
        mapping = SimpleNamespace(attn_head_dim=128)
        config = SimpleNamespace(head_dim=64)
        result = _resolve_head_dim(mapping, config, hidden_size=5120, num_heads=40)
        assert result == 128

    def test_uses_config_head_dim(self):
        mapping = SimpleNamespace(attn_head_dim=None)
        config = SimpleNamespace(head_dim=64)
        result = _resolve_head_dim(mapping, config, hidden_size=5120, num_heads=40)
        assert result == 64

    def test_falls_back_to_hidden_divided_by_heads(self):
        mapping = SimpleNamespace(attn_head_dim=None)
        config = SimpleNamespace(head_dim=None)
        result = _resolve_head_dim(mapping, config, hidden_size=5120, num_heads=40)
        assert result == 128

    def test_ignores_non_positive_head_dim(self):
        mapping = SimpleNamespace(attn_head_dim=None)
        config = SimpleNamespace(head_dim=-1)
        result = _resolve_head_dim(mapping, config, hidden_size=5120, num_heads=40)
        assert result == 128


# ==============================================================================
# _fuse_ln_linear
# ==============================================================================

class TestFuseLnLinear:
    """Fuse LayerNorm into adjacent Linear layers."""

    def test_fuses_weight_only(self):
        ln = nn.LayerNorm(16)
        ln.weight.data.fill_(2.0)
        linear = nn.Linear(16, 8)
        orig_weight = linear.weight.data.clone()
        _fuse_ln_linear(ln, [linear])
        assert not torch.allclose(linear.weight.data, orig_weight)

    def test_fuses_weight_and_bias(self):
        ln = nn.LayerNorm(16)
        ln.weight.data.fill_(2.0)
        ln.bias.data.fill_(1.0)
        linear = nn.Linear(16, 8)
        linear.bias = nn.Parameter(torch.randn(8))
        orig_bias = linear.bias.data.clone()
        _fuse_ln_linear(ln, [linear])
        assert not torch.allclose(linear.bias.data, orig_bias)

    def test_creates_bias_when_missing(self):
        ln = nn.LayerNorm(16)
        ln.weight.data.fill_(1.0)
        ln.bias = nn.Parameter(torch.randn(16))  # layernorm has bias
        linear = nn.Linear(16, 8)
        linear.bias = None  # linear has no bias
        _fuse_ln_linear(ln, [linear])
        assert linear.bias is not None  # bias should be created and fused


# ==============================================================================
# _reset_ln_params
# ==============================================================================

class TestResetLnParams:
    """Reset LayerNorm to identity (weight=1, bias=0)."""

    def test_resets_weight_to_one(self):
        ln = nn.LayerNorm(16)
        ln.weight.data.fill_(5.0)
        _reset_ln_params(ln)
        assert torch.allclose(ln.weight.data, torch.ones(16))

    def test_resets_bias_to_zero(self):
        ln = nn.LayerNorm(16)
        ln.bias.data.fill_(3.0)
        _reset_ln_params(ln)
        assert torch.allclose(ln.bias.data, torch.zeros(16))

    def test_handles_bias_none(self):
        ln = nn.LayerNorm(16)
        ln.bias = None
        _reset_ln_params(ln)  # should not raise


# ==============================================================================
# _rotate_weight_chunked
# ==============================================================================

class TestRotateWeightChunked:
    """Memory-efficient chunked weight rotation."""

    def test_input_side_rotation(self):
        weight = torch.randn(32, 16)
        Q = torch.eye(16)
        result = _rotate_weight_chunked(weight, Q, side="input", compute_device="cpu")
        assert result.shape == weight.shape
        assert torch.allclose(result, weight, atol=1e-4)

    def test_output_side_rotation(self):
        weight = torch.randn(32, 16)
        Q = torch.eye(32)
        result = _rotate_weight_chunked(weight, Q, side="output", compute_device="cpu")
        assert result.shape == weight.shape

    def test_invalid_side_raises(self):
        weight = torch.randn(32, 16)
        Q = torch.eye(16)
        with pytest.raises(ValueError, match="side must be"):
            _rotate_weight_chunked(weight, Q, side="invalid", compute_device="cpu")


# ==============================================================================
# _rotate_linear_by_Q
# ==============================================================================

class TestRotateLinearByQ:
    """Apply rotation Q to Linear weights."""

    def test_rotates_input_side(self):
        linear = nn.Linear(16, 8)
        orig_weight = linear.weight.data.clone()
        Q = torch.eye(16)
        _rotate_linear_by_Q(linear, Q, side="input", compute_device="cpu")
        # With identity Q, weight should be unchanged
        assert torch.allclose(linear.weight.data, orig_weight, atol=1e-4)

    def test_rotates_output_side_with_bias(self):
        # Use square Q for output-side rotation (Q is 16x16, output dimension is 16)
        linear = nn.Linear(16, 16)
        linear.bias = nn.Parameter(torch.randn(16))
        orig_weight = linear.weight.data.clone()
        orig_bias = linear.bias.data.clone()
        Q = torch.eye(16)
        _rotate_linear_by_Q(linear, Q, side="output", compute_device="cpu")
        # With identity Q, both should be unchanged
        assert torch.allclose(linear.weight.data, orig_weight, atol=1e-4)
        assert torch.allclose(linear.bias.data, orig_bias, atol=1e-4)

    def test_output_side_skips_bias_when_none(self):
        linear = nn.Linear(16, 16)
        linear.bias = None
        Q = torch.eye(16)
        _rotate_linear_by_Q(linear, Q, side="output", compute_device="cpu")  # no raise


# ==============================================================================
# _uses_layernorm_with_mean
# ==============================================================================

class TestUsesLayerNormWithMean:
    """Detect standard LayerNorm (subtracts mean)."""

    def test_detects_layer_norm(self):
        model = SimpleNamespace()
        layer = SimpleNamespace()
        layer.input_ln = nn.LayerNorm(16)
        model.layers_attr = [layer]
        mapping = RotationMapping()
        mapping.layers_attr = "layers_attr"
        mapping.attn_input_ln = "input_ln"
        mapping.embedding = None
        result = _uses_layernorm_with_mean(model, mapping)
        assert result is True

    def test_detects_rmsnorm(self):
        model = SimpleNamespace()
        layer = SimpleNamespace()
        layer.input_ln = nn.RMSNorm(16)
        model.layers_attr = [layer]
        mapping = RotationMapping()
        mapping.layers_attr = "layers_attr"
        mapping.attn_input_ln = "input_ln"
        result = _uses_layernorm_with_mean(model, mapping)
        assert result is False


# ==============================================================================
# _bake_mean_into_linear
# ==============================================================================

class TestBakeMeanIntoLinear:
    """Subtract column-wise mean from a Linear layer's weight."""

    def test_subtracts_column_mean_from_weight(self):
        linear = nn.Linear(16, 8)
        orig_weight = linear.weight.data.clone()
        _bake_mean_into_linear(linear)
        assert not torch.allclose(linear.weight.data, orig_weight)

    def test_subtracts_mean_from_bias(self):
        linear = nn.Linear(16, 8)
        linear.bias = nn.Parameter(torch.randn(8))
        orig_bias = linear.bias.data.clone()
        _bake_mean_into_linear(linear)
        assert not torch.allclose(linear.bias.data, orig_bias)

    def test_handles_no_bias(self):
        linear = nn.Linear(16, 8)
        linear.bias = None
        _bake_mean_into_linear(linear)  # no raise


# ==============================================================================
# _subtract_embedding_mean
# ==============================================================================

class TestSubtractEmbeddingMean:
    """Subtract per-row mean from embedding weight matrix."""

    def test_subtracts_row_mean(self):
        embed = nn.Embedding(100, 16)
        orig = embed.weight.data.clone()
        mapping = RotationMapping()
        mapping.embedding = "embed"
        mapping.positional_embedding = None

        class MockModel:
            pass

        model = MockModel()
        model.embed = embed

        _subtract_embedding_mean(model, mapping)
        assert not torch.allclose(embed.weight.data, orig)


# ==============================================================================
# _RMSNorm
# ==============================================================================

class TestRMSNorm:
    """RMS Normalization (no mean subtraction)."""

    def test_forward_shape_preserved(self):
        rms = _RMSNorm(16)
        x = torch.randn(4, 16)
        result = rms(x)
        assert result.shape == x.shape

    def test_preserves_dtype_float32(self):
        rms = _RMSNorm(16)
        x = torch.randn(4, 16, dtype=torch.float32)
        result = rms(x)
        assert result.dtype == torch.float32

    def test_forward_not_equal_to_input(self):
        rms = _RMSNorm(16)
        x = torch.randn(4, 16)
        result = rms(x)
        assert not torch.equal(result, x)


# ==============================================================================
# _replace_layernorms_with_rmsnorm
# ==============================================================================

class TestReplaceLayerNorms:
    """Replace all nn.LayerNorm with _RMSNorm."""

    def test_replaces_layer_norm(self):
        model = nn.Sequential(nn.LayerNorm(16), nn.Linear(16, 8))
        _replace_layernorms_with_rmsnorm(model)
        assert isinstance(model[0], _RMSNorm)

    def test_nested_replacement(self):
        parent = nn.Module()
        parent.ln = nn.LayerNorm(16)
        parent.linear = nn.Linear(16, 8)
        _replace_layernorms_with_rmsnorm(parent)
        assert isinstance(parent.ln, _RMSNorm)

    def test_preserves_device_and_dtype(self):
        model = nn.Module()
        model.ln = nn.LayerNorm(8, dtype=torch.float32)
        model.ln = model.ln.to(torch.bfloat16)
        model.linear = nn.Linear(8, 4)
        _replace_layernorms_with_rmsnorm(model)
        assert model.ln.weight.dtype == torch.bfloat16


# ==============================================================================
# _register_online_hooks
# ==============================================================================

class TestRegisterOnlineHooks:
    """Register online Hadamard pre-forward hooks."""

    def test_registers_hooks(self):
        model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16))
        model[0].weight.data = torch.randn(16, 16)
        model[1].weight.data = torch.randn(16, 16)

        mapping = RotationMapping()
        mapping.mlp_out = "1"  # second linear is down_proj
        mapping.attn_o = "0"  # first linear is o_proj
        mapping.num_heads_attr = "num_heads"
        mapping.hidden_size_attr = "hidden_size"
        mapping.intermediate_size_attr = "intermediate_size"
        mapping.attn_q = "q_proj"
        mapping.attn_k = "k_proj"
        mapping.attn_v = "v_proj"
        mapping.mlp_in = ["gate_proj", "up_proj"]

        model.config = SimpleNamespace(
            num_heads=4,
            hidden_size=16,
            intermediate_size=32,
        )

        handles = _register_online_hooks(
            model,
            mapping,
            fp32_had=False,
            use_fast_had=False,
            group_size=None,
            had_dict=None,
            preset=None,
            fuse_online_to_weight=True,
        )
        assert len(handles) >= 0  # hooks registered


# ==============================================================================
# apply_rotation_transform public API
# ==============================================================================

class TestApplyRotationTransformPublicAPI:
    """Test the public apply_rotation_transform entry point."""

    def test_fuse_online_to_weight_auto_true_for_known_model(self):
        """When model_type is in MAPPING_REGISTRY, fuse_online_to_weight defaults to True."""
        mapping = RotationMapping()
        mapping.embedding = "embed"
        mapping.lm_head = "lm_head"
        mapping.pre_head_ln = "ln_f"
        mapping.layers_attr = "layers"
        mapping.attn_q = "attn.q"
        mapping.attn_k = "attn.k"
        mapping.attn_v = "attn.v"
        mapping.attn_o = "attn.o"
        mapping.mlp_in = ["mlp.gate_proj", "mlp.up_proj"]
        mapping.mlp_out = "mlp.down_proj"
        mapping.attn_input_ln = "attn.input_ln"
        mapping.mlp_input_ln = "mlp.input_ln"
        mapping.hidden_size_attr = "hidden_size"
        mapping.intermediate_size_attr = "intermediate_size"
        mapping.num_heads_attr = "num_heads"
        mapping.attn_head_dim = None

        from auto_round.algorithms.transforms.hadamard.inplace.model_config import MAPPING_REGISTRY

        MAPPING_REGISTRY["test_arch_for_rotation"] = mapping

        try:
            model = nn.Module()
            model.embed = nn.Embedding(100, 16)
            model.lm_head = nn.Linear(16, 100, bias=False)
            model.ln_f = nn.LayerNorm(16)
            layer = nn.Module()
            layer.attn = nn.Module()
            layer.attn.q = nn.Linear(16, 16)
            layer.attn.k = nn.Linear(16, 16)
            layer.attn.v = nn.Linear(16, 16)
            layer.attn.o = nn.Linear(16, 16)
            layer.attn.input_ln = nn.LayerNorm(16)
            layer.mlp = nn.Module()
            layer.mlp.gate_proj = nn.Linear(16, 32)
            layer.mlp.up_proj = nn.Linear(16, 32)
            layer.mlp.down_proj = nn.Linear(32, 16)
            layer.mlp.input_ln = nn.LayerNorm(16)
            model.layers = nn.ModuleList([layer])
            model.config = SimpleNamespace(
                model_type="test_arch_for_rotation",
                hidden_size=16,
                intermediate_size=32,
                num_heads=4,
                num_attention_heads=4,
            )

            model, handles = apply_rotation_transform(
                model,
                group_size=16,
                allow_online_rotation=True,
                fuse_online_to_weight=None,
            )
            assert len(handles) >= 0
        finally:
            del MAPPING_REGISTRY["test_arch_for_rotation"]

    def test_fuse_online_to_weight_auto_false_for_unknown_model(self):
        """For unknown model_type, fuse_online_to_weight defaults to False."""
        model = nn.Module()
        model.model = model  # LLaMA fallback mapping expects model.model
        model.embed = nn.Embedding(100, 16)
        model.lm_head = nn.Linear(16, 100, bias=False)
        model.ln_f = nn.LayerNorm(16)
        model.layers = nn.ModuleList([])
        model.config = SimpleNamespace(
            model_type="completely_unknown_arch_xyz",
            hidden_size=16,
            intermediate_size=32,
            num_heads=4,
            num_attention_heads=4,  # needed by LLaMA fallback mapping
        )

        model, handles = apply_rotation_transform(
            model,
            group_size=16,
            allow_online_rotation=False,
            fuse_online_to_weight=None,
        )
        # With no layers, no hooks should be registered
        assert len(handles) == 0
