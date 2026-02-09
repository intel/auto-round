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

"""
Unit tests for AutoScheme CPU RAM optimization (low_cpu_mem_usage option).

This tests the disk offload mechanism for AutoScheme that reduces CPU RAM usage
by offloading block weights to disk during gradient computation.
"""

import os
import shutil

import pytest
import torch

from auto_round import AutoRound, AutoScheme
from auto_round.auto_scheme.delta_loss import (
    AutoSchemeOffloadContext,
    _clear_module_weights,
    _group_layers_by_block,
)
from auto_round.auto_scheme.utils import compute_layer_bits
from auto_round.utils import get_block_names, get_module


class TestAutoSchemeOffloadContext:
    """Tests for AutoSchemeOffloadContext class."""

    def test_context_init_disabled(self):
        """Test that context is properly initialized when disabled."""
        ctx = AutoSchemeOffloadContext(low_cpu_mem_usage=False)
        assert ctx.low_cpu_mem_usage is False
        assert ctx._offload_tempdir is None
        assert ctx._offloaded_blocks == {}

    def test_context_init_enabled(self):
        """Test that context is properly initialized when enabled."""
        ctx = AutoSchemeOffloadContext(low_cpu_mem_usage=True)
        assert ctx.low_cpu_mem_usage is True
        assert ctx._offload_tempdir is None
        assert ctx._offloaded_blocks == {}

    def test_init_offload_dir_disabled(self):
        """Test that init_offload_dir returns None when disabled."""
        ctx = AutoSchemeOffloadContext(low_cpu_mem_usage=False)
        result = ctx.init_offload_dir()
        assert result is None
        assert ctx._offload_tempdir is None

    def test_init_offload_dir_enabled(self):
        """Test that init_offload_dir creates temp directory when enabled."""
        ctx = AutoSchemeOffloadContext(low_cpu_mem_usage=True)
        try:
            result = ctx.init_offload_dir()
            assert result is not None
            assert os.path.isdir(result)
            assert "autoscheme_offload_" in result
        finally:
            ctx.cleanup()

    def test_offload_block_weights_disabled(self):
        """Test that offload_block_weights does nothing when disabled."""
        ctx = AutoSchemeOffloadContext(low_cpu_mem_usage=False)
        module = torch.nn.Linear(4, 4)
        ctx.offload_block_weights("test_block", module)
        assert ctx._offloaded_blocks == {}

    def test_offload_block_weights_enabled(self):
        """Test that offload_block_weights saves weights to disk."""
        ctx = AutoSchemeOffloadContext(low_cpu_mem_usage=True)
        try:
            module = torch.nn.Linear(4, 4)
            ctx.offload_block_weights("test_block", module)

            assert "test_block" in ctx._offloaded_blocks
            metadata = ctx._offloaded_blocks["test_block"]
            assert "save_path" in metadata
            assert os.path.exists(metadata["save_path"])
            
            # Verify weight was cleared
            assert module.weight.numel() == 0
        finally:
            ctx.cleanup()

    def test_load_block_weights_disabled(self):
        """Test that load_block_weights does nothing when disabled."""
        ctx = AutoSchemeOffloadContext(low_cpu_mem_usage=False)
        module = torch.nn.Linear(4, 4)
        ctx.load_block_weights("test_block", module)

    def test_offload_and_load_block_weights(self):
        """Test full cycle of offloading and loading block weights."""
        ctx = AutoSchemeOffloadContext(low_cpu_mem_usage=True)
        try:
            module = torch.nn.Linear(4, 4)
            original_weight = module.weight.clone()
            original_bias = module.bias.clone()

            # Offload
            ctx.offload_block_weights("test_block", module)
            assert module.weight.numel() == 0

            # Load back
            ctx.load_block_weights("test_block", module)

            # Verify weights are restored
            assert module.weight.shape == original_weight.shape
            assert torch.allclose(module.weight, original_weight)
            assert module.bias.shape == original_bias.shape
            assert torch.allclose(module.bias, original_bias)
        finally:
            ctx.cleanup()

    def test_offload_block_weights_idempotent(self):
        """Test that offloading same block twice doesn't create duplicate files."""
        ctx = AutoSchemeOffloadContext(low_cpu_mem_usage=True)
        try:
            module = torch.nn.Linear(4, 4)

            ctx.offload_block_weights("test_block", module)
            first_path = ctx._offloaded_blocks["test_block"]["save_path"]

            # Create new module and try to offload again
            module2 = torch.nn.Linear(4, 4)
            ctx.offload_block_weights("test_block", module2)

            # Should still have same path (second offload skipped)
            assert ctx._offloaded_blocks["test_block"]["save_path"] == first_path
        finally:
            ctx.cleanup()

    def test_cleanup(self):
        """Test that cleanup removes temp directory."""
        ctx = AutoSchemeOffloadContext(low_cpu_mem_usage=True)
        ctx.init_offload_dir()
        tempdir = ctx._offload_tempdir
        assert os.path.isdir(tempdir)

        ctx.cleanup()

        assert not os.path.exists(tempdir)
        assert ctx._offload_tempdir is None
        assert ctx._offloaded_blocks == {}

    def test_save_and_load_original_block_weights(self):
        """Test saving and loading original (unwrapped) block weights."""
        ctx = AutoSchemeOffloadContext(low_cpu_mem_usage=True)
        try:
            module = torch.nn.Linear(8, 4)
            original_weight = module.weight.data.clone()

            ctx.save_original_block_weights("block.0", module)
            assert "block.0" in ctx._original_blocks

            # Clear and load back
            _clear_module_weights(module)
            assert module.weight.numel() == 0

            ctx.load_original_block_weights("block.0", module)
            assert module.weight.numel() == original_weight.numel()
            assert torch.allclose(module.weight.data, original_weight)
        finally:
            ctx.cleanup()

    def test_save_original_idempotent(self):
        """Test that saving original block twice doesn't overwrite."""
        ctx = AutoSchemeOffloadContext(low_cpu_mem_usage=True)
        try:
            module = torch.nn.Linear(4, 4)
            ctx.save_original_block_weights("b", module)
            path1 = ctx._original_blocks["b"]["save_path"]

            # Modify module and save again — should be a no-op
            module.weight.data.fill_(999.0)
            ctx.save_original_block_weights("b", module)
            path2 = ctx._original_blocks["b"]["save_path"]
            assert path1 == path2
        finally:
            ctx.cleanup()

    def test_reset_scheme_state(self):
        """Test that reset_scheme_state clears wrapped state but keeps originals."""
        ctx = AutoSchemeOffloadContext(low_cpu_mem_usage=True)
        try:
            module = torch.nn.Linear(4, 4)
            ctx.save_original_block_weights("b", module)
            ctx.offload_block_weights("b", module)

            assert len(ctx._offloaded_blocks) == 1
            assert len(ctx._original_blocks) == 1

            ctx.reset_scheme_state()
            assert len(ctx._offloaded_blocks) == 0
            assert len(ctx._original_blocks) == 1
        finally:
            ctx.cleanup()

    def test_cleanup_removes_both_dirs(self):
        """Test that cleanup removes both original and offload directories."""
        ctx = AutoSchemeOffloadContext(low_cpu_mem_usage=True)
        module = torch.nn.Linear(4, 4)
        ctx.save_original_block_weights("b", module)
        ctx.offload_block_weights("b", module)

        orig_dir = ctx._original_dir
        offload_dir = ctx._offload_tempdir
        assert os.path.isdir(orig_dir)
        assert os.path.isdir(offload_dir)

        ctx.cleanup()
        assert not os.path.exists(orig_dir)
        assert not os.path.exists(offload_dir)


class TestClearModuleWeights:
    """Tests for _clear_module_weights helper function."""

    def test_clear_linear_weights(self):
        """Test clearing weights from a Linear module."""
        module = torch.nn.Linear(16, 8)
        _clear_module_weights(module)

        assert module.weight.numel() == 0
        assert module.bias.numel() == 0

    def test_clear_caches_numel(self):
        """Test that clearing weights caches the original numel."""
        module = torch.nn.Linear(32, 16)
        expected = 32 * 16
        _clear_module_weights(module)
        assert hasattr(module, "_cached_weight_numel")
        assert module._cached_weight_numel == expected

    def test_compute_layer_bits_with_empty_weight(self):
        """Test compute_layer_bits works after weight is cleared."""
        layer = torch.nn.Linear(64, 128)
        layer.bits = 4
        layer.group_size = 32
        layer.data_type = "int"
        layer.sym = True
        layer.super_group_size = None
        layer.super_bits = None

        bits_before, _ = compute_layer_bits(layer, False)
        _clear_module_weights(layer)
        bits_after, _ = compute_layer_bits(layer, False)
        assert bits_before == bits_after

    def test_clear_module_none(self):
        """Test clearing weights with None module doesn't crash."""
        _clear_module_weights(None)  # Should not raise

    def test_clear_module_no_weights(self):
        """Test clearing module without weights doesn't crash."""
        module = torch.nn.ReLU()
        _clear_module_weights(module)  # Should not raise


class TestAutoSchemeDataclassLowCpuMem:
    """Tests for low_cpu_mem_usage parameter in AutoScheme dataclass."""

    def test_auto_scheme_default_low_cpu_mem_usage(self):
        """Test that low_cpu_mem_usage defaults to False."""
        scheme = AutoScheme(avg_bits=4, options="W4A16")
        assert scheme.low_cpu_mem_usage is False

    def test_auto_scheme_low_cpu_mem_usage_enabled(self):
        """Test that low_cpu_mem_usage can be enabled."""
        scheme = AutoScheme(avg_bits=4, options="W4A16", low_cpu_mem_usage=True)
        assert scheme.low_cpu_mem_usage is True

    def test_auto_scheme_low_cpu_mem_usage_with_low_gpu_mem_usage(self):
        """Test that both low_cpu_mem_usage and low_gpu_mem_usage can be set."""
        scheme = AutoScheme(
            avg_bits=4,
            options="W4A16",
            low_cpu_mem_usage=True,
            low_gpu_mem_usage=True,
        )
        assert scheme.low_cpu_mem_usage is True
        assert scheme.low_gpu_mem_usage is True


class TestGroupLayersByBlock:
    """Tests for _group_layers_by_block helper."""

    def test_basic_grouping(self):
        layers = ["model.layers.0.attn.q", "model.layers.0.attn.k",
                  "model.layers.1.mlp.fc1", "lm_head"]
        blocks = ["model.layers.0", "model.layers.1"]
        groups, non_block = _group_layers_by_block(layers, blocks)
        assert groups["model.layers.0"] == ["model.layers.0.attn.q", "model.layers.0.attn.k"]
        assert groups["model.layers.1"] == ["model.layers.1.mlp.fc1"]
        assert non_block == ["lm_head"]

    def test_empty_layers(self):
        groups, non_block = _group_layers_by_block([], ["b0"])
        assert groups == {"b0": []}
        assert non_block == []


class TestAutoSchemeIntegration:
    """Integration tests for AutoScheme with low_cpu_mem_usage."""

    save_dir = "./saved"

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self):
        yield
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_auto_scheme_with_low_cpu_mem_disabled(self, tiny_opt_model_path):
        """Test AutoScheme works normally with low_cpu_mem_usage disabled."""
        model_name = tiny_opt_model_path
        scheme = AutoScheme(
            avg_bits=4,
            options="W2A16,W4A16",
            nsamples=1,
            ignore_scale_zp_bits=True,
            low_cpu_mem_usage=False,
            low_gpu_mem_usage=True,
        )
        ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1)
        _, layer_config = ar.quantize()
        assert layer_config is not None
        assert len(layer_config) > 0

    def test_auto_scheme_with_low_cpu_mem_enabled(self, tiny_opt_model_path):
        """Test AutoScheme works with low_cpu_mem_usage enabled."""
        model_name = tiny_opt_model_path
        scheme = AutoScheme(
            avg_bits=4,
            options="W2A16,W4A16",
            nsamples=1,
            ignore_scale_zp_bits=True,
            low_cpu_mem_usage=True,
            low_gpu_mem_usage=True,
        )
        ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1)
        _, layer_config = ar.quantize()
        assert layer_config is not None
        assert len(layer_config) > 0

    def test_auto_scheme_low_cpu_mem_results_consistent(self, tiny_opt_model_path):
        """Test that results are consistent with and without low_cpu_mem_usage."""
        model_name = tiny_opt_model_path

        # Without low_cpu_mem_usage
        scheme1 = AutoScheme(
            avg_bits=4,
            options="W4A16",
            nsamples=1,
            ignore_scale_zp_bits=True,
            low_cpu_mem_usage=False,
            low_gpu_mem_usage=True,
        )
        ar1 = AutoRound(model=model_name, scheme=scheme1, iters=0, nsamples=1, seed=42)
        _, layer_config1 = ar1.quantize()

        # With low_cpu_mem_usage
        scheme2 = AutoScheme(
            avg_bits=4,
            options="W4A16",
            nsamples=1,
            ignore_scale_zp_bits=True,
            low_cpu_mem_usage=True,
            low_gpu_mem_usage=True,
        )
        ar2 = AutoRound(model=model_name, scheme=scheme2, iters=0, nsamples=1, seed=42)
        _, layer_config2 = ar2.quantize()

        # Layer configs should have same keys
        assert set(layer_config1.keys()) == set(layer_config2.keys())


class TestAutoSchemeOffloadContextWithModel:
    """Tests for AutoSchemeOffloadContext with actual model blocks."""

    def test_offload_model_block(self, tiny_opt_model_path):
        """Test offloading an actual model block."""
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(tiny_opt_model_path, torch_dtype=torch.float32)
        block_names = get_block_names(model)[0]

        if len(block_names) == 0:
            pytest.skip("Model has no blocks")

        block_name = block_names[0]
        block = get_module(model, block_name)

        ctx = AutoSchemeOffloadContext(low_cpu_mem_usage=True)
        try:
            # Get original param count
            original_params = sum(p.numel() for p in block.parameters())

            # Offload
            ctx.offload_block_weights(block_name, block)

            # Check that weights were cleared
            current_params = sum(p.numel() for p in block.parameters())
            assert current_params < original_params

            # Load back
            ctx.load_block_weights(block_name, block)

            # Check params are restored
            restored_params = sum(p.numel() for p in block.parameters())
            assert restored_params == original_params
        finally:
            ctx.cleanup()

    def test_offload_all_blocks(self, tiny_opt_model_path):
        """Test offloading all model blocks."""
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(tiny_opt_model_path, torch_dtype=torch.float32)
        block_names = get_block_names(model)[0]

        if len(block_names) == 0:
            pytest.skip("Model has no blocks")

        ctx = AutoSchemeOffloadContext(low_cpu_mem_usage=True)
        try:
            # Offload all blocks
            ctx.offload_all_blocks(model, block_names)

            # Verify all blocks were offloaded
            for block_name in block_names:
                assert block_name in ctx._offloaded_blocks
                block = get_module(model, block_name)
                # Check weights are cleared (should have very few params)
                block_params = sum(p.numel() for p in block.parameters())
                assert block_params == 0 or block_params < 100  # Allow for some edge cases
        finally:
            ctx.cleanup()

