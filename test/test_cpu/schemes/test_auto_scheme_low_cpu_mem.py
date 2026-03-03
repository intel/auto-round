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

import shutil

import pytest
import torch

from auto_round import AutoRound, AutoScheme
from auto_round.auto_scheme.utils import compute_layer_bits
from auto_round.utils import get_block_names, get_module
from auto_round.utils.offload import AutoSchemeOffloadContext
from auto_round.utils.offload import _clear_submodule_weights as _clear_module_weights
from auto_round.utils.offload import group_layers_by_block


class TestAutoSchemeOffloadContext:
    """Tests for AutoSchemeOffloadContext class."""

    def test_context_init_disabled(self):
        """Test that context is properly initialized when disabled."""
        ctx = AutoSchemeOffloadContext(low_cpu_mem_usage=False)
        assert ctx.low_cpu_mem_usage is False
        assert len(ctx._cleared_blocks) == 0

    def test_context_init_enabled(self):
        """Test that context is properly initialized when enabled."""
        ctx = AutoSchemeOffloadContext(low_cpu_mem_usage=True, model_dir="/tmp/fake")
        assert ctx.low_cpu_mem_usage is True
        assert ctx.model_dir == "/tmp/fake"
        assert len(ctx._cleared_blocks) == 0

    def test_clear_block_clears_weights(self):
        """Test that _clear_block clears weights from a module."""
        ctx = AutoSchemeOffloadContext(low_cpu_mem_usage=True)
        module = torch.nn.Linear(4, 4)
        ctx._clear_block(module)

        # Verify weight was cleared
        assert module.weight.numel() == 0

    def test_attach_noop_when_disabled(self):
        """Test that attach does nothing when disabled."""
        ctx = AutoSchemeOffloadContext(low_cpu_mem_usage=False)
        model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        original_numel = model[0].weight.numel()
        ctx.attach(model, ["0"])
        assert model[0].weight.numel() == original_numel  # unchanged

    def test_ensure_block_noop_when_disabled(self):
        """Test that ensure_block_for_layer does nothing when disabled."""
        ctx = AutoSchemeOffloadContext(low_cpu_mem_usage=False)
        model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        ctx.ensure_block_for_layer(model, "0.weight")  # should not raise

    def test_cleanup_resets_tracking(self):
        """Test that cleanup resets internal tracking state."""
        ctx = AutoSchemeOffloadContext(low_cpu_mem_usage=True)
        module = torch.nn.Linear(4, 4)
        ctx._clear_block(module)
        ctx._cleared_blocks.add("test")
        assert len(ctx._cleared_blocks) == 1

        ctx.cleanup()
        assert len(ctx._cleared_blocks) == 0

    def test_reset_scheme_state(self):
        """Test that reset_scheme_state clears tracking."""
        ctx = AutoSchemeOffloadContext(low_cpu_mem_usage=True)
        ctx._cleared_blocks.add("b")
        assert len(ctx._cleared_blocks) == 1

        ctx.reset_scheme_state()
        assert len(ctx._cleared_blocks) == 0

    def test_model_dir_property(self):
        """Test model_dir getter/setter."""
        ctx = AutoSchemeOffloadContext(low_cpu_mem_usage=True, model_dir="/path/a")
        assert ctx.model_dir == "/path/a"
        ctx.model_dir = "/path/b"
        assert ctx.model_dir == "/path/b"


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
        _clear_module_weights(module, cache_numel=True)
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
        _clear_module_weights(layer, cache_numel=True)
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
        """Test that low_cpu_mem_usage defaults to True."""
        scheme = AutoScheme(avg_bits=4, options="W4A16")
        assert scheme.low_cpu_mem_usage is True

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
    """Tests for group_layers_by_block helper."""

    def test_basic_grouping(self):
        layers = ["model.layers.0.attn.q", "model.layers.0.attn.k", "model.layers.1.mlp.fc1", "lm_head"]
        blocks = ["model.layers.0", "model.layers.1"]
        groups, non_block = group_layers_by_block(layers, blocks)
        assert groups["model.layers.0"] == ["model.layers.0.attn.q", "model.layers.0.attn.k"]
        assert groups["model.layers.1"] == ["model.layers.1.mlp.fc1"]
        assert non_block == ["lm_head"]

    def test_empty_layers(self):
        groups, non_block = group_layers_by_block([], ["b0"])
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

    def test_clear_and_load_model_block(self, tiny_opt_model_path):
        """Test clearing and reloading an actual model block from checkpoint files."""
        from transformers import AutoModelForCausalLM

        from auto_round.utils.offload import load_block_from_model_files

        model = AutoModelForCausalLM.from_pretrained(tiny_opt_model_path, torch_dtype=torch.float32)
        block_names = get_block_names(model)[0]

        if len(block_names) == 0:
            pytest.skip("Model has no blocks")

        block_name = block_names[0]
        block = get_module(model, block_name)

        ctx = AutoSchemeOffloadContext(low_cpu_mem_usage=True, model_dir=tiny_opt_model_path)

        # Get original weights
        original_params = sum(p.numel() for p in block.parameters())
        original_weight = next(block.parameters()).data.clone()

        # Clear
        ctx._clear_block(block)
        current_params = sum(p.numel() for p in block.parameters())
        assert current_params < original_params

        # Load back from model files
        load_block_from_model_files(tiny_opt_model_path, block_name, block)
        restored_params = sum(p.numel() for p in block.parameters())
        assert restored_params == original_params

        # Verify values match
        restored_weight = next(block.parameters()).data
        assert torch.allclose(restored_weight, original_weight)

    def test_attach_and_detach(self, tiny_opt_model_path):
        """Test clearing all model blocks via attach and loading them back via detach."""
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(tiny_opt_model_path, torch_dtype=torch.float32)
        block_names = get_block_names(model)[0]

        if len(block_names) == 0:
            pytest.skip("Model has no blocks")

        ctx = AutoSchemeOffloadContext(low_cpu_mem_usage=True, model_dir=tiny_opt_model_path)

        # Save original param counts
        original_counts = {}
        for bn in block_names:
            blk = get_module(model, bn)
            original_counts[bn] = sum(p.numel() for p in blk.parameters())

        # Attach clears all blocks and registers hooks
        ctx.attach(model, block_names)
        for bn in block_names:
            blk = get_module(model, bn)
            block_params = sum(p.numel() for p in blk.parameters())
            assert block_params == 0

        # Detach removes hooks and reloads all blocks
        ctx.detach(model, block_names)
        for bn in block_names:
            blk = get_module(model, bn)
            block_params = sum(p.numel() for p in blk.parameters())
            assert block_params == original_counts[bn]
