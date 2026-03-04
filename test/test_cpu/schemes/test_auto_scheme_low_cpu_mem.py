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
from auto_round.utils.offload import OffloadManager, _clear_module_weights


class TestOffloadManager:
    """Tests for OffloadManager class."""

    def test_context_init_disabled(self):
        """Test that manager is properly initialized when disabled."""
        mgr = OffloadManager(enabled=False)
        assert mgr.enabled is False

    def test_context_init_enabled(self):
        """Test that manager is properly initialized when enabled."""
        mgr = OffloadManager(enabled=True, mode="clean", model_dir="/tmp/fake")
        assert mgr.enabled is True
        assert mgr.model_dir == "/tmp/fake"
        assert mgr.mode == "clean"

    def test_clear_clears_weights(self):
        """Test that _clear clears weights from a module."""
        mgr = OffloadManager(enabled=True, mode="clean")
        module = torch.nn.Linear(4, 4)
        mgr._clear(module)

        # Verify weight was cleared
        assert module.weight.numel() == 0

    def test_add_hooks_noop_when_disabled(self):
        """Test that add_offload_hooks does nothing when disabled."""
        mgr = OffloadManager(enabled=False)
        model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        original_numel = model[0].weight.numel()
        mgr.add_offload_hooks(model, ["0"])
        assert model[0].weight.numel() == original_numel  # unchanged

    def test_ensure_loaded_noop_when_disabled(self):
        """Test that ensure_loaded does nothing when disabled."""
        mgr = OffloadManager(enabled=False)
        model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        mgr.ensure_loaded(model, "0.weight")  # should not raise

    def test_cleanup_resets_state(self):
        """Test that cleanup resets internal state."""
        mgr = OffloadManager(enabled=True, mode="clean")
        module = torch.nn.Linear(4, 4)
        mgr._clear(module)
        mgr._module_names = ["test"]
        assert len(mgr._module_names) == 1

        mgr.cleanup()
        assert len(mgr._module_names) == 0

    def test_reset_clears_tracking(self):
        """Test that reset clears tracking state."""
        mgr = OffloadManager(enabled=True, mode="offload")
        mgr._saved["b"] = {"save_path": "/tmp/x"}
        assert len(mgr._saved) == 1

        mgr.reset()
        assert len(mgr._saved) == 0

    def test_model_dir_property(self):
        """Test model_dir getter/setter."""
        mgr = OffloadManager(enabled=True, mode="clean", model_dir="/path/a")
        assert mgr.model_dir == "/path/a"
        mgr.model_dir = "/path/b"
        assert mgr.model_dir == "/path/b"

    def test_context_manager(self):
        """Test OffloadManager as context manager."""
        with OffloadManager(enabled=True, mode="offload") as mgr:
            assert mgr.enabled is True
        # After exit, state should be cleaned up
        assert mgr._saved == {}

    def test_reload_auto_cleans_temp_file(self):
        """Test that reload() removes the temp file and auto-cleans tempdir."""
        model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        original_weight = model[0].weight.data.clone()
        mgr = OffloadManager(enabled=True, mode="offload")
        mgr.offload(model, "0")
        assert model[0].weight.numel() == 0
        assert mgr._tempdir is not None
        tempdir = mgr._tempdir

        # reload restores weights and auto-cleans temp file + tempdir
        mgr.reload(model, "0")
        assert model[0].weight.numel() > 0
        assert torch.allclose(model[0].weight.data, original_weight)
        assert len(mgr._saved) == 0
        assert mgr._tempdir is None  # auto-cleaned

    def test_reload_all_auto_cleans(self):
        """Test that reload() with no names auto-cleans all temp files."""
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Linear(4, 4),
        )
        mgr = OffloadManager(enabled=True, mode="offload")
        mgr.offload(model, "0")
        mgr.offload(model, "1")
        assert len(mgr._saved) == 2

        mgr.reload(model)
        assert len(mgr._saved) == 0
        assert mgr._tempdir is None  # auto-cleaned

    def test_context_manager_auto_reloads(self):
        """Test that context manager exit auto-reloads offloaded modules."""
        model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        original_weight = model[0].weight.data.clone()
        with OffloadManager(enabled=True, mode="offload") as mgr:
            mgr.offload(model, "0")
            assert model[0].weight.numel() == 0
        # After exiting context, weights should be restored
        assert model[0].weight.numel() > 0
        assert torch.allclose(model[0].weight.data, original_weight)


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


class TestOffloadManagerWithModel:
    """Tests for OffloadManager with actual model blocks."""

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

        mgr = OffloadManager(enabled=True, mode="clean", model_dir=tiny_opt_model_path, cache_numel=True)

        # Get original weights
        original_params = sum(p.numel() for p in block.parameters())
        original_weight = next(block.parameters()).data.clone()

        # Clear
        mgr._clear(block)
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

        mgr = OffloadManager(enabled=True, mode="clean", model_dir=tiny_opt_model_path, cache_numel=True)

        # Save original param counts
        original_counts = {}
        for bn in block_names:
            blk = get_module(model, bn)
            original_counts[bn] = sum(p.numel() for p in blk.parameters())

        # add_offload_hooks clears all blocks and registers hooks
        mgr.add_offload_hooks(model, block_names)
        for bn in block_names:
            blk = get_module(model, bn)
            block_params = sum(p.numel() for p in blk.parameters())
            assert block_params == 0

        # remove_offload_hooks removes hooks and reloads all blocks
        mgr.remove_offload_hooks(model, block_names)
        for bn in block_names:
            blk = get_module(model, bn)
            block_params = sum(p.numel() for p in blk.parameters())
            assert block_params == original_counts[bn]
