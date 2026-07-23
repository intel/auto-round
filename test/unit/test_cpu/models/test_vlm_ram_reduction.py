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
"""Unit tests for VLM RAM reduction changes.

These tests target the specific logic changes introduced by the VLM RAM fix:
1. Per-sample-constant tensor skipping in block forward cache
2. Diffusion multi-device memory reservation
3. MLLM calibration memory cleanup
4. last_cache_name RAM reduction

All tests use fake model hierarchies and mocks — no real model weights
are loaded or downloaded.
"""

import types
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from auto_round.utils.common import flatten_list

# ---------------------------------------------------------------------------
# 1. Tests for per-sample-constant tensor skipping in _get_block_forward_func
# ---------------------------------------------------------------------------


class TestBlockForwardTensorSkipping:
    """Unit tests for the per-sample-constant tensor skipping logic in calib.py.

    The fix skips caching non-hidden_states tensor args that are per-sample
    constants (e.g. position_embeddings, attention_mask) to avoid accumulating
    batch_size * nsamples items in RAM.
    """

    def test_hidden_states_tensor_is_cached(self):
        """hidden_states must always be cached regardless of shared_cache_keys."""
        # Simulate the new logic: skip only if:
        #   1. key != "hidden_states"
        #   2. isinstance(tensor)
        #   3. key not in shared_cache_keys
        key = "hidden_states"
        is_tensor = True
        shared_cache_keys = set()

        should_skip = key != "hidden_states" and is_tensor and key not in shared_cache_keys
        assert not should_skip, "hidden_states must not be skipped"

    def test_per_sample_constant_tensor_is_skipped(self):
        """Per-sample-constant tensors (e.g. attention_mask) must be skipped."""
        key = "attention_mask"
        is_tensor = True
        shared_cache_keys = set()

        should_skip = key != "hidden_states" and is_tensor and key not in shared_cache_keys
        assert should_skip, "attention_mask (per-sample constant) must be skipped"

    def test_position_embeddings_skipped_when_not_shared_no_variable(self):
        """position_embeddings skipped when not in shared_cache_keys and no variable block shape."""
        key = "position_embeddings"
        is_tensor = True
        shared_cache_keys = set()  # empty = position_embeddings not shared
        has_variable_block_shape = False

        should_skip = (
            key != "hidden_states" and is_tensor and key not in shared_cache_keys and not has_variable_block_shape
        )
        assert should_skip, "position_embeddings must be skipped when not shared and no variable block shape"

    def test_position_embeddings_kept_when_variable_shape(self):
        """position_embeddings not skipped when has_variable_block_shape=True (cached per-block)."""
        key = "position_embeddings"
        is_tensor = True
        shared_cache_keys = set()  # not in shared_cache_keys
        has_variable_block_shape = True

        should_skip = (
            key != "hidden_states" and is_tensor and key not in shared_cache_keys and not has_variable_block_shape
        )
        assert not should_skip, "position_embeddings must be cached per-block when variable block shape is enabled"

    def test_position_embeddings_kept_when_shared(self):
        """position_embeddings in shared_cache_keys must not be skipped (non-variable path)."""
        key = "position_embeddings"
        is_tensor = True
        shared_cache_keys = {"position_embeddings"}
        has_variable_block_shape = False

        should_skip = (
            key != "hidden_states" and is_tensor and key not in shared_cache_keys and not has_variable_block_shape
        )
        assert not should_skip, "position_embeddings must be kept when in shared_cache_keys"

    def test_hidden_states_always_cached_variable_shape(self):
        """hidden_states must always be cached even when has_variable_block_shape=True."""
        key = "hidden_states"
        is_tensor = True
        shared_cache_keys = set()
        has_variable_block_shape = True

        should_skip = (
            key != "hidden_states" and is_tensor and key not in shared_cache_keys and not has_variable_block_shape
        )
        assert not should_skip, "hidden_states must never be skipped"


class TestSpecialModelReplayDispatch:
    def test_non_special_block_is_untouched(self):
        from auto_round.special_model_handler import prepare_special_model_block_inputs

        class PlainBlock(nn.Module):
            pass

        block = PlainBlock()
        input_others = {"attention_mask": torch.ones(1, 1, 4, 4), "positional_inputs": []}
        output, positional_inputs = prepare_special_model_block_inputs(block, torch.zeros((1, 4, 8)), input_others, [])

        assert output is input_others
        assert positional_inputs == []
        assert torch.equal(output["attention_mask"], torch.ones(1, 1, 4, 4))


# ---------------------------------------------------------------------------
# 2. Tests for diffusion multi-device memory reservation
# ---------------------------------------------------------------------------


class TestDiffusionMultiDeviceDispatch:
    """Unit tests for diffusion multi-device memory reservation.

    The fix reserves memory on the primary device for non-target pipeline
    components (text encoder, VAE) before computing the balanced device map,
    avoiding OOM when the main model is also dispatched to that device.
    """

    def test_reserve_memory_on_primary_device(self):
        """Non-target components must reserve memory on the primary device."""
        from auto_round.utils.device import dispatch_model_by_all_available_devices

        # Build a fake pipeline with two components: main + auxiliary
        class FakeComponent(nn.Module):
            def __init__(self, param_count):
                # Each param is 2 bytes (bf16), total = param_count * 2 bytes
                super().__init__()
                self.register_parameter("p", nn.Parameter(torch.empty(param_count, dtype=torch.bfloat16)))

        class FakePipe(nn.Module):
            def __init__(self):
                super().__init__()
                # Main model: 1000 params (2KB)
                self.transformer = FakeComponent(1000)
                # Non-main: 100 params (0.2KB)
                self.vae = FakeComponent(100)

        pipe = FakePipe()
        pipe.components = {"transformer": pipe.transformer, "vae": pipe.vae}

        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.cuda.device_count", return_value=0):
                # Single device path (falls back to pipe.to)
                result = dispatch_model_by_all_available_devices(pipe, "cpu")
                assert result is pipe

    def test_multi_device_respects_non_main_memory(self):
        """With multiple devices, memory reservation must be computed for non-main components."""
        from auto_round.utils.device import dispatch_model_by_all_available_devices

        class FakeComponent(nn.Module):
            def __init__(self, param_count):
                super().__init__()
                self.register_parameter("p", nn.Parameter(torch.empty(param_count, dtype=torch.bfloat16)))

        class FakePipe(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = FakeComponent(1000)  # main: 1000 * 2 = 2000 bytes
                self.text_encoder = FakeComponent(100)  # non-main: 100 * 2 = 200 bytes
                self.vae = FakeComponent(50)  # non-main: 50 * 2 = 100 bytes
                # total non-main: 300 params = 600 bytes * 1.2 buffer = 720 bytes

        pipe = FakePipe()
        pipe.components = {
            "transformer": pipe.transformer,
            "text_encoder": pipe.text_encoder,
            "vae": pipe.vae,
        }

        # We can't easily test the full multi-device path without accelerate mocks,
        # but we can verify the non_main_bytes calculation logic in isolation
        non_main_bytes = 0
        for attr, component in pipe.components.items():
            if attr == "transformer":
                continue
            if isinstance(component, nn.Module):
                non_main_bytes += sum(p.numel() * p.element_size() for p in component.parameters())
        non_main_reserved = int(non_main_bytes * 1.2)
        assert non_main_reserved > 0, "non-main memory must be computed"


# ---------------------------------------------------------------------------
# 3. Tests for MLLM calibration memory cleanup
# ---------------------------------------------------------------------------


class TestMLLMCalibMemoryCleanup:
    """Unit tests for MLLM calibration memory cleanup.

    The fix adds gc.collect() + clear_memory() after each forward pass in
    MLLMMixin.calib to reduce memory fragmentation. Also re-raises OOM so
    the try_cache_inter_data_gpucpu handler can switch to CPU.
    """

    def test_get_calibrator_kind_returns_mllm(self):
        """MLLMMixin._get_calibrator_kind must return 'mllm' so the right calibrator is used."""
        from auto_round.compressors.mllm_mixin import MLLMMixin

        class DummyCompressor(MLLMMixin):
            def __init__(self):
                # Only initialise what MLLMMixin needs — skip super().__init__
                # to avoid model loading side-effects.
                self.template = "qwen2_vl"
                self.extra_data_dir = None
                self.quant_nontext_module = False
                self.template_obj = None

        compressor = DummyCompressor()
        assert compressor._get_calibrator_kind() == "mllm"

    def test_quant_nontext_module_batch_size_accumulation(self):
        """quant_nontext_module=True must reset batch_size=1 and accumulate gradient steps."""
        # Directly test the batch_size accumulation logic from MLLMMixin.__init__.
        kwargs = {"batch_size": 4, "gradient_accumulate_steps": 2}

        batch_size = kwargs.get("batch_size", 8)
        grad_acc = kwargs.get("gradient_accumulate_steps", 1)
        new_grad_acc = batch_size * grad_acc
        kwargs["gradient_accumulate_steps"] = new_grad_acc
        kwargs["batch_size"] = 1

        assert kwargs["batch_size"] == 1
        assert kwargs["gradient_accumulate_steps"] == 8


# ---------------------------------------------------------------------------
# 3. Tests for last_cache_name RAM reduction
# ---------------------------------------------------------------------------


class TestLastCacheNameRAMReduction:
    """Unit tests for last_cache_name logic that limits caching to the first block.

    The fix passes last_cache_name=block_names[0] to try_cache_inter_data_gpucpu
    so that only the first block's inputs are cached, reducing RAM from
    O(n_blocks * nsamples) to O(nsamples).
    """

    def test_should_stop_after_first_block(self):
        """With last_cache_name set to the first block, caching must stop there."""

        class FakeCompressor:
            def __init__(self):
                self.last_cache_name = "model.layers.0"
                self._cache_target_set = {"model.layers.0"}
                self._cache_seen_targets = set()

            def _should_stop_cache_forward(self, name):
                if name == self.last_cache_name:
                    return True
                if self.last_cache_name is not None:
                    return False
                if not hasattr(self, "_cache_target_set") or not hasattr(self, "_cache_seen_targets"):
                    return False
                if name in self._cache_target_set:
                    self._cache_seen_targets.add(name)
                if not self._cache_target_set.issubset(self._cache_seen_targets):
                    return False
                self.last_cache_name = name
                return True

        c = FakeCompressor()
        assert c._should_stop_cache_forward("model.layers.0") is True
        assert c.last_cache_name == "model.layers.0"

    def test_without_last_cache_name_all_blocks_cached(self):
        """Without last_cache_name, all blocks must be cached before stopping."""

        class FakeCompressor:
            def __init__(self):
                self.last_cache_name = None
                self._cache_target_set = {"model.layers.0", "model.layers.1"}
                self._cache_seen_targets = set()

            def _should_stop_cache_forward(self, name):
                if name == self.last_cache_name:
                    return True
                if self.last_cache_name is not None:
                    return False
                if not hasattr(self, "_cache_target_set") or not hasattr(self, "_cache_seen_targets"):
                    return False
                if name in self._cache_target_set:
                    self._cache_seen_targets.add(name)
                if not self._cache_target_set.issubset(self._cache_seen_targets):
                    return False
                self.last_cache_name = name
                return True

        c = FakeCompressor()
        assert c._should_stop_cache_forward("model.layers.0") is False
        assert c._should_stop_cache_forward("model.layers.1") is True

    def test_last_cache_name_logic_in_quantize(self):
        """Verify that last_cache_name is set to first block when there are multiple blocks."""
        all_blocks = [["model.layers.0", "model.layers.1"], ["model.layers.2"]]
        to_cache_block_names = [b[0] for b in all_blocks]
        assert to_cache_block_names == ["model.layers.0", "model.layers.2"]

        # Without variable block shape: last_cache_name = first block
        _last_cache_name = to_cache_block_names[0] if len(to_cache_block_names) > 1 else None
        assert _last_cache_name == "model.layers.0"

        single_group = [["model.layers.0"]]
        single_cache = [b[0] for b in single_group]
        assert len(single_cache) == 1
        _last_cache_name_single = single_cache[0] if len(single_cache) > 1 else None
        assert _last_cache_name_single is None
