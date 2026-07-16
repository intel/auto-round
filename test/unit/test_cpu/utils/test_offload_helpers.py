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
"""Tests for ``auto_round/utils/offload.py``.

CPU-friendly: tests the small helpers (``_flatten_names``,
``OffloadManager.__init__``, ``OffloadManager.has``, ``estimate_module_size_gb``,
``_clear_module_weights``, ``_load_state_dict_into_module``) on a tiny model.
The full save/reload path requires a real model checkpoint and is covered
elsewhere.
"""

import os
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# _flatten_names
# ---------------------------------------------------------------------------
class TestFlattenNames:
    def test_flat_already(self):
        from auto_round.utils.offload import OffloadManager

        assert OffloadManager._flatten_names(["a", "b", "c"]) == ["a", "b", "c"]

    def test_nested_one_level(self):
        from auto_round.utils.offload import OffloadManager

        result = OffloadManager._flatten_names([["a", "b"], "c", ["d"]])
        assert result == ["a", "b", "c", "d"]

    def test_empty_input(self):
        from auto_round.utils.offload import OffloadManager

        assert OffloadManager._flatten_names([]) == []

    def test_deeply_nested(self):
        from auto_round.utils.offload import OffloadManager

        result = OffloadManager._flatten_names([["a", ["b", "c"]], "d"])
        # Inner ["b", "c"] is itself a list, so it gets treated as a single item;
        # the helper only handles one level of nesting
        assert result == ["a", ["b", "c"], "d"]


# ---------------------------------------------------------------------------
# OffloadManager.__init__
# ---------------------------------------------------------------------------
class TestOffloadManagerInit:
    def test_default_construction(self):
        from auto_round.utils.offload import OffloadManager

        mgr = OffloadManager()
        assert mgr.mode == "offload"
        assert mgr.enabled is True
        assert mgr._saved == {}
        assert mgr._tempdir is None

    def test_disabled_construction(self):
        from auto_round.utils.offload import OffloadManager

        mgr = OffloadManager(enabled=False)
        assert mgr.enabled is False

    def test_clean_mode(self):
        from auto_round.utils.offload import OffloadManager

        mgr = OffloadManager(mode="clean", model_dir="/some/dir")
        assert mgr.mode == "clean"
        assert mgr.model_dir == "/some/dir"

    def test_cache_numel_flag(self):
        from auto_round.utils.offload import OffloadManager

        mgr = OffloadManager(cache_numel=True)
        assert mgr.cache_numel is True


# ---------------------------------------------------------------------------
# OffloadManager.has / reset
# ---------------------------------------------------------------------------
class TestOffloadManagerStateQueries:
    def test_has_returns_false_initially(self):
        from auto_round.utils.offload import OffloadManager

        mgr = OffloadManager()
        assert mgr.has("any.module") is False

    def test_has_offload_mode_returns_true_after_save(self):
        from auto_round.utils.offload import OffloadManager

        mgr = OffloadManager(mode="offload")
        mgr._saved["model.layer"] = {"save_path": "/tmp/foo"}
        assert mgr.has("model.layer") is True
        assert mgr.has("model.other") is False

    def test_has_clean_mode_always_false(self):
        from auto_round.utils.offload import OffloadManager

        mgr = OffloadManager(mode="clean")
        mgr._saved["model.layer"] = {"save_path": "/tmp/foo"}  # pretend
        # In clean mode, has() always returns False
        assert mgr.has("model.layer") is False

    def test_reset_clears_saved(self):
        from auto_round.utils.offload import OffloadManager

        mgr = OffloadManager(mode="offload")
        mgr._saved["a"] = {"save_path": "/tmp/a"}
        mgr._current_loaded = "a"
        mgr._last_loaded = "a"
        mgr.reset()
        assert mgr._saved == {}
        assert mgr._current_loaded is None
        assert mgr._last_loaded is None


# ---------------------------------------------------------------------------
# estimate_module_size_gb
# ---------------------------------------------------------------------------
class TestEstimateModuleSize:
    def test_empty_module(self):
        from auto_round.utils.offload import OffloadManager

        m = nn.Module()
        size = OffloadManager.estimate_module_size_gb(m)
        assert size == 0.0

    def test_small_module(self):
        from auto_round.utils.offload import OffloadManager

        m = nn.Linear(10, 10)  # 110 fp32 params
        size = OffloadManager.estimate_module_size_gb(m)
        # 110 * 4 bytes = 440 bytes; in GB = 440 / 1024^3
        assert size > 0
        assert size < 1e-6  # way less than a GB


# ---------------------------------------------------------------------------
# _clear_module_weights
# ---------------------------------------------------------------------------
class TestClearModuleWeights:
    def test_clear_sets_weight_to_empty(self):
        from auto_round.utils.offload import _clear_module_weights

        layer = nn.Linear(4, 4)
        assert layer.weight.numel() == 16
        _clear_module_weights(layer)
        assert layer.weight.numel() == 0

    def test_clear_caches_numel_and_shape(self):
        from auto_round.utils.offload import _clear_module_weights

        layer = nn.Linear(4, 4)
        _clear_module_weights(layer, cache_numel=True)
        assert layer._cached_weight_numel == 16
        assert layer._cached_weight_shape == (4, 4)

    def test_clear_none_is_noop(self):
        from auto_round.utils.offload import _clear_module_weights

        # Should not raise
        _clear_module_weights(None)

    def test_clear_skips_orig_layer(self):
        from auto_round.utils.offload import _clear_module_weights

        class _Wrapper(nn.Module):
            pass

        inner = nn.Linear(4, 4)
        wrapper = _Wrapper()
        wrapper.orig_layer = inner
        # Should skip clearing when orig_layer is set
        _clear_module_weights(wrapper)
        # inner.weight should remain intact
        assert inner.weight.numel() == 16

    def test_clear_with_restorable_filter(self):
        from auto_round.utils.offload import _clear_module_weights

        layer = nn.Linear(4, 4)
        # Layer has both weight and bias; only clear weight
        _clear_module_weights(layer, restorable_params={"weight"})
        assert layer.weight.numel() == 0
        # bias should remain
        assert layer.bias.numel() == 4

    def test_clear_with_restorable_excluding_weight(self):
        from auto_round.utils.offload import _clear_module_weights

        layer = nn.Linear(4, 4)
        # Restorable set does NOT include weight -> weight must remain
        _clear_module_weights(layer, restorable_params={"bias"})
        assert layer.weight.numel() == 16
        assert layer.bias.numel() == 0


# ---------------------------------------------------------------------------
# _load_state_dict_into_module
# ---------------------------------------------------------------------------
class TestLoadStateDictIntoModule:
    def test_restores_linear_weight(self):
        from auto_round.utils.offload import (
            _clear_module_weights,
            _load_state_dict_into_module,
        )

        layer = nn.Linear(4, 4)
        saved = {"weight": layer.weight.detach().clone(), "bias": layer.bias.detach().clone()}

        _clear_module_weights(layer)
        assert layer.weight.numel() == 0

        _load_state_dict_into_module(saved, layer)
        # Weight should be restored
        assert layer.weight.numel() == 16
        assert torch.allclose(layer.weight.data, saved["weight"])

    def test_skips_missing_submodule(self):
        """If a nested attribute is missing, the loader must silently skip."""
        from auto_round.utils.offload import _load_state_dict_into_module

        # state_dict has a key whose intermediate path doesn't exist
        state_dict = {"nonexistent_sub.weight": torch.zeros(4, 4)}
        # Should not raise
        _load_state_dict_into_module(state_dict, nn.Linear(4, 4))


# ---------------------------------------------------------------------------
# _resolve_model_dir
# ---------------------------------------------------------------------------
class TestResolveModelDir:
    def test_existing_directory_returned_as_is(self, tmp_path):
        from auto_round.utils.offload import _resolve_model_dir

        assert _resolve_model_dir(str(tmp_path)) == str(tmp_path)

    def test_nonexistent_path_falls_through(self, tmp_path):
        """If snapshot_download fails, the original input is returned."""
        from auto_round.utils.offload import _resolve_model_dir

        with patch(
            "huggingface_hub.snapshot_download",
            side_effect=Exception("not on hub"),
        ):
            result = _resolve_model_dir(str(tmp_path / "missing"))
            assert result == str(tmp_path / "missing")
