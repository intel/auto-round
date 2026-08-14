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

import os
import queue
import tempfile

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")

from auto_round.auto_scheme.delta_loss import (
    _ProgressQueueProxy,
    _assign_scheme_worker_devices,
    _autoscheme_cache_config,
    _autoscheme_cache_key,
    _autoscheme_cache_path,
    _can_parallel_scheme_scoring,
    _drain_progress_queue,
    _find_compatible_autoscheme_cache,
    _get_next_scheme_bits,
    _get_scheme_bits,
    _get_scheme_worker_count,
    _is_per_op_cache_compatible,
    _load_autoscheme_scores,
    _merge_worker_memory_reports,
    _save_autoscheme_scores,
    _scheme_repr,
    _stable_model_id,
    choose_bits_per_layer_with_path,
    move_module_to_tuning_device,
)

# ==============================================================================
# choose_bits_per_layer_with_path (DP)
# ==============================================================================


class TestChooseBitsPerLayerWithPath:
    def test_min_loss_solution(self):
        layers = {
            "a": [("W4A16", 4, 1.0, ["a"]), ("W3A16", 3, 2.0, ["a"])],
            "b": [("W4A16", 4, 1.5, ["b"]), ("W8A16", 8, 0.5, ["b"])],
        }
        best_loss, path = choose_bits_per_layer_with_path(layers, P=12)
        assert best_loss == 1.5
        assert path == [(["a"], "W4A16"), (["b"], "W8A16")]

    def test_infeasible_returns_none(self):
        layers = {"a": [("W4A16", 8, 1.0, ["a"])]}
        assert choose_bits_per_layer_with_path(layers, P=4) == (None, None)

    def test_max_states_one(self):
        layers = {
            "a": [("W4A16", 4, 1.0, ["a"]), ("W3A16", 3, 2.0, ["a"])],
            "b": [("W4A16", 4, 1.5, ["b"]), ("W8A16", 8, 0.5, ["b"])],
        }
        best_loss, path = choose_bits_per_layer_with_path(layers, P=12, max_states=1)
        assert best_loss == 1.5
        assert len(path) == 2

    def test_single_layer(self):
        layers = {"a": [("W4A16", 4, 1.0, ["a"]), ("W8A16", 8, 0.1, ["a"])]}
        best_loss, path = choose_bits_per_layer_with_path(layers, P=8)
        assert best_loss == 0.1
        assert path == [(["a"], "W8A16")]


# ==============================================================================
# scheme-bit helpers
# ==============================================================================


class TestSchemeBits:
    def test_get_bits_string(self):
        assert _get_scheme_bits("W4A16") == 4
        assert _get_scheme_bits("INT8") == 8

    def test_get_bits_dict(self):
        assert _get_scheme_bits({"bits": 6}) == 6
        assert _get_scheme_bits({"data_type": "int"}) == 16

    def test_next_scheme_bits(self):
        assert _get_next_scheme_bits(["W4A16", "W8A16", "W3A16"], [0, 1, 2], floor_bits=4) == 8

    def test_next_scheme_bits_none(self):
        assert _get_next_scheme_bits(["W4A16"], [0], floor_bits=4) is None


class TestSchemeRepr:
    def test_preset_name_normalized(self):
        r = _scheme_repr("W4A16")
        assert isinstance(r, dict)
        assert r["bits"] == 4

    def test_unknown_string_uppercased(self):
        assert _scheme_repr("nope") == "NOPE"

    def test_dict_filters_none(self):
        assert _scheme_repr({"bits": 4, "sym": None}) == {"bits": 4}

    def test_other_types_str(self):
        assert _scheme_repr(123) == "123"


class TestStableModelId:
    def test_hub_id_basename(self):
        assert _stable_model_id("Qwen/Qwen3-0.6B/") == "Qwen3-0.6B"

    def test_local_path_basename(self):
        assert _stable_model_id("/models/opt-125m/") == "opt-125m"

    def test_non_string_passthrough(self):
        assert _stable_model_id(None) is None


# ==============================================================================
# autoscheme cache
# ==============================================================================


def _cache_config():
    return _autoscheme_cache_config(
        "Qwen3-0.6B", "pile", 128, 2048, 8, ["l0", "l1"], {"l0": "W4A16"}, "W4A16", False, True
    )


class TestAutoschemeCache:
    def test_config_sorts_layer_names(self):
        cfg = _cache_config()
        assert cfg["quant_layer_names"] == ["l0", "l1"]
        assert cfg["model_id"] == "Qwen3-0.6B"

    def test_cache_key_is_16_hex(self):
        k = _autoscheme_cache_key(
            "Qwen3-0.6B", "pile", 128, 2048, 8, ["l0", "l1"], {"l0": "W4A16"}, "W4A16", False, True
        )
        assert len(k) == 16
        int(k, 16)  # must be hex

    def test_cache_path_uses_index(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ["AR_AUTO_SCHEME_CACHE"] = d
            try:
                p = _autoscheme_cache_path("deadbeef", 3)
                assert os.path.basename(p) == "scheme_03_deadbeef.json"
                assert os.path.isdir(d)
            finally:
                os.environ.pop("AR_AUTO_SCHEME_CACHE", None)

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "scheme_00_abc.json")
            _save_autoscheme_scores(path, "abc", 0, {"bits": 4}, {"l0": [4, 0.5]}, 1.0, 100, _cache_config())
            loaded = _load_autoscheme_scores(path)
            assert loaded["version"] == 1
            assert loaded["layer_scores"] == {"l0": [4, 0.5]}
            assert loaded["total_params"] == 100

    def test_load_missing_returns_none(self):
        assert _load_autoscheme_scores("/nonexistent/path.json") is None

    def test_load_wrong_version_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"version": 99, "score_granularity": "per_op"}')
            assert _load_autoscheme_scores(path) is None


class TestPerOpCacheCompatibility:
    def test_matches_exact_layers(self):
        data = {"layer_scores": {"l0": [4, 0.5], "l1": [4, 0.6]}}
        assert _is_per_op_cache_compatible(data, ["l0", "l1"], {}) is True

    def test_missing_layer_not_compatible(self):
        data = {"layer_scores": {"l0": [4, 0.5]}}
        assert _is_per_op_cache_compatible(data, ["l0", "l1"], {}) is False

    def test_fixed_layer_excluded(self):
        data = {"layer_scores": {"l0": [4, 0.5]}}
        assert _is_per_op_cache_compatible(data, ["l0", "l1"], {"l1": "W4A16"}) is True

    def test_find_compatible_cache(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _cache_config()
            path = os.path.join(d, "scheme_00_abc.json")
            _save_autoscheme_scores(path, "abc", 0, {"bits": 4}, {"l0": [4, 0.5]}, 1.0, 100, cfg)
            found = _find_compatible_autoscheme_cache(path, cfg, ["l0"], {}, 100)
            assert found is not None
            assert found["_cache_path"] == path


# ==============================================================================
# worker scheduling
# ==============================================================================


class TestWorkerScheduling:
    def test_assign_round_robin(self):
        assert _assign_scheme_worker_devices(3, ["cuda:0", "cuda:1"]) == ["cuda:0", "cuda:1", "cuda:0"]

    def test_assign_empty_devices_raises(self):
        with pytest.raises(ValueError, match="at least one device"):
            _assign_scheme_worker_devices(2, [])

    def test_worker_count_equals_schemes(self):
        assert _get_scheme_worker_count(3, 2) == 3

    def test_worker_count_requires_gpu(self):
        with pytest.raises(ValueError, match="at least one GPU"):
            _get_scheme_worker_count(3, 0)

    def test_can_parallel_all_conditions(self):
        assert _can_parallel_scheme_scoring(True, "m", 1, 3, False, False, False) is True

    def test_can_parallel_false_conditions(self):
        assert _can_parallel_scheme_scoring(True, None, 1, 3, False, False, False) is False
        assert _can_parallel_scheme_scoring(True, "m", 1, 3, True, False, False) is False
        assert _can_parallel_scheme_scoring(True, "m", 1, 3, False, True, True) is False
        assert _can_parallel_scheme_scoring(True, "m", 1, 1, False, False, False) is False
        assert _can_parallel_scheme_scoring(False, "m", 1, 3, False, False, False) is False


class TestProgressQueueProxy:
    def test_update_and_write(self):
        q = queue.Queue()
        p = _ProgressQueueProxy(q)
        p.update(2)
        p.write("hi")
        assert list(q.queue) == [("update", 2), ("write", "hi")]

    def test_drain_progress_queue(self):
        q = queue.Queue()
        q.put(("update", 3))
        q.put(("write", "msg"))

        class _Pbar:
            def __init__(self):
                self.updated = 0
                self.written = []

            def update(self, n):
                self.updated += n

            def write(self, m):
                self.written.append(m)

        pbar = _Pbar()
        _drain_progress_queue(q, pbar)
        assert pbar.updated == 3
        assert pbar.written == ["msg"]


class TestMergeMemoryReports:
    def test_merges_worker_reports(self):
        class _Monitor:
            def __init__(self):
                self.peak_ram = 10.0
                self.peak_vram = {"0": 5.0}

        mon = _Monitor()
        _merge_worker_memory_reports(mon, [{"device": "0", "peak_ram": 3.0, "peak_vram": 2.0}])
        assert mon.peak_ram >= 13.0
        assert mon.peak_vram["0"] >= 5.0


class TestMoveModuleToTuningDevice:
    def test_moves_wrapper_and_leaf(self):
        class _Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 4)
                self.lin.tuning_device = "cpu"

        m = _Inner()
        move_module_to_tuning_device(m, major_device="cpu")
        assert m.lin.weight.device.type == "cpu"

    def test_moves_direct_parameter(self):
        class _Leaf(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Parameter(torch.randn(4))

        m = _Leaf()
        move_module_to_tuning_device(m, major_device="cpu")
        assert m.p.device.type == "cpu"
