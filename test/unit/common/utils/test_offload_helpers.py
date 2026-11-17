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

import json
import os
import sys
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
        assert layer.bias.numel() == 0

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

    @pytest.mark.parametrize(
        "filename",
        [
            "model.safetensors.index.json",
            "diffusion_pytorch_model.safetensors.index.json",
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.bin.index.json",
            "diffusion_pytorch_model.bin",
        ],
    )
    def test_revision_resolves_partial_snapshot_from_cached_weight_index(self, tmp_path, filename):
        """A usable weight snapshot need not contain every repository file."""
        from auto_round.utils.offload import _resolve_model_dir

        snapshot = tmp_path / "snapshots" / "commit"
        index_path = snapshot / filename
        index_path.parent.mkdir(parents=True)
        index_path.write_text('{"weight_map": {}}')
        incomplete = RuntimeError("snapshot is missing README.md")
        with (
            patch(
                "huggingface_hub.try_to_load_from_cache",
                side_effect=lambda repo, name, revision: str(index_path) if name == filename else None,
            ) as cached,
            patch("huggingface_hub.snapshot_download", side_effect=incomplete) as download,
        ):
            resolved = _resolve_model_dir("org/model", revision="commit")

        assert resolved == str(snapshot)
        cached.assert_any_call("org/model", filename, revision="commit")
        download.assert_not_called()

    def test_offload_manager_resolves_model_id_once(self, tmp_path):
        """Block reloads must reuse the snapshot resolved when the manager is created."""
        from auto_round.utils.offload import OffloadManager

        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        model = nn.Sequential(nn.Linear(2, 2))
        with (
            patch("auto_round.utils.offload._resolve_model_dir", return_value=str(snapshot)) as resolve,
            patch("auto_round.utils.offload.load_block_from_model_files") as load_block,
        ):
            manager = OffloadManager(mode="clean", model_dir="org/model", model_revision="commit")
            manager.reload(model, "0")

        assert manager.model_dir == str(snapshot)
        resolve.assert_called_once_with("org/model", revision="commit")
        load_block.assert_called_once_with(str(snapshot), "0", model[0])

    def test_offload_manager_uses_diffusion_component_subfolder(self, tmp_path):
        """Diffusers block reloads must read from the active pipeline component."""
        from auto_round.utils.offload import OffloadManager

        component_dir = tmp_path / "transformer_2"
        component_dir.mkdir()
        model = nn.Sequential(nn.Linear(2, 2))
        model._autoround_checkpoint_subfolder = "transformer_2"
        with patch("auto_round.utils.offload.load_block_from_model_files") as load_block:
            manager = OffloadManager(mode="clean", model_dir=str(tmp_path))
            manager.reload(model, "0")

        load_block.assert_called_once_with(str(component_dir), "0", model[0])

    @pytest.mark.parametrize("component", ["unet", "transformer", "transformer_2"])
    def test_diffusion_component_clear_and_reload(self, tmp_path, component):
        from safetensors.torch import save_file

        from auto_round.utils.offload import OffloadManager

        component_dir = tmp_path / component
        component_dir.mkdir()
        model = nn.Sequential(nn.Linear(2, 2))
        model._autoround_checkpoint_subfolder = component
        expected = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        save_file(expected, str(component_dir / "diffusion_pytorch_model.safetensors"))
        model[0].register_parameter("calib_scale", nn.Parameter(torch.ones(1)))
        manager = OffloadManager(mode="clean", model_dir=str(tmp_path))
        manager.offload(model, "0")
        assert model[0].weight.numel() == model[0].bias.numel() == 0
        assert model[0].calib_scale.numel() == 1
        manager.reload(model, "0")
        for name, tensor in expected.items():
            torch.testing.assert_close(model.state_dict()[name], tensor)

    def test_offload_manager_does_not_eagerly_resolve_without_revision(self):
        """Revision-less and non-Hugging Face sources retain the legacy lazy path."""
        from auto_round.utils.offload import OffloadManager

        with patch("auto_round.utils.offload._resolve_model_dir") as resolve:
            manager = OffloadManager(mode="clean", model_dir="modelscope/model")

        assert manager.model_dir == "modelscope/model"
        resolve.assert_not_called()

    def test_revision_resolution_preserves_hub_error(self):
        """A pinned lookup failure must not be hidden behind a local-path error."""
        from auto_round.utils.offload import _resolve_model_dir

        error = RuntimeError("snapshot unavailable")
        with (
            patch("huggingface_hub.try_to_load_from_cache", return_value=None),
            patch("huggingface_hub.snapshot_download", side_effect=error),
            pytest.raises(RuntimeError) as exc_info,
        ):
            _resolve_model_dir("org/model", revision="commit")

        assert exc_info.value is error


class TestBuildWeightMap:
    @pytest.mark.parametrize(
        "filename",
        [
            "model.safetensors",
            "diffusion_pytorch_model.safetensors",
            "pytorch_model.bin",
            "diffusion_pytorch_model.bin",
        ],
    )
    def test_single_checkpoint(self, tmp_path, filename):
        from safetensors.torch import save_file

        from auto_round.utils.offload import _build_weight_map

        state_dict = {"blocks.0.weight": torch.ones(2, 2)}
        checkpoint = tmp_path / filename
        if filename.endswith(".safetensors"):
            save_file(state_dict, str(checkpoint))
        else:
            torch.save(state_dict, checkpoint)

        assert _build_weight_map(str(tmp_path)) == {"blocks.0.weight": filename}

    def test_custom_diffusers_safetensors_index(self, tmp_path):
        from auto_round.utils.offload import _build_weight_map

        weight_map = {"blocks.0.weight": "diffusion_pytorch_model-00001-of-00002.safetensors"}
        index_path = tmp_path / "diffusion_pytorch_model.safetensors.index.json"
        index_path.write_text(json.dumps({"weight_map": weight_map}))

        assert _build_weight_map(str(tmp_path)) == weight_map


# ---------------------------------------------------------------------------
# OffloadManager stale-dir sweep
# ---------------------------------------------------------------------------
class TestOffloadStaleDirSweep:
    """Startup reclaim of offload temp dirs left behind by dead processes."""

    def _offload_base(self, tmp_path, monkeypatch) -> str:
        """Point ``AR_WORK_SPACE`` at a *relative* path under ``tmp_path``.

        A relative workspace keeps the test independent of path casing (and of
        whether the filesystem is case-sensitive), unlike lowercasing an
        absolute ``tmp_path``.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("AR_WORK_SPACE", "ar_work_space")
        monkeypatch.delenv("AR_RESUME_DIR", raising=False)
        base = os.path.join("ar_work_space", "offload")
        os.makedirs(base, exist_ok=True)
        return base

    @staticmethod
    def _make_stale_dir(base: str, name: str, pid: int, host: str | None = None) -> str:
        """Create a temp dir laid out the way a crashed run leaves it."""
        from auto_round.utils.offload import _OWNER_MARKER, _hostname_tag

        path = os.path.join(base, name)
        os.makedirs(path)
        with open(os.path.join(path, _OWNER_MARKER), "w") as f:
            json.dump({"pid": pid, "host": host if host is not None else _hostname_tag()}, f)
        return path

    def test_ensure_dir_is_pid_tagged_and_sweeps_stale(self, tmp_path, monkeypatch):
        from auto_round.utils.offload import OffloadManager

        base = self._offload_base(tmp_path, monkeypatch)
        stale = self._make_stale_dir(base, "compressor_424242_deadbeef", 424242)
        mgr = OffloadManager(enabled=True, mode="offload", offload_dir_prefix="compressor")
        # Only our own pid counts as alive.
        monkeypatch.setattr(OffloadManager, "_pid_alive", staticmethod(lambda pid: pid == os.getpid()))
        tempdir = mgr._ensure_dir()
        try:
            assert not os.path.exists(stale), "stale dir from a dead pid should have been swept"
            assert os.path.basename(tempdir).startswith(f"compressor_{os.getpid()}_")
        finally:
            mgr._cleanup_tempdir()
        assert not os.path.exists(tempdir)

    def test_sweep_keeps_live_resume_and_legacy_dirs(self, tmp_path, monkeypatch):
        from auto_round.utils.offload import OffloadManager

        base = self._offload_base(tmp_path, monkeypatch)
        dead = self._make_stale_dir(base, "compressor_424242_deadbeef", 424242)
        live = self._make_stale_dir(base, f"compressor_{os.getpid()}_cafebabe", os.getpid())
        # Directories without an owner marker cannot be proven stale: the
        # resume dir is shared across runs by design, and pre-marker dirs left
        # by an older AutoRound are indistinguishable from a live run.
        unmarked = [os.path.join(base, "compressor_resume"), os.path.join(base, "compressor_0gbcamv0")]
        for d in unmarked:
            os.makedirs(d)
        mgr = OffloadManager(enabled=True, mode="offload", offload_dir_prefix="compressor")
        monkeypatch.setattr(OffloadManager, "_pid_alive", staticmethod(lambda pid: pid == os.getpid()))
        mgr._sweep_stale_dirs(base)
        assert not os.path.exists(dead)
        assert os.path.isdir(live)
        for d in unmarked:
            assert os.path.isdir(d)

    def test_sweep_is_cross_prefix(self, tmp_path, monkeypatch):
        """A crash in AutoScheme must not leak a dir the compressor never handles."""
        from auto_round.utils.offload import OffloadManager

        base = self._offload_base(tmp_path, monkeypatch)
        foreign = self._make_stale_dir(base, "autoscheme_424242_deadbeef", 424242)
        mgr = OffloadManager(enabled=True, mode="offload", offload_dir_prefix="compressor")
        monkeypatch.setattr(OffloadManager, "_pid_alive", staticmethod(lambda pid: pid == os.getpid()))
        mgr._sweep_stale_dirs(base)
        assert not os.path.exists(foreign), "every prefix shares offload/, so all dead dirs go"

    def test_sweep_skips_dirs_owned_by_another_host(self, tmp_path, monkeypatch):
        """A pid from another machine says nothing about our process table."""
        from auto_round.utils.offload import OffloadManager

        base = self._offload_base(tmp_path, monkeypatch)
        remote = self._make_stale_dir(base, "compressor_424242_deadbeef", 424242, host="some-other-host")
        mgr = OffloadManager(enabled=True, mode="offload", offload_dir_prefix="compressor")
        monkeypatch.setattr(OffloadManager, "_pid_alive", staticmethod(lambda pid: False))
        mgr._sweep_stale_dirs(base)
        assert os.path.isdir(remote), "shared-storage leftovers from another host must be left alone"

    def test_sweep_tolerates_vanishing_base_dir(self, tmp_path, monkeypatch):
        """A concurrent cleaner removing offload/ must not break _ensure_dir()."""
        from auto_round.utils.offload import OffloadManager

        base = self._offload_base(tmp_path, monkeypatch)
        mgr = OffloadManager(enabled=True, mode="offload", offload_dir_prefix="compressor")
        monkeypatch.setattr("os.listdir", lambda _: (_ for _ in ()).throw(FileNotFoundError("gone")))
        mgr._sweep_stale_dirs(base)  # must not raise

    def test_resume_mode_keeps_resume_dir_but_reclaims_dead_dirs(self, tmp_path, monkeypatch):
        from auto_round.utils.offload import OffloadManager

        base = self._offload_base(tmp_path, monkeypatch)
        monkeypatch.setenv("AR_RESUME_DIR", str(tmp_path / "resume"))
        stale = self._make_stale_dir(base, "compressor_424242_deadbeef", 424242)
        # The resume dir itself is shared across runs by design, so it carries
        # no owner marker and can never be swept.
        os.makedirs(os.path.join(base, "compressor_resume"))
        mgr = OffloadManager(enabled=True, mode="offload", offload_dir_prefix="compressor")
        monkeypatch.setattr(OffloadManager, "_pid_alive", staticmethod(lambda pid: False))
        tempdir = mgr._ensure_dir()
        try:
            assert tempdir == os.path.join(base, "compressor_resume")
            # Resume mode is exactly the crash-recovery path, so stale dirs
            # from the dead run are reclaimed while the resume dir survives.
            assert not os.path.exists(stale)
            assert os.path.isdir(tempdir)
        finally:
            mgr._cleanup_tempdir()

    def test_pid_alive_handles_out_of_range_pid(self):
        """An untrustworthy pid tag must read as alive, never as dead."""
        from auto_round.utils.offload import OffloadManager

        if sys.platform == "win32":  # pragma: no cover - POSIX-only overflow
            pytest.skip("OverflowError from os.kill is POSIX-specific")
        assert OffloadManager._pid_alive(2**64) is True
        assert OffloadManager._pid_alive(0) is False
