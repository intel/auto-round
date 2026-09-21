# coding=utf-8
# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Block-path best-params snapshot placement: one ladder for every lane.

Attempt-based placement for the block-path best-params snapshot:
beside the weights -> freest visible accelerator -> host (+WARNING), every
step attempted for real with failure falling through, the successful route
sticky per block, the host terminal. The outside-block lane keeps its own
select_snapshot_device reservation."""

import torch
import torch.nn as nn

from auto_round.compressors.utils import snapshot_best_params


class _Wrap(nn.Module):
    """Minimal wrapper-shaped module: params dict with a fp32 value."""

    def __init__(self, out_f, in_f, device="cpu"):
        super().__init__()
        self.orig_layer = nn.Linear(in_f, out_f, bias=False)
        self.params = {"value": torch.zeros(out_f, in_f, dtype=torch.float32, device=device)}
        self.device = torch.device(device)


class _FakeAR:
    """Device-manager backend stub: the ladder routes through get_ar_device."""

    def __init__(self, free_by_index, count):
        self._free_by_index, self._count = free_by_index, count

    def is_available(self):
        return True

    @property
    def device_count(self):
        return self._count

    def mem_get_info(self, index=0):
        return self._free_by_index.get(index, 0), 1 << 40


def _cuda_stub(monkeypatch, free_bytes_by_index, device_count=1, dev_type="cuda"):
    """Stub the accelerator introspection the ladder consults; CPU-only tests."""
    import auto_round.utils.device_manager as dm_mod

    fake = _FakeAR(free_bytes_by_index, device_count)
    monkeypatch.setattr(dm_mod, "get_ar_device", lambda t: fake)


class TestSnapshotBestParams:
    def test_cpu_cache_device_keeps_host_snapshot(self):
        blk = _Wrap(8, 4)
        out = snapshot_best_params(blk, "cpu")
        assert out["value"].device.type == "cpu"

    def test_free_gpu_attempts_beside_weights_first(self, monkeypatch):
        blk = _Wrap(8, 4)
        _cuda_stub(monkeypatch, {0: 8 << 30})
        out = snapshot_best_params(blk, "cuda:0")
        # first attempt = per-parameter local duplication, no prediction gates it
        assert out["value"].device.type == blk.params["value"].device.type

    def test_local_failure_falls_to_freest_peer(self, monkeypatch):
        import auto_round.compressors.utils as utils_mod

        blk = _Wrap(8, 4)
        # two idle peers: cuda:1 (8GiB) and cuda:2 (16GiB) - freest first
        _cuda_stub(monkeypatch, {0: 1 << 20, 1: 8 << 30, 2: 16 << 30}, device_count=3)
        calls = []
        monkeypatch.setattr(
            utils_mod, "collect_best_params_local", lambda b: (_ for _ in ()).throw(RuntimeError("OOM"))
        )
        monkeypatch.setattr(utils_mod, "collect_best_params", lambda block, dev: calls.append(dev) or {"value": None})
        snapshot_best_params(blk, "cuda:0")
        assert calls and calls[0] == torch.device("cuda", 2)  # freest, not first-index

    def test_all_accelerators_fail_parks_on_host_with_warning(self, monkeypatch):
        import auto_round.compressors.utils as utils_mod

        blk = _Wrap(8, 4)
        _cuda_stub(monkeypatch, {0: 1 << 20}, device_count=1)  # no peer at all
        monkeypatch.setattr(
            utils_mod, "collect_best_params_local", lambda b: (_ for _ in ()).throw(RuntimeError("OOM"))
        )
        warned = []
        monkeypatch.setattr(utils_mod.logger, "warning", lambda *a, **k: warned.append(a), raising=False)
        out = snapshot_best_params(blk, "cuda:0")
        assert out["value"].device.type == "cpu"  # host fallback, never raises
        assert warned  # loud, not silent

    def test_peer_failure_falls_through_to_host(self, monkeypatch):
        import auto_round.compressors.utils as utils_mod

        blk = _Wrap(8, 4)
        _cuda_stub(monkeypatch, {0: 1 << 20, 1: 4 << 30}, device_count=2)
        monkeypatch.setattr(
            utils_mod, "collect_best_params_local", lambda b: (_ for _ in ()).throw(RuntimeError("OOM"))
        )
        monkeypatch.setattr(
            utils_mod,
            "collect_best_params",
            lambda block, dev: (
                (_ for _ in ()).throw(RuntimeError("OOM")) if str(dev) != "cpu" else {"value": torch.zeros(1)}
            ),
        )
        warned = []
        monkeypatch.setattr(utils_mod.logger, "warning", lambda *a, **k: warned.append(a), raising=False)
        out = snapshot_best_params(blk, "cuda:0")
        assert out["value"].device.type == "cpu"
        assert warned

    def test_successful_route_is_sticky(self, monkeypatch):
        import auto_round.compressors.utils as utils_mod

        blk = _Wrap(8, 4)
        _cuda_stub(monkeypatch, {0: 1 << 20, 1: 8 << 30}, device_count=2)
        calls = []
        monkeypatch.setattr(
            utils_mod, "collect_best_params_local", lambda b: (_ for _ in ()).throw(RuntimeError("OOM"))
        )
        monkeypatch.setattr(utils_mod, "collect_best_params", lambda block, dev: calls.append(dev) or {"value": None})
        snapshot_best_params(blk, "cuda:0")
        assert calls == [torch.device("cuda", 1)]
        # second improving iteration: the peer is retried first (local would
        # raise again, but the peer is tried BEFORE re-enumerating)
        snapshot_best_params(blk, "cuda:0")
        assert calls == [torch.device("cuda", 1), torch.device("cuda", 1)]

    def test_host_is_terminal(self, monkeypatch):
        import auto_round.compressors.utils as utils_mod

        blk = _Wrap(8, 4)
        _cuda_stub(monkeypatch, {0: 1 << 20, 1: 8 << 30}, device_count=2)
        monkeypatch.setattr(
            utils_mod, "collect_best_params_local", lambda b: (_ for _ in ()).throw(RuntimeError("OOM"))
        )
        monkeypatch.setattr(
            utils_mod,
            "collect_best_params",
            lambda block, dev: (
                (_ for _ in ()).throw(RuntimeError("OOM")) if str(dev) != "cpu" else {"value": torch.zeros(1)}
            ),
        )
        snapshot_best_params(blk, "cuda:0")
        assert blk._snapshot_route == "host"
        # later iterations stay on the host even though local now works
        monkeypatch.setattr(utils_mod, "collect_best_params_local", lambda b: {"value": blk.params["value"].clone()})
        out = snapshot_best_params(blk, "cuda:0")
        assert out["value"].device.type == "cpu"


class TestNonCudaAccelerators:
    def test_xpu_home_floor_exceeded_parks_on_host(self, monkeypatch):
        import auto_round.compressors.utils as utils_mod

        blk = _Wrap(8, 4)
        _cuda_stub(monkeypatch, {0: 1 << 20}, device_count=1, dev_type="xpu")
        warned = []
        monkeypatch.setattr(utils_mod.logger, "warning", lambda *a, **k: warned.append(a), raising=False)
        monkeypatch.setattr(
            utils_mod, "collect_best_params_local", lambda b: (_ for _ in ()).throw(RuntimeError("OOM"))
        )
        out = snapshot_best_params(blk, "xpu:0")
        assert out["value"].device.type == "cpu"
        assert warned


class TestCensusStrDevice:
    def test_str_device_is_normalized_not_dropped(self, monkeypatch):
        """device_manager.device is a str; a str must reach the allocator
        probe instead of silently failing the .type check (the live 00:10 run
        printed no census lines at all for exactly this reason)."""
        import auto_round.utils.device as dev_mod

        seen = {}

        class _FakeCuda:
            @staticmethod
            def mem_get_info(dev=None):
                seen["dev"] = dev
                return 8 << 30, 23 << 30

            @staticmethod
            def memory_allocated(dev=None):
                return 1 << 30

            @staticmethod
            def memory_reserved(dev=None):
                return 2 << 30

        monkeypatch.setattr(dev_mod.logger, "isEnabledFor", lambda lvl: True, raising=False)
        monkeypatch.setattr(dev_mod.torch, "cuda", _FakeCuda, raising=False)
        logs = []
        monkeypatch.setattr(dev_mod.logger, "debug", lambda *a, **k: logs.append(a), raising=False)
        dev_mod.log_cuda_memory_census("str-device probe", "cuda:0", walk=False)
        assert seen["dev"] == torch.device("cuda:0")  # normalized, probed
        assert logs, "header must print for a str device"


class TestBestParamsReleaseBeforeReclone:
    def test_best_params_released_before_reclone(self):
        """The old best-params snapshot must be dropped before the new clone.

        Holding both doubles the snapshot footprint for the duration of the
        clone - weight-sized fp32 rounding values make that window OOM-class
        for huge layers on a 24 GB card. Source contract: every snapshot
        clone site in quantize_block is preceded by the release.
        """
        import inspect
        import re as _re

        from auto_round.algorithms.quantization.sign_round import quantizer as qmod

        src = inspect.getsource(qmod.SignRoundQuantizer.quantize_block)
        releases = [m.start() for m in _re.finditer(r"best_params = None", src)]
        clones = [m.start() for m in _re.finditer(r"snapshot_best_params\(", src)]
        assert len(releases) == 2  # improving-iteration site + last-iteration site
        assert len(clones) == 2
        for c in clones:
            assert any(r < c for r in releases)  # a release precedes every clone
