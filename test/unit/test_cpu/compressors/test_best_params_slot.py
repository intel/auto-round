# Copyright (c) 2025 Intel Corporation
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

"""Tests for the huge-layer best-params snapshot slot (compressors/utils.py).

Covers: the window-floor home gate (free - snapshot >= row-window floor, so a
24 GB card keeps the snapshot beside shrunken windows instead of paying the
host round trip per improving iteration), the idle-peer rung, the host
fallback, and the pre-reserved slot's refresh path.
"""

import logging

import pytest
import torch

from auto_round.compressors import utils as cu


class _FakeWrapper(torch.nn.Module):
    """Minimal stand-in for a WrapperLinear: a params dict plus a home device."""

    def __init__(self, n_elems=1000, home="cpu"):
        super().__init__()
        self.device = torch.device(home)
        self.orig_layer = None  # collect_best_params keys off this attribute
        self.params = {
            "value": torch.zeros(n_elems, dtype=torch.float32),
            "min_scale": torch.ones(4, dtype=torch.float32),
            "max_scale": torch.ones(4, dtype=torch.float32),
        }


def _need_bytes(wrapper):
    return sum(t.numel() * t.element_size() for t in wrapper.params.values())


@pytest.fixture()
def cuda_probes(monkeypatch):
    """Route torch.cuda free-memory and device-count probes to a fake table."""

    class _Probes:
        def __init__(self):
            self.free = {}
            self.count = 0

        def install(self):
            # the ladders route through the device-manager abstraction
            import auto_round.utils.device_manager as dm_mod

            probes = self

            class _FakeAR:
                def is_available(self):
                    return True

                @property
                def device_count(self):
                    return probes.count

                def mem_get_info(self, index=0):
                    dev = torch.device("cuda", index)
                    return probes.free.get(dev, 0), 0

            monkeypatch.setattr(dm_mod, "get_ar_device", lambda t: _FakeAR())
            monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev: (self.free.get(dev, 0), 0))
            monkeypatch.setattr(torch.cuda, "device_count", lambda: self.count)

    probes = _Probes()
    probes.install()
    return probes


class TestWindowFloor:
    def test_floor_from_real_shapes(self):
        w = _FakeWrapper()
        w.params["value"] = torch.zeros(512, 4096)
        # 1024 min rows x in_features x fp32 x ~6 live arrays
        assert cu.snapshot_window_floor_bytes(w) == 1024 * 4096 * 4 * 6

    def test_non_2d_value_is_zero(self):
        w = _FakeWrapper()
        w.params["value"] = torch.zeros(4096)
        assert cu.snapshot_window_floor_bytes(w) == 0

    def test_missing_params_is_zero(self):
        w = _FakeWrapper()
        w.params = {}
        assert cu.snapshot_window_floor_bytes(w) == 0


class TestSelectSnapshotDevice:
    def _big_wrapper(self, rows, in_features, home="cuda:0"):
        w = _FakeWrapper(n_elems=1, home=home)
        w.params["value"] = torch.zeros(rows, in_features)
        w.params["min_scale"] = torch.ones(4)
        w.params["max_scale"] = torch.ones(4)
        return w

    def test_home_when_snapshot_fits_beside_window_floor(self, cuda_probes):
        # the 24 GB case: free below 2x the snapshot, but snapshot + floor fit
        w = self._big_wrapper(rows=65536, in_features=4096)  # ~1 GiB snapshot
        need = _need_bytes(w)
        floor = cu.snapshot_window_floor_bytes(w)  # 100 MiB
        assert need > floor
        cuda_probes.free = {torch.device("cuda:0"): need + floor + (512 * 2**20)}
        cuda_probes.count = 1
        assert cu.select_snapshot_device(w) == torch.device("cuda:0")

    def test_home_declined_when_floor_would_not_fit(self, cuda_probes):
        w = self._big_wrapper(rows=4096, in_features=65536)  # ~1 GiB, floor ~1.5 GiB
        need = _need_bytes(w)
        floor = cu.snapshot_window_floor_bytes(w)
        assert floor > need
        # free exceeds the snapshot, but the remainder is below the window floor
        cuda_probes.free = {torch.device("cuda:0"): need + (512 * 2**20)}
        cuda_probes.count = 1
        assert cu.select_snapshot_device(w) == torch.device("cpu")

    def test_peer_when_home_floor_fails(self, cuda_probes):
        w = self._big_wrapper(rows=4096, in_features=65536)  # floor > free - need
        need = _need_bytes(w)
        cuda_probes.free = {
            torch.device("cuda:0"): need + (512 * 2**20),  # no room for the floor
            torch.device("cuda:1"): need * 4,  # comfortably above need / 0.9
        }
        cuda_probes.count = 2
        assert cu.select_snapshot_device(w) == torch.device("cuda:1")

    def test_peer_needs_headroom(self, cuda_probes):
        w = self._big_wrapper(rows=65536, in_features=4096)
        need = _need_bytes(w)
        cuda_probes.free = {
            torch.device("cuda:0"): need,  # no room for the floor
            torch.device("cuda:1"): need,  # exactly need: the 10% headroom declines it
        }
        cuda_probes.count = 2
        assert cu.select_snapshot_device(w) == torch.device("cpu")

    def test_non_cuda_home_returns_host(self, cuda_probes):
        cuda_probes.free = {torch.device("cpu"): 10 * 2**30}
        cuda_probes.count = 0
        w = _FakeWrapper(n_elems=256 * 2**20, home="cpu")
        assert cu.select_snapshot_device(w) == torch.device("cpu")

    def test_probe_failure_falls_back_to_host(self, cuda_probes):
        cuda_probes.free = {}  # every probe returns free=0 -> nothing fits
        cuda_probes.count = 4
        w = self._big_wrapper(rows=65536, in_features=4096)
        assert cu.select_snapshot_device(w) == torch.device("cpu")


class TestBestParamsSlot:
    def test_host_mode_matches_historical_behavior(self):
        w = _FakeWrapper(n_elems=1000, home="cpu")
        slot = cu.BestParamsSlot(w)
        assert slot.device == torch.device("cpu")
        assert slot.buffers is None
        snapshot = slot.refresh(w)
        assert set(snapshot.keys()) == set(w.params.keys())
        for key, tensor in snapshot.items():
            assert torch.equal(tensor, w.params[key].data)
            assert tensor.device.type == "cpu"

    def test_gpu_mode_reserves_and_refreshes_in_place(self, cuda_probes, monkeypatch):
        w = _FakeWrapper(n_elems=1, home="cuda:0")
        w.params["value"] = torch.zeros(65536, 4096)  # ~1 GiB
        w.params["min_scale"] = torch.ones(4)
        w.params["max_scale"] = torch.ones(4)
        need = _need_bytes(w)
        cuda_probes.free = {torch.device("cuda:0"): need + (512 * 2**20)}
        cuda_probes.count = 1
        # CPU test box: emulate device placement while keeping the cuda selection
        allocated = {}
        real_empty_like = torch.empty_like

        def fake_empty_like(tensor, device=None):
            allocated[device] = allocated.get(device, 0) + tensor.numel() * tensor.element_size()
            return real_empty_like(tensor)

        monkeypatch.setattr(torch, "empty_like", fake_empty_like)

        captured = []

        class _Capture(logging.Handler):
            def emit(self, record):
                captured.append(record)

        handler = _Capture(level=logging.INFO)
        library_logger = cu.logger
        library_logger.addHandler(handler)
        try:
            slot = cu.BestParamsSlot(w)
        finally:
            library_logger.removeHandler(handler)

        assert slot.device == torch.device("cuda:0")
        assert set(slot.buffers.keys()) == set(w.params.keys())
        assert allocated[torch.device("cuda:0")] == pytest.approx(need)
        reserve_lines = [r for r in captured if "slot reserved" in r.getMessage()]
        assert len(reserve_lines) == 1 and "cuda:0" in reserve_lines[0].getMessage()

        w.params["value"].add_(1.0)
        snapshot = slot.refresh(w)
        assert torch.equal(snapshot["value"], w.params["value"].data)

    def test_reservation_failure_degrades_to_host(self, cuda_probes, monkeypatch):
        w = _FakeWrapper(n_elems=1, home="cuda:0")
        w.params["value"] = torch.zeros(65536, 4096)
        w.params["min_scale"] = torch.ones(4)
        w.params["max_scale"] = torch.ones(4)
        need = _need_bytes(w)
        cuda_probes.free = {torch.device("cuda:0"): need + (512 * 2**20)}
        cuda_probes.count = 1

        def failing_empty_like(tensor, device=None):
            raise RuntimeError("CUDA out of memory")

        monkeypatch.setattr(torch, "empty_like", failing_empty_like)
        slot = cu.BestParamsSlot(w)
        assert slot.device == torch.device("cpu")
        assert slot.buffers is None
        snapshot = slot.refresh(w)
        assert torch.equal(snapshot["value"], w.params["value"].data)
