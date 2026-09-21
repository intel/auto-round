# coding=utf-8
# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""_resolve_pack_device_: int64 pack intermediates (~16 B/elem) must fit the
live free VRAM or the pack drops to the host.

A 248k-vocab lm_head (1.27B params -> ~18.9GiB intermediates) drops to cpu when
only ~8GiB is free; a body-sized linear (~89M params -> ~1.3GiB) stays on the
device; non-cuda defaults and probe failures pass through untouched.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from auto_round.compressors.utils import _resolve_pack_device_


class _FakeAR:
    def __init__(self, free, total):
        self._free, self._total = free, total

    def is_available(self):
        return True

    @property
    def device_count(self):
        return 1

    def mem_get_info(self, index=0):
        return self._free, self._total


_GIB = 2**30


class TestResolvePackDevice:
    def test_huge_layer_drops_to_cpu_when_free_vram_low(self):
        w = SimpleNamespace(numel=lambda: 1271398400)  # ~18.9GiB at 16 B/elem
        with patch("auto_round.utils.device_manager.get_ar_device", lambda t: _FakeAR(8 * _GIB, 23 * _GIB)):
            out = _resolve_pack_device_(w, torch.device("cuda:0"))
        assert out.type == "cpu"

    def test_body_sized_layer_stays_on_device(self):
        w = SimpleNamespace(numel=lambda: 89_000_000)  # ~1.3GiB at 16 B/elem
        with patch("auto_round.utils.device_manager.get_ar_device", lambda t: _FakeAR(8 * _GIB, 23 * _GIB)):
            out = _resolve_pack_device_(w, torch.device("cuda:0"))
        assert out.type == "cuda"

    def test_cpu_default_short_circuits_without_probe(self):
        probed = []

        def _probe(dev):
            probed.append(dev)
            raise AssertionError("cpu default must skip the probe")

        with patch("auto_round.utils.device_manager.get_ar_device", _probe):
            out = _resolve_pack_device_(SimpleNamespace(numel=lambda: 10**9), torch.device("cpu"))
        assert out.type == "cpu"
        assert probed == []

    def test_probe_failure_falls_back_to_default(self):
        def _boom(dev):
            raise RuntimeError("no cuda here")

        with patch("auto_round.utils.device_manager.get_ar_device", _boom):
            out = _resolve_pack_device_(SimpleNamespace(numel=lambda: 10**9), torch.device("cuda:0"))
        assert out.type == "cuda"

    def test_none_weight_passes_through(self):
        assert _resolve_pack_device_(None, torch.device("cuda:0")).type == "cuda"
