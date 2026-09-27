# coding=utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""Tests for CompressionOrchestrator._attach_pool_placement (calibration-data pools).

Covers the documented no-op contract (policy off, CPU-parked lane) and the
fits-home consolidation call with its reserved-bytes bookkeeping.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch


def _fake_orchestrator(cache_device, mode, batch_size=2, iters=0, runner_device="cuda:0"):
    from auto_round.compressors.orchestrator import CompressionOrchestrator

    runner = SimpleNamespace(device=runner_device, pool_placement="unset")
    composer = SimpleNamespace(
        block_forward=runner,
        block_quantizer=SimpleNamespace(iters=iters),
        need_quanted_input=lambda: False,
    )
    self = SimpleNamespace(
        alg_composer=composer,
        compress_context=SimpleNamespace(cache_device=cache_device, calibration_data_device=mode),
        calibration_context=SimpleNamespace(batch_size=batch_size),
    )
    return CompressionOrchestrator, self, runner


class _FakePlacement:
    def __init__(self, devices, counts=None):
        self.devices = devices
        self._counts = counts or {str(d): 2 for d in devices}

    def counts(self):
        return dict(self._counts)


class TestAttachPoolPlacement(unittest.TestCase):
    def _pools(self):
        return [[torch.randn(2, 3, 8) for _ in range(2)]]  # one pool of 2 chunks

    def test_cpu_parked_lane_never_consolidates(self):
        # low_gpu_mem_usage: cache_device is the host; consolidation must not
        # migrate host-parked pools onto the compute GPU (docstring contract)
        cls, self_, runner = _fake_orchestrator(cache_device="cpu", mode="auto")
        with mock.patch("auto_round.utils.pool_placement.consolidate_pool_onto") as cons:
            cls._attach_pool_placement(self_, object(), self._pools()[0])
        cons.assert_not_called()
        self.assertIsNone(runner.pool_placement)

    def test_policy_off_skips_consolidation(self):
        cls, self_, runner = _fake_orchestrator(cache_device="cuda:0", mode="off")
        with mock.patch("auto_round.utils.pool_placement.consolidate_pool_onto") as cons:
            cls._attach_pool_placement(self_, object(), self._pools()[0])
        cons.assert_not_called()
        self.assertIsNone(runner.pool_placement)

    def test_mode_flows_into_the_real_resolver(self):
        """Regression (review R4-2): mode was read but never passed, so off/cpu/csv
        only worked when the resolver was mocked. Exercise the REAL resolver with
        only the free-memory probe faked."""
        cls, self_, runner = _fake_orchestrator(cache_device="cuda:0", mode="off")
        probe = {"cuda:0": 80 << 30, "cuda:1": 80 << 30}
        with mock.patch("auto_round.utils.device.probe_usable_bytes", side_effect=lambda d: probe.get(d)):
            cls._attach_pool_placement(self_, object(), self._pools()[0])
        # a healthy CUDA fleet must NOT resurrect the disabled policy
        self.assertIsNone(runner.pool_placement)

        cls, self_, runner = _fake_orchestrator(cache_device="cuda:0", mode="cpu")
        with mock.patch("auto_round.utils.device.probe_usable_bytes", side_effect=lambda d: probe.get(d)):
            cls._attach_pool_placement(self_, object(), self._pools()[0])
        self.assertEqual(runner.pool_placement.devices, ["cpu"])  # real resolver, host parking

        cls, self_, runner = _fake_orchestrator(cache_device="cuda:0", mode="cuda:1")
        with mock.patch("auto_round.utils.device.probe_usable_bytes", side_effect=lambda d: probe.get(d)), mock.patch(
            "auto_round.utils.pool_placement.consolidate_pool_onto"
        ):  # no CUDA here; consolidation is not under test
            cls._attach_pool_placement(self_, object(), self._pools()[0])
        # forced csv: the plan draws from the forced devices only
        self.assertTrue(set(runner.pool_placement.devices) <= {"cuda:1"})

    def test_iters_gt0_consumer_retargets_outputs(self):
        """iters>0: the output plan consumer is the block quantizer's per-block
        loss device (falling back to the runner device); at iters=0 or
        single-device lanes the consumer stays unset."""
        # loss device set on the quantizer wins over the runner device
        cls, self_, runner = _fake_orchestrator(cache_device="cuda:0", mode="auto", iters=20, runner_device="cuda:0")
        self_.alg_composer.block_quantizer._loss_device = "cuda:1"
        with mock.patch(
            "auto_round.utils.pool_placement.resolve_placement_for_pool", return_value=None
        ) as resolve, mock.patch("auto_round.utils.pool_placement.consolidate_pool_onto"):
            cls._attach_pool_placement(self_, object(), self._pools()[0])
        self.assertEqual(resolve.call_args.kwargs.get("consumer"), "cuda:1")
        self.assertEqual(resolve.call_args.kwargs.get("mode"), "auto")

        # quantizer without a loss device: runner device fallback (== primary
        # on real multi-GPU lanes, so the retarget stays unset there)
        cls, self_, runner = _fake_orchestrator(cache_device="cuda:0", mode="auto", iters=20, runner_device="cuda:1")
        with mock.patch(
            "auto_round.utils.pool_placement.resolve_placement_for_pool", return_value=None
        ) as resolve2, mock.patch("auto_round.utils.pool_placement.consolidate_pool_onto"):
            cls._attach_pool_placement(self_, object(), self._pools()[0])
        self.assertEqual(resolve2.call_args.kwargs.get("consumer"), "cuda:1")

        # iters=0: no tune loop, outputs follow the cache primary as before
        cls, self_, runner = _fake_orchestrator(cache_device="cuda:0", mode="auto", iters=0, runner_device="cuda:1")
        self_.alg_composer.block_quantizer._loss_device = "cuda:1"
        with mock.patch(
            "auto_round.utils.pool_placement.resolve_placement_for_pool", return_value=None
        ) as resolve, mock.patch("auto_round.utils.pool_placement.consolidate_pool_onto"):
            cls._attach_pool_placement(self_, object(), self._pools()[0])
        self.assertIsNone(resolve.call_args.kwargs.get("consumer"))

        # runner on the primary itself: degenerate, consumer stays unset
        cls, self_, runner = _fake_orchestrator(cache_device="cuda:0", mode="auto", iters=20, runner_device="cuda:0")
        with mock.patch(
            "auto_round.utils.pool_placement.resolve_placement_for_pool", return_value=None
        ) as resolve, mock.patch("auto_round.utils.pool_placement.consolidate_pool_onto"):
            cls._attach_pool_placement(self_, object(), self._pools()[0])
        self.assertIsNone(resolve.call_args.kwargs.get("consumer"))

    def test_host_placement_skips_consolidation_but_keeps_placement(self):
        # mode=cpu resolves a real placement parking pools on the host
        cls, self_, runner = _fake_orchestrator(cache_device="cuda:0", mode="cpu")
        placement = _FakePlacement(("cpu",))
        with mock.patch(
            "auto_round.utils.pool_placement.resolve_placement_for_pool", return_value=placement
        ), mock.patch("auto_round.utils.pool_placement.consolidate_pool_onto") as cons:
            cls._attach_pool_placement(self_, object(), self._pools()[0])
        cons.assert_not_called()
        self.assertIs(runner.pool_placement, placement)

    def test_single_device_plan_consolidates_with_full_reserve(self):
        cls, self_, runner = _fake_orchestrator(cache_device="cuda:0", mode="auto")
        placement = _FakePlacement(("cuda:0",))
        pools = self._pools()[0]
        with mock.patch(
            "auto_round.utils.pool_placement.resolve_placement_for_pool", return_value=placement
        ) as resolve, mock.patch("auto_round.utils.pool_placement.consolidate_pool_onto") as cons:
            cls._attach_pool_placement(self_, object(), pools)
        cons.assert_called_once()
        _, target, _block, _bs = cons.call_args.args[:4]
        reserved = cons.call_args.kwargs["reserved_bytes"]
        self.assertEqual(target, "cuda:0")
        # chains=1 (need_quanted_input False): reserve equals the pool bytes
        self.assertEqual(reserved, sum(t.numel() * t.element_size() for t in pools))
        self.assertIs(runner.pool_placement, placement)
        resolve.assert_called_once()


if __name__ == "__main__":
    unittest.main()
