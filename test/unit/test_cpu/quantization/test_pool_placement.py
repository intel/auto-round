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

"""Unit tests for --calibration_data_device calibration-data placement (policy + routing).

No CUDA required: free-memory probing and device candidates are injected, and the
BlockForwardRunner integration uses meta/cpu device targets to observe routing.
"""

import unittest
from unittest import mock

import torch

from auto_round.algorithms.block_runner import BlockForwardRunner
from auto_round.utils import pool_placement as pp


def _fake_probe(free_map):
    return lambda d: free_map.get(d)


class TestSpreadPlan(unittest.TestCase):
    def test_proportional_interleaved(self):
        plan = pp._spread_plan(["a", "b"], [6, 4], 10)
        self.assertEqual(len(plan), 10)
        self.assertEqual(plan.count("a"), 6)
        self.assertEqual(plan.count("b"), 4)
        # interleaved: no two consecutive chunks on the same device when both remain
        self.assertNotEqual(plan[0], plan[1])

    def test_zero_capacity_falls_back_to_devices(self):
        plan = pp._spread_plan(["a"], [0], 5)
        self.assertEqual(plan, ["a"])

    def test_single_device(self):
        plan = pp._spread_plan(["a"], [100], 5)
        self.assertEqual(plan, ["a"] * 5)


class TestPoolPlacement(unittest.TestCase):
    def test_counts_and_wrap(self):
        p = pp.PoolPlacement(["a", "b"], [5, 5], 4)
        self.assertEqual(p.counts(), {"a": 2, "b": 2})
        # index beyond plan wraps deterministically
        self.assertEqual(p.device_for_index(4), p.device_for_index(0))


class TestResolvePoolPlacement(unittest.TestCase):
    GB = 2**30

    def _resolve(self, mode="auto", free=None, pool=1 * GB, need=2 * GB, primary="cuda:0", candidates=None):
        free = free if free is not None else {"cuda:0": 20 * self.GB}
        candidates = candidates if candidates is not None else ["cuda:0", "cuda:1", "cuda:2"]
        return pp.resolve_pool_placement(pool, 128, primary, need, candidates, _fake_probe(free), mode=mode)

    def test_off_env_returns_none(self):
        self.assertIsNone(self._resolve(mode="off"))

    def test_cpu_mode_parks_on_host(self):
        plan = self._resolve(mode="cpu")
        self.assertEqual(plan.devices, ["cpu"])
        self.assertEqual(plan.counts(), {"cpu": 128})

    def test_need_estimator_first_principles(self):
        """Need = COMPUTED working allowance (2 batch IO generations + widest
        projection transient, floor-guarded) + reserve; iters>0 adds 14B/param
        of tuning state for candidate-homed parameters only. Pool bytes beyond
        one batch are NOT part of the need (the gates count them separately)."""
        reserve = pp._RESERVE_BYTES
        floor = int(0.125 * 2**30)
        # block with hidden 4 (modal in_features) and widest out 4
        m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
        pool = [torch.zeros(8, 4, dtype=torch.float32) for _ in range(2)]  # 2 samples of 32 floats
        esize = 4  # fp32 params
        batch_bytes = 2 * 32 * 4 // 2 * 2  # per-chain 256B, batch 2 of 2 -> 256B
        transient = int((32 * 2 / 4) * 4 * esize * 2)  # tokens=16, widest=4, x2 buffers
        pool_bytes = 2 * 32 * 4  # one chain, both samples
        expected = max(2 * batch_bytes + transient, floor) + reserve + 2 * pool_bytes  # + window term
        need0 = pp.placement_need_bytes(m, pool, 2)
        self.assertEqual(need0, expected)
        # iters=0: identical (no tuning state at zero-shot)
        self.assertEqual(pp.placement_need_bytes(m, pool, 2, iters=0, primary="cuda:0"), need0)
        # degenerate reads floor-guard + window: no block, bare-tensor pool
        self.assertEqual(pp.placement_need_bytes(None, torch.zeros(4), 2), floor + reserve + 2 * 4 * 4)
        # iters>0: +14 B per candidate-homed parameter; peers' params ignored
        n_params = sum(p.numel() for p in m.parameters())
        dev = str(next(m.parameters()).device)
        self.assertEqual(pp.placement_need_bytes(m, pool, 2, iters=20, primary=dev), need0 + n_params * 14)
        self.assertEqual(pp.placement_need_bytes(m, pool, 2, iters=20, primary="cuda:7"), need0)

    def test_working_allowance_scales_with_block_and_pool(self):
        """The allowance is model-derived, not flat: wider projections and
        bigger batches charge more; tiny blocks sit on the floor."""
        m_small = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
        m_wide = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8), torch.nn.Linear(8, 512))
        # structural helpers: widest projection + modal hidden detected
        self.assertEqual(pp._widest_out_and_hidden(m_small), (8, 8))
        self.assertEqual(pp._widest_out_and_hidden(m_wide), (512, 8))
        # real-scale case clears the floor: hidden 4096, widest 4096
        m_big = torch.nn.Sequential(torch.nn.Linear(4096, 4096), torch.nn.Linear(4096, 4096))
        pool = [torch.zeros(4096, 4096) for _ in range(2)]
        esize = 4  # fp32 params
        batch_bytes = 2 * 4096 * 4096 * esize  # per-chain pool = both samples
        tokens = 4096 * 2
        transient = int(tokens * 4096 * esize * 2)
        self.assertEqual(
            pp._working_allowance_bytes(m_big, pool, 2), max(2 * batch_bytes + transient, pp._WORKING_FLOOR_BYTES)
        )
        # half batch: smaller IO + transient terms
        pool4 = [torch.zeros(4096, 4096) for _ in range(4)]
        half = pp._working_allowance_bytes(m_big, pool4, 2)
        full = pp._working_allowance_bytes(m_big, pool4, 4)
        self.assertGreater(full, half)
        # tiny block: floor
        self.assertEqual(pp._working_allowance_bytes(None, [torch.zeros(2)], 1), pp._WORKING_FLOOR_BYTES)

    def test_consumer_retargets_single_away_from_cache_primary(self):
        """iters>0: the tune consumer (not the cache primary) is the
        single-device target; the primary demotes to a plain peer."""
        plan = pp.resolve_pool_placement(
            8 * self.GB,
            128,
            "cuda:0",
            int(3.5 * self.GB),
            ["cuda:0", "cuda:1", "cuda:2"],
            _fake_probe({"cuda:0": 20 * self.GB, "cuda:1": 12 * self.GB, "cuda:2": 11 * self.GB}),
            consumer="cuda:1",
        )
        self.assertEqual(plan.devices, ["cuda:1"])  # 12 - 3.5 >= 8 fits on the consumer

    def test_consumer_needs_headroom_like_the_primary_did(self):
        """The consumer is charged the working-set need: no room -> spread
        (with the consumer need-charged, the primary NOT charged)."""
        plan = pp.resolve_pool_placement(
            8 * self.GB,
            128,
            "cuda:0",
            int(10 * self.GB),  # consumer cannot host the pool beside its need
            ["cuda:0", "cuda:1", "cuda:2"],
            _fake_probe({"cuda:0": 20 * self.GB, "cuda:1": 12 * self.GB, "cuda:2": 11 * self.GB}),
            consumer="cuda:1",
        )
        counts = plan.counts()
        # need-floor: the consumer's capacity floors at ~zero, so it gets the
        # smallest share while the primary (now an uncharged peer) leads
        self.assertLess(counts.get("cuda:1", 0), counts.get("cuda:0", 0))
        self.assertLessEqual(counts.get("cuda:1", 0), counts.get("cuda:2", 0))

    def test_occupied_incoming_pool_blocks_fake_single_fit(self):
        """The single-device gate must count pool bytes already resident on the
        candidate -- without it the plan alternates single/spread across blocks
        (a single plan loads the device, the next resolve sees less free)."""
        base = {"cuda:0": 20 * self.GB, "cuda:1": 13 * self.GB, "cuda:2": 11 * self.GB}
        need = int(3.5 * self.GB)
        pool = 8 * self.GB
        free_fit = 13 * self.GB  # 13 - 3.5 >= 8: would look like a fit...
        with_occupied = pp.resolve_pool_placement(
            pool, 128, "cuda:0", need, ["cuda:0", "cuda:1", "cuda:2"], _fake_probe(base), consumer="cuda:1"
        )
        self.assertEqual(with_occupied.devices, ["cuda:1"])
        # ...until the incoming input pool (already on the consumer) is charged
        blocked = pp.resolve_pool_placement(
            pool,
            128,
            "cuda:0",
            need,
            ["cuda:0", "cuda:1", "cuda:2"],
            _fake_probe(base),
            consumer="cuda:1",
            occupied_bytes=free_fit - 0,  # charge the resident 13 GiB fully
        )
        # 13 - 3.5 - 13 < 8: single vetoed, consumer capacity floored
        counts = blocked.counts()
        self.assertLessEqual(counts.get("cuda:1", 0), counts.get("cuda:2", 0))
        self.assertLess(counts.get("cuda:1", 0), counts.get("cuda:0", 0))

    def test_occupied_incoming_pool_blocks_primary_single_too(self):
        """Regression (server block-2 OOM): a single plan on the cache primary
        must charge the input pool the PREVIOUS single plan left resident
        there -- otherwise every other block re-singles the loaded device."""
        need = int(2.4 * self.GB)
        pool = 4 * self.GB
        free_ok = pp.resolve_pool_placement(
            pool,
            128,
            "cuda:0",
            need,
            ["cuda:0", "cuda:1", "cuda:2"],
            _fake_probe({"cuda:0": 13 * self.GB, "cuda:1": 12 * self.GB, "cuda:2": 11 * self.GB}),
        )
        self.assertEqual(free_ok.devices, ["cuda:0"])  # 13 - 2.4 >= 4: fits
        loaded = pp.resolve_pool_placement(
            pool,
            128,
            "cuda:0",
            need,
            ["cuda:0", "cuda:1", "cuda:2"],
            _fake_probe({"cuda:0": 8 * self.GB, "cuda:1": 12 * self.GB, "cuda:2": 11 * self.GB}),
            occupied_bytes=4 * self.GB,  # previous block's single plan left it here
        )
        # 8 - 2.4 - 4 < 4: single vetoed; the primary's spread share is floored
        self.assertLessEqual(loaded.counts().get("cuda:0", 0), loaded.counts().get("cuda:2", 0))

    def test_degenerate_consumer_falls_back_to_primary(self):
        """consumer == primary or non-cuda consumer: primary-first as before."""
        plan = pp.resolve_pool_placement(
            8 * self.GB,
            128,
            "cuda:0",
            int(3.5 * self.GB),
            ["cuda:0", "cuda:1"],
            _fake_probe({"cuda:0": 12 * self.GB, "cuda:1": 12 * self.GB}),
            consumer="cuda:0",  # same as primary
        )
        self.assertEqual(plan.devices, ["cuda:0"])
        plan2 = pp.resolve_pool_placement(
            8 * self.GB,
            128,
            "cuda:0",
            int(3.5 * self.GB),
            ["cuda:0", "cuda:1"],
            _fake_probe({"cuda:0": 12 * self.GB, "cuda:1": 12 * self.GB}),
            consumer="cpu",  # non-cuda: ignored
        )
        self.assertEqual(plan2.devices, ["cuda:0"])

    def test_huge_moe_need_still_shards_onto_peers(self):
        """Regression: the need constrains the primary alone, never the fleet.

        A MoE block's need runs to hundreds of GiB under the x7 expert
        multiplier; charging it against total free vetoed every shard plan
        ('insufficient: total - need < pool'), parking the pools on the
        busiest device and OOM-ing the next block.
        """
        plan = self._resolve(
            pool=4 * self.GB,
            need=342 * self.GB,
            free={"cuda:0": 2 * self.GB, "cuda:1": 10 * self.GB, "cuda:2": 10 * self.GB},
        )
        self.assertIsNotNone(plan)
        counts = plan.counts()
        self.assertEqual(counts.get("cuda:0", 0), 0)  # need-floor leaves no chunks on the primary
        self.assertGreater(counts.get("cuda:1", 0), 0)
        self.assertGreater(counts.get("cuda:2", 0), 0)
        self.assertEqual(sum(counts.values()), 128)

    def test_zero_peer_capacity_even_over_commit(self):
        # Old contract returned None (-> primary concentration, the worse
        # failure). Now: when the charged headrooms cannot hold the pool,
        # the water-fill spreads the over-commitment EVENLY (same level t
        # below every device's charged headroom) -- the placement that
        # demonstrably completed blocks on the 95%-utilization lane.
        plan = self._resolve(
            pool=4 * self.GB,
            need=2 * self.GB,
            free={"cuda:0": 1 * self.GB, "cuda:1": 1 * self.GB},
        )
        self.assertIsNotNone(plan)
        counts = plan.counts()
        self.assertEqual(
            sum(
                counts.values(),
            ),
            128,
        )
        self.assertFalse(any(str(d).startswith("cpu") for d in plan.devices))

    def test_cpu_primary_returns_none(self):
        self.assertIsNone(self._resolve(primary="cpu"))

    def test_fits_primary_stays_primary(self):
        plan = self._resolve(pool=4 * self.GB, free={"cuda:0": 20 * self.GB})
        self.assertEqual(plan.devices, ["cuda:0"])
        self.assertEqual(plan.counts(), {"cuda:0": 128})

    def test_margin_blocks_primary_placement(self):
        # 8 GiB pool, 8 free, 2 GiB margin -> only 6 usable on primary: not enough,
        # peers absorb the rest while the margin still protects the primary
        plan = self._resolve(
            pool=8 * self.GB,
            free={"cuda:0": 8 * self.GB, "cuda:1": 10 * self.GB, "cuda:2": 10 * self.GB},
        )
        self.assertEqual(sum(plan.counts().values()), 128)
        self.assertLessEqual(plan.counts().get("cuda:0", 0), 64)

    def test_water_fill_minimizes_max_predicted_usage(self):
        # headroom (8, 10, 30) GiB, pool 8: min-max fills from the deepest
        # device down to a common level -> everything on cuda:2 (level 22),
        # cuda:0/1 untouched: all three end with >= 8 GiB predicted headroom
        # and the fleet's max predicted usage is minimized
        plan = self._resolve(
            pool=8 * self.GB,
            free={"cuda:0": 8 * self.GB, "cuda:1": 10 * self.GB, "cuda:2": 30 * self.GB},
        )
        counts = plan.counts()
        self.assertEqual(counts.get("cuda:2", 0), 128)
        self.assertNotIn("cuda:0", counts)
        self.assertNotIn("cuda:1", counts)
        self.assertEqual(sum(counts.values()), 128)

    def test_activation_charges_shift_pool_off_moe_peers(self):
        # peers homing experts carry routed transients the state charge never
        # priced (measured 3.2 GiB/peer): charging them shrinks peer headroom
        # -> water-fill moves pool bytes onto the (uncharged) entry device
        from auto_round.utils.pool_placement import resolve_pool_placement

        GB = 2**30

        def _plan(act):
            return resolve_pool_placement(
                8 * GB,
                128,
                "cuda:0",
                5 * GB,  # working set: single-device rung declines (12-5 < 8)
                ["cuda:0", "cuda:2", "cuda:3"],
                lambda d: {"cuda:0": 12 * GB, "cuda:2": 13 * GB, "cuda:3": 13 * GB}[d],
                peer_state_bytes={"cuda:2": 4 * GB, "cuda:3": 4 * GB},
                activation_bytes=act,
            )

        uncharged = _plan(None)
        charged = _plan({"cuda:0": 1.6 * GB, "cuda:2": 4.3 * GB, "cuda:3": 4.3 * GB})
        cu, cc = uncharged.counts(), charged.counts()
        # uncharged: h = (7, 9, 9) -> level 5.67 -> (1.3, 3.3, 3.3)
        # charged:   h = (5.4, 2.7, 2.7) -> level 0.93 -> (4.5, 1.8, 1.8)
        self.assertGreater(cc["cuda:0"], cu["cuda:0"])
        self.assertLess(cc["cuda:2"], cu["cuda:2"])
        self.assertLess(cc["cuda:3"], cu["cuda:3"])
        self.assertEqual(sum(cc.values()), 128)

    def test_water_fill_spreads_when_headrooms_comparable(self):
        # frees (8, 10) GiB with the helper's default need=2 charged to the
        # primary: headroom (6, 10), pool 9 (exceeds the primary alone, so the
        # single-device rung declines): level = (16-9)/2 = 3.5 -> cuda:1 takes
        # 6.5, cuda:0 takes 2.5: both end at exactly 3.5 GiB predicted
        # headroom -- equalized max VRAM by construction
        plan = self._resolve(pool=9 * self.GB, free={"cuda:0": 8 * self.GB, "cuda:1": 10 * self.GB})
        counts = plan.counts()
        self.assertEqual(sum(counts.values()), 128)
        self.assertGreater(counts.get("cuda:1", 0), counts.get("cuda:0", 0))
        self.assertAlmostEqual(plan.level_bytes / self.GB, 3.5, delta=0.05)

    def test_insufficient_capacity_even_over_commit(self):
        # No silent CPU fallback ever; and instead of None (whose caller-side
        # default concentrates the pool on the primary), an over-fleet pool
        # is spread by the water-fill with the over-commitment shared evenly
        # across charged headrooms (charges shape the split in both regimes).
        plan = self._resolve(
            pool=100 * self.GB,
            free={"cuda:0": 8 * self.GB, "cuda:1": 10 * self.GB, "cuda:2": 10 * self.GB},
        )
        self.assertIsNotNone(plan)
        counts = plan.counts()
        self.assertEqual(sum(counts.values()), 128)
        self.assertFalse(any(str(d).startswith("cpu") for d in plan.devices))

    def test_forced_csv_overrides_candidates(self):
        plan = self._resolve(
            mode="cuda:7,cuda:3",
            free={"cuda:7": 10 * self.GB, "cuda:3": 10 * self.GB},
        )
        self.assertEqual(set(plan.devices), {"cuda:7", "cuda:3"})

    def test_forced_bare_indices_get_cuda_prefix(self):
        # accelerate device_map style: "1,2" means cuda:1,cuda:2
        plan = self._resolve(
            mode="1,2",
            free={"cuda:1": 10 * self.GB, "cuda:2": 10 * self.GB},
        )
        self.assertEqual(set(plan.devices), {"cuda:1", "cuda:2"})

    def test_probe_none_skips_device(self):
        plan = self._resolve(
            pool=8 * self.GB,
            free={"cuda:0": 8 * self.GB, "cuda:1": 10 * self.GB, "cuda:2": None},
        )
        self.assertNotIn("cuda:2", plan.counts())


class TestPoolBytes(unittest.TestCase):
    def test_tensor_bytes_nested(self):
        t = torch.zeros(2048, 4096, dtype=torch.float32)
        pool = {"hidden_states": [t, t], "mask": t}
        self.assertEqual(pp._tensor_bytes(pool), 3 * t.numel() * 4)

    def test_chunk_count(self):
        pool = {"hidden_states": [torch.zeros(1) for _ in range(7)]}
        self.assertEqual(pp._pool_chunk_count(pool), 7)


class TestCalibDataLine(unittest.TestCase):
    def test_bytes_by_device_counts_referenced_leaves(self):
        a = torch.zeros(10)  # 40B
        b = torch.zeros(6)  # 24B
        per_dev, total = pp._bytes_by_device({"h": [a, b], "m": a})
        self.assertEqual(per_dev, {"cpu": 104})  # 'm' references a again
        self.assertEqual(total, 104)

    def test_monitor_grammar(self):
        fp = [torch.zeros(1, 64, 64) for _ in range(4)]  # 64KiB
        q = [torch.zeros(1, 64, 64) for _ in range(4)]
        plan = pp.PoolPlacement(["cpu", "cpu"], [1, 1], 8)
        line = pp.calib_data_line([fp, q], None, plan, 64 * 64 * 4 * 8, 8, "cpu")
        self.assertIn("'input': 0.00GB", line)
        self.assertIn("'output': 0.00GB", line)
        self.assertIn("'aux': 0.00GB", line)
        self.assertIn("'per_device': {'cpu':", line)
        self.assertNotIn("cuda", line)

    def test_short_device_keys(self):
        self.assertEqual(pp._short_device_key("cuda:3"), "3")
        self.assertEqual(pp._short_device_key("cpu"), "cpu")

    def test_plan_devices_short_keys_and_fallback(self):
        plan = pp.PoolPlacement(["cuda:0", "cuda:1"], [1, 1], 2)
        line = pp.calib_data_line([], None, plan, 4 * 2**30, 2, "cuda:0")
        self.assertIn("'0':", line)
        self.assertIn("'1':", line)
        # no plan -> outputs land on the primary
        line2 = pp.calib_data_line([], None, None, 2**30, 1, "cuda:0")
        self.assertIn("'0': 1.00GB", line2)


class TestConsolidate(unittest.TestCase):
    class _OnDevice(torch.Tensor):
        # real tensor (bytes/isinstance work) with an overridden device view
        @property
        def device(self):
            return torch.device(self._fake_device)

    def _tensor_on(self, dev):
        t = torch.zeros(2).as_subclass(self._OnDevice)
        t._fake_device = dev
        return t

    def _ctx(self, free_gib, need_gib):
        return (
            mock.patch.object(pp, "placement_need_bytes", return_value=int(need_gib * 2**30)),
            mock.patch(
                "auto_round.utils.device.probe_usable_bytes",
                return_value=int(free_gib * 2**30),
            ),
        )

    def test_local_when_already_on_target(self):
        pool = [self._tensor_on("cuda:0") for _ in range(3)]
        with self._ctx(10, 0)[0], self._ctx(10, 0)[1]:
            with mock.patch.object(pp, "_move_pool_to") as mv:
                self.assertEqual(pp.consolidate_pool_onto([pool], "cuda:0", object(), 8), "local")
        mv.assert_not_called()

    def test_consolidated_moves_in_place(self):
        pool = [self._tensor_on("cuda:1"), self._tensor_on("cuda:2")]
        with self._ctx(10, 0)[0], self._ctx(10, 0)[1]:
            with mock.patch.object(pp, "_move_pool_to") as mv:
                self.assertEqual(pp.consolidate_pool_onto([pool], "cuda:0", object(), 8), "consolidated")
        mv.assert_any_call(pool, "cuda:0")

    def test_spread_when_not_fitting(self):
        pool = [self._tensor_on("cuda:1")]
        with self._ctx(10, 9)[0], self._ctx(10, 9)[1]:
            with mock.patch.object(pp, "_tensor_bytes", return_value=5 * 2**30), mock.patch.object(
                pp, "_move_pool_to"
            ) as mv:
                self.assertEqual(pp.consolidate_pool_onto([pool], "cuda:0", object(), 8), "spread")
        mv.assert_not_called()

    def test_consolidation_reachable_under_first_principles_need(self):
        """Regression: the old 2x-pool need made attach-time consolidation
        mathematically unreachable (need ~= 2x pool + reserve always exceeded
        any free next to the pools it was gating). With pools counted by the
        gate's own terms, a genuine fit consolidates again."""
        GBi = 2**30
        pool = [torch.zeros(1) for _ in range(4)]
        # 4 GiB of pool tensors, 8 GiB free on the target cuda:0,
        # need 3.5 GiB (working + reserve)
        with mock.patch.object(pp, "_tensor_bytes", return_value=4 * GBi), mock.patch.object(
            pp, "_move_pool_to"
        ) as mv, mock.patch("auto_round.utils.device.probe_usable_bytes", return_value=8 * GBi), mock.patch.object(
            pp, "placement_need_bytes", return_value=int(3.5 * GBi)
        ):
            self.assertEqual(pp.consolidate_pool_onto([pool], "cuda:0", object(), 8), "consolidated")
        mv.assert_called()
        # genuinely tight: free - need < pool -> spread, never moved
        with mock.patch.object(pp, "_tensor_bytes", return_value=4 * GBi), mock.patch.object(
            pp, "_move_pool_to"
        ) as mv2, mock.patch(
            "auto_round.utils.device.probe_usable_bytes", return_value=int(7 * GBi)
        ), mock.patch.object(
            pp, "placement_need_bytes", return_value=int(3.5 * GBi)
        ):
            self.assertEqual(pp.consolidate_pool_onto([pool], "cuda:0", object(), 8), "spread")
        mv2.assert_not_called()

    def test_non_cuda_target_spread(self):
        self.assertEqual(pp.consolidate_pool_onto([[torch.zeros(1)]], "cpu", object(), 8), "spread")

    def test_empty_is_local(self):
        self.assertEqual(pp.consolidate_pool_onto([None, []], "cuda:0", object(), 8), "local")


class TestRunnerRouting(unittest.TestCase):
    def _runner(self):
        r = BlockForwardRunner(batch_dim=0, batch_size=2, device="cpu", cache_device="cpu", enable_torch_compile=False)
        r.block_forward = lambda block, hidden, others, amp, amp_dtype, dev, x: hidden
        return r

    def test_placement_routes_per_sample_outputs(self):
        r = self._runner()
        r.pool_placement = pp.PoolPlacement(["meta", "cpu"], [1, 1], 4)
        inputs = [torch.zeros(1, 1) for _ in range(4)]
        outs = r.forward(object(), inputs, {})
        types = [o.device.type for o in outs]
        self.assertEqual(types, ["meta", "cpu", "meta", "cpu"])

    def test_no_placement_keeps_cache_device(self):
        r = self._runner()
        self.assertIsNone(getattr(r, "pool_placement", None))
        inputs = [torch.zeros(1, 1) for _ in range(4)]
        outs = r.forward(object(), inputs, {})
        self.assertTrue(all(o.device.type == "cpu" for o in outs))

    def test_indices_mode_ignores_placement(self):
        # indices-mode concatenates outputs: mixed devices would break the cat,
        # so placement must not apply there
        r = self._runner()
        r.pool_placement = pp.PoolPlacement(["meta", "cpu"], [1, 1], 4)
        inputs = [torch.zeros(1, 1) for _ in range(4)]
        out = r.forward(object(), inputs, {}, indices=torch.tensor([0, 1]))
        self.assertEqual(out.device.type, "cpu")

    def test_mixed_device_batch_gathers_before_cat(self):
        # regression: _select_batch cat'd per-sample tensors on their (mixed)
        # park devices before the compute-device move -> torch.cat crash.
        # Duck-typed tensors (a CPU box has no second real device; meta cannot
        # be copied out of).
        r = self._runner()

        class _FakeT:
            def __init__(self, dev):
                self.device = torch.device(dev)
                self.moved_to = None

            def to(self, target):
                self.moved_to = target
                return torch.zeros(2, 1)

        mixed = [_FakeT("cpu"), _FakeT("meta"), _FakeT("cpu")]
        out = r._gather_same_device(mixed, torch.device("cpu"))
        self.assertTrue(all(t.moved_to == torch.device("cpu") for t in mixed))
        self.assertTrue(all(o.device.type == "cpu" for o in out))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a real second device")
    def test_mixed_device_batch_select_integration(self):
        # per-sample tensors carry a leading batch dim in production
        # ([1, seq, hidden]); _select_batch cats them along batch_dim
        r = self._runner()
        r.device = "cuda:0"
        inputs = [torch.zeros(1, 2, 1, device="cpu")] * 2 + [torch.zeros(1, 2, 1, device="cuda:0")] * 2
        sel = r._select_batch(inputs, {"m": [torch.zeros(1, 2, 1) for _ in range(4)]}, torch.tensor([0, 1, 2, 3]))
        self.assertEqual(sel[0].device.type, "cuda")
        self.assertEqual(sel[0].shape[0], 4)  # 4 selected samples, batch dim leading

    def test_structured_values_pass_through_gather(self):
        # diffusion inputs (e.g. freqs) can be nested lists; the gather helper
        # must not assume tensors and crash on .device (CI: prefetched batch)
        r = self._runner()
        vals = [[torch.zeros(1)], [torch.zeros(1)]]
        self.assertIs(r._gather_same_device(vals, torch.device("meta")), vals)

    def test_uniform_device_batch_untouched(self):
        r = self._runner()
        a = torch.zeros(2, 1)
        inputs = [a, a.clone(), a.clone(), a.clone()]
        out = r._gather_same_device(inputs, torch.device("meta"))
        self.assertIs(out[0], a)  # same objects, no copies

    def test_explicit_cache_device_call_overrides_placement(self):
        r = self._runner()
        r.pool_placement = pp.PoolPlacement(["meta", "cpu"], [1, 1], 4)
        inputs = [torch.zeros(1, 1) for _ in range(4)]
        outs = r.forward(object(), inputs, {}, cache_device="cpu")
        self.assertTrue(all(o.device.type == "cpu" for o in outs))


if __name__ == "__main__":
    unittest.main()
