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
"""CPU-tier tests for the ported single-process DDP tuning surface.

Covers the device-agnostic contract: shard-constrained samplers, plan
resolution with injected free-VRAM data, transport encoding math, pool
distribution layout, gather-on-CPU paths, env defaults, and the
block-runner device-safe concatenation. Mirror/replica behavior on real
CUDA devices lives in the CUDA tier (test_ddp_mirror.py + e2e).
"""

import logging

import pytest
import torch

from auto_round.algorithms.block_runner import _cat_device_safe
from auto_round.algorithms.parallel.data_parallel import (
    _transport_segment,
    distribute_pool,
    parallel_state,
    resolve_ddp_plan,
    resolve_parallel_world,
)


def _with_policy(q, world=2):
    """Attach a ParallelPolicy-bearing compress_context to a fake quantizer
    (the policy rides on the compress context; env vars are gone)."""
    from types import SimpleNamespace

    from auto_round.algorithms.parallel.data_parallel import ParallelPolicy

    q.compress_context = SimpleNamespace(parallel_policy=ParallelPolicy(world=world))
    return q


class TestResolveDDPPlan:
    def _plan(self, world, free, footprint=100):
        return resolve_ddp_plan(
            world,
            torch.device("cuda", 0),  # device OBJECTS only -- no CUDA runtime touched
            8,
            visible_devices=[0, 1, 2, 3],
            vram_free_bytes={torch.device("cuda", i): free[i] for i in range(4)},
            mirror_footprint_bytes=footprint,
            margin_bytes=0,  # toy byte values in these tests
        )

    def test_full_world_when_vram_fits(self):
        plan = self._plan(4, [1000, 1000, 1000, 1000])
        assert plan.enabled and plan.world == 4
        assert plan.shard_size == 2

    def test_device_without_mirror_fit_is_kept_with_warning_note(self):
        plan = self._plan(4, [1000, 10, 1000, 1000])  # cuda:1 estimate too small
        assert plan.enabled
        assert torch.device("cuda", 1) in plan.devices
        assert plan.world == 4  # the VRAM check is advisory: the requested world stays
        assert any("low free VRAM on" in n and "device kept" in n for n in plan.notes)

    def test_all_devices_low_vram_still_engages_full_world(self):
        plan = self._plan(4, [5, 5, 5, 5], footprint=100)
        assert plan.enabled and plan.world == 4
        assert sum(1 for n in plan.notes if "low free VRAM on" in n) == 3  # mirrors only, not home

    def test_batch_smaller_than_world_demotes_to_pow2(self):
        from auto_round.algorithms.parallel.data_parallel import resolve_ddp_plan as _rdp

        def _resolve(batch, world):
            return _rdp(
                world,
                torch.device("cuda", 0),
                batch,
                visible_devices=[0, 1, 2, 3],
                vram_free_bytes={torch.device("cuda", i): 1 << 40 for i in range(4)},
                mirror_footprint_bytes=1,
                margin_bytes=0,
            )

        # batch 1 cannot shard: serial (world collapses to 1; the reason
        # goes to the log)
        p = _resolve(1, 4)
        assert p.world == 1
        # batch 3, world 4 -> halved to 2 (3 >= 2, one replica gets 2 samples)
        p = _resolve(3, 4)
        assert p.world == 2 and any("world set to 2" in str(n) for n in p.notes)
        # batch 6, world 4 -> no demotion (6 >= 4)
        p = _resolve(6, 4)
        assert p.world == 4
        # batch divisible by world stays
        p = _resolve(8, 4)
        assert p.world == 4

    def test_batch_demotion_is_exact_when_pow2_floor_off(self):
        from auto_round.algorithms.parallel.data_parallel import resolve_ddp_plan as _rdp

        def _resolve(batch, world):
            return _rdp(
                world,
                torch.device("cuda", 0),
                batch,
                visible_devices=[0, 1, 2, 3],
                vram_free_bytes={torch.device("cuda", i): 1 << 40 for i in range(4)},
                mirror_footprint_bytes=1,
                margin_bytes=0,
                pow2_floor=False,
            )

        # iters=0 worlds carry no pow2 constraint: the demotion is exact
        p = _resolve(3, 4)
        assert p.world == 3 and any("world set to 3" in str(n) for n in p.notes)
        p = _resolve(6, 3)
        assert p.world == 3  # no demotion
        p = _resolve(1, 4)
        assert p.world == 1  # serial: a replica needs at least one sample

    def test_non_power_of_two_request_resolves_but_caller_gates(self):
        # resolve itself is policy-free: a non-pow2 world with enough
        # visible devices resolves enabled; the engagement resolver and the
        # CLI policy entry enforce the pow2 rule only for iters > 0 (the
        # gradient exchange needs it; iters=0 runs have no exchange)
        plan = resolve_ddp_plan(
            3,
            torch.device("cuda", 0),
            12,
            visible_devices=[0, 1, 2],
            vram_free_bytes={torch.device("cuda", i): 1 << 30 for i in range(3)},
            mirror_footprint_bytes=1,
            margin_bytes=0,
        )
        assert plan.world == 3 and plan.enabled


class TestResolveParallelWorld:
    """Policy entry: off|auto|N resolution (the CLI delegates here)."""

    def test_off_returns_one(self):
        assert resolve_parallel_world("off", 8) == 1
        assert resolve_parallel_world(None, 8) == 1
        assert resolve_parallel_world("", 8) == 1

    def test_auto_takes_largest_pow2_fitting_visible_at_iters_above_zero(self):
        assert resolve_parallel_world("auto", 8, iters=10) == 8
        assert resolve_parallel_world("auto", 6, iters=10) == 4
        assert resolve_parallel_world("auto", 5, iters=10) == 4

    def test_auto_takes_all_visible_at_iters_zero(self):
        # no gradient exchange runs at iters=0, so every visible device works
        assert resolve_parallel_world("auto", 6, iters=0) == 6
        assert resolve_parallel_world("auto", 8, iters=0) == 8

    def test_auto_needs_two_visible(self):
        with pytest.raises(RuntimeError, match="at least 2 visible CUDA devices"):
            resolve_parallel_world("auto", 1)

    def test_explicit_valid_pow2(self):
        assert resolve_parallel_world("2", 4) == 2
        assert resolve_parallel_world("4", 4) == 4

    def test_explicit_non_pow2_rejected_at_iters_above_zero(self):
        with pytest.raises(RuntimeError, match="power of two"):
            resolve_parallel_world("6", 8, iters=10)

    def test_explicit_non_pow2_allowed_at_iters_zero(self):
        assert resolve_parallel_world("6", 8, iters=0) == 6
        assert resolve_parallel_world("3", 3, iters=0) == 3

    def test_explicit_below_two_rejected(self):
        with pytest.raises(RuntimeError, match="at least 2"):
            resolve_parallel_world("1", 8)

    def test_explicit_over_visible_fails_loudly(self):
        with pytest.raises(RuntimeError, match="requires at least 8 visible CUDA devices, got 4"):
            resolve_parallel_world("8", 4)

    def test_garbage_rejected(self):
        with pytest.raises(RuntimeError, match=r"accepts off\|auto\|N"):
            resolve_parallel_world("yes", 8)


class TestVisibleCountShortfall:
    def test_visible_fewer_than_requested_world_reports_shortfall(self):
        plan = resolve_ddp_plan(
            4,
            torch.device("cuda", 0),
            8,
            visible_devices=[0, 1],  # home + one other: fewer than the requested world
            vram_free_bytes=None,
            mirror_footprint_bytes=None,
        )
        assert not plan.enabled
        assert any("fewer than the requested world" in n for n in plan.notes)


class TestTransportMath:
    def test_fp32_passthrough(self):
        t = torch.randn(16)
        out = _transport_segment(t, t.device)
        assert out is t

    def test_cross_device_copy_is_dtype_preserving(self):
        t = torch.randn(16) if not torch.cuda.is_available() else torch.randn(16, device="cuda:0")
        out = _transport_segment(t, torch.device("cpu"))
        assert out.dtype == t.dtype and out.shape == t.shape


class TestDistributePool:
    def test_plan_layout_covers_pool_exactly(self):
        # distribute_pool gives device r the contiguous range [r*shard,(r+1)*shard);
        # on uniform CPU devices that is a no-op, so pin the layout arithmetic
        # through the plan the pool follows
        from auto_round.algorithms.parallel.data_parallel import resolve_ddp_plan

        plan = resolve_ddp_plan(
            4,
            torch.device("cuda", 0),
            8,
            visible_devices=[0, 1, 2, 3],  # avoids torch.cuda.device_count() on CUDA-less hosts
            vram_free_bytes={torch.device("cuda", i): 1 << 40 for i in range(4)},
            mirror_footprint_bytes=1,
            margin_bytes=0,
        )
        assert plan.world == 4
        assert plan.shard_size * plan.world == 8
        # device objects in the plan are cuda:N (unusable on CUDA-less hosts);
        # exercise the scatter loop with the degenerate all-cpu device list
        distribute_pool([torch.zeros(2) for _ in range(8)], [torch.device("cpu")] * 4)
        # indivisible / too-small pools are left alone by contract (serial cats handle them)
        distribute_pool([torch.zeros(2)], [torch.device("cpu")] * 4)


class TestSingleDevicePlacement:
    """Parallel tuning tunes whole-block mirrors: a block whose weights span
    several CUDA devices fails the resolver (validation, never a gather)."""

    @staticmethod
    def _param(dev, n=8):
        from types import SimpleNamespace

        return SimpleNamespace(device=torch.device(dev), numel=lambda: n, element_size=lambda: 4)

    @staticmethod
    def _quantizer():
        from types import SimpleNamespace

        q = SimpleNamespace(iters=10, gradient_accumulate_steps=1, enable_lfq=False)
        q._get_scaler = lambda: None
        return q

    def test_block_spanning_two_cuda_devices_declines(self, monkeypatch):
        from types import SimpleNamespace

        from auto_round.algorithms.parallel.data_parallel import resolve_tune_ddp_plan_

        block = SimpleNamespace(parameters=lambda: iter([self._param("cuda:0"), self._param("cuda:1")]))
        with pytest.raises(RuntimeError, match="spans 2 CUDA devices"):
            resolve_tune_ddp_plan_(_with_policy(self._quantizer()), block, [torch.zeros(1)], None, "cuda:0")

    def test_non_cuda_accelerator_home_is_eligible(self, monkeypatch):
        """cuda/xpu/hpu homes are eligible; a fake xpu home with explicit
        devices resolves a plan on CPU (no xpu runtime touched)."""
        from auto_round.algorithms.parallel import data_parallel as dp
        from auto_round.algorithms.parallel.data_parallel import resolve_ddp_plan

        free_map = dp._accel_free_bytes_map("xpu")  # None on a box without xpu
        block = SimpleNamespace(
            parameters=lambda: iter(
                [SimpleNamespace(device=torch.device("xpu", 0), numel=lambda: 8, element_size=lambda: 4)]
            ),
            modules=lambda: iter([]),
        )
        q = SimpleNamespace(iters=10, gradient_accumulate_steps=1, enable_lfq=False)
        q._get_scaler = lambda: None
        # plan resolution itself works for an xpu home with visible devices
        plan = resolve_ddp_plan(
            2,
            torch.device("xpu", 0),
            2,
            visible_devices=[0, 1],
            vram_free_bytes=free_map,
            mirror_footprint_bytes=1,
        )
        assert plan.enabled and plan.world == 2
        assert all(d.type == "xpu" for d in plan.devices)
        # and the full resolver reports (decline, without raising) only on
        # real blockers:
        # an xpu home on a box without xpu resolves without the CUDA-only error
        try:
            resolved = dp.resolve_tune_ddp_plan_(_with_policy(q), block, [torch.zeros(1)], None, "xpu:0")
        except RuntimeError as e:
            assert "not a supported accelerator" not in str(e)
            resolved = None
        if resolved is not None:
            assert resolved.world >= 1

    def test_cpu_resident_weights_pass_span_rule(self, monkeypatch):
        """CPU-pinned subtrees alongside the CUDA home are legal placement."""
        from types import SimpleNamespace

        from auto_round.algorithms.parallel.data_parallel import resolve_tune_ddp_plan_

        block = SimpleNamespace(
            parameters=lambda: iter([self._param("cpu"), self._param("cuda:0")]),
            modules=list,
        )
        # the span rule passes; the resolve then proceeds (it may decline
        # later on the free-VRAM probe, which is CPU-safe) -- the assertion
        # checks the decline list for the placement reason
        try:
            resolve_tune_ddp_plan_(_with_policy(self._quantizer()), block, [torch.zeros(1)], None, "cuda:0")
        except RuntimeError as e:
            assert "spans" not in str(e)

    def test_iters0_engages_non_pow2_world(self, monkeypatch):
        """A world=3 plan engages when the tune loop is empty (iters=0):
        no gradient exchange runs, so the halving-doubling pow2 rule lifts."""
        from auto_round.algorithms.parallel import data_parallel as dp

        q = SimpleNamespace(iters=0, gradient_accumulate_steps=1, enable_lfq=False)
        q._get_scaler = lambda: None
        _with_policy(q, world=3)
        monkeypatch.setattr(dp, "_accel_device_count", lambda _t: 3)
        monkeypatch.setattr(
            dp,
            "_accel_free_bytes_map",
            lambda _t: {torch.device("cuda", i): 1 << 40 for i in range(3)},
        )
        block = SimpleNamespace(
            parameters=lambda: iter(
                [SimpleNamespace(device=torch.device("cuda", 0), numel=lambda: 8, element_size=lambda: 4)]
            ),
            modules=lambda: iter([]),
        )
        plan = dp.resolve_tune_ddp_plan_(q, block, [torch.zeros(1)] * 3, None, "cuda:0", log=False)
        assert plan.enabled and plan.world == 3

    def test_iters_above_zero_declines_non_pow2_world(self, monkeypatch):
        """The same world=3 with a live tune loop raises: the gradient
        exchange (recursive halving-doubling) needs a power-of-two world."""
        from auto_round.algorithms.parallel import data_parallel as dp

        q = SimpleNamespace(iters=10, gradient_accumulate_steps=1, enable_lfq=False)
        q._get_scaler = lambda: None
        _with_policy(q, world=3)
        monkeypatch.setattr(dp, "_accel_device_count", lambda _t: 3)
        monkeypatch.setattr(
            dp,
            "_accel_free_bytes_map",
            lambda _t: {torch.device("cuda", i): 1 << 40 for i in range(3)},
        )
        block = SimpleNamespace(
            parameters=lambda: iter(
                [SimpleNamespace(device=torch.device("cuda", 0), numel=lambda: 8, element_size=lambda: 4)]
            ),
            modules=lambda: iter([]),
        )
        with pytest.raises(RuntimeError, match="power of two"):
            dp.resolve_tune_ddp_plan_(q, block, [torch.zeros(1)] * 3, None, "cuda:0", log=False)

    def test_ddp_engagement_raises_dynamo_cache_limit_by_world(self, monkeypatch):
        """Engaging a parallel plan bumps the dynamo cache limit to base x world."""
        from auto_round.algorithms.parallel import data_parallel as dp
        from auto_round.utils import device as _dev

        calls = []
        monkeypatch.setattr(_dev, "_bump_dynamo_cache_limit", lambda n=None: calls.append(n))
        q = SimpleNamespace(iters=10, gradient_accumulate_steps=1, enable_lfq=False)
        q._get_scaler = lambda: None
        _with_policy(q)
        # fake a two-device visible set (this host has at most one CUDA device)
        monkeypatch.setattr(dp, "_accel_device_count", lambda _t: 2)
        monkeypatch.setattr(
            dp, "_accel_free_bytes_map", lambda _t: {torch.device("cuda", 0): 1 << 40, torch.device("cuda", 1): 1 << 40}
        )
        block = SimpleNamespace(
            parameters=lambda: iter(
                [SimpleNamespace(device=torch.device("cuda", 0), numel=lambda: 8, element_size=lambda: 4)]
            ),
            modules=lambda: iter([]),
        )
        plan = dp.resolve_tune_ddp_plan_(q, block, [torch.zeros(1)] * 2, None, "cuda:0", log=False)
        assert plan.world == 2
        assert calls and calls[-1] == 16 * 2  # default env base 16 x engaged world

    def test_accumulation_engages(self, monkeypatch):
        """gradient_accumulate_steps > 1 no longer gates the lane."""
        from auto_round.algorithms.parallel import data_parallel as dp
        from auto_round.algorithms.parallel.data_parallel import resolve_tune_ddp_plan_

        q = SimpleNamespace(iters=10, gradient_accumulate_steps=2, enable_lfq=False)
        q._get_scaler = lambda: None
        _with_policy(q)
        # fake a two-device visible set (this host has at most one CUDA device)
        monkeypatch.setattr(dp, "_accel_device_count", lambda _t: 2)
        monkeypatch.setattr(
            dp, "_accel_free_bytes_map", lambda _t: {torch.device("cuda", 0): 1 << 40, torch.device("cuda", 1): 1 << 40}
        )
        block = SimpleNamespace(
            parameters=lambda: iter(
                [SimpleNamespace(device=torch.device("cuda", 0), numel=lambda: 8, element_size=lambda: 4)]
            ),
            modules=lambda: iter([]),
        )
        plan = resolve_tune_ddp_plan_(q, block, [torch.zeros(1)] * 2, None, "cuda:0", log=False)
        # engages with a requested world; the accumulation gate is gone
        assert plan.world == 2
        assert not any("accumulate" in str(n) for n in plan.notes)

    def test_accumulated_grad_sign_matches_serial(self):
        """Two replicas, sum-reduced shard losses; the full-value exchange
        leaves each replica with the averaged gradient -- a positive rescale
        of the serial accumulated (summed) gradient, so sign-SGD updates and
        momentum directions match serial."""
        from auto_round.algorithms.parallel import data_parallel as dp

        world = 2
        # replicas hold the SAME parameters (mirrors); each accumulates the
        # sum-reduced gradient over its own data shard. Shard grads derived
        # analytically: grad of (x[shard]**2).sum() lives on the shard slice.
        x = torch.randn(16, requires_grad=False)
        grad0 = torch.zeros(16)
        grad0[:8] = 2 * x[:8]
        grad1 = torch.zeros(16)
        grad1[8:] = 2 * x[8:]
        bufs = [grad0.clone(), grad1.clone()]
        dp.halving_doubling_allreduce(bufs, scale=1.0 / world)
        serial_accumulated = 2 * x  # serial: grads summed over both shards
        for buf in bufs:
            # exchanged buffer x world == serial accumulated gradient
            assert torch.allclose(buf * world, serial_accumulated, atol=1e-6)


class TestCatDeviceSafe:
    def test_same_device_is_plain_cat(self):
        parts = [torch.zeros(2, 3), torch.ones(2, 3)]
        out = _cat_device_safe(parts, dim=0)
        assert torch.equal(out, torch.cat(parts, dim=0))

    def test_empty_selection_raises(self):
        with pytest.raises(ValueError):
            _cat_device_safe([], dim=0)


class TestPolicyDefaults:
    """The policy rides on the compress context; a quantizer without one is
    serial (off), and the resolver reads the policy's world."""

    def test_quantizer_without_context_defaults_to_off(self):
        from types import SimpleNamespace

        from auto_round.algorithms.parallel.data_parallel import _quantizer_policy

        assert _quantizer_policy(SimpleNamespace()).world == 1

    def test_policy_world_round_trip(self):
        from types import SimpleNamespace

        from auto_round.algorithms.parallel.data_parallel import ParallelPolicy, _quantizer_policy

        q = _with_policy(SimpleNamespace(), world=4)
        assert _quantizer_policy(q).world == 4
        assert isinstance(_quantizer_policy(q), ParallelPolicy)

    def test_declared_lfq_constraint_declines(self):
        """Eligibility constraints are declared by the quantizer
        (parallel_constraints); LFQ is a real blocker."""
        from types import SimpleNamespace

        import torch

        from auto_round.algorithms.parallel.data_parallel import resolve_tune_ddp_plan_

        q = _with_policy(SimpleNamespace(iters=10, gradient_accumulate_steps=1))
        q.parallel_constraints = lambda: {"enable_lfq": True}
        block = torch.nn.Sequential(torch.nn.Linear(4, 4))
        with pytest.raises(RuntimeError, match="enable_lfq"):
            resolve_tune_ddp_plan_(q, block, [torch.zeros(1)], None, "cpu")

    def test_base_quantizer_declares_constraints(self):
        """BaseQuantizer.parallel_constraints reports LFQ without the resolver
        introspecting quantizer internals; a grad scaler is no constraint."""
        from auto_round.algorithms.quantization.base import BaseQuantizer

        q = BaseQuantizer.__new__(BaseQuantizer)
        assert q.parallel_constraints() == {"enable_lfq": False}
        q.enable_lfq = True
        assert q.parallel_constraints()["enable_lfq"] is True


class TestTupleKwargSlicing:
    """transformers-v5 rope arrives as position_embeddings=(cos, sin) with a
    per-sample batch dim; shard/batch forwards smaller than the cached batch
    must slice tuple-of-tensors kwargs elementwise or crash in rope."""

    def _runner(self, n=8, batch_size=4):
        from auto_round.algorithms.block_runner import BlockForwardRunner

        inputs = [torch.randn(1, 6, 4) for _ in range(n)]
        return (
            BlockForwardRunner(
                batch_dim=0, batch_size=batch_size, device=torch.device("cpu"), cache_device="cpu", amp=False
            ),
            inputs,
        )

    def test_tuple_kwargs_sliced_to_batch_indices(self):
        runner, inputs = self._runner()
        cos = torch.randn(8, 6, 2)
        sin = torch.randn(8, 6, 2)
        _, others = runner.select_batch(inputs, {"position_embeddings": (cos, sin)}, [3, 5])
        assert isinstance(others["position_embeddings"], tuple)
        assert others["position_embeddings"][0].shape == (2, 6, 2)
        assert others["position_embeddings"][1].shape == (2, 6, 2)
        assert torch.equal(others["position_embeddings"][0][0], cos[3])
        assert torch.equal(others["position_embeddings"][0][1], cos[5])

    def test_broadcast_tuple_elements_pass_through(self):
        runner, inputs = self._runner()
        table = torch.randn(1, 6, 2)  # broadcast-shaped: not per-sample
        _, others = runner.select_batch(inputs, {"position_embeddings": (table, table)}, [3, 5])
        assert others["position_embeddings"][0] is table  # unsliced, broadcastable

    def test_mixed_and_non_tensor_tuples_untouched(self):
        runner, inputs = self._runner()
        val = (torch.randn(8, 6, 2), "flag")
        _, others = runner.select_batch(inputs, {"weird": val}, [3, 5])
        assert others["weird"] is val  # mixed tuple treated as opaque


class TestSharedCacheShardPick:
    """shared_cache_keys kwargs arrive as ONE ENTRY PER BATCH (list[n_pool/batch]).
    A sub-batch draw (DDP shard) must pick the owning batch's entry and slice its
    within-batch rows; full-batch draws keep the legacy pick exactly."""

    def _runner(self, batch_size=8):
        from auto_round.algorithms.block_runner import BlockForwardRunner

        inputs = [torch.randn(1, 6, 4) for _ in range(128)]
        return (
            BlockForwardRunner(
                batch_dim=0,
                batch_size=batch_size,
                device=torch.device("cpu"),
                cache_device="cpu",
                amp=False,
                shared_cache_keys=("position_embeddings",),
            ),
            inputs,
        )

    def _pe(self):
        # 16 batch entries, each an (cos, sin) tuple of [8, S, D] per-sample tensors
        return [(torch.full((8, 6, 2), float(b)), torch.full((8, 6, 2), -float(b))) for b in range(16)]

    def test_shard_picks_owning_batch_and_rows(self):
        runner, inputs = self._runner()
        pe = self._pe()
        _, others = runner.select_batch(inputs, {"position_embeddings": pe}, [32, 33])
        cos, sin = others["position_embeddings"]
        assert cos.shape == (2, 6, 2) and sin.shape == (2, 6, 2)
        assert torch.all(cos == 4.0) and torch.all(sin == -4.0)  # entry 4, rows 0-1

    def test_shard_wrapping_rows_across_batch_boundary_fails_safe(self):
        runner, inputs = self._runner()
        pe = self._pe()
        _, others = runner.select_batch(inputs, {"position_embeddings": pe}, [7, 8])
        # batch 0 entry (indices[0]=7), rows 7 and 0 -- within-entry slice
        cos, _ = others["position_embeddings"]
        assert cos.shape == (2, 6, 2) and torch.all(cos == 0.0)

    def test_full_batch_draw_keeps_legacy_pick(self):
        runner, inputs = self._runner()
        pe = self._pe()
        _, others = runner.select_batch(inputs, {"position_embeddings": pe}, list(range(8, 16)))
        # legacy: multi-index full-batch -> val[0], untouched (batch 0's entry, 8 rows)
        cos, _ = others["position_embeddings"]
        assert isinstance(cos, torch.Tensor) and cos.shape[0] == 8 and torch.all(cos == 0.0)

    def test_single_index_sub_batch_draw_per_batch_list(self):
        runner, inputs = self._runner()
        pe = self._pe()
        _, others = runner.select_batch(inputs, {"position_embeddings": pe}, [5])
        # per-batch list: sample 5 -> batch 0 entry, row 5
        cos, _ = others["position_embeddings"]
        assert cos.shape == (1, 6, 2) and torch.all(cos == 0.0)

    def test_per_sample_list_sub_batch_draw(self):
        runner, inputs = self._runner()
        # per-sample layout: 128 entries, each a [1, 6, 2] tensor
        ps = [torch.full((1, 6, 2), float(i)) for i in range(128)]
        _, others = runner.select_batch(inputs, {"position_embeddings": ps}, [32, 33])
        assert others["position_embeddings"].shape == (2, 6, 2)
        assert torch.all(others["position_embeddings"][0] == 32.0)

    @pytest.mark.parametrize("rank_batch", [1, 2])
    def test_rank_runner_with_its_batch_share(self, rank_batch):
        # a thread rank tunes with its share of the batch (8 // world), while
        # the cache was collected 8 per entry: the collection batch comes from
        # the cache, not from the rank runner's batch_size (regression: at 8
        # ranks a 1-row draw got a whole 8-row entry)
        runner, inputs = self._runner(batch_size=rank_batch)
        pe = self._pe()
        draw = [37, 38][:rank_batch]
        _, others = runner.select_batch(inputs, {"position_embeddings": pe}, draw)
        cos, sin = others["position_embeddings"]
        assert cos.shape == (rank_batch, 6, 2) and torch.all(cos == 4.0) and torch.all(sin == -4.0)

    def test_broadcast_entry_passes_through(self):
        # [1, S, D] tables broadcast over the batch: never row-sliced (on CUDA an
        # out-of-range index_select is a device-side assert, not an exception)
        runner, inputs = self._runner(batch_size=1)
        pe = [(torch.full((1, 6, 2), float(b)), torch.full((1, 6, 2), -float(b))) for b in range(16)]
        _, others = runner.select_batch(inputs, {"position_embeddings": pe}, [37])
        cos, _ = others["position_embeddings"]
        assert cos.shape == (1, 6, 2) and torch.all(cos == 4.0)


class TestLoggingGlobals:
    def test_engaged_log_globals_initialized(self):
        """Regression: the env-strip once dropped the module-level
        ``_ENGAGED_LOGGED_SIG`` init, and the ``global`` read in
        resolve_tune_ddp_plan_ then NameError'd -- but only on GPU runs,
        because a CPU home declines before the sig block (CPU tests cannot
        reach the engaged path)."""
        from auto_round.algorithms.parallel import data_parallel as dp
        from auto_round.algorithms.quantization.search_dispatch import _ENGAGED_LOGGED

        assert dp._ENGAGED_LOGGED_SIG is None
        # _coll_mirror_setup_logged was removed with its once-per-process
        # INFO line; search_dispatch._ENGAGED_LOGGED remains the registry
        assert isinstance(_ENGAGED_LOGGED, set)


class TestActivationPricing:
    """Mirror pricing charges the per-forward activation working set on top
    of the weights (estimate_tuning_block_mem's SIX-value return contract).

    Regression: the rebase onto main left the DDP seam unpacking the
    estimator's pre-main FOUR-tuple, so every block logged ``activation
    pricing skipped (too many values to unpack)`` at INFO and the plan
    budget silently dropped to weights-only."""

    @staticmethod
    def _block():
        layer = torch.nn.Linear(32, 32, bias=False)
        layer.global_name = "m.blk.proj"
        layer.to_quantized = True
        layer.bits = 4
        layer.act_bits = 16
        return torch.nn.Sequential(layer)

    @staticmethod
    def _quantizer():
        q = SimpleNamespace(iters=10, gradient_accumulate_steps=1, enable_lfq=False)
        q._get_scaler = lambda: None
        return _with_policy(q)

    def test_pricing_succeeds_and_adds_bytes_to_the_budget(self, monkeypatch, caplog, _autoround_log_propagate):
        """The resolver's pricing call runs (no skip warning) and its bytes
        enter the mirror budget: cuda:1 fits the weights-only mirror (plus
        the 2GiB margin) but misses once the activation charge (>= the flat
        1GiB ``additional_memory``) lands on top."""
        from auto_round.algorithms.parallel import data_parallel as dp

        block = self._block()
        weight_bytes = sum(p.numel() * p.element_size() for p in block.parameters())
        free = {
            torch.device("cuda", 0): 1 << 40,
            # weights + margin + 64MiB slack: passes weights-only pricing,
            # fails once >=1GiB activation bytes are charged
            torch.device("cuda", 1): weight_bytes + (2 << 30) + (64 << 20),
        }
        monkeypatch.setattr(dp, "_accel_device_count", lambda _t: 2)
        monkeypatch.setattr(dp, "_accel_free_bytes_map", lambda _t: free)
        inputs = [torch.randn(1, 16, 32) for _ in range(2)]
        with caplog.at_level("WARNING"):
            plan = dp.resolve_tune_ddp_plan_(self._quantizer(), block, inputs, None, "cuda:0", log=False)
        assert plan.enabled and plan.world == 2
        assert not any("activation pricing skipped" in r.message for r in caplog.records)
        assert any(
            "low free VRAM on cuda:1" in n for n in plan.notes
        ), "activation bytes never entered the budget (weights-only pricing)"

    def test_estimator_failure_is_a_warning(self, monkeypatch, caplog, _autoround_log_propagate):
        """A failed estimate is a degraded state (weights-only budget): the
        skip line must be WARNING-level, not buried at INFO."""
        from auto_round.algorithms.parallel import data_parallel as dp
        from auto_round.utils import device as _dev

        def _boom(*_a, **_k):
            raise RuntimeError("estimator exploded")

        monkeypatch.setattr(_dev, "estimate_tuning_block_mem", _boom)
        monkeypatch.setattr(dp, "_accel_device_count", lambda _t: 2)
        monkeypatch.setattr(
            dp, "_accel_free_bytes_map", lambda _t: {torch.device("cuda", i): 1 << 40 for i in range(2)}
        )
        inputs = [torch.randn(1, 16, 32) for _ in range(2)]
        with caplog.at_level("WARNING"):
            plan = dp.resolve_tune_ddp_plan_(self._quantizer(), self._block(), inputs, None, "cuda:0", log=False)
        assert plan.enabled and plan.world == 2
        assert any("activation pricing skipped" in r.message and r.levelno == logging.WARNING for r in caplog.records)


class TestEarlyDeviceMapEligibility:
    """F3: the multi-device-map ineligibility fires at policy resolution
    (before model load) with the same message the runtime resolver raises at
    block 0; the runtime check remains the second line of defense."""

    def test_multi_device_map_with_live_loop_raises_before_load(self):
        from auto_round.algorithms.parallel.data_parallel import check_parallel_device_map_compat_

        with pytest.raises(RuntimeError, match="spans 2 CUDA devices; parallel tuning requires"):
            check_parallel_device_map_compat_(4, {"a": "cuda:0", "b": "cuda:1"}, iters=10)

    def test_message_lists_the_fixes(self):
        from auto_round.algorithms.parallel.data_parallel import check_parallel_device_map_compat_

        with pytest.raises(RuntimeError, match="--parallel_quantization off"):
            check_parallel_device_map_compat_(4, {"a": "cuda:0", "b": "cuda:1"}, iters=10)

    def test_single_device_map_passes(self):
        from auto_round.algorithms.parallel.data_parallel import check_parallel_device_map_compat_

        check_parallel_device_map_compat_(4, {"a": "cuda:0"}, iters=10)  # one device
        check_parallel_device_map_compat_(4, "0", iters=10)  # CLI default
        check_parallel_device_map_compat_(4, None, iters=10)  # no map

    def test_iters0_and_world1_are_not_cli_blocked(self):
        from auto_round.algorithms.parallel.data_parallel import check_parallel_device_map_compat_

        multi = {"a": "cuda:0", "b": "cuda:1"}
        check_parallel_device_map_compat_(4, multi, iters=0)  # no tune loop: runtime decides
        check_parallel_device_map_compat_(1, multi, iters=10)  # lane off

    def test_cpu_entries_do_not_count(self):
        from auto_round.algorithms.parallel.data_parallel import check_parallel_device_map_compat_

        check_parallel_device_map_compat_(4, {"a": "cpu", "b": "cpu"}, iters=10)

    def test_cli_policy_resolution_calls_the_early_check(self):
        """Wiring: the CLI calls the early check right after resolving the
        world, before the model loads."""
        import inspect

        from auto_round.cli import main as cli_main

        src = inspect.getsource(cli_main)
        assert "check_parallel_device_map_compat_" in src
        assert src.index("check_parallel_device_map_compat_(") > src.index("resolve_parallel_world(")

    def test_runtime_span_check_still_exists(self):
        """Second line of defense: the resolver's per-block span check."""
        from types import SimpleNamespace

        from auto_round.algorithms.parallel.data_parallel import resolve_tune_ddp_plan_

        block = SimpleNamespace(
            parameters=lambda: iter(
                [
                    SimpleNamespace(device=torch.device("cuda:0"), numel=lambda: 8, element_size=lambda: 4),
                    SimpleNamespace(device=torch.device("cuda:1"), numel=lambda: 8, element_size=lambda: 4),
                ]
            )
        )
        q = SimpleNamespace(iters=10, gradient_accumulate_steps=1, enable_lfq=False)
        q._get_scaler = lambda: None
        with pytest.raises(RuntimeError, match="spans 2 CUDA devices"):
            resolve_tune_ddp_plan_(_with_policy(q), block, [torch.zeros(1)], None, "cuda:0")


class TestZeroShotParallelNotice:
    """F4: an explicit --parallel_quantization N on the zero-shot RTN path
    (no calibration forward) emits exactly ONE INFO line; auto/off stay
    silent; nothing refuses to run."""

    def test_explicit_request_logs_one_info_line(self, caplog, _autoround_log_propagate):
        from auto_round.algorithms.parallel.data_parallel import ParallelPolicy, log_zero_shot_parallel_noop_

        with caplog.at_level("INFO"):
            log_zero_shot_parallel_noop_(ParallelPolicy(world=4, source="explicit"))
        infos = [r for r in caplog.records if "has no effect on this run" in r.message]
        assert len(infos) == 1 and infos[0].levelno == logging.INFO
        assert "--parallel_quantization 4" in infos[0].message
        assert "zero-shot RTN needs no calibration forward" in infos[0].message
        assert "quantizing serially" in infos[0].message

    def test_auto_and_off_are_silent(self, caplog, _autoround_log_propagate):
        from auto_round.algorithms.parallel.data_parallel import ParallelPolicy, log_zero_shot_parallel_noop_

        with caplog.at_level("INFO"):
            log_zero_shot_parallel_noop_(ParallelPolicy(world=8, source="auto"))
            log_zero_shot_parallel_noop_(ParallelPolicy(world=1, source="off"))
            log_zero_shot_parallel_noop_(None)
        assert not any("has no effect" in r.message for r in caplog.records)

    def test_zero_shot_path_wires_the_notice(self):
        """Wiring: _quantize_zero_shot consults the compress-context policy."""
        import inspect

        from auto_round.compressors import orchestrator

        src = inspect.getsource(orchestrator.CompressionOrchestrator._quantize_zero_shot)
        assert "log_zero_shot_parallel_noop_" in src


class TestPeakVramReplicaDevices:
    """F5: the final peak-VRAM summary must cover DDP replica devices engaged
    outside the device map (the monitor's callers pass the device-map list)."""

    def test_registered_devices_are_sampled_beyond_the_caller_list(self, monkeypatch, _memory_monitor_clean):
        from auto_round.utils import device as dev_mod

        class _FakeMgr:
            type = "cuda"

            @staticmethod
            def is_available():
                return True

            @staticmethod
            def memory_reserved(i):
                return (i + 1) * 1024**3  # cuda:i reports (i+1) GiB reserved

        mm = _memory_monitor_clean
        mm.register_extra_devices([torch.device("cuda", 3)])
        monkeypatch.setattr(dev_mod, "get_current_device_manager", lambda: _FakeMgr())
        mm.update(["cuda:0"])  # the device-map list, WITHOUT the replica device
        assert "0" in mm.peak_vram and mm.peak_vram["0"] == 1.0
        assert "3" in mm.peak_vram and mm.peak_vram["3"] == 4.0  # replica now visible
        assert "cuda:3" in mm.get_summary() or "'3': 4.0GB" in mm.get_summary()

    def test_registration_survives_reset_and_cpu_entries_are_ignored(self, _memory_monitor_clean):
        mm = _memory_monitor_clean
        mm.register_extra_devices([torch.device("cuda", 2), torch.device("cpu"), None])
        assert mm._extra_devices == {"cuda:2"}
        mm.reset()
        assert mm._extra_devices == {"cuda:2"}  # peaks reset, registration kept

    def test_engaging_plan_registers_its_devices(self, monkeypatch, _memory_monitor_clean):
        from auto_round.algorithms.parallel import data_parallel as dp

        mm = _memory_monitor_clean
        monkeypatch.setattr(dp, "_accel_device_count", lambda _t: 2)
        monkeypatch.setattr(
            dp, "_accel_free_bytes_map", lambda _t: {torch.device("cuda", i): 1 << 40 for i in range(2)}
        )
        block = SimpleNamespace(
            parameters=lambda: iter(
                [SimpleNamespace(device=torch.device("cuda", 0), numel=lambda: 8, element_size=lambda: 4)]
            ),
            modules=lambda: iter([]),
        )
        q = SimpleNamespace(iters=10, gradient_accumulate_steps=1, enable_lfq=False)
        q._get_scaler = lambda: None
        _with_policy(q)
        plan = dp.resolve_tune_ddp_plan_(q, block, [torch.zeros(1)] * 2, None, "cuda:0", log=False)
        assert plan.enabled
        assert mm._extra_devices == {"cuda:0", "cuda:1"}

    def test_serial_plan_registers_nothing(self, _memory_monitor_clean):
        from auto_round.algorithms.parallel.data_parallel import resolve_ddp_plan

        assert resolve_ddp_plan(1, torch.device("cuda", 0), 8).world == 1
        assert _memory_monitor_clean._extra_devices == set()


class TestPerfPrepSpan:
    """F6: the per-block phases line must account for the region between the
    reload mark and the first tick (materialize + dtype convert +
    dispatch_block placement + pool-placement attach) -- ~8s/block on big
    MoE blocks that tqdm counted but total= missed."""

    def test_prep_span_opens_before_attach_and_prints(self):
        import inspect

        from auto_round.compressors import orchestrator

        src = inspect.getsource(orchestrator.CompressionOrchestrator._quantize_blocks)
        first_prep = src.index('_perf.tick("prep")')
        first_attach = src.index('_perf.tick("attach")')
        assert first_prep < first_attach, "the measured chain must open with the prep span"
        assert "prep=" in src

    def test_each_tick_opens_the_region_it_names(self):
        """tick(key) closes the running span and opens ``key``: each tick must
        sit right before the code it names (a tick placed after its region
        shifts every printed label one phase late)."""
        import inspect

        from auto_round.compressors import orchestrator

        src = inspect.getsource(orchestrator.CompressionOrchestrator._quantize_blocks)
        chain = [
            ('_perf.tick("prep")', "materialize_model_(m)"),
            ('_perf.tick("attach")', "self._attach_pool_placement("),
            ('_perf.tick("compress")', "self.alg_composer.compress_block("),
            ('_perf.tick("swap")', "if q_input is not None:"),
            ('_perf.tick("sweep")', "_mm._deferred_search_inputs = None"),
            ('_perf.tick("hooks")', "accelerate.hooks.remove_hook_from_submodules(m)"),
            ('_perf.tick("pack")', "if self.compress_context.is_immediate_packing:"),
            ('_perf.tick("mv")', "mv_module_from_gpu(m)"),
            ('_perf.tick("clear")', "clear_memory(device_list=device_manager.device_list)"),
            ('_perf.tick("monitor")', "memory_monitor.log_summary()"),
            ('_perf.tick("write")', "_bg_writer.dispatch(_bg_write)"),
        ]
        pos = 0
        for tick, region in chain:
            t = src.index(tick, pos)
            r = src.index(region, t)
            assert (
                src.find("_perf.tick(", t + len(tick)) > r or src.find("_perf.tick(", t + len(tick)) == -1
            ), f"{tick} must directly precede {region!r}"
            pos = r


class TestEngagedLineBatchMode:
    """F7: the engaged line logs the batch mode and the effective global batch."""

    def test_engaged_line_reports_batch_mode_and_effective_batch(self, monkeypatch, caplog, _autoround_log_propagate):
        """F7: the engaged line logs the batch mode and the effective global
        batch (per_replica scales it by the world)."""
        from auto_round.algorithms.parallel import data_parallel as dp
        from auto_round.algorithms.parallel.data_parallel import ParallelPolicy

        monkeypatch.setattr(dp, "_accel_device_count", lambda _t: 2)
        monkeypatch.setattr(
            dp, "_accel_free_bytes_map", lambda _t: {torch.device("cuda", i): 1 << 40 for i in range(2)}
        )
        block = SimpleNamespace(
            parameters=lambda: iter(
                [SimpleNamespace(device=torch.device("cuda", 0), numel=lambda: 8, element_size=lambda: 4)]
            ),
            modules=lambda: iter([]),
        )

        def _q(mode):
            q = SimpleNamespace(
                iters=10,
                gradient_accumulate_steps=1,
                enable_lfq=False,
                calibration_context=SimpleNamespace(batch_size=8),
            )
            q._get_scaler = lambda: None
            q.compress_context = SimpleNamespace(parallel_policy=ParallelPolicy(world=2, batch_mode=mode))
            return q

        with caplog.at_level("INFO"):
            dp._ENGAGED_LOGGED_SIG = None  # the engaged line dedups by signature
            dp.resolve_tune_ddp_plan_(_q("split"), block, [torch.zeros(1)] * 32, None, "cuda:0", log=True)
            dp._ENGAGED_LOGGED_SIG = None
            dp.resolve_tune_ddp_plan_(_q("per_replica"), block, [torch.zeros(1)] * 32, None, "cuda:0", log=True)
        engaged = [r.message for r in caplog.records if "[tune-ddp] engaged:" in r.message]
        assert len(engaged) == 2
        assert "batch=split" in engaged[0] and "effective global batch=8" in engaged[0]
        assert "batch=per_replica" in engaged[1] and "effective global batch=16" in engaged[1]


class TestSyncMirrorsToHome:
    """iters=0 RTN/OptRTN under DDP: the quantizer runs main's code on the HOME
    block and never touches the pool; the engine then syncs the quantized home
    state onto the mirrors (sync_mirrors_to_home, called from the composer's
    _settle_block_pool) so the Step-6 cascade still runs sharded."""

    @staticmethod
    def _block():
        class _L(torch.nn.Linear):
            def __init__(self, i):
                super().__init__(4, 4, bias=False)
                self.global_name = f"m.blk.l{i}"
                self.to_quantized = True
                self.bits = 4
                self.sym = True
                self.data_type = "int"
                self.group_size = 4
                self.act_bits = 16
                self.imatrix = torch.rand(4) + 0.5
                self.imatrix_cnt = 1

        class _B(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l0 = _L(0)
                self.l1 = _L(1)

            def forward(self, x):
                return self.l0(x) + self.l1(x)

        return _B()

    @staticmethod
    def _ddp_quantizer(quantizer_cls, block, with_pool=True):
        from types import SimpleNamespace

        from auto_round.algorithms.parallel.data_parallel import MirrorPool

        devices = [torch.device("cpu", 0), torch.device("cpu", 1)]
        q = quantizer_cls.__new__(quantizer_cls)
        q._BaseAlgorithm__run_ctx = None  # unbound: q.model is None (the driver is stubbed)
        pool = MirrorPool(block, devices) if with_pool else None
        q.parallel_state = SimpleNamespace(plan=SimpleNamespace(world=2, devices=devices), pool=pool, pool_used=False)
        return q, pool

    @staticmethod
    def _qdq(layer):
        with torch.no_grad():  # a REAL quantize-dequantize transform on the home layer
            layer.weight.data = ((layer.weight.data * 4).round() / 4) + 0.125

    @staticmethod
    def _mirror_pairs(block, pool):
        for n, m in block.named_modules():
            if hasattr(m, "global_name"):
                for r in range(pool.world):
                    if pool.reps[r] is not block:
                        yield pool.mirror_layer(r, n), m

    def test_optrtn_quantize_block_stays_on_home(self, monkeypatch):
        import auto_round.algorithms.quantization.rtn.batched_search as _bs
        from auto_round.algorithms.quantization.rtn.quantizer import OptimizedRTNQuantizer

        block = self._block()
        home_ids = {id(m) for _, m in block.named_modules() if hasattr(m, "global_name")}
        q, pool = self._ddp_quantizer(OptimizedRTNQuantizer, block)
        staged_on = []

        def _staged(layer, disable_opt_rtn=None, defer_search=False):
            staged_on.append((id(layer), defer_search))
            return layer

        def _driver(model, staged, max_batch=None):
            for _name, layer in staged:
                self._qdq(layer)

        q._quantize_layer_via_rtn = _staged
        monkeypatch.setattr(_bs, "run_batched_rtn_search", _driver)
        q.quantize_block(block, None, {}, None, None, None)
        # main's staged lane ran on the HOME layers only, and the quantizer
        # left the pool alone (the sync is the engine's job)
        assert len(staged_on) == 2
        assert all(iid in home_ids and deferred for iid, deferred in staged_on)
        assert q.parallel_state.pool_used is False
        assert all(not torch.equal(lm.weight.data, hm.weight.data) for lm, hm in self._mirror_pairs(block, pool))

    def test_sync_pushes_home_state_and_marks_pool_consumed(self):
        from auto_round.algorithms.parallel.mirror_sync import sync_mirrors_to_home
        from auto_round.algorithms.quantization.rtn.quantizer import RTNQuantizer

        block = self._block()
        q, pool = self._ddp_quantizer(RTNQuantizer, block)
        for _n, m in block.named_modules():
            if hasattr(m, "global_name"):
                self._qdq(m)
                m.scale = torch.full((4, 1), 0.25)
        assert sync_mirrors_to_home(q, block) is True
        for lm, hm in self._mirror_pairs(block, pool):
            assert torch.equal(lm.weight.data, hm.weight.data)
            assert torch.equal(lm.scale, hm.scale)
        assert q.parallel_state.pool_used is True
        assert not any(hasattr(m, "imatrix") for _n, m in block.named_modules() if hasattr(m, "global_name"))

    def test_sync_without_aligned_pool_is_a_noop(self):
        from auto_round.algorithms.parallel.data_parallel import MirrorPool
        from auto_round.algorithms.parallel.mirror_sync import sync_mirrors_to_home
        from auto_round.algorithms.quantization.rtn.quantizer import RTNQuantizer

        block = self._block()
        q, _pool = self._ddp_quantizer(RTNQuantizer, block, with_pool=False)
        assert sync_mirrors_to_home(q, block) is False
        # a pool built for another block must not be written either
        other = self._block()
        q.parallel_state.pool = MirrorPool(other, q.parallel_state.plan.devices)
        assert sync_mirrors_to_home(q, block) is False
        assert q.parallel_state.pool_used is False
        assert all(hasattr(m, "imatrix") for _n, m in block.named_modules() if hasattr(m, "global_name"))

    def test_composer_settle_syncs_rtn_family_and_keeps_pool(self):
        from auto_round.algorithms.composer import AlgorithmComposer
        from auto_round.algorithms.quantization.rtn.quantizer import RTNQuantizer

        block = self._block()
        q, pool = self._ddp_quantizer(RTNQuantizer, block)
        for _n, m in block.named_modules():
            if hasattr(m, "global_name"):
                self._qdq(m)
        released = []
        composer = AlgorithmComposer.__new__(AlgorithmComposer)
        composer.block_quantizer = q
        composer._coll_ctx = type("_Ctx", (), {"release_pool": lambda self: released.append(1)})()
        assert composer._settle_block_pool(block, pool) is pool
        assert released == []
        for lm, hm in self._mirror_pairs(block, pool):
            assert torch.equal(lm.weight.data, hm.weight.data)

    def test_composer_settle_releases_pool_for_other_quantizers(self):
        from auto_round.algorithms.composer import AlgorithmComposer
        from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer

        block = self._block()
        q, pool = self._ddp_quantizer(SignRoundQuantizer, block)
        released = []
        composer = AlgorithmComposer.__new__(AlgorithmComposer)
        composer.block_quantizer = q
        composer._coll_ctx = type("_Ctx", (), {"release_pool": lambda self: released.append(1)})()
        # not the RTN family and nothing consumed the pool: no sync, released
        assert composer._settle_block_pool(block, pool) is None
        assert released == [1]
        assert q.parallel_state.pool_used is False


class TestRequestedWorldErrors:
    """A requested parallel world is a requirement: infeasible -> RuntimeError, never silent serial."""

    def _fake_quantizer(self):
        from types import SimpleNamespace

        q = SimpleNamespace(
            iters=10,
            gradient_accumulate_steps=1,
            enable_lfq=False,
        )
        q._get_scaler = lambda: None
        return q

    def test_infeasible_world_raises(self, monkeypatch):
        import torch

        from auto_round.algorithms.parallel.data_parallel import resolve_tune_ddp_plan_

        block = torch.nn.Sequential(torch.nn.Linear(4, 4))
        with pytest.raises(RuntimeError, match="ineligible"):
            resolve_tune_ddp_plan_(_with_policy(self._fake_quantizer()), block, [torch.zeros(1)], None, "cpu")

    def test_iters0_is_not_a_decline_reason(self, monkeypatch):
        """The DDP world shards the collection at iters=0 too (campaign
        semantics restored): with a fake quantizer at iters=0 the only
        ineligibility reason left must be the non-CUDA home, never iters."""
        import torch

        from auto_round.algorithms.parallel.data_parallel import resolve_tune_ddp_plan_

        q = self._fake_quantizer()
        q.iters = 0
        block = torch.nn.Sequential(torch.nn.Linear(4, 4))
        with pytest.raises(RuntimeError) as excinfo:
            resolve_tune_ddp_plan_(_with_policy(q), block, [torch.zeros(1)], None, "cpu")
        assert "iters" not in str(excinfo.value)
        assert "not a supported accelerator" in str(excinfo.value)

    def test_no_policy_stays_serial(self, monkeypatch):
        import torch

        from auto_round.algorithms.parallel.data_parallel import resolve_tune_ddp_plan_

        block = torch.nn.Sequential(torch.nn.Linear(4, 4))
        plan = resolve_tune_ddp_plan_(self._fake_quantizer(), block, [torch.zeros(1)], None, "cpu")
        assert plan.world == 1


@pytest.fixture()
def _memory_monitor_clean():
    """Isolate the process-singleton memory monitor's peaks and extra-device
    registration so tests are order-independent."""
    from auto_round.utils.device import memory_monitor

    saved = (
        dict(memory_monitor.peak_vram),
        float(memory_monitor.peak_ram),
        set(getattr(memory_monitor, "_extra_devices", set())),
    )
    memory_monitor.peak_vram = {}
    memory_monitor.peak_ram = 0.0
    memory_monitor._extra_devices = set()
    yield memory_monitor
    memory_monitor.peak_vram, memory_monitor.peak_ram = saved[0], saved[1]
    memory_monitor._extra_devices = saved[2]


@pytest.fixture()
def _autoround_log_propagate():
    """Temporarily enable propagation on the ``autoround`` logger so pytest's
    caplog fixture (handler at the root logger) can capture warnings; the
    logger is configured with propagate=False in production."""
    import logging

    logger = logging.getLogger("autoround")
    original = logger.propagate
    logger.propagate = True
    yield
    logger.propagate = original


def _expert_layer(name):
    """An exact-nn.Linear module that satisfies _split_expert_batches' gates."""
    m = torch.nn.Linear(4, 4)
    m.global_name = name
    m.to_quantized = True
    m.bits = 4
    m.group_size = 4
    m.sym = True  # sym batches under BOTH neuqi and plain opt-RTN; plain asym has no search
    m.act_bits = 16
    return m


def _expert_moe_block():
    """4 same-key experts + one dense module: the standard sharded-search fixture."""

    class _B(torch.nn.Module):
        def __init__(self):
            super().__init__()
            mods = [
                _expert_layer("m.blk.experts.0.gate_proj"),
                _expert_layer("m.blk.experts.1.gate_proj"),
                _expert_layer("m.blk.experts.2.up_proj"),
                _expert_layer("m.blk.experts.3.up_proj"),
            ]
            dense = _expert_layer("m.blk.mlp.down_proj")
            for i, m in enumerate(mods + [dense]):
                setattr(self, f"m{i}", m)

    return _B()


class TestMirrorPoolUnit:
    """MirrorPool construction, hook sync, and release."""

    def _block_with_hook(self):
        class _B(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 4)
                self.lin.global_name = "m.lin"
                self.lin.to_quantized = True
                self.lin.bits = 4
                self.lin.group_size = 4
                self.lin.sym = False
                self.lin.act_bits = 16

            def forward(self, x):
                return self.lin(x)

        b = _B()
        seen = []
        b.lin.register_forward_hook(lambda mod, inp, out: seen.append(out) or out)
        b._seen = seen
        return b

    def test_sync_hooks_copies_and_clears(self):
        from auto_round.algorithms.parallel.data_parallel import MirrorPool

        block = self._block_with_hook()
        pool = MirrorPool(block, [torch.device("cpu", 0), torch.device("cpu", 1)])
        rep = pool.reps[-1]
        assert rep is not block
        # deepcopy carried the hook; sync must keep it working...
        pool.sync_hooks_from(block)
        x = torch.randn(2, 4)
        rep(x)
        assert len(block._seen) == 1
        # ...and after the home hook is gone, sync drops it from the mirror
        block.lin._forward_hooks.clear()
        pool.sync_hooks_from(block)
        rep(x)
        assert len(block._seen) == 1, "stale hook must not fire after sync"

    def test_release_drops_mirrors(self):
        from auto_round.algorithms.parallel.data_parallel import MirrorPool

        block = self._block_with_hook()
        pool = MirrorPool(block, [torch.device("cpu", 0), torch.device("cpu", 1)])
        assert pool.world == 2
        pool.release()
        assert pool.reps == [] and pool.world == 0


class TestComposerPoolHandoff:
    """Step 3.6: the collection pool is handed to the iters=0 quantizer and
    released around quantize_block (kept for the cascade at iters=0, dropped
    before the tune's own mirrors at iters>0, dropped on failure too)."""

    def _parts(self, iters, consumes=True):
        from auto_round.algorithms.composer import AlgorithmComposer
        from auto_round.algorithms.quantization.rtn.quantizer import OptimizedRTNQuantizer

        composer = AlgorithmComposer.__new__(AlgorithmComposer)
        events = []

        class _Ctx:
            devices = None  # no DDP devices -> ensure_pool is a no-op

            def __init__(self, pool):
                self.pool = pool

            def ensure_pool(self, block):
                events.append("ensure")

            def mark(self, what):
                events.append(what)

            def release_pool(self):
                if self.pool is not None:
                    events.append("release")
                    self.pool = None

        class _Pool:
            pass

        pool = _Pool()

        q = OptimizedRTNQuantizer.__new__(OptimizedRTNQuantizer)
        q.iters = iters
        q.enable_quanted_input = False
        q._consumes = consumes

        def _qb(*a, **k):
            # instance-attribute functions stay unbound: close over q directly
            st = parallel_state(q)
            seen = st.pool if st is not None else None
            events.append(("saw_pool", seen))
            if seen is not None and q._consumes:
                parallel_state(q, create=True).pool_used = True
                if q._consumes == "adopt":
                    # adoption transfers ownership (drops the pool ref)
                    parallel_state(q).pool = None

        q.quantize_block = _qb
        ctx = _Ctx(pool)
        ctx.devices = [torch.device("cpu", 0), torch.device("cpu", 1)]  # DDP on
        composer._coll_ctx = ctx
        composer._block = None
        composer.block_quantizer = q
        composer.preprocessors = []
        return composer, ctx, pool, q, events

    def _run_handoff(self, composer):
        # mirrors the Step 3.6 handoff (pool to the quantizer, ANY consumer:
        # OptRTN home quantize or SignRound adoption) + the real post-Step-4
        # _settle_block_pool + the try/finally in compress_block
        if getattr(composer._coll_ctx, "devices", None) and len(composer._coll_ctx.devices) >= 2:
            composer._coll_ctx.ensure_pool(composer._block)
        _pool = getattr(composer._coll_ctx, "pool", None)
        if _pool is not None:
            _state = parallel_state(composer.block_quantizer, create=True)
            _state.pool = _pool
            _state.pool_used = False
        try:
            composer.block_quantizer.quantize_block(None)
            _pool = composer._settle_block_pool(composer._block, _pool)
            composer._coll_ctx.mark("step5+6")
        finally:
            _state = parallel_state(composer.block_quantizer)
            if _state is not None:
                _state.pool = None
            composer._coll_ctx.release_pool()

    def test_iters0_pool_handed_then_kept_for_cascade(self):
        # RTN-lane consumption: attr KEPT + used -> the pool survives into
        # step5+6 (its sync made the mirrors exactly-home for the cascade)
        composer, ctx, pool, q, events = self._parts(iters=0, consumes=True)
        self._run_handoff(composer)
        assert events == ["ensure", ("saw_pool", pool), "step5+6", "release"]
        assert parallel_state(q).pool is None

    def test_iters0_unconsumed_pool_released_before_cascade(self):
        composer, ctx, pool, q, events = self._parts(iters=0, consumes=False)
        self._run_handoff(composer)
        # quantizer left unconsumed (e.g. plan vanished): released right
        # after quantize_block, before the cascade step (the finally release
        # is then a no-op on None)
        assert events == ["ensure", ("saw_pool", pool), "release", "step5+6"]
        assert ctx.pool is None

    def test_iters_positive_pool_adopted_by_tune(self):
        composer, ctx, pool, q, events = self._parts(iters=10, consumes="adopt")
        self._run_handoff(composer)
        # the tune adopts the pool (engage_ wraps mirrors in place) -> kept
        # through the tune, released once at the very end
        assert events == ["ensure", ("saw_pool", pool), "release", "step5+6"]

    def test_failure_still_releases_pool(self):
        composer, ctx, pool, q, events = self._parts(iters=0)

        def _boom(*a, **k):
            raise RuntimeError("quantize failed")

        q.quantize_block = _boom
        try:
            self._run_handoff(composer)
            raised = False
        except RuntimeError:
            raised = True
        assert raised
        assert parallel_state(q).pool is None
        assert ctx.pool is None


class TestRotationHookPoolInteraction:
    """Real Hadamard input-location hooks x MirrorPool: the online rotation
    rides on the mirrors (deepcopy + sync_hooks_from) and mirror forwards are
    bit-identical to the home block; removing the home hook and re-syncing
    drops it from the mirrors (no stale rotation)."""

    def test_hadamard_input_hook_parity_and_resync(self):
        from auto_round.algorithms.parallel.data_parallel import MirrorPool
        from auto_round.algorithms.transforms import RotationConfig
        from auto_round.algorithms.transforms.hadamard.apply import _apply_input_transform

        class _B(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(32, 32)

            def forward(self, x):
                return self.lin(x)

        b = _B()
        _apply_input_transform(b.lin, RotationConfig(hadamard_type="hadamard"), data_type="int")
        pool = MirrorPool(b, [torch.device("cpu", 0), torch.device("cpu", 1)])
        rep = pool.reps[-1]
        assert rep is not b
        assert len(rep.lin._forward_pre_hooks) == 1
        x = torch.randn(2, 32)
        with torch.no_grad():
            y_home = b(x)
            y_rep = rep(x)
        assert torch.equal(y_home, y_rep), "mirror forward must match home under the rotation hook"
        # stale-hook hygiene: removing the home hook + re-sync drops it
        b.lin._forward_pre_hooks.clear()
        pool.sync_hooks_from(b)
        assert len(rep.lin._forward_pre_hooks) == 0
        with torch.no_grad():
            y_rep2 = rep(x)
        assert not torch.equal(y_home, y_rep2), "resync must drop the removed rotation hook"


class TestHomeQuantizeThenCascade:
    """End-to-end seam: the quantizer quantizes the HOME block, the engine
    syncs the mirrors to it, and the cascade forward on the SAME pool outputs
    exactly the serial forward of the quantized home block."""

    def test_cascade_on_pool_matches_serial_home(self, monkeypatch):
        from types import SimpleNamespace

        from auto_round.algorithms.parallel.data_parallel import (
            MirrorPool,
            sharded_nograd_forward,
        )
        from auto_round.algorithms.parallel.mirror_sync import sync_mirrors_to_home
        from auto_round.algorithms.quantization.rtn.quantizer import OptimizedRTNQuantizer

        torch.manual_seed(0)

        class _B(torch.nn.Module):
            def __init__(self):
                super().__init__()
                for i in range(4):
                    setattr(self, f"l{i}", _expert_layer(f"m.blk.l{i}"))

            def forward(self, x):
                for i in range(4):
                    x = getattr(self, f"l{i}")(x)
                return x

        block = _B()  # weights stay RANDOM: FP and quantized forwards differ
        originals = {n: m.weight.data.clone() for n, m in block.named_modules() if hasattr(m, "global_name")}
        devices = [torch.device("cpu", 0), torch.device("cpu", 1)]
        pool = MirrorPool(block, devices)

        q = OptimizedRTNQuantizer.__new__(OptimizedRTNQuantizer)
        q._BaseAlgorithm__run_ctx = None  # unbound: q.model is None (the driver is stubbed)
        q.parallel_state = SimpleNamespace(plan=SimpleNamespace(world=2, devices=devices), pool=pool, pool_used=False)
        import auto_round.algorithms.quantization.rtn.batched_search as _bs

        def _staged(layer, disable_opt_rtn=None, defer_search=False):
            return layer

        def _driver(model, staged, max_batch=None):
            # a REAL quantize-dequantize: round-to-grid + offset, so an
            # FP (unsynced) mirror layer would produce different outputs
            with torch.no_grad():
                for _name, layer in staged:
                    layer.weight.data = ((layer.weight.data * 4).round() / 4) + 0.125

        q._quantize_layer_via_rtn = _staged
        monkeypatch.setattr(_bs, "run_batched_rtn_search", _driver)
        q.quantize_block(block, None, {}, None, None, None)
        assert sync_mirrors_to_home(q, block) is True

        # every home layer carries the EXACT quantize transform
        for n, m in block.named_modules():
            if hasattr(m, "global_name"):
                assert torch.equal(m.weight.data, ((originals[n] * 4).round() / 4) + 0.125)

        inputs = [torch.randn(1, 4) for _ in range(4)]  # n=4, world=2 -> shards

        def _runner(rep, inp, oth, idxs, cache_device=None):
            out = []
            with torch.no_grad():
                for i in idxs:
                    out.append(rep(inp[i]))
            return torch.cat(out, dim=0)

        # mirror layers must carry the exact home state (the post-quantize
        # sync): FP mirrors would diverge here
        for r in range(pool.world):
            if pool.reps[r] is block:
                continue
            for name, m in block.named_modules():
                if hasattr(m, "global_name"):
                    assert torch.equal(pool.mirror_layer(r, name).weight.data, m.weight.data)
        outs = sharded_nograd_forward(_runner, block, inputs, {}, torch.device("cpu"), devices, pool=pool)
        with torch.no_grad():
            serial = [block(t) for t in inputs]
        for a, b in zip(outs, serial):
            assert torch.equal(a, b), "cascade on quantized mirrors must match the serial home forward"
        pool.release()


class TestMirrorPoolMultiPass:
    """The pool must SURVIVE its passes: sharded_nograd_forward's teardown
    once cleared the aliased reps list, silently disabling every later reuse
    (searches, cascade) while still pinning the mirror VRAM."""

    def test_two_pool_passes_keep_the_pool_alive(self):
        import torch as _t

        from auto_round.algorithms.parallel.data_parallel import (
            MirrorPool,
            sharded_nograd_forward,
        )

        class _B(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.lin(x)

        block = _B()
        devices = [torch.device("cpu", 0), torch.device("cpu", 1)]
        pool = MirrorPool(block, devices)
        n_rep_objects = [id(r) for r in pool.reps]
        inputs = [_t.randn(1, 4) for _ in range(4)]

        def _runner(rep, inp, oth, idxs, cache_device=None):
            out = []
            with _t.no_grad():
                for i in idxs:
                    out.append(rep(inp[i]))
            return _t.cat(out, dim=0)

        for pass_idx in range(2):
            outs = sharded_nograd_forward(_runner, block, inputs, {}, torch.device("cpu"), devices, pool=pool)
            assert len(outs) == 4
            # the pool still holds its replicas after the pass teardown
            assert pool.world == 2, f"pass {pass_idx}: pool lost its reps"
            assert [id(r) for r in pool.reps] == n_rep_objects, "replica objects were rebuilt"
        pool.release()
        assert pool.world == 0


class TestMergeMirrorStats:
    """_merge_mirror_stats must keep scalar stats co-traveling with their tensor
    sibling: imatrix_cnt (a python int) was silently dropped, so a cold expert
    (routed rows on mirror shards, zero on home's) merged its imatrix home
    while cnt went missing -- the normalize step then crashed with
    AttributeError: no attribute 'imatrix_cnt'."""

    _STATS = {"imatrix": "sum", "imatrix_cnt": "sum", "act_max": "max"}

    def test_cold_expert_cnt_cotravels_with_imatrix(self):
        from auto_round.algorithms.parallel.data_parallel import _merge_mirror_stats

        home = torch.nn.Linear(4, 4)
        mirror = torch.nn.Linear(4, 4)
        mirror.imatrix = torch.ones(4)
        mirror.imatrix_cnt = 7  # python int, exactly as both collectors write it
        _merge_mirror_stats(home, [mirror], self._STATS)
        assert hasattr(home, "imatrix") and torch.equal(home.imatrix, torch.ones(4))
        assert getattr(home, "imatrix_cnt", None) == 7

    def test_cnt_sums_across_mirrors(self):
        from auto_round.algorithms.parallel.data_parallel import _merge_mirror_stats

        home = torch.nn.Linear(4, 4)
        home.imatrix = torch.zeros(4)
        home.imatrix_cnt = 2
        m1, m2 = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
        m1.imatrix_cnt, m2.imatrix_cnt = 3, 5
        m1.imatrix = torch.ones(4)
        m2.imatrix = torch.ones(4)
        _merge_mirror_stats(home, [m1, m2], self._STATS)
        assert home.imatrix_cnt == 10
        assert torch.equal(home.imatrix, torch.full((4,), 2.0))

    def test_tensor_cnt_also_merges(self):
        from auto_round.algorithms.parallel.data_parallel import _merge_mirror_stats

        home = torch.nn.Linear(4, 4)
        mirror = torch.nn.Linear(4, 4)
        mirror.imatrix = torch.ones(4)
        mirror.imatrix_cnt = torch.tensor(7)
        _merge_mirror_stats(home, [mirror], self._STATS)
        assert home.imatrix_cnt == 7

    def test_non_numeric_attrs_ignored(self):
        from auto_round.algorithms.parallel.data_parallel import _merge_mirror_stats

        home = torch.nn.Linear(4, 4)
        mirror = torch.nn.Linear(4, 4)
        mirror.imatrix_cnt = "bogus"
        _merge_mirror_stats(home, [mirror], self._STATS)
        assert not hasattr(home, "imatrix_cnt")

    def test_mirror_values_are_removed_once_folded(self):
        from auto_round.algorithms.parallel.data_parallel import _merge_mirror_stats

        home = torch.nn.Linear(4, 4)
        mirror = torch.nn.Linear(4, 4)
        mirror.imatrix, mirror.imatrix_cnt, mirror.act_max = torch.ones(4), 3, torch.ones(4)
        _merge_mirror_stats(home, [mirror], self._STATS)
        _merge_mirror_stats(home, [mirror], self._STATS)  # a later pass finds nothing left to add
        assert torch.equal(home.imatrix, torch.ones(4)) and home.imatrix_cnt == 3
        assert not any(hasattr(mirror, a) for a in ("imatrix", "imatrix_cnt", "act_max"))

    def test_hook_pass_then_hookless_pass_matches_serial(self):
        """Pool mirrors live across a block's collection passes: a sharded hook
        pass followed by any other sharded pass must leave the serial totals
        (the second merge used to add the mirrors' share again)."""
        from auto_round.algorithms.parallel.data_parallel import MirrorPool, sharded_nograd_forward

        class _B(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.lin(x)

        def _imatrix_hook(module, inp, _out):
            sq = (inp[0].reshape(-1, inp[0].shape[-1]).float() ** 2).sum(0)
            if hasattr(module, "imatrix"):
                module.imatrix += sq
            else:
                module.imatrix = sq

        def _runner(rep, inp, _oth, idxs, cache_device=None):
            with torch.no_grad():
                return torch.cat([rep(inp[i]) for i in idxs], dim=0)

        torch.manual_seed(0)
        block = _B()
        devices = [torch.device("cpu", i) for i in range(2)]
        pool = MirrorPool(block, devices)
        inputs = [torch.randn(1, 4) for _ in range(8)]
        serial = sum((t.reshape(-1, 4) ** 2).sum(0) for t in inputs)
        stats = {"imatrix": "sum"}
        handle = block.lin.register_forward_hook(_imatrix_hook)
        sharded_nograd_forward(_runner, block, inputs, {}, torch.device("cpu"), devices, merge_stats=stats, pool=pool)
        handle.remove()
        assert torch.allclose(block.lin.imatrix, serial)
        sharded_nograd_forward(_runner, block, inputs, {}, torch.device("cpu"), devices, merge_stats=stats, pool=pool)
        assert torch.allclose(block.lin.imatrix, serial)
        pool.release()


class TestCalibrationStatsContract:
    """A block quantizer declares the statistics its calibration hooks write
    (``calibration_stats``: attr -> "sum" | "max"); the engine folds exactly
    those from the mirrors, keeps undeclared hook passes serial, and warns
    when a sharded pass changes an undeclared attribute on the mirrors."""

    def test_merge_map_is_engine_act_max_plus_declaration(self):
        from types import SimpleNamespace

        from auto_round.algorithms.parallel.data_parallel import calibration_merge_stats

        q = SimpleNamespace(calibration_stats={"hess": "sum"})
        assert calibration_merge_stats(q, ["h"]) == {"act_max": "max", "hess": "sum"}

    def test_undeclared_hooks_keep_the_pass_serial(self):
        from types import SimpleNamespace

        from auto_round.algorithms.parallel.data_parallel import calibration_merge_stats

        assert calibration_merge_stats(SimpleNamespace(calibration_stats=None), ["h"]) is None
        # no quantizer hooks: nothing to declare, the engine's act_max still folds
        assert calibration_merge_stats(SimpleNamespace(calibration_stats=None), []) == {"act_max": "max"}

    def test_unknown_reduction_is_rejected(self):
        from types import SimpleNamespace

        from auto_round.algorithms.parallel.data_parallel import calibration_merge_stats

        with pytest.raises(ValueError, match="supported shard reductions"):
            calibration_merge_stats(SimpleNamespace(calibration_stats={"hess": "mean"}), ["h"])

    def test_imatrix_quantizers_declare_their_stats(self):
        from auto_round.algorithms.quantization.base import BaseQuantizer
        from auto_round.algorithms.quantization.rtn.quantizer import OptimizedRTNQuantizer
        from auto_round.algorithms.quantization.sign_roundv2.quantizer import SignRoundV2Quantizer

        assert BaseQuantizer.calibration_stats is None
        assert OptimizedRTNQuantizer.calibration_stats == {"imatrix": "sum", "imatrix_cnt": "sum"}
        assert SignRoundV2Quantizer.calibration_stats == {"imatrix": "sum"}

    @staticmethod
    def _sharded_hook_pass(hook, stats):
        from auto_round.algorithms.parallel.data_parallel import MirrorPool, sharded_nograd_forward

        class _B(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.lin(x)

        def _runner(rep, inp, _oth, idxs, cache_device=None):
            with torch.no_grad():
                return torch.cat([rep(inp[i]) for i in idxs], dim=0)

        block = _B()
        devices = [torch.device("cpu", i) for i in range(2)]
        pool = MirrorPool(block, devices)
        handle = block.lin.register_forward_hook(hook)
        inputs = [torch.randn(1, 4) for _ in range(8)]
        sharded_nograd_forward(_runner, block, inputs, {}, torch.device("cpu"), devices, merge_stats=stats, pool=pool)
        handle.remove()
        pool.release()
        return block

    def test_undeclared_mirror_write_warns(self, caplog, _autoround_log_propagate):
        def _hook(module, inp, _out):
            module.f9_undeclared_hess = inp[0].sum()

        with caplog.at_level("WARNING"):
            self._sharded_hook_pass(_hook, {"act_max": "max"})
        msgs = [r.message for r in caplog.records if "not a declared calibration statistic" in r.message]
        assert msgs and "f9_undeclared_hess" in msgs[0]

    def test_bookkeeping_list_change_does_not_warn(self, caplog, _autoround_log_propagate):
        # a hook that clears its own handle list (the MoE shape recorder in
        # tune_memory does this) is bookkeeping, not a statistic
        def _hook(module, _inp, _out):
            module.f9_rec_handles_ = []

        with caplog.at_level("WARNING"):
            self._sharded_hook_pass(_hook, {"act_max": "max"})
        assert not [r for r in caplog.records if "not a declared calibration statistic" in r.message]

    def test_undeclared_tensor_list_append_warns(self, caplog, _autoround_log_propagate):
        def _hook(module, inp, _out):
            module.__dict__.setdefault("f9_rows", []).append(inp[0].sum())

        with caplog.at_level("WARNING"):
            self._sharded_hook_pass(_hook, {"act_max": "max"})
        msgs = [r.message for r in caplog.records if "not a declared calibration statistic" in r.message]
        assert msgs and "f9_rows" in msgs[0]

    def test_declared_mirror_write_is_folded_silently(self, caplog, _autoround_log_propagate):
        def _hook(module, inp, _out):
            sq = (inp[0].reshape(-1, 4) ** 2).sum(0)
            if hasattr(module, "f9_declared_stat"):
                module.f9_declared_stat += sq
            else:
                module.f9_declared_stat = sq

        with caplog.at_level("WARNING"):
            block = self._sharded_hook_pass(_hook, {"act_max": "max", "f9_declared_stat": "sum"})
        assert not [r for r in caplog.records if "not a declared calibration statistic" in r.message]
        assert hasattr(block.lin, "f9_declared_stat")


class TestCappedPassReusesPool:
    """A capped hook pass (AR_TUNE_DDP_MAX_COLLECT_FORWARD_DEVICES) must REUSE
    the full-width pool's first K replicas, skipping ephemeral mirrors (the
    old behavior paid a per-pass block deepcopy plus a late full pool
    rebuild per block)."""

    def test_capped_pass_runs_subset_of_pool_replicas(self):
        import torch as _t

        from auto_round.algorithms.parallel.data_parallel import (
            MirrorPool,
            sharded_nograd_forward,
        )

        class _B(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.lin(x)

        block = _B()
        devices = [torch.device("cpu", i) for i in range(4)]
        pool = MirrorPool(block, devices)
        rep_ids = [id(r) for r in pool.reps]
        inputs = [_t.randn(1, 4) for _ in range(8)]
        ran = []

        def _runner(rep, inp, oth, idxs, cache_device=None):
            ran.append(id(rep))
            out = []
            with _t.no_grad():
                for i in idxs:
                    out.append(rep(inp[i]))
            return _t.cat(out, dim=0)

        outs = sharded_nograd_forward(
            _runner, block, inputs, {}, torch.device("cpu"), devices, pool=pool, max_devices=2
        )
        with torch.no_grad():
            serial = [block(t) for t in inputs]
        for a, b in zip(outs, serial):
            assert _t.equal(a, b), "capped pool pass must match the serial forward"
        # only the first K replicas ran, and they are the pool's own objects
        assert set(ran) == set(rep_ids[:2]), f"capped pass must run pool.reps[:2], ran={ran}"
        # the pool keeps its full width and identity set afterwards
        assert pool.world == 4
        assert [id(r) for r in pool.reps] == rep_ids, "replica objects were rebuilt"
        pool.release()

    def test_full_width_pass_with_pool_unchanged(self):
        import torch as _t

        from auto_round.algorithms.parallel.data_parallel import (
            MirrorPool,
            sharded_nograd_forward,
        )

        class _B(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.lin(x)

        block = _B()
        devices = [torch.device("cpu", i) for i in range(2)]
        pool = MirrorPool(block, devices)
        rep_ids = [id(r) for r in pool.reps]
        inputs = [_t.randn(1, 4) for _ in range(4)]

        def _runner(rep, inp, oth, idxs, cache_device=None):
            out = []
            with _t.no_grad():
                for i in idxs:
                    out.append(rep(inp[i]))
            return _t.cat(out, dim=0)

        outs = sharded_nograd_forward(_runner, block, inputs, {}, torch.device("cpu"), devices, pool=pool)
        assert len(outs) == 4
        assert [id(r) for r in pool.reps] == rep_ids
        pool.release()


class TestEnsurePool:
    """The pool is block-wide: built even when no collection pass was
    eligible (n % world != 0 -> serial forwards), because the searches and
    the tune consume resident mirrors regardless of forward eligibility."""

    def test_ensure_pool_builds_once(self):
        from auto_round.algorithms.parallel.tune_parallel import TuneParallelContext

        class _B(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 4)

        ctx = TuneParallelContext()
        ctx.devices = [torch.device("cpu", 0), torch.device("cpu", 1)]
        assert ctx.pool is None
        blk = _B()
        ctx.ensure_pool(blk)
        first = ctx.pool
        assert first is not None and first.world == 2
        ctx.ensure_pool(blk)  # idempotent: same pool object
        assert ctx.pool is first

    def test_ensure_pool_noop_without_devices(self):
        from auto_round.algorithms.parallel.tune_parallel import TuneParallelContext

        ctx = TuneParallelContext()
        ctx.devices = None
        ctx.ensure_pool(object())
        assert ctx.pool is None


class TestContiguousShardSplit:
    """The ceil/floor split: ANY sample count shards (a remainder lands on
    an earlier replica); fewer samples than devices shrinks the pass world
    instead of running serial; pools are withheld when the pass narrows."""

    def test_bounds_remainder_on_earlier_shards(self):
        from auto_round.algorithms.parallel.data_parallel import contiguous_shard_bounds

        assert contiguous_shard_bounds(129, 4) == [0, 33, 65, 97, 129]  # 33/32/32/32
        assert contiguous_shard_bounds(128, 4) == [0, 32, 64, 96, 128]
        assert contiguous_shard_bounds(3, 4) == [0, 1, 2, 3, 3]  # caller clamps world first

    def test_uneven_n_shards_instead_of_serial(self):
        import torch as _t

        from auto_round.algorithms.parallel.data_parallel import sharded_nograd_forward

        class _B(_t.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = _t.nn.Linear(4, 4)

            def forward(self, x):
                return self.lin(x)

        seen_shards = []

        def _runner(rep, inp, oth, idxs, cache_device=None):
            seen_shards.append(list(idxs))
            out = []
            with _t.no_grad():
                for i in idxs:
                    out.append(rep(inp[i]))
            return _t.cat(out, dim=0)

        devices = [_t.device("cpu", 0), _t.device("cpu", 1)]
        inputs = [_t.randn(1, 4) for _ in range(5)]  # 5 % 2 != 0 -> 3/2 split
        outs = sharded_nograd_forward(_runner, _B(), inputs, {}, _t.device("cpu"), devices)
        assert seen_shards == [[0, 1, 2], [3, 4]]
        assert len(outs) == 5

    def test_fewer_samples_than_devices_shrinks_world(self):
        import torch as _t

        from auto_round.algorithms.parallel.data_parallel import MirrorPool, sharded_nograd_forward

        class _B(_t.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = _t.nn.Linear(4, 4)

            def forward(self, x):
                return self.lin(x)

        def _runner(rep, inp, oth, idxs, cache_device=None):
            out = []
            with _t.no_grad():
                for i in idxs:
                    out.append(rep(inp[i]))
            return _t.cat(out, dim=0)

        devices = [_t.device("cpu", 0), _t.device("cpu", 1), _t.device("cpu", 2)]
        inputs = [_t.randn(1, 4) for _ in range(2)]  # n=2 < world=3
        pool = MirrorPool(_B(), devices)  # full-width pool exists
        outs = sharded_nograd_forward(_runner, _B(), inputs, {}, _t.device("cpu"), devices, pool=pool)
        assert len(outs) == 2  # sharded on 2 devices, not serial
        assert pool.world == 3  # pool untouched by the narrowed pass


class TestResolverRtnSafe:
    """The resolver must not assume SignRound-family quantizer methods."""

    def test_quantizer_without_get_scaler_resolves(self, monkeypatch):
        from types import SimpleNamespace

        import torch

        from auto_round.algorithms.parallel.data_parallel import resolve_tune_ddp_plan_

        q = SimpleNamespace(iters=0, gradient_accumulate_steps=1, enable_lfq=False)
        # missing parallel_constraints and calibration_context must still
        # resolve cleanly
        with pytest.raises(RuntimeError, match="not a supported accelerator"):
            resolve_tune_ddp_plan_(
                _with_policy(q), torch.nn.Sequential(torch.nn.Linear(4, 4)), [torch.zeros(1)], None, "cpu"
            )


class TestCollectionShardingFailVisible:
    """The collection context must not swallow the resolver's requirement error."""

    def _composer(self):
        from auto_round.algorithms.composer import AlgorithmComposer

        composer = AlgorithmComposer.__new__(AlgorithmComposer)
        composer.block_forward = type("R", (), {"output_config": ["hidden_states"]})()
        composer.block_quantizer = object()  # opaque; the resolver is monkeypatched
        return composer

    def test_requirement_error_propagates(self, monkeypatch):
        import torch

        import auto_round.algorithms.parallel.tune_parallel as tp

        def _raise(quantizer, block, fp_inputs, fp_outputs, home, world=None, log=True):
            raise RuntimeError("parallel tuning with world=4 is ineligible: no mirror device")

        monkeypatch.setattr(tp, "resolve_tune_ddp_plan_", _raise)
        block = torch.nn.Sequential(torch.nn.Linear(4, 4))
        with pytest.raises(RuntimeError, match="ineligible"):
            self._composer()._collection_context(block, [torch.zeros(1)])

    def test_reachability_error_warns_and_declines(self, monkeypatch, caplog, _autoround_log_propagate):
        import torch

        import auto_round.algorithms.parallel.tune_parallel as tp

        def _raise(quantizer, block, fp_inputs, fp_outputs, home, world=None, log=True):
            raise OSError("boom")

        monkeypatch.setattr(tp, "resolve_tune_ddp_plan_", _raise)
        block = torch.nn.Sequential(torch.nn.Linear(4, 4))
        with caplog.at_level("WARNING"):
            out = self._composer()._collection_context(block, [torch.zeros(1)])
        assert out.devices is None
        assert any("resolver unreachable" in r.message for r in caplog.records)


import unittest  # noqa: E402
from types import SimpleNamespace  # noqa: E402


def _stackable_search_fn(weight, bits, imatrix):
    """Row-independent deterministic search: first element of each leading row."""
    return weight.reshape(weight.shape[0], -1)[:, 0]


class TestSyncPoolDevices:
    """Phase barriers synchronize every distinct CUDA pool device once (CPU skipped)."""

    def test_syncs_each_cuda_device_once(self):
        from auto_round.algorithms.parallel.mirror_sync import _sync_pool_devices

        devices = [
            torch.device("cpu"),
            torch.device("cuda:0"),
            torch.device("cuda:1"),
            torch.device("cuda:0"),
        ]
        calls = []
        orig = torch.cuda.synchronize
        try:
            torch.cuda.synchronize = lambda d: calls.append(str(d))
            elapsed = _sync_pool_devices(devices)
        finally:
            torch.cuda.synchronize = orig
        # one synchronize per distinct cuda device, cpu skipped, elapsed returned
        assert calls == ["cuda:0", "cuda:1"]
        assert isinstance(elapsed, float)


class TestDistributeSearchStats:
    """Tune adoption: the merged home stats land on every mirror."""

    def _pool(self, block, world):
        mirrors = [self._build_mirror(block) for _ in range(world)]
        reps = [block] + mirrors  # home at index 0

        def mirror_layer(r, name):
            return reps[r].get_submodule(name)

        return SimpleNamespace(world=len(reps), reps=reps, mirror_layer=mirror_layer)

    @staticmethod
    def _build_mirror(block):
        import copy as _pcopy

        return _pcopy.deepcopy(block)

    def test_distributes_to_every_mirror(self):
        import torch.nn as nn

        from auto_round.algorithms.parallel.mirror_sync import _distribute_search_stats

        block = nn.Module()
        block.l1 = nn.Linear(4, 4, bias=False)
        block.l1.imatrix = torch.ones(4) * 2
        pool = self._pool(block, 3)
        _distribute_search_stats(pool, block, [("l1", block.l1)])
        for r in (1, 2):
            assert torch.equal(pool.mirror_layer(r, "l1").imatrix, torch.ones(4) * 2)


class TestHomeQuantizeMirrorParity:
    """REAL end-to-end parity (no spies) on CPU mirrors: under a world=2 plan
    the quantizer's own quantize_block (main's code) plus the engine's mirror
    sync must leave the home block AND every mirror bit-identical to a plain
    serial quantize_block run. Every quantized weight/scale/zp/data_type must
    be torch.equal."""

    WIDTHS = ((16, 32), (16, 32), (8, 32), (12, 24))  # two same-shape pairs -> real stacking

    def _layer(self, name, out, inn, gen):
        m = torch.nn.Linear(inn, out, bias=False)
        with torch.no_grad():
            m.weight.copy_(torch.randn(out, inn, generator=gen))
        m.global_name = name
        m.to_quantized = True
        m.bits = 4
        m.group_size = 32
        m.sym = True
        m.data_type = "int"
        m.iters = 0
        m.act_bits = 16
        m.scale_dtype = torch.float16
        m.super_bits = None
        m.super_group_size = None
        m.imatrix = torch.rand(inn, generator=gen) + 0.5
        m.imatrix_cnt = 1  # pre-normalized: quantize_block's divide is identity
        return m

    def _block(self):
        import copy as _copy

        gen = torch.Generator().manual_seed(1234)

        blk = torch.nn.Module()
        for i, (out, inn) in enumerate(self.WIDTHS):
            blk.add_module(f"l{i}", self._layer(f"m.blk.l{i}", out, inn, gen))
        return blk, _copy.deepcopy(blk)

    def _model(self, blk):
        # the tree must resolve the layers' GLOBAL names (m.blk.l<i>): the
        # driver re-attaches quantized results by that path
        model = torch.nn.Module()
        m = torch.nn.Module()
        m.blk = blk
        model.m = m
        return model

    def _quantizer(self, cls, config, model):
        from types import SimpleNamespace

        q_cls = type(
            f"_ParityQ_{cls.__name__}",
            (cls,),
            {
                "config": config,
                "model": model,
                "model_context": SimpleNamespace(amp_dtype=torch.float32, is_moe_model=False),
                "compress_context": SimpleNamespace(enable_torch_compile=False),
            },
        )
        return q_cls.__new__(q_cls)

    def _assert_same(self, ref, got, tag):
        assert torch.equal(ref.weight.data, got.weight.data), f"{tag} weight diverged"
        assert torch.equal(ref.scale, got.scale), f"{tag} scale diverged"
        rz, gz = getattr(ref, "zp", None), getattr(got, "zp", None)
        assert (rz is None) == (gz is None), f"{tag} zp presence diverged"
        if rz is not None:
            if isinstance(rz, torch.Tensor):
                assert torch.equal(rz, gz), f"{tag} zp diverged"
            else:
                assert rz == gz, f"{tag} zp diverged"
        assert getattr(ref, "data_type", None) == getattr(got, "data_type", None), f"{tag} data_type diverged"
        assert getattr(ref, "q_scale_thresh", None) == getattr(got, "q_scale_thresh", None)

    def _run_parity(self, quantizer_cls, config):
        from types import SimpleNamespace

        from auto_round.algorithms.parallel.data_parallel import MirrorPool
        from auto_round.algorithms.parallel.mirror_sync import sync_mirrors_to_home

        blk_s, blk_d = self._block()
        model_s, model_d = self._model(blk_s), self._model(blk_d)
        self._quantizer(quantizer_cls, config, model_s).quantize_block(blk_s, None, {}, None, None, None)

        devices = [torch.device("cpu", 0), torch.device("cpu", 1)]
        pool = MirrorPool(blk_d, devices)
        q = self._quantizer(quantizer_cls, config, model_d)
        q.parallel_state = SimpleNamespace(plan=SimpleNamespace(world=2, devices=devices), pool=pool, pool_used=False)
        q.quantize_block(blk_d, None, {}, None, None, None)
        assert sync_mirrors_to_home(q, blk_d) is True

        for i in range(len(self.WIDTHS)):
            ref = model_s.get_submodule(f"m.blk.l{i}")
            self._assert_same(ref, blk_d.get_submodule(f"l{i}"), f"home l{i}")
            for r in range(pool.world):
                if pool.reps[r] is not blk_d:
                    self._assert_same(ref, pool.mirror_layer(r, f"l{i}"), f"mirror{r} l{i}")

    def test_opt_rtn_home_and_mirrors_match_serial(self):
        from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig
        from auto_round.algorithms.quantization.rtn.quantizer import OptimizedRTNQuantizer

        cfg = OptimizedRTNConfig(bits=4, group_size=32, sym=True, disable_opt_rtn=False, enable_neuqi=False)
        self._run_parity(OptimizedRTNQuantizer, cfg)

    def test_plain_rtn_home_and_mirrors_match_serial(self):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig
        from auto_round.algorithms.quantization.rtn.quantizer import RTNQuantizer

        cfg = RTNConfig(bits=4, group_size=32, sym=True, disable_opt_rtn=True)
        self._run_parity(RTNQuantizer, cfg)
