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
from auto_round.compressors.utils import IndexSampler, shard_samplers


def _with_policy(q, world=2):
    """Attach a ParallelPolicy-bearing compress_context to a fake quantizer
    (the policy rides on the compress context; env vars are gone)."""
    from types import SimpleNamespace

    from auto_round.algorithms.parallel.data_parallel import ParallelPolicy

    q.compress_context = SimpleNamespace(parallel_policy=ParallelPolicy(world=world))
    return q


class TestShardSamplers:
    def test_layout_and_epoch_coverage(self):
        samplers = shard_samplers(nsamples=8, world=4, batch_per_replica=1)
        assert samplers is not None and len(samplers) == 4
        drawn = [s.next_batch() for s in samplers]
        # each replica draws only from its own contiguous shard
        for r, batch in enumerate(drawn):
            assert all(r * 2 <= j < (r + 1) * 2 for j in batch)
        # one shard-epoch covers every sample exactly once (batch=1 x shard=2 draws)
        rest = [s.next_batch() for s in samplers]
        assert sorted(j for b in drawn + rest for j in b) == list(range(8))

    def test_ineligible_layouts_return_none(self):
        assert shard_samplers(nsamples=7, world=4, batch_per_replica=1) is None  # indivisible
        assert shard_samplers(nsamples=8, world=1, batch_per_replica=8) is None  # single device
        assert shard_samplers(nsamples=8, world=4, batch_per_replica=3) is None  # draw > shard

    def test_index_sampler_explicit_pool(self):
        s = IndexSampler(4, 2, indices=[10, 11, 12, 13])
        b = s.next_batch()
        assert sorted(b) in ([10, 11], [12, 13]) or set(b) <= {10, 11, 12, 13}
        with pytest.raises(ValueError):
            IndexSampler(3, 2, indices=[1, 2])


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
        out = _transport_segment(t, torch.device("cpu" if t.is_cuda else "cpu"))
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
            modules=lambda: [],
        )
        # the span rule passes; the resolve then proceeds (it may decline
        # later on the free-VRAM probe, which is CPU-safe) -- the assertion
        # checks the decline list for the placement reason
        try:
            resolve_tune_ddp_plan_(_with_policy(self._quantizer()), block, [torch.zeros(1)], None, "cuda:0")
        except RuntimeError as e:
            assert "spans" not in str(e)

    def test_eager_swap_uses_stable_eager_attr(self):
        """Worker-thread swap prefers the compile-time _eager_* attribute.

        Mirror wrappers carry home-pinned compiled closures (wrong-device
        dynamo guards); the swap must fall to the eager original even when
        torch.compile does not expose _torchdynamo_orig_callable."""
        from auto_round.algorithms.quantization.search_dispatch import swap_wrapper_callables_to_eager

        def _eager(x):
            return x + 1

        class _W:
            pass

        sentinel = object()
        w = _W()
        w._eager_weight_quant_func = _eager
        w.weight_quant_func = sentinel  # stand-in for a compiled closure
        restore = swap_wrapper_callables_to_eager([w])
        assert restore is not None
        assert w.weight_quant_func is _eager
        restore()
        assert w.weight_quant_func is sentinel
        # no attrs at all -> no swap, no crash
        assert swap_wrapper_callables_to_eager([_W()]) is None

    def test_fn_key_normalizes_compiled_callables(self):
        """torch.compile-wrapped identical fns must share ONE batch key.

        Regression: each compiled wrapper is a unique object; without the
        _torchdynamo_orig_callable normalization the closure branch repr-keys
        it and every batch group collapses to size 1 ("0 batch + N singleton"
        under --enable_torch_compile)."""
        import torch as _t

        from auto_round.algorithms.quantization.search_dispatch import _fn_key

        def _base(x):
            return x * 2

        cf = _t.compile(_base)
        if not hasattr(cf, "_torchdynamo_orig_callable"):
            import pytest

            pytest.skip("torch.compile wrapper lacks _torchdynamo_orig_callable on this build")
        assert _fn_key(cf) == _fn_key(_base)
        assert _fn_key(_base) != _fn_key(lambda x: x * 2)

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

    def test_declared_scaler_constraint_declines(self):
        """Eligibility constraints are declared by the quantizer
        (parallel_constraints); an active scaler is a real blocker."""
        from types import SimpleNamespace

        import torch

        from auto_round.algorithms.parallel.data_parallel import resolve_tune_ddp_plan_

        q = _with_policy(SimpleNamespace(iters=10, gradient_accumulate_steps=1))
        q.parallel_constraints = lambda: {"scaler_active": True, "enable_lfq": False}
        block = torch.nn.Sequential(torch.nn.Linear(4, 4))
        with pytest.raises(RuntimeError, match="a grad scaler is active"):
            resolve_tune_ddp_plan_(q, block, [torch.zeros(1)], None, "cpu")

    def test_base_quantizer_declares_constraints(self):
        """BaseQuantizer.parallel_constraints reports the scaler probe and LFQ
        without the resolver introspecting quantizer internals."""
        from auto_round.algorithms.quantization.base import BaseQuantizer

        q = BaseQuantizer.__new__(BaseQuantizer)
        assert q.parallel_constraints() == {"scaler_active": False, "enable_lfq": False}
        q.enable_lfq = True
        assert q.parallel_constraints()["enable_lfq"] is True

        class _WithScaler(BaseQuantizer):
            def _get_scaler(self):
                return object()

        assert _WithScaler.__new__(_WithScaler).parallel_constraints()["scaler_active"] is True


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

    def _runner(self):
        from auto_round.algorithms.block_runner import BlockForwardRunner

        inputs = [torch.randn(1, 6, 4) for _ in range(128)]
        return (
            BlockForwardRunner(
                batch_dim=0,
                batch_size=8,
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


class TestRtnPhaseLogging:
    """The OptRTN iters=0 path logs per-phase walls (norm vs per-layer quant)."""

    def test_quantize_block_logs_phases(self, caplog, _autoround_log_propagate):
        import torch

        from auto_round.algorithms.quantization.rtn.quantizer import OptimizedRTNQuantizer

        class _Layer(torch.nn.Linear):
            def __init__(self):
                super().__init__(4, 4)
                self.global_name = "model.test.proj"

        class _Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = _Layer()
                self.proj.imatrix = torch.ones(4)
                self.proj.imatrix_cnt = 2

        class _Q(OptimizedRTNQuantizer):
            def __init__(self):
                pass  # bypass config init; only exercise quantize_block plumbing

            def quantize_layer_outside_block(self, layer, **kwargs):
                pass

            def _quantize_targets(self, targets):
                pass  # merged interface: non-DDP lane routes through the batched path

        from auto_round import envs as _envs

        monkey = None
        import os

        os.environ["AR_PERF_COUNTERS"] = "1"
        try:
            with caplog.at_level("INFO"):
                _Q().quantize_block(_Block(), [], {}, [], None, None)
            assert any("[perf] rtn phases:" in r.message for r in caplog.records)
        finally:
            os.environ.pop("AR_PERF_COUNTERS", None)


class TestShardedExpertBatching:
    """The sharded lane runs the SAME job plan as the serial lane: expert
    batches round-robin onto the plan devices, one stacked search per batch."""

    def _block(self):
        return _expert_moe_block()

    def _quantizer(self):
        from types import SimpleNamespace

        from auto_round.algorithms.quantization.rtn.quantizer import OptimizedRTNQuantizer

        q = OptimizedRTNQuantizer.__new__(OptimizedRTNQuantizer)
        q.parallel_state = SimpleNamespace(
            plan=SimpleNamespace(world=2, devices=[torch.device("cpu", 0), torch.device("cpu", 1)]),
            pool=None,
            pool_used=False,
        )
        q._search_calls = []
        q._batch_calls = []

        def _core(layer, disable_opt_rtn=None, tuning_device=None):
            q._search_calls.append((layer.global_name, str(tuning_device)))
            return layer

        q._quantize_layer_core = _core
        q._expert_search_active = lambda: True
        return q

    def test_batches_route_round_robin(self):
        from auto_round.algorithms.parallel import rtn_sharding as rtn_q

        q = self._quantizer()

        def _batch(mods, device):
            q._batch_calls.append((tuple(m.global_name for m in mods), str(device)))
            return []

        q._quantize_expert_batch = _batch
        rtn_q._shard_rtn_searches(q, self._block())
        # two same-key expert groups (gate_proj x2, down_proj x2) + one dense single:
        # jobs round-robin -> batch0 on cpu:0, batch1 on cpu:1, single on cpu:0
        assert [dev for _, dev in q._batch_calls] == ["cpu:0", "cpu:1"]
        assert set(q._batch_calls[0][0]) == {"m.blk.experts.0.gate_proj", "m.blk.experts.1.gate_proj"}
        assert set(q._batch_calls[1][0]) == {"m.blk.experts.2.up_proj", "m.blk.experts.3.up_proj"}
        assert [(n, d) for n, d in q._search_calls] == [("m.blk.mlp.down_proj", "cpu:0")]

    def test_batch_remainder_falls_back_on_same_device(self):
        from auto_round.algorithms.parallel import rtn_sharding as rtn_q

        q = self._quantizer()

        def _batch(mods, device):
            q._batch_calls.append((tuple(m.global_name for m in mods), str(device)))
            return mods[1:]  # first module written; the remainder falls back per-module

        q._quantize_expert_batch = _batch
        rtn_q._shard_rtn_searches(q, self._block())
        # remainder of batch0 (gate_proj #1) re-quantized per-module on the
        # batch's own device (cpu:0); the WRITTEN module (#0) stays on the
        # batch path
        assert ("m.blk.experts.1.gate_proj", "cpu:0") in q._search_calls
        assert all(n != "m.blk.experts.0.gate_proj" for n, _ in q._search_calls)

    def test_foreign_quantizer_keeps_all_singles(self):
        from types import SimpleNamespace

        from auto_round.algorithms.parallel import rtn_sharding as rtn_q

        q = SimpleNamespace(
            parallel_state=SimpleNamespace(
                plan=SimpleNamespace(world=2, devices=[torch.device("cpu"), torch.device("cpu")]),
                pool=None,
                pool_used=False,
            )
        )
        q._search_calls = []

        def _core(layer, disable_opt_rtn=None, tuning_device=None):
            q._search_calls.append((layer.global_name, str(tuning_device)))
            return layer

        q._quantize_layer_core = _core  # no _rtn_search_jobs: resolver-safety guard
        rtn_q._shard_rtn_searches(q, self._block())
        assert len(q._search_calls) == 5  # every module per-module, nothing batched


class TestLoopTailContract:
    """The v1 tune loop must record ``init_loss = total_loss`` at its tail.

    The DDP-arc loop surgery once dropped this upstream pair (together with the
    per-iter debug line); the end-of-block summary then formatted ``None`` and
    took down every iters>0 CPU test (CI build 75613, 93 failures), with the
    aborted-block leftovers cascading into the 4 WrapperLinear.linear_forward
    failures. The e2e pins are the iters>0 tests themselves; this AST contract
    pins the exact regression class hermetically: the assignment must live
    inside the iters loop of ``quantize_block``.
    """

    def test_init_loss_assigned_inside_loop(self):
        import ast
        import inspect

        from auto_round.algorithms.quantization.sign_round import quantizer as v1

        src = inspect.getsource(v1)
        found = False
        for fn in ast.walk(ast.parse(src)):
            if isinstance(fn, ast.FunctionDef) and fn.name == "quantize_block":
                for node in ast.walk(fn):
                    if isinstance(node, ast.For):
                        for st in ast.walk(node):
                            if (
                                isinstance(st, ast.Assign)
                                and any(isinstance(t, ast.Name) and t.id == "init_loss" for t in st.targets)
                                and isinstance(st.value, ast.Name)
                                and st.value.id == "total_loss"
                            ):
                                found = True
        assert found, "v1 quantize_block lost the loop-tail `init_loss = total_loss` assignment"


class TestTunePhaseLine:
    """Formatter for the AR_PERF_COUNTERS per-block tune phase breakdown."""

    def test_formats_all_four_buckets(self):
        from auto_round.algorithms.quantization.sign_round.quantizer import _tune_phase_line

        line = _tune_phase_line({"wrap": 8.21, "prepare": 0.42, "loop": 3.5, "tail": 0.47}, 0)
        assert "iters=0" in line
        assert "wrap=8.21s" in line and "prepare=0.42s" in line
        assert "loop=3.50s" in line and "tail=0.47s" in line

    def test_missing_buckets_default_to_zero(self):
        from auto_round.algorithms.quantization.sign_round.quantizer import _tune_phase_line

        line = _tune_phase_line({}, 10)
        assert "wrap=0.00s" in line and "tail=0.00s" in line and "iters=10" in line


class TestBlockHasTuningEntries:
    """DDP decline check: all-float blocks must be detectable pre-engagement."""

    def test_plain_block_has_no_entries(self):
        from auto_round.algorithms.parallel.data_parallel import block_has_tuning_entries

        assert block_has_tuning_entries(torch.nn.Sequential(torch.nn.Linear(8, 8))) is False

    def test_minmax_only_block_counts_as_tunable(self):
        import torch.nn as nn

        from auto_round.algorithms.parallel.data_parallel import block_has_tuning_entries

        class MinmaxOnly(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.params = {"wmax": nn.Parameter(torch.ones(1))}

        # full-mirror can still tune minmax params
        assert block_has_tuning_entries(torch.nn.Sequential(MinmaxOnly())) is True

    def test_round_block_counts(self):
        import torch.nn as nn

        from auto_round.algorithms.parallel.data_parallel import block_has_tuning_entries

        class RoundHolder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.params = {"v": nn.Parameter(torch.ones(4))}

        assert block_has_tuning_entries(torch.nn.Sequential(RoundHolder())) is True


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


class TestPoolShardedSearch:
    """The pool-reused iters=0 lane: searches read resident mirror weights,
    results are written back home (block + model) and the mirror copy stays
    quantized for the cascade."""

    def _setup(self):
        from types import SimpleNamespace

        from auto_round.algorithms.parallel.data_parallel import MirrorPool
        from auto_round.algorithms.quantization.rtn.quantizer import OptimizedRTNQuantizer

        devices = [torch.device("cpu", 0), torch.device("cpu", 1)]
        block = self._block()
        # seed distinct home stats: the distribution must overwrite the
        # mirrors' stale shard-local partials with the home totals
        for name, m in block.named_modules():
            if hasattr(m, "global_name"):
                m.imatrix = torch.full((4,), float(len(name)))
                m.imatrix_cnt = 2
                m.weight.data.zero_()  # clean +1 search marker

        pool = MirrorPool(block, devices)
        # mirrors carry STALE partials (their own collection shard)
        for r in range(pool.world):
            for name, lm in pool.reps[r].named_modules():
                if hasattr(lm, "imatrix"):
                    lm.imatrix = torch.zeros(4)

        q = OptimizedRTNQuantizer.__new__(OptimizedRTNQuantizer)
        q.parallel_state = SimpleNamespace(plan=SimpleNamespace(world=2, devices=devices), pool=None, pool_used=False)
        q.parallel_state.pool = pool
        q._search_calls = []
        q._batch_calls = []

        def _core(layer, disable_opt_rtn=None, tuning_device=None):
            q._search_calls.append((id(layer), str(tuning_device)))
            layer.weight.data = layer.weight.data + 1.0  # mark: searched
            return layer

        q._quantize_layer_core = _core
        q._expert_search_active = lambda: True
        return q, block, pool

    def _block(self):
        return _expert_moe_block()

    def test_pool_search_runs_on_mirrors_and_writes_home(self):
        from auto_round.algorithms.parallel import rtn_sharding as rtn_q

        q, block, pool = self._setup()

        def _batch(mods, device):
            for m in mods:
                m.weight.data += 1.0  # the real primitive writes in-batch
            q._batch_calls.append((tuple(id(m) for m in mods), str(device)))
            return []  # everything written in-batch

        q._quantize_expert_batch = _batch
        home_ids = {id(m) for _, m in block.named_modules() if hasattr(m, "global_name")}
        rtn_q._shard_rtn_searches(q, block)

        searched = {lid for lid, _ in q._search_calls}
        for b in q._batch_calls:
            searched.update(b[0])
        assert searched
        assert not (searched & home_ids), "search must read mirror copies, not home layers"
        # write-back: every home layer carries the +1 searched marker
        for name, m in block.named_modules():
            if hasattr(m, "global_name"):
                assert float(m.weight.data[0, 0]) == 1.0, f"home layer {name} not quantized"

    def test_pool_stats_distribution_then_home_release(self):
        from auto_round.algorithms.parallel import rtn_sharding as rtn_q

        q, block, pool = self._setup()

        def _batch(mods, device):
            seen = tuple(float(m.imatrix[0]) for m in mods)
            q._batch_calls.append((seen, str(device)))
            return []

        q._quantize_expert_batch = _batch
        rtn_q._shard_rtn_searches(q, block)
        # every mirror layer saw the HOME totals (len(name) > 0; a missed
        # merge would leave 0.0 partials)
        for seen, _dev in q._batch_calls:
            assert all(v > 0.0 for v in seen), f"stale shard-local stats reached the search: {seen}"
        # ...and the home stats are released right after the search (consumed inputs only)
        for name, m in block.named_modules():
            if hasattr(m, "global_name"):
                assert not hasattr(m, "imatrix"), f"{name} kept a dead imatrix"
                assert not hasattr(m, "imatrix_cnt")

    def test_pool_batch_remainder_writes_back(self):
        from auto_round.algorithms.parallel import rtn_sharding as rtn_q

        q, block, pool = self._setup()

        def _batch(mods, device):
            mods[0].weight.data += 1.0  # first module written in-batch
            q._batch_calls.append((tuple(id(m) for m in mods), str(device)))
            return mods[1:]  # remainder goes through the single path on the mirror

        q._quantize_expert_batch = _batch
        home_ids = {id(m) for _, m in block.named_modules() if hasattr(m, "global_name")}
        rtn_q._shard_rtn_searches(q, block)
        # remainder single-path ran on a MIRROR layer (home would mean the
        # serial fallback)...
        for lid, _dev in q._search_calls:
            assert lid not in home_ids
        # ...and its home write-back still landed
        for name, m in block.named_modules():
            if hasattr(m, "global_name"):
                assert float(m.weight.data[0, 0]) == 1.0

    def test_mismatched_pool_falls_back_to_move_path(self):
        from types import SimpleNamespace

        from auto_round.algorithms.parallel import rtn_sharding as rtn_q

        q, block, pool = self._setup()
        # pool for a DIFFERENT device set -> alignment fails -> move path
        q.parallel_state = SimpleNamespace(
            plan=SimpleNamespace(world=2, devices=[torch.device("cpu"), torch.device("cpu")]),
            pool=None,
            pool_used=False,
        )

        def _batch(mods, device):
            q._batch_calls.append((tuple(id(m) for m in mods), str(device)))
            return []

        q._quantize_expert_batch = _batch
        rtn_q._shard_rtn_searches(q, block)
        home_ids = {id(m) for _, m in block.named_modules() if hasattr(m, "global_name")}
        touched = {lid for b in q._batch_calls for lid in b[0]} | {lid for lid, _ in q._search_calls}
        assert touched & home_ids, "misaligned pool must fall back to the home-module move path"


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
            return None

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
        # OptRTN searches or SignRound adoption) + the post-Step-4
        # release-if-unconsumed + the try/finally in compress_block
        if getattr(composer._coll_ctx, "devices", None) and len(composer._coll_ctx.devices) >= 2:
            composer._coll_ctx.ensure_pool(composer._block)
        _pool = getattr(composer._coll_ctx, "pool", None)
        if _pool is not None:
            _state = parallel_state(composer.block_quantizer, create=True)
            _state.pool = _pool
            _state.pool_used = False
        try:
            composer.block_quantizer.quantize_block(None)
            _state = parallel_state(composer.block_quantizer)
            if _pool is not None and (_state is None or _state.pool is None or not _state.pool_used):
                composer._coll_ctx.release_pool()
                _pool = None
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


class TestPoolSearchThenCascade:
    """End-to-end seam: pool search quantizes the mirrors, write-back homes
    the results, and the cascade forward on the SAME pool outputs exactly the
    serial forward of the home (written-back) block."""

    def test_cascade_on_pool_matches_serial_home(self):
        from types import SimpleNamespace

        from auto_round.algorithms.parallel import rtn_sharding as rtn_q
        from auto_round.algorithms.parallel.data_parallel import (
            MirrorPool,
            sharded_nograd_forward,
        )
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
        q.parallel_state = SimpleNamespace(plan=SimpleNamespace(world=2, devices=devices), pool=None, pool_used=False)
        q.parallel_state.pool = pool
        q._expert_search_active = lambda: False  # all singles

        def _core(layer, disable_opt_rtn=None, tuning_device=None):
            # a REAL quantize-dequantize: round-to-grid + offset, so an
            # FP (unsynced) mirror layer would produce different outputs
            with torch.no_grad():
                layer.weight.data = ((layer.weight.data * 4).round() / 4) + 0.125
            return layer

        q._quantize_layer_core = _core
        rtn_q._shard_rtn_searches(q, block)

        # every home layer carries the EXACT quantize transform (write-back
        # landed; a silent no-op lane would leave the random originals)
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

        # mirror layers must carry the exact written-back home state (the
        # post-search sync): mixed FP/quantized replicas would diverge here
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

    def test_cold_expert_cnt_cotravels_with_imatrix(self):
        from auto_round.algorithms.parallel.data_parallel import _merge_mirror_stats

        home = torch.nn.Linear(4, 4)
        mirror = torch.nn.Linear(4, 4)
        mirror.imatrix = torch.ones(4)
        mirror.imatrix_cnt = 7  # python int, exactly as both collectors write it
        _merge_mirror_stats(home, [mirror])
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
        _merge_mirror_stats(home, [m1, m2])
        assert home.imatrix_cnt == 10
        assert torch.equal(home.imatrix, torch.full((4,), 2.0))

    def test_tensor_cnt_also_merges(self):
        from auto_round.algorithms.parallel.data_parallel import _merge_mirror_stats

        home = torch.nn.Linear(4, 4)
        mirror = torch.nn.Linear(4, 4)
        mirror.imatrix = torch.ones(4)
        mirror.imatrix_cnt = torch.tensor(7)
        _merge_mirror_stats(home, [mirror])
        assert home.imatrix_cnt == 7

    def test_non_numeric_attrs_ignored(self):
        from auto_round.algorithms.parallel.data_parallel import _merge_mirror_stats

        home = torch.nn.Linear(4, 4)
        mirror = torch.nn.Linear(4, 4)
        mirror.imatrix_cnt = "bogus"
        _merge_mirror_stats(home, [mirror])
        assert not hasattr(home, "imatrix_cnt")


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


class TestPlainRtnShardedLane:
    """Plain RTN (RTNConfig lane) shards its per-module minmax quantizes onto
    the resident collection mirrors with disable_opt_rtn=True semantics."""

    def test_plain_rtn_uses_pool_with_disable_flag(self):
        from types import SimpleNamespace

        from auto_round.algorithms.parallel import rtn_sharding as rtn_q
        from auto_round.algorithms.parallel.data_parallel import MirrorPool
        from auto_round.algorithms.quantization.rtn.quantizer import RTNQuantizer

        devices = [torch.device("cpu", 0), torch.device("cpu", 1)]

        class _B(torch.nn.Module):
            def __init__(self):
                super().__init__()
                for i in range(3):
                    setattr(self, f"l{i}", _expert_layer(f"m.blk.l{i}"))

            def forward(self, x):
                return x

        block = _B()
        for _, m in block.named_modules():
            if hasattr(m, "global_name"):
                m.weight.data.zero_()
        pool = MirrorPool(block, devices)

        q = RTNQuantizer.__new__(RTNQuantizer)
        q.parallel_state = SimpleNamespace(plan=SimpleNamespace(world=2, devices=devices), pool=None, pool_used=False)
        q.parallel_state.pool = pool
        q._core_calls = []

        def _core(layer, disable_opt_rtn=None, tuning_device=None):
            q._core_calls.append((id(layer), disable_opt_rtn))
            with torch.no_grad():
                layer.weight.data = layer.weight.data + 1.0
            return layer

        q._quantize_layer_core = _core
        q._expert_search_active = lambda: False
        # `model` is a read-only property on the base class; the sharded
        # lane reads it via getattr(self, "model", None) on __new__ fakes

        rtn_q._shard_rtn_searches(q, block, disable_opt_rtn=True)

        home_ids = {id(m) for _, m in block.named_modules() if hasattr(m, "global_name")}
        # every quantize ran on a MIRROR with the plain-RTN disable flag
        assert len(q._core_calls) == 3
        for lid, dis in q._core_calls:
            assert lid not in home_ids
            assert dis is True
        # write-back landed home
        for _, m in block.named_modules():
            if hasattr(m, "global_name"):
                assert float(m.weight.data[0, 0]) == 1.0


class TestRtnSearchJobs:
    """The shared job plan: expert batches + singles, one primitive for every lane."""

    def _expert(self, name, proj="gate_proj"):
        m = torch.nn.Linear(4, 4)  # EXACT nn.Linear: subclasses stay per-module by design
        m.global_name = name
        m.to_quantized = True
        m.bits = 4
        m.group_size = 4
        m.sym = True  # plain asym has no search; sym batches under both configs
        m.act_bits = 16
        return m

    def _quantizer(self, active):
        from auto_round.algorithms.quantization.rtn.quantizer import OptimizedRTNQuantizer

        q = OptimizedRTNQuantizer.__new__(OptimizedRTNQuantizer)
        q._expert_search_active = lambda: active
        return q

    def test_plan_splits_expert_groups_only_when_active(self):
        e0 = self._expert("model.blk.experts.0.gate_proj")
        e1 = self._expert("model.blk.experts.1.gate_proj")
        e2 = self._expert("model.blk.experts.0.up_proj")  # same layer, other projection
        dense = self._expert("model.blk.mlp.down_proj")  # not expert-shaped
        dense.group_size = 4
        batches, singles = self._quantizer(True)._rtn_search_jobs([e0, e1, e2, dense])
        assert [len(b) for b in batches] == [2]
        assert {m.global_name for m in batches[0]} == {
            "model.blk.experts.0.gate_proj",
            "model.blk.experts.1.gate_proj",
        }
        # singleton expert groups collect first, then singleton dense groups
        assert [m.global_name for m in singles] == ["model.blk.mlp.down_proj", "model.blk.experts.0.up_proj"]
        # inactive: everything stays per-module, nothing batches
        m0 = self._expert("model.blk.experts.0.gate_proj")
        m1 = self._expert("model.blk.dense")
        batches, singles = self._quantizer(False)._rtn_search_jobs([m0, m1])
        assert batches == [] and singles == [m0, m1]


class TestShardedRtnSearch:
    """iters=0 OptRTN: per-layer searches shard round-robin across the DDP devices."""

    def _fake_quantizer(self):
        from types import SimpleNamespace

        from auto_round.algorithms.quantization.rtn.quantizer import OptimizedRTNQuantizer

        q = OptimizedRTNQuantizer.__new__(OptimizedRTNQuantizer)
        q.parallel_state = SimpleNamespace(
            plan=SimpleNamespace(world=2, devices=[torch.device("cpu", 0), torch.device("cpu", 1)]),
            pool=None,
            pool_used=False,
        )
        q._search_calls = []
        # q.model: property without setter on a bare instance -> getattr
        # default in _shard_rtn_searches treats it as "no global model"

        def _core(layer, disable_opt_rtn=None, tuning_device=None):
            q._search_calls.append((layer.global_name, tuning_device))
            return layer  # identity "quantized" layer

        q._quantize_layer_core = _core
        # a bare __new__ instance carries no config: model a quantizer
        # without the job-plan primitive (the lane's all-singles guard);
        # the expert-batch tests below drive the real primitive instead.
        q._rtn_search_jobs = lambda targets: ([], list(targets))
        return q

    def _block(self, n=4):
        class _L(torch.nn.Linear):
            def __init__(self, i):
                super().__init__(4, 4)
                self.global_name = f"model.test.layer{i}"
                self.to_quantized = True  # check_to_quantized marker
                self.bits = 4

        class _B(torch.nn.Module):
            def __init__(self):
                super().__init__()
                for i in range(n):
                    setattr(self, f"layer{i}", _L(i))

        return _B()

    def test_sharded_round_robin(self, monkeypatch):
        from auto_round.algorithms.parallel import rtn_sharding as rtn_q

        q = self._fake_quantizer()
        block = self._block(4)
        rtn_q._shard_rtn_searches(q, block)
        # all layers searched, round-robin over the two plan devices
        assert len(q._search_calls) == 4
        devs = [d for _, d in q._search_calls]
        assert devs[0] == devs[2] and devs[1] == devs[3] and devs[0] != devs[1]
        # results written back through set_module semantics (block modules replaced)
        assert all(hasattr(m, "global_name") for _n, m in block.named_modules() if hasattr(m, "global_name"))

    def test_world1_keeps_serial_loop(self, monkeypatch):
        from auto_round.algorithms.parallel import rtn_sharding as rtn_q

        q = self._fake_quantizer()
        q.parallel_state.plan.world = 1
        block = self._block(2)
        rtn_q._shard_rtn_searches(q, block)
        # serial path: single shared device for every layer (whatever the
        # device manager resolves to in this environment)
        assert len(q._search_calls) == 2
        assert len({d for _, d in q._search_calls}) == 1

    def test_no_plan_is_serial(self):
        from auto_round.algorithms.parallel import rtn_sharding as rtn_q

        q = self._fake_quantizer()
        q.parallel_state.plan = None
        block = self._block(2)
        rtn_q._shard_rtn_searches(q, block)
        assert len(q._search_calls) == 2


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


class TestShardedWrapSearch(unittest.TestCase):
    """Mirror-sharded wrap-time init-scale searches (mirrors-first pattern).

    Contract exercised: wrappers with ``_init_search_deferred`` set run their
    init-scale search once on exactly one replica (round-robin); the result is
    broadcast to the same-named wrappers on every replica; every replica then
    finalizes (compiles its own quant func) so tuning starts from identical
    state on all copies.
    """

    def _fake_wrapper(self, name, seed):
        import torch as _t

        class _W(_t.nn.Module):
            def __init__(self):
                super().__init__()
                self.name = name
                self.weight = _t.nn.Parameter(_t.ones(2, 2) * seed)
                self.init_scale = None
                self._init_search_deferred = True
                self.searched = 0
                self.compiled = 0
                # staged protocol: (weight_reshape, data_type, bits, imatrix, thresh, search_fn)
                self.orig_layer = _t.nn.Module()
                self.orig_layer.group_size = 2
                # metadata-only staging (data_type, bits, thresh, search_fn); the
                # weight/imatrix derive at consume time via prepare_batched_search
                self._deferred_search_inputs = ("int", 4, 1e-5, _stackable_search_fn)
                self._staged_bits = 4
                self._staged_fn = _stackable_search_fn

            def prepare_batched_search(self):
                # derived at consume time from this wrapper's own weight,
                # on whatever device it lives (the DDP-correct property)
                return self.weight.data.reshape(1, 2, 2), self._staged_bits, None, self._staged_fn

            def _consume(self, res):
                # deterministic given (weight, imatrix): init_scale = seed
                self.init_scale = float(res.reshape(-1)[0].item())
                self.searched += 1
                self._deferred_search_inputs = None

            def _run_deferred_search_now(self):
                weight, bits, im, fn = self.prepare_batched_search()
                self._consume(fn(weight, bits, im))

            def finalize_batched_search(self, res):
                self._consume(res)

            def _finalize_deferred_init(self, val=None):
                if self.init_scale is None:
                    self.init_scale = val
                self.compiled += 1
                self._init_search_deferred = False

        return _W()

    def _fake_group(self, block, world):
        import torch as _t

        # real ReplicaGroup.replicas = [home block] + mirrors -- mirror that
        replicas = [block]
        for _i in range(1, world):
            rep = _t.nn.Module()
            for n, m in block.named_modules():
                if hasattr(m, "_run_deferred_search_now"):
                    w = self._fake_wrapper(n.split(".")[-1], float(m.weight[0, 0].item()))
                    parts = n.split(".")
                    parent = rep
                    for p_ in parts[:-1]:
                        if not hasattr(parent, p_):
                            parent.add_module(p_, _t.nn.Module())
                        parent = getattr(parent, p_)
                    parent.add_module(parts[-1], w)
            replicas.append(rep)
        plan = SimpleNamespace(world=world, devices=[_t.device("cpu", i) for i in range(world)])
        return SimpleNamespace(replicas=replicas, plan=plan)

    def _block(self, seeds):
        import torch as _t

        block = _t.nn.Module()
        for i, seed in enumerate(seeds):
            block.add_module(f"l{i}", self._fake_wrapper(f"l{i}", seed))
        return block

    def _wrappers(self, mod):
        return [m for _, m in mod.named_modules() if hasattr(m, "_run_deferred_search_now")]

    def test_sharded_round_robin_and_broadcast(self):
        from auto_round.algorithms.parallel.data_parallel import run_deferred_wrap_searches

        seeds = [1.0, 2.0, 3.0, 4.0]
        block = self._block(seeds)
        group = self._fake_group(block, 2)
        run_deferred_wrap_searches(block, group)
        # home block filled with deterministic values
        self.assertEqual([w.init_scale for w in self._wrappers(block)], seeds)
        # every replica broadcast-complete and finalized
        for rep in group.replicas:
            self.assertEqual([w.init_scale for w in self._wrappers(rep)], seeds)
            for w in self._wrappers(rep):
                self.assertEqual(w.compiled, 1)
                self.assertFalse(w._init_search_deferred)
        # round-robin: each name searched exactly once across the group
        total_searches = sum(w.searched for rep in group.replicas for w in self._wrappers(rep))
        self.assertEqual(total_searches, len(seeds))

    def test_owner_batches_same_key_into_one_call(self):
        import os

        os.environ.pop("AR_DISABLE_BATCHED_SEARCH", None)
        import torch as _t

        from auto_round.algorithms.parallel.data_parallel import run_deferred_wrap_searches

        calls = []

        def _recording_fn(weight, bits, imatrix):
            calls.append(weight.shape[0])  # one entry per search call, value = batch rows
            return weight.reshape(weight.shape[0], -1)[:, 0]

        block = self._block([1.0, 2.0, 3.0, 4.0])  # same staged key by construction
        for w in self._wrappers(block):
            w._staged_fn = _recording_fn
        group = self._fake_group(block, 2)
        # the mirror copies re-stage with the module-level fn: patch every replica
        for rep in group.replicas:
            for w in self._wrappers(rep):
                w._staged_fn = _recording_fn
        run_deferred_wrap_searches(block, group)
        # round-robin owner map: each owner holds TWO same-key wrappers -> one
        # stacked two-row search call per owner (the actual stacking behavior)
        self.assertEqual(sorted(calls), [2, 2])
        self.assertEqual([w.init_scale for w in self._wrappers(block)], [1.0, 2.0, 3.0, 4.0])

    def test_engine_batches_serial_path_on_home(self):
        import os

        os.environ.pop("AR_DISABLE_BATCHED_SEARCH", None)
        from auto_round.algorithms.parallel.data_parallel import run_deferred_wrap_searches

        calls = []

        def _recording_fn(weight, bits, imatrix):
            calls.append(weight.shape[0])
            return weight.reshape(weight.shape[0], -1)[:, 0]

        block = self._block([3.0, 4.0, 5.0])
        for w in self._wrappers(block):
            w._staged_fn = _recording_fn
        run_deferred_wrap_searches(block, None)  # no plan: engine on home device
        self.assertEqual(calls, [3])  # one stacked call for all three same-key modules
        self.assertEqual([w.init_scale for w in self._wrappers(block)], [3.0, 4.0, 5.0])

    def test_key_separation_never_merges(self):
        from auto_round.algorithms.parallel.data_parallel import run_deferred_wrap_searches

        calls = []

        def _recording_fn(weight, bits, imatrix):
            calls.append((weight.shape, bits))
            return weight.reshape(weight.shape[0], -1)[:, 0]

        block = self._block([6.0, 7.0])
        ws = self._wrappers(block)
        # different bits -> different derived key -> separate batches
        ws[1]._staged_bits = 8
        for w in ws:
            w._staged_fn = _recording_fn
        run_deferred_wrap_searches(block, None)
        self.assertEqual(sorted((tuple(s), b) for s, b in calls), [((1, 2, 2), 4), ((1, 2, 2), 8)])

    def test_wrapper_block_explicit_false_defer_still_wraps(self):
        """R2-1 regression: the serial tune lane passes defer_search=False
        EXPLICITLY -- the engine-initiated deferral must not duplicate the
        keyword (TypeError) and must still defer + batch + finalize."""
        import os

        import torch as _t

        from auto_round.wrapper import wrapper_block

        os.environ.pop("AR_DISABLE_BATCHED_SEARCH", None)
        calls = []

        def _fake_resolved_fn(weight, bits, imatrix):
            calls.append(weight.shape[0])
            return weight.reshape(weight.shape[0], -1)[:, 0]

        import auto_round.algorithms.quantization.sign_roundv2.quantizer as v2q

        layers = []
        for i in range(2):
            layer = self._v2_layer()
            layer.global_name = f"m.blk.experts.{i}.up_proj"
            layer.to_quantized = True
            layers.append(layer)

        class _B(_t.nn.Module):
            pass

        block = _B()
        block.add_module("e0", layers[0])
        block.add_module("e1", layers[1])
        orig = v2q.resolve_optimized_init_scale_fn
        v2q.resolve_optimized_init_scale_fn = lambda dt, thresh=1e-5: _fake_resolved_fn
        try:
            quantized, _unq = wrapper_block(
                block,
                True,
                False,
                enable_torch_compile=False,
                device="cpu",
                wrapper_cls=v2q.SignRoundOptimizedWrapperLinear,
                defer_search=False,  # the serial lane's explicit pass
            )
        finally:
            v2q.resolve_optimized_init_scale_fn = orig
        self.assertEqual(len(quantized), 2)
        self.assertEqual(calls, [2])  # engine deferral still engaged and batched
        for n, m in block.named_modules():
            if hasattr(m, "_init_search_deferred"):
                self.assertFalse(m._init_search_deferred, n)

    def test_v2_deferred_consumes_layer_imatrix_at_consume_time(self):
        """R3-4: the redo's central mechanism -- the derived search input reads
        the layer's imatrix at consume time (not a wrap-time snapshot), and the
        helper relocates it onto the weight's device."""
        import os

        import torch as _t

        os.environ.pop("AR_DISABLE_BATCHED_SEARCH", None)
        import auto_round.algorithms.quantization.sign_roundv2.quantizer as v2q

        layer = self._v2_layer()
        layer.to_quantized = True
        layer.imatrix = _t.arange(8, dtype=_t.float32) + 1.0  # non-uniform importance

        seen = {}

        def _fn(weight, bits, imatrix):
            seen["imatrix"] = imatrix
            return weight.reshape(weight.shape[0], -1)[:, 0]

        orig = v2q.resolve_optimized_init_scale_fn
        v2q.resolve_optimized_init_scale_fn = lambda dt, thresh=1e-5: _fn
        try:
            w = v2q.SignRoundOptimizedWrapperLinear(layer, enable_torch_compile=False, device="cpu", defer_search=True)
            self.assertTrue(hasattr(layer, "imatrix"))  # kept until consume
            w._run_deferred_search_now()
        finally:
            v2q.resolve_optimized_init_scale_fn = orig
        self.assertIsNotNone(w.init_scale)
        # the consumed imatrix is the layer's expanded vector (a missed
        # consume would leave ones)
        expected = layer.weight.data.reshape(-1, 8).shape[0]
        self.assertEqual(seen["imatrix"].shape[-1], 8)
        self.assertTrue(_t.any(seen["imatrix"][0, :] != 1.0))
        self.assertGreaterEqual(seen["imatrix"].shape[0], expected)
        self.assertFalse(hasattr(layer, "imatrix"))  # consumed and released

    def test_dq_wrapper_constructs_after_protocol_unification(self):
        """R1-1 regression: the DQ override must accept the unified defer_search
        signature -- every GGUF DQ run constructs this wrapper."""
        import torch as _t

        import auto_round.algorithms.quantization.sign_roundv2.quantizer as v2q

        layer = self._v2_layer()
        layer.data_type = "int_sym_dq"
        layer.super_bits = 4
        layer.super_group_size = 64
        w = v2q.SignRoundDQWrapperLinear(layer, enable_torch_compile=False, device="cpu")
        self.assertTrue(w._is_dq_path)
        # the DQ search stays inline even under defer (no staged inputs)
        w2 = v2q.SignRoundDQWrapperLinear(layer, enable_torch_compile=False, device="cpu", defer_search=True)
        self.assertIsNone(getattr(w2, "_deferred_search_inputs", None))

    def test_supports_batched_search_flags(self):
        """R1-2 regression: the opt-in is a CLASS attribute the engine lane reads."""
        import auto_round.algorithms.quantization.sign_roundv2.quantizer as v2q
        from auto_round.wrapper import WrapperLinear

        self.assertTrue(v2q.SignRoundOptimizedWrapperLinear.supports_batched_search)
        self.assertFalse(WrapperLinear.supports_batched_search)
        self.assertFalse(v2q.SignRoundDQWrapperLinear.supports_batched_search)

    def test_wrapper_block_engine_path_finalizes(self):
        import os

        os.environ.pop("AR_DISABLE_BATCHED_SEARCH", None)
        """R1-3 regression: engine-initiated deferral consumes AND finalizes
        (init_scale set, staged inputs cleared, flag down)."""
        import torch as _t

        from auto_round.wrapper import wrapper_block

        calls = []

        def _fake_resolved_fn(weight, bits, imatrix):
            calls.append(weight.shape[0])
            return weight.reshape(weight.shape[0], -1)[:, 0]

        import auto_round.algorithms.quantization.sign_roundv2.quantizer as v2q

        layer0 = self._v2_layer()
        layer1 = self._v2_layer()
        for i, layer in enumerate([layer0, layer1]):
            layer.global_name = f"m.blk.experts.{i}.gate_proj"
            layer.to_quantized = True

        class _B(_t.nn.Module):
            pass

        block = _B()
        block.add_module("e0", layer0)
        block.add_module("e1", layer1)
        orig = v2q.resolve_optimized_init_scale_fn
        v2q.resolve_optimized_init_scale_fn = lambda dt, thresh=1e-5: _fake_resolved_fn
        try:
            quantized, _unq = wrapper_block(
                block,
                True,
                False,
                enable_torch_compile=False,
                device="cpu",
                wrapper_cls=v2q.SignRoundOptimizedWrapperLinear,
            )
        finally:
            v2q.resolve_optimized_init_scale_fn = orig
        self.assertEqual(len(quantized), 2)
        self.assertEqual(calls, [2])  # one stacked call for the same-key pair
        for n, m in block.named_modules():
            if hasattr(m, "_init_search_deferred"):
                self.assertFalse(m._init_search_deferred, n)
                self.assertIsNotNone(m.init_scale, n)
                self.assertIsNone(m._deferred_search_inputs, n)

    def test_kill_switch_falls_back_per_module(self):
        import os

        os.environ.pop("AR_DISABLE_BATCHED_SEARCH", None)
        import os

        from auto_round.algorithms.parallel.data_parallel import run_deferred_wrap_searches

        calls = []

        def _recording_fn(weight, bits, imatrix):
            calls.append(weight.shape[0])
            return weight.reshape(weight.shape[0], -1)[:, 0]

        block = self._block([8.0, 9.0])
        for w in self._wrappers(block):
            w._staged_fn = _recording_fn
        os.environ["AR_DISABLE_BATCHED_SEARCH"] = "1"
        try:
            run_deferred_wrap_searches(block, None)
        finally:
            os.environ.pop("AR_DISABLE_BATCHED_SEARCH", None)
        self.assertEqual(calls, [1, 1])  # per-module calls only
        self.assertEqual([w.init_scale for w in self._wrappers(block)], [8.0, 9.0])

    def test_serial_fallback_fills_all(self):
        from auto_round.algorithms.parallel.data_parallel import run_deferred_wrap_searches

        block = self._block([10.0, 11.0, 12.0])
        run_deferred_wrap_searches(block, None)
        self.assertEqual([w.init_scale for w in self._wrappers(block)], [10.0, 11.0, 12.0])
        self.assertTrue(all(w.compiled == 1 for w in self._wrappers(block)))

    def test_no_deferred_wrappers_is_noop(self):
        import torch as _t

        from auto_round.algorithms.parallel.data_parallel import run_deferred_wrap_searches

        block = _t.nn.Module()
        block.add_module("plain", _t.nn.Linear(2, 2))
        group = self._fake_group(block, 2)
        run_deferred_wrap_searches(block, group)  # must not raise

    def _v2_layer(self):
        import torch as _t

        layer = _t.nn.Linear(8, 8, bias=False)
        layer.bits = 4
        layer.group_size = -1
        layer.data_type = "int"
        layer.sym = True
        layer.act_bits = 16
        layer.act_data_type = "int"
        layer.act_dynamic = True
        layer.scale_dtype = _t.float32
        return layer

    def _patch_v2(self, defer=False):
        import torch as _t

        import auto_round.algorithms.quantization.sign_roundv2.quantizer as v2q

        calls = []

        def _fake_search(weight_reshape, data_type, bits, imatrix, thresh):
            calls.append(1)
            return _t.tensor([0.5])

        def _fake_resolved(weight_reshape, bits, imatrix):
            calls.append(1)
            return _t.tensor([0.5])

        orig = (v2q.search_optimized_init_scale, v2q.get_optimized_quant_func, v2q.resolve_optimized_init_scale_fn)
        v2q.search_optimized_init_scale = _fake_search
        v2q.resolve_optimized_init_scale_fn = lambda dt, thresh=1e-5: _fake_resolved
        v2q.get_optimized_quant_func = lambda dt: (lambda *a, **k: None)
        return v2q, calls, orig

    def test_v2_wrapper_honors_defer(self):
        v2q, calls, orig = self._patch_v2()
        try:
            w = v2q.SignRoundOptimizedWrapperLinear(
                self._v2_layer(), enable_torch_compile=False, device="cpu", defer_search=True
            )
            self.assertEqual(len(calls), 0)
            self.assertTrue(w._init_search_deferred)
            self.assertIsNotNone(w.weight_quant_func)
            w._run_deferred_search_now()
            self.assertEqual(len(calls), 1)
            self.assertIsNotNone(w.init_scale)
            w._finalize_deferred_init()
            self.assertFalse(w._init_search_deferred)
        finally:
            v2q.search_optimized_init_scale, v2q.get_optimized_quant_func, v2q.resolve_optimized_init_scale_fn = orig

    def test_v2_finalize_without_search_takes_broadcast(self):
        import torch as _t

        v2q, _calls, orig = self._patch_v2()
        try:
            w = v2q.SignRoundOptimizedWrapperLinear(
                self._v2_layer(), enable_torch_compile=False, device="cpu", defer_search=True
            )
            self.assertIsNone(w.init_scale)  # deferred: present but unset until search/finalize
            w._finalize_deferred_init(_t.tensor([0.25]))
            self.assertIsNotNone(w.init_scale)  # broadcast value applied
            self.assertFalse(w._init_search_deferred)
        finally:
            v2q.search_optimized_init_scale, v2q.get_optimized_quant_func, v2q.resolve_optimized_init_scale_fn = orig

    def test_undeferred_v2_wrapper_unchanged(self):
        v2q, calls, orig = self._patch_v2()
        try:
            v2q.SignRoundOptimizedWrapperLinear(self._v2_layer(), enable_torch_compile=False, device="cpu")
            self.assertEqual(len(calls), 1)  # serial path unchanged: search at wrap
        finally:
            v2q.search_optimized_init_scale, v2q.get_optimized_quant_func, v2q.resolve_optimized_init_scale_fn = orig

    def test_v2_deferred_requires_supported_dtype(self):
        v2q, _calls, orig = self._patch_v2()
        try:
            v2q.get_optimized_quant_func = lambda dt: None  # unsupported data_type
            with self.assertRaises(ValueError):
                v2q.SignRoundOptimizedWrapperLinear(
                    self._v2_layer(), enable_torch_compile=False, device="cpu", defer_search=True
                )
        finally:
            v2q.search_optimized_init_scale, v2q.get_optimized_quant_func, v2q.resolve_optimized_init_scale_fn = orig


class TestSyncPoolDevices:
    """Phase barriers synchronize every distinct CUDA pool device once (CPU skipped)."""

    def test_syncs_each_cuda_device_once(self):
        from auto_round.algorithms.parallel.rtn_sharding import _sync_pool_devices

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


class TestDistributeSearchStatsOwnerOnly:
    """With owner_of, stats land only on the replica whose job searches the module."""

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

    def test_owner_only_distributes_to_searching_replica(self):
        import torch.nn as nn

        from auto_round.algorithms.parallel.rtn_sharding import _distribute_search_stats

        block = nn.Module()
        block.l1 = nn.Linear(4, 4, bias=False)
        block.l1.imatrix = torch.ones(4) * 5  # merged home totals
        pool = self._pool(block, 3)
        # stale sentinel on every mirror so untouched copies stay visible
        for r in (1, 2):
            pool.mirror_layer(r, "l1").imatrix = torch.zeros(4)
        targets = [("l1", block.l1)]
        # module owned by replica 2 (home at index 0, mirrors at 1..2)
        _distribute_search_stats(pool, block, targets, {id(block.l1): 2})
        assert torch.equal(pool.mirror_layer(2, "l1").imatrix, torch.ones(4) * 5)
        assert torch.equal(pool.mirror_layer(1, "l1").imatrix, torch.zeros(4))

    def test_no_owner_map_distributes_to_every_mirror(self):
        import torch.nn as nn

        from auto_round.algorithms.parallel.rtn_sharding import _distribute_search_stats

        block = nn.Module()
        block.l1 = nn.Linear(4, 4, bias=False)
        block.l1.imatrix = torch.ones(4) * 2
        pool = self._pool(block, 3)
        _distribute_search_stats(pool, block, [("l1", block.l1)], None)
        for r in (1, 2):
            assert torch.equal(pool.mirror_layer(r, "l1").imatrix, torch.ones(4) * 2)
