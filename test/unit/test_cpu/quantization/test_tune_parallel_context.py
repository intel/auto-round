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
"""Unit tests for the TuneParallelContext (P1 tune + P3 collection facade).

The engine primitives (ReplicaGroup, sharded_nograd_forward, ...) are faked:
these tests pin the context's own semantics -- shard drawing, loss
accounting, the forgotten-sync guard, thread fan-out ordering, and the
collection-path serial fallbacks. Engagement/eligibility against the real
resolver is covered by test_ddp_core on the live quantize_block path.
"""

import logging
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from auto_round.algorithms.parallel import tune_parallel as tune_parallel_mod
from auto_round.algorithms.parallel.tune_parallel import TuneParallelContext


class _FakeGroup:
    """Minimal ReplicaGroup double: sequential execution, recorded calls."""

    def __init__(self, world=2):
        self.world = world
        self.replicas = [torch.nn.Linear(2, 2) for _ in range(world)]
        self.threaded_calls = []
        self.sync_calls = []
        self.torn_down = False

    def run_threaded(self, fns):
        self.threaded_calls.append(list(fns))
        for fn in fns:  # sequential: deterministic and CPU-safe
            fn()

    def sync_grads(self, params_per_replica, sign_exchange=False):
        self.sync_calls.append((params_per_replica, sign_exchange))

    def teardown(self):
        self.torn_down = True


class _FixedSampler:
    """Yields a fixed global-index schedule (list of lists) in order."""

    def __init__(self, draws):
        self._draws = list(draws)
        self._i = 0

    def next_batch(self):
        out = self._draws[self._i]
        self._i += 1
        return out


def _make_tune_ctx(world=2, nsamples=8, shard_size=2):
    ctx = TuneParallelContext()
    ctx.quantizer = mock.Mock(enable_minmax_tuning=False)
    ctx.block = torch.nn.Linear(2, 2)
    ctx.plan = mock.Mock(shard_size=shard_size, notes=[])
    ctx.group = _FakeGroup(world=world)
    ctx.nsamples = nsamples
    ctx.perf = {"build": 0.0, "warm": 0.0, "fwd": [], "bwd": [], "exch": [], "step": []}
    return ctx


class TestNextShards:
    def test_sampler_path_rebuilds_global_indices_from_shards(self):
        ctx = _make_tune_ctx()
        ctx._samplers = [
            mock.Mock(next_batch=mock.Mock(return_value=[0, 1])),
            mock.Mock(next_batch=mock.Mock(return_value=[2, 3])),
        ]
        shards, gidx = ctx.next_shards(index_sampler=mock.Mock())
        assert shards == [[0, 1], [2, 3]]
        assert gidx == [0, 1, 2, 3]

    def test_fallback_slices_global_indices(self):
        ctx = _make_tune_ctx(world=2)
        sampler = mock.Mock(next_batch=mock.Mock(return_value=[0, 1, 2, 3]))
        shards, gidx = ctx.next_shards(index_sampler=sampler)
        assert shards == [[0, 1], [2, 3]]
        assert gidx == [0, 1, 2, 3]

    def test_shards_none_when_batch_not_divisible(self):
        ctx = _make_tune_ctx(world=2)
        ctx.shards(nsamples=8, global_batch_size=3)
        assert ctx._samplers is None


class TestMeanLoss:
    def test_world_normalized_accounting(self):
        ctx = _make_tune_ctx(world=2)
        losses = [torch.tensor(4.0), torch.tensor(6.0)]
        assert ctx.mean_loss(losses, num_elm=2) == pytest.approx(2.5)  # (4+6)/2/2

    def test_sum_reduced_accounting(self):
        ctx = _make_tune_ctx(world=2)
        losses = [torch.tensor(6.0), torch.tensor(10.0)]
        # sum-reduced: shard sums add to the global sum (16) / num_elm=2 -> 8.0
        assert ctx.mean_loss(losses, num_elm=2, divide_world=False) == pytest.approx(8.0)

    def test_num_elm_nonpositive_treated_as_one(self):
        ctx = _make_tune_ctx(world=2)
        assert ctx.mean_loss([torch.tensor(2.0), torch.tensor(4.0)], num_elm=0) == pytest.approx(3.0)

    def test_none_losses_skipped(self):
        ctx = _make_tune_ctx(world=2)
        assert ctx.mean_loss([torch.tensor(2.0), None], num_elm=1) == pytest.approx(1.0)


class TestRunStep:
    def _step_fn(self, rep, shard, dev, rec):
        rec.fwd, rec.bwd = 0.25, 0.5
        return torch.tensor(float(sum(shard)))

    def test_fan_out_returns_detached_losses_and_walls(self):
        ctx = _make_tune_ctx(world=2)
        losses = ctx.run_step(self._step_fn, [[1, 2], [3, 4]])
        assert [l.item() for l in losses] == [3.0, 7.0]
        assert all(l.requires_grad is False for l in losses)
        assert ctx.perf["fwd"] == [0.25]
        assert ctx.perf["bwd"] == [0.5]

    def test_forgotten_sync_guard_fires_on_second_run_step(self):
        ctx = _make_tune_ctx(world=2)
        # the module logs through the shared 'autoround' logger (propagate=False),
        # which caplog cannot capture -- spy on the module attribute instead
        with mock.patch.object(tune_parallel_mod, "logger") as lg:
            ctx.run_step(self._step_fn, [[1], [2]])
            ctx.run_step(self._step_fn, [[3], [4]])
        assert any("without an intervening sync_grads" in str(c) for c in lg.error.call_args_list)

    def test_guard_silent_after_sync(self):
        ctx = _make_tune_ctx(world=2)
        ctx.params_per_replica = [[], []]
        with mock.patch.object(tune_parallel_mod, "logger") as lg:
            ctx.run_step(self._step_fn, [[1], [2]])
            ctx.sync_grads(sign_exchange=True)
            ctx.run_step(self._step_fn, [[3], [4]])
        assert not any("without an intervening sync_grads" in str(c) for c in lg.error.call_args_list)
        assert ctx.group.sync_calls == [([[], []], True)]


class TestStepAndTeardown:
    def test_step_runs_home_first_then_mirrors(self):
        ctx = _make_tune_ctx(world=2)
        ctx.mirror_optimizers = [mock.Mock(), mock.Mock()]
        ctx.mirror_schedules = [mock.Mock(), mock.Mock()]
        order = []
        home = lambda: order.append("home")  # noqa: E731
        for opt in ctx.mirror_optimizers:
            opt.step.side_effect = lambda o=opt: order.append(f"mirror-{id(o) % 100}")
        ctx.step(home)
        assert order[0] == "home"
        assert len(order) == 3  # home + 2 mirror steps
        for opt in ctx.mirror_optimizers:
            opt.zero_grad.assert_called_once()
        assert ctx.perf["step"] and ctx.perf["step"][0] >= 0.0

    def test_teardown_marks_and_times(self):
        ctx = _make_tune_ctx()
        ctx.teardown()
        assert ctx.group.torn_down
        assert "teardown" in ctx.perf


class TestCollectionContext:
    def test_for_collection_non_list_inputs_keeps_serial(self):
        composer = mock.Mock()
        ctx = TuneParallelContext.for_collection(composer, block=torch.nn.Linear(2, 2), fp_inputs={"a": [1]})
        assert ctx.devices is None

    def test_collect_forward_serial_when_devices_none(self):
        ctx = TuneParallelContext()
        bf = mock.Mock(return_value="out")
        assert ctx.collect_forward(bf, "blk", "inp", "oth", out_dev="cpu") == "out"
        bf.assert_called_once_with("blk", "inp", "oth", cache_device="cpu")

    def test_collect_forward_sharded_when_devices_set(self):
        ctx = TuneParallelContext()
        ctx.devices = [torch.device("cpu"), torch.device("cpu")]
        with mock.patch(
            "auto_round.algorithms.parallel.tune_parallel.sharded_nograd_forward",
            return_value="sharded",
        ) as snf:
            assert ctx.collect_forward("bf", "blk", "inp", "oth") == "sharded"
        snf.assert_called_once_with(
            "bf", "blk", "inp", "oth", None, ctx.devices, merge_stats=True, pool=None, max_devices=0
        )

    def test_collect_forward_hook_pass_cap_from_policy(self):
        ctx = TuneParallelContext()
        ctx.devices = [torch.device("cpu")] * 8
        with mock.patch(
            "auto_round.algorithms.parallel.tune_parallel.sharded_nograd_forward",
            return_value="sharded",
        ) as snf:
            ctx.collect_forward("bf", "blk", "inp", "oth", hook_pass=True)
            # default: no cap (full-width hook passes)
            assert snf.call_args.kwargs["max_devices"] == 0
            # the cap is set on the context (folded from the env into the
            # ParallelPolicy at the entry)
            ctx.collect_forward_cap = 2
            ctx.collect_forward("bf", "blk", "inp", "oth", hook_pass=True)
            assert snf.call_args.kwargs["max_devices"] == 2
            # non-hook passes stay uncapped
            ctx.collect_forward("bf", "blk", "inp", "oth", hook_pass=False)
            assert snf.call_args.kwargs["max_devices"] == 0

    def test_capped_hook_pass_keeps_the_pool(self):
        """The cap limits WHICH pool replicas run, never pool eligibility:
        the full-width pool is passed through with max_devices set."""
        ctx = TuneParallelContext()
        ctx.devices = [torch.device("cpu")] * 8
        ctx.collect_forward_cap = 2
        sentinel = object()
        ctx.pool = sentinel
        blk = torch.nn.Linear(2, 2)  # pool eligibility requires a real Module
        with mock.patch(
            "auto_round.algorithms.parallel.tune_parallel.sharded_nograd_forward",
            return_value="sharded",
        ) as snf:
            assert ctx.collect_forward("bf", blk, [1] * 8, "oth", hook_pass=True) == "sharded"
        assert snf.call_args.kwargs["pool"] is sentinel
        assert snf.call_args.kwargs["max_devices"] == 2

    def test_distribute_pools_noop_without_devices(self):
        ctx = TuneParallelContext()
        pool = [torch.zeros(1), torch.zeros(1)]
        with mock.patch("auto_round.algorithms.parallel.tune_parallel.distribute_pool") as dp:
            ctx.distribute_pools(pool, None)
        dp.assert_not_called()


@pytest.fixture()
def _autoround_log_propagate():
    """Temporarily enable propagation on the ``autoround`` logger so pytest's
    caplog fixture (handler at the root logger) can capture warnings; the
    logger is configured with propagate=False in production."""

    _logger = logging.getLogger("autoround")
    original = _logger.propagate
    _logger.propagate = True
    yield
    _logger.propagate = original


class TestEngagedLaneE2E:
    """End-to-end engaged lane through the REAL quantize_block tune loop.

    A real TuneParallelContext + real ReplicaGroup (deepcopy mirrors on CPU)
    run the full sequence: create -> build_mirror_optimizers -> warmup ->
    loop {next_shards -> run_step -> sync_grads -> mean_loss -> step} ->
    teardown, with a real SignSGD optimizer and real sign exchange.

    Synthetic pools only (no dataset). Serial-vs-dp loss VALUES are not
    compared: disjoint calibration shards change the draws (known, documented
    in the PR). The pinned invariant is the consensus contract instead --
    at teardown every replica holds bit-identical tuned values.
    """

    def _wrapper(self, seed, name):
        import torch as _t

        class _W(_t.nn.Module):
            def __init__(self):
                super().__init__()
                self.orig_layer = type("OL", (), {"bits": 4})()
                v = _t.nn.Parameter(_t.full((2,), float(seed)))
                self.register_parameter("v", v)
                self.params = {"v": v}  # WrapperLinear.params dict contract
                self.name = name

            def forward(self, x):  # x: [1, 2]
                return x * self.params["v"]

        return _W()

    def _block(self):
        import torch as _t

        wrapper_cls = self._wrapper

        class _B(_t.nn.Module):
            def __init__(self):
                super().__init__()
                self.l0 = wrapper_cls(1.0, "l0")
                self.l1 = wrapper_cls(2.0, "l1")

            def forward(self, x):
                return self.l0(x) + self.l1(x)

        return _B()

    def _quantizer(self):
        from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer
        from auto_round.algorithms.quantization.sign_round.sign_sgd import SignSGD

        class _Q(SignRoundQuantizer):
            # plain class attrs shadow the read-only base properties so the
            # harness can inject fakes on an un-initialized instance
            calibration_context = None
            compress_context = None
            model_context = None
            config = None
            block_forward = None
            _config = None

            def __init__(self):
                pass  # bypass config init; only exercise quantize_block plumbing

            def wrapper_block(self, *a, **k):  # block arrives pre-wrapped
                return ["l0", "l1"], []

            def unwrapper_block(self, *a, **k):
                return None

            def _get_loss(self, pred, ref, indices, mse_loss, dev, mask):
                return mse_loss(pred, ref)

            def _maybe_log_low_bit_lr(self, bits):
                return None

        cfg = type(
            "C",
            (),
            {
                "compute_lr": staticmethod(lambda bits: 0.01),
                "compute_minmax_lr": staticmethod(lambda bits: 0.01),
                "is_act_nv_fp": False,
            },
        )()
        q = _Q()
        q.iters = 2
        q.lr = 0.01
        q.minmax_lr = 0.01
        q.momentum = None  # -> sign_exchange=True
        q.optimizer = SignSGD
        q.lr_scheduler = None
        q.enable_minmax_tuning = False
        q.enable_norm_bias_tuning = False
        q.enable_alg_ext = False
        q.enable_lfq = False
        q.not_use_best_mse = True
        q.gradient_accumulate_steps = 1
        q.calibration_context = type("CC", (), {"batch_size": 2})()
        q.compress_context = type(
            "XC",
            (),
            {
                "low_gpu_mem_usage": False,
                "enable_torch_compile": False,
                "clear_memory": staticmethod(lambda: None),
                "cache_device": "cpu",
            },
        )()
        q._config = cfg
        q.config = cfg
        import torch as _t

        q.block_forward = type(
            "BF",
            (),
            {
                "forward": staticmethod(
                    lambda rep, inputs, others, shard, dev: _t.cat([rep(inputs[j]) for j in shard], dim=0)
                )
            },
        )()
        return q

    def _pools(self):
        import torch as _t

        gen = _t.Generator().manual_seed(0)  # identical pools for both lanes
        fp_inputs = [_t.randn(1, 2, generator=gen) for _ in range(4)]
        with _t.no_grad():
            fp_outputs = [(x * 3.0 + 1.0) for x in fp_inputs]
        return fp_inputs, fp_outputs

    def _engage_plan(self, monkeypatch):
        import torch as _t

        import auto_round.algorithms.parallel.tune_parallel as tp
        from auto_round.algorithms.parallel.data_parallel import DDPPlan

        plan = DDPPlan(world=2, devices=[_t.device("cpu"), _t.device("cpu")], shard_size=1)

        def _resolve(quantizer, block, fp_inputs, fp_outputs, home, world=None, log=True):
            return plan

        monkeypatch.setattr(tp, "resolve_tune_ddp_plan_", _resolve)

    def test_engaged_lane_consensus(self, monkeypatch, caplog, _autoround_log_propagate):

        import torch as _t

        import auto_round.algorithms.parallel.data_parallel as dp
        import auto_round.algorithms.quantization.sign_round.quantizer as v1

        # teardown snapshot: capture the consensus state at the last moment
        # the mirrors still exist
        teardown_records = []
        _orig_teardown = dp.ReplicaGroup.teardown

        def _capture(self):
            teardown_records.append(
                [
                    {n: m.params["v"].detach().clone() for n, m in rep.named_modules() if hasattr(m, "params")}
                    for rep in self.replicas
                ]
            )
            _orig_teardown(self)

        monkeypatch.setattr(dp.ReplicaGroup, "teardown", _capture)
        monkeypatch.setattr(v1, "collect_best_params", lambda block, cache_device: {})
        monkeypatch.setattr(v1, "unwrapper_block", lambda block, best_params: None)

        self._engage_plan(monkeypatch)
        block = self._block()
        fp_inputs, fp_outputs = self._pools()
        q = self._quantizer()

        with caplog.at_level(logging.ERROR, logger="auto_round.algorithms.parallel.tune_parallel"):
            q.quantize_block(
                block,
                fp_inputs,
                {},
                fp_outputs,
                None,
                type("BC", (), {"block_index": 0, "block_cnt": 1, "block_name": "b0"})(),
                None,
            )

        # no divergence-guard error fired (every run_step was followed by sync)
        assert not any("without an intervening sync_grads" in r.message for r in caplog.records)
        # the group ran and was torn down exactly once
        assert len(teardown_records) == 1
        replicas = teardown_records[0]
        assert len(replicas) == 2
        # consensus: every replica holds bit-identical tuned values
        for name in ("l0", "l1"):
            assert _t.equal(replicas[0][name], replicas[1][name]), f"{name} diverged across replicas"
        # and the values actually moved from the seeds (optimizer stepped)
        assert not _t.equal(replicas[0]["l0"], _t.full((2,), 1.0))
        assert not _t.equal(replicas[0]["l1"], _t.full((2,), 2.0))

    def test_serial_lane_still_runs_without_engagement(self, monkeypatch, caplog, _autoround_log_propagate):

        import auto_round.algorithms.quantization.sign_round.quantizer as v1

        monkeypatch.setattr(v1, "collect_best_params", lambda block, cache_device: {})
        monkeypatch.setattr(v1, "unwrapper_block", lambda block, best_params: None)
        block = self._block()
        fp_inputs, fp_outputs = self._pools()
        q = self._quantizer()

        # no engagement patch: the real resolver declines on this host ->
        # accel is None -> the serial lane must run unchanged
        with caplog.at_level(logging.ERROR):
            out = q.quantize_block(
                block,
                fp_inputs,
                {},
                fp_outputs,
                None,
                type("BC", (), {"block_index": 0, "block_cnt": 1, "block_name": "b0"})(),
                None,
            )
        assert out == {}
        assert not any("without an intervening sync_grads" in r.message for r in caplog.records)

    def test_block_end_dynamo_reset_helper(self, monkeypatch):
        """The block-boundary helper resets dynamo caches only under compile."""
        import types as _types

        import torch._dynamo as _dynamo_mod

        from auto_round.compressors.orchestrator import _reset_dynamo_caches_

        called = []
        monkeypatch.setattr(_dynamo_mod, "reset", lambda: called.append(1))
        _reset_dynamo_caches_(__import__("types").SimpleNamespace(enable_torch_compile=False))
        assert not called  # compile off: no reset, no import cost
        _reset_dynamo_caches_(__import__("types").SimpleNamespace(enable_torch_compile=True))
        assert called == [1]

    def test_adoption_stat_targets_skip_wrapper_child_names(self):
        """Wrappers that register orig_layer as a child module make
        named_modules yield '<w>.orig_layer' carrying the same stats object;
        the adoption stats distribution must use wrapper-level names only
        (the pool's pre-wrap layer map has no such children) -- regression:
        KeyError('<wrapper>.orig_layer') on a real model whose wrappers
        register ``orig_layer`` as a child module."""
        import torch as _t

        from auto_round.algorithms.parallel.tune_parallel import TuneParallelContext

        class _W(_t.nn.Module):
            def __init__(self, orig):
                super().__init__()
                self.orig_layer = orig  # registered as a CHILD MODULE

            def forward(self, x):
                return self.orig_layer(x)

        orig = _t.nn.Linear(4, 4, bias=False)
        orig.imatrix = _t.ones(4)  # stats live on the orig layer
        block = _t.nn.Module()
        block.linear_attn = _t.nn.Module()
        block.linear_attn.out_proj = _W(orig)  # registers 'linear_attn.out_proj.orig_layer' as a grandchild
        targets = TuneParallelContext._adoption_stat_targets_(block)
        names = [n for n, _m in targets]
        assert "linear_attn.out_proj" in names
        assert "linear_attn.out_proj.orig_layer" not in names
        # the surviving entry reads the stats through the wrapper
        holder = dict(targets)["linear_attn.out_proj"]
        assert holder is orig and hasattr(holder, "imatrix")

    def test_adoption_hook_sync_keeps_rotation_drops_stale(self):
        """The adoption helper re-syncs mirror-layer hooks from the home
        block's CURRENT state: rotation hooks (registered pre-quantization
        and still live on the home layers) survive onto the mirrors; stale
        collection hooks that home already removed are dropped."""
        import torch as _t

        from auto_round.algorithms.parallel.data_parallel import DDPPlan, MirrorPool
        from auto_round.algorithms.parallel.tune_parallel import TuneParallelContext

        class _Lin(_t.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = _t.nn.Parameter(_t.randn(4, 4))

            def forward(self, x):
                return x @ self.weight

        class _B(_t.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = _Lin()
                self.lin.global_name = "m.lin"
                self.lin.to_quantized = True

            def forward(self, x):
                return self.lin(x)

        rot_seen = []

        def _rot_hook(mod, args):
            rot_seen.append(1)
            return (args[0] + 0.0,)

        class _W(_t.nn.Module):
            """Minimal wrapper: nests the layer as orig_layer (hook lives there)."""

            def __init__(self, orig):
                super().__init__()
                self.orig_layer = orig

            def forward(self, x):
                return self.orig_layer(x)

        block = _B()
        block.lin.register_forward_pre_hook(_rot_hook)  # rotation-style hook
        plan = DDPPlan(world=2, devices=[_t.device("cpu", 0), _t.device("cpu", 1)], shard_size=1)
        pool = MirrorPool(block, plan.devices)
        rep1 = pool.reps[1]
        assert rep1 is not block

        def _stale(mod, args):
            return None

        rep1.lin.register_forward_pre_hook(_stale)  # stale collection residue

        # home gets wrapped (as quantize_block would) BEFORE engagement
        block.lin = _W(block.lin)

        wraps = []

        class _Q:
            enable_minmax_tuning = False
            enable_norm_bias_tuning = False
            compress_context = None
            config = None

            def wrapper_block(self, b, *a, **k):
                wraps.append(b)
                b.lin = _W(b.lin)  # real minimal wrap of the mirror's layer
                return ["lin"], []

        q = _Q()
        q.parallel_state = SimpleNamespace(plan=None, pool=pool, pool_used=False)
        group = TuneParallelContext._adopt_or_build_group_(q, block, plan)
        # wrapped each mirror in place, adopted the pool's replica
        assert wraps == [rep1]
        assert group.mirrors == [rep1]
        assert q.parallel_state.pool_used is True
        # mirror hooks now match home EXACTLY: rotation kept, stale dropped
        # (the hook lives on the layer NESTED in the wrapper)
        assert list(rep1.lin.orig_layer._forward_pre_hooks.values()) == [_rot_hook]
        # and the surviving hook actually fires through the wrapper
        x = _t.randn(2, 4)
        n0 = len(rot_seen)
        with _t.no_grad():
            block(x)
            rep1(x)
        assert len(rot_seen) == n0 + 2

    def test_adoption_reuses_pool_mirrors_with_loss_parity(self, monkeypatch, _autoround_log_propagate):
        """engage_ ADOPTS the collection MirrorPool: mirrors are wrapped in
        place (quantizer.wrapper_block per mirror), skipping the second
        deepcopy, and the DP tune runs the identical loss trajectory as the
        deepcopy lane under pinned draws."""
        import torch as _t

        import auto_round.algorithms.parallel.data_parallel as dp
        import auto_round.algorithms.parallel.tune_parallel as tp
        import auto_round.algorithms.quantization.sign_round.quantizer as v1
        from auto_round.algorithms.parallel.data_parallel import DDPPlan, MirrorPool

        monkeypatch.setattr(
            tp, "shard_samplers", lambda ns, world, bpr: [_FixedSampler([[0], [1]]), _FixedSampler([[2], [3]])]
        )
        monkeypatch.setattr(v1, "collect_best_params", lambda block, cache_device: {})
        monkeypatch.setattr(v1, "unwrapper_block", lambda block, best_params: None)

        plan = DDPPlan(world=2, devices=[_t.device("cpu", 0), _t.device("cpu", 1)], shard_size=1)

        def _resolve(quantizer, block, fp_inputs, fp_outputs, home, world=None, log=True):
            return plan

        monkeypatch.setattr(tp, "resolve_tune_ddp_plan_", _resolve)
        block_ctx = type("BC", (), {"block_index": 0, "block_cnt": 1, "block_name": "b0"})()

        loss_records = []

        def _loss_spy(pred, ref, indices, mse_loss, dev, mask):
            val = mse_loss(pred, ref)
            loss_records.append((tuple(indices), val.item()))
            return val

        def _run_dp(with_pool):
            q = self._quantizer()
            q._get_loss = _loss_spy
            # the adoption helper reads these directly; shadow the base
            # properties (config is None on the harness fake)
            q.enable_minmax_tuning = False
            q.enable_norm_bias_tuning = False
            blk = self._block()
            pools = self._pools()
            wraps = []
            _orig_wb = q.wrapper_block

            def _wb(b, *a, **k):
                wraps.append(id(b))
                return _orig_wb(b, *a, **k)

            q.wrapper_block = _wb
            if with_pool:
                q.parallel_state = SimpleNamespace(plan=None, pool=MirrorPool(blk, plan.devices), pool_used=False)
            n = len(loss_records)
            q.quantize_block(blk, pools[0], {}, pools[1], None, block_ctx, None)
            return loss_records[n:], wraps, q

        baseline, wraps_base, _q_base = _run_dp(False)

        # from here on, adoption replaces the deepcopy mirror maker
        def _no_deepcopy(self_, block, dev):
            raise AssertionError("adoption lane must not deepcopy mirrors")

        monkeypatch.setattr(dp.ReplicaGroup, "_make_mirror", _no_deepcopy)
        adopted, wraps_ado, q_ado = _run_dp(True)

        assert len(baseline) == len(adopted) and len(baseline) % 2 == 0
        # the replicas' record ORDER within an iteration races across
        # threads (same as the sibling parity test): group per iteration,
        # compare draw SETS and mean losses
        for it, (b_it, a_it) in enumerate(zip(zip(*[iter(baseline)] * 2), zip(*[iter(adopted)] * 2))):
            assert {idx for idx, _ in b_it} == {idx for idx, _ in a_it}, f"iter {it}: draw sets differ"
            b_mean = sum(v for _, v in b_it) / 2
            a_mean = sum(v for _, v in a_it) / 2
            assert abs(b_mean - a_mean) < 1e-6, f"iter {it}: deepcopy {b_mean} vs adopted {a_mean}"
        # the adoption lane wrapped each mirror in place: one EXTRA wrapper
        # call per non-home replica on top of quantize_block's own home wrap
        assert len(wraps_ado) == len(wraps_base) + 1
        assert q_ado.parallel_state.pool_used is True
        # OWNERSHIP TRANSFER regression pin (review R4-1): adoption must
        # clear the quantizer-side pool ref so the composer drops the ctx
        # pool; the Step-6 cascade then runs on the home block only (the
        # adopted mirrors hold the tune's LAST state; home was unwrapped
        # with BEST)
        assert q_ado.parallel_state.pool is None

    def test_serial_dp_loss_parity_with_controlled_draws(self, monkeypatch, _autoround_log_propagate):
        """Per-iteration serial-vs-dp loss parity when the draws are pinned.

        The dp/serial loss gap seen on real runs comes from DISJOINT random
        draws (each replica samples its own shard). With the draw sequences
        pinned to the same global indices per iteration, the two lanes must
        produce the same per-iteration losses up to float rounding: mean of
        equal-size shard MSE-means == MSE over the concatenated batch, and
        SignSGD consumes signs of the identical mean gradient, so both lanes
        evolve identical parameters and stay on the same loss trajectory.
        """
        import torch as _t

        import auto_round.algorithms.parallel.tune_parallel as tp
        import auto_round.algorithms.quantization.sign_round.quantizer as v1

        loss_records = []  # (indices tuple, mse value) per _get_loss call

        # dp draws: iter1 rep0=[0] rep1=[2]; iter2 rep0=[1] rep1=[3]
        # serial draws (same global indices, same order): iter1 [0,2]; iter2 [1,3]
        monkeypatch.setattr(
            tp, "shard_samplers", lambda ns, world, bpr: [_FixedSampler([[0], [1]]), _FixedSampler([[2], [3]])]
        )
        monkeypatch.setattr(v1, "IndexSampler", lambda nsamples, batch: _FixedSampler([[0, 2], [1, 3]]))

        def _loss_spy(pred, ref, indices, mse_loss, dev, mask):
            val = mse_loss(pred, ref)
            # record the per-ELEMENT mean: the engaged lane sum-reduces shard
            # losses (normalizing later), the serial lane mean-reduces batches;
            # both arms must compare at the same scale
            scale = max(pred.numel(), 1) if getattr(mse_loss, "reduction", "mean") == "sum" else 1
            loss_records.append((tuple(indices), val.item() / scale))
            return val

        monkeypatch.setattr(v1, "collect_best_params", lambda block, cache_device: {})
        monkeypatch.setattr(v1, "unwrapper_block", lambda block, best_params: None)
        block_ctx = type("BC", (), {"block_index": 0, "block_cnt": 1, "block_name": "b0"})()

        # serial lane FIRST: the real resolver declines on this host, the
        # patched IndexSampler pins the global draws to [0,2] then [1,3]
        q_set = self._quantizer()
        q_set._get_loss = _loss_spy
        block_set = self._block()
        pools = self._pools()
        n0 = len(loss_records)
        q_set.quantize_block(block_set, pools[0], {}, pools[1], None, block_ctx, None)
        set_records = loss_records[n0:]
        assert [idx for idx, _ in set_records] == [(0, 2), (1, 3)]

        # dp lane: engage now (resolver patched AFTER the serial run), same
        # pools, same seed values, matching shard draws
        self._engage_plan(monkeypatch)
        q_dp = self._quantizer()
        q_dp._get_loss = _loss_spy
        block_dp = self._block()
        n1 = len(loss_records)
        q_dp.quantize_block(block_dp, pools[0], {}, pools[1], None, block_ctx, None)
        dp_records = loss_records[n1:]
        # warm-up runs first, serially, one call per replica (head-of-shard
        # draws [0] and [2]); then the loop makes two threaded calls per
        # iteration -- the replicas' record ORDER within an iteration races,
        # so group the loop records into per-iteration pairs, compare as sets
        assert len(dp_records) == 6
        dp_iters = [dp_records[2:4], dp_records[4:6]]
        assert {idx for idx, _ in dp_iters[0]} == {(0,), (2,)}
        assert {idx for idx, _ in dp_iters[1]} == {(1,), (3,)}
        dp_iter_losses = [sum(v for _, v in it) / 2 for it in dp_iters]

        for i, ((_, set_val), dp_val) in enumerate(zip(set_records, dp_iter_losses)):
            assert abs(set_val - dp_val) < 1e-6, f"iter {i}: serial {set_val} vs dp {dp_val}"
