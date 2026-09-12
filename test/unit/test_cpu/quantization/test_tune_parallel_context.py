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
from unittest import mock

import pytest
import torch

from auto_round.algorithms.quantization.sign_round.tune_parallel import TuneParallelContext


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

    def test_forgotten_sync_guard_fires_on_second_run_step(self, caplog):
        ctx = _make_tune_ctx(world=2)
        with caplog.at_level(logging.ERROR, logger="auto_round.algorithms.quantization.sign_round.tune_parallel"):
            ctx.run_step(self._step_fn, [[1], [2]])
            ctx.run_step(self._step_fn, [[3], [4]])
        assert any("without an intervening sync_grads" in r.message for r in caplog.records)

    def test_guard_silent_after_sync(self, caplog):
        ctx = _make_tune_ctx(world=2)
        ctx.params_per_replica = [[], []]
        with caplog.at_level(logging.ERROR, logger="auto_round.algorithms.quantization.sign_round.tune_parallel"):
            ctx.run_step(self._step_fn, [[1], [2]])
            ctx.sync_grads(sign_exchange=True)
            ctx.run_step(self._step_fn, [[3], [4]])
        assert not any("without an intervening sync_grads" in r.message for r in caplog.records)
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
            "auto_round.algorithms.quantization.sign_round.tune_parallel.sharded_nograd_forward",
            return_value="sharded",
        ) as snf:
            assert ctx.collect_forward("bf", "blk", "inp", "oth") == "sharded"
        snf.assert_called_once_with("bf", "blk", "inp", "oth", None, ctx.devices, merge_stats=True, max_devices=0)

    def test_collect_forward_hook_pass_caps_at_four(self):
        ctx = TuneParallelContext()
        ctx.devices = [torch.device("cpu")] * 8
        with mock.patch(
            "auto_round.algorithms.quantization.sign_round.tune_parallel.sharded_nograd_forward",
            return_value="sharded",
        ) as snf:
            ctx.collect_forward("bf", "blk", "inp", "oth", hook_pass=True)
        assert snf.call_args.kwargs["max_devices"] == 4

    def test_distribute_pools_noop_without_devices(self):
        ctx = TuneParallelContext()
        pool = [torch.zeros(1), torch.zeros(1)]
        with mock.patch("auto_round.algorithms.quantization.sign_round.tune_parallel.distribute_pool") as dp:
            ctx.distribute_pools(pool, None)
        dp.assert_not_called()


class TestDeferWrapSearches:
    def test_passthrough_of_engine_policy(self):
        with mock.patch(
            "auto_round.algorithms.quantization.sign_round.tune_parallel.pre_wrap_shard_candidate",
            return_value=True,
        ):
            assert TuneParallelContext.defer_wrap_searches() is True


@pytest.fixture()
def _autoround_log_propagate():
    """Temporarily enable propagation on the ``autoround`` logger so pytest's
    caplog fixture (handler at the root logger) can capture warnings; the
    logger is configured with propagate=False in production."""
    import logging

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

        import auto_round.algorithms.quantization.sign_round.tune_parallel as tp
        from auto_round.algorithms.quantization.sign_round.data_parallel import DDPPlan

        plan = DDPPlan(world=2, devices=[_t.device("cpu"), _t.device("cpu")], shard_size=1)

        def _resolve(quantizer, block, fp_inputs, fp_outputs, home, world=None, log=True):
            return plan

        monkeypatch.setattr(tp, "resolve_tune_ddp_plan_", _resolve)

    def test_engaged_lane_consensus(self, monkeypatch, caplog, _autoround_log_propagate):
        import logging

        import torch as _t

        import auto_round.algorithms.quantization.sign_round.data_parallel as dp
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

        with caplog.at_level(logging.ERROR, logger="auto_round.algorithms.quantization.sign_round.tune_parallel"):
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
        import logging

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

        import auto_round.algorithms.quantization.sign_round.quantizer as v1
        import auto_round.algorithms.quantization.sign_round.tune_parallel as tp

        loss_records = []  # (indices tuple, mse value) per _get_loss call

        class _FixedSampler:
            def __init__(self, draws):
                self._draws = list(draws)
                self._i = 0

            def next_batch(self):
                out = self._draws[self._i]
                self._i += 1
                return out

        # dp draws: iter1 rep0=[0] rep1=[2]; iter2 rep0=[1] rep1=[3]
        # serial draws (same global indices, same order): iter1 [0,2]; iter2 [1,3]
        monkeypatch.setattr(
            tp, "shard_samplers", lambda ns, world, bpr: [_FixedSampler([[0], [1]]), _FixedSampler([[2], [3]])]
        )
        monkeypatch.setattr(v1, "IndexSampler", lambda nsamples, batch: _FixedSampler([[0, 2], [1, 3]]))

        def _loss_spy(pred, ref, indices, mse_loss, dev, mask):
            val = mse_loss(pred, ref)
            loss_records.append((tuple(indices), val.item()))
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
