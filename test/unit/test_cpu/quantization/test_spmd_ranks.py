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
"""Thread ranks (auto_round.algorithms.parallel.spmd): collectives, turnstile,
samplers, failure handling, and the unchanged SignRound loop run per rank
on CPU against the serial run."""

import logging
import threading

import pytest
import torch

from auto_round.algorithms.parallel import spmd

CPU2 = [torch.device("cpu"), torch.device("cpu", 1)]


def _run(fns, devices=None):
    devices = devices or CPU2[: len(fns)]
    return spmd.run_ranks(fns, devices, spmd.RankGroup(devices))


class TestCollectives:
    def test_weighted_mean_reproduces_the_union(self):
        # rank 0: 2 tokens with loss 1.0, rank 1: 6 tokens with loss 3.0
        out = _run(
            [
                lambda: spmd.current_rank().all_reduce_weighted_mean(1.0, 2),
                lambda: spmd.current_rank().all_reduce_weighted_mean(3.0, 6),
            ]
        )
        assert out == [2.5, 2.5]

    def test_mean_is_identical_on_every_rank(self):
        vals = [1.0, 2.0, 4.0, 8.0]
        devs = [torch.device("cpu")] + [torch.device("cpu", i) for i in range(1, 4)]
        out = _run([lambda v=v: spmd.current_rank().all_reduce_mean(v) for v in vals], devs)
        assert out == [3.75] * 4

    def test_grads_average_across_ranks_like_one_batch(self):
        torch.manual_seed(0)
        base = torch.nn.Linear(4, 2)
        xs = [torch.randn(3, 4), torch.randn(3, 4)]
        reps = [base, torch.nn.Linear(4, 2)]
        reps[1].load_state_dict(base.state_dict())

        def _fn(r):
            def _go():
                reps[r](xs[r]).pow(2).mean().backward()
                spmd.current_rank().sync_gradients(reps[r])

            return _go

        _run([_fn(0), _fn(1)])
        ref = torch.nn.Linear(4, 2)
        ref.load_state_dict(base.state_dict())
        ref(torch.cat(xs)).pow(2).mean().backward()
        for p_ref, p0, p1 in zip(ref.parameters(), reps[0].parameters(), reps[1].parameters()):
            assert torch.equal(p0.grad, p1.grad)
            assert torch.allclose(p0.grad, p_ref.grad, atol=1e-6)

    def test_rank0_runs_in_the_calling_thread(self):
        main = threading.current_thread()
        out = _run([lambda: threading.current_thread() is main, lambda: threading.current_thread() is main])
        assert out == [True, False]


class TestTurnstileAndFailures:
    def test_first_phases_run_in_rank_order(self):
        events = []
        lock = threading.Lock()

        def _fn(r):
            def _go():
                with lock:
                    events.append(("start", r))
                spmd.current_rank().all_reduce_mean(r)
                with lock:
                    events.append(("after", r))

            return _go

        devs = [torch.device("cpu")] + [torch.device("cpu", i) for i in range(1, 4)]
        _run([_fn(r) for r in range(4)], devs)
        starts = [r for kind, r in events if kind == "start"]
        assert starts == [0, 1, 2, 3]
        # nobody passes the collective before the last rank started
        assert events.index(("start", 3)) < min(events.index(("after", r)) for r in range(4))

    def test_warm_blocks_start_every_rank_at_once(self):
        started = threading.Event()

        def _rank0():
            # rank 1 must start before rank 0 reaches its first collective
            assert started.wait(timeout=5)
            spmd.current_rank().all_reduce_mean(0.0)

        def _rank1():
            started.set()
            spmd.current_rank().all_reduce_mean(1.0)

        spmd.run_ranks([_rank0, _rank1], CPU2, spmd.RankGroup(CPU2, serial_first_phase=False))

    def test_block_signature_tracks_shapes_and_devices(self):
        a, b = torch.nn.Linear(4, 4), torch.nn.Linear(4, 8)
        assert spmd.block_signature(a, CPU2) == spmd.block_signature(torch.nn.Linear(4, 4), CPU2)
        assert spmd.block_signature(a, CPU2) != spmd.block_signature(b, CPU2)
        assert spmd.block_signature(a, CPU2) != spmd.block_signature(a, CPU2[:1])

    def test_peer_failure_raises_the_real_error_without_hanging(self):
        def _ok():
            spmd.current_rank().all_reduce_mean(1.0)
            spmd.current_rank().all_reduce_mean(2.0)

        def _boom():
            spmd.current_rank().all_reduce_mean(1.0)
            raise ValueError("rank 1 failed")

        with pytest.raises(ValueError, match="rank 1 failed"):
            _run([_ok, _boom])

    def test_failure_before_first_collective_releases_the_next_rank(self):
        def _boom():
            raise KeyError("early")

        with pytest.raises(KeyError):
            _run([_boom, lambda: spmd.current_rank().all_reduce_mean(1.0)])


class TestRankScope:
    def test_device_manager_sees_the_rank_device_only_inside_the_rank(self):
        from auto_round.utils.device_manager import device_manager

        outside = device_manager.device
        seen = _run([lambda: device_manager.device, lambda: (device_manager.device, device_manager.device_list)])
        assert seen[0] == "cpu"
        assert seen[1] == ("cpu:1", ["cpu:1"])
        assert device_manager.device == outside

    def test_index_sampler_draws_from_the_rank_shard(self):
        from auto_round.utils.distributed import make_index_sampler

        def _draws():
            s = make_index_sampler(8, 2)
            return sorted(i for _ in range(2) for i in s.next_batch())

        out = _run([_draws, _draws])
        assert out == [[0, 1, 2, 3], [4, 5, 6, 7]]
        assert make_index_sampler(8, 2).nsamples == 8  # outside ranks: the whole pool

    def test_non_zero_ranks_log_warnings_only(self, caplog):
        from auto_round.logger import logger

        logger.propagate = True
        try:
            with caplog.at_level(logging.INFO, logger="autoround"):

                def _log(r):
                    def _go():
                        logger.info("info from rank %d", r)
                        logger.warning("warning from rank %d", r)

                    return _go

                _run([_log(0), _log(1)])
        finally:
            logger.propagate = False
        msgs = [r.getMessage() for r in caplog.records]
        assert "info from rank 0" in msgs and "warning from rank 0" in msgs
        assert "info from rank 1" not in msgs and "warning from rank 1" in msgs


_DS = [
    "CPU-tier rank parity sample with enough tokens for quantization",
    "Another parity sample with a different length padding the batch",
    "Third sample keeps the calibration pool at four entries",
    "Fourth sample completes the pinned draw schedule",
]


class _Fixed:
    def __init__(self, draws):
        self._draws, self._i = list(draws), 0

    def next_batch(self):
        out = self._draws[self._i % len(self._draws)]
        self._i += 1
        return out


def _pinned_sampler(nsamples, batch_size):
    # serial: batches [0, 1] then [2, 3]; rank r draws the r-th sample of the
    # same batch, so both runs see the same samples per iteration
    rank = spmd.current_rank()
    if rank is None:
        return _Fixed([[0, 1], [2, 3]])
    return _Fixed([[0], [2]]) if rank.rank == 0 else _Fixed([[1], [3]])


def _quantize(model_path, monkeypatch, world, **extra):
    import auto_round.algorithms.parallel.tune_parallel as tp
    import auto_round.algorithms.quantization.sign_round.quantizer as sr
    from auto_round import AutoRound
    from auto_round.algorithms.parallel.data_parallel import DDPPlan, ParallelPolicy

    monkeypatch.setattr(sr, "make_index_sampler", _pinned_sampler)
    engaged = []
    if world > 1:
        # CPU ranks: the resolvers decline non-accelerator homes, so pin the
        # tune plan to (cpu, cpu:1) and keep the collection serial
        monkeypatch.setattr(spmd, "resolve_tune_ddp_plan_", lambda *a, **k: DDPPlan(2, CPU2, 2))
        monkeypatch.setattr(tp, "resolve_tune_ddp_plan_", lambda *a, **k: DDPPlan(1, CPU2[:1], 0))
        orig = spmd.quantize_block_on_ranks

        def _counted(*a, **k):
            res = orig(*a, **k)
            engaged.append((res[0], type(a[0]).__name__))
            return res

        monkeypatch.setattr(spmd, "quantize_block_on_ranks", _counted)
    ar = AutoRound(
        model_path,
        bits=4,
        group_size=32,
        sym=extra.pop("sym", False),
        iters=2,
        nsamples=4,
        seqlen=8,
        batch_size=2,
        dataset=list(_DS),
        device_map="cpu",
        enable_torch_compile=False,
        disable_model_free=True,
        **extra,
    )
    if world > 1:
        ar.compress_context.parallel_policy = ParallelPolicy(world=world)
    from auto_round.logger import logger

    losses = []

    class _LossLines(logging.Handler):
        def emit(self, record):
            msg = record.getMessage()
            if "loss iter" in msg:
                losses.append(msg[msg.index("loss iter") :])

    handler = _LossLines(level=logging.INFO)
    logger.addHandler(handler)
    try:
        model, _ = ar.quantize()
    finally:
        logger.removeHandler(handler)
    weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return weights, engaged, losses


def test_signround_on_two_cpu_ranks_matches_serial(tiny_opt_model_path, monkeypatch):
    """The unchanged SignRound loop, run once per rank with averaged
    gradients, lands on the serial run's quantized weights bit-exactly."""
    serial, _, serial_losses = _quantize(tiny_opt_model_path, monkeypatch, world=1)
    ranks, engaged, rank_losses = _quantize(tiny_opt_model_path, monkeypatch, world=2)
    assert engaged and all(e for e, _ in engaged)
    assert set(serial) == set(ranks)
    diff = [k for k in serial if not torch.equal(serial[k], ranks[k])]
    assert not diff, f"rank run diverged from serial: {diff[:8]}"
    # the padded samples carry a valid-token mask: the global loss (masked
    # sums over valid tokens, weighted by each rank's token count) reports
    # the serial value
    assert serial_losses and rank_losses == serial_losses


def test_signroundv2_on_two_cpu_ranks_matches_serial(tiny_opt_model_path, monkeypatch):
    """SignRoundV2 defers its wrap-time init-scale searches; warm ranks shard
    them (each rank searches its share, the scales are exchanged), and the
    result still matches the serial run bit-exactly."""
    sharded = []
    orig = spmd.run_rank_sharded_wrap_searches

    def _counted(rank, wrappers):
        res = orig(rank, wrappers)
        sharded.append(res)
        return res

    monkeypatch.setattr(spmd, "run_rank_sharded_wrap_searches", _counted)
    serial, _, _ = _quantize(tiny_opt_model_path, monkeypatch, world=1, sym=True, enable_alg_ext=True)
    ranks, engaged, _ = _quantize(tiny_opt_model_path, monkeypatch, world=2, sym=True, enable_alg_ext=True)
    assert engaged and all(e and name == "SignRoundV2Quantizer" for e, name in engaged), engaged
    assert any(sharded), "no block sharded its wrap searches"
    diff = [k for k in serial if not torch.equal(serial[k], ranks[k])]
    assert not diff, f"rank run diverged from serial: {diff[:8]}"


def test_pull_tuning_pool_keeps_the_rank_shards_in_place():
    """A rank must not bulk-pull the shared, device-distributed pool onto its
    own device (every other rank's shard would be copied there)."""
    from auto_round.algorithms.quantization.base import BaseQuantizer

    q = BaseQuantizer.__new__(BaseQuantizer)
    pool = [torch.zeros(1), torch.zeros(1)]
    out = _run([lambda: q.pull_tuning_pool(pool, "cpu", None, 1, "pool"), lambda: None])
    assert out[0] is pool


class _FakeWrapper:
    def __init__(self, idx, searched):
        self.idx, self._searched, self.init_scale, self.final = idx, searched, None, None

    def _run_deferred_search_now(self):
        self._searched.append((spmd.current_rank().rank, self.idx))
        self.init_scale = self.final = torch.tensor(float(self.idx))

    def finalize_batched_search(self, init_scale):
        self.init_scale = self.final = init_scale


def test_wrap_searches_shard_across_warm_ranks(monkeypatch):
    from auto_round.algorithms.quantization import search_dispatch
    from auto_round.logger import logger

    monkeypatch.setattr(search_dispatch, "run_batched_wrap_search", lambda ws, **kw: False)
    monkeypatch.setattr(search_dispatch, "perf_counters_enabled", lambda: True)
    searched = []
    per_rank = [[_FakeWrapper(i, searched) for i in range(5)] for _ in range(2)]

    def _fn(r):
        return lambda: spmd.run_rank_sharded_wrap_searches(spmd.current_rank(), per_rank[r])

    lines = []

    class _Lines(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    handler = _Lines(level=logging.INFO)
    logger.addHandler(handler)
    try:
        out = spmd.run_ranks([_fn(0), _fn(1)], CPU2, spmd.RankGroup(CPU2, serial_first_phase=False))
    finally:
        logger.removeHandler(handler)
    assert out == [True, True]
    # every wrapper searched exactly once, by its owner rank (index % world)
    assert sorted(searched, key=lambda t: t[1]) == [(i % 2, i) for i in range(5)]
    for ws in per_rank:
        assert [float(w.final) for w in ws] == [0.0, 1.0, 2.0, 3.0, 4.0]
    # one summary line for the block, covering every rank's share and device
    summary = [m for m in lines if m.startswith("[batched-search] wrap search:")]
    assert len(summary) == 1, lines
    assert "5 modules, 0 batch + 5 singleton search calls" in summary[0], summary
    assert "'cpu': " in summary[0] and "'cpu:1': " in summary[0], summary


def test_wrap_searches_stay_local_in_the_serial_first_phase():
    out = _run([lambda: spmd.run_rank_sharded_wrap_searches(spmd.current_rank(), []), lambda: None])
    assert out[0] is False


def test_adamround_on_two_cpu_ranks_matches_serial(tiny_opt_model_path, monkeypatch):
    """AdamRound runs the same unchanged loop per rank; its update uses
    gradient magnitudes, so the averaged rank gradients must reproduce the
    serial update."""
    serial, _, _ = _quantize(tiny_opt_model_path, monkeypatch, world=1, enable_adam=True)
    ranks, engaged, _ = _quantize(tiny_opt_model_path, monkeypatch, world=2, enable_adam=True)
    assert engaged and all(e and name == "AdamRoundQuantizer" for e, name in engaged), engaged
    diff = [k for k in serial if not torch.equal(serial[k], ranks[k])]
    assert not diff, f"rank run diverged from serial: {diff[:8]}"
