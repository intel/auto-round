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
"""Sharded AWQ searches under --data_parallel (parallel.search_shard).

The search seam on two CPU "devices" against the serial searches: the
smoothing grid (shard-and-reduce over the cached parent calls) and the
per-layer clip search (map) select exactly what the serial loops select;
searches that cannot shard fall back to the serial loop; the composer
attaches the seam around ``pre_quantize_block`` only.
"""

import copy
import threading
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from auto_round import RTNConfig
from auto_round.algorithms.composer import AlgorithmComposer, BlockContext
from auto_round.algorithms.parallel.search_shard import SearchShardSeam, _replicate
from auto_round.algorithms.parallel.tune_parallel import TuneParallelContext
from auto_round.algorithms.transforms.awq import smooth as awq_smooth
from auto_round.algorithms.transforms.awq.base import AWQTransform
from auto_round.algorithms.transforms.awq.config import AWQConfig
from auto_round.algorithms.transforms.awq.mappings import ResolvedMapping
from auto_round.algorithms.transforms.base import BasePreprocessor

CPU2 = [torch.device("cpu"), torch.device("cpu", 1)]
BLOCK = "model.layers.0"


class SpySeam(SearchShardSeam):
    """Records every reduce/map call and the merged reduce result."""

    def __init__(self, devices=CPU2):
        super().__init__(devices)
        self.reduces = []
        self.maps = []

    def reduce(self, module, fn, items):
        merged = super().reduce(module, fn, items)
        self.reduces.append((module, len(items), merged))
        return merged

    def map(self, fn, items):
        res = super().map(fn, items)
        self.maps.append(len(items))
        return res


# ── the seam ──────────────────────────────────────────────────────────────


class TestSeam:
    def test_reduce_sums_contiguous_slices(self):
        block = nn.Linear(2, 2)
        seen = {}

        def fn(rep, remap, items):
            seen[len(seen)] = (rep, list(items))
            return torch.tensor([float(sum(items)), float(len(items))])

        merged = SearchShardSeam(CPU2).reduce(block, fn, [1, 2, 3, 4, 5])
        assert merged.tolist() == [15.0, 5.0]
        slices = sorted(v[1] for v in seen.values())
        assert slices == [[1, 2, 3], [4, 5]]  # ceil/floor, remainder first

    def test_reduce_home_slot_uses_module_and_others_private_copies(self):
        torch.manual_seed(0)
        block = nn.Sequential(nn.Linear(3, 3), nn.ReLU(), nn.Linear(3, 3))
        orig = [p.detach().clone() for p in block.parameters()]
        reps = {}
        lock = threading.Lock()

        def fn(rep, remap, items):
            r_lin = remap(block[2])
            with lock:
                reps[items[0]] = (rep, r_lin)
            if rep is not block:
                r_lin.weight.data.zero_()  # private: must not reach home
            return torch.zeros(1)

        SearchShardSeam(CPU2).reduce(block, fn, [0, 1])
        assert reps[0][0] is block and reps[0][1] is block[2]
        assert reps[1][0] is not block and reps[1][1] is not block[2]
        for p, o in zip(block.parameters(), orig):
            assert torch.equal(p, o)

    def test_replicas_exist_before_the_home_slot_mutates(self):
        block = nn.Linear(4, 4, bias=False)
        block.weight.data.fill_(1.0)
        sums = {}

        def fn(rep, remap, items):
            if rep is block:
                block.weight.data.zero_()  # the home slot mutates in place at once
            sums[items[0]] = float(rep.weight.sum())
            return torch.zeros(1)

        SearchShardSeam(CPU2).reduce(block, fn, [0, 1])
        block.weight.data.fill_(1.0)
        assert sums[1] == 16.0  # the copy was taken before the mutation

    def test_replicate_keeps_ties_and_owns_storage(self):
        m = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
        m[1].weight = m[0].weight
        m.register_buffer("buf", torch.ones(2))
        r = _replicate(m, CPU2[1])
        assert r[0].weight is r[1].weight
        assert r[0].weight.data_ptr() != m[0].weight.data_ptr()
        assert r.buf.data_ptr() != m.buf.data_ptr()

    def test_reduce_and_map_decline_below_two(self):
        seam = SearchShardSeam(CPU2)
        assert seam.reduce(nn.Linear(1, 1), lambda *a: torch.zeros(1), [0]) is None
        assert seam.map(lambda d, xs: xs, [0]) is None
        single = SearchShardSeam(CPU2[:1])
        assert single.reduce(nn.Linear(1, 1), lambda *a: torch.zeros(1), [0, 1]) is None
        assert single.map(lambda d, xs: xs, [0, 1]) is None

    def test_map_keeps_item_order(self):
        out = SearchShardSeam(CPU2).map(lambda d, xs: [x * 10 for x in xs], [1, 2, 3, 4, 5])
        assert out == [10, 20, 30, 40, 50]

    def test_map_rejects_a_short_result(self):
        with pytest.raises(RuntimeError, match="returned"):
            SearchShardSeam(CPU2).map(lambda d, xs: xs[:1], [1, 2, 3, 4])

    def test_errors_propagate_from_a_worker(self):
        def fn(rep, remap, items):
            if rep is not None and items[0] == 1:
                raise ValueError("boom")
            return torch.zeros(1)

        with pytest.raises(ValueError, match="boom"):
            SearchShardSeam(CPU2).reduce(nn.Linear(1, 1), fn, [0, 1])

    def test_workers_run_without_grad(self):
        modes = []

        def fn(dev, xs):
            modes.append(torch.is_grad_enabled())
            return xs

        SearchShardSeam(CPU2).map(fn, [0, 1])
        assert modes == [False, False]

    def test_workers_tolerate_concurrent_traces(self):
        import torch._dynamo.config as dynamo_config

        flags = []
        SearchShardSeam(CPU2).map(lambda d, xs: flags.append(dynamo_config.error_on_nested_fx_trace) or xs, [0, 1])
        assert flags == [False, False]

    def test_context_seam_needs_two_devices(self):
        ctx = TuneParallelContext()
        assert ctx.search_seam() is None
        ctx.devices = CPU2[:1]
        assert ctx.search_seam() is None
        ctx.devices = list(CPU2)
        seam = ctx.search_seam()
        assert isinstance(seam, SearchShardSeam) and seam.devices == CPU2


# ── AWQ smoothing grid ────────────────────────────────────────────────────


class _Mlp(nn.Module):
    def __init__(self, h=16, i=32):
        super().__init__()
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def forward(self, x, scale=None):
        y = self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
        return y if scale is None else y * scale


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.post_attention_layernorm = nn.LayerNorm(16)
        self.mlp = _Mlp()
        for name, m in self.named_modules():
            m.global_name = f"{BLOCK}.{name}" if name else BLOCK

    def forward(self, x):
        return self.mlp(self.post_attention_layernorm(x), scale=torch.tensor(1.5))


def _setup(n_calls=5, parent_name=f"{BLOCK}.mlp", **cfg):
    torch.manual_seed(0)
    block = _Block()
    kw = dict(bits=4, group_size=8, sym=True, data_type="int", n_grid=8)
    kw.update(cfg)
    transform = AWQTransform(AWQConfig(**kw))
    mapping = ResolvedMapping(
        smooth_name=f"{BLOCK}.post_attention_layernorm",
        smooth_layer=block.post_attention_layernorm,
        balance_names=[f"{BLOCK}.mlp.gate_proj", f"{BLOCK}.mlp.up_proj"],
        balance_layers=[block.mlp.gate_proj, block.mlp.up_proj],
        parent_name=parent_name,
        parent=block.mlp,
    )
    xs = [torch.randn(2, 6, 16) for _ in range(n_calls)]
    transform._parent_args_cache[block.mlp] = [((x,), {"scale": torch.rand(1) + 0.5}) for x in xs]
    x_mean = torch.cat([x.flatten(0, -2) for x in xs]).abs().mean(0)
    return block, transform, mapping, x_mean


def _serial_losses(transform, mapping, x_mean):
    strategy = awq_smooth.AWQSmoothStrategy(transform, mapping)
    strategy.prepare(awq_smooth.smooth_group_from_mapping(mapping), x_mean)
    try:
        return [strategy.score(c) for c in strategy.candidates()]
    finally:
        strategy.clear()


class TestShardedGrid:
    @pytest.mark.parametrize(
        "cfg",
        [
            {"duo_scaling": False},
            {"duo_scaling": True},
            {"duo_scaling": "both"},
            {"duo_scaling": True, "smooth_batch_size": 1},  # bounded replay path
        ],
        ids=["no-duo", "duo", "both", "microbatch"],
    )
    def test_matches_the_serial_search(self, cfg):
        block, transform, mapping, x_mean = _setup(**cfg)
        weights = {n: p.detach().clone() for n, p in block.named_parameters()}

        serial = transform._grid_search_scales(mapping, x_mean, BLOCK)
        serial_losses = _serial_losses(transform, mapping, x_mean)

        seam = SpySeam()
        transform.set_parallel_search(seam)
        sharded = transform._grid_search_scales(mapping, x_mean, BLOCK)
        transform.set_parallel_search(None)

        assert len(seam.reduces) == 1
        _module, n_items, merged = seam.reduces[0]
        assert _module is block.mlp
        assert n_items == (10 if cfg.get("smooth_batch_size") else 5)
        losses = (merged[:-1] / merged[-1]).tolist()
        assert losses == pytest.approx(serial_losses, rel=1e-5)
        assert torch.equal(sharded, serial)
        # the home block is left exactly as found (the home slot restores it)
        for n, p in block.named_parameters():
            assert torch.equal(p, weights[n]), n
        assert transform._search_perf["grid_sharded"] == 1

    def test_parent_outside_the_block_stays_serial(self):
        block, transform, mapping, x_mean = _setup(parent_name="model.layers.1.mlp")
        serial = transform._grid_search_scales(mapping, x_mean, BLOCK)
        seam = SpySeam()
        transform.set_parallel_search(seam)
        assert torch.equal(transform._grid_search_scales(mapping, x_mean, BLOCK), serial)
        assert seam.reduces == []

    def test_balance_layer_outside_the_parent_stays_serial(self):
        block, transform, mapping, x_mean = _setup()
        mapping.balance_layers[1] = nn.Linear(16, 32, bias=False)
        seam = SpySeam()
        transform.set_parallel_search(seam)
        transform._grid_search_scales(mapping, x_mean, BLOCK)
        assert seam.reduces == []

    @pytest.mark.parametrize("n_calls", [0, 1])
    def test_too_few_parent_calls_stay_serial(self, n_calls):
        block, transform, mapping, x_mean = _setup(n_calls=max(n_calls, 1))
        if n_calls == 0:
            transform._parent_args_cache[block.mlp] = []  # weight-space fallback
        serial = transform._grid_search_scales(mapping, x_mean, BLOCK)
        seam = SpySeam()
        transform.set_parallel_search(seam)
        assert torch.equal(transform._grid_search_scales(mapping, x_mean, BLOCK), serial)
        assert seam.reduces == []

    def test_empty_replay_declines_to_the_serial_fallback(self):
        block, transform, mapping, x_mean = _setup()
        seam = SpySeam()
        seam.reduce = lambda module, fn, items: torch.zeros(len(transform._get_grid_search_params()) + 1)
        transform.set_parallel_search(seam)
        best = awq_smooth.sharded_grid_search(transform, mapping, x_mean, BLOCK, seam)
        assert best is awq_smooth.SHARD_DECLINED

    def test_no_finite_candidate_returns_none(self):
        block, transform, mapping, x_mean = _setup()
        n = len(transform._get_grid_search_params())
        seam = SpySeam()
        seam.reduce = lambda module, fn, items: torch.tensor([float("nan")] * n + [4.0])
        assert awq_smooth.sharded_grid_search(transform, mapping, x_mean, BLOCK, seam) is None


# ── AWQ clip search ──────────────────────────────────────────────────────


def _clip_setup(**cfg):
    block, transform, mapping, _x_mean = _setup(apply_clip=True, clip_n_grid=4, clip_n_sample_token=64, **cfg)
    down = ResolvedMapping(
        smooth_name=f"{BLOCK}.mlp.up_proj",
        smooth_layer=block.mlp.up_proj,
        balance_names=[f"{BLOCK}.mlp.down_proj"],
        balance_layers=[block.mlp.down_proj],
        parent_name=f"{BLOCK}.mlp.down_proj",
        parent=block.mlp.down_proj,
    )
    transform._clip_input_feat[mapping.smooth_name] = torch.randn(48, 16)
    transform._clip_input_feat[down.smooth_name] = torch.randn(48, 32)
    return block, transform, [mapping, down]


class TestShardedClip:
    @pytest.mark.parametrize("clip_as_init", [False, True])
    def test_matches_the_serial_search(self, clip_as_init):
        block, transform, mappings = _clip_setup(clip_as_init=clip_as_init)
        twin = copy.deepcopy(block)
        twin_maps = [
            ResolvedMapping(
                smooth_name=m.smooth_name,
                smooth_layer=twin.get_submodule(m.smooth_layer.global_name[len(BLOCK) + 1 :]),
                balance_names=m.balance_names,
                balance_layers=[twin.get_submodule(n[len(BLOCK) + 1 :]) for n in m.balance_names],
                parent_name=m.parent_name,
                parent=twin.get_submodule(m.parent_name[len(BLOCK) + 1 :]),
            )
            for m in mappings
        ]

        transform._clip_block(BLOCK, mappings)
        seam = SpySeam()
        transform.set_parallel_search(seam)
        transform._clip_block(BLOCK, twin_maps)

        assert seam.maps == [3]
        assert transform._search_perf["clip_sharded"] is True
        for m, t in zip(mappings, twin_maps):
            for bl, tbl in zip(m.balance_layers, t.balance_layers):
                assert torch.equal(bl.weight, tbl.weight)
                if clip_as_init:
                    assert torch.equal(bl.awq_clip_min, tbl.awq_clip_min)
                    assert torch.equal(bl.awq_clip_max, tbl.awq_clip_max)

    def test_explicit_weight_runs_on_its_device(self):
        _block, transform, mappings = _clip_setup()
        bl = mappings[0].balance_layers[0]
        feat = transform._clip_input_feat[mappings[0].smooth_name]
        ref = transform._compute_best_clip(bl, feat)
        got = transform._compute_best_clip(bl, feat, weight=bl.weight.detach().clone())
        assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])


# ── composer wiring ──────────────────────────────────────────────────────


class _SeamRecorder(BasePreprocessor):
    def __init__(self, fail=False):
        super().__init__()
        self.seen = []
        self.fail = fail

    def register_fp_input_forward_hooks(self, block):
        return []

    def pre_quantize_block(self, ctx):
        self.seen.append(self._parallel_search)
        if self.fail:
            raise RuntimeError("pre failed")


class _StubCollectionContext:
    devices = None
    pool = None

    def __init__(self, seam):
        self._seam = seam

    def distribute_pools(self, *a):
        return None

    def search_seam(self):
        return self._seam

    def collect_forward(self, block_forward, block, inputs, input_others, out_dev=None, **kw):
        return block_forward(block, inputs, input_others)

    def release_pool(self):
        return None


def _composer_with(pre, seam):
    composer = AlgorithmComposer([RTNConfig()])
    fwd = MagicMock(return_value=[torch.zeros(1, 4)])
    fwd.last_output_dict = None
    fwd.enable_torch_compile = False
    composer.block_forward = fwd
    composer.block_quantizer.quantize_block = MagicMock()
    composer.block_quantizer.enable_quanted_input = False
    composer._get_fp_act_hooks = MagicMock(return_value=([], {"act_max": "max"}))
    composer._get_q_act_hooks = MagicMock(return_value=([], {"act_max": "max"}))
    composer.preprocessors = [pre]
    composer._collection_context = lambda block, fp_inputs: _StubCollectionContext(seam)
    return composer


def _ctx():
    return BlockContext(model=None, block_names=[BLOCK], block_name=BLOCK, block_index=0)


class TestComposerWiring:
    def test_seam_is_attached_for_pre_quantize_only(self):
        seam = object()
        pre = _SeamRecorder()
        _composer_with(pre, seam).compress_block(nn.Linear(4, 4), [torch.zeros(1, 4)], {}, block_ctx=_ctx())
        assert pre.seen == [seam]
        assert pre._parallel_search is None

    def test_seam_is_detached_when_pre_quantize_fails(self):
        pre = _SeamRecorder(fail=True)
        with pytest.raises(RuntimeError, match="pre failed"):
            _composer_with(pre, object()).compress_block(nn.Linear(4, 4), [torch.zeros(1, 4)], {}, block_ctx=_ctx())
        assert pre._parallel_search is None

    def test_serial_lane_attaches_nothing(self):
        pre = _SeamRecorder()
        _composer_with(pre, None).compress_block(nn.Linear(4, 4), [torch.zeros(1, 4)], {}, block_ctx=_ctx())
        assert pre.seen == [None]


# ── sharded Step-1 calibration capture ───────────────────────────────────


class TestShardedCalibration:
    def _fresh(self):
        block, transform, mapping, _ = _setup(apply_clip=True, clip_n_sample_token=1000)
        transform._parent_args_cache.clear()
        transform._block_mappings = {BLOCK: [mapping]}
        return block, transform, mapping

    @staticmethod
    def _forward(module, xs):
        with torch.no_grad():
            for x in xs:
                module(x)

    def test_capture_merges_in_sample_order(self, monkeypatch):
        torch.manual_seed(1)
        xs = [torch.randn(2, 6, 16) for _ in range(4)]

        block, serial, mapping = self._fresh()
        handles = serial._register_awq_hooks(block, block, BLOCK)
        self._forward(block, xs)
        for h in handles:
            h.remove()

        block2, sharded, mapping2 = self._fresh()

        # on CPU both fake devices report "cpu": route the replica by a marker
        def _sink(mod):
            return sharded._shard_bufs[CPU2[1] if getattr(mod, "_shard1", False) else CPU2[0]]

        monkeypatch.setattr(sharded, "_capture_sink", _sink)
        with sharded.sharded_calibration(CPU2):
            handles = sharded._register_awq_hooks(block2, block2, BLOCK)
            # the replica carries the same hook closures (deepcopy keeps functions)
            replica = copy.deepcopy(block2)
            for m in replica.modules():
                m._shard1 = True
            self._forward(replica, xs[2:])  # shard 1 finishes first
            self._forward(block2, xs[:2])
            for h in handles:
                h.remove()

        name = mapping.smooth_name
        s_sum, s_cnt = serial._activation_stats[name]
        p_sum, p_cnt = sharded._activation_stats[name]
        assert p_cnt == s_cnt
        assert torch.equal(p_sum, s_sum)  # per-call sums re-accumulated in serial order
        assert torch.equal(sharded._clip_input_feat[name], serial._clip_input_feat[name])
        s_calls = serial._parent_args_cache[block.mlp]
        p_calls = sharded._parent_args_cache[block2.mlp]
        assert len(p_calls) == len(s_calls) == 4
        for (sa, sk), (pa, pk) in zip(s_calls, p_calls):
            assert torch.equal(sa[0], pa[0])
            assert torch.equal(sk["scale"], pk["scale"])
        assert sharded._shard_bufs is None

    def test_sharded_capture_keeps_a_device_copy(self):
        block, transform, mapping = self._fresh()
        x = torch.randn(2, 6, 16)
        seen = {}

        def _probe(mod, a, k):
            seen.setdefault("in", a[0])

        probe = block.mlp.register_forward_pre_hook(_probe, with_kwargs=True)
        with transform.sharded_calibration(CPU2[:1]):
            handles = transform._register_awq_hooks(block, block, BLOCK)
            self._forward(block, [x])
            for h in handles:
                h.remove()
            captured = transform._shard_bufs[CPU2[0]]["parent"][block.mlp][0][0][0]
        probe.remove()
        assert torch.equal(captured, seen["in"])
        assert captured.data_ptr() != seen["in"].data_ptr()  # a copy, not an alias

    def test_failed_pass_discards_the_buffers(self):
        block, transform, mapping = self._fresh()
        with pytest.raises(RuntimeError, match="boom"), transform.sharded_calibration(CPU2):
            transform._shard_bufs[CPU2[0]]["parent"][block.mlp] = [((torch.ones(1),), {})]
            raise RuntimeError("boom")
        assert transform._shard_bufs is None
        assert block.mlp not in transform._parent_args_cache

    def test_default_preprocessor_stays_serial(self):
        assert _SeamRecorder().sharded_calibration(CPU2) is None


class _CaptureScope(BasePreprocessor):
    def __init__(self, shardable=True):
        super().__init__()
        self.shardable = shardable
        self.events = []

    def sharded_calibration(self, devices):
        if not self.shardable:
            return None
        import contextlib

        @contextlib.contextmanager
        def _scope():
            self.events.append(("enter", list(devices)))
            yield
            self.events.append("exit")

        return _scope()


class _RecordingCtx:
    def __init__(self, devices):
        self.devices = devices
        self.calls = []

    def collect_forward(self, block_forward, block, inputs, input_others, **kw):
        self.calls.append(kw)


class TestStep1Routing:
    def _composer(self, devices):
        composer = AlgorithmComposer([RTNConfig()])
        composer.block_forward = MagicMock()
        composer._coll_ctx = _RecordingCtx(devices)
        return composer

    def test_shardable_pass_uses_one_pass_replicas(self):
        composer = self._composer(list(CPU2))
        pre = _CaptureScope()
        composer._preprocessor_calibration_forward(nn.Linear(1, 1), [torch.zeros(1)] * 4, {}, [pre])
        assert composer._coll_ctx.calls == [{"hook_pass": True, "merge_stats": {}, "fresh_replicas": True}]
        assert pre.events == [("enter", list(CPU2)), "exit"]
        composer.block_forward.assert_not_called()

    @pytest.mark.parametrize(
        "devices,shardable,n",
        [(None, True, 4), ([CPU2[0]], True, 4), (list(CPU2), False, 4), (list(CPU2), True, 1)],
        ids=["serial-lane", "one-device", "not-shardable", "fewer-samples-than-devices"],
    )
    def test_serial_otherwise(self, devices, shardable, n):
        composer = self._composer(devices)
        pre = _CaptureScope(shardable)
        composer._preprocessor_calibration_forward(nn.Linear(1, 1), [torch.zeros(1)] * n, {}, [pre])
        assert pre.events == []
        assert composer._coll_ctx.calls == []
        composer.block_forward.assert_called_once()

    def test_fresh_replicas_leave_the_pool_alone(self, monkeypatch):
        from auto_round.algorithms.parallel import tune_parallel

        seen = {}
        monkeypatch.setattr(tune_parallel, "sharded_nograd_forward", lambda *a, **kw: seen.update(pool=kw.get("pool")))
        ctx = TuneParallelContext()
        ctx.devices = list(CPU2)
        inputs = [torch.zeros(1, 2)] * 4
        ctx.collect_forward(MagicMock(), nn.Linear(2, 2), inputs, {}, merge_stats={}, fresh_replicas=True)
        assert seen["pool"] is None and ctx.pool is None
        ctx.collect_forward(MagicMock(), nn.Linear(2, 2), inputs, {}, merge_stats={})
        assert seen["pool"] is not None and ctx.pool is seen["pool"]
