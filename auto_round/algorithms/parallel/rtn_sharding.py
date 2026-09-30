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
"""Engine-owned sharded execution of the RTN/OptRTN iters=0 searches.

The quantizer supplies per-module closures (the single-module search, the
stacked batch search) and declares its job-eligibility inputs; this module
owns the work-sharding mechanics: the shared job plan, replica round-robin,
mirror placement, stat distribution, phase barriers, and the resident-mirror
pool lanes. Deterministic searches make the sharded results bit-identical to
the serial lane."""

import re

import torch
import torch.nn as nn

from auto_round.algorithms.parallel.data_parallel import parallel_state
from auto_round.logger import logger
from auto_round.utils import check_to_quantized

# model.blk.experts.<i>.<proj> -- same-layer expert projections share shapes
_EXPERT_RE = re.compile(r"^(?P<parent>.*)\.experts\.\d+\.(?P<proj>[^.]+)$")


def _shard_rtn_searches(quantizer, block, disable_opt_rtn=None) -> tuple:
    """Run the per-layer RTN/OptRTN iters=0 searches, sharded across the DDP devices.

    Work-sharding on the engaged plan: layer i's search runs on
    ``plan.devices[i % world]`` (the layer already moves to its tuning device
    in the serial path, so this adds no extra weight traffic); the search
    itself is deterministic given (weight, imatrix), so sharded results are
    bit-identical to serial. Falls back to the serial single-device loop when
    no plan is engaged or the world is 1. Returns (total_seconds, n_layers,
    max_layer_seconds).
    """
    import time as _ptime

    from auto_round.utils import set_module as _set_module

    _st = parallel_state(quantizer)
    plan = _st.plan if _st is not None else None
    targets = [(n, m) for n, m in block.named_modules() if hasattr(m, "global_name") and check_to_quantized(m)]
    if plan is None or getattr(plan, "world", 1) < 2 or len(targets) < 2:
        _tq, _mx = 0.0, 0.0
        for _n, m in targets:
            _t0 = _ptime.perf_counter()
            quantizer._quantize_layer_core(m)
            _d = _ptime.perf_counter() - _t0
            _tq += _d
            _mx = max(_mx, _d)
        return _tq, len(targets), _mx

    world = plan.world
    home = plan.devices[0]

    from auto_round.algorithms.parallel.data_parallel import MirrorPool  # noqa: F401

    pool = _st.pool if _st is not None else None
    # construction devices are the authoritative layout (mirror params live on
    # them by construction); parameter sniffing would misjudge indexed-cpu stubs
    use_pool = (
        isinstance(pool, MirrorPool)
        and pool.world == world
        and pool.block is block
        and list(pool.devices) == list(plan.devices)
    )

    def _one(dev, mod):
        if dev.type == "cuda":
            with torch.cuda.device(dev):
                return quantizer._quantize_layer_core(mod, tuning_device=dev, disable_opt_rtn=disable_opt_rtn)
        return quantizer._quantize_layer_core(mod, tuning_device=dev, disable_opt_rtn=disable_opt_rtn)

    # the shared job plan (expert batches + singles), exactly as the serial
    # lane would split it; the JOBS round-robin across the plan devices -- a
    # batch job's stacked search runs entirely on its designated device, and
    # its per-module remainder (mid-batch OOM) falls back through the single
    # path on the same device. Quantizers without the primitive (non-OptRTN
    # families) keep the all-singles plan.
    name_of = {id(m): n for n, m in targets}
    _tplan0 = _ptime.perf_counter()
    plan_jobs = getattr(quantizer, "_rtn_search_jobs", None)
    if callable(plan_jobs):
        batches, singles = plan_jobs([m for _, m in targets])
    else:
        batches, singles = [], [m for _, m in targets]
    jobs = [("batch", b) for b in batches] + [("single", m) for m in singles]
    # replica r runs job idx (idx % world); only that replica's copy of a
    # module is ever searched, so stats distribution can target it alone
    owner_of = {}
    for _ji, (_kind, _item) in enumerate(jobs):
        for _m in (_item if isinstance(_item, list) else [_item]):
            owner_of[id(_m)] = _ji % world
    _t_plan = _ptime.perf_counter() - _tplan0
    if batches and not use_pool:
        from auto_round.algorithms.quantization.search_dispatch import log_engaged_once

        log_engaged_once("rtn searches sharded across devices")
    results: dict = {}

    model = getattr(quantizer, "model", None)

    def _place(m, q_layer):
        # place one single-lane result home: back into the block and (when the
        # quantizer carries the global model) into the model, matching the
        # serial path's placement
        q_layer = q_layer.to(home)
        _replace_module(block, name_of[id(m)], q_layer)
        if isinstance(model, torch.nn.Module):
            _set_module(model, q_layer.global_name, q_layer)

    def _run(idx):
        import threading as _the

        from auto_round.algorithms.quantization.search_dispatch import (
            _mark_worker_eager,
            swap_wrapper_callables_to_eager,
        )

        # main-thread invocations (the serial compile warm-up jobs) keep
        # compiled construction; worker threads run eager
        if _the.current_thread() is not _the.main_thread():
            _mark_worker_eager()
        r = idx % world
        dev = plan.devices[r]
        kind, item = jobs[idx]
        # worker threads skip lazy compilation: swap compiled wrapper
        # callables to their eager originals for this job, restore after
        _mods = item if isinstance(item, list) else [item]
        _restore_eager = swap_wrapper_callables_to_eager(_mods)
        try:
            _run_body(idx, r, dev, kind, item)
        finally:
            if _restore_eager is not None:
                _restore_eager()

    def _run_body(idx, r, dev, kind, item):
        if use_pool:
            # searches read the already-resident MIRROR weights: zero search
            # traffic (the layer-move path below stays for the no-pool case)
            if kind == "batch":
                m_item = [pool.mirror_layer(r, name_of[id(m)]) for m in item]
            else:
                m_item = pool.mirror_layer(r, name_of[id(item)])

            def _go_pool():
                rest = quantizer._quantize_expert_batch(m_item, dev)
                rest_ids = {id(mm) for mm in rest}
                # EVERY batch module is written back home: in-batch-written
                # mirrors carry the result directly; the remainder first goes
                # through the per-module single path on the same device
                for hm, mm in zip(item, m_item):
                    q = _one(dev, mm) if id(mm) in rest_ids else mm
                    _place_pool_result(pool, r, name_of[id(hm)], q, home, block, model, _set_module)

            def _go_pool_single():
                results[idx] = (r, _one(dev, m_item))

            if kind == "batch":
                if dev.type == "cuda":
                    with torch.cuda.device(dev):
                        _go_pool()
                else:
                    _go_pool()
            else:
                _go_pool_single()
            return

        if kind == "batch":

            def _go():
                for m in quantizer._quantize_expert_batch(item, dev):
                    _place(m, _one(dev, m))  # unwritten remainder: per-module, same device

            if dev.type == "cuda":
                with torch.cuda.device(dev):
                    _go()
            else:
                _go()
        else:
            results[idx] = _one(dev, item)

    from auto_round.algorithms.parallel.data_parallel import run_threaded_spawn

    _t_stats = 0.0
    _t_bar = 0.0
    if use_pool:
        _ts0 = _ptime.perf_counter()
        _distribute_search_stats(pool, block, targets, owner_of)
        _t_stats = _ptime.perf_counter() - _ts0
        _t_bar += _sync_pool_devices(pool.devices)
        from auto_round.algorithms.quantization.search_dispatch import log_engaged_once

        log_engaged_once("rtn searches on resident mirrors")

    # No serial warm-up: worker threads construct wrappers with compile
    # disabled and run swapped-eager callables (nothing compiles in-thread),
    # and the NeUQI Triton/coarse sweeps serialize their own first launches
    # under _sweep_warm_lock.
    _t0 = _ptime.perf_counter()
    run_threaded_spawn([lambda i=i: _run(i) for i in range(len(jobs))])
    _tq = _ptime.perf_counter() - _t0
    if use_pool:
        _t_bar += _sync_pool_devices(pool.devices)
    # one aggregate trace per block, mirroring the wrap-search lane's line:
    # jobs run in parallel (one thread per job, devices round-robin), so the
    # wall is the true elapsed search time; the gated [perf] rtn phases line
    # carries the per-module mean/max roll-up
    _n_mods = sum(len(b) for _k, b in jobs if _k == "batch") + sum(1 for _k, _m in jobs if _k == "single")
    _n_bjobs = sum(1 for _k, _b in jobs if _k == "batch")
    _n_sjobs = len(jobs) - _n_bjobs
    _tp0 = _ptime.perf_counter()
    for idx, (kind, item) in enumerate(jobs):
        if kind == "single":
            if use_pool:
                r, q = results[idx]
                _place_pool_result(pool, r, name_of[id(item)], q, home, block, model, _set_module)
            else:
                _place(item, results[idx])
    _t_sync = 0.0
    if use_pool:
        # every mirror must forward the FULLY quantized block for the cascade
        # (the search quantized each layer on its own replica only)
        _t_bar += _sync_pool_devices(pool.devices)
        _tsy0 = _ptime.perf_counter()
        _sync_quantized_to_mirrors(pool, block, targets)
        # proactive stat release, matching the wrapper lanes (which delete
        # imatrix on the layer they searched): the home copies are dead once
        # the stats were distributed and the results written back
        for name, _m in targets:
            _hl = block.get_submodule(name)
            for _attr in ("imatrix", "imatrix_cnt"):
                if hasattr(_hl, _attr):
                    delattr(_hl, _attr)
        # tell the composer this quantizer consumed the pool (Step 6 may run
        # the cascade on it); unconsumed pools are released before Step 6
        parallel_state(quantizer, create=True).pool_used = True
        _t_sync = _ptime.perf_counter() - _tsy0
    logger.debug(
        "[batched-search] rtn searches: %d modules, %d jobs (%d batch + %d single), "
        "%.2fs wall (plan=%.2fs stats=%.2fs sync=%.2fs bar=%.2fs)",
        _n_mods,
        len(jobs),
        _n_bjobs,
        _n_sjobs,
        _tq,
        _t_plan,
        _t_stats,
        _t_sync,
        _t_bar,
    )
    return _tq, len(targets), _tq / max(len(targets), 1)


_POOL_SEARCH_STATS = ("imatrix", "imatrix_cnt", "act_max", "weight_global_scale")


def _sync_pool_devices(devices) -> float:
    """Synchronize every distinct CUDA pool device; returns elapsed seconds.

    Called at search-phase boundaries: cross-device copies issued in one
    phase (stats distribution, result placement, mirror sync) complete before
    the next phase touches the same tensors, and a CUDA fault surfaces at
    the boundary of the phase that enqueued it instead of at an unrelated
    later call.
    """
    import time as _ptime

    _t0 = _ptime.perf_counter()
    for d in dict.fromkeys(devices):
        if d.type == "cuda":
            torch.cuda.synchronize(d)
    return _ptime.perf_counter() - _t0


def _distribute_search_stats(pool, block, targets, owner_of=None):
    """Copy the home layers' collected stats onto the mirrors that search them.

    The pool mirrors collected their own sample-shard statistics during the
    collection passes; the merged totals live on the home layers (folded by
    ``_merge_mirror_stats``). The searches must see the FULL stats, so each
    searching mirror's layer is overwritten with the home totals before the
    search. With ``owner_of`` (module id -> replica index) each module's
    stats go only to the replica whose job quantizes it; the other mirrors'
    copies are never read and stay stale.
    """
    for name, m in targets:
        stats = {a: getattr(m, a) for a in _POOL_SEARCH_STATS if hasattr(m, a)}
        if not stats:
            continue
        mid = id(m)
        owners = (owner_of[mid],) if owner_of and mid in owner_of else range(pool.world)
        for r in owners:
            rep = pool.reps[r]
            if rep is block:
                continue
            lm = pool.mirror_layer(r, name)
            dev = next(rep.parameters(), None)
            dev = dev.device if dev is not None else torch.device("cpu")
            for a, v in stats.items():
                setattr(lm, a, v.to(dev) if isinstance(v, torch.Tensor) else v)


def _place_pool_result(pool, r, name, q_layer, home, block, model, set_module_fn):
    """Place one mirror-searched quantized layer: home block + global model
    (the canonical copy for export) and the mirror block (quantized for the
    upcoming cascade forwards on the pool; when the mirror's device equals
    the home device the SAME object is shared by both trees -- identical
    values by construction)."""
    try:
        q_dev = next(q_layer.parameters()).device
    except StopIteration:
        q_dev = home
    if q_dev == home:
        home_layer = q_layer  # same device: sharing the object is exact
    else:
        # Module.to() moves in place and returns self, so the mirror must
        # keep ITS object: copy first, then move the copy home
        import copy as _pcopy

        home_layer = _pcopy.deepcopy(q_layer).to(home)
    _replace_module(block, name, home_layer)
    if isinstance(model, torch.nn.Module):
        set_module_fn(model, home_layer.global_name, home_layer)
    rep = pool.reps[r]
    if rep is not block:
        _replace_module(rep, name, q_layer)


_POOL_RESULT_ATTRS = ("scale", "zp", "q_scale_thresh", "data_type")


def _sync_quantized_to_mirrors(pool, block, targets):
    """Propagate every target's quantized weight + result attrs home -> mirrors.

    The sharded search quantizes each layer on ITS OWN mirror only (job i on
    mirror i % world), so without this sync each replica would hold a mixed
    FP/quantized block -- and the cascade shards SAMPLES across replicas. One
    weight copy per (target, non-home replica) makes every mirror carry the
    exact written-back home state (the cascade contract). Weight-only iters=0
    contract: activation-quantization WrapperWALayer targets are outside this
    lane (structural mirror/home parity is not guaranteed for them).
    """
    # per-MIRROR threads: each replica's copies are independent, so a serial
    # loop pays the sum of all mirrors' copies while threading collapses the
    # wall to the slowest single mirror
    from auto_round.algorithms.parallel.data_parallel import run_threaded_spawn

    def _sync_mirror(r):
        rep = pool.reps[r]
        if rep is block:
            return
        _p0 = next(rep.parameters(), None)
        dev = _p0.device if _p0 is not None else torch.device("cpu")

        def _copy_targets():
            for name, _m in targets:
                home_layer = block.get_submodule(name)
                w = home_layer.weight.data
                attrs = {a: getattr(home_layer, a) for a in _POOL_RESULT_ATTRS if hasattr(home_layer, a)}
                lm = pool.mirror_layer(r, name)
                lm.weight.data.copy_(w.to(dev, non_blocking=True))
                for a, v in attrs.items():
                    setattr(lm, a, v.to(dev) if isinstance(v, torch.Tensor) else v)

        if dev.type == "cuda":
            with torch.cuda.device(dev):
                try:
                    _copy_targets()
                finally:
                    torch.cuda.synchronize(dev)
        else:
            _copy_targets()

    run_threaded_spawn([lambda r=r: _sync_mirror(r) for r in range(pool.world)])


def _replace_module(block, dotted_name, new_module):
    """Replace ``block.<dotted_name>`` with ``new_module``."""
    parts = dotted_name.split(".")
    parent = block
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def split_expert_batches(targets: list, enable_neuqi: bool = False):
    """Partition same-shape expert projections into batchable groups.

    Experts of one layer share weight shapes (e.g. 192 x ``[1536, 4096]``
    gate/up/down projections). The per-group search is row-independent, so
    a whole group can be quantized in one call by stacking weights along
    the output dim - same results as per-module calls, one search's worth
    of overhead per group.
    """
    grouped = {}
    singles = []
    for m in targets:
        match = _EXPERT_RE.match(getattr(m, "global_name", "") or "")
        # plain asym has no search (min/max init); nothing to batch --
        # enable_neuqi arrives as a declared input (the NeUQI grid search
        # makes asym batchable)
        asym_plain = not bool(getattr(m, "sym", False)) and not enable_neuqi
        eligible = (
            type(m) is nn.Linear
            and isinstance(getattr(m, "group_size", None), int)
            and getattr(m, "group_size", None) > 0
            and getattr(m, "super_bits", None) is None
            and getattr(m, "data_type", "int") == "int"
            and m.weight.shape[1] % m.group_size == 0  # not row-divisible: keep per-module
            and getattr(m, "act_bits", 16) > 8  # act-quant layers need the per-module wrapper path
            and not asym_plain  # plain asym has no search (min/max init); nothing to batch
        )
        if not eligible:
            singles.append(m)
            continue
        if match is None:
            # dense modules stay per-module: same-shape pairs (gate/up) would
            # stack the block's two largest searches onto one device and
            # straggle while the other devices idle. Large expert groups
            # amortize the per-call overhead; dense pairs stay below that
            # scale.
            singles.append(m)
            continue
        key = (
            match["parent"],
            match["proj"],
            tuple(m.weight.shape),
            m.bits,
            m.group_size,
            bool(getattr(m, "sym", False)),
        )
        grouped.setdefault(key, []).append(m)
    # Split every group into CHUNK-SIZED jobs at the plan level: a
    # monolithic whole-group job executes on ONE device while the others
    # idle. Chunk jobs round-robin over the plan devices like any other
    # job, so the workers balance naturally.
    from auto_round.algorithms.quantization.search_dispatch import _wrap_batch_max_elems

    _budget = _wrap_batch_max_elems()
    batches = []
    for g in grouped.values():
        if len(g) < 2:
            singles.extend(g)  # singleton groups stay per-module; every group completes
            continue
        per_mod = max(g[0].weight.numel(), 1)
        cap = max(1, min(64, _budget // per_mod))
        for start in range(0, len(g), cap):
            batches.append(g[start : start + cap])
    return batches, singles
