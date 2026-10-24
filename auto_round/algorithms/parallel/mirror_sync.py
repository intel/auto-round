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
"""Resident mirror-pool synchronization for the data-parallel engine.

The collection passes leave a per-block MirrorPool (one resident block copy
per replica device). This module keeps those mirrors consistent with the HOME
block at phase boundaries: device barriers, home search statistics pushed to
the mirrors before the tune ranks wrap them, and the quantized home state pushed to
the mirrors after a zero-shot RTN/OptRTN quantize so the Step-6 cascade can
still run sharded on them."""

import torch

from auto_round.algorithms.parallel.data_parallel import parallel_state
from auto_round.logger import logger
from auto_round.utils import check_to_quantized


def sync_mirrors_to_home(quantizer, block) -> bool:
    """Make the block's resident mirrors exactly-home after a home-only quantize.

    Zero-shot RTN/OptRTN quantize the HOME block through main's own search
    lane (which already spreads the searches over idle devices). The
    collection mirrors still serve the sharded Step-6 cascade, so push the
    quantized home state (weight + result attrs) onto every replica, release
    the home search stats and mark the pool consumed. Returns False (nothing
    done) when no pool aligned with this block and plan is resident.
    """
    from auto_round.algorithms.parallel.data_parallel import MirrorPool

    _st = parallel_state(quantizer)
    plan = getattr(_st, "plan", None) if _st is not None else None
    pool = getattr(_st, "pool", None) if _st is not None else None
    if not (
        plan is not None
        and isinstance(pool, MirrorPool)
        and pool.world == plan.world
        and pool.block is block
        and list(pool.devices) == list(plan.devices)
    ):
        return False
    targets = [(n, m) for n, m in block.named_modules() if hasattr(m, "global_name") and check_to_quantized(m)]
    _t_bar = _sync_pool_devices(pool.devices)
    _sync_quantized_to_mirrors(pool, block, targets)
    for _n, m in targets:
        for _attr in ("imatrix", "imatrix_cnt"):
            if hasattr(m, _attr):
                delattr(m, _attr)
    _st.pool_used = True
    _t_bar += _sync_pool_devices(pool.devices)
    logger.debug("[mirror-sync] quantized home state synced to %d mirrors (bar=%.2fs)", pool.world - 1, _t_bar)
    return True


_POOL_SEARCH_STATS = ("imatrix", "imatrix_cnt", "act_max", "weight_global_scale")


def _sync_pool_devices(devices) -> float:
    """Synchronize every distinct CUDA pool device; returns elapsed seconds.

    Called at phase boundaries: cross-device copies issued in one
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


def _distribute_search_stats(pool, block, targets):
    """Copy the home layers' collected stats onto every mirror.

    The pool mirrors collected their own sample-shard statistics during the
    collection passes; the merged totals live on the home layers (folded by
    ``_merge_mirror_stats``). Tune ranks wrapping the mirrors must see the FULL
    stats, so each mirror layer is overwritten with the home totals first.
    """
    for name, m in targets:
        stats = {a: getattr(m, a) for a in _POOL_SEARCH_STATS if hasattr(m, a)}
        if not stats:
            continue
        for r in range(pool.world):
            rep = pool.reps[r]
            if rep is block:
                continue
            lm = pool.mirror_layer(r, name)
            dev = next(rep.parameters(), None)
            dev = dev.device if dev is not None else torch.device("cpu")
            for a, v in stats.items():
                setattr(lm, a, v.to(dev) if isinstance(v, torch.Tensor) else v)


_POOL_RESULT_ATTRS = ("scale", "zp", "q_scale_thresh", "data_type")


def _sync_quantized_to_mirrors(pool, block, targets):
    """Propagate every target's quantized weight + result attrs home -> mirrors.

    The quantize ran on the HOME block only, so without this sync each mirror
    would still hold the FP block -- and the cascade shards SAMPLES across
    replicas. One weight copy per (target, non-home replica) makes every
    mirror carry the exact home state (the cascade contract). Weight-only iters=0
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
