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
"""Thread ranks: a DDP-style wrapper for the per-block tuning loop.

The engine runs the algorithm's UNCHANGED ``quantize_block`` once per rank,
one thread per device, like ``torch.nn.parallel.DistributedDataParallel``
runs an unchanged training loop once per process:

- module: each rank tunes its own replica of the block (rank 0 = the home
  block, the others = resident mirrors);
- dataloader: ``make_index_sampler`` hands each rank a sampler over its
  contiguous shard of the (device-distributed) calibration pool;
- gradients: ``sync_gradients()`` from ``setup_ddp_if_needed_`` averages
  the gradients across the ranks (DDP semantics);
- loss: ``all_reduce_loss`` gives every rank the same global loss, so
  best-iteration selection and early stopping agree on every rank.

The algorithm uses only those three hooks, the same ones the torchrun path
provides. Rank 0's result is the block result; the mirrors are discarded.

The first block of each shape runs each rank's first phase (wrap, optimizer
build and first forward/backward, up to its first collective) serially in
rank order: first-time ``torch.compile`` / triton autotuning is not
thread-safe. Later blocks of the same shape on the same devices find their
kernels compiled and run the ranks concurrently from the start.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import random
import threading
from collections.abc import Callable
from typing import Any

import torch

from auto_round.algorithms.parallel.data_parallel import (
    MirrorPool,
    _param_grad_buffers,
    _quantizer_policy,
    _write_back_grads,
    contiguous_shard_bounds,
    distribute_pool,
    halving_doubling_allreduce,
    parallel_state,
    resolve_tune_ddp_plan_,
)
from auto_round.algorithms.parallel.mirror_sync import (
    _POOL_SEARCH_STATS,
    _distribute_search_stats,
    _sync_pool_devices,
)
from auto_round.logger import logger

_tls = threading.local()
# (block shape, devices) signatures whose rank kernels are compiled already
_WARM_SIGNATURES: set = set()
_WARM_LOCK = threading.Lock()


def block_signature(block: torch.nn.Module, devices) -> tuple:
    """What decides whether a block's rank kernels are compiled already."""
    shapes = tuple((n, tuple(p.shape), p.dtype) for n, p in block.named_parameters())
    return (type(block).__name__, shapes, tuple(str(d) for d in devices))


class RankAborted(RuntimeError):
    """Raised in a rank whose peer failed (the group was aborted)."""


class RankGroup:
    """In-process collective group for one block's thread ranks."""

    def __init__(self, devices: list[torch.device], serial_first_phase: bool = True) -> None:
        self.devices = [torch.device(d) for d in devices]
        self.world = len(self.devices)
        self._barrier = threading.Barrier(self.world)
        self._slots: list[Any] = [None] * self.world
        self._result: Any = None
        self._turn = threading.Condition()
        # without the serial first phase every rank starts at once
        self.concurrent = not serial_first_phase
        self._next_turn = 0 if serial_first_phase else self.world
        self._released = [False] * self.world
        self._broken = False
        # one shuffle seed per rank, drawn in rank order from the run's RNG
        self.seeds = [random.getrandbits(63) for _ in range(self.world)]

    # ── first-phase turnstile ────────────────────────────────────────────
    def wait_turn(self, rank: int) -> None:
        with self._turn:
            self._turn.wait_for(lambda: self._broken or self._next_turn >= rank)
        if self._broken:
            raise RankAborted("rank group aborted before this rank started")

    def release_turn(self, rank: int) -> None:
        with self._turn:
            if not self._released[rank]:
                self._released[rank] = True
                self._next_turn = max(self._next_turn, rank + 1)
                self._turn.notify_all()

    def abort(self) -> None:
        with self._turn:
            self._broken = True
            self._turn.notify_all()
        self._barrier.abort()

    # ── collectives ──────────────────────────────────────────────────────
    def _wait(self) -> None:
        try:
            self._barrier.wait()
        except threading.BrokenBarrierError as e:
            raise RankAborted("a peer rank failed") from e

    def exchange(self, rank: int, value: Any, reduce_fn: Callable[[list], Any]) -> Any:
        """Post ``value``; rank 0 reduces every rank's value; all get the result."""
        self.release_turn(rank)
        self._slots[rank] = value
        self._wait()
        if rank == 0:
            try:
                self._result = reduce_fn(list(self._slots))
            except BaseException:
                self.abort()
                raise
        self._wait()
        return self._result

    def all_reduce_grads(self, rank: int, module: torch.nn.Module) -> None:
        """Average the gradients of ``module``'s parameters across the ranks."""
        params = [p for p in module.parameters() if p.grad is not None]

        def _reduce(per_rank):
            counts = {len(ps) for ps in per_rank}
            if len(counts) != 1:
                raise RuntimeError(f"ranks hold different gradient sets: {[len(ps) for ps in per_rank]}")
            bufs = _param_grad_buffers(per_rank)
            if any(b is None for b in bufs):
                return
            halving_doubling_allreduce(bufs, scale=1.0 / self.world)
            for buf, ps in zip(bufs, per_rank):
                _write_back_grads(buf, ps)
            return

        self.exchange(rank, params, _reduce)

    def all_reduce_mean(self, rank: int, value: float) -> float:
        """Mean of a python scalar across the ranks (identical on every rank)."""
        return self.exchange(rank, float(value), lambda vals: sum(vals) / len(vals))

    def all_reduce_weighted_mean(self, rank: int, value: float, weight: float) -> float:
        """sum(value * weight) / sum(weight) across the ranks (identical on every rank)."""

        def _reduce(pairs):
            total = sum(w for _v, w in pairs)
            return sum(v * w for v, w in pairs) / total if total else sum(v for v, _w in pairs) / len(pairs)

        return self.exchange(rank, (float(value), float(weight)), _reduce)


class RankContext:
    """What one rank thread knows about itself."""

    def __init__(self, group: RankGroup, rank: int) -> None:
        self.group = group
        self.rank = rank

    @property
    def world(self) -> int:
        return self.group.world

    @property
    def device(self) -> torch.device:
        return self.group.devices[self.rank]

    def index_sampler(self, nsamples: int, batch_size: int):
        """Sampler over this rank's contiguous shard of the calibration pool."""
        from auto_round.compressors.utils import IndexSampler

        bounds = contiguous_shard_bounds(nsamples, self.world)
        lo, hi = bounds[self.rank], bounds[self.rank + 1]
        return IndexSampler(
            hi - lo,
            min(batch_size, hi - lo),
            indices=list(range(lo, hi)),
            rng=random.Random(self.group.seeds[self.rank]),
        )

    def sync_gradients(self, module: torch.nn.Module) -> None:
        self.group.all_reduce_grads(self.rank, module)

    def all_reduce_mean(self, value: float) -> float:
        return self.group.all_reduce_mean(self.rank, value)

    def all_reduce_weighted_mean(self, value: float, weight: float) -> float:
        return self.group.all_reduce_weighted_mean(self.rank, value, weight)


def run_rank_sharded_wrap_searches(rank: RankContext, wrappers: list) -> bool:
    """Shard a block's deferred wrap-time searches across the ranks.

    Each rank searches the wrappers it owns (index % world) on its own
    replica, the searched init scales are exchanged, and every rank
    finalizes all its wrappers with them. Returns False (the caller runs
    every search locally) when the ranks are in their serial first phase:
    a collective there would release the compile turnstile early.
    """
    if not rank.group.concurrent:
        return False
    from auto_round.algorithms.quantization import search_dispatch

    world, me = rank.world, rank.rank
    own = [w for i, w in enumerate(wrappers) if i % world == me]
    summary = search_dispatch.SearchSummary()
    if own and not search_dispatch.run_batched_wrap_search(own, summary=summary):
        for w in own:
            with summary.timed(rank.device) as st:
                w._run_deferred_search_now()
                st["singletons"] += 1
                st["modules"] += 1
    summary.finish()
    mine = {i: w.init_scale for i, w in enumerate(wrappers) if i % world == me}

    def _merge(per_rank):
        merged = {}
        for part, _summary in per_rank:
            merged.update(part)
        return merged, [s for _part, s in per_rank]

    # the block's summary line: every rank's share, logged once
    scales, summaries = rank.group.exchange(me, (mine, summary), _merge)
    if me == 0:
        search_dispatch.SearchSummary.merged(summaries).log("[batched-search] wrap search:")
    for i, w in enumerate(wrappers):
        if i % world != me:
            val = scales[i]
            w.finalize_batched_search(val.to(rank.device) if torch.is_tensor(val) else val)
    return True


def current_rank() -> RankContext | None:
    """The calling thread's rank, or None outside a thread-rank run."""
    return getattr(_tls, "ctx", None)


class _NonZeroRankFilter(logging.Filter):
    """Rank threads other than rank 0 log warnings and errors only."""

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = current_rank()
        return ctx is None or ctx.rank == 0 or record.levelno >= logging.WARNING


logger.addFilter(_NonZeroRankFilter())


def rank_quantizer(quantizer, world: int, split_batch: bool):
    """Shallow per-rank copy of the quantizer.

    The copy shares the model and contexts (read-only during the tune) and
    gets a private calibration context whose batch size is the rank's share
    of the global batch (``split``) or the full batch (``per_replica``).
    The loss is staged on the rank's own device.
    """
    rq = copy.copy(quantizer)
    run_ctx = quantizer._BaseAlgorithm__run_ctx
    cal = copy.copy(run_ctx.calibration_context)
    if split_batch:
        cal.batch_size = max(1, int(cal.batch_size) // world)
    rq._BaseAlgorithm__run_ctx = dataclasses.replace(run_ctx, calibration_context=cal)
    rq._loss_device = None
    rq._card_0_in_high_risk = False
    return rq


def _decline(reason: str) -> None:
    logger.info("[tune-ranks] declining: %s; serial path", reason)


def quantize_block_on_ranks(
    quantizer, block, fp_inputs, input_others, fp_outputs, q_inputs, block_ctx, input_ids=None
) -> tuple[bool, Any]:
    """Run ``quantizer.quantize_block`` once per data-parallel rank.

    Returns ``(engaged, rank-0 result)``; ``engaged=False`` means the caller
    runs the serial call. Engagement needs a tuning quantizer (iters > 0),
    an engaged DDP plan, list pools whose size the world divides, and a
    per-rank batch that fits its shard.
    """
    from auto_round.utils.device_manager import device_manager

    if int(getattr(quantizer, "iters", 0) or 0) <= 0:
        return False, None
    enable_q = q_inputs is not None and getattr(quantizer, "enable_quanted_input", True)
    active_inputs = q_inputs if enable_q else fp_inputs
    if not isinstance(active_inputs, list) or not isinstance(fp_outputs, list):
        return False, None

    _st = parallel_state(quantizer, create=True)
    # an engaged plan cached by this block's collection pass is reused:
    # re-resolving would price a second set of mirrors next to the resident
    # pool and warn about VRAM the pool itself occupies. A declined or
    # missing plan resolves here, where its reasons are logged.
    plan = _st.plan
    if plan is None or not plan.enabled:
        _st.plan = None
        plan = resolve_tune_ddp_plan_(quantizer, block, active_inputs, fp_outputs, device_manager.device)
    _st.plan = None  # block-scoped: the next block resolves its own
    if not plan.enabled:
        return False, None
    world = plan.world
    nsamples = len(active_inputs)
    policy = _quantizer_policy(quantizer)
    split = str(getattr(policy, "batch_mode", "split") or "split") != "per_replica"
    batch_size = int(quantizer.calibration_context.batch_size)
    acc = int(getattr(quantizer, "gradient_accumulate_steps", 1) or 1)
    if nsamples % world:
        _decline(f"{nsamples} samples do not split evenly over {world} ranks")
        return False, None
    if split and batch_size % world:
        _decline(f"batch size {batch_size} does not split evenly over {world} ranks")
        return False, None
    local_batch = (batch_size // world if split else batch_size) * acc
    if local_batch > nsamples // world:
        _decline(f"per-rank batch {local_batch} exceeds the per-rank shard {nsamples // world}")
        return False, None

    devices = list(plan.devices)
    # data: each device owns its contiguous shard; the rank samplers read
    # only their own shard (make_index_sampler)
    distribute_pool(active_inputs, devices)
    distribute_pool(fp_outputs, devices)

    # module: rank 0 tunes the home block, the others tune mirrors (the
    # collection pool's resident mirrors when aligned, else fresh copies)
    pool = getattr(_st, "pool", None)
    if not (isinstance(pool, MirrorPool) and pool.block is block and list(pool.devices) == devices):
        pool = MirrorPool(block, devices)
    else:
        targets = [(n, m) for n, m in block.named_modules() if any(hasattr(m, a) for a in _POOL_SEARCH_STATS)]
        if targets:
            _distribute_search_stats(pool, block, targets)
        pool.sync_hooks_from(block)
    _sync_pool_devices(devices)
    # the mirrors end the block in their own tuned state: nothing reuses them
    _st.pool = None

    signature = block_signature(block, devices)
    with _WARM_LOCK:
        warm = signature in _WARM_SIGNATURES
    group = RankGroup(devices, serial_first_phase=not warm)
    rank_qs = [rank_quantizer(quantizer, world, split) for _ in range(world)]
    logger.info(
        "[tune-ranks] engaged: world=%d, batch per rank=%d (%s), first phase %s, devices=%s",
        world,
        rank_qs[0].calibration_context.batch_size,
        "split" if split else "per_replica",
        "concurrent" if warm else "serial (compile)",
        [str(d) for d in devices],
    )

    def _rank_fn(r):
        return lambda: rank_qs[r].quantize_block(
            pool.reps[r], fp_inputs, input_others, fp_outputs, q_inputs, block_ctx, input_ids=input_ids
        )

    try:
        results = run_ranks([_rank_fn(r) for r in range(world)], devices, group)
        with _WARM_LOCK:
            _WARM_SIGNATURES.add(signature)
    finally:
        pool.release()
        _st.plan = None
    return True, results[0]


def run_ranks(rank_fns: list[Callable[[], Any]], devices: list[torch.device], group: RankGroup) -> list[Any]:
    """Run one callable per rank; rank 0 runs in the calling thread.

    Every rank sees its own device as ``device_manager.device`` and as the
    current CUDA device. A failure in any rank aborts the group (peers raise
    RankAborted at their next collective) and the first real error is
    re-raised here.
    """
    from auto_round.algorithms.parallel.tune_parallel import _device_scope
    from auto_round.utils.device_manager import device_manager

    world = len(rank_fns)
    results: list[Any] = [None] * world
    errors: list[BaseException | None] = [None] * world

    def _run(r: int) -> None:
        _tls.ctx = RankContext(group, r)
        try:
            with device_manager.thread_device_scope(devices[r]), _device_scope(devices[r]):
                group.wait_turn(r)
                results[r] = rank_fns[r]()
        except BaseException as e:  # noqa: BLE001 - re-raised by the caller below
            errors[r] = e
            group.abort()
        finally:
            group.release_turn(r)
            _tls.ctx = None

    threads = [threading.Thread(target=_run, args=(r,), name=f"ar-rank-{r}", daemon=True) for r in range(1, world)]
    for t in threads:
        t.start()
    _run(0)
    for t in threads:
        t.join()
    real = [e for e in errors if e is not None and not isinstance(e, RankAborted)]
    if real:
        raise real[0]
    first = next((e for e in errors if e is not None), None)
    if first is not None:
        raise first
    return results
