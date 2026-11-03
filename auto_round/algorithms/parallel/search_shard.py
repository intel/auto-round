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
"""Sharded no-grad searches for the preprocessors (``pre_quantize_block``).

Preprocessor searches (AWQ's smoothing-ratio grid, its per-layer clip
search) run on the home block before the collection pool exists. The seam
spreads them over the same devices the collection passes use:

- :meth:`SearchShardSeam.reduce` splits a list of items (cached calibration
  calls) into contiguous ceil/floor slices, runs ``fn`` once per device on a
  private copy of a module, and sums the per-device partial vectors on the
  home device. Additive statistics (loss sums plus an element count) merge
  exactly, so the result does not depend on the split.
- :meth:`SearchShardSeam.map` runs independent per-item searches (one result
  per item) on the devices and returns the results in item order.

Both return ``None`` when fewer than two devices or items remain; the caller
then runs its serial loop. The home block is never written by another device:
the reduce copies are private and die with the call, and ``map`` callers only
read home state. The device threads run under ``torch.no_grad()`` (grad mode is
thread-local) with the device pinned, the same scopes the tune ranks use.
"""

from __future__ import annotations

import copy
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import torch

from auto_round.algorithms.parallel.data_parallel import (
    _block_device,
    _relocate_params,
    contiguous_shard_bounds,
    tolerate_concurrent_traces,
)
from auto_round.logger import logger


def _run_on_devices(fns: Sequence[Callable[[], Any]], devices: Sequence[torch.device]) -> list[Any]:
    """Run one callable per device; slot 0 runs in the calling thread.

    Every callable sees its device as ``device_manager.device`` and as the
    current CUDA device and runs under ``torch.no_grad()``; compiled calls
    stay legal while a peer compiles (``tolerate_concurrent_traces``). The first error
    (by slot) is re-raised after all threads joined.
    """
    from auto_round.algorithms.parallel.tune_parallel import _device_scope
    from auto_round.utils.device_manager import device_manager

    n = len(fns)
    results: list[Any] = [None] * n
    errors: list[BaseException | None] = [None] * n

    def _run(i: int) -> None:
        try:
            with (
                device_manager.thread_device_scope(devices[i]),
                _device_scope(devices[i]),
                torch.no_grad(),
                tolerate_concurrent_traces(),
            ):
                results[i] = fns[i]()
        except BaseException as e:  # noqa: BLE001 - re-raised below
            errors[i] = e

    threads = [threading.Thread(target=_run, args=(i,), name=f"ar-search-{i}", daemon=True) for i in range(1, n)]
    for t in threads:
        t.start()
    _run(0)
    for t in threads:
        t.join()
    first = next((e for e in errors if e is not None), None)
    if first is not None:
        raise first
    return results


def _replicate(module: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Private copy of ``module`` on ``device``.

    Parameters and buffers are copied straight to ``device`` (pre-seeded in
    the deepcopy memo), so the source device never holds a second copy of
    the weights -- the copies of several devices may be built concurrently
    without growing the home device's peak. Tied tensors stay tied. Whatever
    ``.to()`` cannot see (plain tensor attributes, device fields) is
    relocated afterwards.
    """
    memo: dict[int, Any] = {}
    for p in module.parameters():
        if id(p) not in memo:
            memo[id(p)] = torch.nn.Parameter(p.detach().to(device, copy=True), requires_grad=p.requires_grad)
    for b in module.buffers():
        if id(b) not in memo:
            memo[id(b)] = b.detach().to(device, copy=True)
    replica = copy.deepcopy(module, memo)
    _relocate_params(replica, device)
    return replica.to(device)


class SearchShardSeam:
    """Shard-and-reduce / map for no-grad searches over ``devices``."""

    def __init__(self, devices: Sequence[torch.device]) -> None:
        self.devices = [torch.device(d) for d in devices]
        # wall of the last reduce's replica build (the [perf] awq grid split line)
        self.last_build_s = 0.0

    def _world(self, n_items: int) -> int:
        world = min(len(self.devices), n_items)
        return world if world >= 2 else 0

    def reduce(
        self,
        module: torch.nn.Module,
        fn: Callable[[torch.nn.Module, Callable[[torch.nn.Module], torch.nn.Module], list], torch.Tensor],
        items: Sequence,
    ) -> torch.Tensor | None:
        """Sum ``fn``'s per-device partials over contiguous item slices.

        ``fn(replica, remap, items_slice)`` returns a 1-D tensor of the same
        length on every device. ``replica`` is ``module`` itself on the
        module's own device (``fn`` must restore any state it mutates) and a
        private relocated deepcopy on every other device; ``remap(m)`` maps a
        submodule of ``module`` to its counterpart in the replica (a module
        outside ``module`` raises ``KeyError``: the caller must gate on
        containment). The partials are summed on the home device in device
        order. Returns ``None`` when the items cannot shard.
        """
        items = list(items)
        world = self._world(len(items))
        if not world:
            return None
        devices = self.devices[:world]
        home_dev = _block_device(module)
        names = {id(m): n for n, m in module.named_modules()}
        bounds = contiguous_shard_bounds(len(items), world)
        home_slot = next((r for r, d in enumerate(devices) if d == home_dev), -1)

        # phase 1: every private copy exists before any evaluation starts --
        # the home slot evaluates on the module itself and may mutate it
        # in place, which must not race with a copy still reading it
        _t_build = time.perf_counter()
        replicas = _run_on_devices(
            [
                (lambda: module) if r == home_slot else (lambda d=d: _replicate(module, d))
                for r, d in enumerate(devices)
            ],
            devices,
        )
        self.last_build_s = time.perf_counter() - _t_build

        def _job(r: int):
            def _go():
                lookup = dict(replicas[r].named_modules())

                def remap(m: torch.nn.Module) -> torch.nn.Module:
                    return lookup[names[id(m)]]

                return fn(replicas[r], remap, items[bounds[r] : bounds[r + 1]])

            return _go

        # phase 2: the evaluations; the copies die with the call
        try:
            partials = _run_on_devices([_job(r) for r in range(world)], devices)
        finally:
            replicas.clear()
        merged = partials[0].to(home_dev)
        for part in partials[1:]:
            merged = merged + part.to(home_dev)
        logger.debug("[search-shard] reduce: world=%d items=%d", world, len(items))
        return merged

    def map(self, fn: Callable[[torch.device, list], list], items: Sequence) -> list | None:
        """Run ``fn(device, items_slice)`` per device; results in item order.

        ``fn`` returns one result per item of its slice and must only read
        shared (home) state. Returns ``None`` when the items cannot shard.
        """
        items = list(items)
        world = self._world(len(items))
        if not world:
            return None
        devices = self.devices[:world]
        bounds = contiguous_shard_bounds(len(items), world)
        parts = _run_on_devices(
            [lambda r=r, d=d: fn(d, items[bounds[r] : bounds[r + 1]]) for r, d in enumerate(devices)],
            devices,
        )
        for r, part in enumerate(parts):
            if len(part) != bounds[r + 1] - bounds[r]:
                raise RuntimeError(
                    f"search map on {devices[r]} returned {len(part)} results for {bounds[r + 1] - bounds[r]} items"
                )
        logger.debug("[search-shard] map: world=%d items=%d", world, len(items))
        return [res for part in parts for res in part]
