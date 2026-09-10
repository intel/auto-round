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

"""Opt-in, bounded CUDA staging for diffusion SignRound batches."""

import queue
import random
import threading

import torch
from torch.utils._pytree import tree_flatten, tree_map

from auto_round.logger import logger


def _batch_plan(sampler, iters, batch_size):
    """Forecast the cyclic sampler without advancing it or the global RNG."""
    rng = random.Random()
    rng.setstate(random.getstate())
    indices, position = list(sampler.indices), sampler.index
    for _ in range(iters):
        if position + sampler.batch_size > sampler.nsamples:
            rng.shuffle(indices)
            position = 0
        selected = indices[position : position + sampler.batch_size]
        position += sampler.batch_size
        for start in range(0, len(selected), batch_size):
            yield selected[start : start + batch_size]


def _signature(batch):
    leaves, spec = tree_flatten(batch)
    signature = []
    for value in leaves:
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu" or value.layout != torch.strided:
                return None
            signature.append((value.shape, value.dtype))
        elif value is None or isinstance(value, (str, bool, int, float)):
            signature.append((type(value), value))
        else:
            return None
    return spec, signature


def _nbytes(batch):
    return sum(t.numel() * t.element_size() for t in tree_flatten(batch)[0] if isinstance(t, torch.Tensor))


def _auto_budget_bytes(free_bytes):
    # Keep the warmed allocator pool for activations/workspace, and leave at
    # least half the unreserved memory (and at least 1 GiB) as additional margin.
    return max(0, min(free_bytes // 2, free_bytes - 2**30))


class DiffusionTuningCache:
    """Two reusable host/device slots, with optional in-place best snapshots.

    The budget covers these persistent CUDA buffers only, not the training
    activations/workspace. Pinned host memory is bounded by the two batch slots.
    The caller must close the cache after backward, including on early exit.
    """

    @classmethod
    def create(cls, block, runner, inputs, others, outputs, sampler, iters, budget_gib, device):
        if runner.batch_size != 1 or not isinstance(inputs, dict) or iters <= 0:
            return None
        cache = cls()
        cache.runner, cache.inputs, cache.others, cache.outputs = runner, inputs, others, outputs
        cache.device = torch.device(device)
        cache.plan = iter(_batch_plan(sampler, iters, runner.batch_size))
        # Start the generator now, before the original sampler advances.
        first = next(cache.plan)
        template = cache._select(first)
        cache.signature = _signature(template)
        if cache.signature is None:
            return None
        cache.sources = (
            block.params
            if hasattr(block, "orig_layer")
            else {name: module.params for name, module in block.named_modules() if hasattr(module, "orig_layer")}
        )
        staging_bytes = 2 * _nbytes(template)
        best_bytes = _nbytes(cache.sources)
        if budget_gib == "auto":
            torch.cuda.synchronize(cache.device)
        free_bytes = torch.cuda.mem_get_info(cache.device)[0]
        budget = (
            _auto_budget_bytes(free_bytes) if budget_gib == "auto" else int(min(budget_gib, free_bytes / 2**30) * 2**30)
        )
        if budget_gib == "auto":
            logger.info("Diffusion tuning auto cache budget after warmup: %.2f GiB.", budget / 2**30)
        if staging_bytes > budget:
            logger.info("Diffusion tuning prefetch skipped: two batch buffers exceed the cache budget.")
            return None
        cache.stop = threading.Event()
        cache.free, cache.ready = queue.Queue(), queue.Queue()
        cache.slots, cache.current, cache.thread = [], None, None
        cache.best = None
        cache.stream = torch.cuda.Stream(device=cache.device)
        try:
            for slot_id in range(2):
                host = tree_map(
                    lambda t: torch.empty_like(t, device="cpu", pin_memory=True) if isinstance(t, torch.Tensor) else t,
                    template,
                )
                gpu = tree_map(
                    lambda t: torch.empty_like(t, device=cache.device) if isinstance(t, torch.Tensor) else t, template
                )
                cache.slots.append(dict(host=host, gpu=gpu, ready=torch.cuda.Event(), finished=None))
                cache.free.put(slot_id)
            if staging_bytes + best_bytes <= budget:
                cache.best = tree_map(lambda t: torch.empty_like(t, device=cache.device), cache.sources)
            # Allocations happen on the compute stream; hand them to the copy stream.
            cache.stream.wait_stream(torch.cuda.current_stream(cache.device))
            cache.thread = threading.Thread(target=cache._produce, args=(first,), daemon=True)
            cache.thread.start()
        except torch.OutOfMemoryError:
            cache.close()
            logger.info("Diffusion tuning prefetch skipped: buffer allocation ran out of memory.")
            return None
        logger.info(
            "Diffusion tuning prefetch enabled with %.2f GiB of GPU buffers; best parameters on %s.",
            (staging_bytes + (best_bytes if cache.best is not None else 0)) / 2**30,
            "GPU" if cache.best is not None else "CPU",
        )
        return cache

    def _select(self, indices):
        inputs, others = self.runner.select_batch(self.inputs, self.others, torch.tensor(indices, dtype=torch.long))
        return inputs, others, self.outputs[indices[0]]

    def _produce(self, first):
        try:
            with torch.cuda.device(self.device):
                indices = first
                while not self.stop.is_set():
                    try:
                        slot_id = self.free.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    slot = self.slots[slot_id]
                    if slot["finished"] is not None:
                        slot["finished"].synchronize()
                    batch = self._select(indices)
                    if _signature(batch) != self.signature:
                        self.ready.put(None)
                        return
                    with torch.cuda.stream(self.stream):
                        source = tree_flatten(batch)[0]
                        host = tree_flatten(slot["host"])[0]
                        gpu = tree_flatten(slot["gpu"])[0]
                        for src, pinned, dest in zip(source, host, gpu):
                            if isinstance(src, torch.Tensor):
                                pinned.copy_(src)
                                dest.copy_(pinned, non_blocking=True)
                        slot["ready"].record(self.stream)
                    self.ready.put((list(indices), slot_id))
                    indices = next(self.plan, None)
                    if indices is None:
                        return
        except Exception as error:
            self.ready.put(error)

    def get(self, indices):
        if self.stop.is_set():
            return None
        if self.current is not None:
            slot = self.slots[self.current]
            slot["finished"] = torch.cuda.Event()
            slot["finished"].record(torch.cuda.current_stream(self.device))
            self.free.put(self.current)
            self.current = None
        item = self.ready.get(timeout=60)
        if isinstance(item, Exception):
            raise item
        if item is None or item[0] != list(indices):
            # Dynamic shapes or a custom forward using Python RNG can invalidate
            # the forecast. Consume the actual sampler batch on the original path.
            self.close()
            logger.info("Diffusion tuning prefetch stopped: batch structure or sample order changed.")
            return None
        _, self.current = item
        slot = self.slots[self.current]
        torch.cuda.current_stream(self.device).wait_event(slot["ready"])
        return slot["gpu"]

    def forward(self, block, batch, cache_device):
        inputs, others, _ = batch
        shared = self.runner.shared_cache_keys
        inputs = {
            key: [value] if isinstance(value, torch.Tensor) or key in shared else value for key, value in inputs.items()
        }
        others = {
            key: (
                [value]
                if key != "positional_inputs" and (isinstance(value, (torch.Tensor, list)) or key in shared)
                else value
            )
            for key, value in others.items()
        }
        return self.runner.forward(block, inputs, others, [0], cache_device)

    def collect_best_params(self):
        for dest, source in zip(tree_flatten(self.best)[0], tree_flatten(self.sources)[0]):
            dest.copy_(source.detach())
        return self.best

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        # Both streams can still reference buffers on exceptions/early stopping.
        self.stream.synchronize()
        torch.cuda.current_stream(self.device).synchronize()
        self.slots.clear()
        self.current = None
