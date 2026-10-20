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
"""Block-scoped tuning data access, independent of the optimization algorithm."""

from functools import partial

import torch

from auto_round.algorithms.block_runner import BlockForwardRunner
from auto_round.algorithms.tuning_cache import resolve_tuning_cache_cls
from auto_round.compressors.utils import IndexSampler, snapshot_best_params
from auto_round.utils.device_manager import device_manager


class BlockTuningSession:
    """Own batch access and optional staging for one block's optimization.

    Algorithms use the same interface with or without prefetch. A session must
    enclose forward, backward and optimizer steps so its staging buffers remain
    valid until all consumers finish, including on exceptions and early exit.
    """

    def __init__(
        self,
        block,
        runner,
        inputs,
        others,
        outputs,
        *,
        nsamples,
        global_batch_size,
        iters,
        device,
        loss_device,
        model_context,
        compress_context,
        parallel=None,
        lfq_last_block=False,
        prefetch=True,
    ):
        self.block, self.runner = block, runner
        self.inputs, self.others, self.outputs = inputs, others, outputs
        self.iters = iters
        self.sampler = (
            parallel.index_sampler(nsamples, global_batch_size)
            if parallel is not None
            else IndexSampler(nsamples, global_batch_size)
        )
        self._sample_indices = parallel.sample_indices if parallel is not None else lambda sampler: sampler.next_batch()
        self.device = device
        self.loss_device = loss_device
        self.cache_device = compress_context.cache_device
        # Keep transient predictions on the compute device in low-memory mode;
        # the runner's persistent output-cache policy must remain unchanged.
        self.forward_device = (
            device if compress_context.low_gpu_mem_usage and not str(device).startswith("cpu") else None
        )
        self._cache = None
        self._cache_factory = None
        self._warmup_iterations = 0
        cache_cls, budget = (
            self._resolve_tuning_cache_plan(model_context, compress_context, lfq_last_block) if prefetch else (None, 0)
        )
        if cache_cls is not None:
            self._cache_factory = partial(cache_cls.create, budget_gib=budget, device=device)
            self._warmup_iterations = 1 if budget == "auto" else 0

    def _resolve_tuning_cache_plan(self, model_context, compress_context, lfq_last_block):
        """Keep the upstream diffusion and LLM/VLM staging eligibility policy."""
        if not (
            compress_context.low_gpu_mem_usage
            and str(self.device).startswith("cuda")
            and len(device_manager.device_list) == 1
            and (self.loss_device is None or torch.device(self.loss_device) == torch.device(self.device))
        ):
            return None, 0
        if getattr(model_context, "is_diffusion", False):
            budget = getattr(model_context, "diffusion_tuning_cache_size", 0)
            if budget != "auto" and budget <= 0:
                return None, 0
        else:
            if not (isinstance(self.inputs, list) and self.runner is not None) or lfq_last_block:
                return None, 0
            budget = "auto"
        return resolve_tuning_cache_cls(model_context), budget

    def pull_pools(self, pull_tuning_pool, batch_size):
        """Apply hot-pool placement without moving prefetch references off host."""
        entry_device = str(getattr(self.runner, "device", self.device))
        if isinstance(self.inputs, list):
            self.inputs = pull_tuning_pool(
                self.inputs,
                entry_device,
                self.block,
                batch_size,
                "tune] block input activations",
                charge_activation=True,
            )
        if self.outputs and self.loss_device is not None and self._cache_factory is None:
            self.outputs = pull_tuning_pool(
                self.outputs,
                self.loss_device,
                self.block,
                batch_size,
                "tune] fp reference outputs",
                charge_activation=False,
            )
        return self.inputs, self.outputs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._cache is not None:
            self._cache.close()

    def iterations(self):
        """Yield iterations, initializing staging after warmup.

        The caller finishes the whole optimizer iteration before requesting the
        next item. Auto budgeting therefore sees the same warmed allocator as
        the original synchronous-first-iteration implementation.
        """
        for iteration in range(self.iters):
            if self._cache_factory is not None and iteration == self._warmup_iterations:
                self._cache = self._cache_factory(
                    self.block,
                    self.runner,
                    self.inputs,
                    self.others,
                    self.outputs,
                    self.sampler,
                    self.iters - iteration,
                )
            yield iteration

    def next_batch(self):
        """Return the next sample indices without changing sampling semantics."""
        return self._sample_indices(self.sampler)

    def forward(self, indices):
        """Return prediction and reference using staged or synchronous data."""
        staged = self._cache.get(indices) if self._cache is not None else None
        if staged is None:
            selected = [self.outputs[index] for index in indices]
            gathered = BlockForwardRunner._gather_same_device(
                selected, str(self.loss_device) if self.loss_device is not None else str(selected[0].device)
            )
            reference = torch.cat(gathered, dim=0).to(self.loss_device)
            prediction = self.runner.forward(self.block, self.inputs, self.others, indices, self.forward_device)
        else:
            reference = staged[2]
            prediction = self._cache.forward(self.block, staged, self.forward_device)
        return prediction, reference

    def collect_best_params(self):
        """Snapshot parameters using the active storage policy."""
        if self._cache is not None and self._cache.best is not None:
            return self._cache.collect_best_params()
        return snapshot_best_params(self.block, self.cache_device)
