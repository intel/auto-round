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

from auto_round.compressors.utils import IndexSampler, collect_best_params
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
    ):
        self.block, self.runner = block, runner
        self.inputs, self.others, self.outputs = inputs, others, outputs
        self.iters = iters
        self.sampler = IndexSampler(nsamples, global_batch_size)
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
        # Backend selection belongs to data infrastructure, not to algorithms.
        # Non-diffusion workflows do not even import the optional backend.
        if getattr(model_context, "is_diffusion", False):
            from auto_round.compressors.diffusion.tuning_cache import DiffusionTuningCache

            if DiffusionTuningCache.is_supported(model_context, compress_context, device, loss_device):
                budget = model_context.diffusion_tuning_cache_size
                self._cache_factory = partial(DiffusionTuningCache.create, budget_gib=budget, device=device)
                self._warmup_iterations = 1 if budget == "auto" else 0

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
        return self.sampler.next_batch()

    def forward(self, indices):
        """Return prediction and reference using staged or synchronous data."""
        staged = self._cache.get(indices) if self._cache is not None else None
        if staged is None:
            reference = torch.cat([self.outputs[index] for index in indices], dim=0).to(self.loss_device)
            prediction = self.runner.forward(self.block, self.inputs, self.others, indices, self.forward_device)
        else:
            reference = staged[2]
            prediction = self._cache.forward(self.block, staged, self.forward_device)
        return prediction, reference

    def collect_best_params(self):
        """Snapshot parameters using the active storage policy."""
        if self._cache is not None and self._cache.best is not None:
            return self._cache.collect_best_params()
        return collect_best_params(self.block, self.cache_device)
