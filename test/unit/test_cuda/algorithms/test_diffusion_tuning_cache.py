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

import copy
import random
import threading

import pytest
import torch

from auto_round.algorithms.block_runner import BlockForwardRunner
from auto_round.compressors.diffusion.tuning_cache import DiffusionTuningCache
from auto_round.compressors.utils import IndexSampler

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")


@pytest.mark.parametrize(
    "capacity,allocation_oom,steps", [(0, False, 9), (1, False, 9), (3, False, 9), (3, True, 9), (3, False, 2)]
)
def test_resident_cache_reuses_samples_with_identical_updates(capacity, allocation_oom, steps, monkeypatch):
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4, device="cuda:0"))
            self.orig_layer = None
            self.params = {"weight": self.weight}

        def forward(self, hidden_states, temb):
            temb.add_(1)  # A block may modify tensor kwargs in place.
            return hidden_states * self.weight + temb

    block = Block()
    original = copy.deepcopy(block)
    runner = BlockForwardRunner(batch_size=1, device="cuda:0", cache_device="cpu", amp=False, is_diffusion=True)
    inputs = {"hidden_states": torch.arange(24, dtype=torch.float32).reshape(3, 2, 4) / 24}
    others = {"temb": torch.arange(24, dtype=torch.float32).reshape(3, 2, 4), "positional_inputs": []}
    outputs = [torch.ones(1, 2, 4) * i for i in range(3)]
    sample_bytes = 3 * outputs[0].numel() * outputs[0].element_size()
    best_bytes = block.weight.numel() * block.weight.element_size()
    budget = ((2 + capacity) * sample_bytes + best_bytes) / 2**30
    empty_like = torch.empty_like
    allocations = 0

    def allocate(*args, **kwargs):
        nonlocal allocations
        if threading.current_thread() is not threading.main_thread():
            allocations += 1
            if allocation_oom and allocations == 5:
                # Fail partway through the second sample, after retaining one.
                raise torch.OutOfMemoryError("resident allocation")
        return empty_like(*args, **kwargs)

    monkeypatch.setattr(torch, "empty_like", allocate)
    state = random.getstate()
    cache = None
    try:
        sampler = IndexSampler(3, 1)
        cache = DiffusionTuningCache.create(block, runner, inputs, others, outputs, sampler, 9, budget, "cuda:0")
        assert cache is not None
        for _ in range(steps):
            indices = sampler.next_batch()
            batch = cache.get(indices)
            expected = runner.forward(original, inputs, others, indices, "cuda:0")
            actual = cache.forward(block, batch, "cuda:0")
            expected_loss = (expected - outputs[indices[0]].cuda()).square().mean()
            actual_loss = (actual - batch[2]).square().mean()
            expected_loss.backward()
            actual_loss.backward()
            torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
            torch.testing.assert_close(
                batch[0]["hidden_states"].cpu(), inputs["hidden_states"][indices], rtol=0, atol=0
            )
            torch.testing.assert_close(batch[2].cpu(), outputs[indices[0]], rtol=0, atol=0)
            torch.testing.assert_close(block.weight.grad, original.weight.grad, rtol=0, atol=0)
            with torch.no_grad():
                for model in (original, block):
                    model.weight.sub_(model.weight.grad * 0.01)
                    model.zero_grad()
        if steps == 9:
            retained = 1 if allocation_oom else capacity
            assert len(cache.resident) == retained
            assert cache.resident_bytes == retained * sample_bytes
            assert cache.resident_bytes <= cache.resident_budget
            assert cache.hits == retained * 2
            assert cache.loads == 9 - cache.hits
            assert allocations == (5 if allocation_oom else retained * 3)
        else:
            # Unexpected sample order stops prefetch and releases resident entries.
            assert cache.get([-1]) is None
        torch.testing.assert_close(block.weight, original.weight, rtol=0, atol=0)
        torch.testing.assert_close(cache.collect_best_params()["weight"], block.weight, rtol=0, atol=0)
    finally:
        if cache is not None:
            cache.close()
        random.setstate(state)
    assert not cache.thread.is_alive() and not cache.resident and not cache.slots
