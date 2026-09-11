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

import random

import pytest
import torch

from auto_round.algorithms.block_runner import BlockForwardRunner
from auto_round.compressors.diffusion.tuning_cache import DiffusionTuningCache, _batch_plan
from auto_round.compressors.utils import IndexSampler


@pytest.mark.parametrize("global_batch", [1, 2])
def test_prefetch_plan_preserves_sampler_and_rng(global_batch):
    state = random.getstate()
    try:
        sampler = IndexSampler(5, global_batch)
        before = random.getstate()
        predicted = list(_batch_plan(sampler, 8, 1))
        assert random.getstate() == before
        assert sampler.index == 0
        assert predicted == [[i] for _ in range(8) for i in sampler.next_batch()]
    finally:
        random.setstate(state)


@pytest.mark.parametrize("shared", [False, True])
def test_prefetched_batch_preserves_input_structure(shared):
    class Block(torch.nn.Module):
        def forward(self, hidden_states, scale, temb, freqs):
            return hidden_states * scale + temb + freqs[0] + freqs[1]

    cache = DiffusionTuningCache()
    cache.runner = BlockForwardRunner(
        batch_size=1,
        device="cpu",
        cache_device="cpu",
        amp=False,
        is_diffusion=True,
        shared_cache_keys=("freqs",) if shared else (),
    )
    cache.inputs = {"hidden_states": torch.arange(24).reshape(3, 2, 4), "scale": 2.0}
    cache.others = {
        "temb": torch.ones(3, 2, 4),
        "freqs": [[torch.ones(1, 2, 4), torch.ones(1, 2, 4) * 2] for _ in range(3)],
        "positional_inputs": [],
    }
    cache.outputs = [torch.ones(1, 2, 4) * i for i in range(3)]
    batch = cache._select([1])
    expected = cache.runner.forward(Block(), cache.inputs, cache.others, [1], "cpu")
    torch.testing.assert_close(cache.forward(Block(), batch, "cpu"), expected)
    torch.testing.assert_close(batch[2], cache.outputs[1])


@pytest.mark.parametrize("failure", ["budget", "oom"])
def test_resident_cache_stops_growing_and_preserves_existing_samples(failure, monkeypatch):
    cache = DiffusionTuningCache()
    cache.device = "cpu"
    cache.resident, cache.resident_bytes, cache.resident_full = {}, 0, False
    sample = torch.ones(4)
    size = sample.numel() * sample.element_size()
    cache.resident_budget = size * (2 if failure == "oom" else 1)
    first = cache._allocate_resident((0,), sample)
    assert first is not None

    if failure == "oom":

        def allocate(*args, **kwargs):
            raise torch.OutOfMemoryError("resident allocation")

        monkeypatch.setattr(torch, "empty_like", allocate)

    assert cache._allocate_resident((1,), sample) is None
    assert cache.resident_full and cache.resident_bytes == size
    assert list(cache.resident) == [(0,)] and cache.resident[(0,)] is first
