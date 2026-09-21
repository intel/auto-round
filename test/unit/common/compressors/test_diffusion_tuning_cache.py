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
from types import SimpleNamespace

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
@pytest.mark.parametrize("prefetched", [None, False, True])
def test_prefetched_batch_preserves_input_structure(monkeypatch, shared, prefetched):
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
    monkeypatch.setattr(cache, "get", lambda indices: batch if prefetched else None)
    prediction, reference = cache.runner.forward_with_reference(
        Block(),
        cache.inputs,
        cache.others,
        cache.outputs,
        [1],
        "cpu",
        tuning_cache=cache if prefetched is not None else None,
    )
    torch.testing.assert_close(prediction, expected)
    torch.testing.assert_close(reference, cache.outputs[1])


@pytest.mark.parametrize(
    "diffusion,budget,low_memory,device,devices,loss_device,expected",
    [
        (True, "auto", True, "cuda:0", ["cuda:0"], "cuda:0", True),
        (True, 1, True, "cuda:0", ["cuda:0"], None, True),
        (True, 0, True, "cuda:0", ["cuda:0"], "cuda:0", False),
        (False, "auto", True, "cuda:0", ["cuda:0"], "cuda:0", False),
        (True, "auto", False, "cuda:0", ["cuda:0"], "cuda:0", False),
        (True, "auto", True, "cpu", ["cpu"], "cpu", False),
        (True, "auto", True, "cuda:0", ["cuda:0", "cuda:1"], "cuda:0", False),
        (True, "auto", True, "cuda:0", ["cuda:0"], "cuda:1", False),
    ],
)
def test_tuning_cache_scope(monkeypatch, diffusion, budget, low_memory, device, devices, loss_device, expected):
    from auto_round.compressors.diffusion import tuning_cache

    monkeypatch.setattr(tuning_cache, "device_manager", SimpleNamespace(device_list=devices))
    model = SimpleNamespace(is_diffusion=diffusion, diffusion_tuning_cache_size=budget)
    compress = SimpleNamespace(low_gpu_mem_usage=low_memory)
    assert DiffusionTuningCache.is_enabled(model, compress, device, loss_device) is expected
