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
