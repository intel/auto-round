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
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer
from auto_round.algorithms.tuning import BlockTuningSession
from auto_round.compressors.diffusion.tuning_cache import DiffusionTuningCache
from auto_round.compressors.utils import IndexSampler, collect_best_params
from auto_round.utils.device_manager import device_manager

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")


class TuningBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5, device="cuda:0"))
        self.orig_layer = torch.nn.Identity()
        self.params = {"scale": self.scale}

    def forward(self, hidden_states):
        return hidden_states * self.scale


@pytest.fixture
def tuning_args(monkeypatch):
    monkeypatch.setattr(
        "auto_round.compressors.diffusion.tuning_cache.device_manager", SimpleNamespace(device_list=["cuda:0"])
    )
    state = random.getstate()
    yield dict(
        runner=BlockForwardRunner(
            batch_size=1, device="cuda:0", cache_device="cpu", amp=False, is_diffusion=True, enable_torch_compile=False
        ),
        inputs=[torch.full((1, 2, 4), float(i + 1)) for i in range(4)],
        others={},
        outputs=[torch.full((1, 2, 4), float(i + 1) * 0.8) for i in range(4)],
        nsamples=4,
        global_batch_size=2,
        iters=5,
        device="cuda:0",
        loss_device="cuda:0",
        model_context=SimpleNamespace(is_diffusion=True),
        compress_context=SimpleNamespace(cache_device="cpu", low_gpu_mem_usage=True),
    )
    random.setstate(state)


@pytest.mark.parametrize("budget", [0, 1e-12, 0.01, "auto"])
def test_real_staging_matches_legacy_loss_gradients_and_sampling(tuning_args, monkeypatch, budget):
    caches = []
    create = DiffusionTuningCache.create

    def capture(*args, **kwargs):
        cache = create(*args, **kwargs)
        caches.append(cache)
        return cache

    monkeypatch.setattr(DiffusionTuningCache, "create", capture)
    runner = tuning_args["runner"]

    def train(use_session):
        random.seed(123)
        block = TuningBlock()
        optimizer = torch.optim.SGD(block.parameters(), lr=0.001)
        history = []
        best = None

        def step(indices, prediction, reference):
            loss = (prediction - reference).square().mean()
            loss.backward()
            history.append((list(indices), loss.detach().cpu(), block.scale.grad.detach().cpu().clone()))

        if use_session:
            tuning_args["model_context"].diffusion_tuning_cache_size = budget
            with BlockTuningSession(block=block, **tuning_args) as session:
                for _ in session.iterations():
                    for index in session.next_batch():
                        prediction, reference = session.forward([index])
                        step([index], prediction, reference)
                    best = session.collect_best_params()
                    optimizer.step()
                    optimizer.zero_grad()
        else:
            sampler = IndexSampler(4, 2)
            for _ in range(5):
                for index in sampler.next_batch():
                    reference = tuning_args["outputs"][index].to("cuda:0")
                    prediction = runner.forward(block, tuning_args["inputs"], {}, [index], "cuda:0")
                    step([index], prediction, reference)
                best = collect_best_params(block, "cpu")
                optimizer.step()
                optimizer.zero_grad()
        return history, best["scale"].detach().cpu(), block.scale.detach().cpu()

    expected, actual = train(False), train(True)
    for left, right in zip(expected[0], actual[0]):
        assert left[0] == right[0]
        torch.testing.assert_close(left[1], right[1])
        torch.testing.assert_close(left[2], right[2])
    torch.testing.assert_close(expected[1], actual[1])
    torch.testing.assert_close(expected[2], actual[2])
    if budget in (0.01, "auto"):
        assert len(caches) == 1 and caches[0] is not None
        assert caches[0].best["scale"].is_cuda
        assert not caches[0].thread.is_alive() and not caches[0].slots
    elif budget:
        assert caches == [None]
    else:
        assert not caches


@pytest.mark.parametrize("exit_kind", ["early", "forward", "backward", "optimizer", "producer"])
def test_real_staging_cleans_up_on_early_exit_and_errors(tuning_args, monkeypatch, exit_kind):
    block = TuningBlock()
    tuning_args["model_context"].diffusion_tuning_cache_size = 0.01
    caches = []
    create = DiffusionTuningCache.create
    error = RuntimeError("injected tuning failure")

    def capture(*args, **kwargs):
        cache = create(*args, **kwargs)
        assert cache is not None
        caches.append(cache)
        return cache

    monkeypatch.setattr(DiffusionTuningCache, "create", capture)
    if exit_kind == "forward":

        def fail_forward(*args):
            raise error

        block.register_forward_pre_hook(fail_forward)
    elif exit_kind == "backward":

        def fail_backward(gradient):
            raise error

        block.scale.register_hook(fail_backward)
    elif exit_kind == "producer":
        select = DiffusionTuningCache._select

        def fail_select(self, indices):
            # Keep synchronous template selection intact, fail in the worker.
            if hasattr(self, "thread"):
                raise error
            return select(self, indices)

        monkeypatch.setattr(DiffusionTuningCache, "_select", fail_select)

    def run():
        with BlockTuningSession(block=block, **tuning_args) as session:
            for _ in session.iterations():
                prediction, reference = session.forward([session.next_batch()[0]])
                (prediction - reference).square().mean().backward()
                if exit_kind == "optimizer":
                    raise error
                break

    if exit_kind == "early":
        run()
    else:
        with pytest.raises(RuntimeError, match="injected tuning failure") as caught:
            run()
        assert caught.value is error
    assert len(caches) == 1
    assert not caches[0].thread.is_alive() and not caches[0].slots


@pytest.mark.parametrize("is_diffusion,budget", [(True, 0), (True, 0.01), (True, "auto"), (False, "auto")])
def test_signround_quantize_block_uses_shared_session(monkeypatch, is_diffusion, budget):
    """Exercise real wrappers, optimizer and unwrapping without model downloads."""
    previous_device_map = device_manager.device_map
    random_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    device_manager.configure("cuda:0")
    caches = []
    create = DiffusionTuningCache.create

    def capture(*args, **kwargs):
        cache = create(*args, **kwargs)
        caches.append(cache)
        return cache

    monkeypatch.setattr(DiffusionTuningCache, "create", capture)
    try:
        config = SignRoundConfig(
            iters=3, lr=0.01, minmax_lr=0.01, bits=4, group_size=4, sym=True, data_type="int", act_bits=16
        )
        block = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False)).to("cuda:0")
        layer = block[0]
        for name, value in vars(config.scheme).items():
            setattr(layer, name, value)
        layer.scale_dtype = torch.float32
        layer.global_name = "blocks.0.0"
        inputs = [torch.ones(1, 2, 4) * (i + 1) for i in range(3)]
        runner = BlockForwardRunner(
            batch_size=1,
            device="cuda:0",
            cache_device="cpu",
            amp=False,
            is_diffusion=is_diffusion,
            enable_torch_compile=False,
        )
        with torch.no_grad():
            outputs = runner.forward(block, inputs, {})
        quantizer = SignRoundQuantizer(config)
        quantizer.bind(
            SimpleNamespace(
                model_context=SimpleNamespace(
                    model=block, is_diffusion=is_diffusion, diffusion_tuning_cache_size=budget, amp=False
                ),
                compress_context=SimpleNamespace(
                    cache_device="cpu", low_gpu_mem_usage=True, enable_torch_compile=False, clear_memory=lambda: None
                ),
                calibration_context=SimpleNamespace(batch_size=1, batch_dim=0),
                scheme_context=config.scheme,
            )
        )
        quantizer.bind_block_forward_runner(runner)
        snapshots = quantizer.quantize_block(
            block,
            inputs,
            {},
            outputs,
            None,
            SimpleNamespace(block_index=0, block_cnt=1, block_name="blocks.0"),
        )
        assert snapshots
        assert isinstance(block[0], torch.nn.Linear)
        assert torch.isfinite(block[0].weight).all()
        if is_diffusion and budget:
            assert len(caches) == 1 and caches[0] is not None
            assert not caches[0].thread.is_alive() and not caches[0].slots
        else:
            assert not caches
    finally:
        device_manager.configure(previous_device_map)
        random.setstate(random_state)
        torch.random.set_rng_state(torch_state)
