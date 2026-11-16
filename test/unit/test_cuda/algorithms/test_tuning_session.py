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
from auto_round.algorithms.tuning_cache import BlockTuningPrefetchCache, DiffusionTuningCache
from auto_round.compressors.utils import IndexSampler, snapshot_best_params
from auto_round.schemes import QuantizationScheme
from auto_round.utils.device_manager import device_manager

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")


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
    monkeypatch.setattr("auto_round.algorithms.tuning.device_manager", SimpleNamespace(device_list=["cuda:0"]))
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
@requires_cuda
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
                best = snapshot_best_params(block, "cpu")
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
@requires_cuda
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


@pytest.fixture
def quantize_setup(monkeypatch):
    """Build the same real integration on CPU before any CUDA-only skip."""
    previous_device_map = device_manager.device_map
    random_state = random.getstate()
    torch_state = torch.random.get_rng_state()

    def build(device="cpu", is_diffusion=False, budget="auto", **config_kwargs):
        if device == "cpu":
            monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CPU integration initialized CUDA"))
        device_manager.configure(device)
        config = SignRoundConfig(iters=3, lr=0.01, minmax_lr=0.01, **config_kwargs)
        scheme = QuantizationScheme(bits=4, group_size=4, sym=True, data_type="int", act_bits=16)
        config.finalize_scheme(scheme)
        block = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False)).to(device)
        layer = block[0]
        for name, value in vars(scheme).items():
            setattr(layer, name, value)
        layer.scale_dtype = torch.float32
        layer.global_name = "blocks.0.0"
        inputs = [torch.ones(1, 2, 4) * (i + 1) for i in range(3)]
        runner = BlockForwardRunner(
            batch_size=1,
            device=device,
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
                scheme_context=scheme,
            )
        )
        quantizer.bind_block_forward_runner(runner)
        return quantizer, block, inputs, outputs

    try:
        yield build
    finally:
        device_manager.configure(previous_device_map)
        random.setstate(random_state)
        torch.random.set_rng_state(torch_state)


def quantize(quantizer, block, inputs, outputs):
    return quantizer.quantize_block(
        block, inputs, {}, outputs, None, SimpleNamespace(block_index=0, block_cnt=1, block_name="blocks.0")
    )


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=requires_cuda)])
@pytest.mark.parametrize("is_diffusion,budget", [(True, 0), (True, 0.01), (True, "auto"), (False, "auto")])
def test_signround_quantize_block_uses_shared_session(monkeypatch, quantize_setup, device, is_diffusion, budget):
    """Exercise identical scheme binding, real wrappers and optimizer on CPU and CUDA."""
    quantizer, block, inputs, outputs = quantize_setup(device, is_diffusion, budget)
    caches = []
    sessions = []
    cache_cls = DiffusionTuningCache if is_diffusion else BlockTuningPrefetchCache
    create = cache_cls.create
    factory = quantizer.tuning_session

    def capture(*args, **kwargs):
        cache = create(*args, **kwargs)
        caches.append(cache)
        return cache

    def capture_session(*args, **kwargs):
        session = factory(*args, **kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(cache_cls, "create", capture)
    monkeypatch.setattr(quantizer, "tuning_session", capture_session)
    snapshots = quantize(quantizer, block, inputs, outputs)
    assert snapshots
    assert len(sessions) == 1 and sessions[0]._state == "closed"
    # The composer, not quantize_block(), owns final unwrapping.
    assert hasattr(block[0], "orig_layer")
    assert torch.isfinite(block[0].weight).all()
    if device != "cpu" and (not is_diffusion or budget):
        assert len(caches) == 1 and caches[0] is not None
        assert not caches[0].thread.is_alive() and not caches[0].slots
    else:
        assert not caches


@pytest.mark.parametrize("mode", ["joint", "frozen_ar_refine", "staged_refine"])
@pytest.mark.parametrize("not_use_best_mse", [False, True])
def test_teq_session_schedule_and_fixed_selection(monkeypatch, quantize_setup, mode, not_use_best_mse):
    quantizer, block, inputs, outputs = quantize_setup(not_use_best_mse=not_use_best_mse)
    log_alpha = torch.nn.Parameter(torch.zeros(4))
    block[0]._teq_input_context = {
        "log_alpha": log_alpha,
        "min_log": -10.0,
        "max_log": 10.0,
        "apply_input_transform": True,
        "wrappers": [],
    }
    quantizer.teq_refinement_mode = mode
    quantizer.teq_joint_lr = 0.001
    quantizer.teq_refine_iters = 2
    quantizer.teq_ar_first_iters = 1
    sessions, sampled, forwarded, snapshots, selection = [], [], [], [], []
    factory = quantizer.tuning_session
    runner_forward = quantizer.block_forward.forward

    def capture_forward(*args, **kwargs):
        if sessions and sessions[0]._state == "closed":
            selection.append(list(args[3]))
        return runner_forward(*args, **kwargs)

    monkeypatch.setattr(quantizer.block_forward, "forward", capture_forward)

    def capture_session(*args, **kwargs):
        assert kwargs["iters"] == 6
        assert kwargs["prefetch"] is (mode != "staged_refine")
        session = factory(*args, **kwargs)
        sessions.append(session)
        next_batch, forward, collect = session.next_batch, session.forward, session.collect_best_params

        def sample():
            indices = next_batch()
            sampled.append(list(indices))
            return indices

        def run(indices):
            forwarded.append(list(indices))
            return forward(indices)

        def snapshot():
            snapshots.append(len(forwarded))
            return collect()

        monkeypatch.setattr(session, "next_batch", sample)
        monkeypatch.setattr(session, "forward", run)
        monkeypatch.setattr(session, "collect_best_params", snapshot)
        return session

    monkeypatch.setattr(quantizer, "tuning_session", capture_session)
    result = quantize(quantizer, block, inputs, outputs)
    assert result and sessions[0]._state == "closed"
    assert len(forwarded) == 6
    assert len(sampled) == (3 if mode == "staged_refine" else 6)
    if mode == "staged_refine":
        assert forwarded[1:4] == [[0], [0], [0]]
        assert sessions[0]._cache_factory is None
        if not not_use_best_mse:
            assert "selected_loss" in quantizer.teq_last_refinement
            assert selection == [[0], [0]]
    if mode != "staged_refine" or not_use_best_mse:
        assert not selection
    if not_use_best_mse:
        assert snapshots[-1] == 6


@pytest.mark.parametrize("not_use_best_mse", [False, True])
def test_previous_best_snapshot_released_before_allocating(monkeypatch, quantize_setup, not_use_best_mse):
    import inspect

    quantizer, block, inputs, outputs = quantize_setup(not_use_best_mse=not_use_best_mse)
    collect = BlockTuningSession.collect_best_params
    snapshots = []

    def check_release(session):
        caller = inspect.currentframe().f_back
        assert caller.f_locals["best_params"] is None
        del caller
        snapshots.append(collect(session))
        return snapshots[-1]

    monkeypatch.setattr(BlockTuningSession, "collect_best_params", check_release)
    assert quantize(quantizer, block, inputs, outputs)
    assert snapshots


@pytest.mark.parametrize("exit_kind", ["normal", "forward", "backward", "optimizer", "pull"])
def test_quantize_block_closes_staging_before_lfq_downstream(monkeypatch, quantize_setup, exit_kind):
    quantizer, block, inputs, outputs = quantize_setup()
    events = []
    downstream = [torch.nn.Identity(), torch.nn.Identity()]
    cache = SimpleNamespace(best=None, get=lambda indices: None, close=lambda: events.append("cache"))
    monkeypatch.setattr(BlockTuningSession, "_resolve_tuning_cache_plan", lambda *args: (DiffusionTuningCache, 1))
    monkeypatch.setattr(DiffusionTuningCache, "create", lambda *args, **kwargs: cache)
    monkeypatch.setattr(quantizer, "_compute_valid_token_mask", lambda ids: None)
    monkeypatch.setattr(quantizer, "_lfq_is_kd_block", lambda ctx: True)
    monkeypatch.setattr(quantizer, "_get_lfq_downstream_blocks", lambda ctx, device: downstream)
    monkeypatch.setattr(quantizer, "_forward_lfq_downstream", lambda pred, *args: pred)
    monkeypatch.setattr(quantizer, "lfq_loss", lambda pred, *args, **kwargs: pred.square().mean())
    monkeypatch.setattr(
        "auto_round.algorithms.quantization.sign_round.quantizer.mv_module_from_gpu",
        lambda module: events.append(next(i for i, item in enumerate(downstream) if item is module)),
    )

    def fail(*args, **kwargs):
        raise RuntimeError("injected integration failure")

    if exit_kind == "forward":
        monkeypatch.setattr(quantizer.block_forward, "forward", fail)
    elif exit_kind == "backward":
        monkeypatch.setattr(quantizer, "_scale_loss_and_backward", fail)
    elif exit_kind == "optimizer":
        monkeypatch.setattr(quantizer, "_step", fail)
    elif exit_kind == "pull":
        monkeypatch.setattr(quantizer, "pull_tuning_pool", fail)

    def run():
        return quantizer.quantize_block(
            block,
            inputs,
            {},
            outputs,
            None,
            SimpleNamespace(block_index=0, block_cnt=1, block_name="blocks.0"),
            input_ids=[torch.ones(1, 2, dtype=torch.long) for _ in inputs],
        )

    if exit_kind == "normal":
        assert run()
    else:
        with pytest.raises(RuntimeError, match="injected integration failure"):
            run()
    assert events == ([0, 1] if exit_kind == "pull" else ["cache", 0, 1])
