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

"""CPU-only contracts for block tuning data access and diffusion staging."""

import queue
import random
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from auto_round.algorithms.block_runner import BlockForwardRunner
from auto_round.algorithms.quantization.base import BaseQuantizer
from auto_round.algorithms.tuning import BlockTuningSession
from auto_round.compressors.diffusion.tuning_cache import DiffusionTuningCache
from auto_round.compressors.utils import IndexSampler


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def reject_cuda(*args, **kwargs):
        pytest.fail("CPU-only tuning tests must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", reject_cuda)
    state = random.getstate()
    yield
    random.setstate(state)


@pytest.fixture
def session_args():
    block = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        block.weight.copy_(torch.eye(2))
    return dict(
        block=block,
        runner=BlockForwardRunner(batch_size=1, device="cpu", amp=False, enable_torch_compile=False),
        inputs=[torch.full((1, 2), float(i + 1)) for i in range(5)],
        others={},
        outputs=[torch.full((1, 2), float(i + 2)) for i in range(5)],
        nsamples=5,
        global_batch_size=1,
        iters=3,
        device="cpu",
        loss_device="cpu",
        model_context=SimpleNamespace(is_diffusion=True, diffusion_tuning_cache_size=1),
        compress_context=SimpleNamespace(cache_device="cpu", low_gpu_mem_usage=True),
    )


@pytest.fixture
def backend(monkeypatch):
    cache = SimpleNamespace(
        best=None,
        get=Mock(return_value=None),
        forward=Mock(),
        collect_best_params=Mock(),
        close=Mock(),
    )
    supported = Mock(return_value=True)
    create = Mock(return_value=cache)
    monkeypatch.setattr(DiffusionTuningCache, "is_supported", supported)
    monkeypatch.setattr(DiffusionTuningCache, "create", create)
    return SimpleNamespace(cache=cache, create=create, supported=supported)


@pytest.mark.parametrize("global_batch_size", [1, 2])
def test_synchronous_forward_and_cyclic_sampling(session_args, monkeypatch, global_batch_size):
    create = Mock(side_effect=AssertionError("unsupported CPU session must not create a cache"))
    monkeypatch.setattr(DiffusionTuningCache, "create", create)
    session_args.update(global_batch_size=global_batch_size, iters=8)
    rng = random.getstate()
    expected_sampler = IndexSampler(5, global_batch_size)
    expected_batches = [expected_sampler.next_batch() for _ in range(8)]
    random.setstate(rng)

    with BlockTuningSession(**session_args) as session:
        for iteration in session.iterations():
            indices = session.next_batch()
            assert indices == expected_batches[iteration]
            prediction, reference = session.forward(indices)
            torch.testing.assert_close(prediction, torch.cat([session_args["inputs"][i] for i in indices]))
            torch.testing.assert_close(reference, torch.cat([session_args["outputs"][i] for i in indices]))
            assert prediction.device.type == reference.device.type == "cpu"
    create.assert_not_called()
    assert session_args["runner"].cache_device == "cpu"


@pytest.mark.parametrize("model_context", [SimpleNamespace(is_diffusion=False), SimpleNamespace()])
def test_non_diffusion_never_consults_backend(session_args, backend, model_context):
    session_args["model_context"] = model_context
    with BlockTuningSession(**session_args) as session:
        for _ in session.iterations():
            session.forward(session.next_batch())
    backend.supported.assert_not_called()
    backend.create.assert_not_called()


@pytest.mark.parametrize("budget,first_cached_iteration", [(1, 0), ("auto", 1)])
def test_cache_creation_timing_and_completed_warmup(session_args, backend, budget, first_cached_iteration):
    session_args["model_context"].diffusion_tuning_cache_size = budget
    block = session_args["block"]
    optimizer = torch.optim.SGD(block.parameters(), lr=0.1)
    initial_weight = block.weight.detach().clone()
    completed_steps = []

    def create(*args, **kwargs):
        assert len(completed_steps) == first_cached_iteration
        if budget == "auto":
            assert not torch.equal(block.weight, initial_weight)
        return backend.cache

    backend.create.side_effect = create
    with BlockTuningSession(**session_args) as session:
        for iteration in session.iterations():
            assert backend.create.call_count == int(iteration >= first_cached_iteration)
            prediction, reference = session.forward(session.next_batch())
            optimizer.zero_grad()
            (prediction - reference).square().mean().backward()
            optimizer.step()
            completed_steps.append(iteration)
            backend.cache.close.assert_not_called()

    args, kwargs = backend.create.call_args
    assert args[:5] == tuple(session_args[key] for key in ("block", "runner", "inputs", "others", "outputs"))
    assert args[5] is session.sampler
    assert args[6] == session_args["iters"] - first_cached_iteration
    assert kwargs == {"budget_gib": budget, "device": "cpu"}
    backend.create.assert_called_once()
    backend.cache.close.assert_called_once_with()


@pytest.mark.parametrize("budget,iters", [(1, 0), ("auto", 0), ("auto", 1)])
def test_short_runs_do_not_create_cache(session_args, backend, budget, iters):
    session_args["model_context"].diffusion_tuning_cache_size = budget
    session_args["iters"] = iters
    with BlockTuningSession(**session_args) as session:
        assert list(session.iterations()) == list(range(iters))
    backend.create.assert_not_called()
    backend.cache.close.assert_not_called()


@pytest.mark.parametrize("fallback", ["create", "get"])
def test_missing_cache_or_batch_falls_back_to_synchronous_forward(session_args, backend, fallback):
    if fallback == "create":
        backend.create.return_value = None
    runner = session_args["runner"]
    runner.forward = Mock(wraps=runner.forward)
    with BlockTuningSession(**session_args) as session:
        for _ in session.iterations():
            indices = session.next_batch()
            prediction, reference = session.forward(indices)
            torch.testing.assert_close(prediction, session_args["inputs"][indices[0]])
            torch.testing.assert_close(reference, session_args["outputs"][indices[0]])
    backend.create.assert_called_once()
    assert runner.forward.call_count == session_args["iters"]
    backend.cache.forward.assert_not_called()
    if fallback == "get":
        assert backend.cache.get.call_count == session_args["iters"]
        backend.cache.close.assert_called_once_with()
    else:
        backend.cache.get.assert_not_called()
        backend.cache.close.assert_not_called()


@pytest.mark.parametrize(
    "device,low_memory,forward_device", [("cpu", True, None), ("cuda:0", True, "cuda:0"), ("cuda:0", False, None)]
)
def test_staged_forward_uses_backend_once_without_synchronous_forward(
    session_args, backend, device, low_memory, forward_device
):
    # CUDA is only a routing label: the fake backend returns CPU tensors.
    session_args["device"] = device
    session_args["compress_context"].low_gpu_mem_usage = low_memory
    staged = ({}, {}, torch.full((1, 2), 42.0))
    prediction = torch.ones(1, 2)
    backend.cache.get.return_value = staged
    backend.cache.forward.return_value = prediction
    session_args["runner"].forward = Mock(side_effect=AssertionError("duplicate synchronous forward"))
    with BlockTuningSession(**session_args) as session:
        next(session.iterations())
        indices = session.next_batch()
        actual, reference = session.forward(indices)
    assert actual is prediction
    assert reference is staged[2]
    backend.cache.get.assert_called_once_with(indices)
    backend.cache.forward.assert_called_once_with(session_args["block"], staged, forward_device)
    session_args["runner"].forward.assert_not_called()
    assert session_args["runner"].cache_device == "cpu"


@pytest.mark.parametrize("storage", ["backend", "ordinary", "no_cache"])
def test_collect_best_params_storage_policy(session_args, backend, storage):
    block = session_args["block"]
    block.orig_layer = torch.nn.Identity()
    block.params = {"weight": block.weight}
    if storage == "backend":
        # Represent GPU-owned storage with a CPU sentinel; allocation is not under test.
        backend.cache.best = {"weight": torch.empty_like(block.weight)}
        backend.cache.collect_best_params.return_value = backend.cache.best
    elif storage == "no_cache":
        backend.create.return_value = None
    with BlockTuningSession(**session_args) as session:
        next(session.iterations())
        snapshot = session.collect_best_params()
        if storage == "backend":
            assert snapshot is backend.cache.best
            backend.cache.collect_best_params.assert_called_once_with()
        else:
            backend.cache.collect_best_params.assert_not_called()
            torch.testing.assert_close(snapshot["weight"], block.weight)
            assert snapshot["weight"].device.type == "cpu"
            with torch.no_grad():
                block.weight.add_(1)
            assert not torch.equal(snapshot["weight"], block.weight)


@pytest.mark.parametrize("exit_kind", ["normal", "break", "return", "forward", "backward", "optimizer", "producer"])
def test_cache_closes_after_all_consumers_and_propagates_errors(session_args, backend, exit_kind):
    error = RuntimeError(f"{exit_kind} failed")
    block = session_args["block"]
    optimizer = torch.optim.SGD(block.parameters(), lr=0.1)
    if exit_kind == "forward":
        block.register_forward_pre_hook(Mock(side_effect=error))
    elif exit_kind == "backward":
        block.weight.register_hook(Mock(side_effect=error))
    elif exit_kind == "optimizer":
        optimizer.step = Mock(side_effect=error)
    elif exit_kind == "producer":
        # Exercise the real producer-error queue handoff, before any CUDA calls.
        failed_cache = DiffusionTuningCache()
        failed_cache.stop = threading.Event()
        failed_cache.current = None
        failed_cache.ready = queue.Queue()
        failed_cache.ready.put(error)
        backend.cache.get.side_effect = failed_cache.get

    completed = []

    def tune():
        with BlockTuningSession(**session_args) as session:
            for iteration in session.iterations():
                backend.cache.close.assert_not_called()
                prediction, reference = session.forward(session.next_batch())
                (prediction - reference).square().mean().backward()
                backend.cache.close.assert_not_called()
                optimizer.step()
                optimizer.zero_grad()
                completed.append(iteration)
                backend.cache.close.assert_not_called()
                if exit_kind == "break":
                    break
                if exit_kind == "return":
                    return "early"

    if exit_kind in {"normal", "break", "return"}:
        assert tune() == ("early" if exit_kind == "return" else None)
        assert len(completed) == (3 if exit_kind == "normal" else 1)
    else:
        with pytest.raises(RuntimeError) as caught:
            tune()
        assert caught.value is error
        assert not completed
    backend.cache.close.assert_called_once_with()


@pytest.mark.parametrize("consume_first", [False, True])
def test_iterations_is_single_use_without_overwriting_cache(session_args, backend, consume_first):
    with BlockTuningSession(**session_args) as session:
        iterator = session.iterations()
        if consume_first:
            next(iterator)
        with pytest.raises(RuntimeError, match="only be called once"):
            session.iterations()
        assert backend.create.call_count == int(consume_first)
    assert backend.cache.close.call_count == int(consume_first)


@pytest.mark.parametrize("phase", ["before_enter", "after_exit"])
@pytest.mark.parametrize("operation", ["iterations", "next_batch", "forward", "collect_best_params"])
def test_data_access_requires_active_context(session_args, backend, phase, operation):
    session = BlockTuningSession(**session_args)
    if phase == "after_exit":
        with session:
            next(session.iterations())
    calls_before = backend.create.call_count
    with pytest.raises(RuntimeError, match="active context"):
        args = ([0],) if operation == "forward" else ()
        getattr(session, operation)(*args)
    assert backend.create.call_count == calls_before
    backend.cache.get.assert_not_called()
    backend.cache.collect_best_params.assert_not_called()


@pytest.mark.parametrize("nested", [False, True])
def test_session_cannot_be_entered_twice(session_args, backend, nested):
    session = BlockTuningSession(**session_args)
    with session:
        next(session.iterations())
        if nested:
            with pytest.raises(RuntimeError, match="only be entered once"), session:
                pytest.fail("nested entry must be rejected")
            backend.cache.close.assert_not_called()
    if not nested:
        with pytest.raises(RuntimeError, match="only be entered once"), session:
            pytest.fail("closed session must not reopen")
    backend.cache.close.assert_called_once_with()


@pytest.mark.parametrize("advance_before_exit", [False, True])
def test_retained_iterator_cannot_run_after_exit(session_args, backend, advance_before_exit):
    session_args["model_context"].diffusion_tuning_cache_size = "auto"
    with BlockTuningSession(**session_args) as session:
        iterator = session.iterations()
        if advance_before_exit:
            assert next(iterator) == 0
    with pytest.raises(RuntimeError, match="active context"):
        next(iterator)
    backend.create.assert_not_called()


def test_close_failure_still_marks_session_closed(session_args, backend):
    backend.cache.close.side_effect = RuntimeError("close failed")
    session = BlockTuningSession(**session_args)
    with pytest.raises(RuntimeError, match="close failed"), session:
        next(session.iterations())
    with pytest.raises(RuntimeError, match="active context"):
        session.next_batch()


def test_base_quantizer_creates_usable_session(session_args, monkeypatch):
    monkeypatch.setattr("auto_round.algorithms.quantization.base.device_manager", SimpleNamespace(device="cpu"))
    quantizer = SimpleNamespace(
        block_forward=session_args["runner"],
        model_context=session_args["model_context"],
        compress_context=session_args["compress_context"],
    )
    kwargs = {key: session_args[key] for key in ("nsamples", "global_batch_size", "iters", "loss_device")}
    with BaseQuantizer.tuning_session(
        quantizer,
        session_args["block"],
        session_args["inputs"],
        session_args["others"],
        session_args["outputs"],
        **kwargs,
    ) as session:
        assert isinstance(session, BlockTuningSession)
        for _ in session.iterations():
            indices = session.next_batch()
            prediction, reference = session.forward(indices)
            torch.testing.assert_close(prediction + 1, reference)


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, True),
        ({"budget": "auto"}, True),
        ({"budget": 0}, False),
        ({"budget": -1}, False),
        ({"is_diffusion": False}, False),
        ({"low_memory": False}, False),
        ({"device": "cpu", "loss_device": "cpu"}, False),
        ({"devices": ["cuda:0", "cuda:1"]}, False),
        ({"devices": []}, False),
        ({"loss_device": None}, True),
        ({"loss_device": torch.device("cuda:0")}, True),
        ({"loss_device": "cuda:1"}, False),
        ({"loss_device": "cpu"}, False),
    ],
    ids=[
        "explicit",
        "auto",
        "disabled",
        "negative",
        "non_diffusion",
        "not_low_memory",
        "cpu",
        "multi_gpu",
        "no_device",
        "implicit_loss",
        "same_loss",
        "other_gpu_loss",
        "cpu_loss",
    ],
)
def test_real_diffusion_cache_support_policy(monkeypatch, overrides, expected):
    policy = dict(
        budget=1, is_diffusion=True, low_memory=True, device="cuda:0", loss_device="cuda:0", devices=["cuda:0"]
    )
    policy.update(overrides)
    monkeypatch.setattr(
        "auto_round.compressors.diffusion.tuning_cache.device_manager", SimpleNamespace(device_list=policy["devices"])
    )
    model = SimpleNamespace(is_diffusion=policy["is_diffusion"], diffusion_tuning_cache_size=policy["budget"])
    compress = SimpleNamespace(low_gpu_mem_usage=policy["low_memory"])
    assert DiffusionTuningCache.is_supported(model, compress, policy["device"], policy["loss_device"]) is expected
