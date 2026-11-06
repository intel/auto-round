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
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from auto_round.parallel.execution import ParallelExecution


def test_custom_loop_retains_autograd_and_update_boundary():
    model = torch.nn.Linear(4, 4)
    reference = copy.deepcopy(model)
    samples = [torch.randn(2, 4)]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    sampler = SimpleNamespace(next_batch=lambda: [0])
    with ParallelExecution().prepare_run(model, samples, "cpu", whole_model=True) as run:
        assert run.samples is samples
        run.prepare_training(None)
        for _ in range(2):
            optimizer.zero_grad()
            reference_optimizer.zero_grad()
            indices = run.next_batch(sampler)
            loss = run.forward(run.samples[indices[0]]).square().mean()
            loss.backward()
            reference(samples[0]).square().mean().backward()
            run.step(optimizer.step)
            reference_optimizer.step()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)
    with pytest.raises(RuntimeError, match="already exited"):
        run.forward(samples[0])


def test_training_binds_after_new_parameters_and_syncs_before_master_copy():
    model = torch.nn.Linear(2, 2)
    events = []
    execution = ParallelExecution()

    class TrainingWrapper(torch.nn.Module):
        def forward(self, inputs):
            events.append("training_forward")
            return {"logits": model(inputs) + model.tuning}

    def prepare(algorithm, module):
        assert module is model
        assert model.tuning.requires_grad
        events.append("prepare")

        def synchronize():
            events.append("sync")
            model.tuning.grad.mul_(2)

        return TrainingWrapper(), synchronize

    with (
        patch.object(execution, "prepare_gradients", side_effect=prepare),
        execution.prepare_run(model, device="cpu") as run,
    ):
        with pytest.raises(RuntimeError, match="prepare_training"):
            run.step(lambda: None)
        model.tuning = torch.nn.Parameter(torch.ones(2))
        run.prepare_training(None)
        run.forward(torch.ones(1, 2), output="logits").sum().backward()

        def update():
            events.append("copy_master_and_step")
            torch.testing.assert_close(model.tuning.grad, torch.full((2,), 2.0))

        run.step(update)
        with pytest.raises(RuntimeError, match="once"):
            run.prepare_training(None)
    assert events == ["prepare", "training_forward", "sync", "copy_master_and_step"]
    assert run._training_model is model
    assert run._sync_gradients is None


def test_scope_releases_training_wrapper_on_exception():
    model = torch.nn.Linear(2, 2)
    execution = ParallelExecution()
    wrapper = torch.nn.Sequential(model)
    with (
        patch.object(execution, "prepare_gradients", return_value=(wrapper, Mock())),
        pytest.raises(ValueError, match="failure"),
        execution.prepare_run(model, device="cpu") as run,
    ):
        run.prepare_training(None)
        raise ValueError("failure")
    assert run._training_model is model
    assert run._sync_gradients is None


def test_whole_model_prepares_residency_before_cached_or_zero_shot_paths():
    model = torch.nn.Linear(2, 2)
    model.scale = torch.ones(1)
    model.tuning_device = "cuda:7"
    native_hook = model.register_forward_pre_hook(lambda module, args: None)
    with ParallelExecution().prepare_run(model, device="meta", whole_model=True):
        assert model.weight.device.type == "meta"
        assert model.scale.device.type == "meta"
        assert not hasattr(model, "tuning_device")
        assert native_hook.id in model._forward_pre_hooks
    native_hook.remove()


def test_block_scope_preserves_existing_placement():
    model = torch.nn.Linear(2, 2)
    model.tuning_device = "cpu"
    execution = ParallelExecution({"pp_size": 2})
    with (
        patch.object(execution, "prepare_whole_model", side_effect=AssertionError("block already placed")),
        execution.prepare_run(model, device="cpu"),
    ):
        assert model.tuning_device == "cpu"
    with (
        pytest.raises(NotImplementedError, match="naive pipeline parallelism"),
        execution.prepare_run(model, device="cpu", whole_model=True),
    ):
        pytest.fail("must reject before tuning")


def test_tp_requires_native_parameters():
    with pytest.raises(ValueError, match="DTensor"):
        ParallelExecution({"tp_size": 2}).prepare_run(torch.nn.Linear(2, 2))


def test_collectives_use_the_model_mesh_not_the_global_group():
    model = torch.nn.Linear(2, 2)
    group = object()
    model.weight.device_mesh = SimpleNamespace(ndim=1, mesh=torch.tensor([3, 5]), get_group=lambda: group)

    def broadcast(values, *, src, group):
        assert src == 3
        values[0] = [7]

    sampler = SimpleNamespace(next_batch=Mock(side_effect=AssertionError("only mesh leader samples")))
    with (
        patch("auto_round.parallel.tp.runtime.is_dtensor", side_effect=lambda value: value is model.weight),
        patch.object(dist, "get_rank", return_value=5),
        patch.object(dist, "broadcast_object_list", side_effect=broadcast) as send,
        patch.object(dist, "all_reduce") as reduce,
        ParallelExecution().prepare_run(model, device="cpu") as run,
    ):
        assert run.samples == [7]
        assert run.next_batch(sampler) == [7]
        assert run.cache.ensure(False, lambda: None)
    assert send.call_count == 2
    assert all(call.kwargs["group"] is group for call in send.call_args_list)
    assert reduce.call_args.kwargs["group"] is group


def test_local_cache_fingerprints_and_hit_callbacks(tmp_path):
    model = torch.nn.Linear(2, 2).to(torch.bfloat16)
    model.register_buffer("counter", torch.tensor(0))
    samples = [{"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)}]
    with ParallelExecution().prepare_run(model, samples, "cpu") as run:
        assert run.cache.workspace(tmp_path) == str(tmp_path)
        signature = run.cache.signature(samples)
        assert signature["seqlen"] == 2
        model.counter.add_(1)
        assert signature != run.cache.signature(samples)
        signature = run.cache.signature(samples)
        samples[0]["attention_mask"][0, 1] = 0
        assert signature != run.cache.signature(samples)
        build = Mock()
        assert not run.cache.ensure(True, build)
        build.assert_not_called()
        assert run.cache.ensure(False, build)
        build.assert_called_once_with()


def test_dp_cache_namespace_does_not_require_tp_coordination(tmp_path):
    run = ParallelExecution().prepare_run(torch.nn.Linear(2, 2), device="cpu")
    with (
        patch.object(dist, "is_initialized", return_value=True),
        patch.object(dist, "get_world_size", return_value=2),
        patch.object(dist, "get_rank", return_value=1),
        patch.object(dist, "all_reduce", side_effect=AssertionError("independent DP cache")),
    ):
        assert Path(run.cache.workspace(tmp_path)).name == "dp_rank_1"
        assert run.cache.ensure(False, lambda: None)


def _prepared_worker(rank, rendezvous, workspace):
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2, timeout=timedelta(seconds=45)
    )
    try:
        torch.manual_seed(42)
        model = torch.nn.Linear(4, 4, bias=False)
        reference = copy.deepcopy(model)
        mesh = dist.init_device_mesh("cpu", (2,))
        parallelize_module(model, mesh, ColwiseParallel(use_local_output=False))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        samples = [torch.ones(2, 4) * (rank + 1)]
        execution = ParallelExecution({"tp_size": 2})
        with execution.prepare_run(model, samples, "cpu", whole_model=True) as run:
            torch.testing.assert_close(run.samples[0], torch.ones(2, 4))
            assert Path(run.cache.workspace(workspace)).name == f"tp_rank_{rank}"
            sampler = SimpleNamespace(next_batch=lambda: [rank, rank + 2])
            assert run.next_batch(sampler) == [0, 2]
            calls = []

            def build():
                calls.append(run.forward(run.samples[0]).detach())

            assert run.cache.ensure(rank == 0, build)
            assert len(calls) == 1
            assert not run.cache.ensure(True, build)
            signature = run.cache.signature(run.samples)
            if rank == 1:
                with torch.no_grad():
                    model.weight.to_local().add_(0.25)
            changed = run.cache.signature(run.samples)
            assert signature != changed  # Includes changes to the *other* shard.
            if rank == 1:
                with torch.no_grad():
                    model.weight.to_local().sub_(0.25)
            with patch(
                "auto_round.parallel.distributed.setup_ddp_if_needed_", side_effect=AssertionError("TP is not DP")
            ):
                run.prepare_training(None)
            original_full_tensor = DTensor.full_tensor

            def no_weight_gather(value, *args, **kwargs):
                assert tuple(value.shape) != (4, 4), "must not gather model weights"
                return original_full_tensor(value, *args, **kwargs)

            with patch.object(DTensor, "full_tensor", no_weight_gather):
                prediction = run.forward(run.samples[0])
                expected = reference(run.samples[0])
                torch.testing.assert_close(prediction, expected)
                prediction.square().mean().backward()
                expected.square().mean().backward()
                torch.testing.assert_close(model.weight.grad.to_local(), reference.weight.grad.chunk(2, 0)[rank])
                run.step(optimizer.step)
                reference_optimizer.step()
                torch.testing.assert_close(model.weight.to_local(), reference.weight.chunk(2, 0)[rank])
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires distributed Gloo")
def test_two_rank_prepared_run(tmp_path):
    with patch.object(sys, "path", [entry for entry in sys.path if Path(entry).is_absolute()]):
        workers = mp.spawn(
            _prepared_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path / "cache")), nprocs=2, join=False
        )
    try:
        deadline = time.monotonic() + 120
        while not workers.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("PreparedRun TP test timed out.")
    finally:
        for process in workers.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
