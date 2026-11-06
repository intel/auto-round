# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
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
from accelerate import Accelerator
from accelerate.state import AcceleratorState, PartialState
from accelerate.utils import DistributedType

from auto_round.parallel import ParallelConfig
from auto_round.parallel.distributed import is_distributed
from auto_round.parallel.tp.evaluation import _DISTRIBUTED_ENV_VARS, rank_zero_evaluation


@pytest.fixture(autouse=True)
def reset_distributed_caches():
    AcceleratorState._reset_state(reset_partial_state=True)
    is_distributed.cache_clear()
    yield
    AcceleratorState._reset_state(reset_partial_state=True)
    is_distributed.cache_clear()


def _mock_process_group(monkeypatch, rank):
    group = {"initialized": True, "events": []}
    monkeypatch.setattr(dist, "is_initialized", lambda: group["initialized"])
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "barrier", lambda: group["events"].append("barrier"))

    def destroy():
        group["events"].append("destroy")
        group["initialized"] = False

    monkeypatch.setattr(dist, "destroy_process_group", destroy)
    return group


@pytest.mark.parametrize("fail", [False, True])
def test_eval_clears_training_state_and_restores_environment(monkeypatch, fail):
    group = _mock_process_group(monkeypatch, rank=0)
    for key in _DISTRIBUTED_ENV_VARS:
        monkeypatch.setenv(key, "training-value")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    original = dict(os.environ)
    # Both Borg caches can outlive the training process group.
    AcceleratorState._shared_state["num_processes"] = 2
    PartialState._shared_state["num_processes"] = 2
    assert is_distributed()

    def evaluate():
        with rank_zero_evaluation() as should_evaluate:
            assert should_evaluate
            assert group["events"] == ["barrier", "destroy"]
            assert not dist.is_initialized()
            assert not is_distributed()
            assert not AcceleratorState._shared_state
            assert not PartialState._shared_state
            assert not any(key in os.environ for key in _DISTRIBUTED_ENV_VARS)
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
            accelerator = Accelerator(cpu=True)
            assert accelerator.num_processes == 1
            assert accelerator.distributed_type == DistributedType.NO
            if fail:
                raise RuntimeError("evaluation failed")

    if fail:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            evaluate()
    else:
        evaluate()
    assert dict(os.environ) == original
    assert not AcceleratorState._shared_state
    assert not PartialState._shared_state


def test_nonzero_rank_closes_group_without_evaluating(monkeypatch):
    group = _mock_process_group(monkeypatch, rank=1)
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    with rank_zero_evaluation() as should_evaluate:
        assert not should_evaluate
        assert not dist.is_initialized()
        assert os.environ["RANK"] == "1"
        assert os.environ["WORLD_SIZE"] == "2"
    assert group["events"] == ["barrier", "destroy"]


@pytest.mark.parametrize("format_name", ["fake", "auto_round"])
@pytest.mark.parametrize("tp_size,rank", [(None, 0), (1, 0), (2, 0), (2, 1)])
def test_cli_evaluates_only_after_tp_cleanup(monkeypatch, format_name, tp_size, rank):
    from auto_round.cli.main import start

    has_parallel_config = tp_size is not None
    tp_size = tp_size or 1
    group = _mock_process_group(monkeypatch, rank)
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", str(tp_size))
    monkeypatch.setenv("LOCAL_RANK", str(rank))
    model, tokenizer = Mock(), Mock()
    compressor = SimpleNamespace(
        tokenizer=tokenizer,
        model_context=SimpleNamespace(model=torch.nn.Linear(1, 1, device="cpu")),
        quantize_and_save=Mock(return_value=(model, "exported-model")),
    )
    if has_parallel_config:
        compressor.parallel_config = ParallelConfig(tp_size=tp_size)

    def evaluate(*args):
        assert args[:5] == (model, tokenizer, compressor, "exported-model", [format_name])
        if tp_size > 1:
            assert not dist.is_initialized()
            assert "LOCAL_RANK" not in os.environ
            assert "WORLD_SIZE" not in os.environ
        else:
            assert group["events"] == []
            assert os.environ["WORLD_SIZE"] == "1"

    with (
        patch("auto_round.AutoRound", return_value=compressor),
        patch("auto_round.utils.clear_memory"),
        patch("transformers.utils.versions.require_version"),
        patch("auto_round.eval.evaluation.run_model_evaluation", side_effect=evaluate) as evaluation,
    ):
        start(
            argv=[
                "local-model",
                "--alg",
                "qad",
                "--tp_size",
                str(tp_size),
                "--format",
                format_name,
                "--device",
                "cpu",
                "--tasks",
                "lambada_openai",
            ]
        )
    assert evaluation.call_count == (1 if rank == 0 else 0)
    assert group["events"] == (["barrier", "destroy"] if tp_size > 1 else [])


def _evaluation_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2", LOCAL_WORLD_SIZE="2", OMP_NUM_THREADS="1")
    dist.init_process_group(
        "gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        assert is_distributed()
        training_accelerator = Accelerator(cpu=True)
        assert training_accelerator.num_processes == 2
        with rank_zero_evaluation() as should_evaluate:
            assert not dist.is_initialized()
            assert not is_distributed()
            if should_evaluate:
                # This is the same Accelerate singleton used by lm-eval's HFLM.
                evaluation_accelerator = Accelerator(cpu=True)
                assert evaluation_accelerator.num_processes == 1
                assert evaluation_accelerator.distributed_type == DistributedType.NO
                with patch.object(dist, "all_gather", side_effect=AssertionError("unexpected collective")):
                    result = evaluation_accelerator.gather(torch.tensor([42]))
                assert result.tolist() == [42]
        Path(output, f"rank_{rank}.json").write_text(json.dumps({"evaluated": should_evaluate}))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires distributed Gloo")
def test_two_rank_training_transitions_to_single_process_eval(tmp_path):
    with patch.object(sys, "path", [entry for entry in sys.path if Path(entry).is_absolute()]):
        workers = mp.spawn(
            _evaluation_worker,
            args=(str(tmp_path / "rendezvous"), str(tmp_path)),
            nprocs=2,
            join=False,
        )
    try:
        deadline = time.monotonic() + 60
        while not workers.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("Evaluation transition hung waiting for another rank.")
    finally:
        for process in workers.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
    assert json.loads((tmp_path / "rank_0.json").read_text()) == {"evaluated": True}
    assert json.loads((tmp_path / "rank_1.json").read_text()) == {"evaluated": False}
