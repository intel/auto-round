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

import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from test.unit.test_cpu.algorithms.test_parallel_execution import _make_checkpoint, _run_workers
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2 or not dist.is_nccl_available(),
    reason="requires two CUDA devices and NCCL",
)
def test_two_rank_blockwise_tp_cpu_residency(tmp_path):
    checkpoint = _make_checkpoint(tmp_path)
    _run_workers(tmp_path, checkpoint, device_type="cuda")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2 or not dist.is_nccl_available(),
    reason="requires two CUDA devices and NCCL",
)
def test_local_tp_task_api_and_cli(tmp_path):
    from auto_round import ParallelConfig, SignRoundConfig, quantize_and_save

    checkpoint = _make_checkpoint(tmp_path)
    tokens = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]])
    folders = quantize_and_save(
        str(checkpoint),
        str(tmp_path / "api"),
        parallel_config=ParallelConfig(tensor_parallel=2),
        alg_configs=SignRoundConfig(iters=2, group_size=8, not_use_best_mse=True),
        dataset=torch.utils.data.DataLoader(tokens, batch_size=1),
        nsamples=2,
        seqlen=4,
        batch_size=1,
        model_dtype="float32",
        amp=False,
        enable_torch_compile=False,
        low_cpu_mem_usage=False,
        save_kwargs={"format": "fake"},
    )
    assert len(folders) == 1
    assert (Path(folders[0]) / "config.json").is_file()
    assert list(Path(folders[0]).glob("*.safetensors"))
    assert not dist.is_initialized(), "the parent must not acquire a worker process group"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "auto_round",
            "quantize",
            str(checkpoint),
            "--tensor_parallel",
            "2",
            "--algorithm",
            "rtn",
            "--group_size",
            "8",
            "--format",
            "fake",
            "--output_dir",
            str(tmp_path / "cli"),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert list((tmp_path / "cli").rglob("model*.safetensors"))


def _whole_model_worker(rank, rendezvous, checkpoint, algorithm):
    from auto_round import AutoRound
    from auto_round.algorithms.quantization.qad.config import QADConfig
    from auto_round.algorithms.quantization.qat.config import QATConfig
    from auto_round.algorithms.quantization.sign_round.config import SignRoundV2Config
    from auto_round.parallel.runtime import is_dtensor

    torch.set_num_threads(1)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2, timeout=timedelta(seconds=120)
    )
    try:
        tokens = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]])
        config_class = QADConfig if algorithm == "qad" else QATConfig
        compressor = AutoRound(
            checkpoint,
            scheme="W2A16",
            parallel_config={"tp_size": 2},
            alg_configs=[
                SignRoundV2Config(iters=1, group_size=8),
                config_class(
                    iters=1,
                    group_size=8,
                    topk=8,
                    gradient_checkpointing=True,
                    qad_workspace=str(Path(checkpoint).parent / "teacher"),
                ),
            ],
            dataset=torch.utils.data.DataLoader(tokens, batch_size=1),
            nsamples=2,
            seqlen=4,
            batch_size=1,
            model_dtype="float32",
            low_cpu_mem_usage=False,
            enable_torch_compile=False,
        )
        compressor.post_init()
        quantizer = compressor.alg_composer.block_quantizer
        forward = quantizer._forward_logits
        calls = []

        def check_forward(model, inputs, vocab_size, run=None):
            assert model.model.embed_tokens.weight.device == device
            assert all(parameter.device == device for parameter in model.parameters())
            input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs
            assert input_ids.device == device
            calls.append(True)
            return forward(model, inputs, vocab_size, run)

        with patch.object(quantizer, "_forward_logits", side_effect=check_forward):
            model, _ = compressor.quantize()
        assert calls
        assert is_dtensor(model.model.layers[0].self_attn.q_proj.weight)
        assert model.model.embed_tokens.weight.device == device
        with torch.no_grad():
            assert torch.isfinite(model(input_ids=tokens.to(device)).logits).all()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2 or not dist.is_nccl_available(),
    reason="requires two CUDA devices and NCCL",
)
@pytest.mark.parametrize("algorithm", ["qad", "qat"])
def test_two_rank_block_to_whole_model_residency(tmp_path, algorithm):
    checkpoint = _make_checkpoint(tmp_path)
    with patch.object(sys, "path", [entry for entry in sys.path if Path(entry).is_absolute()]):
        workers = mp.spawn(
            _whole_model_worker,
            args=(str(tmp_path / "rendezvous"), str(checkpoint), algorithm),
            nprocs=2,
            join=False,
        )
    try:
        deadline = time.monotonic() + 180
        while not workers.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("TP block-to-whole-model tuning timed out.")
    finally:
        for process in workers.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
