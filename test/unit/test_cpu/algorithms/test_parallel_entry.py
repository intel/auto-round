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

import importlib
import inspect
from dataclasses import asdict
from datetime import timedelta
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist

from auto_round import AutoRound, ParallelConfig, quantize_and_save
from auto_round.cli.main import _build_parallel_config, start
from auto_round.cli.parser import build_quantize_parser
from auto_round.parallel.tp import launcher


@pytest.fixture(autouse=True)
def clean_launcher_environment(monkeypatch):
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("name", ["engine", "mirror_sync", "spmd", "tune_parallel"])
def test_parallel_engines_live_in_shared_package(name):
    module = importlib.import_module(f"auto_round.parallel.dp.{name}")
    assert module.__package__ == "auto_round.parallel.dp"


def test_config_aliases_and_roundtrip():
    config = ParallelConfig(tensor_parallel=2)
    assert config == ParallelConfig(tp_size=2)
    assert config.tensor_parallel == 2
    assert ParallelConfig(pipeline_parallel=3).pipeline_parallel == 3
    assert ParallelConfig.from_value(asdict(config)) == config
    raw = {"data_parallel": 4, "parallel_batch": "per_replica"}
    assert ParallelConfig.from_value(raw).data_parallel == "4"
    assert raw["data_parallel"] == 4
    assert ParallelConfig(data_parallel=1).data_parallel == "off"


def test_dp_uses_one_unambiguous_python_config_field():
    assert "dp_size" not in inspect.signature(ParallelConfig).parameters
    assert "dp_size" not in asdict(ParallelConfig())
    with pytest.raises(TypeError, match="data_parallel"):
        ParallelConfig.from_value({"dp_size": 2})
    with pytest.raises(TypeError, match="data_parallel"):
        AutoRound("not-loaded", dp_size=2)
    with pytest.raises(TypeError, match="data_parallel"):
        quantize_and_save("not-loaded", "output", dp_size=2)
    assert ParallelConfig.from_value({"data_parallel": 2}).data_parallel == "2"


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "invalid", None])
def test_invalid_dp_values(value):
    with pytest.raises(ValueError, match="data_parallel"):
        ParallelConfig(data_parallel=value)


@pytest.mark.parametrize(
    "options",
    [
        {"tp_size": 2, "tensor_parallel": 4},
        {"pp_size": 2, "pipeline_parallel": 4},
    ],
)
def test_conflicting_aliases(options):
    with pytest.raises(ValueError, match="Conflicting"):
        ParallelConfig(**options)


@pytest.mark.parametrize("entry", [AutoRound, quantize_and_save])
@pytest.mark.parametrize("option", ["tensor_parallel", "pipeline_parallel", "data_parallel", "parallel_batch"])
def test_api_parallelism_only_uses_parallel_config(entry, option):
    assert option not in inspect.signature(entry).parameters
    args = ("not-loaded", "output") if entry is quantize_and_save else ("not-loaded",)
    with pytest.raises(TypeError, match="parallel_config"):
        entry(*args, **{option: 2})


def test_invalid_parallel_batch():
    with pytest.raises(ValueError, match="parallel_batch"):
        ParallelConfig(parallel_batch="invalid")


@pytest.mark.parametrize(
    "options",
    [
        {"tensor_parallel": 2, "data_parallel": 2},
        {"pipeline_parallel": 2, "data_parallel": "auto"},
        {"pipeline_parallel": 2, "tensor_parallel": 2},
    ],
)
def test_hybrids_rejected(options):
    with pytest.raises(NotImplementedError):
        ParallelConfig(**options)


@pytest.mark.parametrize("flag,dest", [("tensor_parallel", "tp_size"), ("pipeline_parallel", "pp_size")])
def test_cli_aliases(flag, dest):
    parser = build_quantize_parser()
    assert getattr(parser.parse_args([f"--{flag}", "2"]), dest) == 2
    with pytest.raises(SystemExit):
        parser.parse_args([f"--{flag}", "2", f"--{dest}", "4"])


@pytest.mark.parametrize("value", ["off", "auto", "2", "4"])
@pytest.mark.parametrize("equals", [False, True])
@pytest.mark.parametrize("flag", ["--dp", "--dp_size"])
def test_cli_dp_alias(value, equals, flag):
    parser = build_quantize_parser()
    options = [f"{flag}={value}"] if equals else [flag, value]
    args = parser.parse_args(options)
    full_args = parser.parse_args(["--data_parallel", value])
    assert args.data_parallel == value
    assert vars(args) == vars(full_args)
    assert _build_parallel_config(args) == _build_parallel_config(full_args)
    assert flag in parser._option_string_actions
    assert flag in parser.format_help()


@pytest.mark.parametrize("flag", ["--data_parallel", "--dp", "--dp_size"])
def test_dp_prevents_implicit_tp(monkeypatch, flag):
    monkeypatch.setenv("WORLD_SIZE", "4")
    args = build_quantize_parser().parse_args([flag, "auto"])
    assert _build_parallel_config(args)["tp_size"] == 1


@pytest.mark.parametrize("iters,expected", [(0, 3), (2, 2)])
def test_shared_dp_policy(iters, expected):
    config = ParallelConfig(data_parallel="auto", parallel_batch="per_replica")
    with patch("torch.cuda.device_count", return_value=3):
        policy = config.resolve_data_parallel_policy("0", iters=iters)
    assert (policy.world, policy.source, policy.batch_mode) == (expected, "auto", "per_replica")


def test_dp_rejects_external_world_before_device_query(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with patch("torch.cuda.device_count", side_effect=AssertionError), pytest.raises(NotImplementedError):
        ParallelConfig(data_parallel=2).resolve_data_parallel_policy("0")


@pytest.mark.parametrize("options", [{"data_parallel": 2}, {"pipeline_parallel": 2}])
def test_autoround_parallel_config_reaches_builder(options):
    with patch("auto_round.autoround._CompressorBuilder") as builder:
        AutoRound("local-model", parallel_config=options)
    assert builder.call_args.kwargs["parallel_config"] == ParallelConfig(**options)


def test_cli_launches_before_model_load():
    with (
        patch.object(launcher, "launch", return_value="launched") as launch,
        patch("auto_round.AutoRound", side_effect=AssertionError("model must load in workers")),
    ):
        assert start(argv=["local-model", "--tensor_parallel", "2"]) == "launched"
    assert launch.call_args.args[0].tp_size == 2


def test_launch_config_and_rank_zero_result():
    worker = Mock(return_value={0: ["checkpoint"], 1: []})
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=2),
        patch("torch.distributed.launcher.api.elastic_launch", return_value=worker) as elastic,
    ):
        assert launcher.launch(ParallelConfig(tensor_parallel=2), abs, -1) == ["checkpoint"]
    config = elastic.call_args.args[0]
    assert config.nproc_per_node == 2
    assert config.rdzv_endpoint == "localhost:0"
    assert config.max_restarts == 0
    worker.assert_called_once_with(abs, (-1,))


def test_launch_checks_visible_devices():
    with patch("torch.cuda.is_available", return_value=False), pytest.raises(ValueError, match="visible CUDA"):
        launcher.launch(ParallelConfig(tensor_parallel=2), abs, -1)


def test_external_group_not_relaunched_or_destroyed(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    with patch.object(dist, "destroy_process_group") as destroy:
        assert launcher.launch(ParallelConfig(tensor_parallel=2), abs, -3) == 3
    destroy.assert_not_called()
    monkeypatch.setenv("WORLD_SIZE", "4")
    with pytest.raises(ValueError, match="WORLD_SIZE"):
        launcher.needs_local_launch(ParallelConfig(tensor_parallel=2))


def test_initialized_single_rank_group_not_relaunched():
    with (
        patch.object(dist, "is_initialized", return_value=True),
        patch.object(dist, "get_world_size", return_value=1),
        pytest.raises(ValueError, match="WORLD_SIZE"),
    ):
        launcher.needs_local_launch(ParallelConfig(tensor_parallel=2))


def test_worker_failure_cleans_owned_group():
    def fail():
        raise RuntimeError("worker failed")

    with (
        patch.object(dist, "is_initialized", return_value=True),
        patch.object(dist, "destroy_process_group") as destroy,
        pytest.raises(RuntimeError, match="worker failed"),
    ):
        launcher._worker(fail, ())
    destroy.assert_called_once_with()


def test_task_api_returns_folders_and_preserves_input():
    save_options = {"format": "fake"}
    compressor = Mock()
    compressor.quantize_and_save.return_value = (object(), ["output"])
    with patch("auto_round.AutoRound", return_value=compressor) as ctor:
        result = quantize_and_save(
            "local-model", "output", save_kwargs=save_options, parallel_config={"data_parallel": 1}
        )
    assert result == ["output"]
    assert ctor.call_args.kwargs["parallel_config"].data_parallel == "off"
    assert save_options == {"format": "fake"}
    compressor.quantize_and_save.assert_called_once_with("output", format="fake")


def test_task_api_rejects_preloaded_objects():
    with pytest.raises(TypeError, match="model name/path"):
        quantize_and_save(torch.nn.Linear(2, 2), "output")
    with pytest.raises(ValueError, match="tokenizer"):
        quantize_and_save("local-model", "output", tokenizer=object())
    with pytest.raises(ValueError, match="return_folders"):
        quantize_and_save("local-model", "output", save_kwargs={"return_folders": True})


def _gloo_worker():
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    value = torch.tensor([dist.get_rank() + 1])
    dist.all_reduce(value)
    return int(value.item())


def _failing_worker():
    raise RuntimeError("intentional launcher failure")


def test_real_local_launcher_gloo():
    # Exercise process creation/collectives without requiring two GPUs. Only the
    # parent eligibility probe is mocked; workers use real CPU/Gloo collectives.
    with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=2):
        assert launcher.launch(ParallelConfig(tensor_parallel=2), _gloo_worker) == 3


def test_real_local_launcher_propagates_failure():
    from torch.distributed.elastic.multiprocessing.errors import ChildFailedError

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=2),
        pytest.raises(ChildFailedError),
    ):
        launcher.launch(ParallelConfig(tensor_parallel=2), _failing_worker)


def test_api_dp_policy_attached_before_model_loading():
    from transformers import LlamaConfig, LlamaForCausalLM

    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    with (
        patch("torch.cuda.device_count", return_value=2),
        patch("auto_round.parallel.dp.engine.check_parallel_device_map_compat_"),
    ):
        compressor = AutoRound(
            model,
            tokenizer=Mock(),
            alg_configs="auto_round",
            parallel_config=ParallelConfig(data_parallel=2, parallel_batch="per_replica"),
            device_map="cpu",
            low_cpu_mem_usage=False,
        )
    assert compressor.compress_context.parallel_policy.world == 2
    assert compressor.compress_context.parallel_policy.batch_mode == "per_replica"


def test_task_api_offline_quantize_and_save(tmp_path):
    from pathlib import Path
    from test.unit.test_cpu.algorithms.test_parallel_execution import _make_checkpoint

    from auto_round import RTNConfig

    checkpoint = _make_checkpoint(tmp_path)
    folders = quantize_and_save(
        str(checkpoint),
        str(tmp_path / "quantized"),
        alg_configs=RTNConfig(group_size=8, disable_opt_rtn=True),
        device_map="cpu",
        low_cpu_mem_usage=False,
        disable_model_free=True,
        save_kwargs={"format": "fake"},
    )
    assert isinstance(folders, list) and len(folders) == 1
    assert (Path(folders[0]) / "config.json").is_file()
