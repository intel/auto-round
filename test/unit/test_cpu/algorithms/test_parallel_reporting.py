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

import logging
from datetime import timedelta
from io import StringIO
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from auto_round import ParallelConfig
from auto_round.algorithms.config_resolver import _prepare_entry_kwargs
from auto_round.cli.main import _build_parallel_config
from auto_round.cli.parser import build_quantize_parser
from auto_round.logger import RankZeroFilter, logger
from auto_round.utils.device import memory_monitor, set_non_auto_device_map
from auto_round.utils.device_manager import DeviceManager, device_manager


def test_dp_splits_global_batch_without_an_extra_flag(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    args = build_quantize_parser().parse_args(["local-model", "--data_parallel", "2"])
    config = ParallelConfig.from_value(_build_parallel_config(args))
    with patch("torch.cuda.device_count", return_value=2):
        policy = config.resolve_data_parallel_policy("0", iters=2)
    assert policy.world == 2
    assert policy.batch_mode == "split"


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("shared_device", [False, True])
def test_tp_memory_is_merged_once_on_rank_zero(monkeypatch, rank, shared_device):
    monkeypatch.setattr(device_manager, "_tensor_parallel", True)
    monkeypatch.setattr(memory_monitor, "peak_ram", 1.0)
    index = 0 if shared_device else rank
    local_peaks = {str(index): 2.5 + rank}
    monkeypatch.setattr(memory_monitor, "peak_vram", local_peaks.copy())

    def gather(reports, local):
        assert local == {str(index): local_peaks[str(index)]}
        reports[:] = [{"0": 2.5}, {str(0 if shared_device else 1): 3.5}]

    output = StringIO()
    handler = logging.StreamHandler(output)
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        with (
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "get_world_size", return_value=2),
            patch.object(dist, "get_rank", return_value=rank),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=index),
            patch("torch.cuda.max_memory_reserved", return_value=local_peaks[str(index)] * 1024**3),
            patch.object(memory_monitor, "update"),
            patch.object(dist, "all_gather_object", side_effect=gather) as collective,
        ):
            summary = device_manager.log_memory_summary()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
    collective.assert_called_once()
    assert memory_monitor.peak_vram == local_peaks, "merged peer statistics must not pollute local sampling"
    if rank == 0:
        expected = "{'0': 3.5GB}" if shared_device else "{'0': 2.5GB, '1': 3.5GB}"
        assert f"'peak_vram': {expected}" in summary
        assert len(output.getvalue().splitlines()) == 1
        assert "TP rank" not in output.getvalue()
    else:
        assert summary is None
        assert not output.getvalue()


def test_rank_one_cannot_bypass_routine_log_filter(monkeypatch):
    monkeypatch.setenv("RANK", "1")
    ordinary = logging.LogRecord("autoround", logging.INFO, __file__, 1, "ordinary", (), None)
    memory = logging.LogRecord("autoround", logging.INFO, __file__, 1, "memory", (), None)
    with patch.object(dist, "is_initialized", return_value=False):
        assert not RankZeroFilter().filter(ordinary)
        assert not RankZeroFilter().filter(memory)


def test_arbitrary_tp_log_sites_do_not_communicate_or_print_partial_reports(monkeypatch):
    monkeypatch.setattr(device_manager, "_tensor_parallel", True)
    with (
        patch.object(dist, "all_gather_object", side_effect=AssertionError("not an all-rank boundary")),
        patch.object(logger, "info") as info,
    ):
        memory_monitor.log_summary()
    info.assert_not_called()


def test_device_manager_tp_reporting_state_can_be_reset(monkeypatch):
    monkeypatch.setattr(device_manager, "_tensor_parallel", False)
    device_manager.set_tensor_parallel(True)
    assert DeviceManager().is_tensor_parallel
    device_manager.set_tensor_parallel(False)
    assert not device_manager.is_tensor_parallel


def test_non_tp_memory_reporting_does_not_consult_process_group(monkeypatch):
    monkeypatch.setattr(device_manager, "_tensor_parallel", False)
    with (
        patch.object(dist, "is_initialized", side_effect=AssertionError("non-TP reporting is local")),
        patch.object(dist, "all_gather_object", side_effect=AssertionError("not TP")),
        patch.object(memory_monitor, "log_summary", return_value="summary") as log,
    ):
        assert device_manager.log_memory_summary("done", "warning") == "summary"
    log.assert_called_once_with("done", "warning")


@pytest.mark.parametrize("initialized", [False, True])
def test_tp_memory_reporting_without_multiple_ranks_stays_local(monkeypatch, initialized):
    monkeypatch.setattr(device_manager, "_tensor_parallel", True)
    monkeypatch.setattr(memory_monitor, "peak_vram", {"0": 2.5})
    with (
        patch.object(dist, "is_initialized", return_value=initialized),
        patch.object(dist, "get_world_size", return_value=1),
        patch.object(dist, "all_gather_object", side_effect=AssertionError("no multi-rank group")),
        patch.object(memory_monitor, "log_summary", return_value="summary") as log,
    ):
        assert device_manager.log_memory_summary("done", "warning") == "summary"
    log.assert_called_once_with("done", "warning", peak_vram={"0": 2.5})
    assert log.call_args.kwargs["peak_vram"] is not memory_monitor.peak_vram


def test_tp_memory_reporting_without_cuda_still_participates(monkeypatch):
    monkeypatch.setattr(device_manager, "_tensor_parallel", True)

    def gather(reports, local):
        assert local == {}
        reports[:] = [{}, {"0": 3.5}]

    with (
        patch.object(dist, "is_initialized", return_value=True),
        patch.object(dist, "get_world_size", return_value=2),
        patch.object(dist, "get_rank", return_value=0),
        patch("torch.cuda.is_available", return_value=False),
        patch.object(memory_monitor, "update_cpu") as update,
        patch.object(dist, "all_gather_object", side_effect=gather) as collective,
        patch.object(memory_monitor, "log_summary", return_value="summary") as log,
    ):
        assert device_manager.log_memory_summary() == "summary"
    update.assert_called_once_with()
    collective.assert_called_once()
    log.assert_called_once_with("", "info", peak_vram={"0": 3.5})


def _memory_reporting_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        device_manager.set_tensor_parallel(True)
        local_peaks = {"0": 2.5 + rank}
        memory_monitor.peak_vram = local_peaks.copy()
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.max_memory_reserved", return_value=local_peaks["0"] * 1024**3),
            patch.object(memory_monitor, "update"),
            patch.object(logger, "info") as info,
        ):
            for phase in ("cache", "block"):
                summary = device_manager.log_memory_summary(phase)
                assert memory_monitor.peak_vram == local_peaks
                if rank == 0:
                    assert "'peak_vram': {'0': 3.5GB}" in summary
                else:
                    assert summary is None
            assert info.call_count == (2 if rank == 0 else 0)
    finally:
        device_manager.set_tensor_parallel(False)
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires Gloo")
def test_two_rank_memory_reporting(tmp_path):
    torch.multiprocessing.spawn(
        _memory_reporting_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True
    )


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:1", "cpu", "xpu:0", "hpu:0", "cuda:0,cuda:1"])
def test_device_ids_are_not_parsed_as_module_patterns(device):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    with patch("auto_round.utils.device.logger.warning") as warning:
        set_non_auto_device_map(model, device)
    warning.assert_not_called()
    assert not hasattr(model[0], "tuning_device")


def test_explicit_module_maps_still_apply_and_invalid_patterns_warn():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    set_non_auto_device_map(model, {"0": "cuda:1"})
    assert model[0].tuning_device == "cuda:1"
    set_non_auto_device_map(model, "1:cuda:0")
    assert model[1].tuning_device == "cuda:0"
    with patch("auto_round.utils.device.logger.warning") as warning:
        set_non_auto_device_map(model, {"nonexistent_layer": "cuda:0"})
    warning.assert_called_once()
    assert "does not match any modules" in warning.call_args.args[0]


def test_non_tp_summary_retains_single_device_format(monkeypatch):
    monkeypatch.setattr(device_manager, "_tensor_parallel", False)
    monkeypatch.setattr(memory_monitor, "peak_vram", {"0": 2.5})
    assert "'peak_vram': 2.5GB" in memory_monitor.get_summary()


@pytest.mark.parametrize(
    "algorithm,options",
    [
        (None, {"iters": 2, "bits": 4, "group_size": 32, "sym": True}),
        ("rtn", {"disable_opt_rtn": True, "bits": 4, "group_size": 32, "sym": True}),
    ],
)
def test_supported_direct_options_do_not_emit_recommendation_warnings(algorithm, options):
    with patch("auto_round.algorithms.config_resolver.logger.warning") as warning:
        configs, runtime = _prepare_entry_kwargs(algorithm, options)
    warning.assert_not_called()
    assert runtime["scheme_overrides"] == {"bits": 4, "group_size": 32, "sym": True}
    assert not hasattr(configs[0], "bits")
