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

import logging
from io import StringIO
from unittest.mock import patch

import pytest
import torch.distributed as dist

from auto_round import AutoRound, ParallelConfig
from auto_round.cli.main import _build_entry_base_kwargs
from auto_round.cli.parser import build_quantize_parser
from auto_round.compressors.base import BaseOrchestrator
from auto_round.logger import logger


@pytest.mark.parametrize(
    "world_size,options,expected",
    [
        (None, [], (1, 1, 1)),
        ("1", [], (1, 1, 1)),
        ("2", [], (2, 1, 1)),
        ("8", [], (8, 1, 1)),
        ("2", ["--tp_size", "1"], (1, 1, 1)),
        ("2", ["--tp_size", "2"], (2, 1, 1)),
        ("2", ["--pp_size", "2"], (1, 2, 1)),
        ("2", ["--dp_size", "2"], (1, 1, 2)),
    ],
)
def test_cli_infers_tp_from_world_size(monkeypatch, world_size, options, expected):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    if world_size is not None:
        monkeypatch.setenv("WORLD_SIZE", world_size)
    # Visible GPU count is not the number of distributed participants.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    args = build_quantize_parser().parse_args(["local-model", "--alg", "qad", *options])
    kwargs = _build_entry_base_kwargs(args, low_cpu_mem_usage=True, enable_torch_compile=None, layer_config={})
    assert kwargs["parallel_config"] == dict(zip(("tp_size", "pp_size", "dp_size"), expected))


@pytest.mark.parametrize("rank", [None, "0", "1", "3"])
def test_routine_logs_only_on_global_rank_zero(monkeypatch, rank):
    monkeypatch.delenv("RANK", raising=False)
    if rank is not None:
        monkeypatch.setenv("RANK", rank)
    # LOCAL_RANK=0 on another node is not global rank zero.
    monkeypatch.setenv("LOCAL_RANK", "0")
    output = StringIO()
    handler = logging.StreamHandler(output)
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        with patch.object(dist, "is_initialized", return_value=False):
            logger.info("routine info")
            logger.warning("routine warning")
            logger.error("rank failure")
            logger.critical("critical failure")
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
    logs = output.getvalue()
    assert ("routine info" in logs) == (rank in (None, "0"))
    assert ("routine warning" in logs) == (rank in (None, "0"))
    assert "rank failure" in logs
    assert "critical failure" in logs


@pytest.mark.parametrize("rank", [0, 1])
def test_logger_uses_initialized_global_rank(monkeypatch, rank):
    from auto_round.logger import RankZeroFilter

    monkeypatch.delenv("RANK", raising=False)
    record = logging.LogRecord("autoround", logging.INFO, __file__, 1, "routine", (), None)
    with (
        patch.object(dist, "is_available", return_value=True),
        patch.object(dist, "is_initialized", return_value=True),
        patch.object(dist, "get_rank", return_value=rank),
    ):
        assert RankZeroFilter().filter(record) == (rank == 0)


@pytest.mark.parametrize("requested", [False, True])
def test_tp_compile_reason_survives_finalization_without_enable_hint(requested):
    compressor = object.__new__(BaseOrchestrator)
    compressor.parallel_config = ParallelConfig(tp_size=2)
    compressor._torch_compile_default_off_reason = "the user disabled it"
    compressor._precheck_torch_compile(requested)
    assert compressor.enable_torch_compile is False
    with (
        patch.object(compressor, "_get_torch_compile_guard_state", return_value=(False, False)),
        patch("auto_round.compressors.base.logger.info") as info,
    ):
        compressor._finalize_torch_compile()
    info.assert_called_once_with(
        "`torch.compile` is disabled, as %s", "native TP QDQ does not support torch.compile yet"
    )
    assert compressor.enable_torch_compile is False


def test_tp_explicit_compile_request_fails_before_model_loading():
    with (
        patch.object(ParallelConfig, "resolve_device_map", return_value="cpu"),
        pytest.raises(NotImplementedError, match="Native TP QDQ does not support torch.compile"),
    ):
        AutoRound("local-model", alg_configs="qad", parallel_config={"tp_size": 2}, enable_torch_compile=True)


def test_non_tp_compile_can_remain_enabled():
    compressor = object.__new__(BaseOrchestrator)
    compressor.parallel_config = ParallelConfig()
    with (
        patch.object(compressor, "_get_torch_compile_guard_state", return_value=(False, False)),
        patch.object(compressor, "_torch_compile_unsupported_arch_reason", return_value=None),
        patch.object(compressor, "_torch_compile_disabled_reason", return_value=None),
    ):
        compressor._precheck_torch_compile(True)
    assert compressor.enable_torch_compile is True
