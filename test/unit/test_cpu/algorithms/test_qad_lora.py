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

"""Low-rank rounding keeps the FP grid and native TP transport unchanged."""

import copy
import sys
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from auto_round.algorithms.quantization.qad.config import QADConfig
from auto_round.compressors.utils import collect_best_params
from auto_round.data_type import reshape_pad_tensor_by_group_size
from auto_round.wrapper import WrapperLinear, wrapper_block


def _configure(layer, group_size=4, symmetric=True):
    for key, value in dict(bits=4, group_size=group_size, sym=symmetric, data_type="int", act_bits=16).items():
        setattr(layer, key, value)
    layer.scale_dtype = torch.float32
    return layer.requires_grad_(False)


@pytest.mark.parametrize("rank", [-1, True, 1.5, "2"])
def test_invalid_lora_rank(rank):
    with pytest.raises(ValueError, match="qad_lora_rank"):
        QADConfig(qad_lora_rank=rank)


@pytest.mark.parametrize("alpha", [0, -1, float("inf"), float("nan")])
def test_invalid_lora_alpha(alpha):
    with pytest.raises(ValueError, match="qad_lora_alpha"):
        QADConfig(qad_lora_rank=2, qad_lora_alpha=alpha)


def test_lora_cli_registration():
    from auto_round.cli.parser import build_quantize_parser

    args = build_quantize_parser().parse_args(
        ["local-model", "--algs", "qad", "--qad_lora_rank", "8", "--qad_lora_alpha", "16"]
    )
    assert args.qad_lora_rank == 8 and args.qad_lora_alpha == 16
    assert QADConfig().qad_lora_rank == 0


@pytest.mark.parametrize("group_size", [-1, 4])
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("conv1d", [False, True])
def test_lora_only_changes_rounding_and_restores_best_snapshot(group_size, symmetric, conv1d):
    from transformers.pytorch_utils import Conv1D

    torch.manual_seed(17)
    layer = _configure(Conv1D(16, 13) if conv1d else torch.nn.Linear(13, 16), group_size, symmetric)
    original_weight = layer.weight.detach().clone()
    dense = WrapperLinear(copy.deepcopy(layer), enable_torch_compile=False)
    low_rank = WrapperLinear(layer, enable_torch_compile=False, qad_lora_rank=2, qad_lora_alpha=4)
    assert "value" not in low_rank.params
    assert low_rank.value.numel() == 1 and not low_rank.value.requires_grad
    assert low_rank.lora_A.numel() + low_rank.lora_B.numel() == 2 * (13 + 16)
    assert sum(p.numel() for p in low_rank.params.values()) < sum(p.numel() for p in dense.params.values())
    x = torch.randn(2, 3, 13)
    torch.testing.assert_close(low_rank(x), dense(x))
    torch.testing.assert_close(low_rank.min_scale, dense.min_scale)
    torch.testing.assert_close(low_rank.max_scale, dense.max_scale)
    optimizer = torch.optim.AdamW(low_rank.params.values(), lr=0.01, weight_decay=0)
    for step in range(2):
        optimizer.zero_grad()
        low_rank(x).square().sum().backward()
        assert torch.isfinite(low_rank.lora_B.grad).all() and low_rank.lora_B.grad.abs().sum() > 0
        if step:
            assert torch.isfinite(low_rank.lora_A.grad).all() and low_rank.lora_A.grad.abs().sum() > 0
        optimizer.step()
    torch.testing.assert_close(layer.weight, original_weight)
    expected = low_rank(x).detach()
    snapshot = collect_best_params(torch.nn.Sequential(low_rank), "cpu")["0"]
    assert set(snapshot) == set(low_rank.params)
    with torch.no_grad():
        low_rank.lora_A.add_(1)
        low_rank.lora_B.add_(1)
    result = low_rank.unwrapper(snapshot)
    torch.testing.assert_close(result(x), expected)
    assert not any("lora" in name for name, _ in result.named_parameters())


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("next_rank", [0, 2])
def test_lora_preserves_frozen_warm_start(optimized, next_rank):
    from auto_round.algorithms.quantization.sign_roundv2.quantizer import SignRoundOptimizedWrapperLinear
    from auto_round.wrapper import store_warm_start_params_and_unwrap

    cls = SignRoundOptimizedWrapperLinear if optimized else WrapperLinear
    layer = _configure(torch.nn.Linear(16, 16))
    wrapper = cls(layer, enable_torch_compile=False)
    with torch.no_grad():
        wrapper.value.uniform_(-0.4, 0.4)
    x = torch.randn(2, 16)
    expected = wrapper(x).detach()
    block = torch.nn.Sequential(wrapper)
    best = collect_best_params(block, "cpu")
    store_warm_start_params_and_unwrap(block, best)
    wrapper_block(block, True, False, enable_torch_compile=False, qad_lora_rank=2)
    low_rank = block[0]
    assert isinstance(low_rank, cls)
    assert "value" not in low_rank.params
    torch.testing.assert_close(low_rank.value, best["0"]["value"])
    assert not low_rank.value.requires_grad
    torch.testing.assert_close(block(x), expected)
    with torch.no_grad():
        low_rank.lora_B.uniform_(-0.3, 0.3)
    expected = block(x).detach()
    best = collect_best_params(block, "cpu")
    effective_value = low_rank._rounding_value(best["0"]).detach().clone()
    store_warm_start_params_and_unwrap(block, best)
    assert "lora_A" not in block[0]._ar_warm_start
    torch.testing.assert_close(block[0]._ar_warm_start["value"], effective_value)
    wrapper_block(block, True, False, enable_torch_compile=False, qad_lora_rank=next_rank)
    torch.testing.assert_close(block(x), expected)


def test_lora_rounding_is_fp32_under_autocast():
    wrapper = WrapperLinear(_configure(torch.nn.Linear(16, 16)), enable_torch_compile=False, qad_lora_rank=2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        value = wrapper._rounding_value()
        assert value.dtype == torch.float32
        wrapper(torch.randn(2, 16)).float().square().sum().backward()
    assert wrapper.lora_B.grad.dtype == torch.float32
    assert torch.isfinite(wrapper.lora_B.grad).all()


def test_disabled_round_tuning_does_not_create_lora():
    wrapper = WrapperLinear(
        _configure(torch.nn.Linear(16, 16)), enable_torch_compile=False, enable_round_tuning=False, qad_lora_rank=2
    )
    assert not hasattr(wrapper, "lora_A") and "value" not in wrapper.params


@pytest.mark.parametrize("optimized", [False, True])
def test_disabled_rounding_preserves_warm_start_and_unwrap(optimized):
    from auto_round.algorithms.quantization.sign_roundv2.quantizer import SignRoundOptimizedWrapperLinear
    from auto_round.wrapper import store_warm_start_params_and_unwrap

    torch.manual_seed(31)
    layer = _configure(torch.nn.Linear(16, 16))
    cls = SignRoundOptimizedWrapperLinear if optimized else WrapperLinear
    block = torch.nn.Sequential(cls(layer, enable_torch_compile=False))
    with torch.no_grad():
        block[0].value.uniform_(-0.4, 0.4)
    x = torch.randn(2, 16)
    expected = block(x).detach()
    best = collect_best_params(block, "cpu")
    store_warm_start_params_and_unwrap(block, best)
    wrapper_block(block, True, False, enable_torch_compile=False, enable_round_tuning=False, lora_rank=2)
    frozen = block[0]
    assert "value" in frozen._buffers and "value" not in frozen.params
    assert not hasattr(frozen, "lora_A")
    torch.testing.assert_close(frozen.value, best["0"]["value"])
    torch.testing.assert_close(block(x), expected)
    frozen_best = collect_best_params(block, "cpu")
    store_warm_start_params_and_unwrap(block, frozen_best)
    torch.testing.assert_close(layer._ar_warm_start["value"], best["0"]["value"])
    torch.testing.assert_close(frozen.unwrapper(frozen_best["0"])(x), expected)


def _tp_lora_worker(rank, rendezvous):
    from torch.distributed.tensor import DTensor
    from transformers.distributed.tensor_parallel import ColwiseParallel, RowwiseParallel

    from auto_round.parallel.qdq import TensorParallelQDQLinear

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        mesh = dist.init_device_mesh("cpu", (2,))
        torch.manual_seed(9)
        col = _configure(torch.nn.Linear(16, 16))
        row = _configure(torch.nn.Linear(16, 8))
        references = [WrapperLinear(copy.deepcopy(layer), enable_torch_compile=False) for layer in (col, row)]
        wrappers = []
        for layer, style, reference in zip((col, row), (ColwiseParallel(), RowwiseParallel()), references):
            style.shard_param(layer, "weight", mesh)
            style.shard_param(layer, "bias", mesh)
            style.install_forward(layer, mesh)
            wrapper = TensorParallelQDQLinear(layer, enable_torch_compile=False, qad_lora_rank=2)
            with torch.no_grad():
                wrapper.lora_B.uniform_(-0.2, 0.2)
            assert wrapper.lora_A.shape[1] == layer.weight.to_local().shape[1]
            assert wrapper.lora_B.shape[0] == layer.weight.to_local().shape[0]
            # Gathering is permitted only to construct the dense test oracle.
            full_delta = wrapper._distributed_weight(wrapper.lora_B @ wrapper.lora_A).full_tensor()
            grouped, _, _ = reshape_pad_tensor_by_group_size(full_delta.detach(), layer.group_size)
            reference.value.data.copy_(grouped)
            wrappers.append(wrapper)
        x = torch.randn(2, 3, 16)
        actual_x, expected_x = x.clone().requires_grad_(), x.clone().requires_grad_()
        with patch.object(DTensor, "full_tensor", side_effect=AssertionError("LoRA must not gather weights")):
            actual = wrappers[1](wrappers[0](actual_x))
            actual.square().mean().backward()
        expected = references[1](references[0](expected_x))
        expected.square().mean().backward()
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(actual_x.grad, expected_x.grad, atol=2e-5, rtol=2e-4)
        for axis, wrapper, reference in zip((0, 1), wrappers, references):
            dense_grad = reference.value.grad.reshape(reference.weight.shape).chunk(2, dim=axis)[rank]
            torch.testing.assert_close(
                wrapper.lora_A.grad, wrapper.lora_B.detach().t() @ dense_grad, atol=3e-6, rtol=3e-5
            )
            torch.testing.assert_close(
                wrapper.lora_B.grad, dense_grad @ wrapper.lora_A.detach().t(), atol=3e-6, rtol=3e-5
            )
            best = {key: value.detach().clone() for key, value in wrapper.params.items()}
            ref_best = {key: value.detach().clone() for key, value in reference.params.items()}
            ref_result = reference.unwrapper(ref_best)
            with patch.object(DTensor, "full_tensor", side_effect=AssertionError("Unwrap must stay shard-local")):
                result = wrapper.unwrapper(best)
            assert isinstance(result.weight, DTensor)
            torch.testing.assert_close(result.weight.full_tensor(), ref_result.weight)
            torch.testing.assert_close(result.scale.full_tensor(), ref_result.scale)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires distributed Gloo")
def test_two_rank_lora_rounding(tmp_path):
    pytest.importorskip("transformers.distributed.tensor_parallel")
    with patch.object(sys, "path", [entry for entry in sys.path if Path(entry).is_absolute()]):
        context = mp.spawn(_tp_lora_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=False)
    try:
        import time

        deadline = time.monotonic() + 120
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("LoRA TP workers exceeded the test timeout.")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
