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

import pytest
import torch

import auto_round.algorithms.transforms.svdquant.residual as residual_module
from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig
from auto_round.algorithms.transforms.svdquant.residual import (
    ResidualQuantScheme,
    iterate_residual_decomposition,
    rtn_qdq_residual,
    truncated_svd,
)
from auto_round.algorithms.transforms.svdquant.wrapper import SVDQuantLinear
from auto_round.data_type.mxfp import quant_mx_rceil


def test_rtn_qdq_residual_matches_deployable_mxfp4_quantizer():
    weight = torch.linspace(-3.0, 3.0, steps=3 * 64, dtype=torch.float32).reshape(3, 64)
    scheme = ResidualQuantScheme(data_type="mx_fp4e2m1", bits=4, group_size=32, sym=True)

    actual = rtn_qdq_residual(weight, scheme)
    expected, _, _ = quant_mx_rceil(
        tensor=weight,
        bits=4,
        group_size=32,
        data_type="mx_fp4e2m1",
    )

    torch.testing.assert_close(actual, expected)
    assert actual.shape == weight.shape
    assert actual.dtype == weight.dtype
    assert actual.device == weight.device


def test_svdquant_linear_combines_residual_and_low_rank_branches_after_smoothing():
    residual = torch.nn.Linear(4, 3, bias=True)
    lora_down = torch.nn.Linear(4, 2, bias=False)
    lora_up = torch.nn.Linear(2, 3, bias=False)
    smooth = torch.tensor([0.5, 1.0, 2.0, 4.0])
    wrapper = SVDQuantLinear(residual, lora_down, lora_up, smooth)
    inputs = torch.randn(2, 4)

    smoothed = inputs * smooth
    expected = residual(smoothed) + lora_up(lora_down(smoothed))

    torch.testing.assert_close(wrapper(inputs), expected)


def test_svdquant_config_defaults_to_data_free_single_iteration():
    config = SVDQuantConfig()

    assert config.rank == 32
    assert config.smooth_enabled is False
    assert config.smooth_max_calibration_calls == 128
    assert config.residual_iters == 1
    assert config.residual_early_stop is False
    assert config.need_calib is False


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"rank": -1}, "rank"),
        ({"smooth_enabled": 1}, "smooth_enabled"),
        ({"smooth_num_grids": 1}, "smooth_num_grids"),
        ({"smooth_max_calibration_calls": 0}, "smooth_max_calibration_calls"),
        ({"low_rank_dtype": "bf116"}, "low_rank_dtype"),
        ({"residual_iters": 0}, "residual_iters"),
        ({"residual_quant_method": "signround"}, "residual_quant_method"),
    ],
)
def test_svdquant_config_rejects_invalid_structural_options(kwargs, field):
    with pytest.raises(ValueError, match=field):
        SVDQuantConfig(**kwargs)


@pytest.mark.parametrize("dtype", ["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
def test_svdquant_config_accepts_supported_low_rank_dtype_aliases(dtype):
    assert SVDQuantConfig(low_rank_dtype=dtype).low_rank_dtype == dtype


def test_truncated_svd_returns_shared_down_factor_for_stacked_projection_group():
    torch.manual_seed(0)
    qkv = torch.randn(12, 8, dtype=torch.float32)

    low_rank, down, up = truncated_svd(qkv, rank=3)

    assert low_rank.shape == qkv.shape
    assert down.shape == (3, 8)
    assert up.shape == (12, 3)
    torch.testing.assert_close(low_rank, up @ down)
    q_up, k_up, v_up = up.split((4, 4, 4), dim=0)
    torch.testing.assert_close(torch.cat((q_up @ down, k_up @ down, v_up @ down)), low_rank)


def test_residual_iteration_keeps_the_best_materialized_candidate():
    torch.manual_seed(1)
    weight = torch.randn(8, 32, dtype=torch.float32)
    scheme = ResidualQuantScheme(data_type="mx_fp4e2m1", bits=4, group_size=32, sym=True)

    first_low_rank, first_down, first_up = truncated_svd(weight, rank=2)
    first_qdq = rtn_qdq_residual(weight - first_low_rank, scheme)
    first_error = torch.sum((weight - (first_qdq + first_up @ first_down)).square()).item()

    result = iterate_residual_decomposition(
        weight,
        rank=2,
        scheme=scheme,
        iterations=4,
        early_stop=False,
        residual_dtype=torch.float32,
        low_rank_dtype=torch.float32,
    )

    assert 1 <= result.selected_iteration <= 4
    assert result.error <= first_error
    assert result.residual.shape == weight.shape
    torch.testing.assert_close(result.residual + result.up @ result.down, weight)


def test_residual_iteration_early_stops_after_candidate_worsens(monkeypatch):
    torch.manual_seed(2)
    weight = torch.randn(4, 32, dtype=torch.float32)
    scheme = ResidualQuantScheme(data_type="mx_fp4e2m1", bits=4, group_size=32, sym=True)
    calls = 0

    def qdq_then_degrade(residual, _scheme):
        nonlocal calls
        calls += 1
        return residual if calls == 1 else residual + 1

    monkeypatch.setattr(residual_module, "rtn_qdq_residual", qdq_then_degrade)

    result = iterate_residual_decomposition(
        weight,
        rank=2,
        scheme=scheme,
        iterations=10,
        early_stop=True,
        residual_dtype=torch.float32,
        low_rank_dtype=torch.float32,
    )

    assert calls == 2
    assert result.selected_iteration == 1
