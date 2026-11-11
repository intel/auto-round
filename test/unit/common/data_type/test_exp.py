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

from auto_round.data_type.exp import quant_tensor_exp_nvint4, quant_tensor_exp_nvint4_zp, quant_tensor_exp_nvsint4
from auto_round.data_type.utils import (
    get_optimized_quant_func,
    get_quant_func,
    resolve_optimized_init_scale_fn,
    resolve_optimized_init_scale_zero_fn,
)
from auto_round.schemes import preset_name_to_scheme


@pytest.mark.parametrize(
    ("name", "expected_dtype", "expected_sym"),
    [
        ("EXP-FP4", "nvfp4_v2", True),
        ("EXP-NVSINT4", "exp_nvsint4", True),
        ("EXP-NVINT4", "exp_nvint4", True),
        ("EXP-NVINT4-ZP", "exp_nvint4_zp", False),
    ],
)
def test_exp_scheme_presets(name, expected_dtype, expected_sym):
    scheme = preset_name_to_scheme(name)
    assert scheme.bits == 4
    assert scheme.group_size == 16
    assert scheme.data_type == expected_dtype
    assert scheme.sym is expected_sym


@pytest.mark.parametrize(
    ("data_type", "quant_func"),
    [
        ("exp_nvsint4", quant_tensor_exp_nvsint4),
        ("exp_nvint4", quant_tensor_exp_nvint4),
        ("exp_nvint4_zp", quant_tensor_exp_nvint4_zp),
    ],
)
def test_exp_int4_qdq_scale_format_and_gradient(data_type, quant_func):
    torch.manual_seed(7)
    tensor = torch.randn(4, 16, requires_grad=True)
    scale_coeff = torch.ones(4, requires_grad=True)
    kwargs = {}
    kwargs["init_scale"] = torch.ones(4, 1)
    if data_type == "exp_nvint4_zp":
        kwargs["init_zp"] = torch.full((4, 1), 8.0)

    qdq, scale, zero = quant_func(
        tensor,
        bits=4,
        group_size=16,
        max_scale=scale_coeff,
        **kwargs,
    )

    assert qdq.shape == tensor.shape
    assert scale.shape == (4, 1)
    assert torch.isfinite(qdq).all()
    assert torch.isfinite(scale).all()
    if data_type == "exp_nvint4_zp":
        assert zero.shape == (4, 1)
    else:
        assert zero == 8
    if data_type == "exp_nvsint4":
        assert torch.equal(scale, scale.to(torch.float8_e4m3fn).float())
    else:
        from auto_round.data_type.nvfp import cast_to_ue5m3

        assert torch.equal(scale, cast_to_ue5m3(scale))

    qdq.square().mean().backward()
    assert scale_coeff.grad is not None
    assert torch.isfinite(scale_coeff.grad).all()
    assert torch.count_nonzero(scale_coeff.grad) > 0


@pytest.mark.parametrize(
    ("data_type", "quant_func"),
    [
        ("exp_nvsint4", quant_tensor_exp_nvsint4),
        ("exp_nvint4", quant_tensor_exp_nvint4),
        ("exp_nvint4_zp", quant_tensor_exp_nvint4_zp),
    ],
)
def test_exp_int4_rtn_dispatch_and_group_size(data_type, quant_func):
    selected, selected_name = get_quant_func(data_type, bits=4, sym=data_type != "exp_nvint4_zp", iters=1)
    assert selected is quant_func
    assert selected_name == data_type

    with pytest.raises(ValueError, match="group_size=16"):
        quant_func(torch.randn(2, 32), bits=4, group_size=32)


def test_exp_int4_optimized_tuning_dispatch():
    assert resolve_optimized_init_scale_fn("exp_nvsint4") is not None
    assert get_optimized_quant_func("exp_nvsint4") is quant_tensor_exp_nvsint4
    assert resolve_optimized_init_scale_fn("exp_nvint4") is not None
    assert get_optimized_quant_func("exp_nvint4") is quant_tensor_exp_nvint4
    assert resolve_optimized_init_scale_zero_fn("exp_nvint4_zp") is not None
    assert get_optimized_quant_func("exp_nvint4_zp") is quant_tensor_exp_nvint4_zp


def test_exp_nvint4_zp_opt_rtn_does_not_use_neuqi(monkeypatch):
    from auto_round.data_type.exp import search_exp_nvint4_zp_scale_zero

    def fail_if_called(*args, **kwargs):
        raise AssertionError("NeUQI must remain opt-in for exp_nvint4_zp")

    monkeypatch.setattr("auto_round.data_type.neuqi.neuqi_search_scale_zero", fail_if_called)
    weight = torch.randn(4, 16)
    scale, zp = search_exp_nvint4_zp_scale_zero(weight, bits=4)

    assert scale.shape == (4, 1)
    assert zp.shape == (4, 1)
    assert torch.isfinite(scale).all()
    assert ((zp >= 0) & (zp <= 15)).all()


def test_exp_nvint4_opt_rtn_accepts_wrapper_kwargs():
    from auto_round.data_type.exp import opt_rtn_exp_nvint4

    qdq, scale, zero = opt_rtn_exp_nvint4(
        torch.randn(4, 16),
        bits=4,
        group_size=16,
        init_scale=None,
        min_scale=torch.tensor(1.0),
        max_scale=torch.tensor(1.0),
    )

    assert qdq.shape == (4, 16)
    assert scale.shape == (4, 1)
    assert (scale > 1e-3).all()
    assert zero == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_exp_nvint4_opt_rtn_moves_wrapper_scale_to_weight_device():
    from auto_round.data_type.exp import opt_rtn_exp_nvint4

    qdq, scale, zero = opt_rtn_exp_nvint4(
        torch.randn(4, 16, device="cuda"),
        bits=4,
        group_size=16,
        init_scale=None,
        min_scale=torch.tensor(1.0),
        max_scale=torch.tensor(1.0),
    )

    assert qdq.is_cuda and scale.is_cuda
    assert (scale > 1e-3).all()
    assert zero == 8


@pytest.mark.parametrize(
    ("data_type", "sym", "rtn_name", "opt_rtn_name", "quant_func"),
    [
        ("nvfp4_v2", True, "rtn_nvfp4_v2", "opt_rtn_nvfp4_v2", None),
        ("exp_nvsint4", True, "rtn_exp_nvsint4", "opt_rtn_exp_nvsint4", quant_tensor_exp_nvsint4),
        ("exp_nvint4", True, "rtn_exp_nvint4", "opt_rtn_exp_nvint4", quant_tensor_exp_nvint4),
        ("exp_nvint4_zp", False, "rtn_exp_nvint4_zp", "opt_rtn_exp_nvint4_zp", quant_tensor_exp_nvint4_zp),
    ],
)
def test_exp_schemes_support_rtn_opt_rtn_and_tuning(data_type, sym, rtn_name, opt_rtn_name, quant_func):
    from auto_round.data_type import QUANT_FUNC_WITH_DTYPE
    from auto_round.data_type.nvfp import nvfp4_v2

    expected_quant_func = quant_func or nvfp4_v2
    weight = torch.randn(4, 16)

    rtn_func, rtn_selected = get_quant_func(data_type, bits=4, sym=sym, disable_opt_rtn=True, group_size=16, iters=0)
    assert rtn_selected == rtn_name
    assert rtn_func is QUANT_FUNC_WITH_DTYPE[rtn_name]

    opt_rtn_func, opt_rtn_selected = get_quant_func(
        data_type, bits=4, sym=sym, disable_opt_rtn=False, group_size=16, iters=0
    )
    assert opt_rtn_selected == opt_rtn_name
    assert opt_rtn_func is QUANT_FUNC_WITH_DTYPE[opt_rtn_name]

    tuned_func, tuned_selected = get_quant_func(
        data_type, bits=4, sym=sym, disable_opt_rtn=False, group_size=16, iters=1
    )
    assert tuned_selected == data_type
    assert tuned_func is expected_quant_func

    for selected_func in (rtn_func, opt_rtn_func, tuned_func):
        qdq, scale, _ = selected_func(weight, bits=4, group_size=16)
        assert qdq.shape == weight.shape
        assert torch.isfinite(qdq).all()
        assert torch.isfinite(scale).all()
