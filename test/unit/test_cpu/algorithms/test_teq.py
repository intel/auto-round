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

"""CPU tests for TEQ transform integration."""

import pytest
import torch

from auto_round import AutoRound, SignRoundConfig, TEQConfig
from auto_round.algorithms.registry import resolve_alg_config, resolve_algorithm_names
from auto_round.algorithms.transforms.teq.base import _TEQLinearFakeQuant
from auto_round.data_type.utils import compute_optimized_init_scale, get_quant_func
from auto_round.wrapper import WrapperLinear


def test_teq_algorithm_registry():
    assert resolve_algorithm_names("teq,auto_round") == ["teq", "auto_round"]
    assert isinstance(resolve_alg_config("teq"), TEQConfig)


def test_teq_config_optional_awq_init_and_replay_limits():
    config = TEQConfig(awq_init=True, awq_init_n_grid=8, nsamples=2, batch_size=1, sample_seqlen=16, skip_moe=False)

    assert config.teq_awq_init is True
    assert config.teq_awq_init_n_grid == 8
    assert config.teq_nsamples == 2
    assert config.teq_batch_size == 1
    assert config.teq_sample_seqlen == 16
    assert config.teq_skip_moe is False


def test_teq_research_modes_require_explicit_opt_in():
    with pytest.raises(ValueError, match="experimental=True"):
        TEQConfig(optimization_mode="joint")

    config = TEQConfig(optimization_mode="joint", experimental=True, refine_iters=3, joint_lr=2e-4)
    assert config.teq_optimization_mode == "joint"
    assert config.teq_refine_iters == 3
    assert config.teq_joint_lr == 2e-4


@pytest.mark.parametrize("mode", ["joint", "frozen_ar_refine"])
def test_teq_research_mode_wrapper_is_identity_and_trainable(mode):
    layer = torch.nn.Linear(5, 4, bias=True)
    layer.bits = 16
    layer.group_size = -1
    layer.data_type = "int"
    layer.sym = True
    layer.act_bits = 16
    layer.scale_dtype = torch.float32
    log_alpha = torch.nn.Parameter(torch.zeros(5))
    context = {
        "log_alpha": log_alpha,
        "min_log": -10.0,
        "max_log": 10.0,
        "apply_input_transform": True,
        "wrappers": [],
    }
    layer._teq_input_context = context
    wrapper = WrapperLinear(layer, enable_torch_compile=False, device="cpu")
    inputs = torch.randn(3, 5)

    torch.testing.assert_close(wrapper(inputs), layer(inputs))
    wrapper(inputs).square().mean().backward()
    assert torch.isfinite(log_alpha.grad).all()
    assert wrapper.params["teq_log_alpha"] is log_alpha


@pytest.mark.parametrize("data_type", ["int", "mx_fp"])
def test_teq_qdq_matches_common_w4a16_schemes(data_type):
    torch.manual_seed(1)
    layer = torch.nn.Linear(35, 6, bias=False)
    params = {"bits": 4, "group_size": 32, "data_type": data_type, "sym": True}
    quant_func, _ = get_quant_func(data_type, 4, True, disable_opt_rtn=True, group_size=32, iters=1)
    log_alpha = torch.nn.Parameter(torch.linspace(-0.2, 0.2, 35))
    wrapper = _TEQLinearFakeQuant(
        layer,
        log_alpha,
        params,
        quant_func,
        min_scale=1e-5,
        max_scale=10.0,
    )
    transformed = layer.weight * log_alpha.exp().view(1, -1)
    expected, _, _ = quant_func(transformed, **params)

    torch.testing.assert_close(wrapper._qdq_weight(transformed), expected)


def test_teq_mxfp_init_scale_handles_group_padding_and_channel_importance():
    weight = torch.randn(7, 35)
    imatrix = torch.linspace(0.5, 1.5, 35)
    init_scale = compute_optimized_init_scale(weight, "mx_fp", 4, 32, imatrix=imatrix)

    assert init_scale.numel() == 14
    assert torch.isfinite(init_scale).all()


def test_teq_log_scale_preserves_fp_equivalence_and_gradient():
    layer = torch.nn.Linear(4, 3, bias=True)
    log_alpha = torch.nn.Parameter(torch.tensor([0.2, -0.1, 0.4, -0.3]))

    def identity_quant(tensor, **kwargs):
        return tensor, None, None

    wrapper = _TEQLinearFakeQuant(
        layer,
        log_alpha,
        {"bits": 16, "group_size": -1, "data_type": "int", "sym": True},
        identity_quant,
        min_scale=1e-5,
        max_scale=10.0,
    )
    inputs = torch.randn(2, 5, 4)
    expected = layer(inputs)
    actual = wrapper(inputs)

    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert log_alpha.grad is not None
    assert torch.isfinite(log_alpha.grad).all()


def test_teq_static_activation_training_is_rejected():
    config = TEQConfig(bits=4, group_size=32, data_type="int", act_bits=8, act_data_type="int", act_dynamic=False)
    from auto_round.algorithms.transforms.teq.base import TEQTransform

    transform = TEQTransform(config)
    layer = torch.nn.Linear(4, 3, bias=False)
    layer.global_name = "layer"

    with pytest.raises(NotImplementedError, match="static activation"):
        transform._make_wrapper(layer, torch.nn.Parameter(torch.zeros(4)))


def test_teq_mxfp4_signroundv2_smoke(tiny_opt_model_path):
    ar = AutoRound(
        tiny_opt_model_path,
        scheme="MXFP4",
        alg_configs=[
            TEQConfig(iters=1, awq_init=True, awq_init_n_grid=2),
            SignRoundConfig(iters=1, enable_alg_ext=True),
        ],
        nsamples=1,
        seqlen=8,
        dataset=["local TEQ calibration sample with enough tokens for quantization"],
        low_cpu_mem_usage=False,
    )

    model, layer_config = ar.quantize()

    assert [type(pre).__name__ for pre in ar.alg_composer.preprocessors] == ["TEQTransform"]
    assert type(ar.alg_composer.block_quantizer).__name__ == "SignRoundV2Quantizer"
    assert model is not None
    assert layer_config
    assert all(cfg["data_type"] == "mx_fp" for cfg in layer_config.values() if cfg["bits"] == 4)


def test_teq_experimental_refinement_smoke(tiny_opt_model_path):
    ar = AutoRound(
        tiny_opt_model_path,
        bits=4,
        group_size=32,
        alg_configs=[
            TEQConfig(optimization_mode="joint", experimental=True, refine_iters=1, joint_lr=1e-4),
            SignRoundConfig(iters=1, enable_alg_ext=True),
        ],
        nsamples=1,
        seqlen=8,
        dataset=["local TEQ refinement calibration sample with enough tokens for quantization"],
        low_cpu_mem_usage=False,
    )

    model, layer_config = ar.quantize()

    assert model is not None
    assert layer_config
    assert not any(hasattr(module, "_teq_input_context") for module in model.modules())
    result = ar.alg_composer.block_quantizer.teq_last_refinement
    assert result["selected_loss"] <= result["baseline_loss"]
