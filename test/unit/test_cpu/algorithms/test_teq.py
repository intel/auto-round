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
import torch.nn as nn

from auto_round import AutoRound, AWQConfig, SignRoundConfig, TEQConfig
from auto_round.algorithms.registry import resolve_alg_config, resolve_algorithm_names
from auto_round.algorithms.transforms.hadamard.transforms import build_hadamard_transform
from auto_round.algorithms.transforms.teq.base import TEQTransform, _TEQLinearFakeQuant
from auto_round.algorithms.transforms.teq.protocol import (
    TEQ_EQUIVALENT_TRANSFORMS_ATTR,
    install_teq_equivalent_input_hook,
    set_teq_equivalent_transforms,
)
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


def test_teq_config_accepts_lion_for_pre_ar_optimization():
    config = TEQConfig(optimizer="lion")
    assert config.teq_optimizer == "lion"

    with pytest.raises(ValueError, match="optimizer"):
        TEQConfig(optimizer="sgd")


def test_teq_sqrt_weight_init_is_geometrically_normalized_before_clamping():
    layer = torch.nn.Linear(8, 4, bias=False)
    with torch.no_grad():
        channel_scale = torch.logspace(-3, 1, layer.in_features)
        layer.weight.copy_(channel_scale.unsqueeze(0).expand_as(layer.weight))

    transform = TEQTransform(TEQConfig(sqrt_w_init=True, min_scale=0.05, max_scale=3.0))
    alpha = transform._initial_alpha([layer])

    assert torch.isfinite(alpha).all()
    assert alpha.min() >= 0.05
    assert alpha.max() <= 3.0
    assert (alpha == 3.0).float().mean() < 0.5
    # The raw inverse-sqrt values have geometric mean 10. Normalization keeps
    # the non-saturated center near one instead of clipping every channel high.
    assert alpha.median() < 1.0
    assert torch.count_nonzero((alpha > 0.5) & (alpha < 2.0)) >= 2


def test_teq_research_modes_require_explicit_opt_in():
    with pytest.raises(ValueError, match="experimental=True"):
        TEQConfig(optimization_mode="joint")

    config = TEQConfig(optimization_mode="joint", experimental=True, refine_iters=3, joint_lr=2e-4)
    assert config.teq_optimization_mode == "joint"
    assert config.teq_refine_iters == 3
    assert config.teq_joint_lr == 2e-4


def test_teq_staged_refine_requires_a_valid_ar_split():
    config = TEQConfig(optimization_mode="staged_refine", experimental=True, ar_first_iters=100)
    assert config.teq_ar_first_iters == 100

    with pytest.raises(ValueError, match="ar_first_iters"):
        TEQConfig(optimization_mode="staged_refine", experimental=True, ar_first_iters=0)


def test_teq_staged_refine_accepts_lion_optimizer():
    config = TEQConfig(
        optimization_mode="staged_refine", experimental=True, ar_first_iters=100, refine_optimizer="lion"
    )
    assert config.teq_refine_optimizer == "lion"


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


@pytest.mark.parametrize("hadamard_type", ["hadamard", "random_hadamard"])
def test_teq_fake_qdq_preserves_rotation_equivalence(hadamard_type):
    """TEQ must train in the same rotated basis that final AR wrappers use."""
    torch.manual_seed(3)
    # Use the production MXFP4 rotation block size.  RandomHadamardTransform
    # deliberately defaults to 32 as well.
    layer = torch.nn.Linear(32, 3, bias=True)
    log_alpha = torch.nn.Parameter(torch.linspace(-0.3, 0.4, 32))
    weight_transform = build_hadamard_transform(hadamard_type, block_size=32, location="weight")
    input_transform = build_hadamard_transform(hadamard_type, block_size=32, location="input", inverse=True)
    # Match production metadata storage: it is intentionally not a registered
    # child module or state-dict entry.
    set_teq_equivalent_transforms(layer, weight=weight_transform, input=input_transform)

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
    inputs = torch.randn(2, 5, 32)

    # A non-uniform alpha is intentional: it catches the invalid order
    # ``Hadamard(x) / alpha`` / ``Hadamard(W) * alpha``.  Per-channel alpha
    # must be applied before each side of the equivalent rotation.
    torch.testing.assert_close(wrapper(inputs), layer(inputs), rtol=1e-5, atol=1e-6)
    assert f"{TEQ_EQUIVALENT_TRANSFORMS_ATTR}.weight" not in layer.state_dict()


def test_teq_fake_qdq_replays_original_hooks_when_replaced():
    """Online rotations attached to the original layer cannot disappear in TEQ."""
    layer = torch.nn.Linear(4, 3, bias=False)
    layer.register_forward_pre_hook(lambda _, args: (args[0] * 2.0,))
    layer.register_forward_hook(lambda _, _args, output: output + 1.0)
    log_alpha = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.2, -0.1]))

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
    inputs = torch.randn(3, 4)

    torch.testing.assert_close(wrapper(inputs), layer(inputs))


def test_ar_wrapper_preserves_rotation_teq_order():
    """Final AR wrapper must use the same non-commuting order as fake TEQ."""
    torch.manual_seed(4)
    layer = torch.nn.Linear(32, 3, bias=True)
    layer.bits = 16
    layer.group_size = -1
    layer.data_type = "int"
    layer.sym = True
    layer.act_bits = 16
    layer.scale_dtype = torch.float32
    context = {
        "log_alpha": torch.nn.Parameter(torch.linspace(-0.2, 0.3, 32)),
        "min_log": -10.0,
        "max_log": 10.0,
        "apply_input_transform": True,
        "wrappers": [],
    }
    layer._teq_input_context = context
    set_teq_equivalent_transforms(
        layer,
        weight=build_hadamard_transform("hadamard", block_size=32, location="weight"),
        input=build_hadamard_transform("hadamard", block_size=32, location="input", inverse=True),
    )
    wrapper = WrapperLinear(layer, enable_torch_compile=False, device="cpu")
    inputs = torch.randn(2, 32)

    torch.testing.assert_close(wrapper(inputs), layer(inputs), rtol=1e-5, atol=1e-6)


def test_committed_rotation_weight_installs_exactly_one_input_hook():
    """A committed rotated QDQ weight must stay paired in later block forwards."""
    torch.manual_seed(5)
    layer = torch.nn.Linear(32, 4, bias=True)
    layer.bits = 16
    layer.group_size = -1
    layer.data_type = "int"
    layer.sym = True
    layer.act_bits = 16
    layer.scale_dtype = torch.float32
    set_teq_equivalent_transforms(
        layer,
        weight=build_hadamard_transform("hadamard", block_size=32, location="weight"),
        input=build_hadamard_transform("hadamard", block_size=32, location="input", inverse=True),
    )
    original_weight = layer.weight.detach().clone()
    inputs = torch.randn(3, 32)
    expected = torch.nn.functional.linear(inputs, original_weight, layer.bias)
    wrapper = WrapperLinear(layer, enable_torch_compile=False, device="cpu")
    committed = wrapper.unwrapper({})
    assert len(committed._forward_pre_hooks) == 1
    torch.testing.assert_close(committed(inputs), expected, rtol=1e-5, atol=1e-6)
    install_teq_equivalent_input_hook(committed)
    assert len(committed._forward_pre_hooks) == 1


def test_awq_grid_search_uses_rotation_aware_candidate_path():
    """AWQ must score ``QDQ(rotate(W * s))`` with ``rotate(x / s)``."""
    from auto_round.algorithms.transforms.awq.base import AWQTransform
    from auto_round.algorithms.transforms.awq.mappings import ResolvedMapping

    class Parent(nn.Module):
        def __init__(self):
            super().__init__()
            self.balance = Balance()

        def forward(self, x):
            return self.balance(x)

    class Balance(nn.Linear):
        def __init__(self):
            super().__init__(32, 3, bias=False)
            self.seen_inputs = []

        def forward(self, x):
            self.seen_inputs.append(x.detach().clone())
            return super().forward(x)

    parent = Parent()
    parent.balance.global_name = "balance"
    set_teq_equivalent_transforms(
        parent.balance,
        weight=build_hadamard_transform("hadamard", block_size=32, location="weight"),
        input=build_hadamard_transform("hadamard", block_size=32, location="input", inverse=True),
    )
    transform = AWQTransform(AWQConfig(bits=16, group_size=32, sym=True, data_type="int", n_grid=2, duo_scaling=False))
    mapping = ResolvedMapping(
        smooth_name="smooth",
        smooth_layer=parent.balance,
        balance_names=["balance"],
        balance_layers=[parent.balance],
        parent_name="",
        parent=parent,
    )
    inputs = torch.randn(2, 32)
    transform._parent_args_cache[parent] = [((inputs,), {})]
    original = parent(inputs).detach()
    original_weight = parent.balance.weight.detach().clone()
    observed = []

    def identity_qdq(weight, params, **kwargs):
        observed.append(weight.detach().clone())
        return weight

    transform._qdq_tool.qdq = identity_qdq
    x_mean = torch.linspace(0.2, 2.0, 32)
    transform._grid_search_scales(mapping, x_mean)
    assert len(observed) == 2
    rotated = parent.balance._autoround_teq_transforms["weight"](original_weight)
    torch.testing.assert_close(observed[0], rotated)
    scales = x_mean / (x_mean.max() * x_mean.min()).sqrt()
    expected_second = parent.balance._autoround_teq_transforms["weight"](original_weight * scales.view(1, -1))
    torch.testing.assert_close(observed[1], expected_second)
    # The parent reference remains in its original FP basis; each candidate's
    # child pre-hook provides the matching ``rotate(x / s)`` input basis.
    input_transform = parent.balance._autoround_teq_transforms["input"]
    torch.testing.assert_close(parent.balance.seen_inputs[1], input_transform(inputs))
    torch.testing.assert_close(parent.balance.seen_inputs[2], input_transform(inputs / scales))
    # Identity QDQ leaves each candidate function-preserving; this verifies
    # the parent reference did not need to be changed into the rotated basis.
    torch.testing.assert_close(parent(inputs), original)


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


def test_teq_staged_refine_smoke(tiny_opt_model_path):
    ar = AutoRound(
        tiny_opt_model_path,
        bits=4,
        group_size=32,
        alg_configs=[
            TEQConfig(
                iters=1,
                awq_init=True,
                awq_init_n_grid=2,
                optimization_mode="staged_refine",
                experimental=True,
                ar_first_iters=1,
                refine_iters=1,
                joint_lr=1e-5,
            ),
            SignRoundConfig(iters=2, enable_alg_ext=True),
        ],
        nsamples=1,
        seqlen=8,
        dataset=["local staged TEQ refinement calibration sample with enough tokens"],
        low_cpu_mem_usage=False,
    )

    model, layer_config = ar.quantize()

    assert model is not None
    assert layer_config
    assert not any(hasattr(module, "_teq_input_context") for module in model.modules())
