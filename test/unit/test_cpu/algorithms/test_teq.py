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

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from auto_round import AutoRound, AWQConfig, SignRoundConfig, TEQConfig
from auto_round.algorithms.registry import resolve_alg_config, resolve_algorithm_names
from auto_round.algorithms.transforms.hadamard.transforms import build_hadamard_transform
from auto_round.algorithms.transforms.smoothing.errors import NoFiniteCandidateError
from auto_round.algorithms.transforms.teq.base import TEQTransform, _TEQLinearFakeQuant
from auto_round.algorithms.transforms.teq.protocol import (
    TEQ_EQUIVALENT_TRANSFORMS_ATTR,
    install_teq_equivalent_input_hook,
    set_teq_equivalent_transforms,
)
from auto_round.algorithms.transforms.teq.smooth import (
    TEQSmoothCalibration,
    iter_batch_calls,
    run_teq_smoothing,
    truncate_forward_value,
)
from auto_round.data_type.utils import compute_optimized_init_scale, get_quant_func
from auto_round.schemes import QuantizationScheme
from auto_round.wrapper import WrapperLinear


def _make_awq_transform():
    from auto_round.algorithms.transforms.awq.base import AWQTransform
    from auto_round.algorithms.transforms.awq.qdq import QDQTool

    scheme = QuantizationScheme(bits=16, group_size=32, sym=True, data_type="int")
    transform = AWQTransform(AWQConfig(n_grid=2, duo_scaling=False))
    transform.bind(SimpleNamespace(model_context=None, compress_context=None, scheme_context=scheme))
    transform._qdq_tool = QDQTool(
        bits=scheme.bits, group_size=scheme.group_size, sym=scheme.sym, data_type=scheme.data_type
    )
    return transform


def test_teq_algorithm_registry():
    assert resolve_algorithm_names("teq,auto_round") == ["teq", "auto_round"]
    assert isinstance(resolve_alg_config("teq"), TEQConfig)


def test_teq_uses_run_scheme_with_layer_overrides():
    config = TEQConfig()
    assert "TEQConfig(" in repr(config)
    with pytest.raises(TypeError, match="AutoRound"):
        TEQConfig(bits=4)
    scheme = QuantizationScheme(bits=8, group_size=32, sym=False, data_type="int", act_bits=16)
    transform = TEQTransform(config)
    transform.bind(SimpleNamespace(model_context=None, compress_context=None, scheme_context=scheme))
    layer = nn.Linear(32, 4)
    layer.global_name = "proj"
    params = transform._resolve_params(layer)
    for field in ("bits", "group_size", "sym", "data_type", "act_bits"):
        assert params[field] == getattr(scheme, field)
        assert not hasattr(config, field)
    transform.layer_config = {"proj": {"bits": 4, "group_size": 16}}
    params = transform._resolve_params(layer)
    assert params["bits"] == 4
    assert params["group_size"] == 16
    assert params["sym"] is False


def test_teq_default_recipe_is_cross_size_candidate():
    config = TEQConfig()

    assert config.teq_iters == 5
    assert config.teq_lr == pytest.approx(1.5e-4)
    assert config.teq_optimizer == "adam"
    assert config.teq_awq_init is True
    assert config.teq_awq_init_n_grid == 20
    assert config.teq_min_scale == 1e-5
    assert config.teq_max_scale == 3.0
    assert config.teq_nsamples == 32
    assert config.teq_batch_size == 1
    assert config.teq_sample_seqlen == 512


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


def test_teq_smooth_replay_microbatches_and_truncates_sequence():
    hidden = torch.randn(2, 8, 4)
    attention_mask = torch.ones(2, 8, dtype=torch.long)

    truncated_hidden, truncated_mask = truncate_forward_value((hidden, attention_mask), 5)
    assert truncated_hidden.shape == (2, 5, 4)
    assert truncated_mask.shape == (2, 5)

    calls = list(iter_batch_calls((hidden, attention_mask), {}, batch_size=1))
    assert len(calls) == 2
    assert all(args[0].shape == (1, 8, 4) and args[1].shape == (1, 8) for args, _ in calls)


def test_teq_nonfinite_training_falls_back_without_aborting_autoround(monkeypatch):
    import auto_round.algorithms.transforms.teq.smooth as teq_smooth

    initial_alpha = torch.tensor([0.5, 1.0, 2.0])

    class Owner:
        min_scale = 0.1
        max_scale = 3.0
        iters = 5
        awq_init = False

        def __init__(self):
            self.applied = None

        def _apply_scales(self, mapping, scales):
            del mapping
            self.applied = scales.detach().clone()

        def _log_scale_summary(self, *args):
            del args

    def fail_training(*args, **kwargs):
        del args, kwargs
        raise NoFiniteCandidateError

    monkeypatch.setattr(teq_smooth, "smooth_group_from_mapping", lambda mapping: mapping)
    monkeypatch.setattr(teq_smooth.SmoothEngine, "run", fail_training)
    owner = Owner()
    mapping = nn.Module()
    mapping.smooth_name = "block.smooth"
    calibration = TEQSmoothCalibration([], [], initial_alpha)

    outcome = run_teq_smoothing(owner, mapping, calibration)

    assert outcome.trained is False
    torch.testing.assert_close(owner.applied, initial_alpha)
    torch.testing.assert_close(outcome.alpha, initial_alpha)


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


def test_signround_dq_wrapper_composes_teq_scaling_and_rotation():
    """The main-branch double-quant path must quantize the TEQ-transformed weight."""
    from auto_round.algorithms.quantization.sign_roundv2.quantizer import SignRoundDQWrapperLinear

    layer = torch.nn.Linear(4, 3, bias=False)
    layer.bits = 4
    layer.group_size = 4
    layer.super_group_size = 16
    layer.data_type = "int"
    layer.sym = True
    layer.act_bits = 16
    layer.act_group_size = -1
    layer.act_sym = True
    layer.act_data_type = "int"
    layer.act_dynamic = True
    layer.scale_dtype = torch.float16
    layer.disable_opt_rtn = True
    layer.iters = 1
    layer.tuning_device = "cpu"
    alpha = torch.tensor([1.5, 2.0, 2.5, 3.0])
    layer._teq_input_context = {
        "log_alpha": torch.nn.Parameter(alpha.log()),
        "min_log": -10.0,
        "max_log": 10.0,
        "apply_input_transform": True,
        "wrappers": [],
    }
    set_teq_equivalent_transforms(layer, weight=lambda weight: weight * 3.0, input=lambda value: value)
    wrapper = SignRoundDQWrapperLinear(layer, enable_torch_compile=False, device="cpu")
    expected = layer.weight.detach() * alpha.view(1, -1) * 3.0
    observed = []

    def fake_search(weight, value):
        del value
        observed.append(weight.detach().clone())
        return {"scale": torch.ones(1), "d_scale": torch.ones(1)}

    wrapper._run_search = fake_search
    wrapper.weight_quant_func = lambda weight, scale, bits, v=None: weight
    result = wrapper._qdq_weight(wrapper.value, wrapper.min_scale, wrapper.max_scale)

    torch.testing.assert_close(observed[0], expected)
    torch.testing.assert_close(result[0], expected)


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
    transform = _make_awq_transform()
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
    # Index 0 is the manual reference call above and index 1 is the grid
    # search's FP reference replay. Candidate inputs start at index 2.
    torch.testing.assert_close(parent.balance.seen_inputs[2], input_transform(inputs))
    torch.testing.assert_close(parent.balance.seen_inputs[3], input_transform(inputs / scales))
    # Identity QDQ leaves each candidate function-preserving; this verifies
    # the parent reference did not need to be changed into the rotated basis.
    torch.testing.assert_close(parent(inputs), original)


def test_teq_static_activation_training_is_rejected():
    scheme = QuantizationScheme(
        bits=4, group_size=32, data_type="int", act_bits=8, act_data_type="int", act_dynamic=False
    )
    from auto_round.algorithms.transforms.teq.base import TEQTransform

    transform = TEQTransform(TEQConfig())
    transform.bind(SimpleNamespace(model_context=None, compress_context=None, scheme_context=scheme))
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
    assert type(ar.alg_composer.active_quantizer).__name__ == "SignRoundV2Quantizer"
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
    result = ar.alg_composer.active_quantizer.teq_last_refinement
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


def _make_teq_fp_linear(in_features: int, out_features: int, bias: bool = False) -> nn.Linear:
    """A 16-bit Linear whose ``_qdq_weight`` is an exact (identity-quant) transform."""
    layer = nn.Linear(in_features, out_features, bias=bias)
    layer.bits = 16
    layer.group_size = -1
    layer.data_type = "int"
    layer.sym = True
    layer.act_bits = 16
    layer.scale_dtype = torch.float32
    return layer


def _make_teq_context(size: int, stale: torch.Tensor, apply_input_transform: bool = True) -> dict:
    return {
        "log_alpha": nn.Parameter(stale.clone()),
        "min_log": -10.0,
        "max_log": 10.0,
        "smooth_name": "smooth",
        "apply_input_transform": apply_input_transform,
        "output_wrapped": False,
    }


def test_unwrapper_block_settles_shared_teq_scales_before_baking():
    """A layer consuming two scales must bake both at their *selected* values.

    ``up_proj`` sits in two mappings: it consumes one scale on its input side
    and another (``down_proj``'s) on its output side.  ``unwrapper_block``
    walks modules in order, so without a settle pass ``up_proj`` would bake the
    last-iteration value of the output scale instead of the selected one.
    """
    from auto_round.wrapper import unwrapper_block

    torch.manual_seed(7)
    up = _make_teq_fp_linear(5, 4)
    down = _make_teq_fp_linear(4, 3)
    up_weight = up.weight.detach().clone()
    down_weight = down.weight.detach().clone()

    # Stale values are what the last tuning iteration left behind.
    context_in = _make_teq_context(5, torch.full((5,), 0.9))
    context_out = _make_teq_context(4, torch.full((4,), -0.7))
    up._teq_input_context = context_in
    up._teq_output_contexts = [context_out]
    down._teq_input_context = context_out

    block = nn.Module()
    block.up = WrapperLinear(up, enable_torch_compile=False, device="cpu")
    block.down = WrapperLinear(down, enable_torch_compile=False, device="cpu")

    best_in = torch.linspace(-0.2, 0.3, 5)
    best_out = torch.linspace(0.4, -0.1, 4)
    unwrapper_block(
        block,
        {"up": {"teq_log_alpha": best_in.clone()}, "down": {"teq_log_alpha": best_out.clone()}},
    )

    expected_up = up_weight * best_in.exp().view(1, -1) / best_out.exp().view(-1, 1)
    expected_down = down_weight * best_out.exp().view(1, -1)
    torch.testing.assert_close(up.weight.detach(), expected_up)
    torch.testing.assert_close(down.weight.detach(), expected_down)


def test_unwrapper_block_bakes_teq_weight_exactly_once():
    """Unwrapping must not quantize (or rotate) an already committed weight."""
    from auto_round.wrapper import unwrapper_block

    torch.manual_seed(8)
    layer = _make_teq_fp_linear(32, 4)
    original = layer.weight.detach().clone()
    context = _make_teq_context(32, torch.zeros(32))
    layer._teq_input_context = context
    set_teq_equivalent_transforms(
        layer,
        weight=build_hadamard_transform("hadamard", block_size=32, location="weight"),
        input=build_hadamard_transform("hadamard", block_size=32, location="input", inverse=True),
    )

    block = nn.Module()
    block.proj = WrapperLinear(layer, enable_torch_compile=False, device="cpu")

    best = torch.linspace(-0.25, 0.25, 32)
    unwrapper_block(block, {"proj": {"teq_log_alpha": best.clone()}})

    rotate = layer._autoround_teq_transforms["weight"]
    expected = rotate(original * best.exp().view(1, -1))
    torch.testing.assert_close(layer.weight.detach(), expected, rtol=1e-5, atol=1e-6)
    # A second rotation would be the signature of a redundant write-back.
    assert not torch.allclose(layer.weight.detach(), rotate(expected), rtol=1e-4, atol=1e-5)


class _StubModelTEQTransform(TEQTransform):
    """Expose a plain module as ``model`` without a full algorithm run context."""

    @property
    def model(self):
        return self._stub_model


@pytest.mark.parametrize("smooth_is_quantized", [True, False])
def test_teq_finalize_folds_scale_into_unquantized_smooth_linear(smooth_is_quantized):
    """An excluded smooth Linear must still receive the full output transform.

    When the smooth layer is quantized, its ``WrapperLinear`` already folded the
    scale into the weight and only the bias is left.  When it is excluded (via
    ignore_layers / mixed precision) no wrapper ever runs, so finalization owns
    the whole fold -- otherwise the committed model is not function-preserving.
    """
    from auto_round.algorithms.transforms.awq.mappings import ResolvedMapping

    torch.manual_seed(9)
    smooth = _make_teq_fp_linear(6, 4, bias=True)
    balance = _make_teq_fp_linear(4, 3)
    model = nn.Module()
    model.smooth = smooth
    model.balance = balance

    scales = torch.linspace(1.5, 2.5, 4)
    context = _make_teq_context(4, scales.log(), apply_input_transform=False)
    context["output_wrapped"] = smooth_is_quantized
    balance._teq_input_context = context
    smooth._teq_output_contexts = [context]

    smooth_weight = smooth.weight.detach().clone()
    smooth_bias = smooth.bias.detach().clone()

    transform = _StubModelTEQTransform(TEQConfig(optimization_mode="joint", experimental=True))
    transform._stub_model = model
    mapping = ResolvedMapping(
        smooth_name="smooth",
        smooth_layer=smooth,
        balance_names=["balance"],
        balance_layers=[balance],
        parent_name="",
        parent=model,
    )
    transform._finalize_ar_refinement_mapping(mapping)

    # The bias is unquantized in both cases, so it is always folded here.
    torch.testing.assert_close(smooth.bias.detach(), smooth_bias / scales)
    if smooth_is_quantized:
        torch.testing.assert_close(smooth.weight.detach(), smooth_weight)
    else:
        torch.testing.assert_close(smooth.weight.detach(), smooth_weight / scales.view(-1, 1))
    assert not hasattr(balance, "_teq_input_context")


@pytest.mark.parametrize("rotation_aware", [True, False])
def test_awq_grid_search_drops_imatrix_for_rotation_aware_candidates(rotation_aware):
    """Diagonal importance statistics are invalid in the rotated basis.

    ``imatrix`` is a per-input-channel vector collected in the original basis.
    A rotation-aware candidate is quantized after the equivalent weight
    transform, and a non-diagonal Hadamard does not preserve a diagonal
    importance vector, so reusing it would weight the wrong channels.
    """
    from auto_round.algorithms.transforms.awq.mappings import ResolvedMapping

    torch.manual_seed(11)
    balance = nn.Linear(32, 3, bias=False)
    balance.global_name = "balance"
    balance.imatrix = torch.rand(32) + 0.5
    parent = nn.Module()
    parent.balance = balance
    if rotation_aware:
        set_teq_equivalent_transforms(
            balance,
            weight=build_hadamard_transform("hadamard", block_size=32, location="weight"),
            input=build_hadamard_transform("hadamard", block_size=32, location="input", inverse=True),
        )

    transform = _make_awq_transform()
    mapping = ResolvedMapping(
        smooth_name="smooth",
        smooth_layer=balance,
        balance_names=["balance"],
        balance_layers=[balance],
        parent_name="",
        parent=parent,
    )
    seen_imatrix = []

    def identity_qdq(weight, params, **kwargs):
        seen_imatrix.append(kwargs.get("imatrix"))
        return weight

    transform._qdq_tool.qdq = identity_qdq
    transform._grid_search_scales(mapping, torch.linspace(0.2, 2.0, 32))

    assert seen_imatrix
    if rotation_aware:
        assert all(imatrix is None for imatrix in seen_imatrix)
    else:
        assert all(imatrix is not None for imatrix in seen_imatrix)
