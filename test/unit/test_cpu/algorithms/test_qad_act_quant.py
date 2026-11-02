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

"""Offline CPU regressions for QAD with activation quantization."""

from types import SimpleNamespace

import pytest
import torch

from auto_round import envs
from auto_round.algorithms.composer import AlgorithmComposer
from auto_round.algorithms.quantization.qad.config import QADConfig
from auto_round.algorithms.quantization.qad.quantizer import QADQuantizer
from auto_round.algorithms.quantization.sign_roundv2.quantizer import SignRoundOptimizedWrapperLinear
from auto_round.schemes import PRESET_SCHEMES
from auto_round.wrapper import WrapperLinear, WrapperWALayer, store_warm_start_params_and_unwrap


def _linear(scheme):
    layer = torch.nn.Linear(32, 32, bias=False)
    for key, value in PRESET_SCHEMES[scheme].to_dict().items():
        if value is not None:
            setattr(layer, key, value)
    layer.scale_dtype = torch.float32
    layer.requires_grad_(False)
    return layer


def _wrap(layer, cls=WrapperLinear):
    return cls(layer, enable_torch_compile=False, device="cpu")


@pytest.mark.parametrize("scheme", ["MXFP4", "FP8_STATIC"])
@pytest.mark.parametrize("env_scale", [1.0, 0.75])
def test_finalize_restores_best_activation_scales(monkeypatch, scheme, env_scale):
    monkeypatch.setattr(envs, "AR_ENABLE_ACT_MINMAX_TUNING", True)
    monkeypatch.setattr(envs, "AR_ACT_SCALE", env_scale)
    torch.manual_seed(42)
    layer = _linear(scheme)
    if scheme == "FP8_STATIC":
        layer.act_max = torch.tensor([10.0])
    wrapper = _wrap(layer)
    inputs = torch.randn(2, 32)
    with torch.no_grad():
        wrapper.act_min_scale.fill_(0.6)
        wrapper.act_max_scale.fill_(0.5)
        best = {key: value.detach().clone() for key, value in wrapper.params.items()}
        expected = wrapper(inputs).clone()
        _, expected_scale, _ = wrapper._qdq_act(
            inputs, wrapper.act_min_scale, wrapper.act_max_scale, getattr(layer, "act_max", None)
        )
        wrapper.act_min_scale.fill_(1.0)
        wrapper.act_max_scale.fill_(1.0)
        finalized = wrapper.unwrapper(best)

    torch.testing.assert_close(finalized(inputs), expected)
    assert finalized.orig_layer.act_min_scale.item() == pytest.approx(0.6)
    assert finalized.orig_layer.act_max_scale.item() == pytest.approx(0.5)
    assert not finalized.orig_layer.act_max_scale.requires_grad
    if scheme == "FP8_STATIC":
        torch.testing.assert_close(layer.act_scale, expected_scale)
        torch.testing.assert_close(layer.act_max, torch.tensor([10.0]))


@pytest.mark.parametrize("scheme", ["NVFP4", "NVFP4_E5M3_GS"])
@pytest.mark.parametrize("fixed_global_scale", [False, True])
def test_nvfp_finalized_qdq_matches_training(scheme, fixed_global_scale):
    torch.manual_seed(42)
    layer = _linear(scheme)
    layer.act_max = torch.tensor([100.0])
    if fixed_global_scale:
        layer.input_global_scale = torch.tensor([26.88])
    wrapper = _wrap(layer)
    inputs = torch.randn(2, 32)
    with torch.no_grad():
        expected = wrapper(inputs).clone()
        finalized = wrapper.unwrapper({})
    torch.testing.assert_close(finalized(inputs), expected)


@pytest.mark.parametrize("wrapper_cls", [WrapperLinear, SignRoundOptimizedWrapperLinear])
@pytest.mark.parametrize("scheme", ["MXFP4", "NVFP4"])
def test_activation_warm_start(monkeypatch, wrapper_cls, scheme):
    monkeypatch.setattr(envs, "AR_ENABLE_ACT_MINMAX_TUNING", True)
    torch.manual_seed(42)
    layer = _linear(scheme)
    layer.act_max = torch.tensor([100.0])
    layer.input_global_scale = torch.tensor([26.88])
    wrapper = _wrap(layer, wrapper_cls)
    best = {key: value.detach().clone() for key, value in wrapper.params.items()}
    best["act_min_scale"].fill_(0.6)
    best["act_max_scale"].fill_(0.5)
    block = torch.nn.Sequential(wrapper)
    store_warm_start_params_and_unwrap(block, {"0": best})
    resumed = _wrap(block[0], wrapper_cls)
    torch.testing.assert_close(resumed.act_min_scale, best["act_min_scale"])
    torch.testing.assert_close(resumed.act_max_scale, best["act_max_scale"])
    torch.testing.assert_close(layer.act_max, torch.tensor([100.0]))
    torch.testing.assert_close(resumed.input_global_scale, layer.input_global_scale)


class _TinyLM(torch.nn.Module):
    def __init__(self, scheme):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=32)
        self.embedding = torch.nn.Embedding(32, 32)
        self.first = _linear(scheme)
        self.second = _linear(scheme)
        self.requires_grad_(False)

    def forward(self, input_ids, **kwargs):
        hidden = self.first(self.embedding(input_ids))
        return SimpleNamespace(logits=self.second(hidden))


def _quantizer(tmp_path, use_teacher=False):
    quantizer = QADQuantizer(
        QADConfig(
            iters=2,
            lr=0.01,
            topk=8,
            qad_workspace=str(tmp_path),
            qad_kd_weight=1 if use_teacher else 0,
            qad_ce_weight=0 if use_teacher else 1,
            gradient_checkpointing=False,
        )
    )
    quantizer.bind(
        SimpleNamespace(
            model_context=SimpleNamespace(amp=False),
            compress_context=SimpleNamespace(enable_torch_compile=False),
            calibration_context=SimpleNamespace(batch_size=1),
        )
    )
    quantizer._get_scaler = lambda: None
    return quantizer


@pytest.mark.parametrize("scheme", ["MXFP4", "NVFP4", "NVFP4_E5M3_GS", "FP8_STATIC"])
@pytest.mark.parametrize("teacher_cache", ["disabled", "cold", "warm"])
def test_whole_model_activation_calibration(tmp_path, scheme, teacher_cache):
    torch.manual_seed(42)
    model = _TinyLM(scheme)
    quantizer = _quantizer(tmp_path, teacher_cache != "disabled")
    token_ids = [torch.tensor([1, 2, 3, 4]), torch.tensor([4, 3, 2, -100])]
    if teacher_cache == "warm":
        # All samples are already cached: calibration must not depend on cache misses.
        quantizer._build_teacher_cache_from_model(model, token_ids, "cpu")
    gradients = []

    def backward(scaler, loss):
        loss.backward()
        gradients.append([model.first.value.grad.abs().sum().item(), model.second.value.grad.abs().sum().item()])

    quantizer._scale_loss_and_backward = backward
    composer = object.__new__(AlgorithmComposer)
    composer._run_tuning(model, lambda: quantizer.quantize_model(model, token_ids, device="cpu"))
    assert gradients and all(first > 0 and second > 0 for first, second in gradients)
    for layer in (model.first, model.second):
        assert isinstance(layer, WrapperWALayer)
        assert not layer.orig_layer._forward_hooks
        if scheme == "MXFP4":
            assert not hasattr(layer.orig_layer, "act_max")
        else:
            assert torch.isfinite(layer.orig_layer.act_max).all()
            assert layer.orig_layer.act_max.numel() == 1
            assert layer.orig_layer.act_max.item() > 0
        if scheme.startswith("NVFP4"):
            from auto_round.data_type.nvfp import calculate_gparam_by_dtype

            # The NVFP exporter can now derive its required static input scale.
            original = layer.orig_layer
            scale = calculate_gparam_by_dtype(original.act_max, original.act_data_type, original.act_group_size)
            assert torch.isfinite(scale).all() and (scale > 0).all()

    if scheme.startswith("NVFP4"):
        from auto_round.export.export_to_autoround.export_to_nvfp_mx import pack_layer

        for name in ("first", "second"):
            original = getattr(model, name).orig_layer
            expected_scale = calculate_gparam_by_dtype(
                original.act_max, original.act_data_type, original.act_group_size
            )
            pack_layer(name, model, backend=None, device="cpu")
            torch.testing.assert_close(getattr(model, name).input_global_scale.reshape(-1), expected_scale.reshape(-1))


def test_calibration_prepares_fused_nvfp_scales(tmp_path, monkeypatch):
    monkeypatch.setattr(envs, "AR_NVFP4_FUSED_LAYER_GLOBAL_SCALE", True)
    model = torch.nn.Module()
    for name in ("q_proj", "k_proj", "v_proj"):
        setattr(model, name, _linear("NVFP4"))
    with torch.no_grad():
        model.q_proj.weight.fill_(0.1)
        model.k_proj.weight.fill_(0.2)
        model.v_proj.weight.fill_(0.4)
    quantizer = _quantizer(tmp_path)

    def forward(module, *args):
        inputs = torch.ones(1, 32)
        for name in ("q_proj", "k_proj", "v_proj"):
            getattr(module, name)(inputs)

    monkeypatch.setattr(quantizer, "_forward_logits", forward)
    quantizer._calibrate_activations(model, [torch.tensor([1, 2])], "cpu")
    for projection in (model.q_proj, model.k_proj, model.v_proj):
        torch.testing.assert_close(projection.weight_global_scale, model.v_proj.weight_global_scale)
        torch.testing.assert_close(projection.act_max, torch.ones(1))


def test_calibration_removes_hooks_on_failure(tmp_path, monkeypatch):
    model = _TinyLM("NVFP4")
    model.train()
    quantizer = _quantizer(tmp_path)

    def fail_forward(*args):
        assert model.first._forward_hooks
        raise RuntimeError("calibration failed")

    monkeypatch.setattr(quantizer, "_forward_logits", fail_forward)
    with pytest.raises(RuntimeError, match="calibration failed"):
        quantizer._calibrate_activations(model, [torch.tensor([1, 2])], "cpu")
    assert model.training
    assert not model.first._forward_hooks
    assert not model.second._forward_hooks
