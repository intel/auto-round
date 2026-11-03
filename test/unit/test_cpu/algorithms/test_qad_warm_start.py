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

"""CPU regressions for SignRound -> QAD state and output hand-off."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from auto_round.algorithms.composer import AlgorithmComposer, BlockContext
from auto_round.algorithms.quantization.qad.config import QADConfig
from auto_round.algorithms.quantization.qad.quantizer import QADQuantizer
from auto_round.algorithms.quantization.sign_roundv2.quantizer import SignRoundOptimizedWrapperLinear
from auto_round.wrapper import WrapperLinear, store_warm_start_params_and_unwrap, wrapper_block


class _SnapshotLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.orig_layer = torch.nn.Linear(1, 1, bias=False)
        self.orig_layer.requires_grad_(False)
        self.value = torch.nn.Parameter(torch.zeros(()))
        self.params = {"value": self.value}


class _SnapshotModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = _SnapshotLayer()
        self.fp = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(vocab_size=2)


@pytest.mark.parametrize("use_last", [False, True])
@pytest.mark.parametrize("tune_fp", [False, True])
def test_qad_snapshot_matches_measured_loss(monkeypatch, use_last, tune_fp):
    """Best loss is at the initial state; deliberately make each update worse."""
    model = _SnapshotModel()
    model.fp.requires_grad_(tune_fp)
    quantizer = QADQuantizer(
        QADConfig(
            iters=2,
            lr=0.1,
            qad_kd_weight=0,
            qad_ce_weight=1,
            gradient_checkpointing=False,
            tune_non_quant_params=tune_fp,
            not_use_best_mse=use_last,
        )
    )
    quantizer.bind(
        SimpleNamespace(
            model_context=SimpleNamespace(amp=False),
            compress_context=SimpleNamespace(enable_torch_compile=False),
            calibration_context=SimpleNamespace(batch_size=1),
        )
    )
    quantizer.wrapper_block = lambda *args, **kwargs: (["layer"], [])
    quantizer._forward_logits = lambda *args: model.layer.value + model.fp
    quantizer._qad_loss = lambda logits, *args: logits.square()
    quantizer._get_scaler = lambda: None
    quantizer._scale_loss_and_backward = lambda scaler, loss: loss.backward()
    snapshots = []

    def worsen_state(scaler, optimizer, schedule):
        with torch.no_grad():
            model.layer.value.add_(1)
            if tune_fp:
                model.fp.add_(1)

    def capture_unwrap(block, params):
        snapshots.append(params["layer"]["value"].clone())

    quantizer._step = worsen_state
    monkeypatch.setattr("auto_round.algorithms.composer.unwrapper_block", capture_unwrap)
    monkeypatch.setattr("auto_round.algorithms.quantization.qad.quantizer.clear_memory", lambda: None)
    composer = object.__new__(AlgorithmComposer)
    composer._run_tuning(
        model, lambda: quantizer.quantize_model(model, [torch.ones(2, dtype=torch.long)] * 2, device="cpu")
    )

    expected = 2.0 if use_last else 0.0
    assert model.layer.value.item() == 2.0  # Both optimizer steps actually ran.
    torch.testing.assert_close(snapshots[0], torch.tensor(expected))
    torch.testing.assert_close(model.fp, torch.tensor(expected if tune_fp else 0.0))


def _make_linear():
    layer = torch.nn.Linear(64, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.linspace(-0.9, 1.0, 128).reshape(2, 64))
    layer.requires_grad_(False)
    layer.bits = 2
    layer.group_size = 64
    layer.sym = True
    layer.data_type = "int"
    layer.act_bits = 16
    layer.scale_dtype = torch.float32
    return layer


def _wrap(layer, wrapper_cls):
    return wrapper_cls(layer, device="cpu", enable_torch_compile=False)


@pytest.mark.parametrize("wrapper_cls", [WrapperLinear, SignRoundOptimizedWrapperLinear])
@pytest.mark.parametrize("output_dict", [False, True])
def test_warmup_propagates_best_quantized_output(monkeypatch, wrapper_cls, output_dict):
    """Propagate QDQ output, preserve FP reference/weights and the QAD init grid."""
    layer = _make_linear()
    block = torch.nn.Sequential(layer)
    fp_weight = layer.weight.detach().clone()
    inputs = [torch.linspace(-1, 2, 64).reshape(1, 64)]
    fp_output = layer(inputs[0]).detach()
    expected_outputs = []
    calls = []

    def forward(module, samples, others):
        calls.append(module[0])
        result = [module(sample) for sample in samples]
        forward.last_output_dict = {"hidden_states": result} if output_dict else None
        return result

    forward.last_output_dict = None

    def tune(module, *args, **kwargs):
        wrapped = _wrap(module[0], wrapper_cls)
        module[0] = wrapped
        best = {key: value.detach().clone() for key, value in wrapped.params.items()}
        best["value"].fill_(0.4)
        best["min_scale"].fill_(0.8)
        best["max_scale"].fill_(0.8)
        with torch.no_grad():
            for key, value in best.items():
                wrapped.params[key].copy_(value)
            expected_outputs.append(module(inputs[0]).detach())
            wrapped.value.fill_(-0.4)  # Final iterate is deliberately not the best.
        return {"0": best}

    composer = object.__new__(AlgorithmComposer)
    composer._preserve_weights = True
    composer.block_forward = forward
    composer.block_quantizer = SimpleNamespace(
        enable_quanted_input=True,
        quantize_block=tune,
        finalize_block=Mock(),
        register_fp_input_forward_hooks=lambda block: [],
        register_qinput_forward_hooks=lambda block: [],
    )
    composer.preprocessors = []
    composer._rotation_members = []
    composer.scheme = None
    composer._get_fp_act_hooks = lambda block: []
    composer._get_q_act_hooks = lambda block: []
    monkeypatch.setattr("auto_round.algorithms.composer.clear_memory", lambda: None)
    ctx = BlockContext(model=block, block_names=["0"], block_name="0", block_index=0)
    quantized, reference = composer.compress_block(block, inputs, {}, ctx)

    q_result = quantized["hidden_states"] if output_dict else quantized
    fp_result = reference["hidden_states"] if output_dict else reference
    torch.testing.assert_close(q_result[0], expected_outputs[0])
    torch.testing.assert_close(fp_result[0], fp_output)
    assert not torch.allclose(q_result[0], fp_output)
    assert len(calls) == 2  # FP reference, then the still-wrapped best QDQ forward.
    assert isinstance(calls[1], wrapper_cls)
    assert block[0] is layer
    torch.testing.assert_close(layer.weight, fp_weight)
    torch.testing.assert_close(layer._ar_warm_start["value"], torch.full((2, 64), 0.4))

    # QAD re-wraps exactly these FP layers with the transferred best parameters.
    wrapper_block(block, True, False, enable_torch_compile=False, device="cpu")
    resumed = block[0]
    assert isinstance(resumed, wrapper_cls)
    with torch.no_grad():
        torch.testing.assert_close(resumed(inputs[0]), q_result[0])
    if wrapper_cls is SignRoundOptimizedWrapperLinear:
        torch.testing.assert_close(resumed.init_scale, layer._ar_warm_start["init_scale"])


def test_warmup_restores_all_fp_layers_when_forward_raises():
    layers = [_make_linear(), _make_linear()]
    weights = [layer.weight.detach().clone() for layer in layers]
    block = torch.nn.ModuleList([_wrap(layer, WrapperLinear) for layer in layers])
    best = {
        name: {key: value.detach().clone() for key, value in module.params.items()}
        for name, module in block.named_children()
    }

    def fail_forward(module):
        assert all(isinstance(layer, WrapperLinear) for layer in module)
        raise RuntimeError("forward failed")

    with pytest.raises(RuntimeError, match="forward failed"):
        store_warm_start_params_and_unwrap(block, best, before_unwrap=fail_forward)

    for index, layer in enumerate(layers):
        assert block[index] is layer
        torch.testing.assert_close(layer.weight, weights[index])
        assert "value" in layer._ar_warm_start
