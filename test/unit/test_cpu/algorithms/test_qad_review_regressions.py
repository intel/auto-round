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

"""Offline regressions for QAD review findings and chained tuning state."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from auto_round import envs
from auto_round.algorithms.composer import AlgorithmComposer, BlockContext
from auto_round.algorithms.quantization.qad.config import QADConfig
from auto_round.algorithms.quantization.qad.quantizer import QADQuantizer
from auto_round.algorithms.quantization.sign_round.config import AdamRoundConfig, SignRoundConfig, SignRoundV2Config
from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer
from auto_round.algorithms.quantization.sign_roundv2.quantizer import SignRoundOptimizedWrapperLinear
from auto_round.calibration.llm import LLMCalibrator
from auto_round.wrapper import (
    preserve_original_weights,
    store_warm_start_params_and_unwrap,
    wrapper_block,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(42)


def _linear():
    layer = torch.nn.Linear(8, 8)
    for key, value in dict(bits=4, group_size=8, sym=True, data_type="int", act_bits=16).items():
        setattr(layer, key, value)
    layer.scale_dtype = torch.float32
    return layer


class _TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="tiny", vocab_size=8, _name_or_path="same-checkpoint")
        self.embedding = torch.nn.Embedding(8, 8)
        self.proj = _linear()
        self.calls = []

    def forward(self, input_ids, attention_mask, **kwargs):
        self.calls.append((input_ids.detach().clone(), attention_mask.detach().clone()))
        # Context-dependent outputs expose corrupted internal EOS tokens.
        hidden = (self.embedding(input_ids) * attention_mask.unsqueeze(-1)).cumsum(dim=1)
        return SimpleNamespace(logits=self.proj(hidden))


def _quantizer(tmp_path, **kwargs):
    config = QADConfig(iters=2, lr=0.01, gradient_checkpointing=False, qad_workspace=str(tmp_path), topk=3, **kwargs)
    quantizer = QADQuantizer(config)
    quantizer.bind(
        SimpleNamespace(
            model_context=SimpleNamespace(amp=False),
            compress_context=SimpleNamespace(enable_torch_compile=False),
            calibration_context=SimpleNamespace(batch_size=2, nsamples=2),
        )
    )
    quantizer._get_scaler = lambda: None
    return quantizer


def _calibrator():
    calibrator = object.__new__(LLMCalibrator)
    calibrator.tokenizer = SimpleNamespace(pad_token_id=2, eos_token_id=2)
    return calibrator


@pytest.mark.parametrize("last_padded", [False, True])
def test_original_context_is_independent_of_labels(tmp_path, last_padded):
    calibrator = _calibrator()
    ids = torch.tensor([[1, 2, 3, 2]])
    mask = torch.tensor([[1, 1, 1, 0 if last_padded else 1]])
    sample = calibrator._model_input_samples(ids, mask)[0]
    assert sample["labels"][0, 1] == 2  # Internal EOS is not padding.
    sample["labels"][0, 1] = -100  # Loss exclusion must not change context.
    quantizer = _quantizer(tmp_path)
    model = _TinyLM()
    expected = model(input_ids=ids, attention_mask=mask).logits
    inputs = quantizer._load_model_batch([0], [sample], "cpu")
    torch.testing.assert_close(quantizer._forward_logits(model, inputs, 8), expected)
    quantizer._build_teacher_cache_from_model(model, [sample], "cpu")
    labels, _, _ = quantizer._load_teacher_batch([0], "cpu")
    assert labels[0, 1] == -100
    torch.testing.assert_close(model.calls[-1][0], ids)
    torch.testing.assert_close(model.calls[-1][1], mask)
    quantizer._teacher_cache.shutdown()


def test_collect_and_reuse_original_calibration_inputs(tmp_path):
    calibrator = _calibrator()
    ids = torch.tensor([[1, 2, 3, 2], [4, 2, 5, 2]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    calibrator.dataset = [{"input_ids": ids, "attention_mask": mask}]
    calibrator.seqlen = 4
    samples = calibrator.collect_token_ids(2, include_model_inputs=True)
    torch.testing.assert_close(torch.cat([sample["input_ids"] for sample in samples]), ids)
    torch.testing.assert_close(samples[0]["labels"], torch.tensor([[1, 2, 3, -100]]))
    calibrator.model_input_samples = samples
    quantizer = _quantizer(tmp_path)
    selected = quantizer.get_calibration_token_ids([sample["labels"] for sample in samples], calibrator)
    assert selected[0] is samples[0]
    # More samples than the common pool must use the same raw-input collection.
    calibrator.model_input_samples = samples[:1]
    quantizer.config.nsamples = 2
    selected = quantizer.get_calibration_token_ids([samples[0]["labels"]], calibrator)
    torch.testing.assert_close(selected[1]["attention_mask"], mask[1:])


def test_block_calibration_does_not_mutate_whole_model_attention(tmp_path):
    calibrator = _calibrator()
    ids = torch.tensor([[1, 2, 3], [4, 2, 5]])
    original_mask = torch.ones_like(ids)
    data = {"input_ids": ids.clone(), "attention_mask": original_mask.clone()}
    calibrator.dataset = [data]
    calibrator.model = _TinyLM()
    calibrator.model.device = "cpu"
    calibrator.inputs = {}
    calibrator.batch_size = 2
    calibrator.seqlen = 3
    calibrator.calib(nsamples=1, bs=2)
    # Block caching forces a nontrivial final attention position; the dataset
    # and whole-model snapshot must retain the user's unmodified attention.
    assert calibrator.model.calls[-1][1][0, -1] == 0
    torch.testing.assert_close(data["attention_mask"], original_mask)
    torch.testing.assert_close(calibrator.model_input_samples[0]["attention_mask"], original_mask[:1])
    quantizer = _quantizer(tmp_path)
    quantizer.config.nsamples = 1
    selected = quantizer.get_calibration_token_ids(calibrator.inputs["input_ids"], calibrator)
    assert len(selected) == 1
    torch.testing.assert_close(selected[0]["labels"], ids[:1])


def test_last_token_remains_a_next_token_target(tmp_path):
    calibrator = _calibrator()
    ids = torch.tensor([[1, 3]])
    labels = calibrator._mask_ids_for_loss(ids)
    torch.testing.assert_close(labels, ids)
    eos_ids = torch.tensor([[1, 2]])
    torch.testing.assert_close(calibrator._mask_ids_for_loss(eos_ids), eos_ids)
    logits = torch.randn(1, 2, 8, requires_grad=True)
    quantizer = _quantizer(tmp_path)
    loss = quantizer._qad_ce_loss(logits, labels)
    torch.testing.assert_close(loss, F.cross_entropy(logits[:, 0], ids[:, 1]))
    loss.backward()
    assert logits.grad[:, 0].abs().sum() > 0
    assert logits.grad[:, 1].abs().sum() == 0
    no_tokenizer = object.__new__(LLMCalibrator)
    no_tokenizer.tokenizer = None
    torch.testing.assert_close(no_tokenizer._mask_ids_for_loss(ids), ids)
    torch.testing.assert_close(
        no_tokenizer._mask_ids_for_loss(torch.tensor([[1, 3, 3, 3]])), torch.tensor([[1, 3, -100, -100]])
    )


def test_custom_calibrator_requires_recoverable_inputs(tmp_path):
    quantizer = _quantizer(tmp_path)
    raw_ids = [torch.tensor([1, 2])]
    selected = quantizer.get_calibration_token_ids(raw_ids, SimpleNamespace())
    assert selected[0] is raw_ids[0]
    with pytest.raises(ValueError, match="original input_ids"):
        quantizer.get_calibration_token_ids([torch.tensor([1, -100])], SimpleNamespace())
    sample = _calibrator()._model_input_samples(torch.tensor([[1, 2]]))[0]
    assert quantizer.get_calibration_token_ids([sample], SimpleNamespace())[0] is sample


def test_teacher_signature_includes_bfloat16_weights_and_scalar_buffers():
    model = _TinyLM().to(torch.bfloat16)
    model.register_buffer("counter", torch.tensor(0))
    ids = [torch.tensor([1, 2])]
    original = QADQuantizer._teacher_cache_signature(model, ids)
    model.counter.add_(1)
    assert original != QADQuantizer._teacher_cache_signature(model, ids)


def test_teacher_cache_checks_actual_weights_and_masks(tmp_path):
    model = _TinyLM()
    sample = _calibrator()._model_input_samples(torch.tensor([[1, 2, 3]]))[0]
    quantizer = _quantizer(tmp_path)
    quantizer._build_teacher_cache_from_model(model, [sample], "cpu")
    assert len(model.calls) == 1
    quantizer._build_teacher_cache_from_model(model, [sample], "cpu")
    assert len(model.calls) == 1  # Identical weights/data safely reuse the cache.
    other = copy.deepcopy(model)
    other.calls = []
    with torch.no_grad():
        other.proj.weight.add_(0.25)
    quantizer._build_teacher_cache_from_model(other, [sample], "cpu")
    assert len(other.calls) == 1  # Same config/path, different in-memory teacher.
    with torch.no_grad():
        other.embedding.weight.add_(0.1)
    quantizer._build_teacher_cache_from_model(other, [sample], "cpu")
    assert len(other.calls) == 2  # Also revalidate a reused quantizer instance.
    sample["attention_mask"][0, 1] = 0
    quantizer._build_teacher_cache_from_model(other, [sample], "cpu")
    assert len(other.calls) == 3
    quantizer._teacher_cache.shutdown()


@pytest.mark.parametrize("use_teacher", [False, True])
@pytest.mark.parametrize("tune_fp", [False, True])
def test_whole_model_freezes_base_and_clears_placement(tmp_path, use_teacher, tune_fp):
    model = _TinyLM()
    original_params = list(model.parameters())
    for module in model.modules():
        module.tuning_device = "cuda:7"  # CPU test catches accidental use of stale metadata.
    for parameter in original_params:
        parameter.grad = torch.ones_like(parameter)
    quantizer = _quantizer(tmp_path, qad_kd_weight=int(use_teacher), qad_ce_weight=1, tune_non_quant_params=tune_fp)
    samples = _calibrator()._model_input_samples(torch.tensor([[1, 2, 3], [4, 2, 5]]))
    gradients = []

    def backward(scaler, loss):
        loss.backward()
        assert model.proj.value.grad is not None
        assert model.proj.orig_layer.weight.grad is None
        assert model.proj.orig_layer.bias.grad is None
        assert (model.embedding.weight.grad is not None) == tune_fp
        assert not any(hasattr(module, "tuning_device") for module in model.modules())
        gradients.append(model.proj.value.grad.abs().sum().item())

    quantizer._scale_loss_and_backward = backward
    composer = object.__new__(AlgorithmComposer)
    composer._run_tuning(model, lambda: quantizer.quantize_model(model, samples, device="cpu"))
    assert len(gradients) == 2 and all(value > 0 for value in gradients)
    assert not model.proj.weight.requires_grad
    assert model.embedding.weight.requires_grad == tune_fp
    assert all(2 in ids for ids, _ in model.calls)


@pytest.mark.parametrize("config_cls", [SignRoundConfig, SignRoundV2Config, AdamRoundConfig])
def test_enable_qad_routes_to_whole_model(config_cls):
    composer = AlgorithmComposer([config_cls(enable_qad=True, iters=2, nblocks=1, lfq_top_k=5)])
    quantizer = composer.block_quantizer
    assert isinstance(quantizer, QADQuantizer)
    assert quantizer.whole_model_tuning
    assert quantizer.requires_original_weights
    assert quantizer.iters == 2
    assert quantizer.qad_top_k == 5


@pytest.mark.parametrize("nblocks", [1, 2, 3])
@pytest.mark.parametrize("last_blocks", [1, 2, 3])
def test_lfq_uses_physical_group_endpoint(nblocks, last_blocks):
    quantizer = SignRoundQuantizer(SignRoundConfig(enable_lfq=True, lfq_num_blocks=last_blocks))
    names = [f"model.layers.{index}" for index in range(8)]
    starts = list(range(0, 8, nblocks))
    for start in starts:
        group = names[start : start + nblocks]
        ctx = BlockContext(
            model=torch.nn.Module(),
            block_names=group,
            block_name=group[0],
            block_index=start,
            block_cnt=len(starts),
            all_block_names=names,
        )
        assert quantizer._lfq_is_kd_block(ctx) == (min(start + nblocks, 8) > 8 - last_blocks)


@pytest.mark.parametrize("topk", [-10, -2, 0])
def test_invalid_topk_rejected_immediately(topk):
    with pytest.raises(ValueError, match="topk"):
        QADConfig(topk=topk)


@pytest.mark.parametrize("topk", [-1, 1, 32])
def test_valid_topk_accepted(topk):
    assert QADConfig(topk=topk).topk == topk


@pytest.mark.parametrize("source_log,target_log", [(True, True), (False, True), (True, False)])
def test_log_minmax_warm_start_preserves_coefficients(monkeypatch, source_log, target_log):
    monkeypatch.setattr(envs, "AR_ENABLE_LOG_MINMAX", source_log)
    layer = _linear().requires_grad_(False)
    wrapped = SignRoundOptimizedWrapperLinear(layer, enable_torch_compile=False)
    best = {key: value.detach().clone() for key, value in wrapped.params.items()}
    best["min_scale"].fill_(-0.2 if source_log else 0.8)
    best["max_scale"].fill_(0.3 if source_log else 1.2)
    block = torch.nn.Sequential(wrapped)
    store_warm_start_params_and_unwrap(block, {"0": best})
    monkeypatch.setattr(envs, "AR_ENABLE_LOG_MINMAX", target_log)
    resumed = SignRoundOptimizedWrapperLinear(layer, enable_torch_compile=False)
    min_coeff, max_coeff = resumed._prepare_minmax_scale(resumed.min_scale, resumed.max_scale)
    for coefficient, key in [(min_coeff, "min_scale"), (max_coeff, "max_scale")]:
        expected = best[key].exp() if source_log else best[key]
        torch.testing.assert_close(coefficient, expected)
    torch.testing.assert_close(resumed.init_scale, wrapped.init_scale)


@pytest.mark.parametrize("norm_cls", [torch.nn.LayerNorm, LlamaRMSNorm])
def test_bias_and_norm_warm_start_matches_previous_best_output(norm_cls):
    block = torch.nn.Sequential(_linear(), norm_cls(8)).requires_grad_(False)
    original = {key: value.clone() for key, value in block.state_dict().items()}
    inputs = torch.randn(2, 8)
    with preserve_original_weights(block):
        wrapper_block(block, True, True, enable_torch_compile=False)
        best = {}
        for name, wrapped in block.named_children():
            best[name] = {key: value.detach().clone() for key, value in wrapped.params.items()}
            for key, param in wrapped.params.items():
                if key in ("bias_v", "v"):
                    best[name][key].fill_(0.2)
                    with torch.no_grad():
                        param.copy_(best[name][key])
        expected = block(inputs).detach()
        store_warm_start_params_and_unwrap(block, best)
    for key, value in block.state_dict().items():
        torch.testing.assert_close(value, original[key])
    wrapper_block(block, True, True, enable_torch_compile=False)
    torch.testing.assert_close(block[0].bias_v, best["0"]["bias_v"])
    torch.testing.assert_close(block[1].v, best["1"]["v"])
    torch.testing.assert_close(block(inputs), expected)


def test_log_minmax_incompatible_warm_start_uses_identity(monkeypatch):
    monkeypatch.setattr(envs, "AR_ENABLE_LOG_MINMAX", True)
    layer = _linear().requires_grad_(False)
    layer._ar_warm_start = {"min_scale": torch.ones(3), "max_scale": torch.ones(3), "minmax_parameterization": "log"}
    resumed = SignRoundOptimizedWrapperLinear(layer, enable_torch_compile=False)
    torch.testing.assert_close(resumed.min_scale, torch.zeros_like(resumed.min_scale))
    torch.testing.assert_close(resumed.max_scale, torch.zeros_like(resumed.max_scale))
