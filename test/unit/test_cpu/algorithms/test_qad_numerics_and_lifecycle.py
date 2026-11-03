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

"""Offline QAD numerical stability and cache/wrapper lifecycle regressions."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from filelock import Timeout

from auto_round.algorithms.quantization.qad.config import QADConfig
from auto_round.algorithms.quantization.qad.quantizer import QADQuantizer, _QADDistillLoss
from auto_round.algorithms.quantization.qad.teacher_cache import TeacherTopKCache
from auto_round.algorithms.quantization.sign_round.lfq import LFQMixin
from auto_round.wrapper import WrapperLinear, WrapperWALayer, store_warm_start_params_and_unwrap


def test_full_vocabulary_fp16_targets_are_shift_invariant():
    target = torch.full((1, 2, 3), 1 / 3, dtype=torch.float16)
    assert target.float().sum(-1)[0, 0] != 1
    labels = torch.tensor([[1, 2]])
    losses, gradients = [], []
    for offset in (0.0, 16.0):
        logits = torch.full((1, 2, 3), offset, requires_grad=True)
        loss = _QADDistillLoss.apply(logits, None, target, labels, 1.0, 1e-6, 2)
        loss.backward()
        losses.append(loss.detach())
        gradients.append(logits.grad)
    torch.testing.assert_close(losses[0], losses[1], atol=1e-5, rtol=0)
    torch.testing.assert_close(losses[0], torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(gradients[0], gradients[1], atol=2e-6, rtol=0)
    torch.testing.assert_close(gradients[0].sum(-1), torch.zeros(1, 2), atol=1e-6, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("full_mode", [False, True])
def test_low_precision_gradient_is_cast_after_subtraction_and_scaling(dtype, full_mode):
    logits = torch.zeros(2, 2, 2, dtype=dtype, requires_grad=True)
    target = torch.tensor([0.5001, 0.4999]).expand(2, 2, 2).clone()
    indices = None if full_mode else torch.tensor([0, 1]).expand(2, 2, 2)
    labels = torch.tensor([[0, 1], [-100, -100]])
    loss = _QADDistillLoss.apply(logits, indices, target, labels, 1.0, 1e-6, 1)
    (loss * 1024).backward()
    reference = logits.detach().float().requires_grad_()
    expected_loss = F.kl_div(F.log_softmax(reference[0, 0], -1), target[0, 0], reduction="sum")
    (expected_loss * 1024).backward()
    torch.testing.assert_close(logits.grad, reference.grad.to(dtype), atol=1e-5, rtol=0.01)
    assert logits.grad[0, 0].abs().min() > 0
    assert torch.count_nonzero(logits.grad[1]) == 0
    assert torch.count_nonzero(logits.grad[:, 1]) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fp_optimizer_masters_work_with_grad_scaler(dtype):
    param = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=dtype))
    original = param.detach().clone()
    optimizer_params, pairs = QADQuantizer._prepare_fp_optimizer_params([param])
    assert optimizer_params[0].dtype == torch.float32
    optimizer = torch.optim.AdamW(optimizer_params, lr=0.01, weight_decay=0)
    scaler = torch.amp.GradScaler("cpu", init_scale=16)
    for _ in range(2):
        scaler.scale(param.float().square().sum()).backward()
        QADQuantizer._copy_fp_grads_to_masters(pairs)
        assert optimizer_params[0].grad.dtype == torch.float32
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        QADQuantizer._copy_fp_masters_to_model(pairs)
    assert param.dtype == dtype
    assert not torch.equal(param, original)
    assert param.abs().max() < original.abs().max()
    if pairs:
        assert param.grad is None


@pytest.mark.parametrize("hf_loss", [False, True])
@pytest.mark.parametrize("seqlen", [1, 3])
def test_all_ignored_ce_is_graph_connected_zero(hf_loss, seqlen):
    model = SimpleNamespace()
    if hf_loss:
        model.loss_function = Mock(side_effect=AssertionError("CE must not be called with no valid labels"))
    hidden = torch.randn(2, seqlen, 4, requires_grad=True)
    labels = torch.full((2, seqlen), -100)
    # Bind the lightweight loss harness without relying on a downloaded model.
    harness = SimpleNamespace(lfq_loss_type="ce", _lfq_logits=lambda hidden: hidden, model=model)
    loss = LFQMixin.lfq_loss(harness, hidden, labels)
    assert loss.item() == 0 and torch.isfinite(loss)
    loss.backward()
    torch.testing.assert_close(hidden.grad, torch.zeros_like(hidden))
    if hf_loss:
        model.loss_function.assert_not_called()


def test_different_cache_signatures_do_not_invalidate_active_readers(tmp_path):
    cache = TeacherTopKCache(str(tmp_path), 2, signature={"teacher": "one"})
    ids = torch.tensor([1, 2])
    logits = torch.tensor([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    try:
        cache.build_sample(0, ids, logits)
        cache.prefetch([0])
        with ThreadPoolExecutor(1) as executor:
            other = executor.submit(TeacherTopKCache, str(tmp_path), 2, signature={"teacher": "two"}).result()
        try:
            other.build_sample(0, ids, -logits)
            assert cache.workspace != other.workspace
            loaded_ids, index, probs = cache.load(0, "cpu")
            torch.testing.assert_close(loaded_ids, ids)
            torch.testing.assert_close(index, logits.topk(2).indices)
            torch.testing.assert_close(probs, logits.topk(2).values.softmax(-1))
        finally:
            other.shutdown()
    finally:
        cache.shutdown()


def test_same_cache_signature_has_exclusive_lifecycle_lock(tmp_path):
    signature = {"teacher": "same"}
    cache = TeacherTopKCache(str(tmp_path), 2, signature=signature)
    try:
        with pytest.raises(Timeout):
            TeacherTopKCache(str(tmp_path), 2, signature=signature)
        cache.build_sample(0, torch.tensor([1, 2]), torch.randn(2, 3))
        cache.prefetch([0])
    finally:
        cache.shutdown()
    reopened = TeacherTopKCache(str(tmp_path), 2, signature=signature)
    try:
        assert reopened.exists(0)
    finally:
        reopened.shutdown()


def test_cache_lock_released_when_qad_setup_fails(tmp_path, monkeypatch):
    quantizer = QADQuantizer(QADConfig(qad_workspace=str(tmp_path)))
    cache = TeacherTopKCache(str(tmp_path), 2, signature={"teacher": "one"})
    quantizer._teacher_cache = cache

    def fail(*args):
        raise RuntimeError("wrapper setup failed")

    monkeypatch.setattr(quantizer, "_quantize_model", fail)
    with pytest.raises(RuntimeError, match="wrapper setup failed"):
        quantizer.quantize_model(torch.nn.Linear(2, 2), [torch.tensor([1, 2])])
    reopened = TeacherTopKCache(str(tmp_path), 2, signature={"teacher": "one"})
    reopened.shutdown()


def test_warm_start_skips_persistent_activation_wrapper():
    linear = torch.nn.Linear(4, 4).requires_grad_(False)
    for key, value in dict(bits=4, group_size=4, sym=True, data_type="int", act_bits=16).items():
        setattr(linear, key, value)
    linear.scale_dtype = torch.float32
    tuned = WrapperLinear(linear, enable_torch_compile=False)
    # A persistent QDQ wrapper has orig_layer, but no tuning params/unwrapper.
    persistent = object.__new__(WrapperWALayer)
    torch.nn.Module.__init__(persistent)
    persistent.orig_layer = torch.nn.Linear(4, 4)
    block = torch.nn.ModuleList([persistent, tuned])
    best = {"1": {key: value.detach().clone() for key, value in tuned.params.items()}}
    store_warm_start_params_and_unwrap(block, best)
    assert block[0] is persistent
    assert block[1] is linear
    assert "value" in linear._ar_warm_start


@pytest.mark.parametrize("algorithm", ["qad", "qat", "rtn_qad"])
def test_whole_model_and_rtn_qad_without_downloads(tmp_path, monkeypatch, algorithm):
    from tokenizers import Tokenizer, models
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from auto_round import AutoRound
    from auto_round.algorithms.quantization.qat.config import QATConfig
    from auto_round.algorithms.quantization.rtn.config import RTNConfig

    torch.manual_seed(42)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    original = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, **{str(i): i for i in range(4, 32)}}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="<unk>")),
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
    )
    kwargs = dict(iters=2, bits=4, group_size=8, nsamples=2, gradient_checkpointing=False)
    configs = [QATConfig(**kwargs)] if algorithm == "qat" else [QADConfig(qad_workspace=str(tmp_path), **kwargs)]
    if algorithm == "rtn_qad":
        configs.insert(0, RTNConfig(bits=4, group_size=8, act_bits=8, act_data_type="int", act_dynamic=True))
        configs[-1].act_bits = 8
        configs[-1].act_data_type = "int"
        configs[-1].act_dynamic = True
    tokens = torch.randint(3, 32, (2, 4))
    compressor = AutoRound(
        model,
        tokenizer=tokenizer,
        alg_configs=configs,
        dataset=torch.utils.data.DataLoader(tokens, batch_size=1),
        nsamples=2,
        seqlen=4,
        batch_size=1,
        device_map="cpu",
        low_cpu_mem_usage=False,
        enable_torch_compile=False,
    )
    if algorithm != "rtn_qad":
        monkeypatch.setattr(compressor, "cache_data", Mock(side_effect=AssertionError("unused activation cache")))
    quantized, _ = compressor.quantize()
    assert quantized is model
    projection = model.model.layers[0].self_attn.q_proj
    assert not torch.equal(projection.weight, original)
    if algorithm != "rtn_qad":
        assert not compressor.inputs
        assert not compressor.calibration.inputs
    assert torch.isfinite(model(input_ids=tokens).logits).all()
