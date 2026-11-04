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

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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


@pytest.mark.parametrize("dtype", [None, torch.float16, torch.bfloat16, torch.float32])
def test_full_vocabulary_cache_storage_dtype(tmp_path, dtype):
    kwargs = {} if dtype is None else {"full_dtype": dtype}
    cache = TeacherTopKCache(str(tmp_path), -1, signature={"teacher": "one"}, **kwargs)
    expected_dtype = torch.float32 if dtype is None else dtype
    ids = torch.tensor([1, 2])
    logits = torch.tensor([[0.0, -20.0, -40.0], [0.0, -10.0, -30.0]])
    try:
        cache.build_sample(0, ids, logits)
        stored = torch.load(cache._path(0), map_location="cpu")
        assert stored["prob"].dtype == expected_dtype
        assert cache.is_valid_for(0, ids)
        cache.prefetch([0])
        loaded_ids, index, probs = cache.load(0, "cpu")
        assert index is None
        assert probs.dtype == torch.float32
        torch.testing.assert_close(loaded_ids, ids)
        torch.testing.assert_close(probs, logits.softmax(-1).to(expected_dtype).float())
        if expected_dtype in (torch.bfloat16, torch.float32):
            assert (probs[:, -1] > 0).all()
        # A file with an incompatible storage dtype must not be silently reused.
        other_dtype = torch.float16 if expected_dtype != torch.float16 else torch.float32
        stored["prob"] = stored["prob"].to(other_dtype)
        torch.save(stored, cache._path(0))
        assert not cache.is_valid_for(0, ids)
    finally:
        cache.shutdown()


def test_full_vocabulary_cache_dtype_namespaces_are_isolated(tmp_path):
    caches = []
    try:
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            cache = TeacherTopKCache(str(tmp_path), -1, signature={"teacher": "one"}, full_dtype=dtype)
            caches.append(cache)
            manifest = json.loads(Path(cache.workspace, "manifest.json").read_text())
            # FP16 retains its legacy manifest; BF16 and FP32 record their dtype explicitly.
            assert manifest.get("full_dtype", str(torch.float16)) == str(dtype)
        assert len({cache.workspace for cache in caches}) == 3
    finally:
        for cache in caches:
            cache.shutdown()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_topk_cache_stays_fp32_regardless_of_full_dtype(tmp_path, dtype):
    cache = TeacherTopKCache(str(tmp_path), 2, full_dtype=dtype)
    ids = torch.tensor([1, 2])
    logits = torch.tensor([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    try:
        cache.build_sample(0, ids, logits)
        stored = torch.load(cache._path(0), map_location="cpu")
        assert stored["prob"].dtype == torch.float32
        _, index, probs = cache.load(0, "cpu")
        torch.testing.assert_close(index, logits.topk(2).indices)
        torch.testing.assert_close(probs, logits.topk(2).values.softmax(-1))
    finally:
        cache.shutdown()


@pytest.mark.parametrize("dtype", [torch.float64, torch.int32, None])
def test_teacher_cache_rejects_unsupported_storage_dtype(tmp_path, dtype):
    with pytest.raises(ValueError, match="full_dtype.*bfloat16"):
        TeacherTopKCache(str(tmp_path), -1, full_dtype=dtype)


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


@pytest.fixture
def tiny_qad_model_and_tokenizer():
    from tokenizers import Tokenizer, models
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

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
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, **{str(i): i for i in range(4, 32)}}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="<unk>")),
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
    )
    return model, tokenizer


@pytest.mark.parametrize("dataset_kind", ["dataloader", "named"])
@pytest.mark.parametrize("qad_nsamples,available", [(None, 1024), (1024, 1024), (1024, 256)])
def test_signround_then_qad_uses_independent_sample_counts(
    tmp_path, monkeypatch, tiny_qad_model_and_tokenizer, dataset_kind, qad_nsamples, available
):
    from auto_round import AutoRound
    from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
    from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer
    from auto_round.calibration.llm import LLMCalibrator

    model, tokenizer = tiny_qad_model_and_tokenizer
    tokens = torch.randint(3, 32, (available, 4))
    requested_samples = []

    def get_dataloader(tokenizer, seqlen, dataset, seed, batch_size, nsamples, **kwargs):
        requested_samples.append(nsamples)
        return torch.utils.data.DataLoader(tokens[:nsamples], batch_size=batch_size)

    monkeypatch.setattr("auto_round.calib_dataset.get_dataloader", get_dataloader)
    dataset = "local-regression" if dataset_kind == "named" else torch.utils.data.DataLoader(tokens, batch_size=16)
    warmup_counts, qad_counts = [], []
    original_quantize_block = SignRoundQuantizer.quantize_block

    def quantize_block(self, block, fp_inputs, *args, **kwargs):
        warmup_counts.append(len(fp_inputs))
        return original_quantize_block(self, block, fp_inputs, *args, **kwargs)

    def quantize_model(self, model, samples, device=None):
        expected = min(qad_nsamples or 128, available)
        qad_counts.append(len(samples))
        assert len(samples) == expected
        torch.testing.assert_close(torch.cat([sample["input_ids"] for sample in samples]), tokens[:expected])
        return {}

    monkeypatch.setattr(SignRoundQuantizer, "quantize_block", quantize_block)
    monkeypatch.setattr(QADQuantizer, "quantize_model", quantize_model)
    collected_requests = []
    original_collect = LLMCalibrator.collect_token_ids

    def collect(calibration, nsamples, **kwargs):
        collected_requests.append((nsamples, kwargs))
        return original_collect(calibration, nsamples, **kwargs)

    monkeypatch.setattr(LLMCalibrator, "collect_token_ids", collect)
    compressor = AutoRound(
        model,
        tokenizer=tokenizer,
        scheme="W4A16",
        alg_configs=[
            SignRoundConfig(iters=2, group_size=8),
            QADConfig(qad_nsamples=qad_nsamples, qad_workspace=str(tmp_path), group_size=8),
        ],
        dataset=dataset,
        nsamples=128,
        seqlen=4,
        batch_size=16,
        device_map="cpu",
        low_cpu_mem_usage=False,
        enable_torch_compile=False,
    )
    compressor.quantize()

    assert warmup_counts == [128]
    assert qad_counts == [min(qad_nsamples or 128, available)]
    assert compressor.calibration_context.nsamples == 128
    if qad_nsamples is None:
        assert collected_requests == []
    else:
        assert collected_requests == [(1024, {"include_model_inputs": True})]
    if dataset_kind == "named":
        assert requested_samples == ([128] if qad_nsamples is None else [128, 1024])


@pytest.mark.parametrize("algorithm", ["qad", "qat", "rtn_qad"])
@pytest.mark.parametrize("lora_rank", [0, 2])
def test_whole_model_and_rtn_qad_without_downloads(
    tmp_path, monkeypatch, tiny_qad_model_and_tokenizer, algorithm, lora_rank
):
    from auto_round import AutoRound
    from auto_round.algorithms.quantization.qat.config import QATConfig
    from auto_round.algorithms.quantization.rtn.config import RTNConfig

    model, tokenizer = tiny_qad_model_and_tokenizer
    original = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    kwargs = dict(iters=2, bits=4, group_size=8, nsamples=2, gradient_checkpointing=False, qad_lora_rank=lora_rank)
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
