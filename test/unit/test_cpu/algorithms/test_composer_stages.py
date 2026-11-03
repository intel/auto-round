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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from auto_round.algorithms.composer import AlgorithmComposer
from auto_round.algorithms.quantization.qad.config import QADConfig
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.wrapper import preserve_original_weights


def _pipeline(configs, monkeypatch):
    monkeypatch.delenv("AR_RESUME_DIR", raising=False)
    composer = AlgorithmComposer(configs)
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16))
    model.requires_grad_(False)
    composer._orchestrator = SimpleNamespace(
        model_context=SimpleNamespace(model=model),
        compress_context=SimpleNamespace(is_immediate_saving=False, is_immediate_packing=False),
        calibration=SimpleNamespace(nsamples=3),
        _reload_model_for_tuning=Mock(),
    )
    return composer, model


def test_three_stages_preserve_original_bf16_without_algorithm_hooks(monkeypatch):
    composer, model = _pipeline([SignRoundConfig(), SignRoundConfig(), QADConfig(nsamples=2)], monkeypatch)
    original = model[0].weight.detach().clone()
    block_calls, output_values = [], []

    def run_blocks(replay):
        block_calls.append((composer.block_quantizer, replay))
        torch.testing.assert_close(model[0].weight, original)

        def tune():
            with torch.no_grad():
                model[0].weight.fill_(3)
            return {}

        composer._run_tuning(model, tune, lambda module: output_values.append(module[0].weight.clone()))
        torch.testing.assert_close(model[0].weight, original)

    def tune_model(module, tokens, device=None):
        torch.testing.assert_close(module[0].weight, original)
        assert tokens == [1, 2]
        with torch.no_grad():
            module[0].weight.fill_(7)
        return {}

    composer.quantizers[-1].quantize_model = Mock(side_effect=tune_model)
    assert composer.compress_model([1, 2, 3], run_blocks) is True
    assert [q for q, _ in block_calls] == list(composer.quantizers[:2])
    assert [replay for _, replay in block_calls] == [False, True]
    assert len(output_values) == 2
    assert all(torch.all(value == 3) for value in output_values)
    assert torch.all(model[0].weight == 7)
    assert model[0].weight.dtype == torch.bfloat16
    assert composer._preserve_weights is False
    composer._orchestrator._reload_model_for_tuning.assert_called_once()
    for quantizer in composer.quantizers:
        assert not hasattr(quantizer, "begin_stage")
        assert not hasattr(quantizer, "end_stage")
        assert not hasattr(quantizer, "warm_start_collect")


def test_block_stages_consume_previous_weights_when_originals_not_required(monkeypatch):
    composer, model = _pipeline([SignRoundConfig(), SignRoundConfig()], monkeypatch)
    original = model[0].weight.detach().clone()
    seen = []

    def run_blocks(replay):
        seen.append((replay, model[0].weight.detach().clone()))

        def tune():
            with torch.no_grad():
                model[0].weight.add_(1)
            return {}

        composer._run_tuning(model, tune)

    assert composer.compress_model([1], run_blocks) is False
    torch.testing.assert_close(seen[0][1], original)
    torch.testing.assert_close(seen[1][1], original + 1)
    torch.testing.assert_close(model[0].weight, original + 2)


def test_stage_failure_restores_weights_and_composer(monkeypatch):
    composer, model = _pipeline([SignRoundConfig(), QADConfig()], monkeypatch)
    terminal = composer.block_quantizer
    original_layer = model[0]
    original = original_layer.weight.detach().clone()
    for quantizer in composer.quantizers:
        quantizer.finalize_run = Mock()

    def run_blocks(replay):
        def tune():
            with torch.no_grad():
                original_layer.weight.fill_(5)
            model[0] = torch.nn.Identity()
            raise RuntimeError("tuning failed")

        composer._run_tuning(model, tune)

    with pytest.raises(RuntimeError, match="tuning failed"):
        composer.compress_model([1], run_blocks)
    assert model[0] is original_layer
    torch.testing.assert_close(model[0].weight, original)
    assert composer.block_quantizer is terminal
    assert composer._preserve_weights is False
    for quantizer in composer.quantizers:
        quantizer.finalize_run.assert_called_once()


def test_weight_preservation_restores_dtype_and_requires_grad():
    model = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    original = model.weight.detach().clone()
    with preserve_original_weights(model):
        model.weight.data = torch.zeros_like(model.weight, dtype=torch.float32)
        model.weight.requires_grad_(False)
    torch.testing.assert_close(model.weight, original)
    assert model.weight.dtype == torch.bfloat16
    assert model.weight.requires_grad


@pytest.mark.parametrize("option", ["is_immediate_saving", "is_immediate_packing"])
def test_multi_stage_streaming_rejected_before_tuning(monkeypatch, option):
    composer, _ = _pipeline([SignRoundConfig(), QADConfig()], monkeypatch)
    setattr(composer._orchestrator.compress_context, option, True)
    run_blocks = Mock()
    with pytest.raises(NotImplementedError, match="immediate saving/packing"):
        composer.compress_model([1], run_blocks)
    run_blocks.assert_not_called()


def test_multi_stage_resume_rejected_before_tuning(monkeypatch):
    composer, _ = _pipeline([SignRoundConfig(), QADConfig()], monkeypatch)
    monkeypatch.setenv("AR_RESUME_DIR", "unused-resume-dir")
    run_blocks = Mock()
    with pytest.raises(NotImplementedError, match="Resuming a multi-stage"):
        composer.compress_model([1], run_blocks)
    run_blocks.assert_not_called()


@pytest.mark.parametrize("block_stages", [1, 2])
def test_tiny_model_block_to_model_tuning_without_downloads(monkeypatch, block_stages):
    from tokenizers import Tokenizer, models
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from auto_round import AutoRound
    from auto_round.algorithms.quantization.qat.config import QATConfig

    monkeypatch.delenv("AR_RESUME_DIR", raising=False)
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
    model = LlamaForCausalLM(config).to(torch.bfloat16).eval()
    original_weight = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, **{str(i): i for i in range(4, 32)}}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="<unk>")),
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
    )
    tokens = torch.randint(3, 32, (3, 8))
    dataset = torch.utils.data.DataLoader(tokens, batch_size=1)
    algorithms = [SignRoundConfig(iters=1, bits=4, group_size=8) for _ in range(block_stages)]
    algorithms.append(QATConfig(iters=1, bits=4, group_size=8, nsamples=3, gradient_checkpointing=False))
    compressor = AutoRound(
        model,
        tokenizer=tokenizer,
        alg_configs=algorithms,
        dataset=dataset,
        nsamples=2,
        seqlen=8,
        batch_size=1,
        device_map="cpu",
        low_cpu_mem_usage=False,
        enable_torch_compile=False,
    )
    quantized, layer_config = compressor.quantize()
    assert quantized is model
    assert layer_config
    assert not torch.equal(model.model.layers[0].self_attn.q_proj.weight, original_weight)
    assert all(not hasattr(module, "orig_layer") for module in model.modules())
    with torch.no_grad():
        assert torch.isfinite(model(input_ids=tokens[:1]).logits).all()
