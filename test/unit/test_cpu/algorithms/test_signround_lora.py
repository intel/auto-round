# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline SignRound low-rank rounding configuration and tuning regressions."""

from unittest.mock import Mock

import pytest
import torch

from auto_round import AutoRound
from auto_round.algorithms.quantization.qad.config import QADConfig
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig, SignRoundV2Config
from auto_round.wrapper import WrapperLinear


def _scaling_wrapper(rank, alpha=None):
    layer = torch.nn.Linear(8, 8).requires_grad_(False)
    for key, value in dict(bits=4, group_size=4, sym=True, data_type="int", act_bits=16).items():
        setattr(layer, key, value)
    layer.scale_dtype = torch.float32
    return WrapperLinear(layer, lora_rank=rank, lora_alpha=alpha, enable_torch_compile=False)


@pytest.mark.parametrize("rank", [2, 64])
@pytest.mark.parametrize("iters", [1, 4])
@pytest.mark.parametrize("direction", [-1, 1])
def test_signsgd_default_scaling_bounds_extreme_updates(rank, iters, direction):
    from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer
    from auto_round.algorithms.quantization.sign_round.sign_sgd import SignSGD

    wrapper = _scaling_wrapper(rank)
    # Align the initial A entries and every update to attain the worst-case
    # bound, rather than relying on random cancellation in B @ A.
    a0 = wrapper.lora_A.detach().abs().max().item()
    with torch.no_grad():
        wrapper.lora_A.fill_(a0)
        wrapper.value.fill_(0.25)  # Previous stages are outside the new-offset bound.
    optimizer = SignSGD([{"params": [wrapper.lora_A], "lr": 0.1}, {"params": [wrapper.lora_B], "lr": 0.05}], lr=0.1)
    schedule = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=iters)
    quantizer = SignRoundQuantizer(SignRoundConfig(iters=iters, lora_rank=rank))
    quantizer._configure_lora_scaling(torch.nn.Sequential(wrapper), optimizer)
    budget_a, budget_b = 0.1 * (iters + 1) / 2, 0.05 * (iters + 1) / 2
    assert wrapper.lora_scaling == pytest.approx(4 / (rank * (a0 + budget_a) * budget_b))
    assert wrapper.qad_lora_scaling == wrapper.lora_scaling
    for _ in range(iters):
        wrapper.lora_A.grad = -torch.ones_like(wrapper.lora_A)
        wrapper.lora_B.grad = -direction * torch.ones_like(wrapper.lora_B)
        optimizer.step()
        schedule.step()
        offset = wrapper._rounding_value() - wrapper.value
        assert offset.abs().max().item() <= 4 + 1e-5
    torch.testing.assert_close(offset, torch.full_like(offset, direction * 4.0))


@pytest.mark.parametrize("alpha", [None, 8.0])
@pytest.mark.parametrize("adam", [False, True])
def test_lora_scaling_preserves_explicit_alpha_and_other_optimizers(alpha, adam):
    from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer
    from auto_round.algorithms.quantization.sign_round.sign_sgd import SignSGD

    wrapper = _scaling_wrapper(2, alpha)
    optimizer_cls = torch.optim.AdamW if adam else SignSGD
    optimizer = optimizer_cls([wrapper.lora_A, wrapper.lora_B], lr=0.1)
    quantizer = SignRoundQuantizer(SignRoundConfig(iters=4, lora_rank=2, lora_alpha=alpha))
    quantizer._configure_lora_scaling(wrapper, optimizer)
    if alpha is not None or adam:
        assert wrapper.lora_scaling == (alpha / 2 if alpha is not None else 1.0)
    else:
        assert wrapper.lora_scaling != 1.0


@pytest.mark.parametrize("case", ["zero_iters", "zero_lr", "custom_scheduler", "no_lora"])
def test_lora_auto_scaling_without_default_budget(case, monkeypatch):
    from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer
    from auto_round.algorithms.quantization.sign_round.sign_sgd import SignSGD
    from auto_round.logger import logger

    wrapper = _scaling_wrapper(0 if case == "no_lora" else 2)
    optimizer = SignSGD(wrapper.params.values(), lr=0 if case == "zero_lr" else 0.1)
    quantizer = SignRoundQuantizer(SignRoundConfig(iters=0 if case == "zero_iters" else 4, lora_rank=2))
    if case == "custom_scheduler":
        quantizer.lr_scheduler = object()
    warning = Mock()
    monkeypatch.setattr(logger, "warning_once", warning)
    quantizer._configure_lora_scaling(wrapper, optimizer)
    assert wrapper.lora_scaling == 1.0
    assert bool(warning.call_args_list) is (case == "custom_scheduler")


@pytest.mark.parametrize("config_cls", [SignRoundConfig, SignRoundV2Config])
@pytest.mark.parametrize("rank", [-1, True, 1.5, "2"])
def test_invalid_signround_lora_rank(config_cls, rank):
    with pytest.raises(ValueError, match="lora_rank"):
        config_cls(lora_rank=rank)


@pytest.mark.parametrize("alpha", [0, -1, float("inf"), float("nan")])
def test_invalid_signround_lora_alpha(alpha):
    with pytest.raises(ValueError, match="lora_alpha"):
        SignRoundConfig(lora_rank=2, lora_alpha=alpha)


@pytest.mark.parametrize("prefix", ["", "signround_"])
def test_signround_lora_cli(prefix):
    from auto_round.cli.parser import build_quantize_parser

    args = build_quantize_parser().parse_args(
        ["local-model", "--algs", "sign_round", f"--{prefix}lora_rank", "8", f"--{prefix}lora_alpha", "16"]
    )
    assert (args.lora_rank, args.lora_alpha) == (8, 16)
    assert SignRoundConfig().lora_rank == 0


def test_qad_specific_settings_override_shared_lora():
    shared = QADConfig(lora_rank=2, lora_alpha=4)
    assert (shared.qad_lora_rank, shared.qad_lora_alpha) == (2, 4)
    specific = QADConfig(lora_rank=2, lora_alpha=4, qad_lora_rank=8, qad_lora_alpha=16)
    assert (specific.qad_lora_rank, specific.qad_lora_alpha) == (8, 16)


@pytest.mark.parametrize(
    "flag,enabled",
    [
        (None, True),
        ("--qad_enable_round_tuning", True),
        ("--qad_disable_round_tuning", False),
        ("--no-qad_enable_round_tuning", False),
    ],
)
def test_qad_rounding_cli(flag, enabled):
    from auto_round.cli.parser import build_quantize_parser

    args = build_quantize_parser().parse_args(["local-model", "--algs", "qad"] + ([flag] if flag else []))
    assert getattr(args, "qad_enable_round_tuning", True) is enabled
    assert QADConfig(qad_enable_round_tuning=enabled).qad_enable_round_tuning is enabled


@pytest.mark.parametrize(
    "case", ["signround", "v2", "adam", "qad", "qad_adamw", "qad_disabled", "no_lora", "zero_iters"]
)
def test_lora_signsgd_warning_uses_actual_optimizer(monkeypatch, case):
    from auto_round.algorithms.quantization.adam_round.adam import AdamRoundQuantizer
    from auto_round.algorithms.quantization.qad.quantizer import QADQuantizer
    from auto_round.algorithms.quantization.sign_round.config import AdamRoundConfig
    from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer
    from auto_round.logger import logger

    warning = Mock()
    monkeypatch.setattr(logger, "warning_once", warning)
    if case.startswith("qad"):
        config = QADConfig(
            qad_lora_rank=2,
            qad_optimizer="adamw" if case == "qad_adamw" else "signround",
            qad_enable_round_tuning=case != "qad_disabled",
        )
        quantizer = QADQuantizer(config)
    else:
        cls = AdamRoundConfig if case == "adam" else SignRoundV2Config if case == "v2" else SignRoundConfig
        config = cls(lora_rank=0 if case == "no_lora" else 2, iters=0 if case == "zero_iters" else 2)
        quantizer = (AdamRoundQuantizer if case == "adam" else SignRoundQuantizer)(config)
    messages = [call.args[0] for call in warning.call_args_list if "LoRA rounding with SignSGD" in call.args[0]]
    should_warn = case in ("signround", "v2", "qad")
    assert bool(messages) == should_warn
    if should_warn:
        from auto_round.algorithms.quantization.sign_round.sign_sgd import SignSGD

        assert quantizer.optimizer is SignSGD
        assert ("--qad_optimizer adamw" if case == "qad" else "--enable_adam") in messages[0]


@pytest.fixture
def tiny_model_and_tokenizer():
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


def _compressor(model_and_tokenizer, algorithm):
    model, tokenizer = model_and_tokenizer
    tokens = torch.randint(3, 32, (2, 4))
    compressor = AutoRound(
        model,
        tokenizer=tokenizer,
        alg_configs=algorithm,
        dataset=torch.utils.data.DataLoader(tokens, batch_size=1),
        nsamples=2,
        seqlen=4,
        batch_size=1,
        device_map="cpu",
        low_cpu_mem_usage=False,
        enable_torch_compile=False,
    )
    compressor.post_init()
    return compressor, tokens


@pytest.mark.parametrize("minmax", [False, True])
def test_qad_rounding_disabled_keeps_other_tuning(tiny_model_and_tokenizer, monkeypatch, minmax):
    config = QADConfig(
        iters=2,
        bits=4,
        group_size=8,
        qad_lora_rank=2,
        qad_enable_round_tuning=False,
        enable_minmax_tuning=minmax,
        qad_kd_weight=0,
        qad_ce_weight=1,
        gradient_checkpointing=False,
    )
    compressor, tokens = _compressor(tiny_model_and_tokenizer, config)
    wrapper_init = WrapperLinear.__init__
    wrappers = []

    def capture_wrapper(self, *args, **kwargs):
        wrapper_init(self, *args, **kwargs)
        wrappers.append(self)

    monkeypatch.setattr(WrapperLinear, "__init__", capture_wrapper)
    model, _ = compressor.quantize()
    assert wrappers
    for wrapper in wrappers:
        assert not wrapper.enable_round_tuning and "value" not in wrapper.params
        assert not hasattr(wrapper, "lora_A")
        assert not wrapper.value.requires_grad and wrapper.value.grad is None
        assert ("max_scale" in wrapper.params) is minmax
    assert torch.isfinite(model(input_ids=tokens).logits).all()


@pytest.mark.parametrize("config_cls", [SignRoundConfig, SignRoundV2Config])
@pytest.mark.parametrize("minmax", [False, True])
def test_signround_block_optimizes_low_rank_factors(tiny_model_and_tokenizer, monkeypatch, config_cls, minmax):
    config = config_cls(iters=3, bits=4, group_size=8, lora_rank=2, enable_minmax_tuning=minmax, not_use_best_mse=True)
    compressor, tokens = _compressor(tiny_model_and_tokenizer, config)
    quantizer = compressor.alg_composer.active_quantizer
    wrapper_init = WrapperLinear.__init__
    wrappers = []

    def capture_wrapper(self, *args, **kwargs):
        wrapper_init(self, *args, **kwargs)
        wrappers.append(self)

    monkeypatch.setattr(WrapperLinear, "__init__", capture_wrapper)
    optimized = []
    optimizer_cls = quantizer.optimizer

    def capture_optimizer(groups, **kwargs):
        optimizer = optimizer_cls(groups, **kwargs)
        optimized.extend(parameter for group in optimizer.param_groups for parameter in group["params"])
        return optimizer

    monkeypatch.setattr(quantizer, "optimizer", capture_optimizer)
    model, _ = compressor.quantize()
    assert wrappers
    for wrapper in wrappers:
        assert "value" not in wrapper.params and not wrapper.value.requires_grad
        for factor in (wrapper.lora_A, wrapper.lora_B):
            assert sum(parameter is factor for parameter in optimized) == 1
        assert wrapper.lora_B.abs().sum() > 0
    assert not any(hasattr(module, "params") for module in model.modules())
    assert torch.isfinite(model(input_ids=tokens).logits).all()


@pytest.mark.parametrize("minmax", [False, True])
def test_outside_block_optimizer_uses_both_factors(tiny_model_and_tokenizer, monkeypatch, minmax):
    config = SignRoundConfig(iters=3, bits=4, group_size=8, lora_rank=2, enable_minmax_tuning=minmax)
    compressor, _ = _compressor(tiny_model_and_tokenizer, config)
    quantizer = compressor.alg_composer.active_quantizer
    layer = compressor.model.lm_head
    for name, value in dict(bits=4, group_size=8, sym=True, data_type="int", act_bits=16).items():
        setattr(layer, name, value)
    layer.scale_dtype = torch.float32
    layer.global_name = "lm_head"
    optimized = []
    optimizer_cls = quantizer.optimizer

    def capture_optimizer(parameters, **kwargs):
        optimizer = optimizer_cls(parameters, **kwargs)
        optimized.extend(parameter for group in optimizer.param_groups for parameter in group["params"])
        return optimizer

    monkeypatch.setattr(quantizer, "optimizer", capture_optimizer)
    quantizer.quantize_layer_outside_block(layer, [torch.randn(1, 4, 32) for _ in range(2)])
    factors = [parameter for parameter in optimized if parameter.ndim == 2]
    assert sorted(tuple(parameter.shape) for parameter in factors) == [(2, 32), (32, 2)]
    assert len({id(parameter) for parameter in optimized}) == len(optimized)
    assert factors[1].abs().sum() > 0
    lm_head = compressor.model.lm_head
    inputs = torch.randn(1, lm_head.in_features, dtype=lm_head.weight.dtype, device=lm_head.weight.device)
    assert torch.isfinite(lm_head(inputs)).all()


@pytest.mark.parametrize("qad_iters", [1, 3])
@pytest.mark.parametrize("optimizer_name", ["signround", "adamw"])
@pytest.mark.parametrize("alpha", [None, 4.0])
def test_qad_whole_model_uses_effective_lora_budget(
    tiny_model_and_tokenizer, monkeypatch, qad_iters, optimizer_name, alpha
):
    from auto_round.algorithms.quantization.qad.quantizer import QADQuantizer

    config = QADConfig(
        iters=20,
        qad_iters=qad_iters,
        bits=4,
        group_size=8,
        qad_lora_rank=2,
        qad_lora_alpha=alpha,
        qad_optimizer=optimizer_name,
        qad_kd_weight=0,
        qad_ce_weight=1,
        gradient_checkpointing=False,
        not_use_best_mse=True,
    )
    compressor, tokens = _compressor(tiny_model_and_tokenizer, config)
    quantizer = compressor.alg_composer.active_quantizer
    assert isinstance(quantizer, QADQuantizer)
    assert quantizer.iters == qad_iters
    assert quantizer.lr == pytest.approx(1 / qad_iters)
    configure_scaling = quantizer._configure_lora_scaling
    configured = []

    def capture_scaling(model, optimizer):
        budgets = {
            id(parameter): float(group["lr"]) * (qad_iters + 1) / 2
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        expected = []
        for module in model.modules():
            if not isinstance(module, WrapperLinear) or not hasattr(module, "lora_A"):
                continue
            scaling = alpha / 2 if alpha is not None else 1.0
            if optimizer_name == "signround" and alpha is None:
                a_bound = module.lora_A.detach().abs().max().item() + budgets[id(module.lora_A)]
                b_bound = module.lora_B.detach().abs().max().item() + budgets[id(module.lora_B)]
                scaling = 4 / (module.lora_rank * a_bound * b_bound)
            expected.append((module, scaling))
        assert expected
        configure_scaling(model, optimizer)
        for module, scaling in expected:
            assert module.lora_scaling == pytest.approx(scaling)
        configured.append(True)

    step = quantizer._step
    updates = []

    def capture_step(scaler, optimizer, scheduler):
        assert scheduler.total_iters == qad_iters
        updates.append(True)
        return step(scaler, optimizer, scheduler)

    monkeypatch.setattr(quantizer, "_configure_lora_scaling", capture_scaling)
    monkeypatch.setattr(quantizer, "_step", capture_step)
    model, _ = compressor.quantize()
    assert len(configured) == 1
    assert len(updates) == qad_iters
    assert torch.isfinite(model(input_ids=tokens).logits).all()
