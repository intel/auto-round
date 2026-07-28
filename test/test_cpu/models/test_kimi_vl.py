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

from functools import wraps
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from auto_round.algorithms.block_runner import BlockForwardRunner
from auto_round.algorithms.quantization.sign_round.config import (
    AdamRoundConfig,
    SignRoundConfig,
    SignRoundV2Config,
)
from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer
from auto_round.compressors.utils import set_layer_config
from auto_round.modeling.kimi_vl import enable_kimi_vl_moe_grad
from auto_round.schemes import _handle_special_schemes, preset_name_to_scheme
from auto_round.utils.device_manager import device_manager

HIDDEN_SIZE = 32


class TinyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.up_proj = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.down_proj = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)

    def forward(self, hidden_states):
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class DeepseekV3MoE(nn.Module):
    """Small Kimi-shaped MoE whose eval path reproduces the remote-code decorator."""

    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([TinyExpert(), TinyExpert(), TinyExpert()])
        self.shared_experts = TinyExpert()
        self.ep_size = 1

    def forward(self, hidden_states):
        flat_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        inactive_logits = torch.full_like(flat_states[:, 0], -32)
        gate_logits = torch.stack((flat_states[:, 0], -flat_states[:, 0], inactive_logits), dim=-1).float()
        topk_weight, topk_ids = torch.softmax(gate_logits, dim=-1).topk(2, dim=-1)
        routed_states = self.moe_infer(flat_states, topk_ids, topk_weight).reshape_as(hidden_states)
        return routed_states + self.shared_experts(hidden_states)

    @torch.no_grad()
    def moe_infer(self, hidden_states, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = hidden_states[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for expert_idx, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[expert_idx]
            outputs.append(expert(sorted_tokens[start_idx:end_idx]))
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        return (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )


class DeepseekV3MoESubclass(DeepseekV3MoE):
    pass


class DeepseekMoE(nn.Module):
    """Original DeepSeek-style scatter-reduce body, intentionally out of scope."""

    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([TinyExpert(), TinyExpert(), TinyExpert()])
        self.num_experts_per_tok = 2

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok

        for expert_idx, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if expert_idx == 0 else tokens_per_expert[expert_idx - 1]
            if start_idx == end_idx:
                continue

            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_out = self.experts[expert_idx](x[exp_token_idx])
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(
                0,
                exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out,
                reduce="sum",
            )
        return expert_cache


class UnrelatedMoE(nn.Module):
    @torch.no_grad()
    def moe_infer(self, hidden_states):
        return hidden_states


class MultipleMoEBlocks(nn.Module):
    def __init__(self):
        super().__init__()
        self.moes = nn.ModuleList([DeepseekV3MoE(), DeepseekV3MoE()])


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.kv_b_proj = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.o_proj = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)


class TinyDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttention()
        self.mlp = DeepseekV3MoE()

    def forward(self, hidden_states):
        return self.mlp(hidden_states)


class TinyTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TinyDecoderLayer()])


class TinyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyTextModel()


class TinyVisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)


class TinyMultimodalProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)


class TinyKimiVL(nn.Module):
    def __init__(self, model_type="kimi_vl"):
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type, tie_word_embeddings=False)
        self.language_model = TinyLanguageModel()
        self.vision_tower = TinyVisionTower()
        self.multi_modal_projector = TinyMultimodalProjector()
        self.lm_head = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)


class RecordingSignRoundQuantizer(SignRoundQuantizer):
    """Record the actual block loss and optimizer state around one SignSGD step."""

    def _get_loss(self, *args, **kwargs):
        loss = super()._get_loss(*args, **kwargs)
        self.recorded_loss_requires_grad = loss.requires_grad
        return loss

    def _step(self, scaler, optimizer, lr_schedule):
        self.recorded_grads = {}
        self.params_before_step = {}
        for module_name, module in self.recorded_wrappers.items():
            for param_name, parameter in module.params.items():
                key = f"{module_name}.{param_name}"
                self.recorded_grads[key] = None if parameter.grad is None else parameter.grad.detach().clone()
                self.params_before_step[key] = parameter.detach().clone()

        super()._step(scaler, optimizer, lr_schedule)
        self.params_after_step = {
            f"{module_name}.{param_name}": parameter.detach().clone()
            for module_name, module in self.recorded_wrappers.items()
            for param_name, parameter in module.params.items()
        }


def _make_model(model_type="kimi_vl"):
    torch.manual_seed(7)
    return TinyKimiVL(model_type=model_type).eval()


def _configure_scheme(model, scheme_name):
    layer_config = _handle_special_schemes(
        scheme_name,
        {},
        model,
        supported_types=(nn.Linear,),
        inner_supported_types=(),
        mllm=True,
    )
    layer_config, _, _ = set_layer_config(
        model,
        layer_config,
        default_scheme=preset_name_to_scheme(scheme_name),
        default_scale_dtype=torch.float32,
        supported_types=(nn.Linear,),
        inner_supported_types=(),
        quant_block_list=[["language_model.model.layers.0"]],
        is_mllm=True,
    )
    return layer_config


def _record_kimi_vl_scope_during_wrapper(quantizer):
    wrap_block = quantizer.wrapper_block
    if getattr(wrap_block, "_records_kimi_vl_scope", False):
        return

    @wraps(wrap_block)
    def record_wrappers(*args, **kwargs):
        block = args[0]
        moe_state = block.mlp.__dict__.get("_auto_round_kimi_vl_moe_grad_state")
        quantizer.recorded_moe_grad_depth = getattr(moe_state, "depth", 0)
        quantizer.recorded_moe_infer = block.mlp.moe_infer.__func__
        quantizer.recorded_moe_grad_modules = tuple(
            module for module in quantizer.model.modules() if "_auto_round_kimi_vl_moe_grad_state" in module.__dict__
        )
        result = wrap_block(*args, **kwargs)
        quantizer.recorded_wrappers = {
            name: module for name, module in block.named_modules() if hasattr(module, "orig_layer")
        }
        return result

    record_wrappers._records_kimi_vl_scope = True
    quantizer.wrapper_block = record_wrappers


def _make_quantizer(model, quantizer_cls=RecordingSignRoundQuantizer, config_cls=SignRoundConfig):
    config = config_cls(
        bits=4,
        group_size=32,
        sym=True,
        data_type="int",
        act_bits=16,
        act_group_size=-1,
        act_sym=True,
        act_data_type="float",
        act_dynamic=True,
        iters=1,
        lr=0.1,
        minmax_lr=0.1,
        not_use_best_mse=True,
    )
    quantizer = quantizer_cls(config)
    orchestrator = SimpleNamespace(
        model_context=SimpleNamespace(
            model=model,
            amp=True,
            amp_dtype=torch.float32,
            is_diffusion=False,
        ),
        compress_context=SimpleNamespace(
            low_gpu_mem_usage=False,
            enable_torch_compile=False,
            cache_device="cpu",
            clear_memory=lambda: None,
        ),
        calibration_context=SimpleNamespace(batch_size=1, batch_dim=0),
        scale_dtype=torch.float32,
        scheme_context=config.scheme,
    )
    quantizer.bind(orchestrator)
    quantizer.bind_block_forward_runner(
        BlockForwardRunner(batch_dim=0, batch_size=1, device="cpu", cache_device="cpu", amp=False)
    )
    _record_kimi_vl_scope_during_wrapper(quantizer)
    return quantizer


def test_kimi_vl_signround_without_prepare_run_updates_active_routed_experts():
    device_manager.configure("cpu")
    model = _make_model()
    _configure_scheme(model, "W4A16_MIXED")
    block = model.language_model.model.layers[0]
    moe = block.mlp
    original_moe_infer = moe.moe_infer.__func__

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    sample = torch.linspace(-1.0, 1.0, steps=2 * HIDDEN_SIZE).reshape(2, HIDDEN_SIZE)
    reference = block(sample).unsqueeze(0).detach()
    quantizer = _make_quantizer(model)
    quantizer.quantize_block(
        block,
        fp_inputs=[sample],
        input_others={},
        fp_outputs=[reference],
        q_inputs=None,
        block_ctx=SimpleNamespace(),
    )

    assert quantizer.recorded_moe_grad_depth == 1
    assert quantizer.recorded_moe_infer is original_moe_infer.__wrapped__
    assert quantizer.recorded_loss_requires_grad
    assert len(quantizer.recorded_wrappers) == 9
    assert all(".experts." in name and "shared" not in name for name in quantizer.recorded_wrappers)

    active_keys = [key for key in quantizer.recorded_grads if ".experts.0." in key or ".experts.1." in key]
    inactive_keys = [key for key in quantizer.recorded_grads if ".experts.2." in key]
    assert len(active_keys) == 18
    assert len(inactive_keys) == 9

    for key in active_keys:
        gradient = quantizer.recorded_grads[key]
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0
    assert any(
        not torch.equal(quantizer.params_before_step[key], quantizer.params_after_step[key]) for key in active_keys
    )

    assert all(quantizer.recorded_grads[key] is None for key in inactive_keys)
    assert all(
        torch.equal(quantizer.params_before_step[key], quantizer.params_after_step[key]) for key in inactive_keys
    )
    assert not any(module.training for module in model.modules())
    assert "moe_infer" not in moe.__dict__
    assert moe.moe_infer.__func__ is original_moe_infer


def test_kimi_vl_moe_grad_context_preserves_output_and_restores_method():
    model = _make_model().to(torch.bfloat16)
    block = model.language_model.model.layers[0]
    moe = block.mlp
    sample = torch.linspace(-0.5, 0.5, steps=2 * HIDDEN_SIZE, dtype=torch.bfloat16).reshape(2, HIDDEN_SIZE)
    original_method = moe.moe_infer.__func__
    with torch.no_grad():
        decorated_output = block(sample)

    with enable_kimi_vl_moe_grad(model, block, iters=1):
        assert "moe_infer" in moe.__dict__
        assert moe.moe_infer.__func__ is original_method.__wrapped__
        grad_enabled_output = block(sample)
        assert grad_enabled_output.requires_grad
        assert not any(module.training for module in model.modules())

    torch.testing.assert_close(grad_enabled_output, decorated_output)
    grad_enabled_output.float().square().sum().backward()
    for expert in moe.experts[:2]:
        for parameter in expert.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert torch.count_nonzero(parameter.grad) > 0
    assert all(parameter.grad is None for parameter in moe.experts[2].parameters())
    assert "moe_infer" not in moe.__dict__
    assert moe.moe_infer.__func__ is original_method

    with pytest.raises(RuntimeError, match="expected test error"):
        with enable_kimi_vl_moe_grad(model, block, iters=1):
            raise RuntimeError("expected test error")

    assert "moe_infer" not in moe.__dict__
    assert moe.moe_infer.__func__ is original_method


def test_kimi_vl_moe_grad_context_manager_explicit_reentrancy_contract():
    """Explicit nesting tests the context-manager contract, not the product flow."""
    model = _make_model()
    block = model.language_model.model.layers[0]
    moe = block.mlp
    original_method = moe.moe_infer.__func__

    with enable_kimi_vl_moe_grad(model, block, iters=1):
        raw_method = moe.moe_infer.__func__
        state = moe.__dict__["_auto_round_kimi_vl_moe_grad_state"]
        assert state.depth == 1
        with enable_kimi_vl_moe_grad(model, block, iters=1):
            assert moe.moe_infer.__func__ is raw_method
            assert state.depth == 2
        assert moe.moe_infer.__func__ is raw_method
        assert state.depth == 1

    assert "moe_infer" not in moe.__dict__
    assert moe.moe_infer.__func__ is original_method


@pytest.mark.parametrize(
    ("model_type", "iters"),
    [
        ("deepseek_v3", 1),
        ("kimi_vl", 0),
    ],
)
def test_kimi_vl_moe_grad_context_is_scoped(model_type, iters):
    model = _make_model(model_type=model_type)
    block = model.language_model.model.layers[0]
    moe = block.mlp
    original_method = moe.moe_infer.__func__

    with enable_kimi_vl_moe_grad(model, block, iters=iters):
        assert "moe_infer" not in moe.__dict__
        assert moe.moe_infer.__func__ is original_method


def test_kimi_vl_moe_grad_context_requires_wrapped_method():
    model = _make_model()
    block = model.language_model.model.layers[0]
    moe = block.mlp
    original_method = moe.moe_infer.__func__
    moe.moe_infer = MethodType(original_method.__wrapped__, moe)

    with pytest.raises(RuntimeError, match=r"callable moe_infer\.__wrapped__ is unavailable") as exc_info:
        with enable_kimi_vl_moe_grad(model, block, iters=1):
            pass

    assert "iters=0 (RTN)" in str(exc_info.value)
    assert "remote-code revision" in str(exc_info.value)
    assert moe.moe_infer.__func__ is original_method.__wrapped__


def test_kimi_vl_moe_grad_context_rejects_non_grad_mode_wrapper():
    model = _make_model()
    block = model.language_model.model.layers[0]
    moe = block.mlp
    raw_method = moe.moe_infer.__func__.__wrapped__

    @wraps(raw_method)
    def generic_wrapper(*args, **kwargs):
        return raw_method(*args, **kwargs)

    moe.moe_infer = MethodType(generic_wrapper, moe)
    with pytest.raises(RuntimeError, match="not a recognized torch grad-mode decorator") as exc_info:
        with enable_kimi_vl_moe_grad(model, block, iters=1):
            pass

    assert "iters=0 (RTN)" in str(exc_info.value)
    assert "remote-code revision" in str(exc_info.value)
    assert moe.moe_infer.__func__ is generic_wrapper


def test_kimi_vl_prepare_run_rejects_non_grad_mode_wrapper_before_block_iteration():
    model = _make_model()
    moe = model.language_model.model.layers[0].mlp
    raw_method = moe.moe_infer.__func__.__wrapped__

    @wraps(raw_method)
    def generic_wrapper(*args, **kwargs):
        return raw_method(*args, **kwargs)

    moe.moe_infer = MethodType(generic_wrapper, moe)
    quantizer = _make_quantizer(model)
    with pytest.raises(RuntimeError, match="not a recognized torch grad-mode decorator"):
        quantizer.prepare_run()

    assert "_auto_round_kimi_vl_moe_grad_state" not in moe.__dict__
    assert moe.moe_infer.__func__ is generic_wrapper


def test_kimi_vl_moe_grad_context_accepts_inference_mode_wrapper():
    model = _make_model()
    block = model.language_model.model.layers[0]
    moe = block.mlp
    raw_method = moe.moe_infer.__func__.__wrapped__
    inference_wrapper = torch.inference_mode()(raw_method)
    moe.moe_infer = MethodType(inference_wrapper, moe)

    with enable_kimi_vl_moe_grad(model, block, iters=1):
        assert moe.moe_infer.__func__ is raw_method

    assert moe.moe_infer.__func__ is inference_wrapper


def test_kimi_vl_moe_grad_context_rejects_expert_parallel_path():
    model = _make_model()
    block = model.language_model.model.layers[0]
    moe = block.mlp
    original_method = moe.moe_infer.__func__
    moe.ep_size = 2

    with pytest.raises(RuntimeError, match="expert-parallel all_to_all") as exc_info:
        with enable_kimi_vl_moe_grad(model, block, iters=1):
            pass

    assert "ep_size=1" in str(exc_info.value)
    assert "iters=0 (RTN)" in str(exc_info.value)
    assert "remote-code revision" in str(exc_info.value)
    assert "moe_infer" not in moe.__dict__
    assert moe.moe_infer.__func__ is original_method


def test_kimi_vl_prepare_run_rejects_expert_parallel_path_before_block_iteration():
    model = _make_model()
    moe = model.language_model.model.layers[0].mlp
    original_method = moe.moe_infer.__func__
    moe.ep_size = 2
    quantizer = _make_quantizer(model)

    with pytest.raises(RuntimeError, match="expert-parallel all_to_all"):
        quantizer.prepare_run()

    assert "_auto_round_kimi_vl_moe_grad_state" not in moe.__dict__
    assert moe.moe_infer.__func__ is original_method


def test_kimi_vl_moe_grad_context_supports_deepseek_v3_moe_subclasses():
    model = _make_model()
    block = model.language_model.model.layers[0]
    block.mlp = DeepseekV3MoESubclass().eval()
    original_method = block.mlp.moe_infer.__func__

    with enable_kimi_vl_moe_grad(model, block, iters=1):
        assert block.mlp.moe_infer.__func__ is original_method.__wrapped__

    assert "moe_infer" not in block.mlp.__dict__
    assert block.mlp.moe_infer.__func__ is original_method


def test_kimi_vl_moe_grad_context_ignores_unrelated_moe_classes():
    model = _make_model()
    block = UnrelatedMoE().eval()
    original_method = block.moe_infer.__func__

    with enable_kimi_vl_moe_grad(model, block, iters=1):
        assert "moe_infer" not in block.__dict__
    assert block.moe_infer.__func__ is original_method


def test_kimi_vl_moe_grad_context_excludes_deepseek_scatter_reduce_moe():
    model = _make_model()
    block = DeepseekMoE().eval()
    original_method = block.moe_infer.__func__
    hidden_states = torch.randn(2, HIDDEN_SIZE)
    flat_expert_indices = torch.tensor([0, 2, 0, 2])
    flat_expert_weights = torch.tensor([[0.6], [0.4], [0.7], [0.3]])
    output = original_method.__wrapped__(
        block,
        hidden_states,
        flat_expert_indices,
        flat_expert_weights,
    )

    with pytest.raises(RuntimeError, match="ScatterReduceBackward0"):
        output.sum().backward()

    with enable_kimi_vl_moe_grad(model, block, iters=1):
        assert "moe_infer" not in block.__dict__
        assert block.moe_infer.__func__ is original_method


def test_kimi_vl_moe_grad_context_validates_all_targets_before_patching():
    model = _make_model()
    block = MultipleMoEBlocks().eval()
    first, second = block.moes
    first_original = first.moe_infer.__func__
    second_original = second.moe_infer.__func__
    second.moe_infer = MethodType(second_original.__wrapped__, second)

    with pytest.raises(RuntimeError, match=r"callable moe_infer\.__wrapped__ is unavailable"):
        with enable_kimi_vl_moe_grad(model, block, iters=1):
            pass

    assert "moe_infer" not in first.__dict__
    assert first.moe_infer.__func__ is first_original
    assert second.moe_infer.__func__ is second_original.__wrapped__


def test_kimi_vl_moe_grad_context_patches_and_restores_multiple_targets():
    model = _make_model()
    block = MultipleMoEBlocks().eval()
    original_methods = [moe.moe_infer.__func__ for moe in block.moes]

    with enable_kimi_vl_moe_grad(model, block, iters=1):
        for moe, original_method in zip(block.moes, original_methods):
            assert moe.moe_infer.__func__ is original_method.__wrapped__

    for moe, original_method in zip(block.moes, original_methods):
        assert "moe_infer" not in moe.__dict__
        assert moe.moe_infer.__func__ is original_method


def test_kimi_vl_quantize_block_restores_method_on_error():
    model = _make_model()
    block = model.language_model.model.layers[0]
    moe = block.mlp
    original_method = moe.moe_infer.__func__
    quantizer = _make_quantizer(model)
    quantizer.prepare_run()

    def raise_expected_error(*args, **kwargs):
        raise RuntimeError("expected quantize error")

    quantizer.wrapper_block = raise_expected_error
    with pytest.raises(RuntimeError, match="expected quantize error"):
        quantizer.quantize_block(
            block,
            fp_inputs=[],
            input_others={},
            fp_outputs=[],
            q_inputs=None,
            block_ctx=SimpleNamespace(),
        )

    assert "moe_infer" not in moe.__dict__
    assert moe.moe_infer.__func__ is original_method
    quantizer.finalize_run()


def test_kimi_vl_lazy_scope_does_not_patch_sibling_block():
    device_manager.configure("cpu")
    model = _make_model()
    model.language_model.model.layers.append(TinyDecoderLayer())
    _configure_scheme(model, "W4A16_MIXED")
    block = model.language_model.model.layers[0]
    moe = block.mlp
    sibling_moe = model.language_model.model.layers[1].mlp
    original_method = moe.moe_infer.__func__
    sibling_original_method = sibling_moe.moe_infer.__func__
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    sample = torch.linspace(-1.0, 1.0, steps=2 * HIDDEN_SIZE).reshape(2, HIDDEN_SIZE)
    reference = block(sample).unsqueeze(0).detach()
    quantizer = _make_quantizer(model)
    quantizer.quantize_block(
        block,
        fp_inputs=[sample],
        input_others={},
        fp_outputs=[reference],
        q_inputs=None,
        block_ctx=SimpleNamespace(),
    )

    assert quantizer.recorded_moe_grad_depth == 1
    assert quantizer.recorded_moe_grad_modules == (moe,)
    assert quantizer.recorded_moe_infer is original_method.__wrapped__
    assert "moe_infer" not in moe.__dict__
    assert moe.moe_infer.__func__ is original_method
    assert "moe_infer" not in sibling_moe.__dict__
    assert sibling_moe.moe_infer.__func__ is sibling_original_method


@pytest.mark.parametrize(
    ("quantizer_cls", "config_cls"),
    [
        pytest.param("adam", AdamRoundConfig, id="adam-round"),
        pytest.param("signround-v2", SignRoundV2Config, id="signround-v2"),
    ],
)
def test_signround_variants_apply_and_restore_kimi_vl_gradient_scope(quantizer_cls, config_cls):
    from auto_round.algorithms.quantization.adam_round.adam import AdamRoundQuantizer
    from auto_round.algorithms.quantization.sign_roundv2.quantizer import SignRoundV2Quantizer

    variant_cls = {
        "adam": AdamRoundQuantizer,
        "signround-v2": SignRoundV2Quantizer,
    }[quantizer_cls]
    device_manager.configure("cpu")
    model = _make_model()
    _configure_scheme(model, "W4A16_MIXED")
    block = model.language_model.model.layers[0]
    moe = block.mlp
    original_method = moe.moe_infer.__func__
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    sample = torch.linspace(-1.0, 1.0, steps=2 * HIDDEN_SIZE).reshape(2, HIDDEN_SIZE)
    reference = block(sample).unsqueeze(0).detach()
    quantizer = _make_quantizer(model, quantizer_cls=variant_cls, config_cls=config_cls)
    quantizer.prepare_run()
    _record_kimi_vl_scope_during_wrapper(quantizer)
    quantizer.quantize_block(
        block,
        fp_inputs=[sample],
        input_others={},
        fp_outputs=[reference],
        q_inputs=None,
        block_ctx=SimpleNamespace(),
    )
    quantizer.finalize_run()

    assert quantizer.recorded_moe_grad_depth == 1
    assert quantizer.recorded_moe_infer is original_method.__wrapped__
    assert "moe_infer" not in moe.__dict__
    assert moe.moe_infer.__func__ is original_method


@pytest.mark.parametrize(
    ("scheme_name", "non_routed_bits"),
    [
        ("W4A16", 4),
        ("W4A16_MIXED", 16),
    ],
)
def test_kimi_vl_w4a16_mixed_keeps_non_routed_modules_unquantized(scheme_name, non_routed_bits):
    model = _make_model()
    layer_config = _configure_scheme(model, scheme_name)
    block = model.language_model.model.layers[0]

    assert block.mlp.experts[0].gate_proj.bits == 4
    assert block.mlp.experts[0].gate_proj.data_type == "int"
    assert block.mlp.shared_experts.gate_proj.bits == non_routed_bits
    assert block.self_attn.q_proj.bits == non_routed_bits
    assert block.self_attn.kv_a_proj_with_mqa.bits == non_routed_bits
    assert block.self_attn.kv_b_proj.bits == non_routed_bits
    assert block.self_attn.o_proj.bits == non_routed_bits
    assert "lm_head" not in layer_config
    assert not hasattr(model.lm_head, "bits")

    if scheme_name == "W4A16_MIXED":
        assert model.vision_tower.proj.bits == 16
        assert model.multi_modal_projector.proj.bits == 16
