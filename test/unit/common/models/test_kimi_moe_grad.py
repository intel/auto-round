import types

import pytest
import torch
from torch import nn

from auto_round.special_model_handler import update_module


class _MLP(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class _MoEGate(nn.Module):
    def __init__(self, hidden, n_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.weight = nn.Parameter(torch.randn(n_experts, hidden))

    def forward(self, x):
        scores = nn.functional.linear(x.view(-1, x.shape[-1]), self.weight).softmax(dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1)
        return topk_idx, topk_weight


class DeepseekV3MoE(nn.Module):
    """Same eval-mode forward/moe_infer structure as Kimi-VL's remote DeepseekV3MoE."""

    def __init__(self, hidden=16, inter=8, n_experts=4, top_k=2):
        super().__init__()
        self.ep_size, self.ep_rank, self.experts_per_rank = 1, 0, n_experts
        self.experts = nn.ModuleList([_MLP(hidden, inter) for _ in range(n_experts)])
        self.gate = _MoEGate(hidden, n_experts, top_k)
        self.shared_experts = _MLP(hidden, inter)

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        return y + self.shared_experts(identity)

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0).cpu().numpy()
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        outputs, start_idx = [], 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            outputs.append(self.experts[i](sorted_tokens[start_idx:end_idx]))
            start_idx = end_idx
        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        return (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )


class KimiSparseMoeBlock(DeepseekV3MoE):
    """Kimi-Linear / Kimi-K3 remote MoE block shares the same eval-mode moe_infer structure."""


def _build_model(model_type, moe_cls):
    torch.manual_seed(0)
    model = nn.Module()
    model.config = types.SimpleNamespace(model_type=model_type)
    model.mlp = moe_cls()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _routed_expert_loss(model, x):
    routed = list(model.mlp.experts.parameters())
    for p in routed:
        p.requires_grad_(True)
    with torch.enable_grad():
        loss = model.mlp(x).pow(2).mean()
    return loss, routed


@pytest.mark.parametrize(
    "model_type, moe_cls, patched",
    [
        ("kimi_vl", DeepseekV3MoE, True),
        ("kimi_linear", KimiSparseMoeBlock, True),
        ("kimi_k3", KimiSparseMoeBlock, True),
        ("deepseek_v3", DeepseekV3MoE, False),
    ],
)
def test_kimi_routed_experts_receive_grad(model_type, moe_cls, patched):
    model = _build_model(model_type, moe_cls)
    x = torch.randn(2, 8, 16)
    with torch.no_grad():
        ref = model.mlp(x)

    model = update_module(model, cleanup_original=False)
    loss, routed = _routed_expert_loss(model, x)

    if not patched:
        assert not loss.requires_grad
        return
    loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in routed)
    assert not model.training

    with torch.no_grad():
        out = model.mlp(x)
    assert not out.requires_grad
    assert torch.equal(out, ref)
