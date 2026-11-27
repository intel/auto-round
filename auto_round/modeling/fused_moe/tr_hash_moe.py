# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Expose the TR-HASH HF checkpoint's routed and attention projections to AutoRound."""

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache

from auto_round.modeling.fused_moe.replace_modules import ReplacementModuleBase
from auto_round.modeling.fused_moe.utils import _update_parameter

TR_HASH_PROJECTIONS = {"expert_gate": "gate_proj", "expert_up": "up_proj", "expert_down": "down_proj"}


def tr_hash_linear_weight(weight: torch.Tensor, index: int) -> torch.Tensor:
    # Native experts use x @ W[index]; Linear stores (out_features, in_features).
    return weight[index].T.contiguous()


def _apply_rotary(states, cos, sin):
    left, right = states.chunk(2, dim=-1)
    return states * cos + torch.cat((-right, left), dim=-1) * sin


class LinearTRHashAttention(ReplacementModuleBase):
    """Use module calls for Q/K/V so tuning wrappers and packed kernels both run."""

    def __init__(self, original):
        super().__init__(original)
        self.layer_idx = original.layer_idx
        self.head_dim = original.head_dim
        self.num_key_value_groups = original.num_key_value_groups
        self.dropout = original.dropout
        self.q_proj = original.q_proj
        self.k_proj = original.k_proj
        self.v_proj = original.v_proj
        self.o_proj = original.o_proj
        self.q_norm = original.q_norm
        self.k_norm = original.k_norm
        self.rotary_emb = original.rotary_emb
        self.train(original.training)

    @classmethod
    def original_module_class(cls) -> str:
        return "TRHashAttention"

    @classmethod
    def from_original(cls, original, config):
        return cls(original)

    def forward(self, hidden_states, attention_mask, position_ids, past_key_values, use_cache, cache_position):
        query = self.q_proj(hidden_states).unflatten(-1, (-1, self.head_dim)).transpose(1, 2)
        key = self.k_proj(hidden_states).unflatten(-1, (-1, self.head_dim)).transpose(1, 2)
        value = self.v_proj(hidden_states).unflatten(-1, (-1, self.head_dim)).transpose(1, 2)
        if self.q_norm is not None:
            query, key = self.q_norm(query), self.k_norm(key)
        cos, sin = (t.unsqueeze(1) for t in self.rotary_emb(position_ids, query.dtype))
        query, key = _apply_rotary(query, cos, sin), _apply_rotary(key, cos, sin)

        if isinstance(past_key_values, Cache):
            key, value = past_key_values.update(key, value, self.layer_idx, {"cache_position": cache_position})
            next_cache = past_key_values
        else:
            if past_key_values is not None:
                previous_key, previous_value = past_key_values
                key = torch.cat((previous_key, key), dim=-2)
                value = torch.cat((previous_value, value), dim=-2)
            next_cache = (key, value)

        attn = F.scaled_dot_product_attention(
            query,
            key.repeat_interleave(self.num_key_value_groups, dim=1),
            value.repeat_interleave(self.num_key_value_groups, dim=1),
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=attention_mask is None and query.shape[-2] == key.shape[-2],
        )
        return self.o_proj(attn.transpose(1, 2).flatten(-2)), next_cache if use_cache else None


class LinearTRHashExpertEngine(ReplacementModuleBase):
    def __init__(self, original):
        super().__init__(original)
        self.num_experts = original.num_experts
        self.top_k = original.top_k
        self.vocab_size = original.vocab_size
        self.route_weights = original.route_weights
        self.shared_output_scale = original.shared_output_scale
        self.routed_output_scale = original.routed_output_scale
        self.shared_gate = original.shared_gate
        self.shared_up = original.shared_up
        self.shared_down = original.shared_down
        for name in ("route_table", "fused_route_codes", "fused_expert_pairs"):
            self.register_buffer(name, getattr(original, name))

        self.experts = nn.ModuleList(nn.Module() for _ in range(self.num_experts))
        for source, projection in TR_HASH_PROJECTIONS.items():
            weight = getattr(original, source)
            _, in_features, out_features = weight.shape
            for expert in self.experts:
                expert.add_module(
                    projection, nn.Linear(in_features, out_features, bias=False, device="meta", dtype=weight.dtype)
                )

    @classmethod
    def original_module_class(cls) -> str:
        return "TRHashExpertEngine"

    @classmethod
    def from_original(cls, original, config):
        return cls(original)

    def _materialize_weights(self):
        original = self._get_original_module()
        if original is None:
            return  # The inference skeleton reads the packed weights instead.
        for source, projection in TR_HASH_PROJECTIONS.items():
            weight = getattr(original, source)
            if not weight.is_meta:
                for index, expert in enumerate(self.experts):
                    _update_parameter(getattr(expert, projection), "weight", tr_hash_linear_weight(weight, index))
                # The original engine is not used after replacement; release just
                # its stacked storage, without touching shared modules or buffers.
                setattr(original, source, nn.Parameter(torch.empty_like(weight, device="meta"), weight.requires_grad))

    def forward(self, hidden_states, token_ids):
        flat_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        if self.shared_gate is None:
            shared = torch.zeros_like(flat_states)
        else:
            shared = self.shared_down(F.silu(self.shared_gate(flat_states)) * self.shared_up(flat_states))

        routes = self.route_table[:, token_ids.clamp(0, self.vocab_size - 1)].reshape(self.top_k, -1)
        route_weights = flat_states.new_tensor(self.route_weights).view(-1, 1)
        routed = torch.zeros_like(flat_states)
        for index, expert in enumerate(self.experts):
            token_weight = (routes.eq(index).to(flat_states.dtype) * route_weights).sum(dim=0)
            active_states = flat_states * token_weight.ne(0).to(flat_states.dtype).unsqueeze(-1)
            intermediate = F.silu(expert.gate_proj(active_states)) * expert.up_proj(active_states)
            routed.add_(expert.down_proj(intermediate) * token_weight.unsqueeze(-1))

        return (self.shared_output_scale * shared + self.routed_output_scale * routed).reshape_as(hidden_states)
