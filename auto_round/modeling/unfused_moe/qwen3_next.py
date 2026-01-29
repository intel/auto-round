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

import torch
import torch.nn as nn
from torch.nn import functional as F


class LinearQwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers.models.qwen3_next.modeling_qwen3_next import (
            Qwen3NextExperts,
            Qwen3NextMLP,
            Qwen3NextTopKRouter,
        )

        self.gate = Qwen3NextTopKRouter(config)
        self.num_experts = config.num_experts  # needed
        self.experts = nn.ModuleList(
            [Qwen3NextMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )
        self.experts = Qwen3NextExperts(config)
        self.shared_expert = Qwen3NextMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def expert_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            # gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            # current_hidden_states = self.act_fn(gate) * up
            # current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            expert_layer = self.experts[expert_idx]
            current_hidden_states = expert_layer(current_state)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self.expert_forward(hidden_states_reshaped, selected_experts, routing_weights)

        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output

        expert_output += shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output
