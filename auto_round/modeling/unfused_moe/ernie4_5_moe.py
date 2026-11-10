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

from auto_round.modeling.fused_moe.utils import sequential_moe_forward


class LinearErnie4_5_MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers.models.ernie4_5_moe.modeling_ernie4_5_moe import Ernie4_5_MoeMLP, Ernie4_5_MoeTopKRouter

        self.hidden_dim = config.hidden_size
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_k
        self.gate = Ernie4_5_MoeTopKRouter(config)
        self.experts = nn.ModuleList(
            [Ernie4_5_MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )

        self.shared_experts = None
        if config.moe_num_shared_experts > 0:
            self.shared_experts = Ernie4_5_MoeMLP(config, config.moe_intermediate_size * config.moe_num_shared_experts)

    def experts_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        return sequential_moe_forward(hidden_states, top_k_index, top_k_weights, self.experts, self.num_experts)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_dim)

        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)

        # transformers' Ernie4_5_MoeTopKRouter.forward returns
        # (router_logits, routing_weights, selected_experts); unpack weights
        # and indices in that order to match the current transformers API.
        _, top_k_weights, top_k_index = self.gate(hidden_states)
        final_hidden_states = self.experts_forward(hidden_states, top_k_index, top_k_weights)

        if self.shared_experts is not None:
            final_hidden_states = final_hidden_states + shared_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, self.hidden_dim)
        return final_hidden_states.to(hidden_states.dtype)
