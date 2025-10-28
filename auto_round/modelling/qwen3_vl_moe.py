# Copyright (c) 2025 Intel Corporation
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
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_utils import no_init_weights as skip_weights_initialize
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

# from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP
from transformers.models.qwen3_vl_moe.modular_qwen3_vl_moe import Qwen3VLMoeTextExperts, Qwen3VLMoeTextSparseMoeBlock

__all__ = ["get_replacement_info"]


class QwenSingleExpert(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.intermediate_size = config.moe_intermediate_size
        self.hidden_size = config.hidden_size
        self.activation_fn = ACT2FN[config.hidden_act]

        # Combined gate/up projection
        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expert forward pass with combined gate/up projection"""
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up[:, : self.intermediate_size], gate_up[:, self.intermediate_size :]
        activated = self.activation_fn(gate) * up
        return self.down_proj(activated)


class Qwen3VLSequentialMoeTextExperts(nn.Module):
    def __init__(self, config, original: Qwen3VLMoeTextExperts):
        super().__init__()
        self.num_experts = config.num_experts
        self.intermediate_size = config.moe_intermediate_size
        self.hidden_size = config.hidden_size
        self.act_fn = ACT2FN[config.hidden_act]

        # Create individual expert modules
        new_module = nn.ModuleList(
            [
                QwenSingleExpert(
                    hidden_size=self.hidden_size, intermediate_size=self.intermediate_size, activation_fn=self.act_fn
                )
                for _ in range(self.num_experts)
            ]
        )
        dtype = original.gate_up_proj.dtype
        new_module = new_module.to(dtype)
        # Transfer weights from original module
        device = next(original.parameters()).device
        # For meta device, only set the experts without copying weights
        if device == torch.device("meta"):
            self.experts = new_module
            return
        for i, expert in enumerate(new_module):
            # Set weights for the new expert module
            expert.gate_up_proj.weight.data.copy_(original.gate_up_proj[i].transpose(0, 1))
            expert.down_proj.weight.data.copy_(original.down_proj[i].transpose(0, 1))
        # self.experts = new_module

    def forward(
        self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, router_indices: torch.Tensor
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # [num_tokens, hidden_size]

        # Process through all experts
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(hidden_states))

        # Stack and weight expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=0)  # [num_experts, num_tokens, hidden_size]
        weighted_outputs = expert_outputs * routing_weights.t().unsqueeze(-1)
        # Note: !!! keep dtype consistency when summing
        next_states = weighted_outputs.sum(dim=0, dtype=hidden_states.dtype)  # [num_tokens, hidden_size]
        return next_states.reshape(batch_size, -1, self.hidden_size)


class Qwen3VLSequentialMoeTextSparseMoeBlock(nn.Module):
    def __init__(self, config, original):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.gate = original.gate
        # self.experts = Qwen3VLMoeTextExperts(config)

        # Create individual expert modules

        target_device = next(original.experts.parameters()).device
        with skip_weights_initialize(), torch.device(target_device):
            new_module = nn.ModuleList([QwenSingleExpert(config) for _ in range(self.num_experts)])
        original_experts = original.experts
        dtype = original_experts.gate_up_proj.dtype
        new_module = new_module.to(dtype)
        # Transfer weights from original module
        device = next(original_experts.parameters()).device
        # For meta device, only set the experts without copying weights
        if device == torch.device("meta"):
            self.experts = new_module
            return
        for i, expert in enumerate(new_module):
            # Set weights for the new expert module
            expert.gate_up_proj.weight.data.copy_(original_experts.gate_up_proj[i].transpose(0, 1))
            expert.down_proj.weight.data.copy_(original_experts.down_proj[i].transpose(0, 1))
        self.experts = new_module
        # since all the models use norm_topk_prob, we don't need to have a extra check for it
        # self.norm_topk_prob = config.norm_topk_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.gate(hidden_states)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        router_weights = torch.zeros_like(router_logits).scatter_(1, router_indices, routing_weights)
        hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_size)
        # routed_out = self.experts(hidden_states, router_weights, router_indices)
        # Start of expert processing
        routing_weights = router_weights
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # [num_tokens, hidden_size]

        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(hidden_states))

        # Stack and weight expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=0)  # [num_experts, num_tokens, hidden_size]
        weighted_outputs = expert_outputs * routing_weights.t().unsqueeze(-1)
        # Note: !!! keep dtype consistency when summing
        next_states = weighted_outputs.sum(dim=0, dtype=hidden_states.dtype)  # [num_tokens, hidden_size]
        routed_out = next_states.reshape(batch_size, -1, self.hidden_size)
        # End of expert processing
        return routed_out, router_logits


def get_replacement_info(config):
    return (
        Qwen3VLSequentialMoeTextSparseMoeBlock,
        config.get_text_config(),
        Qwen3VLMoeTextSparseMoeBlock.__name__,
    )
