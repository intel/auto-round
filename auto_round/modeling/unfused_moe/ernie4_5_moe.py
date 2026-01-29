import torch
import torch.nn as nn


class LinearErnie4_5_MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers.models.ernie4_5_moe.modeling_ernie4_5_moe import Ernie4_5_MoeTopKRouter, Ernie4_5_MoeMLP

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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_dim)

        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)

        _, top_k_index, top_k_weights = self.gate(hidden_states)
        final_hidden_states = self.experts_forward(hidden_states, top_k_index, top_k_weights)

        if self.shared_experts is not None:
            final_hidden_states = final_hidden_states + shared_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, self.hidden_dim)
        return final_hidden_states.to(hidden_states.dtype)