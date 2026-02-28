# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeMLP

from auto_round.modeling.fused_moe.replace_modules import ReplacementModuleBase
from auto_round.utils import clear_memory, unsupported_meta_device
from transformers.utils.versions import require_version
require_version("transformers>=5.2.0")

from auto_round.modeling.fused_moe.utils import _update_parameter


class LinearQwen3_5MoeSparseMoeBlock(ReplacementModuleBase):
    def __init__(self, original, config):
        super().__init__(original)
        self.gate = original.gate
        text_config = config.get_text_config()
        self.shared_expert = original.shared_expert
        with torch.device("meta"):
            self.experts = SequentialQwen3_5MoeExperts(text_config, original.experts)
        self.shared_expert_gate = original.shared_expert_gate
        self.num_experts = text_config.num_experts

    @classmethod
    def original_module_class(cls) -> str:
        """Return the class name of the module this replaces."""
        return "Qwen3_5MoeSparseMoeBlock"

    def _materialize_weights(self) -> None:
        original = self._get_original_module()
        self.experts._materialize_weights(original.experts)
        clear_memory()

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
            current_hidden_states = self.experts[expert_idx](current_state)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self.experts_forward(hidden_states_reshaped, selected_experts, routing_weights)

        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output

        expert_output += shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output

    @classmethod
    def from_original(
        cls,
        original,
        config,
        **kwargs,
    ):
        """Create an instance from the original module."""
        return cls(original, config)


class SequentialQwen3_5MoeExperts(torch.nn.ModuleList):
    def __init__(self, config, original):
        super().__init__()
        self.num_experts = original.gate_up_proj.shape[0]
        intermediate_size = config.moe_intermediate_size

        with torch.device("meta"):
            super().__init__([Qwen3_5MoeMLP(config, intermediate_size) for _ in range(self.num_experts)])

    def _materialize_weights(self, original) -> None:
        intermediate_size = original.down_proj.shape[-1]
        if not unsupported_meta_device(original):
            for i in range(self.num_experts):
                gate_up = original.gate_up_proj[i]
                down = original.down_proj[i]

                gate_proj = gate_up[:intermediate_size, :]
                up_proj = gate_up[intermediate_size:, :]

                _update_parameter(self[i].gate_proj, "weight", gate_proj.contiguous())
                _update_parameter(self[i].up_proj, "weight", up_proj.contiguous())
                _update_parameter(self[i].down_proj, "weight", down.contiguous())
            del gate_up, down, gate_proj, up_proj
            original.to_empty(device="meta")  # release original experts parameters
            clear_memory()
