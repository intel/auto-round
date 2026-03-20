# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MoE module replacements for Qwen3-Omni model.

Qwen3-Omni has MoE blocks in both thinker and talker:
- Thinker: Qwen3OmniMoeThinkerTextSparseMoeBlock (experts + gate, no shared expert)
- Talker: Qwen3OmniMoeTalkerTextSparseMoeBlock (experts + gate + shared_expert + shared_expert_gate)

This module provides replacement classes that unfuse fused expert weights (3D Parameters)
into individual nn.Linear layers, enabling per-expert quantization with meta device optimization.
"""

import torch

from auto_round.modeling.fused_moe.replace_modules import ReplacementModuleBase
from auto_round.modeling.fused_moe.utils import _update_parameter
from auto_round.utils import clear_memory, unsupported_meta_device

# ---------------------------------------------------------------------------
# Thinker MoE replacement (no shared expert)
# ---------------------------------------------------------------------------


class LinearQwen3OmniThinkerSparseMoeBlock(ReplacementModuleBase):
    """Calibration replacement for Qwen3OmniMoeThinkerTextSparseMoeBlock.

    Unfuses fused expert weights into individual nn.Linear layers for
    per-expert quantization.  Uses meta device to avoid doubling memory.

    Structure: gate (router) + experts (unfused).
    """

    def __init__(self, original, config):
        super().__init__(original)
        self.gate = original.gate
        self.num_experts = original.experts.num_experts
        text_config = config.thinker_config.text_config
        with torch.device("meta"):
            self.experts = SequentialQwen3OmniThinkerExperts(text_config, original.experts)

    @classmethod
    def original_module_class(cls) -> str:
        return "Qwen3OmniMoeThinkerTextSparseMoeBlock"

    def _materialize_weights(self) -> None:
        original = self._get_original_module()
        self.experts._materialize_weights(original.experts)
        clear_memory()

    def experts_forward(self, hidden_states, top_k_index, top_k_weights):
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
            current_hidden_states = self.experts[expert_idx](current_state)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self.experts_forward(hidden_states_reshaped, selected_experts, routing_weights)
        return expert_output.reshape(batch_size, sequence_length, hidden_dim)

    @classmethod
    def from_original(cls, original, config, **kwargs):
        return cls(original, config)


# ---------------------------------------------------------------------------
# Talker MoE replacement (with shared expert, same pattern as qwen3_5_moe)
# ---------------------------------------------------------------------------


class LinearQwen3OmniTalkerSparseMoeBlock(ReplacementModuleBase):
    """Calibration replacement for Qwen3OmniMoeTalkerTextSparseMoeBlock.

    Unfuses fused expert weights and preserves the shared_expert + shared_expert_gate.
    Similar to Qwen3.5-MoE pattern.

    Structure: gate (router) + experts (unfused) + shared_expert + shared_expert_gate.
    """

    def __init__(self, original, config):
        super().__init__(original)
        self.gate = original.gate
        self.shared_expert = original.shared_expert
        self.shared_expert_gate = original.shared_expert_gate
        self.num_experts = original.experts.num_experts
        text_config = config.talker_config.text_config
        with torch.device("meta"):
            self.experts = SequentialQwen3OmniTalkerExperts(text_config, original.experts)

    @classmethod
    def original_module_class(cls) -> str:
        return "Qwen3OmniMoeTalkerTextSparseMoeBlock"

    def _materialize_weights(self) -> None:
        original = self._get_original_module()
        self.experts._materialize_weights(original.experts)
        clear_memory()

    def experts_forward(self, hidden_states, top_k_index, top_k_weights):
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
            current_hidden_states = self.experts[expert_idx](current_state)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self.experts_forward(hidden_states_reshaped, selected_experts, routing_weights)

        shared_expert_output = torch.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output

        expert_output = expert_output + shared_expert_output
        return expert_output.reshape(batch_size, sequence_length, hidden_dim)

    @classmethod
    def from_original(cls, original, config, **kwargs):
        return cls(original, config)


# ---------------------------------------------------------------------------
# Sequential expert containers (unfused nn.Linear per expert)
# ---------------------------------------------------------------------------


class SequentialQwen3OmniThinkerExperts(torch.nn.ModuleList):
    """Unfused per-expert nn.Linear layers for Qwen3-Omni thinker MoE.

    Replaces fused 3D Parameters (gate_up_proj, down_proj) with individual
    Qwen3OmniMoeThinkerTextMLP modules per expert.
    """

    def __init__(self, config, original):
        super().__init__()
        self.num_experts = original.gate_up_proj.shape[0]
        intermediate_size = config.moe_intermediate_size

        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeThinkerTextMLP,
        )

        with torch.device("meta"):
            super().__init__([Qwen3OmniMoeThinkerTextMLP(config, intermediate_size) for _ in range(self.num_experts)])

    def _materialize_weights(self, original) -> None:
        """Unfuse fused expert weights into individual nn.Linear layers.

        gate_up_proj shape: (num_experts, 2 * moe_intermediate, hidden)
        down_proj shape:    (num_experts, hidden, moe_intermediate)
        """
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
            original.to_empty(device="meta")  # release original fused parameters
            clear_memory()


class SequentialQwen3OmniTalkerExperts(torch.nn.ModuleList):
    """Unfused per-expert nn.Linear layers for Qwen3-Omni talker MoE.

    Replaces fused 3D Parameters (gate_up_proj, down_proj) with individual
    Qwen3OmniMoeTalkerTextMLP modules per expert.
    """

    def __init__(self, config, original):
        super().__init__()
        self.num_experts = original.gate_up_proj.shape[0]
        intermediate_size = config.moe_intermediate_size

        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeTalkerTextMLP,
        )

        with torch.device("meta"):
            super().__init__([Qwen3OmniMoeTalkerTextMLP(config, intermediate_size) for _ in range(self.num_experts)])

    def _materialize_weights(self, original) -> None:
        """Unfuse fused expert weights into individual nn.Linear layers.

        gate_up_proj shape: (num_experts, 2 * moe_intermediate, hidden)
        down_proj shape:    (num_experts, hidden, moe_intermediate)
        """
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
            original.to_empty(device="meta")  # release original fused parameters
            clear_memory()
