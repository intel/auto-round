# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeMLP
from transformers.utils.versions import require_version

from auto_round.modeling.fused_moe.fusion_spec import build_standard_moe_fusion_spec, register_moe_fusion_spec
from auto_round.modeling.fused_moe.replace_modules import ReplacementModuleBase
from auto_round.modeling.fused_moe.utils import _update_parameter, sequential_moe_forward
from auto_round.utils import clear_memory, unsupported_meta_device

require_version("transformers>=5.2.0")


class LinearQwen3_5MoeSparseMoeBlock(ReplacementModuleBase):
    supports_gguf_fused_moe = True

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
        return "Qwen3_5MoeSparseMoeBlock"

    def _materialize_weights(self) -> None:
        original = self._get_original_module()
        self.experts._materialize_weights(original.experts)
        clear_memory()

    def experts_forward(self, hidden_states, top_k_index, top_k_weights):
        return sequential_moe_forward(hidden_states, top_k_index, top_k_weights, self.experts, self.num_experts)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self.experts_forward(hidden_states_reshaped, selected_experts, routing_weights)

        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output
        expert_output += shared_expert_output
        return expert_output.reshape(batch_size, sequence_length, hidden_dim)

    @classmethod
    def from_original(cls, original, config, **kwargs):
        return cls(original, config)


class SequentialQwen3_5MoeExperts(torch.nn.ModuleList):
    def __init__(self, config, original):
        super().__init__()
        self.num_experts = original.gate_up_proj.shape[0]
        intermediate_size = config.moe_intermediate_size

        with torch.device("meta"):
            super().__init__([Qwen3_5MoeMLP(config, intermediate_size) for _ in range(self.num_experts)])

        register_moe_fusion_spec(
            self,
            build_standard_moe_fusion_spec(
                detected_projections={
                    "gate_up_proj": {"split_into": ["gate_proj", "up_proj"]},
                    "down_proj": {},
                },
                num_experts=self.num_experts,
                checkpoint_transposed=False,
                module=original,
            ),
        )

    def _materialize_weights(self, original) -> None:
        intermediate_size = original.down_proj.shape[-1]
        if not unsupported_meta_device(original):
            for expert_idx in range(self.num_experts):
                gate_up = original.gate_up_proj[expert_idx]
                down = original.down_proj[expert_idx]

                gate_proj = gate_up[:intermediate_size, :]
                up_proj = gate_up[intermediate_size:, :]

                _update_parameter(self[expert_idx].gate_proj, "weight", gate_proj.contiguous())
                _update_parameter(self[expert_idx].up_proj, "weight", up_proj.contiguous())
                _update_parameter(self[expert_idx].down_proj, "weight", down.contiguous())
            del gate_up, down, gate_proj, up_proj
            original.to_empty(device="meta")
            clear_memory()
