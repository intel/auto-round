# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import transformers
from packaging import version

from auto_round.modelling.replace_modules import ReplacementModuleBase
from auto_round.utils import clear_memory, logger, unsupported_meta_device

transformers_version = version.parse(transformers.__version__)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import Glm4MoeLiteConfig
    from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
    Glm4MoeLiteMoE
)


def _update_parameter(
    module: torch.nn.Module,
    name: str,
    data: torch.Tensor,
) -> None:
    param = getattr(module, name)
    param.data.copy_(data)


class LinearGlm4MoeLiteMoE(ReplacementModuleBase):
    """
    Calibration version of Glm4MoeLiteMoE that sends all tokens to all
    experts.
    """

    is_permanent = True

    def __init__(
        self,
        original: "Glm4MoeLiteMoE",
        config: "Glm4MoeLiteConfig",
        calibrate_all_experts: bool = False,
    ):
        super().__init__()
        self.config = config
        self.top_k = original.top_k
        self.n_routed_experts = original.n_routed_experts
        self.n_group = original.n_group
        self.topk_group = original.topk_group
        self.norm_topk_prob = original.norm_topk_prob
        self.routed_scaling_factor = original.routed_scaling_factor
        self.gate = original.gate
        self.route_tokens_to_experts = original.route_tokens_to_experts
        self.calibrate_all_experts = calibrate_all_experts
        self.experts = SequentialGlm4MoeLiteNaiveMoe(config, original.experts)
        self.shared_experts = original.shared_experts
    
    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        # topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        experts_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                topk_indices, num_classes=len(self.experts)
            )
            expert_mask = expert_mask.permute(2, 0, 1)
            
        for expert_idx, expert in enumerate(self.experts):
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)
            has_tokens = token_indices.numel() > 0

            if self.calibrate_all_experts:
                # When calibrating, run all tokens through the expert to gather stats.
                # The output is still calculated using only the routed tokens.
                expert_output_full = expert(hidden_states)
                if not has_tokens:
                    # No tokens routed to this expert, but stats were gathered.
                    continue
                expert_output = expert_output_full[token_indices]
            else:
                # Standard MoE behavior: only process tokens routed to this expert.
                if not has_tokens:
                    continue
                expert_output = expert(hidden_states[token_indices])

            # Common logic for combining expert outputs
            expert_weights = topk_weights[token_indices, weight_indices]
            weighted_output = expert_output * expert_weights.unsqueeze(-1)
            experts_hidden_states.index_add_(0, token_indices, weighted_output.to(experts_hidden_states.dtype))

        # return experts_hidden_states
        hidden_states = experts_hidden_states.view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states

    @classmethod
    def original_module_class(cls) -> str:
        """Return the class name of the module this replaces."""
        return "Glm4MoeLiteMoE"

    @classmethod
    def from_original(
        cls,
        original: "Glm4MoeLiteMoE",
        config: "Glm4MoeLiteConfig",
        **kwargs,
    ) -> "LinearGlm4MoeLiteMoE":
        """Create an instance from the original module."""
        return cls(original, config)


class SequentialGlm4MoeLiteNaiveMoe(torch.nn.ModuleList):
    def __init__(self, config, original):
        super().__init__()
        self.num_experts = original.gate_up_proj.shape[0]
        intermediate_size = config.moe_intermediate_size
        from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
            Glm4MoeLiteMLP,
        )

        target_device = next(original.parameters()).device
        with torch.device(target_device):
            super().__init__([Glm4MoeLiteMLP(config, intermediate_size) for _ in range(self.num_experts)])

        if not unsupported_meta_device(original):
            for i in range(self.num_experts):
                gate_up = original.gate_up_proj[i]
                down = original.down_proj[i]

                gate_proj = gate_up[:intermediate_size,:]
                up_proj = gate_up[intermediate_size:,:]

                _update_parameter(self[i].gate_proj, "weight", gate_proj)
                _update_parameter(self[i].up_proj, "weight", up_proj)
                _update_parameter(self[i].down_proj, "weight", down)
            del gate_up, down, gate_proj, up_proj
            original.to_empty(device="meta")  # release original experts parameters
            clear_memory()

