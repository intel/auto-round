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

# Note: adapted from: https://huggingface.co/stepfun-ai/Step-3.5-Flash/blob/main/modeling_step3p5.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from auto_round.modeling.fused_moe.replace_modules import ReplacementModuleBase
from auto_round.modeling.fused_moe.utils import _update_parameter
from auto_round.utils import clear_memory, unsupported_meta_device


class Step3p5ExpertMLP(nn.Module):
    """Single expert MLP with gate_proj, up_proj, down_proj as nn.Linear."""

    def __init__(self, hidden_size, intermediate_size, limit=None):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN["silu"]
        self.limit = limit

    def forward(self, x):
        up = self.up_proj(x)
        gate = self.act_fn(self.gate_proj(x))
        if self.limit is not None:
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
        return self.down_proj(gate * up)


class SequentialStep3p5MoeExperts(torch.nn.ModuleList):
    """Converts fused MoELinear weights into individual Step3p5ExpertMLP modules."""

    def __init__(self, original):
        super().__init__()
        self.num_experts = original.up_proj.num_experts
        hidden_size = original.up_proj.in_features
        intermediate_size = original.up_proj.out_features
        limit = original.limit

        with torch.device("meta"):
            super().__init__([Step3p5ExpertMLP(hidden_size, intermediate_size, limit) for _ in range(self.num_experts)])

    def _materialize_weights(self, original) -> None:
        """Split fused MoELinear weights into individual expert nn.Linear weights.

        Args:
            original: The original Step3p5MoEMLP module containing fused MoELinear
                      modules (gate_proj, up_proj, down_proj).
        """
        if not unsupported_meta_device(original.up_proj):
            for i in range(self.num_experts):
                _update_parameter(self[i].gate_proj, "weight", original.gate_proj.weight[i].contiguous())
                _update_parameter(self[i].up_proj, "weight", original.up_proj.weight[i].contiguous())
                _update_parameter(self[i].down_proj, "weight", original.down_proj.weight[i].contiguous())
            original.gate_proj.to_empty(device="meta")
            original.up_proj.to_empty(device="meta")
            original.down_proj.to_empty(device="meta")
            clear_memory()


class LinearStep3p5MoEMLP(ReplacementModuleBase):
    """Replacement for Step3p5MoEMLP that splits fused MoELinear into
    individual nn.Linear per expert for quantization support."""

    def __init__(self, original, config=None):
        super().__init__(original)
        self.num_experts = original.num_experts
        self.top_k = original.top_k
        self.hidden_size = original.hidden_size
        self.moe_intermediate_size = original.moe_intermediate_size

        self.gate = original.gate
        self.need_fp32_gate = original.need_fp32_gate
        self.routed_scaling_factor = original.routed_scaling_factor

        # routing function
        self.use_moe_router_bias = original.use_moe_router_bias
        if self.use_moe_router_bias:
            self.router_bias = original.router_bias
            self.custom_routing_function = self.router_bias_func
        else:
            self.custom_routing_function = original.custom_routing_function

        # split fused experts into sequential
        with torch.device("meta"):
            self.experts = SequentialStep3p5MoeExperts(original)

    @classmethod
    def original_module_class(cls) -> str:
        """Return the class name of the module this replaces."""
        return "Step3p5MoEMLP"

    def _materialize_weights(self) -> None:
        original = self._get_original_module()
        self.experts._materialize_weights(original)
        clear_memory()

    def router_bias_func(self, gating_output: torch.Tensor, topk: int, renormalize: bool):
        gate_prob = torch.sigmoid(gating_output.float())
        gate_prob_with_bias = gate_prob + self.router_bias.unsqueeze(0)
        _, indices = torch.topk(gate_prob_with_bias, k=topk, dim=1)
        topk_prob = torch.gather(gate_prob, 1, indices)
        expert_topk_weight = topk_prob
        if renormalize:
            expert_topk_weight = expert_topk_weight / (torch.sum(expert_topk_weight, dim=-1, keepdim=True) + 1e-20)
        return expert_topk_weight, indices

    def forward(self, hidden_states):
        """Forward pass with sequential expert computation."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if self.need_fp32_gate:
            router_logits = torch.matmul(hidden_states.to(torch.float32), self.gate.weight.t().to(torch.float32))
        else:
            router_logits = self.gate(hidden_states)

        if self.custom_routing_function:
            routing_weights, selected_experts = self.custom_routing_function(
                router_logits, self.top_k, renormalize=True
            )
        else:
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        routing_weights = routing_weights * self.routed_scaling_factor

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = self.experts[expert_idx](current_state) * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states

    @classmethod
    def from_original(
        cls,
        original,
        config=None,
        **kwargs,
    ):
        """Create an instance from the original module."""
        return cls(original, config)
