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
from transformers.modeling_utils import no_init_weights as skip_weights_initialize
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP

__all__ = ["get_replacement_info"]


def _update_parameter(
    module: torch.nn.Module,
    name: str,
    data: torch.Tensor,
) -> None:
    param = getattr(module, name)
    param.data.copy_(data)


class GPTOssSingleExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dtype: torch.dtype | None = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.alpha = 1.702
        self.limit = 7.0
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=True, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=True, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        act = (up + 1) * glu
        return self.down_proj(act)


class SequentialGPTOSSMoEV1(nn.Module):
    """
    Replaces GPT-OSS fused-expert MoE with per-expert `GPTOssSingleExpert` modules.
    Copies weights from fused tensors and reuses the original router and optional shared_expert.
    """

    def __init__(self, config: GptOssConfig, original: GptOssMLP):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        dtype_str = getattr(config, "torch_dtype", None) or getattr(config, "dtype", None)
        dtype = torch.bfloat16 if str(dtype_str).endswith("bfloat16") else torch.float32
        top_k = config.num_experts_per_tok
        self.hidden_size = hidden_size
        self.intermediate = intermediate_size
        self.top_k = top_k
        self.router = original.router
        self.shared_expert = getattr(original, "shared_expert", None)

        # Number of experts
        E = original.experts.gate_up_proj.shape[0]
        self.num_experts = E

        # Build per-expert MLPs
        self.experts = nn.ModuleList()
        target_device = next(original.experts.parameters()).device
        with skip_weights_initialize(), torch.device(target_device):
            for _ in range(E):
                self.experts.append(GPTOssSingleExpert(hidden_size, intermediate_size, dtype=dtype))

        gup = original.experts.gate_up_proj  # [E, H, 2I]
        gup_b = original.experts.gate_up_proj_bias  # [E, 2I]
        dwn = original.experts.down_proj  # [E, I, H]
        dwn_b = original.experts.down_proj_bias  # [E, H]

        for i, mlp in enumerate(self.experts):
            _update_parameter(mlp.gate_proj, "weight", original.experts.gate_up_proj[i, :, ::2].T)
            _update_parameter(mlp.up_proj, "weight", original.experts.gate_up_proj[i, :, 1::2].T)
            _update_parameter(mlp.down_proj, "weight", original.experts.down_proj[i].T)

            _update_parameter(mlp.gate_proj, "bias", original.experts.gate_up_proj_bias[i, ::2])
            _update_parameter(mlp.up_proj, "bias", original.experts.gate_up_proj_bias[i, 1::2])
            _update_parameter(mlp.down_proj, "bias", original.experts.down_proj_bias[i])  # [H]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, H = hidden_states.shape
        x = hidden_states.reshape(-1, H)

        # Use the original router (it returns scores and indices already softmaxed over top-k)
        router_scores, router_indices = self.router(x)  # scores: [tokens, E], indices: [tokens, k]

        out = self.shared_expert(x) if self.shared_expert is not None else torch.zeros_like(x)

        # Accumulate expert outputs for chosen experts only
        for j in range(self.top_k):
            idx = router_indices[:, j]
            w = router_scores[torch.arange(idx.size(0), device=idx.device), idx].unsqueeze(-1)
            # breakpoint()
            unique_experts = torch.unique(idx)
            for e in unique_experts:
                mask = idx == e
                out[mask] += self.experts[e](x[mask]) * w[mask]

        out = out.view(B, T, H)
        router_scores = router_scores.view(B * T, -1)  # shape doesn't matter much; it’s ignored by the decoder
        return out, router_scores


class SequentialGPTOSSMoEV2(SequentialGPTOSSMoEV1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, H = hidden_states.shape
        x = hidden_states.reshape(-1, H)

        # Use the original router (it returns scores and indices already softmaxed over top-k)
        router_scores, router_indices = self.router(x)  # scores: [tokens, E], indices: [tokens, k]

        out = self.shared_expert(x) if self.shared_expert is not None else torch.zeros_like(x)
        num_all_tokens, total_num_experts = x.size(0), self.num_experts
        mask_weights = torch.zeros((num_all_tokens, total_num_experts), dtype=x.dtype, device=x.device)
        # experts_mask = torch.zeros(
        #     (num_all_tokens, total_num_experts), dtype=x.dtype, device=x.device
        # )
        topk_ids, experts_mask = router_indices, router_scores
        topk_ids = topk_ids.to(torch.int64)
        # topk_weights = topk_weights.to(x.dtype)

        # # experts_mask.scatter_(-1, topk_ids, topk_weights)
        # experts_mask = router_scores
        mask_weights.scatter_(-1, topk_ids, 1)

        mask_weights = mask_weights[:num_all_tokens, :total_num_experts]
        mask_weights = mask_weights.transpose(0, 1)
        experts_mask = experts_mask[:num_all_tokens, :total_num_experts]
        experts_mask = experts_mask.transpose(0, 1)
        num_experts = total_num_experts
        for expert_index in range(num_experts):
            mask_weight = mask_weights[expert_index].unsqueeze(1)
            current_state_static = x * mask_weight
            expert = self.experts[expert_index]
            expert_output = expert(current_state_static)
            expert_output = expert_output * experts_mask[expert_index].unsqueeze(1)
            if expert_index == 0:
                final_hidden_states = expert_output
            else:
                final_hidden_states += expert_output
        return final_hidden_states.view(B, T, H), router_scores.view(B * T, -1)


class SequentialGPTOSSMoEV3(SequentialGPTOSSMoEV1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, H = hidden_states.shape
        x = hidden_states.reshape(-1, H)

        # Use the original router (it returns scores and indices already softmaxed over top-k)
        router_scores, router_indices = self.router(x)  # scores: [tokens, E], indices: [tokens, k]

        out = self.shared_expert(x) if self.shared_expert is not None else torch.zeros_like(x)
        num_all_tokens, total_num_experts = x.size(0), self.num_experts
        mask_weights = torch.zeros((num_all_tokens, total_num_experts), dtype=x.dtype, device=x.device)
        # experts_mask = torch.zeros(
        #     (num_all_tokens, total_num_experts), dtype=x.dtype, device=x.device
        # )
        topk_ids, experts_mask = router_indices, router_scores
        topk_ids = topk_ids.to(torch.int64)
        # topk_weights = topk_weights.to(x.dtype)

        # # experts_mask.scatter_(-1, topk_ids, topk_weights)
        # experts_mask = router_scores
        mask_weights.scatter_(-1, topk_ids, 1)
        top_k_weights = router_scores
        top_k_index = router_indices
        expert_mask = experts_mask
        final_hidden_states = torch.zeros_like(hidden_states)
        # expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_mask = topk_ids.unsqueeze(0).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states[None, top_x].reshape(-1, hidden_states.shape[-1])
            current_hidden_states = self.experts[expert_idx](current_state) * top_k_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        # return final_hidden_states

        return final_hidden_states.view(B, T, H), router_scores.view(B * T, -1)


import os

USE_V1 = os.environ.get("OSS_V1", "0").lower() in ("1", "true", "yes")
if USE_V1:
    SequentialGPTOSSMoE = SequentialGPTOSSMoEV1
else:
    SequentialGPTOSSMoE = SequentialGPTOSSMoEV2


def get_replacement_info(config):
    return (
        SequentialGPTOSSMoE,
        config.get_text_config(),
        GptOssMLP.__name__,
    )
