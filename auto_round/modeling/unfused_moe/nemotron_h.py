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


class LinearNemotronHMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers.models.nemotron_h.modeling_nemotron_h import (
            NemotronHMLP,
            NemotronHTopkRouter,
        )

        self.config = config
        self.gate = NemotronHTopkRouter(config)
        self.n_routed_experts = config.n_routed_experts
        self.experts = nn.ModuleList(
            [NemotronHMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.n_routed_experts)]
        )
        self.shared_experts = NemotronHMLP(
            config=config,
            intermediate_size=config.moe_shared_expert_intermediate_size,
        )
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok

        if config.moe_latent_size is not None:
            self.fc1_latent_proj = nn.Linear(config.hidden_size, config.moe_latent_size, bias=config.mlp_bias)
            self.fc2_latent_proj = nn.Linear(config.moe_latent_size, config.hidden_size, bias=config.mlp_bias)
        else:
            self.fc1_latent_proj = nn.Identity()
            self.fc2_latent_proj = nn.Identity()

    def route_tokens_to_experts(self, router_logits: torch.Tensor):
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def experts_forward(
        self,
        flat: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.zeros_like(flat)

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=self.n_routed_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().squeeze(-1)

        for expert_idx in expert_hit.tolist():
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue
            current_state = flat[token_idx]
            current_out = self.experts[expert_idx](current_state)
            current_out = current_out * topk_weights[token_idx, top_k_pos, None]
            out.index_add_(0, token_idx, current_out.to(out.dtype))

        return out

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]

        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        topk_weights = topk_weights.to(hidden_states.dtype)

        flat = hidden_states.view(-1, hidden_dim)
        flat = self.fc1_latent_proj(flat)

        out = self.experts_forward(flat, topk_indices, topk_weights)

        out = self.fc2_latent_proj(out)
        out = out.view(*orig_shape)
        out = out + self.shared_experts(residuals)
        return out


def nemotron_h_default_layer_config_patterns():
    from auto_round.modeling.unfused_moe.nemotron_h_setup import _NEMOTRON_H_OUT_PROJ_PATTERN

    return {
        _NEMOTRON_H_OUT_PROJ_PATTERN: {
            "scale_dtype": torch.bfloat16,
        },
    }


def apply_nemotron_h_post_load(
    model: nn.Module,
    *,
    ssm_core_dtype=torch.float32,
    router_bias_dtype=torch.float32,
    enable_high_precision_overrides: bool = True,
):
    from auto_round.modeling.unfused_moe.nemotron_h_setup import (
        _apply_high_precision_overrides,
        _patch_zamba2_group_size,
    )

    summary = {"zamba2_patched": 0, "high_precision_restored": 0}
    config = getattr(model, "config", None)
    if config is None or getattr(config, "model_type", None) != "nemotron_h":
        return summary

    if getattr(model, "_autoround_nh_post_load_applied", False):
        return summary

    summary["zamba2_patched"] = _patch_zamba2_group_size(model)
    if enable_high_precision_overrides:
        summary["high_precision_restored"] = _apply_high_precision_overrides(
            model,
            ssm_core_dtype=ssm_core_dtype,
            router_bias_dtype=router_bias_dtype,
        )
    try:
        setattr(model, "_autoround_nh_post_load_applied", True)
    except Exception:
        pass
    return summary
