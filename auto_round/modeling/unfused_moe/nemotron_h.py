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
"""Unfused-MoE handler for Nemotron-H (Mamba2 + Attention + MoE hybrid).

Nemotron-H stores routed-MoE expert weights as bundled 3-D Parameters
inside ``NemotronHExperts``:

    self.up_proj   : nn.Parameter[num_experts, intermediate, hidden]
    self.down_proj : nn.Parameter[num_experts, hidden, intermediate]

AutoRound's quantization path walks ``nn.Linear`` modules and never sees
bundled Parameters, so without this handler it quantizes 0 routed
experts — ~94 % of the model. ``LinearNemotronHMoE`` re-expresses the
routed experts as a ``ModuleList[NemotronHMLP]`` whose entries each
contain regular ``nn.Linear`` ``up_proj`` / ``down_proj`` modules. The
on-disk safetensors checkpoint already stores experts under individual
keys (``backbone.layers.{i}.mixer.experts.{e}.up_proj.weight``), so the
state-dict loads straight into the ModuleList.

The forward path — group-top-k routing with
``e_score_correction_bias`` and ``routed_scaling_factor`` — is a
verbatim port of ``transformers.models.nemotron_h.modeling_nemotron_h.NemotronHMoE``;
that logic is load-bearing for both inference accuracy and
calibration-time token routing.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearNemotronHMoE(nn.Module):
    """Linear-discoverable replacement for ``NemotronHMoE``.

    Forward semantics are byte-identical to the upstream block; the
    sole structural difference is that routed experts live in
    ``self.experts: nn.ModuleList[NemotronHMLP]`` rather than a single
    bundled ``NemotronHExperts`` module.
    """

    def __init__(self, config, layer_idx: int | None = None):
        super().__init__()
        from transformers.models.nemotron_h.modeling_nemotron_h import (
            NemotronHMLP,
            NemotronHTopkRouter,
        )

        self.config = config

        # Router — its only learnable piece is a single Parameter
        # ``[num_experts, hidden_size]`` plus a buffer
        # ``e_score_correction_bias``. AutoRound finds no nn.Linear inside
        # NemotronHTopkRouter so it is automatically left in BF16.
        self.gate = NemotronHTopkRouter(config)

        # Routed experts — the whole point of this handler. Each expert
        # is a standard 2-linear ReLU^2 MLP with the same shape as one
        # slice of the bundled NemotronHExperts tensors.
        self.n_routed_experts = config.n_routed_experts
        self.experts = nn.ModuleList(
            [NemotronHMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.n_routed_experts)]
        )

        # Shared expert(s) — already use nn.Linear in upstream.
        self.shared_experts = NemotronHMLP(
            config=config,
            intermediate_size=config.moe_shared_expert_intermediate_size,
        )

        # Routing hyper-parameters mirrored from upstream.
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok

        # Optional latent projection. ``Identity`` for Nemotron-Cascade-2
        # because ``moe_latent_size is None``.
        if config.moe_latent_size is not None:
            self.fc1_latent_proj = nn.Linear(config.hidden_size, config.moe_latent_size, bias=config.mlp_bias)
            self.fc2_latent_proj = nn.Linear(config.moe_latent_size, config.hidden_size, bias=config.mlp_bias)
        else:
            self.fc1_latent_proj = nn.Identity()
            self.fc2_latent_proj = nn.Identity()

    # ------------------------------------------------------------------ #
    # Identical to upstream NemotronHMoE.route_tokens_to_experts. The
    # group-top-k + score-correction-bias path is load-bearing — do not
    # simplify.
    # ------------------------------------------------------------------ #
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]

        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        topk_weights = topk_weights.to(hidden_states.dtype)

        flat = hidden_states.view(-1, hidden_dim)
        flat = self.fc1_latent_proj(flat)

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

        out = self.fc2_latent_proj(out)
        out = out.view(*orig_shape)
        out = out + self.shared_experts(residuals)
        return out
