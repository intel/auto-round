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

import contextlib
import gc
import os

import torch
import transformers.models.gpt_oss as transformers_gpt_oss
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


@contextlib.contextmanager
def align_module_device(module: torch.nn.Module):
    device = next(module.parameters()).device
    # return with torch.device(device)
    try:
        yield device
    except:
        pass


from transformers.modeling_utils import no_init_weights as skip_weights_initialize


def update_offload_parameter(
    module: torch.nn.Module,
    name: str,
    data: torch.Tensor,
) -> None:
    param: torch.nn.Parameter = getattr(module, name)
    param.data.copy_(data)


def _get_top_k(config):
    # GPT-OSS MoE: experts per token
    return getattr(config, "num_experts_per_tok", None) or getattr(config, "num_experts_per_token", 1)


class GPTOSSMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dtype=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.alpha = 1.702
        self.limit = 7.0
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=True, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=True, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True, dtype=dtype)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        act = (up + 1) * glu
        return self.down_proj(act)


class SequentialGPTOSSMoE(nn.Module):
    """
    Replaces GPT-OSS fused-expert MoE with per-expert GPTOSSMLP modules.
    Copies weights from fused tensors and reuses the original router and optional shared_expert.
    """

    def __init__(self, config, original):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        dtype_str = getattr(config, "torch_dtype", None) or getattr(config, "dtype", None)
        dtype = torch.bfloat16 if str(dtype_str).endswith("bfloat16") else torch.float32
        top_k = _get_top_k(config)
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
        with skip_weights_initialize(), align_module_device(original.experts):
            for _ in range(E):
                self.experts.append(GPTOSSMLP(hidden_size, intermediate_size, dtype=dtype))

        gup = original.experts.gate_up_proj  # [E, H, 2I]
        gup_b = original.experts.gate_up_proj_bias  # [E, 2I]
        dwn = original.experts.down_proj  # [E, I, H]
        dwn_b = original.experts.down_proj_bias  # [E, H]

        with align_module_device(self.experts):
            for i, mlp in enumerate(self.experts):
                update_offload_parameter(mlp.gate_proj, "weight", original.experts.gate_up_proj[i, :, ::2].T)
                update_offload_parameter(mlp.up_proj, "weight", original.experts.gate_up_proj[i, :, 1::2].T)
                update_offload_parameter(mlp.down_proj, "weight", original.experts.down_proj[i].T)

                update_offload_parameter(mlp.gate_proj, "bias", original.experts.gate_up_proj_bias[i, ::2])
                update_offload_parameter(mlp.up_proj, "bias", original.experts.gate_up_proj_bias[i, 1::2])
                update_offload_parameter(mlp.down_proj, "bias", original.experts.down_proj_bias[i])  # [H]

    def forward(self, hidden_states):
        B, T, H = hidden_states.shape
        x = hidden_states.reshape(-1, H)

        # Use the original router (it returns scores and indices already softmaxed over top-k)
        router_scores, router_indices = self.router(x)  # scores: [tokens, E], indices: [tokens, k]

        out = self.shared_expert(x) if self.shared_expert is not None else torch.zeros_like(x)

        # Accumulate expert outputs for chosen experts only
        for j in range(self.top_k):
            idx = router_indices[:, j]
            w = router_scores[torch.arange(idx.size(0), device=idx.device), idx].unsqueeze(-1)
            unique_experts = torch.unique(idx)
            for e in unique_experts:
                mask = idx == e
                out[mask] += self.experts[e](x[mask]) * w[mask]

        out = out.view(B, T, H)
        router_scores = router_scores.view(B * T, -1)  # shape doesn't matter much; it’s ignored by the decoder
        return out, router_scores


def get_replacement_info(config):
    return (
        SequentialGPTOSSMoE,
        config.get_text_config(),
        transformers_gpt_oss.modeling_gpt_oss.GptOssMLP.__name__,
    )
