import torch
from torch import nn
from transformers.models.deepseek_v3.modeling_deepseek_v3 import ACT2FN
import torch.nn.functional as F
from transformers.initialization import no_init_weights
# from .configuration_qwen3_moe import Qwen3MoeConfig
from transformers.modeling_utils import PreTrainedModel
import transformers.initialization as init

import logging
logger = logging.getLogger(__name__)

def oot_normal_(
    tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, generator: torch.Generator | None = None
) -> torch.Tensor:
    return tensor

init.normal_ = oot_normal_


class Qwen3MoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class OoTQwen3MoeExperts(torch.nn.ModuleList):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        # append expert modules
        with no_init_weights():
            for _ in range(self.num_experts):
                self.append(Qwen3MoeMLP(config, intermediate_size=self.intermediate_dim))
        # self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        # self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]
        # print(f"Initialized OoTQwen3MoeExperts with {self.num_experts} experts.")

    def forward(
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
            current_hidden_states = self[expert_idx](current_state)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states



from contextlib import contextmanager
@contextmanager
def empty_ctx():
    yield


import transformers.models.qwen3_moe.modeling_qwen3_moe as m

# Monkeypatch: bypass Qwen-specific init, keep only base PreTrainedModel init
@torch.no_grad()
def oot_init_weights(self, module):
    return PreTrainedModel._init_weights(self, module)


m.Qwen3MoePreTrainedModel._init_weights = oot_init_weights

def apply_transformer_patches_qwen():
    # Patch DeepseekV3NaiveMoe to use OoTDeepseekV3MoE
    from transformers.models.qwen3_moe import modeling_qwen3_moe as qwen3_moe_modeling
    qwen3_moe_modeling.Qwen3MoeExperts = OoTQwen3MoeExperts
    # import use_experts_implementation
    import transformers.integrations as transformers_integrations
    from transformers.models import qwen3 as qwen3
    transformers_integrations.skip_weights_initialize = empty_ctx
    logger.warning("Patched DeepseekV3NaiveMoe to use OoTDeepseekV3MoE")