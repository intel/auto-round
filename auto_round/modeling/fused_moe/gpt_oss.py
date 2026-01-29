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
import transformers
from packaging import version
from torch import nn

transformers_version = version.parse(transformers.__version__)
if transformers_version < version.parse("5.0.0"):
    from transformers.modeling_utils import no_init_weights
else:
    from transformers.initialization import no_init_weights

from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP


from auto_round.modeling.fused_moe.replace_modules import ReplacementModuleBase
from auto_round.modeling.fused_moe.utils import _update_parameter
from auto_round.utils import clear_memory, unsupported_meta_device


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


class SequentialGPTOSSMoE(ReplacementModuleBase):
    """
    Replaces GPT-OSS fused-expert MoE with per-expert `GPTOssSingleExpert` modules.
    Copies weights from fused tensors and reuses the original router and optional shared_expert.
    """

    def __init__(self, original: GptOssMLP, config: GptOssConfig):
        super().__init__(original)
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
        if isinstance(original.experts.gate_up_proj, nn.ModuleList):
            E = len(original.experts.gate_up_proj)
        else:
            E = original.experts.gate_up_proj.shape[0]
        self.num_experts = E

        # Build per-expert MLPs
        self.experts = nn.ModuleList()
        target_device = next(original.experts.parameters()).device
        with no_init_weights(), torch.device("meta"):
            for _ in range(E):
                self.experts.append(GPTOssSingleExpert(hidden_size, intermediate_size, dtype=dtype))

    def _materialize_weights(self) -> None:
        original = self._get_original_module()
        if not unsupported_meta_device(original):
            is_modulelist = isinstance(original.experts.gate_up_proj, nn.ModuleList)
            for i, mlp in enumerate(self.experts):
                if is_modulelist:
                    gate_up_linear = original.experts.gate_up_proj[i]
                    down_linear = original.experts.down_proj[i]
                    # gate_up weight is (2*intermediate, hidden)
                    gate_up_weight = gate_up_linear.weight
                    _update_parameter(mlp.gate_proj, "weight", gate_up_weight[: self.intermediate].contiguous())
                    _update_parameter(mlp.up_proj, "weight", gate_up_weight[self.intermediate :].contiguous())
                    _update_parameter(mlp.down_proj, "weight", down_linear.weight.contiguous())

                    if gate_up_linear.bias is not None:
                        _update_parameter(mlp.gate_proj, "bias", gate_up_linear.bias[: self.intermediate].contiguous())
                        _update_parameter(mlp.up_proj, "bias", gate_up_linear.bias[self.intermediate :].contiguous())
                    if down_linear.bias is not None:
                        _update_parameter(mlp.down_proj, "bias", down_linear.bias.contiguous())
                else:
                    _update_parameter(mlp.gate_proj, "weight", original.experts.gate_up_proj[i, :, ::2].T)
                    _update_parameter(mlp.up_proj, "weight", original.experts.gate_up_proj[i, :, 1::2].T)
                    _update_parameter(mlp.down_proj, "weight", original.experts.down_proj[i].T)

                    _update_parameter(mlp.gate_proj, "bias", original.experts.gate_up_proj_bias[i, ::2])
                    _update_parameter(mlp.up_proj, "bias", original.experts.gate_up_proj_bias[i, 1::2])
                    _update_parameter(mlp.down_proj, "bias", original.experts.down_proj_bias[i])  # [H]
            original.experts.to_empty(device="meta")  # release original experts parameters
            clear_memory()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, H = hidden_states.shape
        x = hidden_states.reshape(-1, H)

        # Use the original router (it returns scores and indices already softmaxed over top-k)
        router_out = self.router(x)
        if isinstance(router_out, (tuple, list)):
            if len(router_out) == 2:
                router_scores, router_indices = router_out
            else:
                router_scores, router_indices = router_out[0], router_out[1]
        else:
            raise ValueError("Unexpected router output type for GPT-OSS router")

        final_hidden_states = self.shared_expert(x) if self.shared_expert is not None else torch.zeros_like(x)
        num_all_tokens, total_num_experts = x.size(0), self.num_experts
        mask_weights = torch.zeros((num_all_tokens, total_num_experts), dtype=x.dtype, device=x.device)
        topk_ids, experts_mask = router_indices, router_scores
        topk_ids = topk_ids.to(torch.int64)

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
            final_hidden_states += expert_output
        return final_hidden_states.view(B, T, H), router_scores.view(B * T, -1)

    @classmethod
    def original_module_class(cls) -> str:
        """Return the class name of the module this replaces."""
        return "GptOssMLP"

    @classmethod
    def from_original(
        cls,
        original: GptOssMLP,
        config: GptOssConfig,
        **kwargs,
    ) -> "SequentialGPTOSSMoE":
        """Create an instance from the original module."""
        return cls(original, config)