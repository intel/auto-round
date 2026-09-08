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

from auto_round.modeling.fused_moe.fusion_spec import build_standard_moe_fusion_spec, register_moe_fusion_spec
from auto_round.modeling.fused_moe.replace_modules import ReplacementModuleBase
from auto_round.modeling.fused_moe.utils import _update_parameter, grouped_or_sequential_moe_forward
from auto_round.utils import clear_memory, unsupported_meta_device


class GPTOssExperts(torch.nn.ModuleList):
    """Per-expert ``GPTOssSingleExpert`` container with the GPT-OSS gating exposed for the
    grouped experts forward.

    GPT-OSS does not use the generic ``act_fn(gate) * up`` gate, so it provides ``_apply_gate``
    (the hook the grouped path prefers): the ``[gate; up]`` clamp, the ``glu = gate * sigmoid(
    gate * alpha)`` GLU, and the ``(up + 1) * glu`` scaling -- byte-for-byte what
    ``GPTOssSingleExpert.forward`` does, so the grouped result matches the per-expert loop.
    """

    def __init__(self, experts: list[nn.Module], num_experts: int, alpha: float, limit: float):
        super().__init__(experts)
        self.num_experts = num_experts
        self.alpha = alpha
        self.limit = limit

    def _apply_gate(self, gate_up_out: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up_out.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        return (up + 1) * glu


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

    supports_gguf_fused_moe = True

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
        E = original.experts.gate_up_proj.shape[0]
        self.num_experts = E

        # Build per-expert MLPs
        target_device = next(original.experts.parameters()).device
        with no_init_weights(), torch.device("meta"):
            expert_mlps = [GPTOssSingleExpert(hidden_size, intermediate_size, dtype=dtype) for _ in range(E)]
        # Expose num_experts + the GPT-OSS gating so the grouped experts forward can run.
        self.experts = GPTOssExperts(expert_mlps, num_experts=E, alpha=expert_mlps[0].alpha, limit=expert_mlps[0].limit)
        register_moe_fusion_spec(
            self.experts,
            build_standard_moe_fusion_spec(
                detected_projections={
                    "gate_up_proj": {"split_into": ["gate_proj", "up_proj"], "concat_dim": 0},
                    "down_proj": {},
                },
                num_experts=E,
                checkpoint_transposed=True,
                module=original.experts,
            ),
        )

    def _materialize_weights(self) -> None:
        original = self._get_original_module()
        if not unsupported_meta_device(original):
            for i, mlp in enumerate(self.experts):
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

        # Use the original router (it returns logits, scores and indices)
        router_out = self.router(x)
        if len(router_out) == 3:
            _, router_scores, router_indices = router_out
        else:
            router_scores, router_indices = router_out

        shared = self.shared_expert(x) if self.shared_expert is not None else torch.zeros_like(x)

        # ``router_indices``/``router_scores`` are the gathered (token, top_k) ids and weights
        # (the original built its per-expert score matrix by scattering exactly these). Feeding
        # them to the grouped experts forward reproduces the old ``sum_e expert(x) * score[e]``
        # dense-mask loop -- routed pairs only, weighted then reduced over top_k.
        expert_output = grouped_or_sequential_moe_forward(x, router_indices, router_scores, self.experts, self.num_experts)
        final_hidden_states = shared + expert_output

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
