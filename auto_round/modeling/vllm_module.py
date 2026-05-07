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


from typing import Any

import torch
import transformers
from packaging import version
from torch import nn

transformers_version = version.parse(transformers.__version__)
if transformers_version < version.parse("5.0.0"):
    from transformers.configuration_utils import PretrainedConfig as PreTrainedConfig
    from transformers.modeling_utils import no_init_weights
else:
    from transformers.initialization import no_init_weights
    from transformers.configuration_utils import PreTrainedConfig

from transformers.activations import ACT2FN
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
)

from auto_round.modeling.fused_moe.utils import _update_parameter
from auto_round.modeling.replace_modules import ReplacementModuleBase
from auto_round.utils import LazyImport, clear_memory, unsupported_meta_device

SharedFusedMoE = LazyImport("vllm.model_executor.layers.fused_moe.SharedFusedMoE")
FusedMoE = LazyImport("vllm.model_executor.layers.fused_moe.FusedMoE")


state_dict_mapping = {
    "experts._": "",
    "gate_up_proj.gate_proj.": "gate_proj.",
    "gate_up_proj.up_proj.": "up_proj.",
    "qkv_proj.q_proj.": "q_proj.",
    "qkv_proj.k_proj.": "k_proj.",
    "qkv_proj.v_proj.": "v_proj.",
    "self_attn.mla_attn.mla_attn.": "self_attn.",
    "self_attn.mla_attn.": "self_attn.",
}

key_to_remove = [
    "attn._q_scale",
    "attn._k_scale",
    "attn._v_scale",
    "attn._prob_scale",
]


def state_dict_hook(module, state_dict, prefix, *args):
    keys = state_dict.keys()
    weight_to_update = []
    weight_to_remove = []
    for key in keys:
        for k in state_dict_mapping:
            if k in key:
                weight_to_update.append(key)
                break
        for k in key_to_remove:
            if k in key:
                weight_to_remove.append(key)
                break

    for name in weight_to_remove:
        state_dict.pop(name)

    for name in weight_to_update:
        if name in state_dict:
            new_name = name
            for old_pattern, new_pattern in state_dict_mapping.items():
                new_name = new_name.replace(old_pattern, new_pattern)
            state_dict[new_name] = state_dict.pop(name)


class VllmAttention(ReplacementModuleBase):
    """Replaces Vllm Attention with VllmAttention modules."""

    def __init__(self, original: Attention, config: PreTrainedConfig):
        super().__init__(original)

    @classmethod
    def original_module_class(cls) -> str:
        """Return the class name of the module this replaces."""
        return "Attention"

    @classmethod
    def from_original(
        cls,
        original: Attention,
        config: PreTrainedConfig,
        **kwargs,
    ) -> "VllmAttention":
        """Create an instance from the original module."""
        original.register_state_dict_post_hook(state_dict_hook)
        return original


class VllmQKVParallelLinear(ReplacementModuleBase):
    """Replaces Vllm QKVParallelLinear with VllmQKVParallelLinear modules."""

    def __init__(self, original: QKVParallelLinear, config: PreTrainedConfig):
        super().__init__(original)
        dtype_str = getattr(config, "torch_dtype", None) or getattr(config, "dtype", None)
        dtype = torch.bfloat16 if str(dtype_str).endswith("bfloat16") else torch.float32
        hidden_size = original.hidden_size
        with no_init_weights(), torch.device("meta"):
            self.q_proj = nn.Linear(hidden_size, original.head_size * original.num_heads, bias=False, dtype=dtype)
            self.k_proj = nn.Linear(hidden_size, original.head_size * original.num_kv_heads, bias=False, dtype=dtype)
            self.v_proj = nn.Linear(hidden_size, original.v_head_size * original.num_kv_heads, bias=False, dtype=dtype)
        self._materialize_weights()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        return torch.cat([q, k, v], dim=0)

    def _materialize_weights(self) -> None:
        original = self._get_original_module()
        if not unsupported_meta_device(original):
            _update_parameter(self.q_proj, "weight", original.weight[: self.q_proj.weight.shape[1], :])
            _update_parameter(
                self.k_proj,
                "weight",
                original.weight[
                    self.q_proj.weight.shape[1] : self.q_proj.weight.shape[1] + self.k_proj.weight.shape[1], :
                ],
            )
            _update_parameter(self.v_proj, "weight", original.weight[-self.v_proj.weight.shape[1] :, :])
            original.to_empty(device="meta")
            clear_memory()

    @classmethod
    def original_module_class(cls) -> str:
        """Return the class name of the module this replaces."""
        return "QKVParallelLinear"

    @classmethod
    def from_original(
        cls,
        original: QKVParallelLinear,
        config: PreTrainedConfig,
        **kwargs,
    ) -> "VllmQKVParallelLinear":
        """Create an instance from the original module."""
        return cls(original, config)


class VllmMergedColumnParallelLinear(ReplacementModuleBase):
    """Replaces Vllm MergedColumnParallelLinear with VllmMergedColumnParallelLinear modules."""

    def __init__(self, original: MergedColumnParallelLinear, config: PreTrainedConfig):
        super().__init__(original)
        dtype_str = getattr(config, "torch_dtype", None) or getattr(config, "dtype", None)
        dtype = torch.bfloat16 if str(dtype_str).endswith("bfloat16") else torch.float32
        with no_init_weights(), torch.device("meta"):
            self.gate_proj = nn.Linear(original.input_size, original.output_size // 2, bias=False, dtype=dtype)
            self.up_proj = nn.Linear(original.input_size, original.output_size // 2, bias=False, dtype=dtype)
        self._materialize_weights()

    def _materialize_weights(self) -> None:
        original = self._get_original_module()
        if not unsupported_meta_device(original):
            _update_parameter(self.gate_proj, "weight", original.weight[: self.gate_proj.weight.shape[0], :])
            _update_parameter(self.up_proj, "weight", original.weight[self.up_proj.weight.shape[0] :, :])
            original.to_empty(device="meta")  # release original experts parameters
            clear_memory()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_out = self.gate_proj(hidden_states)
        up_out = self.up_proj(hidden_states)
        return torch.cat([gate_out, up_out], dim=1), None

    @classmethod
    def original_module_class(cls) -> str:
        """Return the class name of the module this replaces."""
        return "MergedColumnParallelLinear"

    @classmethod
    def from_original(
        cls,
        original: MergedColumnParallelLinear,
        config: PreTrainedConfig,
        **kwargs,
    ) -> "VllmMergedColumnParallelLinear":
        """Create an instance from the original module."""
        new_module = cls(original, config)
        new_module.register_state_dict_post_hook(state_dict_hook)
        return new_module


class MoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=None, act=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class SequentialExperts(torch.nn.ModuleList):
    def __init__(self, config, original):
        self.num_experts = original.w13_weight.shape[0]
        target_device = next(original.parameters()).device
        with no_init_weights(), torch.device("meta"):
            super().__init__([MoeMLP(config, act=original.activation) for _ in range(self.num_experts)])

    def _materialize_weights(self, original) -> None:
        if not unsupported_meta_device(original):
            for i in range(self.num_experts):
                gate_up = original.w13_weight[i]
                down = original.w2_weight[i]
                gate_proj = gate_up[: gate_up.shape[0] // 2, :]
                up_proj = gate_up[gate_up.shape[0] // 2 :, :]
                _update_parameter(self[i].gate_proj, "weight", gate_proj.contiguous())
                _update_parameter(self[i].up_proj, "weight", up_proj.contiguous())
                _update_parameter(self[i].down_proj, "weight", down.contiguous())
            del gate_up, down, gate_proj, up_proj
            original.w13_weight.to("meta")  # release original experts parameters
            original.w2_weight.to("meta")
            del original.w13_weight
            del original.w2_weight
            clear_memory()


class VllmSharedFusedMoE(ReplacementModuleBase):
    """Replaces Vllm SharedFusedMoE with VllmSharedFusedMoE modules."""

    def __init__(self, original: Any, config: PreTrainedConfig):
        super().__init__(original)
        dtype_str = getattr(config, "torch_dtype", None) or getattr(config, "dtype", None)
        dtype = torch.bfloat16 if str(dtype_str).endswith("bfloat16") else torch.float32
        with no_init_weights(), torch.device("meta"):
            self._experts = SequentialExperts(config, original)

        self._shared_experts = original._shared_experts
        self._gate = original._gate
        self.router = original.router
        self.is_internal_router = original.is_internal_router
        self._materialize_weights()

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        router_logits, _ = self._gate(hidden_states)
        topk_weights, topk_indices = self.router.select_experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                topk_indices.to(torch.long), num_classes=self._experts.num_experts
            )
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self._experts.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_hidden_states = self._experts[expert_idx](current_state) * topk_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return self._shared_experts(hidden_states), final_hidden_states

    def _materialize_weights(self) -> None:
        original = self._get_original_module()
        self._experts._materialize_weights(original)
        clear_memory()

    @classmethod
    def original_module_class(cls) -> str:
        """Return the class name of the module this replaces."""
        return "SharedFusedMoE"

    @classmethod
    def from_original(
        cls,
        original: Any,
        config: PreTrainedConfig,
        **kwargs,
    ) -> "VllmSharedFusedMoE":
        """Create an instance from the original module."""
        new_module = cls(original, config)
        new_module.register_state_dict_post_hook(state_dict_hook)
        return new_module


def apply_moe(
    layer: torch.nn.Module,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor | None,
    shared_experts_input: torch.Tensor | None = None,
) -> torch.Tensor:
    hidden_states = x
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(topk_ids.to(torch.long), num_classes=layer._experts.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == layer._experts.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        current_hidden_states = layer._experts[expert_idx](current_state) * topk_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states


class VllmFusedMoE(ReplacementModuleBase):
    """Replaces Vllm FusedMoE with VllmFusedMoE modules."""

    def __init__(self, original: Any, config: PreTrainedConfig):
        super().__init__(original)

    @classmethod
    def original_module_class(cls) -> str:
        """Return the class name of the module this replaces."""
        return "FusedMoE"

    @classmethod
    def from_original(
        cls,
        original: Any,
        config: PreTrainedConfig,
        **kwargs,
    ) -> "VllmSharedFusedMoE":
        """Create an instance from the original module."""
        with no_init_weights(), torch.device("meta"):
            experts = SequentialExperts(config, original)
        experts._materialize_weights(original)
        original._experts = experts
        original.runner.quant_method.apply = apply_moe
        original.register_state_dict_post_hook(state_dict_hook)
        return original
