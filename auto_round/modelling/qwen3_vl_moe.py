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
from torch import nn
from transformers.activations import ACT2FN

from auto_round.utils import logger, unsupported_meta_device

transformers_version = version.parse(transformers.__version__)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import Qwen3VLMoeConfig, Qwen3VLMoeTextConfig
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
        Qwen3VLMoeTextSparseMoeBlock,
    )


def _update_parameter(
    module: torch.nn.Module,
    name: str,
    data: torch.Tensor,
) -> None:
    param = getattr(module, name)
    param.data.copy_(data)


# Adapted from https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modeling/qwen3_vl_moe.py
class LinearQwen3VLMoeTextSparseMoeBlock(torch.nn.Module):
    """
    Calibration version of Qwen3VLMoeTextSparseMoeBlock that sends all tokens to all
    experts.
    """

    is_permanent = True

    def __init__(
        self,
        original: "Qwen3VLMoeTextSparseMoeBlock",
        config: "Qwen3VLMoeConfig",
        calibrate_all_experts: bool = False,
    ):
        super().__init__()
        text_config: "Qwen3VLMoeTextConfig" = config.get_text_config()

        self.hidden_size = text_config.hidden_size
        self.num_experts = text_config.num_experts
        self.top_k = original.top_k
        # Note: gate was changed to be a Linear layer in transformers==4.57.0
        # https://github.com/JJJYmmm/transformers/commit/f5dea1c694af8c994c769170813a8702332119ee
        self.gate = original.gate
        self.calibrate_all_experts = calibrate_all_experts
        self.experts = SequentialQwen3VLMoeTextExperts(text_config, original.experts)
        if not transformers_version <= version.parse(
            "4.57.3"
        ):  # remove conversion_mapping for qwen3_vl_moe when transformers>=5.0
            from transformers.conversion_mapping import register_checkpoint_conversion_mapping

            register_checkpoint_conversion_mapping(config.model_type, [], overwrite=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
        # get topk experts per token
        # routing_weight: (num_tokens, top_k)
        # routing_indices: (num_tokens, top_k)
        routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        next_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # convert router indices into OHE list
        # reshape to be (num_experts, top_k, batch_size * sequence_length)
        expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx, expert_layer in enumerate(self.experts):
            idx, token_idx = torch.where(expert_mask[expert_idx].squeeze(0))

            if self.calibrate_all_experts:
                expert_out = expert_layer(hidden_states)[token_idx]
            else:
                expert_out = expert_layer(hidden_states[token_idx])

            if len(token_idx) > 0:
                # if there are tokens meant for this expert, further scale the expert
                # output by the score
                weighted_output = expert_out * routing_weights[token_idx, idx, None]
                next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
        next_states = next_states.reshape(batch_size, sequence_length, hidden_dim)

        if transformers_version <= version.parse("4.57.3"):
            return next_states, router_logits
        else:
            return next_states


class SequentialQwen3VLMoeTextExperts(torch.nn.ModuleList):
    def __init__(self, config, original):
        super().__init__()
        self.num_experts = original.gate_up_proj.shape[0]
        intermediate_size = config.moe_intermediate_size
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
            Qwen3VLMoeTextMLP,
        )

        super().__init__([Qwen3VLMoeTextMLP(config, intermediate_size) for _ in range(self.num_experts)])

        if not unsupported_meta_device(original):
            for i in range(self.num_experts):
                gate_up = original.gate_up_proj[i]
                down = original.down_proj[i]

                gate_proj = gate_up[:, :intermediate_size]
                up_proj = gate_up[:, intermediate_size:]

                _update_parameter(self[i].gate_proj, "weight", gate_proj.t().contiguous())
                _update_parameter(self[i].up_proj, "weight", up_proj.t().contiguous())
                _update_parameter(self[i].down_proj, "weight", down.t().contiguous())


def get_replacement_info(config):
    return (LinearQwen3VLMoeTextSparseMoeBlock, config, "Qwen3VLMoeTextSparseMoeBlock")
