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
# Note: adapted from # https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modeling/llama4.py
import torch
import transformers
from packaging import version

transformers_version = version.parse(transformers.__version__)
if transformers_version < version.parse("5.0.0"):
    from transformers.modeling_utils import no_init_weights
else:
    from transformers.initialization import no_init_weights
from transformers.models.llama4.modeling_llama4 import Llama4Config, Llama4TextMLP, Llama4TextMoe

from auto_round.modeling.replace_modules import ReplacementModuleBase
from auto_round.modeling.utils import _update_parameter
from auto_round.utils import clear_memory, unsupported_meta_device


class SequentialLlama4TextExperts(torch.nn.ModuleList):
    def __init__(self, config, original):
        self.num_experts = original.gate_up_proj.shape[0]
        target_device = next(original.parameters()).device
        with no_init_weights(), torch.device("meta"):
            super().__init__([Llama4TextMLP(config) for _ in range(self.num_experts)])

    def _materialize_weights(self, original) -> None:
        if not unsupported_meta_device(original):
            intermediate_size = original.down_proj.shape[1]

            for i in range(self.num_experts):
                gate_up = original.gate_up_proj[i]
                down = original.down_proj[i]
                gate_proj = gate_up[:, :intermediate_size]
                up_proj = gate_up[:, intermediate_size:]
                _update_parameter(self[i].gate_proj, "weight", gate_proj.t().contiguous())
                _update_parameter(self[i].up_proj, "weight", up_proj.t().contiguous())
                _update_parameter(self[i].down_proj, "weight", down.t().contiguous())
            del gate_up, down, gate_proj, up_proj
            original.to_empty(device="meta")  # release original experts parameters


class SequentialLlama4TextMoe(ReplacementModuleBase):
    def __init__(self, original, config):
        super().__init__(original)
        config = config.text_config
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        with torch.device("meta"):
            self.experts = SequentialLlama4TextExperts(config, original.experts)

        self.router = original.router
        self.shared_expert = original.shared_expert

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = self.router(hidden_states)
        if isinstance(router_logits, tuple):
            router_scores, router_logits = router_logits
            router_scores = router_scores.t()
        else:
            # transformers < 4.54.0 only returns router_logits
            router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=1)

            router_scores = (
                torch.full_like(router_logits, float("-inf"))
                .scatter_(1, router_indices, router_top_value)
                .transpose(0, 1)
            )
            router_scores = torch.sigmoid(router_scores.float()).to(hidden_states.dtype)

        out = self.shared_expert(hidden_states)
        for i in range(self.num_experts):
            out += self.experts[i](hidden_states) * router_scores[i].reshape(-1, 1)

        return out, router_logits

    @classmethod
    def original_module_class(cls) -> str:
        """Return the class name of the module this replaces."""
        return "Llama4TextMoe"

    @classmethod
    def from_original(
        cls,
        original: torch.nn.Module,
        config: Llama4Config,
        **kwargs,
    ) -> "SequentialLlama4TextMoe":
        """Create an instance from the original module."""
        return cls(original, config)

    def _materialize_weights(self) -> None:
        original = self._get_original_module()
        self.experts._materialize_weights(original.experts)
        clear_memory()
