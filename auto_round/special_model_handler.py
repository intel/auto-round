# Copyright (c) 2023 Intel Corporation
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


mllms_with_limited_bs = ("llava", "qwen2_vl", "phi3_v", "mllama")  # Limitations on batch_size

SUPPORT_ONLY_TEXT_MODELS = [
    "phi3_v",
    "cogvlm2",
    "llava",
    "qwen2_vl",
    "qwen2_5_vl",
    "deepseek_vl_v2",
    "chatglm",
    "idefics3",
    "llama4",
    "internvl_chat",
]

NOT_SUPPORT_ONLY_TEXT_MODELS = ["mllama", "mistral3_2"]

SPECIAL_SHARED_CACHE_KEYS = {
    "Gemma3ForConditionalGeneration": ("position_embeddings_global", "position_embeddings_local")
}
SPECIAL_SHARED_CACHE_KEYS["MiniMaxText01ForCausalLM"] = ("slope_rate",)

CONVERT_EXPERT_TO_LINEAR_MODELS = ["llama4", "gpt_oss"]

def _get_moe_converter(config):
    import torch
    from transformers.modeling_utils import no_init_weights

    # https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modeling/llama4.py
    if config.model_type == "llama4":
        from transformers.models.llama4.modeling_llama4 import Llama4TextMLP

        class SequentialLlama4TextExperts(torch.nn.ModuleList):
            def __init__(self, config, original):
                self.num_experts = original.gate_up_proj.shape[0]
                with no_init_weights():
                    super().__init__([Llama4TextMLP(config) for _ in range(self.num_experts)])
                intermediate_size = original.down_proj.shape[1]

                for i in range(self.num_experts):
                    gate_up = original.gate_up_proj[i]
                    down = original.down_proj[i]
                    gate_proj = gate_up[:, :intermediate_size]
                    up_proj = gate_up[:, intermediate_size:]

                    self[i].gate_proj.weight.data = gate_proj.t().contiguous()
                    self[i].up_proj.weight.data = up_proj.t().contiguous()
                    self[i].down_proj.weight.data = down.t().contiguous()
 
        class SequentialLlama4TextMoe(torch.nn.Module):
            def __init__(self, config, original):
                super().__init__()
                self.top_k = config.num_experts_per_tok
                self.hidden_dim = config.hidden_size
                self.num_experts = config.num_local_experts
                self.experts = SequentialLlama4TextExperts(config, original.experts)
                self.router = original.router
                self.shared_expert = original.shared_expert

            def forward(self, hidden_states: torch.Tensor):
                hidden_states = hidden_states.reshape(-1, self.hidden_dim)
                router_logits = self.router(hidden_states)
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
                return out, router_scores

        return SequentialLlama4TextMoe, config.get_text_config(), "Llama4TextMoe"

    elif config.model_type == "gpt_oss":
        class SequentialGptOssExperts(torch.nn.Module):
            def __init__(self, config, original):
                super().__init__()
                self.intermediate_size = config.intermediate_size
                self.num_experts = config.num_local_experts
                self.hidden_size = config.hidden_size
                self.expert_dim = self.intermediate_size
                self.alpha = original.alpha
                self.limit = original.limit

                with no_init_weights():
                    self.gate_up_projs = torch.nn.ModuleList([
                        torch.nn.Linear(self.hidden_size, 2 * self.expert_dim)
                        for _ in range(self.num_experts)
                    ])

                    self.down_projs = torch.nn.ModuleList([
                        torch.nn.Linear(self.expert_dim, self.hidden_size)
                        for _ in range(self.num_experts)
                    ])

                for i in range(self.num_experts):
                    self.gate_up_projs[i].weight.data = original.gate_up_proj[i].t().contiguous()
                    self.gate_up_projs[i].bias.data = original.gate_up_proj_bias[i].contiguous()

                    self.down_projs[i].weight.data = original.down_proj[i].t().contiguous()
                    self.down_projs[i].bias.data = original.down_proj_bias[i].contiguous()

            def forward(self, hidden_states, router_indices=None, routing_weights=None):
                batch_size = hidden_states.shape[0]
                hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
                num_experts = routing_weights.shape[1]

                if self.training:
                    next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)

                    with torch.no_grad():
                        expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=num_experts)
                        expert_mask = expert_mask.permute(2, 1, 0)
                        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

                    for expert_idx in expert_hitted[:]:
                        with torch.no_grad():
                            _, token_idx = torch.where(expert_mask[expert_idx[0]])

                        current_state = hidden_states[token_idx]

                        gate_up = self.gate_up_projs[expert_idx](current_state)
                        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                        gate = gate.clamp(min=None, max=self.limit)
                        up = up.clamp(min=-self.limit, max=self.limit)

                        glu = gate * torch.sigmoid(gate * self.alpha)
                        gated_output = (up + 1) * glu

                        out = self.down_projs[expert_idx](gated_output)

                        weighted_output = out[0] * routing_weights[token_idx, expert_idx, None]
                        next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))

                    next_states = next_states.view(batch_size, -1, self.hidden_size)
                else:
                    hidden_states = hidden_states.repeat(num_experts, 1)
                    hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)

                    gate_up = torch.stack([proj(hidden_states[i]) for i, proj in enumerate(self.gate_up_projs)])
                    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                    gate = gate.clamp(min=None, max=self.limit)
                    up = up.clamp(min=-self.limit, max=self.limit)

                    glu = gate * torch.sigmoid(gate * self.alpha)
                    next_states = torch.stack([proj((up[i] + 1) * glu[i]) for i, proj in enumerate(self.down_projs)])

                    next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
                    next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
                    next_states = next_states.sum(dim=0)

                return next_states


        return SequentialGptOssExperts, config, "GptOssExperts"


def postprocess(model):
    if model.config.model_type == "gpt_oss":
        import torch
        import tqdm
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts
        for name, module in model.named_modules():
            cls_name = module.__class__.__name__
            if cls_name == "SequentialGptOssExperts":
                new_module = GptOssExperts(model.config)
                new_module.gate_up_proj.data = torch.stack([i.weight.data.t() for i in module.gate_up_projs])
                new_module.gate_up_proj_bias.data = torch.stack([i.bias.data for i in module.gate_up_projs])
                new_module.down_proj.data = torch.stack([i.weight.data.t() for i in module.down_projs])
                new_module.down_proj_bias.data = torch.stack([i.bias.data for i in module.down_projs])
                parent, child = name.rsplit(".", maxsplit=1)
                parent = model.get_submodule(parent)
                setattr(parent, child, new_module)
    return model


def _handle_special_model(model):
    if model.config.model_type == "deepseek_vl_v2":
        from functools import partial

        model.forward = partial(_deepseek_vl2_forward, model)

    return model


def _convert_to_linear(model):
    if model.config.model_type in CONVERT_EXPERT_TO_LINEAR_MODELS:
        from auto_round.utils import clear_memory
        from tqdm import tqdm

        new_moe_class, convert_config, orig_cls_name = _get_moe_converter(model.config)
        model = model.to("cpu")
        clear_memory()

        for name, module in tqdm(model.named_modules(), desc="Converting model"):
            cls_name = module.__class__.__name__
            if cls_name == orig_cls_name:
                new_module = new_moe_class(config=convert_config, original=module)
                parent, child = name.rsplit(".", maxsplit=1)
                parent = model.get_submodule(parent)
                setattr(parent, child, new_module)
    return model


def _get_deepseek_vl2_multimodal_block(model, quant_vision=False):
    model.forward = model.language.forward
    block_names = []
    if quant_vision:
        block_names.append([f"vision.blocks.{i}" for i in range(len(model.vision.blocks))])
        block_names.append([f"projector.layers.{i}" for i in range(len(model.projector.layers))])
    block_names.append([f"language.model.layers.{i}" for i in range(len(model.language.model.layers))])
    return block_names


SPECIAL_MULTIMODAL_BLOCK = {"deepseek_vl_v2": _get_deepseek_vl2_multimodal_block}


def _deepseek_vl2_forward(
    model,
    input_ids=None,
    position_ids=None,
    attention_mask=None,
    past_key_values=None,
    inputs_embeds=None,
    images=None,
    images_seq_mask=None,
    images_spatial_crop=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    cache_position=None,
    **kwargs,
):
    inputs_embeds = model.prepare_inputs_embeds(
        input_ids=input_ids,
        images=images,
        images_seq_mask=images_seq_mask,
        images_spatial_crop=images_spatial_crop,
    )
    return model.language(
        input_ids=None,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )


def check_mllm_model_batch(model, batch_size, gradient_accumulate_steps=1):
    """
    Checks model configuration to determine if it's necessary to limit bs to avoid potential input shape mismatches.
    """
    for key in mllms_with_limited_bs:
        if hasattr(model, "config") and key in model.config.model_type and batch_size != 1:
            accumulate_steps = batch_size * gradient_accumulate_steps
            print(
                "To avoid the tensor concat mismatch problem, modified parameters to "
                f"batch_size=1. As an alternative, set the gradient_accumulate_steps={accumulate_steps}"
            )
            return 1, accumulate_steps
    return batch_size, gradient_accumulate_steps
