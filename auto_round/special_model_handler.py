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
from auto_round.utils import logger

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
    "glm4v_moe",
]

NOT_SUPPORT_ONLY_TEXT_MODELS = ["mllama", "mistral3_2"]

SPECIAL_SHARED_CACHE_KEYS = {
    "Gemma3ForConditionalGeneration": ("position_embeddings_global", "position_embeddings_local")
}
SPECIAL_SHARED_CACHE_KEYS["MiniMaxText01ForCausalLM"] = ("slope_rate",)

CONVERT_EXPERT_TO_LINEAR_MODELS = ["llama4"]


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

        return SequentialLlama4TextMoe, config.get_text_config(), "Llama4TextMoe"

    else:
        raise ValueError(f"Currently moe converter only supports llama4 model_type, but get {config.model_type}")


def _handle_special_model(model):
    if model.config.model_type == "deepseek_vl_v2":
        from functools import partial

        model.forward = partial(_deepseek_vl2_forward, model)
    return model


def _handle_moe_model(model, formats=None):
    if formats is not None and any(["gguf" in format_ for format_ in formats]):
        return model
    if hasattr(model.config, "model_type") and model.config.model_type in CONVERT_EXPERT_TO_LINEAR_MODELS:
        from tqdm import tqdm

        from auto_round.utils import clear_memory

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

        logger.warning("Llama4 experts are converted, the quantized model can not run on transformers.")
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
            logger.warning(
                "To avoid the tensor concat mismatch problem, modified parameters to "
                f"batch_size=1. As an alternative, set the gradient_accumulate_steps={accumulate_steps}"
            )
            return 1, accumulate_steps
    return batch_size, gradient_accumulate_steps
