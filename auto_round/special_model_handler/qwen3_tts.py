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
from auto_round.special_model_handler.core import SPECIAL_MULTIMODAL_BLOCK


def _get_qwen3_tts_multimodal_block(model, quant_vision=False):
    """Get block names for Qwen3-TTS model.

    Qwen3-TTS (Qwen3TTSForConditionalGeneration) has a talker backbone with the
    following possible module layouts depending on the modeling code version:
    - tts_model.model.layers: Main TTS transformer decoder layers
    - talker.model.layers: Alternative attribute name (matches talker_config)
    - model.layers: Standard fallback (via the model.model.layers attribute)

    Only the talker backbone is quantized. Audio codec layers are not quantized
    because audio calibration data is not yet supported.

    Args:
        model: The Qwen3TTSForConditionalGeneration model instance.
        quant_vision: Unused. Accepted for interface compatibility with
            SPECIAL_MULTIMODAL_BLOCK registry.
    """
    block_names = []
    # Try tts_model.model.layers
    if hasattr(model, "tts_model") and hasattr(getattr(model.tts_model, "model", None), "layers"):
        block_names.append([f"tts_model.model.layers.{i}" for i in range(len(model.tts_model.model.layers))])
    # Try talker.model.layers (alternative attr name from talker_config)
    if not block_names and hasattr(model, "talker") and hasattr(getattr(model.talker, "model", None), "layers"):
        block_names.append([f"talker.model.layers.{i}" for i in range(len(model.talker.model.layers))])
    # Fallback: model.layers named-module path (model.model.layers attribute)
    if not block_names and hasattr(model, "model") and hasattr(model.model, "layers"):
        block_names.append([f"model.layers.{i}" for i in range(len(model.model.layers))])
    return block_names


def _qwen3_tts_forward(
    model,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    cache_position=None,
    **kwargs,
):
    """Forward function for Qwen3-TTS model.

    Qwen3-TTS uses a discrete multi-codebook LM architecture. During calibration
    we route the forward through the talker, converting text input_ids to
    inputs_embeds via text_embedding + text_projection.
    """
    tts_backbone = None
    for candidate in [getattr(model, "tts_model", None), getattr(model, "talker", None)]:
        if candidate is None:
            continue
        has_text_embed = hasattr(getattr(candidate, "model", None), "text_embedding")
        has_text_proj = hasattr(candidate, "text_projection")
        if has_text_embed and has_text_proj:
            tts_backbone = candidate
            break
    # Fallback: use whichever backbone exists even without text pathway
    if tts_backbone is None:
        tts_backbone = getattr(model, "tts_model", None) or getattr(model, "talker", None)

    if tts_backbone is not None:
        # Convert text input_ids to inputs_embeds through text pathway
        if input_ids is not None and inputs_embeds is None:
            text_embedding = getattr(tts_backbone.model, "text_embedding", None)
            text_projection = getattr(tts_backbone, "text_projection", None)
            if text_embedding is None or text_projection is None:
                raise RuntimeError(
                    "Qwen3-TTS backbone is missing text_embedding or text_projection. "
                    "Cannot convert text input_ids to inputs_embeds for calibration."
                )
            inputs_embeds = text_projection(text_embedding(input_ids))
            input_ids = None

        return tts_backbone(
            input_ids=input_ids,
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
            **kwargs,
        )
    # Fallback: model has standard forward
    return model.__class__.forward(
        model,
        input_ids=input_ids,
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
        **kwargs,
    )


SPECIAL_MULTIMODAL_BLOCK["qwen3_tts"] = _get_qwen3_tts_multimodal_block
