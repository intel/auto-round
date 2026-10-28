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


def _get_qwen2_5_omni_multimodal_block(model, quant_vision=False):
    """Get block names for Qwen2.5-Omni model.

    Qwen2.5-Omni has the following structure:
    - thinker: Contains audio_tower, visual, model (text decoder)
    - talker: Contains model (talker decoder)
    - token2wav: Audio decoder

    For quantization, we focus on:
    - thinker.model.layers (text decoder layers) - main LLM layers
    - Optionally: visual encoder blocks, audio encoder layers

    talker is excluded by default because quantizing it has been observed to
    degrade audio quality in long-form generation.
    """
    block_names = []

    # Quantize visual encoder blocks if quant_vision is enabled
    if quant_vision:
        if hasattr(model, "thinker") and hasattr(model.thinker, "visual") and hasattr(model.thinker.visual, "blocks"):
            block_names.append([f"thinker.visual.blocks.{i}" for i in range(len(model.thinker.visual.blocks))])
        if hasattr(model, "thinker") and hasattr(model.thinker, "audio_tower"):
            if hasattr(model.thinker.audio_tower, "layers"):
                block_names.append(
                    [f"thinker.audio_tower.layers.{i}" for i in range(len(model.thinker.audio_tower.layers))]
                )

    # Thinker text model layers (main LLM decoder)
    if hasattr(model, "thinker") and hasattr(model.thinker, "model") and hasattr(model.thinker.model, "layers"):
        block_names.append([f"thinker.model.layers.{i}" for i in range(len(model.thinker.model.layers))])

    return block_names



def _get_qwen3_omni_thinker_layer_info(model):
    """Return the thinker layer path and module for supported layouts.

    Transformers historically exposed ``thinker.model.layers``. Newer
    Qwen3-Omni MoE builds nest the decoder under
    ``thinker.model.language_model.layers``. Resolve either layout.
    """
    thinker_model = getattr(getattr(model, "thinker", None), "model", None)
    if thinker_model is None:
        return None, None

    if hasattr(thinker_model, "layers"):
        return "thinker.model.layers", thinker_model.layers

    language_model = getattr(thinker_model, "language_model", None)
    if language_model is not None and hasattr(language_model, "layers"):
        return "thinker.model.language_model.layers", language_model.layers

    return None, None


def _get_qwen3_omni_moe_multimodal_block(model, quant_vision=False):
    """Get block names for Qwen3-Omni MoE model.

    Qwen3-Omni has the following structure:
    - thinker: Contains audio_tower, visual, model (text decoder)
    - talker: Contains model (talker decoder), code_predictor
    - code2wav: Audio decoder

    For quantization, we focus on:
    - thinker.model.layers (text decoder layers) - main LLM layers
    - Optionally: visual encoder blocks, audio encoder layers

    talker is excluded by default because quantizing it has been observed to
    degrade audio quality in long-form generation.
    """
    block_names = []

    # Quantize visual encoder blocks if quant_vision is enabled
    if quant_vision:
        # Vision encoder blocks
        if hasattr(model, "thinker") and hasattr(model.thinker, "visual") and hasattr(model.thinker.visual, "blocks"):
            block_names.append([f"thinker.visual.blocks.{i}" for i in range(len(model.thinker.visual.blocks))])
        # Audio encoder layers
        if hasattr(model, "thinker") and hasattr(model.thinker, "audio_tower"):
            if hasattr(model.thinker.audio_tower, "layers"):
                block_names.append(
                    [f"thinker.audio_tower.layers.{i}" for i in range(len(model.thinker.audio_tower.layers))]
                )

    # Thinker text model layers (main LLM decoder)
    layer_prefix, layers = _get_qwen3_omni_thinker_layer_info(model)
    if layers is not None:
        block_names.append([f"{layer_prefix}.{i}" for i in range(len(layers))])

    return block_names


def _qwen2_5_omni_forward(
    model,
    input_ids=None,
    input_features=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
    feature_attention_mask=None,
    audio_feature_lengths=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    rope_deltas=None,
    labels=None,
    use_cache=None,
    use_audio_in_video=None,
    cache_position=None,
    video_second_per_grid=None,
    **kwargs,
):
    """Forward function for Qwen2.5-Omni model.

    This delegates to the thinker module for calibration, then optionally
    runs a forward through the talker to ensure its layers are also calibrated.
    """
    thinker_output = model.thinker(
        input_ids=input_ids,
        input_features=input_features,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        feature_attention_mask=feature_attention_mask,
        audio_feature_lengths=audio_feature_lengths,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        rope_deltas=rope_deltas,
        labels=labels,
        use_cache=use_cache,
        use_audio_in_video=use_audio_in_video,
        cache_position=cache_position,
        video_second_per_grid=video_second_per_grid,
        output_hidden_states=True,
        return_dict=True,
        **kwargs,
    )

    # Run talker forward if available (for calibration purposes)
    if hasattr(model, "talker") and model.has_talker:
        thinker_hidden = thinker_output.hidden_states[-1] if thinker_output.hidden_states else None

        if thinker_hidden is not None:
            # ---- calibrate thinker_to_talker_proj (nn.Linear) ----
            thinker_embeds = model.thinker.get_input_embeddings()(input_ids)
            proj_dtype = next(model.talker.thinker_to_talker_proj.parameters()).dtype
            talker_inputs_embeds = model.talker.thinker_to_talker_proj(thinker_embeds.to(proj_dtype))

            # ---- calibrate talker decoder layers ----
            talker_dtype = next(model.talker.model.parameters()).dtype
            talker_output = model.talker.model(
                inputs_embeds=talker_inputs_embeds.to(talker_dtype),
                attention_mask=attention_mask,
                use_cache=False,
            )

            # ---- calibrate codec_head (nn.Linear) ----
            if hasattr(model.talker, "codec_head"):
                talker_last_hidden = (
                    talker_output[0]
                    if not hasattr(talker_output, "last_hidden_state")
                    else talker_output.last_hidden_state
                )
                _ = model.talker.codec_head(talker_last_hidden.to(talker_dtype))

    return thinker_output


def _qwen3_omni_moe_forward(
    model,
    input_ids=None,
    input_features=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
    feature_attention_mask=None,
    audio_feature_lengths=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    rope_deltas=None,
    labels=None,
    use_cache=None,
    output_router_logits=None,
    use_audio_in_video=None,
    cache_position=None,
    video_second_per_grid=None,
    **kwargs,
):
    """Forward function for Qwen3-Omni-MoE model.

    This runs forward through both thinker and talker modules for calibration.
    The thinker processes text/vision/audio input, and talker uses thinker's
    hidden states to prepare for speech synthesis.

    In real inference the talker receives inputs built from two projections:
      - ``text_projection``: maps thinker word embeddings → talker hidden dim
        (used for pure-text tokens).
      - ``hidden_projection``: maps thinker intermediate hidden states → talker
        hidden dim (used for multimodal tokens).

    During calibration we exercise both projection paths and then run one
    forward pass through the talker decoder so that every linear layer
    (attention, MoE/MLP, codec_head) observes representative activations.
    """
    # Run thinker forward with output_hidden_states to get hidden states for talker
    thinker_output = model.thinker(
        input_ids=input_ids,
        input_features=input_features,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        feature_attention_mask=feature_attention_mask,
        audio_feature_lengths=audio_feature_lengths,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        rope_deltas=rope_deltas,
        labels=labels,
        use_cache=use_cache,
        output_router_logits=output_router_logits,
        use_audio_in_video=use_audio_in_video,
        cache_position=cache_position,
        video_second_per_grid=video_second_per_grid,
        output_hidden_states=True,
        return_dict=True,
        **kwargs,
    )

    # Run talker forward if available (for calibration purposes)
    if getattr(model, "has_talker", False) and getattr(model, "talker", None) is not None:
        thinker_hidden = thinker_output.hidden_states[-1] if thinker_output.hidden_states else None

        if thinker_hidden is not None:
            # ---- calibrate text_projection (ResizeMLP) ----
            thinker_embeds = model.thinker.get_input_embeddings()(input_ids)
            proj_dtype = next(model.talker.text_projection.parameters()).dtype
            talker_inputs_embeds = model.talker.text_projection(thinker_embeds.to(proj_dtype))

            # ---- calibrate hidden_projection (ResizeMLP) ----
            if hasattr(model.talker, "hidden_projection"):
                hidden_proj_dtype = next(model.talker.hidden_projection.parameters()).dtype
                _ = model.talker.hidden_projection(thinker_hidden.to(hidden_proj_dtype))

            # ---- calibrate talker decoder layers ----
            talker_dtype = next(model.talker.model.parameters()).dtype
            talker_output = model.talker.model(
                inputs_embeds=talker_inputs_embeds.to(talker_dtype),
                attention_mask=attention_mask,
                use_cache=False,
            )

            # ---- calibrate codec_head ----
            if hasattr(model.talker, "codec_head"):
                talker_last_hidden = (
                    talker_output[0]
                    if not hasattr(talker_output, "last_hidden_state")
                    else talker_output.last_hidden_state
                )
                _ = model.talker.codec_head(talker_last_hidden.to(talker_dtype))

    return thinker_output


SPECIAL_MULTIMODAL_BLOCK["qwen2_5_omni"] = _get_qwen2_5_omni_multimodal_block
SPECIAL_MULTIMODAL_BLOCK["qwen3_omni_moe"] = _get_qwen3_omni_moe_multimodal_block
