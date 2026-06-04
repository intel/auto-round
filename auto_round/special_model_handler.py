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
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from auto_round.formats import OutputFormat
from auto_round.modeling.fused_moe.replace_modules import apply_replacements, release_original_module_
from auto_round.utils import is_moe_model_via_config, logger

mllms_with_limited_bs = (
    "llava",
    "qwen2_vl",
    "phi3_v",
    "mllama",
    "qwen2_5_omni",
    "qwen3_omni_moe",
    "glm_image",
    "mimo_audio",
    "qwen3_tts",
)  # Limitations on batch_size

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
    "glm_image",
    "qwen3_vl_moe",
    "qwen2_5_omni",
    "qwen3_omni_moe",
    "gemma3",
    "bagel",
    "mimo_audio",
    "qwen3_tts",
]

NOT_SUPPORT_ONLY_TEXT_MODELS = ["mllama", "mistral3_2"]

# Maps architecture class names to virtual model_type keys.
# Used when config.model_type doesn't uniquely identify the model (e.g. MiMo-Audio).
from auto_round.utils.model import ARCHITECTURE_MODEL_TYPE_MAP, resolve_model_type  # noqa: E402

SPECIAL_SHARED_CACHE_KEYS = {
    "Gemma3ForConditionalGeneration": ("position_embeddings_global", "position_embeddings_local")
}
SPECIAL_SHARED_CACHE_KEYS["MiniMaxText01ForCausalLM"] = ("slope_rate",)
SPECIAL_SHARED_CACHE_KEYS["StableAudioDiTModel"] = ("encoder_hidden_states",)
SPECIAL_SHARED_CACHE_KEYS["Gemma4ForConditionalGeneration"] = ("position_ids",)
SPECIAL_SHARED_CACHE_KEYS["WanTransformer3DModel"] = ("rotary_emb",)
MISTRAL_3_2_MODELS = ["Mistral-Small-3.2", "Magistral-Small", "Devstral-Small"]


def _normalize_gemma4_per_layer_input(positional_inputs, hidden_states):
    if positional_inputs is None or len(positional_inputs) == 0:
        return positional_inputs

    per_layer_input = positional_inputs[0]
    if not isinstance(per_layer_input, torch.Tensor) or per_layer_input.shape[1] == hidden_states.shape[1]:
        return positional_inputs

    hs_seq = hidden_states.shape[1]
    pl_seq = per_layer_input.shape[1]
    if hs_seq <= pl_seq:
        per_layer_input = per_layer_input[:, :hs_seq, :]
    else:
        pad = per_layer_input[:, -1:, :].expand(-1, hs_seq - pl_seq, -1)
        per_layer_input = torch.cat([per_layer_input, pad], dim=1)

    normalized_inputs = list(positional_inputs)
    normalized_inputs[0] = per_layer_input
    return type(positional_inputs)(normalized_inputs) if isinstance(positional_inputs, tuple) else normalized_inputs


def prepare_special_model_block_inputs(block, rotary_input, input_others, positional_inputs=None):
    """Rewrite replay inputs for blocks that need model-specific handling."""

    # Guard: ensure position_ids is a tensor, not a list or None.
    if "position_ids" in input_others:
        pid = input_others["position_ids"]
        if isinstance(pid, list):
            if len(pid) == 1:
                input_others["position_ids"] = pid[0]
            elif len(pid) == 0:
                input_others["position_ids"] = (
                    torch.arange(rotary_input.shape[1], device=rotary_input.device, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(rotary_input.shape[0], -1)
                )
        elif pid is None:
            input_others["position_ids"] = (
                torch.arange(rotary_input.shape[1], device=rotary_input.device, dtype=torch.long)
                .unsqueeze(0)
                .expand(rotary_input.shape[0], -1)
            )

    special_replay_type = getattr(block, "_autoround_special_replay", None)
    if special_replay_type == "gemma4" or special_replay_type == "gemma4_unified":
        prepared_inputs = _prepare_gemma4_replay_inputs(
            block,
            rotary_input,
            position_ids=input_others.get("position_ids"),
            position_embeddings=input_others.get("position_embeddings"),
            attention_mask=input_others.get("attention_mask"),
            shared_kv_states=input_others.get("shared_kv_states"),
            past_key_values=input_others.get("past_key_values"),
            config=getattr(block, "_gemma4_config_ref", None),
        )
        for key, value in prepared_inputs.items():
            if value is not None or key in input_others or key == "shared_kv_states":
                input_others[key] = value
        positional_inputs = _normalize_gemma4_per_layer_input(positional_inputs, rotary_input)
    return input_others, positional_inputs


def _get_gemma4_shared_kv_states_global(block):
    """Return the shared KV states dict for Gemma4 block-wise quantization."""
    ref = getattr(block, "_shared_kv_states_global_ref", None)
    if ref is not None:
        return ref
    return {}


def _get_gemma4_rotary_emb(block, default_rotary_emb=None):
    rotary_emb_ref = getattr(block, "_rotary_emb_ref", None)
    if rotary_emb_ref:
        return rotary_emb_ref[0]
    return getattr(block, "_rotary_emb", default_rotary_emb)


def _rebuild_gemma4_attention_mask(config, hidden_states, position_ids, layer_type, past_key_values=None):
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    mask_builder = create_causal_mask if layer_type == "full_attention" else create_sliding_window_causal_mask
    return mask_builder(
        config=config,
        inputs_embeds=hidden_states,
        attention_mask=None,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )


def _prepare_gemma4_replay_inputs(
    block,
    rotary_input,
    *,
    position_ids=None,
    position_embeddings=None,
    attention_mask=None,
    shared_kv_states=None,
    past_key_values=None,
    config=None,
    default_rotary_emb=None,
    default_shared_kv_states=None,
):
    attn = getattr(block, "self_attn", None)
    layer_type = getattr(attn, "layer_type", None)
    head_dim = getattr(attn, "head_dim", None)

    if attn is not None and hasattr(attn, "store_full_length_kv") and shared_kv_states is None:
        if default_shared_kv_states is not None:
            shared_kv_states = default_shared_kv_states
        else:
            shared_kv_states = _get_gemma4_shared_kv_states_global(block)
            if getattr(block, "layer_idx", None) == 0:
                shared_kv_states.clear()

    need_position_embeddings = position_embeddings is None
    if isinstance(position_embeddings, dict):
        cached_position_embeddings = position_embeddings.get(layer_type) if layer_type is not None else None
        need_position_embeddings = cached_position_embeddings is None
    else:
        cached_position_embeddings = position_embeddings

    if (
        not need_position_embeddings
        and head_dim is not None
        and isinstance(cached_position_embeddings, (tuple, list))
        and cached_position_embeddings
    ):
        need_position_embeddings = cached_position_embeddings[0].shape[-1] != head_dim

    if need_position_embeddings and layer_type is not None and position_ids is not None:
        rotary_emb = _get_gemma4_rotary_emb(block, default_rotary_emb)
        if rotary_emb is not None:
            rebuilt_position_embeddings = rotary_emb(rotary_input, position_ids, layer_type)
            if isinstance(position_embeddings, dict):
                position_embeddings = dict(position_embeddings)
                position_embeddings[layer_type] = rebuilt_position_embeddings
            else:
                position_embeddings = rebuilt_position_embeddings

    if config is not None and layer_type is not None and position_ids is not None:
        try:
            attention_mask = _rebuild_gemma4_attention_mask(
                config,
                hidden_states=rotary_input,
                position_ids=position_ids,
                layer_type=layer_type,
                past_key_values=past_key_values,
            )
        except Exception:
            pass

    return {
        "position_embeddings": position_embeddings,
        "attention_mask": attention_mask,
        "shared_kv_states": shared_kv_states,
    }


def _patch_gemma4_model(model):
    """Patch each Gemma4 decoder layer so it recomputes position_embeddings and
    attention_mask from the cached position_ids when the cached versions have
    wrong dimensions (sliding_attention vs full_attention head_dims differ).

    During auto-round block-wise quantization the cached inputs from block 0
    (always a sliding_attention layer) are reused for every subsequent block.
    Full-attention layers (head_dim=512) would receive position embeddings
    computed for sliding layers (head_dim=256), causing a shape mismatch crash.
    """
    import types as _types

    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel
    except ImportError:
        return model

    # Find the Gemma4TextModel by traversing the hierarchy
    text_model = None
    for _, submodule in model.named_modules():
        if isinstance(submodule, Gemma4TextModel):
            text_model = submodule
            break

    if text_model is None:
        return model

    rotary_emb = text_model.rotary_emb
    # Shared dict to propagate KV state between anchor/sharer layers (like
    # Gemma4TextModel.forward does in newer transformers versions).
    shared_kv_states_global = {}

    for layer in text_model.layers:
        object.__setattr__(layer, "_autoround_special_replay", "gemma4")
        object.__setattr__(layer, "_gemma4_config_ref", text_model.config)
        original_forward = layer.forward
        layer_type = getattr(getattr(layer, "self_attn", None), "layer_type", None)
        if layer_type is None:
            continue
        head_dim = getattr(getattr(layer, "self_attn", None), "head_dim", None)
        is_full = layer_type == "full_attention"

        def _make_layer_forward(orig_fwd, lt, hd, is_full_attn, re, cfg, skv_global):
            import inspect

            orig_params = list(inspect.signature(orig_fwd).parameters)
            orig_has_shared_kv = "shared_kv_states" in orig_params

            def patched_layer_forward(
                self,
                hidden_states,
                per_layer_input=None,
                shared_kv_states=None,
                position_embeddings=None,
                attention_mask=None,
                position_ids=None,
                **kwargs,
            ):
                # Rebuild Gemma4 layer-specific replay inputs from the minimal
                # shared cache so later layers do not need variable block inputs.
                if shared_kv_states is None and getattr(self, "layer_idx", None) == 0:
                    skv_global.clear()
                prepared_inputs = _prepare_gemma4_replay_inputs(
                    self,
                    hidden_states,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    shared_kv_states=shared_kv_states,
                    past_key_values=kwargs.get("past_key_values"),
                    config=cfg,
                    default_rotary_emb=re,
                    default_shared_kv_states=skv_global,
                )
                position_embeddings = prepared_inputs["position_embeddings"]
                attention_mask = prepared_inputs["attention_mask"]
                shared_kv_states = prepared_inputs["shared_kv_states"]

                # per_layer_input is token-specific but is cached as shared positional
                # input (only 1st batch stored). Truncate/pad to match hidden_states seq_len.
                if per_layer_input is not None and per_layer_input.shape[1] != hidden_states.shape[1]:
                    hs_seq = hidden_states.shape[1]
                    pl_seq = per_layer_input.shape[1]
                    if hs_seq <= pl_seq:
                        per_layer_input = per_layer_input[:, :hs_seq, :]
                    else:
                        pad = per_layer_input[:, -1:, :].expand(-1, hs_seq - pl_seq, -1)
                        per_layer_input = torch.cat([per_layer_input, pad], dim=1)

                if orig_has_shared_kv:
                    return orig_fwd(
                        hidden_states,
                        per_layer_input,
                        shared_kv_states,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **kwargs,
                    )
                return orig_fwd(
                    hidden_states,
                    per_layer_input,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **kwargs,
                )

            return patched_layer_forward

        layer.forward = _types.MethodType(
            _make_layer_forward(
                original_forward, layer_type, head_dim, is_full, rotary_emb, text_model.config, shared_kv_states_global
            ),
            layer,
        )

    return model


def _handle_special_model(model):
    model_type = resolve_model_type(model)
    if model_type == "deepseek_vl_v2":
        from functools import partial

        model.forward = partial(_deepseek_vl2_forward, model)
    if model_type == "qwen2_5_omni":
        from functools import partial

        model.forward = partial(_qwen2_5_omni_forward, model)
    if model_type == "qwen3_omni_moe":
        from functools import partial

        model.forward = partial(_qwen3_omni_moe_forward, model)
    if model_type == "qwen3_tts":
        from functools import partial

        model.forward = partial(_qwen3_tts_forward, model)
    if model_type == "mimo_audio":
        from functools import partial

        model.forward = partial(_mimo_audio_forward, model)
    if hasattr(model, "config") and (model_type == "gemma4"):
        import transformers
        from packaging import version

        if version.parse(transformers.__version__) < version.parse("5.6"):
            _patch_gemma4_model(model)
        else:
            _attach_gemma4_rotary_emb(model)
        logger.warning(
            "Applying a monkey patch to Gemma4 to reduce RAM usage. "
            "This patch has only been validated with limited Transformers versions. "
            "Proceed with caution."
        )
    if hasattr(model, "config") and model_type == "gemma4_unified":
        _attach_gemma4_unified_rotary_emb(model)
    return model


def update_module(
    model, formats: list[OutputFormat] = None, trust_remote_code: bool = True, cleanup_original: bool = True
):
    if formats is not None and any([format_.is_gguf() for format_ in formats]):
        return model

    model = apply_replacements(model)

    if cleanup_original:
        release_original_module_(model)

    return model


def _get_deepseek_vl2_multimodal_block(model, quant_vision=False):
    model.forward = model.language.forward
    block_names = []
    if quant_vision:
        block_names.append([f"vision.blocks.{i}" for i in range(len(model.vision.blocks))])
        block_names.append([f"projector.layers.{i}" for i in range(len(model.projector.layers))])
    block_names.append([f"language.model.layers.{i}" for i in range(len(model.language.model.layers))])
    return block_names


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
    degrade audio quality in long-form generation .
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
    if hasattr(model, "thinker") and hasattr(model.thinker, "model") and hasattr(model.thinker.model, "layers"):
        block_names.append([f"thinker.model.layers.{i}" for i in range(len(model.thinker.model.layers))])

    return block_names


def _get_glm_image_multimodal_block(model, quant_vision=False):
    """Get block names for GLM-Image AR model.

    GLM-Image AR model structure:
    - model.visual.blocks: vision encoder
    - model.language_model.layers: autoregressive text backbone

    By default, only text backbone is quantized. Set quant_vision=True to include
    the visual encoder blocks.
    """
    block_names = []

    if quant_vision and hasattr(model, "model") and hasattr(model.model, "visual"):
        if hasattr(model.model.visual, "blocks"):
            block_names.append([f"model.visual.blocks.{i}" for i in range(len(model.model.visual.blocks))])

    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        if hasattr(model.model.language_model, "layers"):
            block_names.append(
                [f"model.language_model.layers.{i}" for i in range(len(model.model.language_model.layers))]
            )

    return block_names


def _get_mimo_audio_multimodal_block(model, quant_vision=False):
    """Get block names for MiMo-Audio model.

    MiMo-Audio (MiMoAudioForCausalLM) has the following structure:
    - model.model.layers: Main Qwen2-based LLM decoder (28 layers)
    - input_local_transformer.layers: Input audio encoder (6 layers)
    - local_transformer.layers: Audio decoder / patch decoder (16 layers)

    Currently only the main LLM backbone is quantized because:
    - Audio encoder/decoder require audio calibration data (not yet supported)
    - Audio modules use smaller hidden dim (1024 vs 4096) with limited quantization benefit

    Args:
        model: The MiMoAudioForCausalLM or MiMoAudioModel instance.
    """
    block_names = []

    # Main LLM decoder layers (backbone - always quantized)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        # MiMoAudioForCausalLM: layers at model.model.layers
        block_names.append([f"model.layers.{i}" for i in range(len(model.model.layers))])
    elif hasattr(model, "layers"):
        # MiMoAudioModel (base): layers at model.layers directly
        block_names.append([f"layers.{i}" for i in range(len(model.layers))])

    return block_names


def _get_qwen3_tts_multimodal_block(model, quant_vision=False):
    """Get block names for Qwen3-TTS model.

    Qwen3-TTS (Qwen3TTSForConditionalGeneration) has a talker backbone with the
    following possible module layouts depending on the modeling code version:
    - tts_model.model.layers: Main TTS transformer decoder layers
    - talker.model.layers: Alternative attribute name (matches talker_config)
    - model.model.layers: Standard fallback

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
    # Fallback: model.model.layers (standard structure)
    if not block_names and hasattr(model, "model") and hasattr(model.model, "layers"):
        block_names.append([f"model.layers.{i}" for i in range(len(model.model.layers))])
    return block_names


def _get_bagel_multimodal_block(model, quant_vision=False):
    """Get block names for BAGEL MoT (Mixture of Transformers) model.

    BAGEL model structure:
    - language_model.model.layers: Qwen2-based LLM with MoT dual paths
    - vit_model: SigLIP vision encoder (not quantized by default)
    - connector: Vision-language MLP connector
    - encoder/decoder: VAE autoencoder
    - time_embedder, vae2llm, llm2vae: bridge modules

    By default, only the language_model layers are quantized.
    """
    block_names = []

    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        if hasattr(model.language_model.model, "layers"):
            block_names.append(
                [f"language_model.model.layers.{i}" for i in range(len(model.language_model.model.layers))]
            )

    return block_names


SPECIAL_MULTIMODAL_BLOCK = {
    "deepseek_vl_v2": _get_deepseek_vl2_multimodal_block,
    "qwen2_5_omni": _get_qwen2_5_omni_multimodal_block,
    "qwen3_omni_moe": _get_qwen3_omni_moe_multimodal_block,
    "glm_image": _get_glm_image_multimodal_block,
    "mimo_audio": _get_mimo_audio_multimodal_block,
    "qwen3_tts": _get_qwen3_tts_multimodal_block,
    "bagel": _get_bagel_multimodal_block,
}


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


def _mimo_audio_forward(
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
    """Forward function for MiMo-Audio model during calibration.

    MiMo-Audio's native forward expects 3D input_ids [B, audio_channels+1, T] with
    interleaved text and audio tokens. During calibration with text-only data, we
    route through the Qwen2Model backbone directly to calibrate the main decoder layers.
    """
    backbone = model.model

    if input_ids is not None and inputs_embeds is None:
        inputs_embeds = backbone.embed_tokens(input_ids)
        input_ids = None

    return backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )


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


def check_mllm_model_batch(model, batch_size, gradient_accumulate_steps=1):
    """
    Checks model configuration to determine if it's necessary to limit bs to avoid potential input shape mismatches.
    """
    effective_type = resolve_model_type(model) or ""
    for key in mllms_with_limited_bs:
        if key in effective_type and batch_size != 1:
            accumulate_steps = batch_size * gradient_accumulate_steps
            logger.warning(
                "To avoid the tensor concat mismatch problem, modified parameters to "
                f"batch_size=1. As an alternative, set the gradient_accumulate_steps={accumulate_steps}"
            )
            return 1, accumulate_steps
    return batch_size, gradient_accumulate_steps


class ModelNameMatcher:
    """model.config.name_or_path"""

    def __init__(self, pattern: str, mode="in"):
        self.pattern = pattern
        self.mode = mode

    def __call__(self, model) -> bool:
        name = getattr(model.config, "name_or_path", "")
        if self.mode == "full":
            return name == self.pattern
        elif self.mode == "in":
            return self.pattern in name
        elif self.mode == "regex":
            return re.search(self.pattern, name) is not None
        else:
            raise ValueError("unsupported mode {self.mode}")

    """Matches config.architectures."""


class ArchitectureMatcher:
    """match config.architectures"""

    def __init__(self, arch: str, mode="in"):
        self.arch = arch
        self.mode = mode

    def __call__(self, model) -> bool:
        archs = getattr(model.config, "architectures", [])
        archs_str = ",".join(archs) if isinstance(archs, list) else str(archs)

        if self.mode == "full":
            return archs_str == self.arch
        elif self.mode == "in":
            return self.arch in archs_str
        elif self.mode == "regex":
            return re.search(self.arch, archs_str) is not None
        else:
            raise ValueError(f"unsupported mode {self.mode}")


class ModelTypeMatcher:
    """match config.architectures"""

    def __init__(self, model_type: str, mode="in"):
        self.model_type = model_type
        self.mode = mode

    def __call__(self, model) -> bool:
        model_type = getattr(model.config, "model_type", None)
        if model_type is None:
            return False

        if self.mode == "full":
            return model_type == self.model_type
        elif self.mode == "in":
            return self.model_type in model_type
        elif self.mode == "regex":
            return re.search(self.model_type, model_type) is not None
        else:
            raise ValueError(f"unsupported mode {self.mode}")


@dataclass
class PreDefinedIgnoreLayers:
    matchers: list[Callable[[Any], bool]]
    ignore_layers: list[str] = field(default_factory=list)


_PRE_DEFINED_IGNORE_LAYERS: list[PreDefinedIgnoreLayers] = []


def register_ignore_layers(
    matchers: list[Callable[[Any], bool]], ignore_layers: list[str | Callable[[torch.nn.Module], str | list[str]]]
):
    rule = PreDefinedIgnoreLayers(matchers, ignore_layers)
    _PRE_DEFINED_IGNORE_LAYERS.append(rule)


# longcat
register_ignore_layers(
    matchers=[
        ArchitectureMatcher(r"Longcat", mode="in"),
    ],
    ignore_layers=[
        "classifier",  # transforms directly call the weights of this layer
    ],
)


def get_glm_flash_ignore_layers(model) -> list[str]:
    num_dense_layer = 1
    if hasattr(model, "config") and hasattr(model.config, "first_k_dense_replace"):
        num_dense_layer = model.config.first_k_dense_replace
    ignore_layers = [f"layers.{i}.mlp" for i in range(num_dense_layer)]
    return ignore_layers


# glmflash
register_ignore_layers(
    matchers=[
        ArchitectureMatcher(r"Glm4MoeLite", mode="in"),
    ],
    ignore_layers=[
        get_glm_flash_ignore_layers,  # vllm issue
    ],
)

# glm5
register_ignore_layers(
    matchers=[
        ModelTypeMatcher(r"glm_moe_dsa", mode="full"),
    ],
    ignore_layers=[get_glm_flash_ignore_layers, "weights_proj"],  # vllm issue
)

# step3p5
register_ignore_layers(
    matchers=[
        ModelTypeMatcher(r"step3p5", mode="full"),
    ],
    ignore_layers=[
        "g_proj",  # shape issue [96, 4096], 96 is not divisible by 64
        "moe.gate",
        "eh_proj",  # MTP layer
        "shared_head",  # MTP layer
        "layers.45",  # MTP layer, requiring g_idx in vLLM, skip it
    ],
)

register_ignore_layers(
    matchers=[
        ModelTypeMatcher(r"kimi_k25", mode="full"),
    ],
    ignore_layers=[
        "vision_tower",
        "mm_projector",
    ],
)


def get_bagel_ignore_layers(model) -> list[str]:
    """Keep BAGEL generation-path modules in FP16.

    BAGEL uses `*_moe_gen` modules for the image-generation path. Quantizing
    them causes quality to collapse during the iterative denoising loop.
    The shared attention projections are also highly sensitive, and preserving
    the top 4 transformer blocks in FP16 gave acceptable image quality in
    validation runs.
    """
    top_fp16_layers = 0

    ignore_layers = [
        "moe_gen",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ]

    num_layers = 0
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        num_layers = len(getattr(model.language_model.model, "layers", []))

    if num_layers > 0:
        for layer_idx in range(max(0, num_layers - top_fp16_layers), num_layers):
            ignore_layers.append(f"language_model.model.layers.{layer_idx}")

    return ignore_layers


register_ignore_layers(
    matchers=[
        ModelTypeMatcher(r"bagel", mode="full"),
    ],
    ignore_layers=[
        get_bagel_ignore_layers,
    ],
)


def get_predefined_ignore_layers(model: torch.nn.Module) -> list[str]:
    layers = []
    for rule in _PRE_DEFINED_IGNORE_LAYERS:
        if all(m(model) for m in rule.matchers):
            for ignore_layer in rule.ignore_layers:
                if isinstance(ignore_layer, str):
                    layers.append(ignore_layer)
                else:
                    res = ignore_layer(model)
                    if isinstance(res, str):
                        layers.append(res)
                    else:
                        layers.extend(res)
            break
    config = getattr(model, "config", None)
    if not layers and is_moe_model_via_config(config):
        for name, _ in model.named_modules():
            if name.endswith(".gate"):
                layers.append(name)

    return list(dict.fromkeys(layers))


def _attach_gemma4_rotary_emb(model):
    """Attach ``_rotary_emb`` to each Gemma4 decoder layer.

    For transformers >= 5.6 the per-layer forward patch is unnecessary, but
    ``block_forward`` still needs access to ``rotary_emb`` (which lives on the
    parent ``Gemma4TextModel``) to recompute ``position_embeddings`` when the
    cached version from block 0 has the wrong dimension.
    """
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel
    except ImportError:
        return

    text_model = None
    for _, submodule in model.named_modules():
        if isinstance(submodule, Gemma4TextModel):
            text_model = submodule
            break

    if text_model is None:
        return

    # Create a single shared dict to propagate KV state between anchor/sharer layers.
    # Gemma4TextModel.forward in newer transformers uses the same pattern.
    shared_kv_states_global = {}

    for layer in text_model.layers:
        # Store in a plain list to prevent nn.Module from registering these
        # as child submodules (which would cause meta-tensor errors during .to(device)).
        object.__setattr__(layer, "_rotary_emb_ref", [text_model.rotary_emb])
        object.__setattr__(layer, "_shared_kv_states_global_ref", shared_kv_states_global)
        object.__setattr__(layer, "_autoround_special_replay", "gemma4")
        object.__setattr__(layer, "_gemma4_config_ref", text_model.config)


def _attach_gemma4_unified_rotary_emb(model):
    """Attach ``_rotary_emb`` to each Gemma4 decoder layer.

    For transformers >= 5.6 the per-layer forward patch is unnecessary, but
    ``block_forward`` still needs access to ``rotary_emb`` (which lives on the
    parent ``Gemma4TextModel``) to recompute ``position_embeddings`` when the
    cached version from block 0 has the wrong dimension.
    """
    try:
        from transformers.models.gemma4_unified import Gemma4UnifiedTextModel
    except ImportError:
        return

    text_model = None
    for _, submodule in model.named_modules():
        if isinstance(submodule, Gemma4UnifiedTextModel):
            text_model = submodule
            break

    if text_model is None:
        return

    # Create a single shared dict to propagate KV state between anchor/sharer layers.
    # Gemma4TextModel.forward in newer transformers uses the same pattern.
    shared_kv_states_global = {}

    for layer in text_model.layers:
        # Store in a plain list to prevent nn.Module from registering these
        # as child submodules (which would cause meta-tensor errors during .to(device)).
        object.__setattr__(layer, "_rotary_emb_ref", [text_model.rotary_emb])
        object.__setattr__(layer, "_shared_kv_states_global_ref", shared_kv_states_global)
        object.__setattr__(layer, "_autoround_special_replay", "gemma4")
        object.__setattr__(layer, "_gemma4_config_ref", text_model.config)


def load_next_step_diffusion(pretrained_model_name_or_path, device_str):
    try:
        from models.gen_pipeline import NextStepPipeline  # pylint: disable=E0401
    except ImportError:
        raise ImportError(
            "NextStepPipeline module not found. "
            + "Please navigate to the model file path and add it to your PYTHONPATH."
        )
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, local_files_only=True, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(pretrained_model_name_or_path, local_files_only=True, trust_remote_code=True)
    # The model is loaded onto the device because more than one block requires input data.
    pipe = NextStepPipeline(tokenizer=tokenizer, model=model).to(device=device_str, dtype=torch.bfloat16)

    def _nextstep_pipeline_fn(pipe, prompts, guidance_scale=7.5, num_inference_steps=28, generator=None, **kwargs):
        """Default pipeline_fn for NextStep models.

        Maps standard :class:`DiffusionCompressor` parameters to NextStep's
        ``generate_image`` API.  Pass a custom ``pipeline_fn`` to
        :class:`DiffusionCompressor` to override defaults or supply
        model-specific kwargs (e.g. ``hw``, ``positive_prompt``,
        ``cfg_schedule``, ``timesteps_shift``).
        """
        for prompt in (prompts if isinstance(prompts, list) else [prompts]):
            pipe.generate_image(
                prompt,
                cfg=guidance_scale,
                num_sampling_steps=num_inference_steps,
                **kwargs,
            )

    pipe._autoround_pipeline_fn = _nextstep_pipeline_fn
    return pipe, model


_PRE_DEFINED_FIXED_ATTR = {"gemma4_unified": {"has_variable_block_shape": True}}


def get_predefined_fixed_attr(model: torch.nn.Module) -> dict | None:
    """Return fixed compressor attributes for models that need special caching.

    For Gemma4 with transformers >= 5.6, each decoder block must cache its own
    inputs because sliding vs full-attention layers require different
    position_embeddings. Returns ``None`` for older transformers, which instead
    rely on the per-layer forward patch applied in ``_handle_special_model``.
    """
    import transformers
    from packaging import version

    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "model_type"):
        return None
    attrs = _PRE_DEFINED_FIXED_ATTR.get(config.model_type)
    return attrs
