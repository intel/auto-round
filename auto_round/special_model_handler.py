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
]

NOT_SUPPORT_ONLY_TEXT_MODELS = ["mllama", "mistral3_2"]

SPECIAL_SHARED_CACHE_KEYS = {
    "Gemma3ForConditionalGeneration": ("position_embeddings_global", "position_embeddings_local")
}
SPECIAL_SHARED_CACHE_KEYS["MiniMaxText01ForCausalLM"] = ("slope_rate",)
MISTRAL_3_2_MODELS = ["Mistral-Small-3.2", "Magistral-Small", "Devstral-Small"]


def _handle_special_model(model):
    if hasattr(model, "config") and model.config.model_type == "deepseek_vl_v2":
        from functools import partial

        model.forward = partial(_deepseek_vl2_forward, model)
    if hasattr(model, "config") and model.config.model_type == "qwen2_5_omni":
        from functools import partial

        model.forward = partial(_qwen2_5_omni_forward, model)
    if hasattr(model, "config") and model.config.model_type == "qwen3_omni_moe":
        from functools import partial

        model.forward = partial(_qwen3_omni_moe_forward, model)
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
    - talker.model.layers (talker decoder layers)
    - Optionally: visual encoder blocks, audio encoder layers
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

    # Talker model layers (if available)
    if hasattr(model, "talker") and hasattr(model.talker, "model") and hasattr(model.talker.model, "layers"):
        block_names.append([f"talker.model.layers.{i}" for i in range(len(model.talker.model.layers))])

    return block_names


def _get_qwen3_omni_moe_multimodal_block(model, quant_vision=False):
    """Get block names for Qwen3-Omni MoE model.

    Qwen3-Omni has the following structure:
    - thinker: Contains audio_tower, visual, model (text decoder)
    - talker: Contains model (talker decoder), code_predictor
    - code2wav: Audio decoder

    For quantization, we focus on:
    - thinker.model.layers (text decoder layers) - main LLM layers
    - talker.model.layers (talker decoder layers)
    - Optionally: visual encoder blocks, audio encoder layers
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

    # Talker model layers (if available)
    if hasattr(model, "talker") and hasattr(model.talker, "model") and hasattr(model.talker.model, "layers"):
        block_names.append([f"talker.model.layers.{i}" for i in range(len(model.talker.model.layers))])

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


def _get_bagel_multimodal_block(model, quant_vision=False):
    """Get block names for BAGEL not (Mixture of Transformers) model.

    BAGEL model structure:
    - language_model.model.layers: Qwen2-based LLM with not dual paths
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


def get_bagel_ignore_layers(model) -> list[str]:
    """Keep BAGEL generation-path modules in FP16.

    BAGEL uses `*_moe_gen` modules for the image-generation path. Quantizing
    them causes quality to collapse during the iterative denoising loop.
    The shared attention projections are also highly sensitive.
    """
    ignore_layers = [
        "moe_gen",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ]

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
