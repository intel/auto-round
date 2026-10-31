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
"""Shared constants, matchers, and cross-cutting dispatch for special models.

Merged from the previous ``constants`` / ``matchers`` / ``dispatch`` modules
(per maintainer feedback on #2514 to reduce granularity).
"""

import re
from functools import partial

import torch

from auto_round.export.formats import OutputFormat
from auto_round.utils import logger
from auto_round.utils.model import ARCHITECTURE_MODEL_TYPE_MAP, resolve_model_type  # noqa: F401

# ---------------------------------------------------------------------------
# Constants / registries
# ---------------------------------------------------------------------------

# Maps architecture class names to virtual model_type keys.
# Used when config.model_type doesn't uniquely identify the model (e.g. MiMo-Audio).

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

SPECIAL_SHARED_CACHE_KEYS = {
    "Gemma3ForConditionalGeneration": ("position_embeddings_global", "position_embeddings_local")
}
SPECIAL_SHARED_CACHE_KEYS["MiniMaxText01ForCausalLM"] = ("slope_rate",)
SPECIAL_SHARED_CACHE_KEYS["StableAudioDiTModel"] = ("encoder_hidden_states",)
SPECIAL_SHARED_CACHE_KEYS["Gemma4ForConditionalGeneration"] = ("position_ids",)
SPECIAL_SHARED_CACHE_KEYS["WanTransformer3DModel"] = ("rotary_emb",)
MISTRAL_3_2_MODELS = ["Mistral-Small-3.2", "Magistral-Small", "Devstral-Small"]


# Populated by model-family modules at import time (and by __init__).
SPECIAL_MULTIMODAL_BLOCK = {}


def check_mllm_only_support_bs1(model: torch.nn.Module):
    """
    Checks model configuration to determine if it's necessary to limit bs to avoid potential input shape mismatches.
    """
    effective_type = resolve_model_type(model) or ""
    for key in mllms_with_limited_bs:
        if key in effective_type:
            return True
    return False


# ---------------------------------------------------------------------------
# Matchers
# ---------------------------------------------------------------------------


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
            raise ValueError(f"unsupported mode {self.mode}")


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
    """match config.model_type"""

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


# ---------------------------------------------------------------------------
# Cross-cutting dispatch / entrypoints
# ---------------------------------------------------------------------------


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
        from auto_round.special_model_handler.gemma4 import (
            _normalize_gemma4_per_layer_input,
            _prepare_gemma4_replay_inputs,
        )

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


def _handle_special_model(model):
    model_type = resolve_model_type(model)
    if model_type == "deepseek_vl_v2":
        from auto_round.special_model_handler.deepseek_vl import _deepseek_vl2_forward

        model.forward = partial(_deepseek_vl2_forward, model)
    if model_type == "qwen2_5_omni":
        from auto_round.special_model_handler.qwen_omni import _qwen2_5_omni_forward

        model.forward = partial(_qwen2_5_omni_forward, model)
    if model_type == "qwen3_omni_moe":
        from auto_round.special_model_handler.qwen_omni import _qwen3_omni_moe_forward

        model.forward = partial(_qwen3_omni_moe_forward, model)
    if model_type == "qwen3_tts":
        from auto_round.special_model_handler.qwen3_tts import _qwen3_tts_forward

        model.forward = partial(_qwen3_tts_forward, model)
    if model_type == "mimo_audio":
        from auto_round.special_model_handler.mimo_audio import _mimo_audio_forward

        model.forward = partial(_mimo_audio_forward, model)
    if hasattr(model, "config") and (model_type == "gemma4"):
        import transformers
        from packaging import version

        from auto_round.special_model_handler.gemma4 import (
            _attach_gemma4_rotary_emb,
            _patch_gemma4_model,
        )

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
        from auto_round.special_model_handler.gemma4 import _attach_gemma4_unified_rotary_emb

        _attach_gemma4_unified_rotary_emb(model)
    return model


def update_module(
    model, formats: list[OutputFormat] | None = None, trust_remote_code: bool = True, cleanup_original: bool = True
):
    # Resolve through the package namespace so tests can monkeypatch
    # ``special_model_handler.apply_replacements`` (legacy module API).
    from auto_round import special_model_handler as _smh

    gguf_export = formats is not None and any(format_.is_gguf() for format_ in formats)
    model = _smh.apply_replacements(model, gguf_export=gguf_export)

    if cleanup_original:
        _smh.release_original_module_(model)

    return model
