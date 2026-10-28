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
from functools import partial

import torch

from auto_round.export.formats import OutputFormat
from auto_round.modeling.fused_moe.replace_modules import apply_replacements, release_original_module_
from auto_round.special_model_handler.gemma4 import (
    _attach_gemma4_rotary_emb,
    _attach_gemma4_unified_rotary_emb,
    _normalize_gemma4_per_layer_input,
    _patch_gemma4_model,
    _prepare_gemma4_replay_inputs,
)
from auto_round.utils import logger
from auto_round.utils.model import resolve_model_type

# Forward helpers are imported lazily inside _handle_special_model only when needed
# in the original file via local imports of model_type branches — keep same structure
# by importing the forward callables at module level (identical binding names).
from auto_round.special_model_handler.deepseek_vl import _deepseek_vl2_forward
from auto_round.special_model_handler.mimo_audio import _mimo_audio_forward
from auto_round.special_model_handler.qwen3_tts import _qwen3_tts_forward
from auto_round.special_model_handler.qwen_omni import _qwen2_5_omni_forward, _qwen3_omni_moe_forward

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
    model, formats: list[OutputFormat] | None = None, trust_remote_code: bool = True, cleanup_original: bool = True
):
    gguf_export = formats is not None and any(format_.is_gguf() for format_ in formats)
    model = apply_replacements(model, gguf_export=gguf_export)

    if cleanup_original:
        release_original_module_(model)

    return model
