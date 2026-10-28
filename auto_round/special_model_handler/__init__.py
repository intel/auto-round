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
"""Special-model handlers split by model/arch family.

Public import path is unchanged: ``from auto_round.special_model_handler import ...``.
"""

from auto_round.special_model_handler.constants import (
    MISTRAL_3_2_MODELS,
    NOT_SUPPORT_ONLY_TEXT_MODELS,
    SPECIAL_MULTIMODAL_BLOCK,
    SPECIAL_SHARED_CACHE_KEYS,
    SUPPORT_ONLY_TEXT_MODELS,
    check_mllm_only_support_bs1,
    mllms_with_limited_bs,
)
from auto_round.special_model_handler.matchers import (
    ArchitectureMatcher,
    ModelNameMatcher,
    ModelTypeMatcher,
)
from auto_round.special_model_handler.ignore_layers import (
    PreDefinedIgnoreLayers,
    PreDefinedTorchCompileOff,
    _PRE_DEFINED_FIXED_ATTR,
    _PRE_DEFINED_IGNORE_LAYERS,
    _PRE_DEFINED_LAYER_CONFIG,
    _PRE_DEFINED_TORCH_COMPILE_OFF,
    get_glm_flash_ignore_layers,
    get_predefined_fixed_attr,
    get_predefined_ignore_layers,
    get_predefined_layer_config,
    get_torch_compile_off_reason,
    register_ignore_layers,
    register_torch_compile_off,
)
from auto_round.special_model_handler.gemma4 import (
    _attach_gemma4_rotary_emb,
    _attach_gemma4_unified_rotary_emb,
    _get_gemma4_rotary_emb,
    _get_gemma4_shared_kv_states_global,
    _normalize_gemma4_per_layer_input,
    _patch_gemma4_model,
    _prepare_gemma4_replay_inputs,
    _rebuild_gemma4_attention_mask,
)
from auto_round.special_model_handler.deepseek_vl import (
    _deepseek_vl2_forward,
    _get_deepseek_vl2_multimodal_block,
)
from auto_round.special_model_handler.qwen_omni import (
    _get_qwen2_5_omni_multimodal_block,
    _get_qwen3_omni_moe_multimodal_block,
    _qwen2_5_omni_forward,
    _qwen3_omni_moe_forward,
)
from auto_round.special_model_handler.glm_image import _get_glm_image_multimodal_block
from auto_round.special_model_handler.mimo_audio import (
    _get_mimo_audio_multimodal_block,
    _mimo_audio_forward,
)
from auto_round.special_model_handler.qwen3_tts import (
    _get_qwen3_tts_multimodal_block,
    _qwen3_tts_forward,
)
from auto_round.special_model_handler.bagel import (
    _get_bagel_multimodal_block,
    get_bagel_ignore_layers,
)
from auto_round.special_model_handler.next_step import load_next_step_diffusion
from auto_round.special_model_handler.cosmos3 import (
    _align_cosmos3_quant_config_to_vllm_omni,
    _bypass_cosmos3_safety_checker,
    _cosmos3_forward_mode,
    _cosmos3_forward_state,
    _get_cosmos3_multimodal_block,
    _set_quantizable_bits,
    _wrap_cosmos3_layer_dual_mode,
    load_cosmos3_diffusion,
)
from auto_round.special_model_handler.dispatch import (
    _handle_special_model,
    prepare_special_model_block_inputs,
    update_module,
)

# Ensure model modules run their SPECIAL_MULTIMODAL_BLOCK registrations.
# (Already imported above; bagel/ignore also register ignore rules.)

# Re-export resolve_model_type for callers that historically imported it from here.
from auto_round.utils.model import resolve_model_type  # noqa: E402

__all__ = [
    "ARCHITECTURE_MODEL_TYPE_MAP",  # not always imported; keep symbols tests touch
    "MISTRAL_3_2_MODELS",
    "NOT_SUPPORT_ONLY_TEXT_MODELS",
    "SPECIAL_MULTIMODAL_BLOCK",
    "SPECIAL_SHARED_CACHE_KEYS",
    "SUPPORT_ONLY_TEXT_MODELS",
    "ArchitectureMatcher",
    "ModelNameMatcher",
    "ModelTypeMatcher",
    "PreDefinedIgnoreLayers",
    "PreDefinedTorchCompileOff",
    "_PRE_DEFINED_FIXED_ATTR",
    "_PRE_DEFINED_IGNORE_LAYERS",
    "_PRE_DEFINED_LAYER_CONFIG",
    "_PRE_DEFINED_TORCH_COMPILE_OFF",
    "_align_cosmos3_quant_config_to_vllm_omni",
    "_attach_gemma4_rotary_emb",
    "_attach_gemma4_unified_rotary_emb",
    "_bypass_cosmos3_safety_checker",
    "_cosmos3_forward_mode",
    "_cosmos3_forward_state",
    "_deepseek_vl2_forward",
    "_get_bagel_multimodal_block",
    "_get_cosmos3_multimodal_block",
    "_get_deepseek_vl2_multimodal_block",
    "_get_gemma4_rotary_emb",
    "_get_gemma4_shared_kv_states_global",
    "_get_glm_image_multimodal_block",
    "_get_mimo_audio_multimodal_block",
    "_get_qwen2_5_omni_multimodal_block",
    "_get_qwen3_omni_moe_multimodal_block",
    "_get_qwen3_tts_multimodal_block",
    "_handle_special_model",
    "_mimo_audio_forward",
    "_normalize_gemma4_per_layer_input",
    "_patch_gemma4_model",
    "_prepare_gemma4_replay_inputs",
    "_qwen2_5_omni_forward",
    "_qwen3_omni_moe_forward",
    "_qwen3_tts_forward",
    "_rebuild_gemma4_attention_mask",
    "_set_quantizable_bits",
    "_wrap_cosmos3_layer_dual_mode",
    "check_mllm_only_support_bs1",
    "get_bagel_ignore_layers",
    "get_glm_flash_ignore_layers",
    "get_predefined_fixed_attr",
    "get_predefined_ignore_layers",
    "get_predefined_layer_config",
    "get_torch_compile_off_reason",
    "load_cosmos3_diffusion",
    "load_next_step_diffusion",
    "mllms_with_limited_bs",
    "prepare_special_model_block_inputs",
    "register_ignore_layers",
    "register_torch_compile_off",
    "resolve_model_type",
    "update_module",
]

# Import ARCHITECTURE_MODEL_TYPE_MAP for completeness (was available via side-import in original).
from auto_round.utils.model import ARCHITECTURE_MODEL_TYPE_MAP  # noqa: E402,F401
