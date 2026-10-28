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
import torch

# Maps architecture class names to virtual model_type keys.
# Used when config.model_type doesn't uniquely identify the model (e.g. MiMo-Audio).
from auto_round.utils.model import ARCHITECTURE_MODEL_TYPE_MAP, resolve_model_type  # noqa: E402,F401

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
