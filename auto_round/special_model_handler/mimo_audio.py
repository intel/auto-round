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


SPECIAL_MULTIMODAL_BLOCK["mimo_audio"] = _get_mimo_audio_multimodal_block
