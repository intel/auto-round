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


SPECIAL_MULTIMODAL_BLOCK["glm_image"] = _get_glm_image_multimodal_block
