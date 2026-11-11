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
from auto_round.special_model_handler.core import SPECIAL_MULTIMODAL_BLOCK, ModelTypeMatcher
from auto_round.special_model_handler.ignore_layers import register_ignore_layers
from auto_round.utils import logger


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


def get_bagel_ignore_layers(model) -> list[str]:
    """Keep BAGEL generation-path modules in BF16.

    BAGEL uses `*_moe_gen` modules for the image-generation path. Quantizing
    them can reduce quality during the iterative denoising loop. Users can
    explicitly opt in with ``AR_QUANTIZE_BAGEL_MOE_GEN=1``.

    The standard Qwen2 attention and MLP projections are deliberately not
    ignored: they form BAGEL's normal text path and should follow AutoRound's
    usual transformer-layer policy.
    """
    from auto_round import envs

    ignore_layers = []
    if not envs.AR_QUANTIZE_BAGEL_MOE_GEN:
        ignore_layers.append("moe_gen")
    else:
        logger.warning(
            "AR_QUANTIZE_BAGEL_MOE_GEN is enabled. Quantizing BAGEL's image-generation experts may reduce image "
            "quality."
        )
    return ignore_layers


register_ignore_layers(
    matchers=[
        ModelTypeMatcher(r"bagel", mode="full"),
    ],
    ignore_layers=[
        get_bagel_ignore_layers,
    ],
)


SPECIAL_MULTIMODAL_BLOCK["bagel"] = _get_bagel_multimodal_block
