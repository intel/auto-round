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
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch

from auto_round.special_model_handler.matchers import ArchitectureMatcher, ModelTypeMatcher
from auto_round.utils import is_moe_model_via_config, torch_version_at_least

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

# GLM-5.3-Flash family: ``glm_moe_dsa`` (text) and ``glm5_next`` (multimodal).
#
# Both keep their whole DSA indexer out of quantization:
#   * ``{GlmMoeDsa,Glm5NextText}Indexer.forward`` is decorated with ``@torch.no_grad()``
#     while owning three quantizable linears (``wq_b``, ``wk``, ``weights_proj``).
#     Wrapping them makes the same compiled quant function run under both grad-disabled
#     (indexer) and grad-enabled (rest of the block) modes. dynamo keys its cache by
#     *code object* -- shared by every WrapperLinear -- so the AOTAutograd backward then
#     fails with "cannot enter context: <Context ...> is already entered".
#   * transformers marks ``indexer.weights_proj`` in ``_keep_in_fp32_modules``, i.e. it
#     is not even supposed to leave FP32.
# The indexer is tiny anyway (``weights_proj`` outputs ``index_n_heads``), so there is
# no compression to gain here. ``get_glm_flash_ignore_layers`` additionally keeps the
# leading dense MLPs in full precision to work around a vLLM loading issue.
register_ignore_layers(
    matchers=[
        ModelTypeMatcher(r"glm_moe_dsa", mode="full"),
    ],
    ignore_layers=[get_glm_flash_ignore_layers, "weights_proj", "indexer"],  # get_glm_flash_ignore_layers: vllm issue
)

register_ignore_layers(
    matchers=[
        ModelTypeMatcher(r"glm5_next", mode="full"),
    ],
    ignore_layers=[
        get_glm_flash_ignore_layers,  # vllm issue
        "weights_proj",
        "indexer",
        "self_attn",
    ],
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

# qwen4: keep hyper_connection and MoE gate modules in full precision.
register_ignore_layers(
    matchers=[
        ArchitectureMatcher(r"Qwen4", mode="in"),
    ],
    ignore_layers=[
        "hyper_connection",
        "mlp.gate",  # MoE router gate
        "shared_expert",
    ],
)

# Architecture-specific layer_config defaults, keyed by ``config.model_type``.
# Nemotron-H Omni wrappers (Super VL, Bolt) keep their Mamba2 ``out_proj`` scales in
# BF16; every other model, including plain ``nemotron_h``, keeps the caller's scale_dtype.
_PRE_DEFINED_LAYER_CONFIG = {
    "nemotron_h_omni": {".*mixer.out_proj$": {"scale_dtype": torch.bfloat16}},
}


def get_predefined_layer_config(model: torch.nn.Module) -> dict:
    """Return predefined layer_config patterns for the model; user entries take precedence."""
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    return _PRE_DEFINED_LAYER_CONFIG.get(model_type, {})


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
            if name.endswith((".gate", ".shared_expert_gate")):
                layers.append(name)

    return list(dict.fromkeys(layers))


# ---------------------------------------------------------------------------
# Architectures for which ``torch.compile`` is force-disabled during tuning.
# ---------------------------------------------------------------------------
@dataclass
class PreDefinedTorchCompileOff:
    """A rule that force-disables ``torch.compile`` for a family of models."""

    matchers: list[Callable[[Any], bool]]
    reason: str


_PRE_DEFINED_TORCH_COMPILE_OFF: list[PreDefinedTorchCompileOff] = []


def register_torch_compile_off(matchers: list[Callable[[Any], bool]], reason: str) -> None:
    """Register an architecture family for which ``torch.compile`` must stay off.

    Any matcher hitting is enough (``any``, not ``all``) so one rule can cover several
    equivalent identifiers (model_type and architecture name).
    """
    _PRE_DEFINED_TORCH_COMPILE_OFF.append(PreDefinedTorchCompileOff(matchers, reason))


def get_torch_compile_off_reason(model) -> str | None:
    """Return why ``torch.compile`` must stay off for this model, else ``None``.

    Accepts a model or anything exposing ``.config``.
    """
    if model is None:
        return None
    proxy = model if hasattr(model, "config") else SimpleNamespace(config=model)
    for rule in _PRE_DEFINED_TORCH_COMPILE_OFF:
        if any(matcher(proxy) for matcher in rule.matchers):
            return rule.reason
    return None


# DeepSeek V2/V3/V3.2/V4 and the GLM-5.3-Flash family (``glm_moe_dsa`` text /
# ``glm5_next`` multimodal) share a DSA indexer + fused MoE with shape-dependent
# control flow. On torch < 2.14.0 compiling the tuning graph there hits dynamo's
# recompile_limit and provides little benefit, so keep those runs in eager mode.
# From torch 2.14.0 onward these constraints are resolved, so we no longer force
# ``torch.compile`` off and instead honor the default / user-provided setting.
if not torch_version_at_least("2.14.0"):
    register_torch_compile_off(
        matchers=[
            ModelTypeMatcher("deepseek", mode="in"),
            ArchitectureMatcher("Deepseek", mode="in"),
            ArchitectureMatcher("DeepSeek", mode="in"),
        ],
        reason="the DeepSeek architecture is incompatible with torch.compile during tuning on torch < 2.14.0",
    )
    register_torch_compile_off(
        matchers=[
            ModelTypeMatcher(r"glm_moe_dsa", mode="full"),
            ModelTypeMatcher(r"glm5_next", mode="full"),
            ArchitectureMatcher(r"Glm5Next", mode="in"),
        ],
        reason="the GLM-5 architecture is incompatible with torch.compile during tuning on torch < 2.14.0",
    )

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
