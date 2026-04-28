# Copyright (c) 2026 Intel Corporation
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
#
# This file contains AWQ mapping definitions and registry structure adapted from
# llm-compressor (https://github.com/vllm-project/llm-compressor), originally
# developed by Neural Magic, Inc. and licensed under the Apache License 2.0.
# The mapping patterns, model-class registry, and AWQMapping/ResolvedMapping
# data structures are aligned with llm-compressor's AWQ modifier so that
# auto-round AWQ produces models compatible with vllm's AWQ inference kernels.
# Reference: llmcompressor/modifiers/awq/mappings.py

"""AWQ layer mapping resolution

Defines the relationship between *smooth layers* (whose output channels are
inversely scaled) and *balance layers* (whose input channels are scaled up) for
the AWQ smoothing algorithm.

Mappings are resolved in order of priority:
  1. User-provided explicit mappings (``AWQConfig.mappings``).
  2. Model-class-name registry (``AWQ_MAPPING_REGISTRY``), matching
     llm-compressor's(vLLM) AWQ architecture support.
  3. ``default_mappings`` fallback for unknown Llama-like architectures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

from auto_round.logger import logger

# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class AWQMapping:
    """Declarative mapping: smooth_layer regex → list of balance_layer regexes.
    Aligned with ``llmcompressor.modifiers.awq.mappings.AWQMapping``.
    """

    smooth_layer: str
    balance_layers: list[str]


@dataclass
class ResolvedMapping:
    """A fully resolved AWQ mapping with concrete module references."""

    smooth_name: str
    smooth_layer: torch.nn.Module
    balance_names: list[str]
    balance_layers: list[torch.nn.Module]
    parent_name: str
    parent: torch.nn.Module


# ── Mapping definitions ─────────────────────────
# Reference: vllm-project/llm-compressor src/llmcompressor/modifiers/awq/mappings.py

default_mappings = [
    AWQMapping(r"input_layernorm$", [r"q_proj$", r"k_proj$", r"v_proj$"]),
    AWQMapping(r"v_proj$", [r"o_proj$"]),
    AWQMapping(r"post_attention_layernorm$", [r"gate_proj$", r"up_proj$"]),
    AWQMapping(r"up_proj$", [r"down_proj$"]),
]

_gemma_mappings = [
    AWQMapping(r"input_layernorm$", [r"q_proj$", r"k_proj$", r"v_proj$"]),
    AWQMapping(r"v_proj$", [r"o_proj$"]),
    AWQMapping(r"pre_feedforward_layernorm$", [r"gate_proj$", r"up_proj$"]),
    AWQMapping(r"up_proj$", [r"down_proj$"]),
]

# Cohere: MLP runs in parallel with attention; both fed by input_layernorm.
_cohere_mappings = [
    AWQMapping(
        r"input_layernorm$",
        [r"self_attn\.q_proj$", r"self_attn\.k_proj$", r"self_attn\.v_proj$", r"mlp\.gate_proj$", r"mlp\.up_proj$"],
    ),
    AWQMapping(r"v_proj$", [r"o_proj$"]),
    AWQMapping(r"up_proj$", [r"down_proj$"]),
]

# Phi: fused qkv_proj and gate_up_proj layers.
_phi_mappings = [
    AWQMapping(r"input_layernorm$", [r"qkv_proj$"]),
    AWQMapping(r"qkv_proj$", [r"o_proj$"]),
    AWQMapping(r"post_attention_layernorm$", [r"gate_up_proj$"]),
    AWQMapping(r"gate_up_proj$", [r"down_proj$"]),
]

# Bloom: different naming convention.
# Note: query_key_value → dense mapping, see
# https://github.com/mit-han-lab/llm-awq/issues/2#issuecomment-1606297469
_bloom_mappings = [
    AWQMapping(r"input_layernorm$", [r"query_key_value$"]),
    AWQMapping(r"post_attention_layernorm$", [r"dense_h_to_4h$"]),
    AWQMapping(r"gelu_impl$", [r"dense_4h_to_h$"]),
]

# DeepSeek V2/V3: multi-head latent attention with compressed KV.
_deepseek_mappings = [
    AWQMapping(r"input_layernorm$", [r"(q|q_a)_proj$", r"kv_a_proj_with_mqa$"]),
    AWQMapping(r"q_a_layernorm$", [r"q_b_proj$"]),
    AWQMapping(r"kv_a_layernorm$", [r"kv_b_proj$"]),
    AWQMapping(r"post_attention_layernorm$", [r"gate_proj$", r"up_proj$"]),
    AWQMapping(r"up_proj$", [r"down_proj$"]),
]

# MoE default: expert-parallel gate/up projections.
_moe_default_mappings = [
    AWQMapping(r"input_layernorm$", [r"q_proj$", r"k_proj$", r"v_proj$"]),
    AWQMapping(r"v_proj$", [r"o_proj$"]),
    AWQMapping(
        r"post_attention_layernorm$",
        [r"mlp\.experts\.\d+\.gate_proj$", r"mlp\.experts\.\d+\.up_proj$"],
    ),
    AWQMapping(r"up_proj$", [r"down_proj$"]),
]

# Exaone4 / Olmo3: only v_proj→o_proj and up_proj→down_proj smoothing.
_exaone4_mappings = [
    AWQMapping(r"v_proj$", [r"o_proj$"]),
    AWQMapping(r"up_proj$", [r"down_proj$"]),
]

# AFMOE: dual normalization — pre_mlp_layernorm feeds MLP, attention has gate_proj.
_afmoe_mappings = [
    AWQMapping(
        r"input_layernorm$",
        [r"self_attn\.q_proj$", r"self_attn\.k_proj$", r"self_attn\.v_proj$", r"self_attn\.gate_proj$"],
    ),
    AWQMapping(r"v_proj$", [r"o_proj$"]),
    AWQMapping(
        r"pre_mlp_layernorm$",
        [r"mlp\..*gate_proj$", r"mlp\..*up_proj$"],
    ),
    AWQMapping(r"up_proj$", [r"down_proj$"]),
]

# ── Model class name → mappings registry ──────────────────────────────────────
# Aligned with llm-compressor AWQ_MAPPING_REGISTRY (llmcompressor v0.10.0).
# Models not in this registry fall back to default_mappings.

AWQ_MAPPING_REGISTRY: dict[str, list[AWQMapping]] = {
    # AFMOE
    "AfmoeForCausalLM": _afmoe_mappings,
    # Bloom
    "BloomForCausalLM": _bloom_mappings,
    # Cohere / Command-R
    "CohereForCausalLM": _cohere_mappings,
    "Cohere2ForCausalLM": _cohere_mappings,
    "Cohere2VisionForConditionalGeneration": _cohere_mappings,
    # DeepSeek V2/V3
    "DeepseekV3ForCausalLM": _deepseek_mappings,
    # Exaone4 / Olmo3
    "Exaone4ForCausalLM": _exaone4_mappings,
    # Gemma 2/3
    "Gemma2ForCausalLM": _gemma_mappings,
    "Gemma3ForCausalLM": _gemma_mappings,
    "Gemma3ForConditionalGeneration": _gemma_mappings,
    # Llama
    "LlamaForCausalLM": default_mappings,
    "Llama4ForConditionalGeneration": default_mappings,
    # Mistral
    "MistralForCausalLM": default_mappings,
    "Mistral3ForConditionalGeneration": default_mappings,
    # Olmo3 (same as Exaone4)
    "Olmo3ForCausalLM": _exaone4_mappings,
    # Phi
    "Phi3ForCausalLM": _phi_mappings,
    "Phi3VForCausalLM": _phi_mappings,
    # Qwen
    "Qwen2ForCausalLM": default_mappings,
    "Qwen2_5OmniThinkerForConditionalGeneration": default_mappings,
    "Qwen3ForCausalLM": default_mappings,
    # Qwen MoE
    "Qwen2MoeForCausalLM": _moe_default_mappings,
    "Qwen3MoeForCausalLM": _moe_default_mappings,
    # Other models using default mappings
    "Glm4MoeForCausalLM": default_mappings,
    "SeedOssForCausalLM": default_mappings,
    "Ernie4_5_MoeForCausalLM": default_mappings,
}


# ── Helper functions ──────────────────────────────────────────────────────────


def _get_module(model: torch.nn.Module, name: str) -> torch.nn.Module | None:
    """Safely retrieve a sub-module by dotted name."""
    try:
        parts = name.split(".")
        m = model
        for p in parts:
            m = getattr(m, p)
        return m
    except AttributeError:
        return None


def _find_parent(model: torch.nn.Module, names: list[str]) -> tuple[str, torch.nn.Module]:
    """Find the lowest common ancestor module for a list of module names."""
    if not names:
        return "", model

    parts_list = [n.split(".") for n in names]
    common = []
    for level_parts in zip(*parts_list):
        if len(set(level_parts)) == 1:
            common.append(level_parts[0])
        else:
            break

    ancestor_name = ".".join(common)
    if ancestor_name:
        ancestor = _get_module(model, ancestor_name)
    else:
        ancestor = model
    return ancestor_name, ancestor


def _extract_block_prefix(name: str) -> str | None:
    """Extract the transformer block prefix from a module name.

    E.g., "model.layers.5.input_layernorm" → "model.layers.5"
    """
    match = re.match(r"(.*\.\d+)\.", name)
    return match.group(1) if match else None


def _get_model_class_name(model: torch.nn.Module) -> str:
    """Get the model class name, handling nested model wrappers."""
    return type(model).__name__


def _get_mappings_for_model(model: torch.nn.Module) -> list[AWQMapping]:
    """Look up mappings for a model from the registry, with default fallback."""
    cls_name = _get_model_class_name(model)

    if cls_name in AWQ_MAPPING_REGISTRY:
        logger.info(f"Using registered AWQ mappings for {cls_name}.")
        return AWQ_MAPPING_REGISTRY[cls_name]

    logger.info(
        f"Architecture '{cls_name}' not found in AWQ mapping registry. " f"Using default mappings (Llama-like)."
    )
    return default_mappings


# ── Public API ────────────────────────────────────────────────────────────────


def resolve_mappings(
    model: torch.nn.Module,
    user_mappings: list[dict] | None = None,
) -> list[ResolvedMapping]:
    """Resolve AWQ mappings for the given model.

    Resolution order:
      1. ``user_mappings`` — explicit dicts with ``smooth_layer`` /
         ``balance_layers`` regex keys.
      2. ``AWQ_MAPPING_REGISTRY`` — model-class-name lookup
      3. ``default_mappings`` — Llama-like fallback.

    Returns:
        List of ``ResolvedMapping`` objects ready for AWQ grid search.
    """
    if user_mappings is not None:
        mapping_defs = [AWQMapping(m["smooth_layer"], m["balance_layers"]) for m in user_mappings]
    else:
        mapping_defs = _get_mappings_for_model(model)

    return _resolve_mapping_defs(model, mapping_defs)


def _resolve_mapping_defs(
    model: torch.nn.Module,
    mapping_defs: list[AWQMapping],
) -> list[ResolvedMapping]:
    """Resolve a list of AWQMapping definitions against the model."""
    resolved = []
    all_names = [n for n, _ in model.named_modules()]

    # Group modules by block prefix
    block_modules: dict[str, list[str]] = {}
    for name in all_names:
        prefix = _extract_block_prefix(name)
        if prefix is not None:
            block_modules.setdefault(prefix, []).append(name)

    if not block_modules:
        logger.warning(
            "AWQ found no repeating block structure in the model. "
            "Provide explicit mappings via AWQConfig(mappings=[...])."
        )
        return resolved

    matched_count = 0

    for prefix, names_in_block in block_modules.items():
        for mapping_def in mapping_defs:
            # Find smooth layer(s) in this block
            smooth_matches = [n for n in names_in_block if re.search(mapping_def.smooth_layer, n)]
            if not smooth_matches:
                continue

            for smooth_name in smooth_matches:
                smooth_layer = _get_module(model, smooth_name)
                if smooth_layer is None:
                    continue
                if not hasattr(smooth_layer, "weight"):
                    continue

                # Find balance layers in the same block
                balance_names = []
                balance_layers = []
                for bp in mapping_def.balance_layers:
                    for n in names_in_block:
                        if re.search(bp, n):
                            m = _get_module(model, n)
                            if m is not None and isinstance(m, torch.nn.Linear):
                                balance_names.append(n)
                                balance_layers.append(m)

                if not balance_layers:
                    continue

                # Verify dimensional compatibility (filters out GQA
                # mismatches for v_proj → o_proj automatically)
                smooth_dim = smooth_layer.weight.shape[0]
                compatible = all(bl.in_features == smooth_dim for bl in balance_layers)
                if not compatible:
                    logger.warning_once(
                        f"Skipping AWQ for '{smooth_name}': incompatible "
                        f"balance layers (smooth_dim={smooth_dim}, "
                        f"balance in_features="
                        f"{[bl.in_features for bl in balance_layers]})"
                    )
                    continue

                parent_name, parent = _find_parent(model, balance_names)
                resolved.append(
                    ResolvedMapping(
                        smooth_name=smooth_name,
                        smooth_layer=smooth_layer,
                        balance_names=balance_names,
                        balance_layers=balance_layers,
                        parent_name=parent_name,
                        parent=parent,
                    )
                )
                matched_count += 1

    if matched_count == 0:
        logger.warning(
            "AWQ resolved 0 mappings. The model architecture may not match "
            "any known pattern. Provide explicit mappings via "
            "AWQConfig(mappings=[...])."
        )
    else:
        first_prefix = next(iter(block_modules))
        n_blocks = len(block_modules)
        mappings_per_block = sum(1 for r in resolved if r.smooth_name.startswith(first_prefix))
        logger.info(
            f"AWQ resolved {matched_count} smooth-balance mappings "
            f"({mappings_per_block} per block × {n_blocks} blocks)."
        )

    return resolved


# ── Model compatibility diagnostics ───────────────────────────────────────────


def check_model_compatibility(
    model: torch.nn.Module,
    user_mappings: list[dict] | None = None,
) -> dict:
    """Check AWQ compatibility and return a diagnostic report.

    Returns a dict with:
        - ``compatible`` (bool): True if at least one mapping was resolved.
        - ``n_mappings`` (int): Number of resolved mappings.
        - ``mappings_per_block`` (int): Mappings in the first block.
        - ``n_blocks`` (int): Number of transformer blocks.
        - ``model_class`` (str): Model class name.
        - ``in_registry`` (bool): Whether model class is in AWQ_MAPPING_REGISTRY.
        - ``warnings`` (list[str]): Any compatibility warnings.
    """
    warnings_list = []
    cls_name = _get_model_class_name(model)
    in_registry = cls_name in AWQ_MAPPING_REGISTRY

    if not in_registry and user_mappings is None:
        warnings_list.append(
            f"Model class '{cls_name}' is not in AWQ_MAPPING_REGISTRY. "
            f"Using default Llama-like mappings. If quantization quality is "
            f"poor, provide explicit mappings via AWQConfig(mappings=[...])."
        )

    resolved = resolve_mappings(model, user_mappings)

    all_prefixes = set()
    for r in resolved:
        prefix = _extract_block_prefix(r.smooth_name)
        if prefix:
            all_prefixes.add(prefix)

    n_blocks = len(all_prefixes)
    mappings_per_block = 0
    if n_blocks > 0 and resolved:
        first_prefix = min(all_prefixes)
        mappings_per_block = sum(1 for r in resolved if r.smooth_name.startswith(first_prefix))

    if not resolved:
        warnings_list.append(
            "No AWQ mappings could be resolved. The model architecture may " "not be supported for auto-detection."
        )

    return {
        "compatible": len(resolved) > 0,
        "n_mappings": len(resolved),
        "mappings_per_block": mappings_per_block,
        "n_blocks": n_blocks,
        "model_class": cls_name,
        "in_registry": in_registry,
        "warnings": warnings_list,
    }
