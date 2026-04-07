# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""Model architecture mapping for Hadamard rotation.

Design inspired by ``llm-compressor`` SpinQuant:
  https://github.com/vllm-project/llm-compressor/tree/main/src/llmcompressor/modifiers/transform/spinquant

Each :class:`RotationMapping` describes *where* the rotation-relevant modules
live inside a model, using **regex patterns** over fully-qualified module names.
Patterns prefixed with ``re:`` are compiled as regexes; plain strings are
matched as suffixes.  This keeps the mapping declarative and model-agnostic.

New architectures can be supported by calling :func:`register_mapping`
with a new :class:`RotationMapping` – **no code changes** to the rotation
logic are required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from auto_round.utils import logger

__all__ = [
    "RotationMapping",
    "register_mapping",
    "get_mapping",
    "infer_mapping_from_model",
    "MAPPING_REGISTRY",
]


# ---------------------------------------------------------------------------
# Mapping dataclass
# ---------------------------------------------------------------------------

@dataclass
class RotationMapping:
    """Declarative description of a transformer architecture for Hadamard rotation.

    Every field is either a **single** pattern or a **list** of patterns.
    A pattern is:
      * ``"re:<regex>"`` – matched via ``re.search`` against full module name.
      * any other string   – matched as a **suffix** of the full module name.

    Top-level modules (resolved from the *model*):
        embedding, lm_head

    Per-layer modules (resolved from each *decoder layer*):
        attn_q, attn_k, attn_v, attn_o   – attention projections
        mlp_in                             – MLP input projections  (gate / up)
        mlp_out                            – MLP output projections (down)
        attn_input_ln, mlp_input_ln        – layernorms to fuse
        pre_head_ln                        – final norm before head (model-level)

    MoE-specific (optional):
        is_moe                             – whether the MLP is MoE
        moe_expert_mlp_in / moe_expert_mlp_out – patterns for *each* expert
        moe_shared_mlp_in / moe_shared_mlp_out – patterns for shared expert

    Config attribute names (read from ``model.config``):
        num_heads_attr, hidden_size_attr, intermediate_size_attr,
        moe_intermediate_size_attr
        head_dim_override – explicit head dim (skip hidden_size // num_heads)
    """

    # -- top-level --
    embedding: str = "re:.*embed_tokens$"
    lm_head: str = "lm_head"

    # -- layers container (dot-path, not regex) --
    layers_attr: str = "model.layers"

    # -- per-layer: attention --
    attn_input_ln: str = "input_layernorm"
    attn_q: str = "re:.*q_proj$"
    attn_k: str = "re:.*k_proj$"
    attn_v: str = "re:.*v_proj$"
    attn_o: str = "re:.*o_proj$"

    # -- per-layer: MLP (dense) --
    mlp_input_ln: str = "post_attention_layernorm"
    mlp_in: List[str] = field(default_factory=lambda: ["re:.*up_proj$", "re:.*gate_proj$"])
    mlp_out: List[str] = field(default_factory=lambda: ["re:.*down_proj$"])

    # -- final norm (model-level, dot-path) --
    pre_head_ln: str = "model.norm"

    # -- MoE --
    is_moe: bool = False
    moe_expert_mlp_in: List[str] = field(
        default_factory=lambda: ["re:.*experts\\.\\d+\\.up_proj$", "re:.*experts\\.\\d+\\.gate_proj$"]
    )
    moe_expert_mlp_out: List[str] = field(
        default_factory=lambda: ["re:.*experts\\.\\d+\\.down_proj$"]
    )
    moe_shared_mlp_in: List[str] = field(
        default_factory=lambda: ["re:.*shared_expert\\.up_proj$", "re:.*shared_expert\\.gate_proj$",
                                  "re:.*shared_experts\\.up_proj$", "re:.*shared_experts\\.gate_proj$"]
    )
    moe_shared_mlp_out: List[str] = field(
        default_factory=lambda: ["re:.*shared_expert\\.down_proj$",
                                  "re:.*shared_experts\\.down_proj$"]
    )

    # -- head dim --
    attn_head_dim: Optional[int] = None  # override; else hidden_size // num_heads

    # -- config attr names --
    num_heads_attr: str = "num_attention_heads"
    hidden_size_attr: str = "hidden_size"
    intermediate_size_attr: str = "intermediate_size"
    moe_intermediate_size_attr: str = "moe_intermediate_size"


# ---------------------------------------------------------------------------
# Pattern matching helper
# ---------------------------------------------------------------------------

def _match(pattern: str, name: str) -> bool:
    """Check whether *name* matches *pattern*.

    ``"re:<regex>"`` → ``re.search``; otherwise suffix match.
    """
    if pattern.startswith("re:"):
        return re.search(pattern[3:], name) is not None
    return name.endswith(pattern)


def find_modules(model_or_layer, patterns, return_names=False):
    """Yield ``(name, module)`` pairs whose name matches *any* pattern.

    Args:
        model_or_layer: An ``nn.Module`` to search.
        patterns: A single pattern string or a list of patterns.
        return_names: If ``True`` return ``(name, module)``; else just ``module``.
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    for name, mod in model_or_layer.named_modules():
        if any(_match(p, name) for p in patterns):
            if return_names:
                yield name, mod
            else:
                yield mod


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MAPPING_REGISTRY: Dict[str, RotationMapping] = {}


def register_mapping(key: str, mapping: RotationMapping) -> RotationMapping:
    """Register a :class:`RotationMapping` under *key* (model_type or architecture)."""
    MAPPING_REGISTRY[key] = mapping
    return mapping


def get_mapping(key: str) -> RotationMapping:
    """Look up a mapping by *key*; fall back to default if not found."""
    if key in MAPPING_REGISTRY:
        return MAPPING_REGISTRY[key]
    logger.warning(
        f"No rotation mapping registered for '{key}', "
        "falling back to default (LLaMA-like) mapping."
    )
    return RotationMapping()


def infer_mapping_from_model(model) -> RotationMapping:
    """Return the best :class:`RotationMapping` for *model*.

    Tries ``model.__class__.__name__`` first, then ``model.config.model_type``.
    """
    arch = model.__class__.__name__
    if arch in MAPPING_REGISTRY:
        return MAPPING_REGISTRY[arch]

    model_type = getattr(getattr(model, "config", None), "model_type", "")
    if model_type in MAPPING_REGISTRY:
        return MAPPING_REGISTRY[model_type]

    logger.warning(
        f"Unrecognised architecture '{arch}' (model_type='{model_type}'). "
        "Falling back to default (LLaMA-like) mapping."
    )
    return RotationMapping()


# ===================================================================
# Built-in mappings
# ===================================================================

# ---- default (LLaMA-like, covers LLaMA / LLaMA-2 / LLaMA-3 / Mistral / Yi / …) ----
_default = RotationMapping()

register_mapping("llama", _default)
register_mapping("LlamaForCausalLM", _default)
register_mapping("mistral", _default)
register_mapping("MistralForCausalLM", _default)

# ---- Qwen2 / Qwen2.5 (dense – same layout as LLaMA) ----
register_mapping("qwen2", _default)
register_mapping("Qwen2ForCausalLM", _default)
register_mapping("qwen2_5", _default)
register_mapping("Qwen2_5ForCausalLM", _default)

# ---- Qwen3 (dense – same layout as LLaMA) ----
register_mapping("qwen3", _default)
register_mapping("Qwen3ForCausalLM", _default)

# ---- Gemma2 ----
register_mapping("gemma2", _default)
register_mapping("Gemma2ForCausalLM", _default)

# ---- Qwen2-MoE / Qwen2.5-MoE ----
_qwen2_moe = RotationMapping(
    is_moe=True,
    # dense layers on some blocks, MoE on others — use generic patterns
    # that match both  mlp.up_proj  and  mlp.experts.N.up_proj
    mlp_in=["re:.*mlp\\.up_proj$", "re:.*mlp\\.gate_proj$"],
    mlp_out=["re:.*mlp\\.down_proj$"],
    moe_expert_mlp_in=[
        "re:.*experts\\.\\d+\\.up_proj$",
        "re:.*experts\\.\\d+\\.gate_proj$",
    ],
    moe_expert_mlp_out=["re:.*experts\\.\\d+\\.down_proj$"],
    moe_shared_mlp_in=[
        "re:.*shared_expert\\.up_proj$",
        "re:.*shared_expert\\.gate_proj$",
    ],
    moe_shared_mlp_out=["re:.*shared_expert\\.down_proj$"],
    moe_intermediate_size_attr="moe_intermediate_size",
)
register_mapping("qwen2_moe", _qwen2_moe)
register_mapping("Qwen2MoeForCausalLM", _qwen2_moe)

# ---- Qwen3-MoE ----
_qwen3_moe = RotationMapping(
    is_moe=True,
    mlp_in=["re:.*mlp\\.up_proj$", "re:.*mlp\\.gate_proj$"],
    mlp_out=["re:.*mlp\\.down_proj$"],
    moe_expert_mlp_in=[
        "re:.*experts\\.\\d+\\.up_proj$",
        "re:.*experts\\.\\d+\\.gate_proj$",
    ],
    moe_expert_mlp_out=["re:.*experts\\.\\d+\\.down_proj$"],
    moe_shared_mlp_in=[
        "re:.*shared_expert\\.up_proj$",
        "re:.*shared_expert\\.gate_proj$",
    ],
    moe_shared_mlp_out=["re:.*shared_expert\\.down_proj$"],
    moe_intermediate_size_attr="moe_intermediate_size",
)
register_mapping("qwen3_moe", _qwen3_moe)
register_mapping("Qwen3MoeForCausalLM", _qwen3_moe)

# ---- Phi-3 / Phi-3.5 ----
_phi3 = RotationMapping(
    mlp_in=["re:.*gate_up_proj$"],
)
register_mapping("phi3", _phi3)
register_mapping("Phi3ForCausalLM", _phi3)

# ---- InternLM2 ----
_internlm2 = RotationMapping(
    attn_q="re:.*wqkv$",
    attn_k="re:.*wqkv$",
    attn_v="re:.*wqkv$",
    attn_o="re:.*wo$",
    mlp_in=["re:.*mlp\\.w1$", "re:.*mlp\\.w3$"],
    mlp_out=["re:.*mlp\\.w2$"],
)
register_mapping("internlm2", _internlm2)
register_mapping("InternLM2ForCausalLM", _internlm2)

# ---- DeepSeek-V2 / V3 ----
_deepseek_v2 = RotationMapping(
    is_moe=True,
    attn_q="re:.*q_proj$",
    attn_k="re:.*kv_a_proj_with_mqa$",
    attn_v="re:.*kv_a_proj_with_mqa$",
    attn_o="re:.*o_proj$",
    mlp_in=["re:.*mlp\\.gate_proj$", "re:.*mlp\\.up_proj$"],
    mlp_out=["re:.*mlp\\.down_proj$"],
    moe_expert_mlp_in=[
        "re:.*experts\\.\\d+\\.gate_proj$",
        "re:.*experts\\.\\d+\\.up_proj$",
    ],
    moe_expert_mlp_out=["re:.*experts\\.\\d+\\.down_proj$"],
    moe_shared_mlp_in=[
        "re:.*shared_experts\\.gate_proj$",
        "re:.*shared_experts\\.up_proj$",
    ],
    moe_shared_mlp_out=["re:.*shared_experts\\.down_proj$"],
    moe_intermediate_size_attr="moe_intermediate_size",
)
register_mapping("deepseek_v2", _deepseek_v2)
register_mapping("deepseek_v3", _deepseek_v2)
register_mapping("DeepseekV2ForCausalLM", _deepseek_v2)
register_mapping("DeepseekV3ForCausalLM", _deepseek_v2)
