# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""Model architecture mapping for Hadamard rotation.

Each :class:`RotationMapping` describes *where* the rotation-relevant modules
live inside a model.  Currently supports LLaMA-2, LLaMA-3, and Qwen-3 (dense).

New architectures can be supported by calling :func:`register_mapping`.
"""

from __future__ import annotations

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

    Attribute names follow the dot-path convention relative to the model or
    each decoder layer.

    Config attribute names (read from ``model.config``):
        num_heads_attr, hidden_size_attr, intermediate_size_attr
        head_dim_override – explicit head dim (skip hidden_size // num_heads)
    """

    # -- top-level modules (dot-path from model root) --
    embedding: str = "model.embed_tokens"
    lm_head: str = "lm_head"
    positional_embedding: Optional[str] = None  # e.g. "model.decoder.embed_positions" for OPT

    # -- layers container (dot-path from model root) --
    layers_attr: str = "model.layers"

    # -- per-layer: attention (dot-path from each layer) --
    attn_input_ln: str = "input_layernorm"
    attn_q: str = "self_attn.q_proj"
    attn_k: str = "self_attn.k_proj"
    attn_v: str = "self_attn.v_proj"
    attn_o: str = "self_attn.o_proj"

    # -- per-layer: MLP (dot-path from each layer) --
    mlp_input_ln: str = "post_attention_layernorm"
    mlp_in: List[str] = field(default_factory=lambda: ["mlp.up_proj", "mlp.gate_proj"])
    mlp_out: str = "mlp.down_proj"

    # -- final norm (dot-path from model root) --
    pre_head_ln: str = "model.norm"

    # -- head dim override (None = hidden_size // num_heads) --
    attn_head_dim: Optional[int] = None

    # -- config attr names --
    num_heads_attr: str = "num_attention_heads"
    hidden_size_attr: str = "hidden_size"
    intermediate_size_attr: str = "intermediate_size"



# ---------------------------------------------------------------------------
# Helper: resolve a dot-path attribute on a module
# ---------------------------------------------------------------------------


def _resolve(root, dot_path: str):
    """Resolve ``'a.b.c'`` to ``root.a.b.c``."""
    obj = root
    for attr in dot_path.split("."):
        obj = getattr(obj, attr)
    return obj


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

    Tries ``model.config.model_type`` first, then ``model.__class__.__name__``.
    """
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    if model_type in MAPPING_REGISTRY:
        return MAPPING_REGISTRY[model_type]

    arch = model.__class__.__name__
    if arch in MAPPING_REGISTRY:
        return MAPPING_REGISTRY[arch]

    logger.warning(
        f"Unrecognised architecture '{arch}' (model_type='{model_type}'). "
        "Falling back to default (LLaMA-like) mapping."
    )
    return RotationMapping()


# ===================================================================
# Built-in mappings
# ===================================================================

# LLaMA-2 / LLaMA-3 / Mistral / Yi — all share the same layout
_default = RotationMapping()

register_mapping("llama", _default)
register_mapping("LlamaForCausalLM", _default)

# Qwen-3 dense — identical layout to LLaMA
register_mapping("qwen3", _default)
register_mapping("Qwen3ForCausalLM", _default)

# Qwen-2 / Qwen-2.5 dense — identical layout to LLaMA
register_mapping("qwen2", _default)
register_mapping("Qwen2ForCausalLM", _default)

# ---- OPT ----
# OPT uses standard LayerNorm (with bias, subtracts mean),
# different module names, and tied lm_head ↔ embedding weights.
_opt = RotationMapping(
    embedding="model.decoder.embed_tokens",
    lm_head="lm_head",
    positional_embedding="model.decoder.embed_positions",
    layers_attr="model.decoder.layers",
    attn_input_ln="self_attn_layer_norm",
    attn_q="self_attn.q_proj",
    attn_k="self_attn.k_proj",
    attn_v="self_attn.v_proj",
    attn_o="self_attn.out_proj",
    mlp_input_ln="final_layer_norm",
    mlp_in=["fc1"],
    mlp_out="fc2",
    pre_head_ln="model.decoder.final_layer_norm",
    intermediate_size_attr="ffn_dim",
)
register_mapping("opt", _opt)
register_mapping("OPTForCausalLM", _opt)

