# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Name-only restoration of the native DeepSeek V4 checkpoint namespace.

Do not use source FP8 converters here: exported NVFP4 tensors already have their
final bytes, dtype and scale layout. Model modules remain in the HF namespace.
"""

import re


def is_deepseek_v4(model):
    return getattr(getattr(model, "config", None), "model_type", None) == "deepseek_v4"


def checkpoint_name(name: str) -> str:
    """Restore a tensor or module path, retaining quantization tensor suffixes."""
    name = name.removeprefix("model.")
    parts = name.split(".")
    if parts[0] in ("embed_tokens", "lm_head"):
        parts[0] = {"embed_tokens": "embed", "lm_head": "head"}[parts[0]]
        return ".".join(parts)
    if name.startswith("hc_head.hc_"):
        return name.replace("hc_head.hc_", "hc_head_", 1)
    name = re.sub(r"^(layers\.\d+)\.(attn|ffn)_hc\.(base|fn|scale)$", r"\1.hc_\2_\3", name)
    name = name.replace(".input_layernorm.", ".attn_norm.")
    name = name.replace(".post_attention_layernorm.", ".ffn_norm.")
    # The scorer is native attn.indexer; the remaining indexer tensors belong
    # under attn.indexer.compressor. Apply these before replacing self_attn.
    name = name.replace(".self_attn.compressor.indexer.scorer.", ".self_attn.indexer.")
    name = name.replace(".self_attn.compressor.indexer.q_b_proj.", ".self_attn.indexer.wq_b.")
    name = name.replace(".self_attn.compressor.indexer.", ".self_attn.indexer.compressor.")
    parts = name.split(".")
    if "self_attn" in parts:
        replacements = {
            "self_attn": "attn",
            "q_a_proj": "wq_a",
            "q_b_proj": "wq_b",
            "kv_proj": "wkv",
            "o_a_proj": "wo_a",
            "o_b_proj": "wo_b",
            "q_a_norm": "q_norm",
            "gate_proj": "wgate",
            "position_bias": "ape",
            "sinks": "attn_sink",
        }
        parts = [replacements.get(part, part) for part in parts]
        name = ".".join(parts).replace(".compressor.kv_norm.", ".compressor.norm.")
        return name
    elif "mlp" in parts:
        replacements = {"mlp": "ffn"}
        if "experts" in parts or "shared_experts" in parts:
            replacements.update(gate_proj="w1", up_proj="w3", down_proj="w2")
        parts = [replacements.get(part, part) for part in parts]
        if parts[-1] == "e_score_correction_bias":
            parts[-1] = "bias"
    return ".".join(parts)


def checkpoint_state_dict(state_dict):
    """Rename dictionary keys without copying or converting any tensors."""
    from auto_round.utils.model_free_utils import split_fused_expert_tensors

    # Unquantized experts may still be stacked in the HF model. Restore the
    # native per-expert matrices using views, without running FP8 conversion.
    state_dict = split_fused_expert_tensors(state_dict)
    result = {}
    for name, tensor in state_dict.items():
        target = checkpoint_name(name)
        if target in result:
            raise ValueError(f"DeepSeek V4 checkpoint name collision: {name!r} -> {target!r}")
        result[target] = tensor
    return result
