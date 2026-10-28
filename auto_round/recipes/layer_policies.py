# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Model-aware precision recommendations, separate from capability checks."""

import torch


def get_model_layer_defaults(model) -> dict:
    """Keep Nemotron-H Omni Mamba2 out-projection scales in BF16."""
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type == "nemotron_h_omni":
        return {".*mixer.out_proj$": {"scale_dtype": torch.bfloat16}}
    return {}


def apply_mixed_precision_policy(
    scheme_name,
    layer_config,
    model,
    supported_types=None,
    inner_supported_types=None,
    quant_lm_head=False,
    mllm=False,
    quant_nontext_module=False,
) -> dict:
    """Apply legacy mixed-precision recommendations without replacing user entries."""
    from auto_round.logger import logger
    from auto_round.utils import contain_any_mm_keys

    if not isinstance(scheme_name, str):
        return layer_config
    if layer_config is None:
        layer_config = {}
    if scheme_name.lower() == "gguf:q2_k_mixed":
        for n, m in model.named_modules():
            if n in layer_config:
                continue
            if isinstance(m, torch.nn.Linear) and contain_any_mm_keys(n) and not quant_nontext_module:
                layer_config[n] = "BF16"
                continue
            if n == "lm_head" or isinstance(m, torch.nn.Embedding):
                layer_config[n] = "GGUF:Q8_0"
            elif isinstance(m, torch.nn.Linear) and ("expert" not in n or "shared_expert" in n) and n != "lm_head":
                if m.weight.ndim == 3:
                    layer_config[n] = "GGUF:Q2_K_S"
                else:
                    layer_config[n] = "GGUF:Q4_K_S"
    if scheme_name.lower() == "w4a16_mixed":
        logger.warning("W4A16_MIXED is experimental and the recipe may change in the future.")
        from auto_round.utils import get_lm_head_name

        lm_head_name = get_lm_head_name(model)
        if supported_types is None:
            from auto_round.utils import SUPPORTED_DTYPES

            supported_types = SUPPORTED_DTYPES
        if inner_supported_types is None:
            from auto_round.utils import INNER_SUPPORTED_LAYER_TYPES

            inner_supported_types = INNER_SUPPORTED_LAYER_TYPES
        for n, m in model.named_modules():
            if n in layer_config:
                continue
            if (
                (type(m) in supported_types or type(m) in inner_supported_types)
                and contain_any_mm_keys(n)
                and quant_nontext_module
            ):
                layer_config[n] = {"bits": 16}
                continue
            if type(m) in supported_types or type(m) in inner_supported_types:
                if "expert" in n and "shared" not in n:
                    layer_config[n] = {"bits": 4, "data_type": "int"}
                elif n != lm_head_name and mllm:
                    layer_config[n] = {"bits": 16}
                elif n != lm_head_name or quant_lm_head:
                    layer_config[n] = {"bits": 8, "data_type": "int"}
    return layer_config
