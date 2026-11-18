# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""Rename DeepSeek V4 exports without reversing lossy weight conversions."""

import copy
import json
import os
import re


def build_checkpoint_name_maps(source_keys):
    """Use forward rules and source keys to disambiguate the reverse mapping.

    In particular, indexer/compressor paths cannot safely be recovered by
    swapping regexes, and HC parameters do not end in ``.weight``. Expert
    fusion converters are deliberately excluded: AR exports separate Linears.
    """
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping
    from transformers.core_model_loading import WeightRenaming

    rules = [r for r in get_checkpoint_conversion_mapping("deepseek_v4") if isinstance(r, WeightRenaming)]
    aliases, modules = {}, {}

    def add(table, alias, original):
        if alias in table and table[alias] != original:
            raise ValueError(f"Ambiguous DeepSeek V4 alias {alias}: {table[alias]} / {original}")
        table[alias] = original

    for original in source_keys:
        if original.startswith("mtp."):
            continue
        name = original
        for rule in rules:
            if len(rule.source_patterns) != 1 or len(rule.target_patterns) != 1:
                raise ValueError("Expected scalar DeepSeek V4 rename rule")
            name = re.sub(rule.source_patterns[0], rule.target_patterns[0], name)
        name = re.sub(
            r"(\.experts\.\d+)\.w([123])\.",
            lambda match: match[1] + "." + {"1": "gate_proj", "2": "down_proj", "3": "up_proj"}[match[2]] + ".",
            name,
        )
        for alias in (name, "model." + name):
            add(aliases, alias, original)
            if alias.endswith(".weight") and original.endswith(".weight"):
                add(modules, alias[:-7], original[:-7])
    return aliases, modules


def restore_tensor_name(name, aliases, modules, source_keys):
    if name in aliases:
        return aliases[name]
    for suffix in (".weight_packed", ".weight_scale", ".weight_global_scale", ".input_global_scale"):
        if name.endswith(suffix) and name[: -len(suffix)] in modules:
            return modules[name[: -len(suffix)]] + suffix
    if name in source_keys:
        return name
    raise ValueError(f"Unmapped DeepSeek V4 export tensor: {name}")


def restore_quantization_names(config, modules, tensor_names):
    """Restore explicit targets and derive ignores from actual saved precision."""
    result = copy.deepcopy(config)
    for group in result["config_groups"].values():
        group["targets"] = [modules.get(name, name) for name in group["targets"]]
    names = set(tensor_names)
    result["ignore"] = sorted(
        name[:-7]
        for name in names
        if name.endswith(".weight")
        and name[:-7] + ".weight_scale" not in names
        and name[:-7] + ".weight_packed" not in names
    )
    result["ignore"].append("re:.*mtp\\..*")
    # vLLM maps the tensor suffix ``head.weight`` but preserves bare class
    # targets. Include its runtime spelling as well as the checkpoint module.
    if "head" in result["ignore"]:
        result["ignore"].append("lm_head")
    if "embed" in result["ignore"]:
        result["ignore"].extend(["embed_tokens", "model.embed_tokens"])
    if "norm" in result["ignore"]:
        result["ignore"].append("model.norm")
    return result


def prepare_native_export(model, state_dict, source_dir):
    """Return a rename-only state dict, preserving each tensor object and dtype."""
    index_path = os.path.join(source_dir, "model.safetensors.index.json")
    with open(index_path, encoding="utf-8") as stream:
        source_keys = json.load(stream)["weight_map"]
    aliases, modules = build_checkpoint_name_maps(source_keys)
    renamed = {}
    for name, tensor in state_dict.items():
        target = restore_tensor_name(name, aliases, modules, source_keys)
        if target in renamed:
            raise ValueError(f"Duplicate DeepSeek V4 export tensor: {target}")
        renamed[target] = tensor
    model.config.quantization_config = restore_quantization_names(model.config.quantization_config, modules, renamed)
    return renamed
