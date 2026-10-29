# Copyright (c) 2025 Intel Corporation
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


import re

import torch

from auto_round.utils import matches_any_regex, to_standard_regex

_EXPERT_SEGMENTS = ("experts", "shared_experts")


# TODO: Remove when the issue is fixed
# https://github.com/vllm-project/llm-compressor/issues/3069
def add_routed_experts_if_moe(targets: list[str], layer_names: list[str]) -> list[str]:
    """Append ``"RoutedExperts"`` for routed expert layouts that llm-compressor misses.

    The generic ``"Linear"`` target does not cover all expert layouts.  Keep the
    safe canonical cases alone (``mlp.experts.*.gate_proj|up_proj|down_proj``),
    but trigger for non-standard expert paths and for any routed
    expert structure that lives outside the standard ``mlp.experts`` naming.
    """
    if "RoutedExperts" in targets:
        return targets

    def _projection_name(name: str) -> str:
        # Eg. '...experts.0.gate_proj.weight' -> ['...', 'experts', '0', 'gate_proj', 'weight']
        parts = name.split(".")
        if len(parts) >= 2:
            return parts[-2]
        return parts[-1]

    for name in layer_names:
        lname = name.lower()
        if "block_sparse_moe" in lname or ".moe." in lname or ".feed_forward." in lname:
            return list(targets) + ["RoutedExperts"]

        if ".experts." not in lname:
            continue

        projection = _projection_name(name)
        if projection in {"gate_proj", "up_proj", "down_proj"}:
            continue
        return list(targets) + ["RoutedExperts"]

    return targets


def ignore_fused_experts_for_vllm(qconfig: dict, ignored_layers: list[str], quantized_layers: list[str]) -> dict:
    """Add vLLM-matchable ignore entries for ignored fused expert modules when ``RoutedExperts`` is targeted.

    vLLM resolves a FusedMoE scheme via ``<module>.0.{gate,up,down}_proj``, which fused layer names such as
    ``<module>.w13_weight`` never match, so the global ``RoutedExperts`` target would quantize them.
    """
    groups = qconfig.get("config_groups") or {}
    if not any("RoutedExperts" in (g.get("targets") or []) for g in groups.values()):
        return qconfig

    ignore = list(qconfig.get("ignore") or [])
    for layer_name in dict.fromkeys(ignored_layers):
        if layer_name.startswith("re:"):
            continue
        parts = layer_name.split(".")
        if parts[-1] in _EXPERT_SEGMENTS:
            module = layer_name
        elif len(parts) >= 2 and parts[-2] in _EXPERT_SEGMENTS and not parts[-1].isdigit():
            module = ".".join(parts[:-1])
        else:
            continue
        # A partially quantized module cannot be described as ignored.
        if any(q == module or q.startswith(f"{module}.") for q in quantized_layers):
            continue
        entry = f"re:^{re.escape(module)}\\.\\d+\\.(gate|up|down)_proj$"
        if entry not in ignore:
            ignore.append(entry)
    qconfig["ignore"] = ignore
    return qconfig


def get_checkpoint_name_reverter(model: torch.nn.Module):
    """Return ``fn(module_name) -> name`` mapping a loaded-model module name to the saved checkpoint namespace."""
    from auto_round.utils.common import (
        get_reverse_checkpoint_conversion_mapping,
        get_reverse_weight_transforms,
        revert_checkpoint_conversion_mapping,
        revert_name_with_weight_transforms,
    )

    transforms = get_reverse_weight_transforms(model)
    mapping = get_reverse_checkpoint_conversion_mapping(model) if transforms is None else None
    if transforms is None and not mapping:
        return lambda name: name

    def _revert(name: str) -> str:
        # Renamings are defined on parameter keys, so revert a full weight key.
        key = f"{name}.weight"
        if transforms is not None:
            reverted = revert_name_with_weight_transforms(key, transforms)
        else:
            reverted = revert_checkpoint_conversion_mapping(key, mapping)
        return reverted[: -len(".weight")] if reverted.endswith(".weight") else name

    return _revert


def revert_quantization_config_names(qconfig: dict, revert) -> dict:
    """Rewrite module names in ``ignore`` and group ``targets`` with ``revert`` so they match the saved weights."""

    def _revert_entry(entry):
        if not isinstance(entry, str):
            return entry
        if not entry.startswith("re:"):
            return revert(entry)
        # Only rewrite regexes that wrap a single escaped module name, e.g. ``re:.*model\.layers\.0\.mlp\.gate.*``.
        match = re.fullmatch(r"(re:(?:\.\*)?)(.*?)((?:\.\*)?)", entry)
        plain = re.sub(r"\\(.)", r"\1", match.group(2))
        if not plain or re.escape(plain) != match.group(2):
            return entry
        return f"{match.group(1)}{re.escape(revert(plain))}{match.group(3)}"

    if qconfig.get("ignore"):
        qconfig["ignore"] = list(dict.fromkeys(_revert_entry(e) for e in qconfig["ignore"]))
    for group in (qconfig.get("config_groups") or {}).values():
        if group.get("targets"):
            group["targets"] = [_revert_entry(t) for t in group["targets"]]
    return qconfig


def finalize_llmcompressor_config_names(
    qconfig: dict, model: torch.nn.Module, ignored_layers: list[str], quantized_layers: list[str]
) -> dict:
    """Revert config module names to the checkpoint namespace, then add vLLM ignore entries for fused experts."""
    revert = get_checkpoint_name_reverter(model)
    qconfig = revert_quantization_config_names(qconfig, revert)
    ignored = list(qconfig.get("ignore") or []) + [
        revert(n) for n in ignored_layers + collect_fused_expert_modules(model) if not n.startswith("re:")
    ]
    return ignore_fused_experts_for_vllm(qconfig, ignored, [revert(n) for n in quantized_layers])


def collect_fused_expert_modules(model: torch.nn.Module) -> list[str]:
    """Return expert modules still holding fused 3-D weights, i.e. never unfused into quantized Linear layers."""
    return [
        name
        for name, module in model.named_modules()
        if name.split(".")[-1] in _EXPERT_SEGMENTS and any(p.dim() == 3 for p in module.parameters(recurse=False))
    ]


def generate_ignore_regex_list(regex_config: dict[str, dict], layer_config: dict[str, dict]) -> list[str]:
    """
    Generate ignore regex list for llm_compressor based on regex_config and layer_config.

    Rules:
    1. Any layer in regex_config with bits >= 16 is ignored.
    2. Any layer in layer_config with bits >= 16 is ignored if not already included.
    3. Output regex patterns are normalized for llm_compressor ('re:...' style).

    Args:
        regex_config (Dict[str, Dict]): dynamic quantization config
        layer_config (Dict[str, Dict]): layer-wise quantization config

    Returns:
        List[str]: List of regex patterns to ignore during quantization.
    """
    prefix = "re:"
    ignore_regex: list[str] = []

    # Step 1: Add regex_config keys with bits >= 16
    for key, cfg in regex_config.items():
        bits = cfg.get("bits")
        if bits > 8:
            ignore_regex.append(prefix + to_standard_regex(key))

    # Step 2: Add all full named layer from layer_config if bits >= 16
    for key, cfg in layer_config.items():
        bits = cfg.get("bits")
        if bits > 8:
            ignore_regex.append(key)

    return ignore_regex
