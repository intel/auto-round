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
from typing import Any

import torch

from auto_round.compressors.base import SERIALIZATION_KEYS
from auto_round.utils.regex import (
    _numeric_values_to_regex,
    compress_regex_patterns,
    matches_any_regex,
    to_standard_regex,
)

_EXPERT_SEGMENTS = ("experts", "shared_experts")
_EXPERT_LAYER = re.compile(r"^(.*\.experts)\.(0|[1-9][0-9]*)\.([^.]+)$")


# TODO: Remove when the issue is fixed
# https://github.com/vllm-project/llm-compressor/issues/3069
def add_routed_experts_if_moe(
    targets: list[str], layer_names: list[str], include_standard_experts: bool = False
) -> list[str]:
    """Append ``"RoutedExperts"`` for routed expert layouts that llm-compressor misses.

    The generic ``"Linear"`` target does not cover all expert layouts.  Keep the
    safe canonical cases alone (``mlp.experts.*.gate_proj|up_proj|down_proj``),
    which vLLM resolves by the ``<experts>.0.gate_proj`` name probe, unless
    ``include_standard_experts`` marks the class-target fallback group.  Trigger for
    non-standard expert paths and for any routed expert structure that lives
    outside the standard ``mlp.experts`` naming.
    """
    if "RoutedExperts" in targets:
        return targets

    standard_projections = {"gate_proj", "up_proj", "down_proj"}
    for name in layer_names:
        lname = name.lower()
        if "block_sparse_moe" in lname or ".moe." in lname or ".feed_forward." in lname:
            return list(targets) + ["RoutedExperts"]

        if ".experts." not in lname:
            continue

        # Accept both module names ('...experts.0.gate_proj') and tensor names ('...gate_proj.weight').
        if standard_projections & set(name.split(".")[-2:]):
            if include_standard_experts:
                return list(targets) + ["RoutedExperts"]
            continue
        return list(targets) + ["RoutedExperts"]

    return targets


def select_fallback_group(group_layers: dict, preferred: Any = None) -> Any:
    """Return the key of the group covering the most layers, breaking ties in favor of ``preferred``."""
    return max(group_layers, key=lambda key: (len(group_layers[key]), key == preferred), default=None)


def assign_config_group_targets(
    group_layers: dict, fallback_targets: list[str] | None = None, fallback_key: Any = None
) -> dict:
    """Map each config group key to its ``targets``.

    The fallback group (``fallback_key``, default: the largest group) uses the class ``fallback_targets`` plus
    ``RoutedExperts`` when it holds experts; exact names in the other groups outrank class targets in both vLLM
    and compressed-tensors. Only pass ``fallback_targets`` when ``ignore`` lists every unquantized module.
    """
    if fallback_targets is not None and fallback_key is None:
        fallback_key = select_fallback_group(group_layers)
    targets = {}
    for key, names in group_layers.items():
        if fallback_targets is not None and key == fallback_key:
            targets[key] = add_routed_experts_if_moe(list(fallback_targets), names, include_standard_experts=True)
        else:
            targets[key] = add_routed_experts_if_moe(list(names), names)
    return targets


def compact_expert_targets(qconfig: dict) -> dict:
    """Replace per-expert exact targets with the vLLM probe names plus one regex per expert container.

    vLLM resolves a FusedMoE scheme only from the exact ``<experts>.0.<proj>`` paths (exact, so its name mapper
    still renames them), while compressed-tensors matches every expert through the regex, which enumerates exactly
    the listed expert indices so it never claims modules of another group.
    """
    for group in (qconfig.get("config_groups") or {}).values():
        targets = group.get("targets") or []
        indices = {}  # (container, projection) -> expert indices
        kept = []
        for target in targets:
            match = _EXPERT_LAYER.match(target)
            if match is None:
                kept.append(target)
                continue
            indices.setdefault((match.group(1), match.group(3)), set()).add(int(match.group(2)))
            if match.group(2) == "0":
                kept.append(target)
        if not indices:
            continue
        projections = {}  # (container, expert index set) -> projections
        for (container, projection), expert_ids in indices.items():
            projections.setdefault((container, frozenset(expert_ids)), []).append(re.escape(projection))
        regexes = []
        for (container, expert_ids), names in projections.items():
            alternatives = names[0] if len(names) == 1 else "(?:" + "|".join(sorted(names)) + ")"
            regexes.append(
                f"re:^{re.escape(container)}\\.{_numeric_values_to_regex(list(expert_ids))}\\.{alternatives}$"
            )
        group["targets"] = kept + compress_regex_patterns(regexes)
    return qconfig


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
        # Exact paths, unlike "re:" entries, are renamed by vLLM's WeightsMapper.
        for projection in ("gate_proj", "up_proj", "down_proj"):
            entry = f"{module}.0.{projection}"
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


def _compact_llmcompressor_regex_config(qconfig: dict) -> dict:
    """Merge ``re:`` ignore entries that differ by one numeric path segment.

    Exact module paths are kept literal: vLLM's ``WeightsMapper`` remaps exact paths but leaves ``re:`` entries
    untouched, so turning exact paths into regexes would break renamed (e.g. VLM) checkpoints.
    """
    if qconfig.get("ignore"):
        qconfig["ignore"] = compress_regex_patterns(qconfig["ignore"])
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
    qconfig = ignore_fused_experts_for_vllm(qconfig, ignored, [revert(n) for n in quantized_layers])
    # Runs after the revert so the expert regexes are built from checkpoint names.
    return _compact_llmcompressor_regex_config(compact_expert_targets(qconfig))


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


# Tuning hyper-parameters worth recording so a llm-compressor checkpoint can be
# traced back to (and re-created with) the AutoRound command that produced it.
AUTOROUND_CONFIG_KEY = "autoround_config"


def _autoround_config_fields() -> tuple[str, ...]:
    """Shared compressor fields plus every algorithm-declared export field.

    ``autoround_version`` is injected into ``serialization_dict`` at export time and
    is not a SerializedCompressorConfig field. Algorithm fields are resolved lazily
    so configs registered after import are still covered.
    """
    from auto_round.algorithms.config import all_algorithm_serialized_fields

    execution_fields = {
        "quant_block_list",
        "to_quant_block_names",
        "scale_dtype",
        "supported_types",
    }
    fields = tuple(key for key in SERIALIZATION_KEYS if key not in execution_fields)
    return ("autoround_version",) + fields + tuple(sorted(all_algorithm_serialized_fields()))


def _to_json_value(value):
    import enum

    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, enum.Enum):
        enum_value = value.value
        if isinstance(enum_value, (bool, int, float, str)):
            return enum_value
        return value.name
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, dict):
        return {str(k): _to_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(v) for v in value]
    return str(value)


def add_autoround_config(quantization_config: dict, serialization_dict: dict | None) -> dict:
    """Record AutoRound tuning hyper-parameters in an llm-compressor ``quantization_config``.

    The values are stored under a single nested ``"autoround_config"`` key so that the
    compressed-tensors / vLLM loaders, which only read the keys they know, are unaffected.
    Only runs when ``quantization_config["provider"] == "auto-round"``, so shared export
    helpers do not write AutoRound-branded metadata for other providers.
    """
    if not isinstance(quantization_config, dict) or not serialization_dict:
        return quantization_config
    if quantization_config.get("provider") != "auto-round":
        return quantization_config
    has_kv_cache_quantization = quantization_config.get("kv_cache_scheme") is not None
    has_attention_quantization = quantization_config.get("attention_input_activations") is not None
    conditional_fields = {
        "static_kv_dtype": has_kv_cache_quantization,
        "static_kv_granularity": has_kv_cache_quantization,
        "static_attention_dtype": has_attention_quantization,
        "static_attention_granularity": has_attention_quantization,
    }
    autoround_config = {}
    for key in _autoround_config_fields():
        if key in conditional_fields and not conditional_fields[key]:
            continue
        value = serialization_dict.get(key)
        if value is None:
            continue
        autoround_config[key] = _to_json_value(value)
    if autoround_config:
        quantization_config[AUTOROUND_CONFIG_KEY] = autoround_config
    return quantization_config
