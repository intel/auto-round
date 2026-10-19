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
"""Load-time support for NVIDIA's Nemotron-H Omni wrapper checkpoints.

``nemotron_h_omni`` (``NemotronH_Omni_Reasoning_V3``, remote code) wraps a Nemotron-H
language model -- native ``nemotron_h`` (e.g. Nemotron 3.5 Super VL) or the remote
``nemotron_h_puzzle`` variant (e.g. Nemotron 3.5 Bolt) -- plus a RADIO vision tower.
Plain ``nemotron_h`` text checkpoints are not touched by anything in this module.
"""

import glob
import json
import os
import re
from collections.abc import Iterable

import torch
import torch.nn as nn

from auto_round.logger import logger
from auto_round.utils.model import get_module

NEMOTRON_H_OMNI_MODEL_TYPE = "nemotron_h_omni"

# On disk the wrapper's language-model trunk is ``language_model.backbone.*``; the module
# tree exposes it as ``language_model.model.*``. The inner model's own ``backbone.`` ->
# ``model.`` rule cannot do this rename for the nested trunk: reversed, ``model.`` also
# matches the tail of ``language_model.``. Use the full prefix instead.
_NESTED_TRUNK_SOURCE = "language_model.backbone."
_NESTED_TRUNK_TARGET = "language_model.model."

_SSM_CORE_PATTERNS = (r"\.mixer\.A_log$", r"\.mixer\.D$", r"\.mixer\.dt_bias$")
_ROUTER_BIAS_PATTERNS = (r"\.mixer\.gate\.e_score_correction_bias$",)
# Composition-level renames registered by the checkpoints' remote code
# (``register_nemotron_h_omni_conversion_mapping`` in ``modeling_nemotron_h_omni.py``).
_REMOTE_OMNI_RENAMES = (
    (r"^mlp1\.0\.", r"vision_projector\.mlp1\.norm\."),
    (r"^mlp1\.1\.", r"vision_projector\.mlp1\.linear1\."),
    (r"^mlp1\.3\.", r"vision_projector\.mlp1\.linear2\."),
    (r"^sound_encoder\.", r"sound_projector\.sound_encoder\."),
    (r"^sound_projection\.", r"sound_projector\.sound_projection\."),
)


def _cfg_get(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def is_nemotron_h_omni_config(config) -> bool:
    """Whether ``config`` (object or ``config.json`` dict) is a Nemotron-H Omni wrapper."""
    return config is not None and _cfg_get(config, "model_type") == NEMOTRON_H_OMNI_MODEL_TYPE


def prefer_remote_omni_class(config, trust_remote_code: bool) -> bool:
    """Whether to build the wrapper from the checkpoint's remote code instead of a native class.

    Transformers >= 5.20 ships a native class with the same name, whose config rejects the
    remote ``nemotron_h_puzzle`` language model (Bolt). Wrappers around a native ``nemotron_h``
    (Super VL) keep the default class resolution.
    """
    if not (trust_remote_code and is_nemotron_h_omni_config(config)):
        return False
    llm_config = _cfg_get(config, "llm_config")
    auto_map = _cfg_get(config, "auto_map") or {}
    return _cfg_get(llm_config, "model_type") == "nemotron_h_puzzle" and "AutoModelForCausalLM" in auto_map


def nemotron_h_omni_load_kwargs(config) -> dict:
    """``from_pretrained`` kwargs for a full (non-streamed) load of an Omni wrapper.

    Transformers scopes a sub-model's registered renames to that sub-model's prefix, so the
    nested-trunk rename is passed as a root-level ``key_mapping`` instead.
    """
    if not is_nemotron_h_omni_config(config):
        return {}
    return {"key_mapping": {r"^language_model\.backbone\.": _NESTED_TRUNK_TARGET}}


def register_nemotron_h_omni_checkpoint_mapping(config, trust_remote_code: bool = False) -> bool:
    """Register the conversion rules an Omni wrapper needs; idempotent, no-op for other models.

    * The nested-trunk rename is appended to the inner model type's mapping, so the upstream
      rules stay in place; it only matches ``language_model.backbone.*`` names, which exist
      only in Omni wrapper checkpoints.
    * When the remote class is used (see :func:`prefer_remote_omni_class`), Transformers would
      look its class name up first and find the *native* class's mapping (projector renamed to
      ``multi_modal_projector``), so the projector would silently stay random. Map the class
      name to the remote code's own composition-level renames instead.
    """
    if not is_nemotron_h_omni_config(config):
        return False
    try:
        from transformers.conversion_mapping import (
            get_checkpoint_conversion_mapping,
            register_checkpoint_conversion_mapping,
        )
        from transformers.core_model_loading import WeightRenaming
    except ImportError:  # transformers < 5
        return False
    if prefer_remote_omni_class(config, trust_remote_code):
        for class_name in _cfg_get(config, "architectures") or []:
            if get_checkpoint_conversion_mapping(class_name) is not None:
                register_checkpoint_conversion_mapping(
                    class_name, [WeightRenaming(src, dst) for src, dst in _REMOTE_OMNI_RENAMES], overwrite=True
                )
    llm_config = _cfg_get(config, "llm_config") or _cfg_get(config, "text_config")
    text_model_type = _cfg_get(llm_config, "model_type") if llm_config is not None else None
    if text_model_type is None:
        return False
    rules = list(get_checkpoint_conversion_mapping(text_model_type) or [])
    if any(
        getattr(rule, "source_patterns", None) == [_NESTED_TRUNK_SOURCE]
        and getattr(rule, "target_patterns", None) == [_NESTED_TRUNK_TARGET]
        for rule in rules
    ):
        return True
    rules.append(WeightRenaming(_NESTED_TRUNK_SOURCE, _NESTED_TRUNK_TARGET))
    register_checkpoint_conversion_mapping(text_model_type, rules, overwrite=True)
    logger.info("Registered %s -> %s for %s", _NESTED_TRUNK_SOURCE, _NESTED_TRUNK_TARGET, text_model_type)
    return True


def _text_config(config):
    for attr in ("text_config", "llm_config"):
        inner = getattr(config, attr, None)
        if inner is not None:
            return inner
    return config


def _patch_zamba2_group_size(model: nn.Module) -> int:
    """Restore ``Zamba2RMSNormGated.group_size``, which a meta/low-memory load can drop."""
    config = _text_config(model.config)
    try:
        group_size = config.mamba_num_heads * config.mamba_head_dim // config.n_groups
    except (AttributeError, ZeroDivisionError):
        return 0
    patched = 0
    for module in model.modules():
        if module.__class__.__name__ != "Zamba2RMSNormGated":
            continue
        if getattr(module, "group_size", None) != group_size:
            object.__setattr__(module, "group_size", group_size)
        patched += 1
    return patched


def _source_shard_map(source_dir: str) -> dict[str, str]:
    """Map checkpoint tensor name -> shard path (index file, single file, or shard headers)."""
    from safetensors import safe_open

    index_path = os.path.join(source_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f).get("weight_map", {})
        if weight_map:
            return {k: os.path.join(source_dir, shard) for k, shard in weight_map.items()}
    shards = [os.path.join(source_dir, "model.safetensors")]
    if not os.path.exists(shards[0]):
        shards = sorted(glob.glob(os.path.join(source_dir, "model-*.safetensors")))
    source_to_shard = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            for k in f.keys():
                source_to_shard[k] = shard
    return source_to_shard


def _resolve_source_dir(name_or_path) -> str | None:
    """Local checkpoint directory for a path or an (already downloaded) hub id."""
    if not name_or_path:
        return None
    if os.path.isdir(name_or_path):
        return name_or_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(name_or_path, allow_patterns=["*.safetensors", "*.safetensors.index.json"])
    except Exception as exc:  # noqa: BLE001 - offline / gated: keep the loaded dtype
        logger.warning("nemotron_h_omni: cannot resolve %s (%s); FP32 restore skipped", name_or_path, exc)
        return None


def _restore_tensors_from_source(
    model: nn.Module, source_dir: str, name_patterns: Iterable[str], dtype: torch.dtype
) -> int:
    """Reload matching tensors from the checkpoint as ``dtype`` (they are stored in FP32)."""
    from safetensors import safe_open

    compiled = [re.compile(p) for p in name_patterns]
    by_shard: dict[str, list[str]] = {}
    for key, shard in _source_shard_map(source_dir).items():
        if any(c.search(key) for c in compiled):
            by_shard.setdefault(shard, []).append(key)
    restored = 0
    for shard, keys in by_shard.items():
        with safe_open(shard, framework="pt", device="cpu") as f:
            for src_key in keys:
                module_path = src_key.replace(_NESTED_TRUNK_SOURCE, _NESTED_TRUNK_TARGET, 1)
                parent_path, attr = module_path.rsplit(".", 1)
                parent = get_module(model, parent_path)
                old = getattr(parent, attr, None) if parent is not None else None
                if not isinstance(old, torch.Tensor):
                    continue
                src = f.get_tensor(src_key)
                if old.shape != src.shape:
                    logger.warning("nemotron_h_omni: shape mismatch for %s, skipping restore", module_path)
                    continue
                src = src.to(dtype)
                if isinstance(old, nn.Parameter):
                    setattr(parent, attr, nn.Parameter(src, requires_grad=old.requires_grad))
                else:
                    persistent = attr not in getattr(parent, "_non_persistent_buffers_set", set())
                    parent.register_buffer(attr, src, persistent=persistent)
                restored += 1
    return restored


def apply_nemotron_h_omni_post_load(model: nn.Module) -> dict:
    """Post-load fix-ups for Omni wrappers; no-op for every other model.

    Restores the Mamba2 recurrence parameters (``A_log``/``D``/``dt_bias``) and the router
    ``e_score_correction_bias`` from the checkpoint in FP32 (the low-precision load
    truncates them), and re-applies the gated-norm ``group_size``.
    """
    summary = {"zamba2_patched": 0, "high_precision_restored": 0}
    config = getattr(model, "config", None)
    if not is_nemotron_h_omni_config(config) or getattr(model, "_autoround_nh_post_load_applied", False):
        return summary
    summary["zamba2_patched"] = _patch_zamba2_group_size(model)
    source_dir = _resolve_source_dir(_cfg_get(config, "_name_or_path"))
    if source_dir is not None:
        patterns = _SSM_CORE_PATTERNS + _ROUTER_BIAS_PATTERNS
        summary["high_precision_restored"] = _restore_tensors_from_source(model, source_dir, patterns, torch.float32)
        logger.info("nemotron_h_omni: restored %d tensor(s) in FP32", summary["high_precision_restored"])
    model._autoround_nh_post_load_applied = True
    return summary
