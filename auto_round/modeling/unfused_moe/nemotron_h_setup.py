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
"""Nemotron-H post-load fix-ups and default layer_config patterns."""

import json
import os
import re
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn

from auto_round.logger import logger

_NEMOTRON_H_OUT_PROJ_PATTERN = ".*mixer.out_proj$"

_NH_SSM_CORE_PATTERNS: tuple[str, ...] = (
    r"\.mixer\.A_log$",
    r"\.mixer\.D$",
    r"\.mixer\.dt_bias$",
)
_NH_ROUTER_BIAS_PATTERNS: tuple[str, ...] = (r"\.mixer\.gate\.e_score_correction_bias$",)


def _patch_zamba2_group_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is None:
        return 0
    try:
        intermediate_size = config.mamba_num_heads * config.mamba_head_dim
        n_groups = config.n_groups
    except AttributeError:
        return 0
    if n_groups <= 0:
        return 0
    group_size = intermediate_size // n_groups

    patched_instances = 0
    patched_classes: set[type] = set()
    for module in model.modules():
        if module.__class__.__name__ != "Zamba2RMSNormGated":
            continue
        cls = module.__class__
        if cls not in patched_classes:
            cls.group_size = group_size
            patched_classes.add(cls)
        if getattr(module, "group_size", None) != group_size:
            object.__setattr__(module, "group_size", group_size)
        patched_instances += 1

    if patched_instances:
        logger.info(
            "nemotron_h: patched %d Zamba2RMSNormGated module(s) + %d class(es) (group_size=%d)",
            patched_instances,
            len(patched_classes),
            group_size,
        )
    return patched_instances


def _resolve_module(root: nn.Module, dotted_path: str) -> nn.Module | None:
    if not dotted_path:
        return root
    mod: nn.Module | None = root
    for part in dotted_path.split("."):
        if mod is None:
            return None
        if part.isdigit():
            try:
                mod = mod[int(part)]  # type: ignore[index]
            except (IndexError, TypeError, KeyError):
                return None
        else:
            mod = getattr(mod, part, None)
    return mod


def _nh_source_to_module(src_key: str) -> str:
    if src_key.startswith("backbone."):
        src_key = "model." + src_key[len("backbone.") :]
    if src_key.endswith(".embedding.weight"):
        src_key = src_key[: -len(".embedding.weight")] + ".embeddings.weight"
    return src_key


def _restore_tensors_from_source(
    model: nn.Module,
    source_dir: str,
    name_patterns: Iterable[str],
    target_dtype: torch.dtype,
    source_to_module: Callable[[str], str] = _nh_source_to_module,
) -> dict[str, tuple[torch.dtype, torch.dtype]]:
    try:
        from safetensors import safe_open
    except ImportError:
        return {}

    if not os.path.isdir(source_dir):
        return {}

    index_path = os.path.join(source_dir, "model.safetensors.index.json")
    single_path = os.path.join(source_dir, "model.safetensors")
    source_to_shard: dict[str, str] = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            idx = json.load(f)
        for k, shard in idx["weight_map"].items():
            source_to_shard[k] = os.path.join(source_dir, shard)
    elif os.path.exists(single_path):
        with safe_open(single_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                source_to_shard[k] = single_path
    else:
        return {}

    compiled = [re.compile(p) for p in name_patterns]
    if not compiled:
        return {}

    by_shard: dict[str, list[str]] = {}
    for k, shard in source_to_shard.items():
        if any(c.search(k) for c in compiled):
            by_shard.setdefault(shard, []).append(k)

    if not by_shard:
        return {}

    restored: dict[str, tuple[torch.dtype, torch.dtype]] = {}
    for shard, keys in by_shard.items():
        with safe_open(shard, framework="pt", device="cpu") as f:
            for src_key in keys:
                try:
                    src_tensor = f.get_tensor(src_key)
                except Exception as exc:
                    logger.warning("failed to read %s from %s: %s", src_key, shard, exc)
                    continue

                module_path = source_to_module(src_key)
                parts = module_path.split(".")
                parent_path, attr_name = ".".join(parts[:-1]), parts[-1]
                parent = _resolve_module(model, parent_path)
                if parent is None or not hasattr(parent, attr_name):
                    continue
                old = getattr(parent, attr_name)
                if not isinstance(old, torch.Tensor):
                    continue
                if old.shape != src_tensor.shape:
                    logger.warning(
                        "restore: shape mismatch for %s — model %s, source %s",
                        module_path,
                        list(old.shape),
                        list(src_tensor.shape),
                    )
                    continue

                new_tensor = src_tensor.to(target_dtype)
                old_dtype = old.dtype
                if isinstance(old, nn.Parameter):
                    setattr(parent, attr_name, nn.Parameter(new_tensor, requires_grad=old.requires_grad))
                else:
                    persistent = attr_name not in getattr(parent, "_non_persistent_buffers_set", set())
                    parent.register_buffer(attr_name, new_tensor, persistent=persistent)
                restored[module_path] = (old_dtype, target_dtype)
    return restored


def _resolve_source_dir(model: nn.Module) -> str | None:
    config = getattr(model, "config", None)
    if config is None:
        return None
    name_or_path = getattr(config, "_name_or_path", None) or getattr(config, "name_or_path", None)
    if not name_or_path:
        return None
    if os.path.isdir(name_or_path):
        return name_or_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(
            name_or_path,
            allow_patterns=[
                "model.safetensors.index.json",
                "model-*.safetensors",
                "model.safetensors",
            ],
        )
    except Exception as exc:
        logger.info("nemotron_h: snapshot_download for %r failed (%s); skipping overrides.", name_or_path, exc)
        return None


def _apply_high_precision_overrides(
    model: nn.Module,
    *,
    ssm_core_dtype: torch.dtype | None,
    router_bias_dtype: torch.dtype | None,
) -> int:
    if ssm_core_dtype is None and router_bias_dtype is None:
        return 0
    source_dir = _resolve_source_dir(model)
    if source_dir is None:
        return 0
    restored: dict = {}
    try:
        if ssm_core_dtype is not None:
            restored.update(_restore_tensors_from_source(model, source_dir, _NH_SSM_CORE_PATTERNS, ssm_core_dtype))
        if router_bias_dtype is not None:
            restored.update(
                _restore_tensors_from_source(model, source_dir, _NH_ROUTER_BIAS_PATTERNS, router_bias_dtype)
            )
    except Exception as exc:
        logger.warning("nemotron_h: high-precision override failed (%s); continuing.", exc)
        return 0
    if restored:
        logger.info("nemotron_h: restored %d tensor(s) at high precision", len(restored))
    return len(restored)
