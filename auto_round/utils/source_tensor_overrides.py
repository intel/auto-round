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

"""Reload selected tensors from a source checkpoint at a higher dtype.

Background
----------
``transformers.AutoModelForCausalLM.from_pretrained(..., torch_dtype=bf16)``
casts every floating-point parameter and buffer to BF16 on load.  For most
weights that is fine, but a few small tensors carry disproportionate
numerical weight:

* SSM recurrence parameters (``A_log``, ``D``, ``dt_bias``) — applied via
  ``exp`` and accumulated into the SSM state across the entire sequence.
  BF16 rounding here compounds.
* Mixer ``conv1d.weight`` — small mix kernel feeding the SSM input
  projections; cross-channel precision matters.
* Routed-MoE ``e_score_correction_bias`` — DeepSeek-V3-style load-balance
  bias; its values dominate the router additive term, so the small
  per-element BF16 rounding can flip top-k expert choice.

This module restores those tensors **in place** from the on-disk
checkpoint at a chosen higher dtype (typically FP32).  It runs after
``from_pretrained`` and before AutoRound's quantization pipeline begins.

Footprint is microscopic — for Nemotron-Cascade-2-30B-A3B the full
high-precision override of all four families adds up to well under a
megabyte across 52 layers.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn

from auto_round.logger import logger


def restore_tensors_from_source(
    model: nn.Module,
    source_dir: str,
    name_patterns: Iterable[str],
    target_dtype: torch.dtype,
    source_to_module: Callable[[str], str] | None = None,
) -> dict[str, tuple[torch.dtype, torch.dtype]]:
    """Reload tensors from ``source_dir`` matching *name_patterns* and replace
    the corresponding in-memory parameters/buffers with the given dtype.

    Args:
        model: Loaded ``nn.Module``.  Modified in place.
        source_dir: Path to the original checkpoint directory containing a
            ``model.safetensors`` (single-shard) or
            ``model.safetensors.index.json`` (sharded).
        name_patterns: Iterable of regex strings.  Match is via ``re.search``
            on the **source** tensor key name (e.g. ``backbone.layers.0.mixer.A_log``).
        target_dtype: Dtype to cast the source tensor to before re-binding.
        source_to_module: Optional callable mapping a source key name to the
            in-memory parameter path.  Used to undo any rename rules the
            checkpoint loader applied (e.g. for Nemotron-H,
            ``backbone.→model.`` and ``embedding.weight→embeddings.weight``).
            Defaults to identity.

    Returns:
        Mapping of restored in-memory parameter path → ``(old_dtype, new_dtype)``.
        Missing source files or unmatched keys are logged and skipped.
    """

    try:
        from safetensors import safe_open
    except ImportError:
        logger.warning("safetensors not available, cannot restore tensors from source")
        return {}

    if not os.path.isdir(source_dir):
        logger.warning("source_dir %s is not a directory; skipping tensor restore", source_dir)
        return {}

    # ------------------------------------------------------------------ #
    # Build source-tensor -> shard-file map
    # ------------------------------------------------------------------ #
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
        logger.warning("no safetensors checkpoint under %s; skipping tensor restore", source_dir)
        return {}

    compiled = [re.compile(p) for p in name_patterns]
    if not compiled:
        return {}

    matched: list[tuple[str, str]] = []
    for k, shard in source_to_shard.items():
        if any(c.search(k) for c in compiled):
            matched.append((k, shard))

    if not matched:
        logger.info(
            "restore_tensors_from_source: no source keys matched patterns %s — nothing to do",
            list(name_patterns),
        )
        return {}

    if source_to_module is None:
        source_to_module = lambda s: s  # noqa: E731

    # Group reads by shard to avoid re-opening shard files.
    by_shard: dict[str, list[str]] = {}
    for k, shard in matched:
        by_shard.setdefault(shard, []).append(k)

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
                if parent is None:
                    logger.warning(
                        "restore: in-memory module %r (from source key %r) not found — skipping",
                        parent_path,
                        src_key,
                    )
                    continue
                if not hasattr(parent, attr_name):
                    logger.warning(
                        "restore: %s has no attribute %r (from source key %r) — skipping",
                        parent_path,
                        attr_name,
                        src_key,
                    )
                    continue

                old = getattr(parent, attr_name)
                if not isinstance(old, torch.Tensor):
                    logger.warning(
                        "restore: %s.%s is not a Tensor (got %s) — skipping",
                        parent_path,
                        attr_name,
                        type(old).__name__,
                    )
                    continue
                if old.shape != src_tensor.shape:
                    logger.warning(
                        "restore: shape mismatch for %s — model has %s, source has %s; skipping",
                        module_path,
                        list(old.shape),
                        list(src_tensor.shape),
                    )
                    continue

                new_tensor = src_tensor.to(target_dtype)
                old_dtype = old.dtype

                if isinstance(old, nn.Parameter):
                    requires_grad = old.requires_grad
                    setattr(parent, attr_name, nn.Parameter(new_tensor, requires_grad=requires_grad))
                else:
                    # Re-register so PyTorch keeps it tracked as a buffer; preserve
                    # ``persistent`` flag if available (PyTorch >= 1.6).
                    persistent = True
                    pers_set = getattr(parent, "_non_persistent_buffers_set", set())
                    if attr_name in pers_set:
                        persistent = False
                    parent.register_buffer(attr_name, new_tensor, persistent=persistent)

                restored[module_path] = (old_dtype, target_dtype)

    if restored:
        # Compress for log readability.
        from collections import Counter

        dtype_counts = Counter((str(o), str(n)) for o, n in restored.values())
        summary = ", ".join(f"{n} tensors {old}->{new}" for (old, new), n in dtype_counts.items())
        logger.info("restore_tensors_from_source: restored %d tensors (%s)", len(restored), summary)
    return restored


def _resolve_module(root: nn.Module, dotted_path: str) -> nn.Module | None:
    if not dotted_path:
        return root
    mod: nn.Module | None = root
    for part in dotted_path.split("."):
        if mod is None:
            return None
        if part.isdigit():
            idx = int(part)
            try:
                mod = mod[idx]  # type: ignore[index]
            except (IndexError, TypeError, KeyError):
                return None
        else:
            mod = getattr(mod, part, None)
    return mod


# ---------------------------------------------------------------------- #
# Convenience helpers for known architectures
# ---------------------------------------------------------------------- #

# Source-key regexes for Nemotron-H high-precision-sensitive tensors.
#
# NOTE: ``mixer.conv1d.weight`` is intentionally NOT included.  Unlike the
# three scalar/per-head params below — which are applied via ``exp`` and
# broadcast-style ops where PyTorch naturally promotes mixed dtypes —
# ``F.conv1d`` requires ``input.dtype == weight.dtype``.  Upcasting
# ``conv1d.weight`` to FP32 while the input stream stays BF16 raises
# ``RuntimeError: type XPUBFloat16Type does not equal XPUFloatType`` at
# the first forward pass.  Handling it cleanly would require either
# patching the Mamba2 forward to cast inputs, or moving to BF16 weights
# — the footprint saving (<500 KB total) does not justify either path.
NEMOTRON_H_SSM_CORE_PATTERNS: tuple[str, ...] = (
    r"\.mixer\.A_log$",
    r"\.mixer\.D$",
    r"\.mixer\.dt_bias$",
)

NEMOTRON_H_ROUTER_BIAS_PATTERNS: tuple[str, ...] = (
    r"\.mixer\.gate\.e_score_correction_bias$",
)


def _nemotron_h_source_to_module(src_key: str) -> str:
    """Apply the Nemotron-H source→in-memory rename rules used by the
    transformers checkpoint loader.

    Mirrors:
      * ``backbone.`` → ``model.``
      * ``embedding.weight`` → ``embeddings.weight``
    """
    if src_key.startswith("backbone."):
        src_key = "model." + src_key[len("backbone.") :]
    if src_key.endswith(".embedding.weight"):
        src_key = src_key[: -len(".embedding.weight")] + ".embeddings.weight"
    return src_key


def restore_nemotron_h_high_precision(
    model: nn.Module,
    source_dir: str,
    *,
    ssm_core_dtype: torch.dtype | None = torch.float32,
    router_bias_dtype: torch.dtype | None = torch.float32,
) -> dict[str, tuple[torch.dtype, torch.dtype]]:
    """Bundle of ``restore_tensors_from_source`` calls covering
    Nemotron-H's high-precision-sensitive tensor families.

    Pass ``None`` for any dtype to skip that family.
    """
    restored: dict[str, tuple[torch.dtype, torch.dtype]] = {}
    if ssm_core_dtype is not None:
        restored.update(
            restore_tensors_from_source(
                model,
                source_dir,
                NEMOTRON_H_SSM_CORE_PATTERNS,
                ssm_core_dtype,
                source_to_module=_nemotron_h_source_to_module,
            )
        )
    if router_bias_dtype is not None:
        restored.update(
            restore_tensors_from_source(
                model,
                source_dir,
                NEMOTRON_H_ROUTER_BIAS_PATTERNS,
                router_bias_dtype,
                source_to_module=_nemotron_h_source_to_module,
            )
        )
    return restored
