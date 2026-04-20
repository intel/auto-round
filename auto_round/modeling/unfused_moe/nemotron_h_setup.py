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

"""Post-load, architecture-specific setup for Nemotron-H-family models.

This module centralises the work that must happen *after*
``AutoModelForCausalLM.from_pretrained`` in order for a Nemotron-H MoE
checkpoint (e.g. ``nvidia/Nemotron-Cascade-2-30B-A3B``) to be
quantized by AutoRound into a model whose decoded output is coherent.

Three concerns are handled here, all strictly gated on
``config.model_type == "nemotron_h"`` by the single public entry
:func:`apply_nemotron_h_post_load`:

1. **Zamba2RMSNormGated ``group_size`` patch** — when transformers loads
   a model with ``low_cpu_mem_usage=True`` it materialises modules via
   ``init_empty_weights`` which calls ``cls.__new__(cls)``.  That
   bypasses ``__init__``, so any attribute set in ``__init__`` (such as
   ``self.group_size`` in ``Zamba2RMSNormGated``) is missing at forward
   time and calibration crashes with ``AttributeError``.  We derive
   ``group_size = mamba_num_heads * mamba_head_dim // n_groups`` from
   the config and set it both per-instance and on the class (the latter
   as a fallback for any subsequent re-wrap).

2. **High-precision override of SSM recurrence + router bias** —
   ``torch_dtype=bf16`` at ``from_pretrained`` time downcasts every
   floating-point tensor including ``A_log``/``D``/``dt_bias``/
   ``conv1d.weight`` (used via ``exp`` and accumulated across the
   sequence) and ``e_score_correction_bias`` (dominates the router
   additive term).  The precision loss cannot be recovered at save
   time; we reload these few small tensors from the source checkpoint
   at FP32.  Total footprint is < 1 MB for a 30B-class MoE.  If the
   source checkpoint is not locally available and cannot be downloaded
   (offline env) the override is logged and skipped — calibration still
   works on the BF16 values.

3. **Default layer_config patterns** — Mamba2 ``mixer.out_proj``
   projects the SSM state (very small-magnitude values) back to the
   hidden stream; FP16 per-group scales collapse to sub-normals for
   many of its groups, making the packed weight unusable.  We promote
   scales for this layer to BF16 by default.  This is exposed via
   :func:`nemotron_h_default_layer_config_patterns` and consumed by
   ``set_layer_config``.

None of this fires for any other model type.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from auto_round.logger import logger

# Regex patterns for the default layer_config overlay.
# (Wrapped in a function so callers always get a fresh dict — these
# entries get mutated downstream by the layer_config pipeline.)
#
# The pattern is written in AutoRound's "standard regex input" form
# accepted by ``set_layer_config`` / ``to_standard_regex``: plain dots
# (not ``\.``) because that helper escapes bare dots itself, while it
# does not re-unescape ``\.`` sequences — meaning a raw-string regex
# like ``r".*\.mixer\.out_proj$"`` would end up with doubled
# backslashes and silently match nothing.
_NEMOTRON_H_OUT_PROJ_PATTERN = ".*mixer.out_proj$"


def nemotron_h_default_layer_config_patterns() -> dict[str, dict[str, Any]]:
    """Return the regex → overlay dict of default quantization-config
    overrides that apply automatically to any Nemotron-H model.

    Only ``scale_dtype`` is set — bits/group_size/sym follow the global
    default.  User-provided ``layer_config`` entries for the same
    pattern take precedence (see ``set_layer_config``).
    """

    return {
        _NEMOTRON_H_OUT_PROJ_PATTERN: {
            # BF16 per-group scales avoid FP16 sub-normal collapse on the
            # SSM output projection, which has very small-magnitude
            # weights.
            "scale_dtype": torch.bfloat16,
        },
    }


def _patch_zamba2_group_size(model: nn.Module) -> int:
    """Set ``group_size`` on every ``Zamba2RMSNormGated`` instance and
    class found in *model*.  Returns the number of instances patched.

    Safe to call unconditionally — modules of other class names are
    ignored.  If ``group_size`` is already correctly set (non-low-cpu
    path) we skip the instance but still class-patch once as a defence
    against later re-wraps.
    """

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
        # Always class-patch once — guards against accelerate/AutoRound
        # re-instantiating the class via ``cls.__new__(cls)`` which would
        # otherwise strand ``group_size`` at the Python default (missing).
        cls = module.__class__
        if cls not in patched_classes:
            cls.group_size = group_size
            patched_classes.add(cls)
        # Instance patch (use object.__setattr__ so torch.nn.Module's
        # __setattr__ doesn't reroute through _buffers/_parameters).
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


def _resolve_source_dir(model: nn.Module) -> str | None:
    """Best-effort resolution of the original checkpoint directory for
    ``restore_nemotron_h_high_precision``.  Returns ``None`` when the
    model was constructed without a ``_name_or_path`` or the snapshot
    cannot be located and downloading is not possible."""

    import os

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
    except Exception as exc:  # pragma: no cover - network / offline
        logger.info(
            "nemotron_h: snapshot_download for %r failed (%s); skipping "
            "high-precision SSM/router overrides — calibration proceeds on BF16 values.",
            name_or_path,
            exc,
        )
        return None


def _apply_high_precision_overrides(
    model: nn.Module,
    *,
    ssm_core_dtype: torch.dtype | None = torch.float32,
    router_bias_dtype: torch.dtype | None = torch.float32,
) -> int:
    """Call ``restore_nemotron_h_high_precision`` against the resolved
    source checkpoint.  Returns the number of tensors restored (0 on
    offline / failure)."""

    if ssm_core_dtype is None and router_bias_dtype is None:
        return 0
    source_dir = _resolve_source_dir(model)
    if source_dir is None:
        return 0
    try:
        from auto_round.utils.source_tensor_overrides import (
            restore_nemotron_h_high_precision,
        )

        restored = restore_nemotron_h_high_precision(
            model,
            source_dir,
            ssm_core_dtype=ssm_core_dtype,
            router_bias_dtype=router_bias_dtype,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "nemotron_h: high-precision override failed (%s); " "continuing with BF16 SSM/router tensors.",
            exc,
        )
        return 0
    if restored:
        logger.info(
            "nemotron_h: restored %d tensor(s) at high precision (ssm_core=%s, router_bias=%s)",
            len(restored),
            ssm_core_dtype,
            router_bias_dtype,
        )
    return len(restored)


def apply_nemotron_h_post_load(
    model: nn.Module,
    *,
    ssm_core_dtype: torch.dtype | None = torch.float32,
    router_bias_dtype: torch.dtype | None = torch.float32,
    enable_high_precision_overrides: bool = True,
) -> dict[str, int]:
    """Apply every Nemotron-H-specific fix-up that must happen between
    ``from_pretrained`` and AutoRound calibration.

    Safe to call on any model — a quick ``model_type`` check at the top
    makes this a no-op for non-NH models.

    Args:
        model: The loaded ``nn.Module`` (in-place modified).
        ssm_core_dtype: Dtype to restore SSM recurrence tensors (``A_log``,
            ``D``, ``dt_bias``, ``conv1d.weight``) to.  ``None`` disables.
        router_bias_dtype: Dtype to restore ``e_score_correction_bias`` to.
            ``None`` disables.
        enable_high_precision_overrides: When ``False``, skip the
            source-checkpoint reload step even if dtypes are supplied
            (useful for offline environments).

    Returns:
        Summary dict ``{"zamba2_patched": int, "high_precision_restored": int}``.
    """

    summary = {"zamba2_patched": 0, "high_precision_restored": 0}
    config = getattr(model, "config", None)
    if config is None or getattr(config, "model_type", None) != "nemotron_h":
        return summary

    # Idempotency sentinel — if a caller has already applied these
    # fix-ups (e.g. a legacy launcher script that pre-patches before
    # handing the model to AutoRound), we skip so we don't re-download
    # the source checkpoint or re-cast tensors already at FP32.
    if getattr(model, "_autoround_nh_post_load_applied", False):
        return summary

    summary["zamba2_patched"] = _patch_zamba2_group_size(model)
    if enable_high_precision_overrides:
        summary["high_precision_restored"] = _apply_high_precision_overrides(
            model,
            ssm_core_dtype=ssm_core_dtype,
            router_bias_dtype=router_bias_dtype,
        )
    try:
        setattr(model, "_autoround_nh_post_load_applied", True)
    except Exception:  # pragma: no cover - nn.Module.__setattr__ edge
        pass
    return summary
