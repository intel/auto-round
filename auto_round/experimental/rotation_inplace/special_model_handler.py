# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""Registry of per-model overrides for Hadamard inplace rotation.

Some models are known to require a specific rotation configuration
(e.g. a particular ``rotation_matrix`` preset) to preserve accuracy or
even to run correctly. Hard-coding these decisions in
``apply_rotation_transform`` quickly becomes messy, so we centralize them
here as a small declarative registry.

Public API
----------

* :func:`register_special_model` – register a new override entry.
* :func:`get_special_overrides` – look up overrides for a given model.
* :func:`apply_special_overrides` – apply all matching overrides on top of
  a kwargs dict, with logging when something actually changes.

Each entry is a :class:`SpecialModelEntry` with:
  * ``name``     – short, human-readable label used in log messages.
  * ``matches``  – callable ``(model) -> bool`` that decides whether the
                   entry applies to a given model.
  * ``overrides`` – dict of keyword arguments that must be forced (e.g.
                    ``{"rotation_matrix": "random_hadamard"}``).
  * ``reason``    – short justification, surfaced in log messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from auto_round.utils import logger

__all__ = [
    "SpecialModelEntry",
    "SPECIAL_MODEL_REGISTRY",
    "register_special_model",
    "get_special_overrides",
    "apply_special_overrides",
]


# ---------------------------------------------------------------------------
# Registry datatype
# ---------------------------------------------------------------------------


@dataclass
class SpecialModelEntry:
    name: str
    matches: Callable[[Any], bool]
    overrides: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


SPECIAL_MODEL_REGISTRY: List[SpecialModelEntry] = []


def register_special_model(entry: SpecialModelEntry) -> SpecialModelEntry:
    """Append *entry* to the registry. First-registered wins on conflict."""
    SPECIAL_MODEL_REGISTRY.append(entry)
    return entry


# ---------------------------------------------------------------------------
# Match helpers
# ---------------------------------------------------------------------------


def _name_contains(model, needle: str) -> bool:
    """Return True iff a (case-insensitive) substring is found in any of the
    common model identifier attributes."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return False
    needle = needle.lower()
    for attr in ("_name_or_path", "name_or_path"):
        val = getattr(cfg, attr, None)
        if isinstance(val, str) and needle in val.lower():
            return True
    return False


def _config_matches(model, model_type: str = None, **expected) -> bool:
    """Return True iff ``model.config.<k> == v`` for every ``k=v`` in *expected*
    (and optionally a matching ``model_type``)."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return False
    if model_type is not None and getattr(cfg, "model_type", "") != model_type:
        return False
    for k, v in expected.items():
        if getattr(cfg, k, None) != v:
            return False
    return True


# ---------------------------------------------------------------------------
# Built-in entries
# ---------------------------------------------------------------------------

# ---- Example: Qwen3-14B (no longer needed, kept as reference) ----
# Qwen3-14B previously required a forced ``random_hadamard`` override because
# the o_proj weight rotation used a single full-dimension Hadamard whose
# butterfly construction violated the Kronecker decomposition assumed by the
# cross-head online hook when ``num_heads`` is not a power of 2.  This has
# been fixed in ``_rotate_weights`` — all presets now use the decomposed
# per-head + cross-head form — so the override is no longer needed.
#
# def _is_qwen3_14b(model) -> bool:
#     if _name_contains(model, "qwen3-14b"):
#         return True
#     return _config_matches(
#         model,
#         model_type="qwen3",
#         hidden_size=5120,
#         num_hidden_layers=40,
#         num_attention_heads=40,
#     )
#
# register_special_model(
#     SpecialModelEntry(
#         name="Qwen3-14B",
#         matches=_is_qwen3_14b,
#         overrides={"rotation_matrix": "random_hadamard"},
#         reason="deterministic Hadamard is known to hurt accuracy on this model",
#     )
# )


# ---------------------------------------------------------------------------
# Public lookup / application helpers
# ---------------------------------------------------------------------------


def get_special_overrides(model) -> Dict[str, Any]:
    """Return the merged override dict for *model* (empty if none match).

    Multiple matching entries are merged in registration order; later entries
    overwrite earlier ones for the same key.
    """
    merged: Dict[str, Any] = {}
    for entry in SPECIAL_MODEL_REGISTRY:
        try:
            if entry.matches(model):
                merged.update(entry.overrides)
        except Exception as e:  # never let a buggy matcher break rotation
            logger.warning("Special-model matcher %r raised %r; skipping.", entry.name, e)
    return merged


def apply_special_overrides(model, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Force *kwargs* values for *model* according to the registry.

    Returns the (possibly modified) ``kwargs`` dict. Emits a warning whenever
    a user-supplied value is overwritten and an info message when an unset
    value is filled in.
    """
    for entry in SPECIAL_MODEL_REGISTRY:
        try:
            if not entry.matches(model):
                continue
        except Exception as e:
            logger.warning("Special-model matcher %r raised %r; skipping.", entry.name, e)
            continue

        for key, forced_value in entry.overrides.items():
            current = kwargs.get(key)
            if current == forced_value or current is None:
                logger.info(
                    "Detected %s: forcing %s=%r (%s).",
                    entry.name,
                    key,
                    forced_value,
                    entry.reason or "model-specific override",
                )
            else:
                logger.warning(
                    "Detected %s: overriding %s=%r with %r (%s).",
                    entry.name,
                    key,
                    current,
                    forced_value,
                    entry.reason or "model-specific override",
                )
            kwargs[key] = forced_value
    return kwargs
