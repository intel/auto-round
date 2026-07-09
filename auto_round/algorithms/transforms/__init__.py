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
"""Weight/activation rotation algorithm package.

This package houses all *pre-quantisation rotation/transform* algorithms –
mathematical operations applied to model weights or activations before the
quantisation step to improve numerical properties.

Current algorithms
------------------
* **hadamard** – Block-diagonal Hadamard rotations (QuaRot / SpinQuant style).
  See :mod:`auto_round.algorithms.transforms.hadamard`.
* **spinquant** – SpinQuant/QuaRot multi-level rotation (R1–R4) with optional
  online hooks, trainable rotations, and known Hadamard matrices for non-pow2.
  See :mod:`auto_round.algorithms.transforms.spinquant`.

Adding a new algorithm
-----------------------
1. Create ``algorithms/transforms/<name>/`` with ``config.py`` and ``apply.py``.
2. Subclass :class:`BaseRotationConfig` and :class:`BaseRotation`; register
   with ``@BaseRotation.register("<name>")``.
3. Re-export from this ``__init__.py``.

Typical usage
-------------
>>> from auto_round.algorithms.transforms import apply_rotation
>>> model = apply_rotation(model, config={"hadamard_type": "random_hadamard"})
"""

from __future__ import annotations

from typing import Any

import torch

from auto_round.algorithms.transforms.base import (
    BaseWeightTransformer,
    BaseRotation,
    BaseRotationConfig,
    SerializerMixin,
    ROTATION_SUPPORTED_SCHEMES,
    check_supported_schemes,
    _ensure_registry_populated,
)
from auto_round.algorithms.transforms.hadamard import (
    HadamardRotation,
    apply_rotation_transform,
    normalize_rotation_config as _normalize_hadamard_config,
    RotationConfig,
)

__all__ = [
    # Base interfaces
    "BaseWeightTransformer",
    "BaseRotation",
    "BaseRotationConfig",
    "SerializerMixin",
    "ROTATION_SUPPORTED_SCHEMES",
    "check_supported_schemes",
    # Config
    "RotationConfig",
    "HadamardRotation",
    "apply_rotation_transform",
    # Unified entry — preprocessing
    "apply_rotation",
    "normalize_rotation_config",
    # Unified entry — serialization (generic dispatch)
    "inject_rotation_buffers_on_layer",
    "inject_rotation_buffers_bulk",
    "save_rotation_config",
    "preregister_rotation_buffers",
    "rebuild_rotation_if_needed",
    "apply_rotation_hooks_from_config",
]


def __getattr__(name):
    if name == "AWQConfig":
        from auto_round.algorithms.transforms.awq.config import AWQConfig

        return AWQConfig
    if name == "AWQTransform":
        from auto_round.algorithms.transforms.awq.base import AWQTransform

        return AWQTransform
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def normalize_rotation_config(
    config: Any,
) -> BaseRotationConfig | None:
    """Normalise any supported config form to the canonical :class:`BaseRotationConfig` subclass.

    Dispatches by inspecting the ``algorithm`` field (or missing field for
    legacy dicts that only carry Hadamard keys).

    Args:
        config: One of: ``None``, :class:`RotationConfig`, a ``dict`` with
                an ``"algorithm"`` key, or a plain Hadamard shorthand string
                (including ``"quarot"`` / ``"spinquant"``).

    Returns:
        The appropriate :class:`BaseRotationConfig` subclass, or ``None``
        when *config* is ``None`` / empty.
    """
    if config is None:
        return None

    if isinstance(config, BaseRotationConfig):
        return config

    if isinstance(config, dict):
        alg = config.get("algorithm", "hadamard")
        if alg == "hadamard":
            return RotationConfig.model_validate(config)
        if alg == "spinquant":
            from auto_round.algorithms.transforms.spinquant.preprocessor import SpinQuantConfig

            # Filter to only valid SpinQuantConfig fields
            import dataclasses

            valid_fields = {f.name for f in dataclasses.fields(SpinQuantConfig)}
            filtered = {k: v for k, v in config.items() if k != "algorithm" and k in valid_fields}
            return SpinQuantConfig(**filtered)
        raise ValueError(
            f"Unknown rotation algorithm: {alg!r}. " f"Registered algorithms: {sorted(BaseRotation._REGISTRY)}"
        )

    if isinstance(config, str):
        key = config.strip().lower()
        if key in ("spinquant", "quarot"):
            from auto_round.algorithms.transforms.spinquant.preprocessor import SpinQuantConfig

            # "quarot" → QuaRot defaults (fixed Hadamard, no training)
            if key == "quarot":
                return SpinQuantConfig(trainable_rotation=False, trainable_smooth=False)
            # "spinquant" → experimental trainable mode
            return SpinQuantConfig(trainable_rotation=True, trainable_smooth=True)
        # Otherwise treat as Hadamard config.
        return RotationConfig.model_validate(_normalize_hadamard_config(config))

    raise TypeError(
        f"Unsupported rotation config type: {type(config).__name__}. "
        "Expected None, dict, str, or a BaseRotationConfig subclass."
    )


def apply_rotation(
    model: torch.nn.Module,
    config: Any,
    data_type: str = "mx_fp",
    **kwargs: Any,
) -> torch.nn.Module:
    """Apply a rotation/transform algorithm to *model*.

    This is the single, algorithm-agnostic entry point.  The correct
    :class:`BaseRotation` subclass is selected automatically from *config*.

    Args:
        model:            Model to transform (modified in-place).
        config:           Rotation configuration.  Accepts:

                          * ``None`` – no-op, returns *model* unmodified.
                          * :class:`RotationConfig` or compatible ``dict``/``str``.
                          * Any :class:`BaseRotationConfig` subclass.

        data_type:        Quantization data type (e.g. ``"mx_fp"``).
        **kwargs:         Forwarded to :meth:`BaseRotation.apply_to_model`.

    Returns:
        The transformed model.
    """
    if config is None:
        return model

    normalised = normalize_rotation_config(config)
    if normalised is None:
        return model

    rotation = BaseRotation.from_config(normalised)
    return rotation.apply_to_model(model, data_type=data_type, **kwargs)


# ---------------------------------------------------------------------------
# Serialization dispatch — generic, algorithm-agnostic entry points
# ---------------------------------------------------------------------------

import logging as _logging

_dispatch_logger = _logging.getLogger("autoround.transforms.dispatch")


def _get_serializer(model: torch.nn.Module) -> SerializerMixin | None:
    """Get the :class:`SerializerMixin` for the model's rotation config.

    Returns ``None`` if no rotation was applied or the rotation algorithm
    does not support serialization.
    """
    config = getattr(model, "_rotation_config", None)
    if config is None:
        config = getattr(model, "_spinquant_config", None)  # legacy
    if config is None:
        return None
    try:
        rotation = BaseRotation.from_config(config)
    except (ValueError, KeyError):
        return None
    if not isinstance(rotation, SerializerMixin):
        return None
    return rotation


def _get_serializer_for_config(
    config_dict: dict,
) -> SerializerMixin | None:
    """Get :class:`SerializerMixin` from a JSON config dict (load side).

    The dict should have an ``"algorithm"`` field; defaults to
    ``"spinquant"`` for backward compatibility.
    """
    algorithm = config_dict.get("algorithm", "spinquant")
    _ensure_registry_populated()
    rotation_cls = BaseRotation._REGISTRY.get(algorithm)
    if rotation_cls is None:
        return None
    # Construct a minimal config for dispatch only.
    config = BaseRotationConfig(algorithm=algorithm)
    try:
        rotation = rotation_cls(config)
    except Exception:
        return None
    if not isinstance(rotation, SerializerMixin):
        return None
    return rotation


# ── Save side (called by export files) ────────────────────────────────


def inject_rotation_buffers_on_layer(
    layer_name: str,
    qlayer: torch.nn.Module,
    model: torch.nn.Module,
) -> None:
    """Per-layer rotation buffer injection (ShardWriter path).

    Dispatches to the correct :class:`SerializerMixin` based on
    ``model._rotation_config``.  No-op if no rotation config is present.

    Called from ``pack_layer()`` in all export files.
    """
    serializer = _get_serializer(model)
    if serializer is not None:
        serializer.inject_buffers_on_layer(layer_name, qlayer, model)


def inject_rotation_buffers_bulk(
    model: torch.nn.Module,
    quantization_config: dict,
) -> None:
    """Bulk rotation buffer injection (non-ShardWriter path).

    Dispatches to the correct :class:`SerializerMixin` based on
    ``model._rotation_config``.  No-op if no rotation config is present.

    Called from ``save_quantized_as_*()`` in all export files.
    """
    serializer = _get_serializer(model)
    if serializer is not None:
        serializer.inject_buffers_bulk(model, quantization_config)


def save_rotation_config(
    model: torch.nn.Module,
    save_dir: str,
) -> None:
    """Persist rotation config to ``config.json``.

    Dispatches to the correct :class:`SerializerMixin` based on
    ``model._rotation_config``.  No-op if no rotation config is present.

    Called from ``save_quantized_as_*()`` in all export files, after
    ``model.save_pretrained()``.
    """
    serializer = _get_serializer(model)
    if serializer is not None:
        serializer.save_config(model, save_dir)


# ── Load side (called by convert_model.py) ────────────────────────────


def preregister_rotation_buffers(
    model: torch.nn.Module,
    quantization_config,
) -> int:
    """Pre-register empty rotation buffers before state_dict loading.

    Extracts the rotation config dict from *quantization_config* by
    checking each registered :class:`SerializerMixin`'s
    :meth:`config_key` (e.g. ``"spinquant_config"``).

    Note: does NOT use ``"rotation_config"`` — that key is reserved for
    the existing Hadamard rotation system with a different schema.

    Returns the number of modules that received buffers, or 0 if
    no rotation config was found.
    """
    # Extract rotation config dict by checking registered serializer keys
    rotation_cfg = None
    _ensure_registry_populated()

    if isinstance(quantization_config, dict):
        for _name, _cls in BaseRotation._REGISTRY.items():
            if not (isinstance(_cls, type) and issubclass(_cls, SerializerMixin)):
                continue
            try:
                legacy_key = _cls.config_key()
            except Exception:
                continue
            rotation_cfg = quantization_config.get(legacy_key)
            if rotation_cfg is not None:
                rotation_cfg = dict(rotation_cfg)
                rotation_cfg.setdefault("algorithm", _name)
                break
    else:
        # Object-style quantization_config (e.g. QuantizationConfig)
        # Check each registered serializer's config_key as attribute
        for _name, _cls in BaseRotation._REGISTRY.items():
            if not (isinstance(_cls, type) and issubclass(_cls, SerializerMixin)):
                continue
            try:
                legacy_key = _cls.config_key()
            except Exception:
                continue
            rotation_cfg = getattr(quantization_config, legacy_key, None)
            if rotation_cfg is not None:
                if isinstance(rotation_cfg, dict):
                    rotation_cfg = dict(rotation_cfg)
                    rotation_cfg.setdefault("algorithm", _name)
                break

    if not rotation_cfg:
        return 0

    serializer = _get_serializer_for_config(rotation_cfg)
    if serializer is None:
        return 0
    return serializer.preregister_buffers(model, rotation_cfg)


def rebuild_rotation_if_needed(model: torch.nn.Module) -> None:
    """Rebuild online rotation hooks after weights are loaded.

    Scans all registered :class:`SerializerMixin` implementations to
    find one whose buffers are present on the model, then calls its
    :meth:`rebuild_online`.
    """
    _ensure_registry_populated()

    for name, rotation_cls in BaseRotation._REGISTRY.items():
        try:
            temp = rotation_cls(BaseRotationConfig(algorithm=name))
        except Exception:
            continue
        if not isinstance(temp, SerializerMixin):
            continue

        # Quick scan: does any module have this method's buffers?
        found = False
        for _, module in model.named_modules():
            if temp.has_rotation_buffers(module):
                found = True
                break
        if found:
            try:
                temp.rebuild_online(model)
            except Exception as e:
                _dispatch_logger.warning(f"Failed to rebuild {name} rotations: {e}")
            return  # Only one rotation method expected per model


def apply_rotation_hooks_from_config(
    model: torch.nn.Module,
    quantization_config,
) -> torch.nn.Module:
    """Apply rotation forward hooks at inference time based on saved config.

    Unified entry point that handles both Hadamard rotation (via
    ``rotation_config`` key) and SerializerMixin-based methods like
    SpinQuant (via ``preregister_rotation_buffers``).

    Called from ``convert_hf_model()`` before weight loading.

    Args:
        model:                The model being loaded.
        quantization_config:  The quantization config (dict or object) read
                              from ``config.json``.

    Returns:
        The model with rotation hooks applied.
    """
    from auto_round.utils import logger

    logger.warning_once(
        "Rotation transform is still in experimental stage and uses forward hooks for inference, "
        "the inference speed might be slow."
    )

    # --- Hadamard rotation (rotation_config key) ---
    rotation_config = (
        quantization_config.get("rotation_config", None)
        if isinstance(quantization_config, dict)
        else getattr(quantization_config, "rotation_config", None)
    )
    if rotation_config:
        data_type = (
            quantization_config.get("data_type", "mx_fp")
            if isinstance(quantization_config, dict)
            else getattr(quantization_config, "data_type", "mx_fp")
        )
        from auto_round.algorithms.transforms.hadamard.apply import apply_rotation_transform
        from auto_round.algorithms.transforms.hadamard.config import RotationConfig as _RC

        cfg = _RC(
            block_size=rotation_config["block_size"],
            hadamard_type=rotation_config["hadamard_type"],
        )
        model = apply_rotation_transform(
            model,
            cfg,
            location="input",
            desc="Register pre forward hook for hadamard transform",
            data_type=data_type,
        )

    # --- SerializerMixin-based methods (SpinQuant / QuaRot / future) ---
    try:
        preregister_rotation_buffers(model, quantization_config)
    except Exception as e:
        _dispatch_logger.warning(f"Failed to pre-register rotation buffers: {e}")

    return model
