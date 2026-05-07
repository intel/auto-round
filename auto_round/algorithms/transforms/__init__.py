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
  See :mod:`auto_round.algorithms.transforms.rotation`.

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
    BaseRotation,
    BaseRotationConfig,
    ROTATION_SUPPORTED_SCHEMES,
    check_supported_schemes,
)
from auto_round.algorithms.transforms.rotation import (
    HadamardRotation,
    apply_rotation_transform,
    normalize_rotation_config as _normalize_hadamard_config,
    RotationConfig,
)

__all__ = [
    # Base interfaces
    "BaseRotation",
    "BaseRotationConfig",
    "ROTATION_SUPPORTED_SCHEMES",
    "check_supported_schemes",
    # Config
    "RotationConfig",
    "HadamardRotation",
    "apply_rotation_transform",
    # Unified entry
    "apply_rotation",
    "normalize_rotation_config",
]


def normalize_rotation_config(
    config: Any,
) -> BaseRotationConfig | None:
    """Normalise any supported config form to the canonical :class:`BaseRotationConfig` subclass.

    Dispatches by inspecting the ``algorithm`` field (or missing field for
    legacy dicts that only carry Hadamard keys).

    Args:
        config: One of: ``None``, :class:`RotationConfig`, a ``dict`` with
                an ``"algorithm"`` key, or a plain Hadamard shorthand string.

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
        raise ValueError(
            f"Unknown rotation algorithm: {alg!r}. " f"Registered algorithms: {sorted(BaseRotation._REGISTRY)}"
        )

    if isinstance(config, str):
        # String shorthand → treat as Hadamard config.
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
