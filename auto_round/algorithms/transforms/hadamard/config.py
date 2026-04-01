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
"""Hadamard rotation algorithm configuration."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from auto_round.algorithms.transforms.base import BaseRotationConfig

__all__ = ["HadamardConfig", "normalize_hadamard_config"]

# Supported Hadamard transform types (also used by HadamardTransform registry).
HADAMARD_TYPES: frozenset[str] = frozenset({"hadamard", "random_hadamard"})


class HadamardConfig(BaseModel, BaseRotationConfig):
    """Configuration for Hadamard rotation transforms.

    This config is designed to be embedded inside a model's ``config.json``
    for serialisation, and is also used at runtime to drive
    :class:`~auto_round.algorithms.transforms.hadamard.apply.HadamardRotation`.

    Attributes:
        algorithm: Fixed to ``"hadamard"`` – identifies this config in the
            :class:`~auto_round.algorithms.transforms.base.BaseRotation` registry.
        block_size: Block size for the block-diagonal Hadamard matrix.
        hadamard_type: Which transform to use (``"hadamard"`` or
            ``"random_hadamard"``).
        random_seed: For ``"random_hadamard"`` – seed the generator for
            reproducibility.  Excluded from serialisation (``exclude=True``)
            because it is a calibration-time detail.
    """

    # Override BaseRotationConfig.algorithm with a literal default.
    algorithm: str = Field(default="hadamard", frozen=True)
    block_size: int = Field(default=32)
    hadamard_type: str = Field(default="hadamard")
    random_seed: bool = Field(default=False, exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("hadamard_type")
    @classmethod
    def _validate_hadamard_type(cls, v: str) -> str:
        if v not in HADAMARD_TYPES:
            raise ValueError(f"Unsupported hadamard_type: {v!r}. " f"Supported values: {sorted(HADAMARD_TYPES)}")
        return v


def normalize_hadamard_config(
    config: str | dict | HadamardConfig | None,
) -> dict[str, Any]:
    """Normalise various input forms to a canonical ``dict`` for :class:`HadamardConfig`.

    Args:
        config: One of:

            * ``None``            → returns ``{}``
            * ``dict``            → validated via :class:`HadamardConfig`
            * :class:`HadamardConfig` → converted to ``dict``
            * ``str`` shorthand  → treated as ``hadamard_type``
              (``"default"`` → default :class:`HadamardConfig`)

    Returns:
        A validated ``dict`` that can be passed to ``HadamardConfig(**result)``.

    Raises:
        ValueError: If the config is invalid.
        TypeError:  If the config type is not recognised.
    """
    if config is None:
        return {}

    if isinstance(config, HadamardConfig):
        return config.model_dump()

    if isinstance(config, dict):
        try:
            return HadamardConfig.model_validate(config).model_dump()
        except Exception as exc:
            raise ValueError(f"Invalid HadamardConfig dict: {exc}") from exc

    if isinstance(config, str):
        key = config.strip()
        if not key:
            return {}
        if key == "default":
            return HadamardConfig().model_dump()
        if key not in HADAMARD_TYPES:
            raise ValueError(
                f"Unrecognised hadamard config string: {key!r}. "
                f"Expected one of {sorted(HADAMARD_TYPES)} or 'default'."
            )
        try:
            return HadamardConfig.model_validate({"hadamard_type": key}).model_dump()
        except Exception as exc:
            raise ValueError(f"Failed to build HadamardConfig from {key!r}: {exc}") from exc

    raise TypeError("hadamard_config must be None, dict, HadamardConfig, or str " f"(got {type(config).__name__})")
