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
"""Rotation/transform configuration (canonical, unified).

This module is the **single source of truth** for the ``RotationConfig``
schema.  The legacy location
``auto_round.experimental.transform.rotation_config`` re-exports from here.

Two implementation backends share this one schema (method B):

* ``backend="inplace"``  – QuaRot-style residual-stream rotation, implemented
  under :mod:`auto_round.experimental.rotation_inplace`.  Works for any
  weight/activation dtype and can optionally fuse the online Hadamard into
  weights (``fuse_online_to_weight=True``).

* ``backend="transform"`` – Per-Linear weight + activation Hadamard with a
  fused triton kernel, implemented under
  :mod:`auto_round.algorithms.transforms.rotation.apply`.  Supports only
  MXFP4 / NVFP4 and cannot fuse online to weight.

* ``backend="auto"`` – dispatcher picks inplace when a fused online rotation
  is requested, transform when the data_type is MX/NV-FP, inplace otherwise.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from auto_round.algorithms.transforms.base import BaseRotationConfig
from auto_round.compressors.utils import is_mx_fp, is_nv_fp
from auto_round.utils import logger

__all__ = [
    "RotationConfig",
    "normalize_rotation_config",
    "to_dict_rotation_config",
    "dump_group_size_to_rotation_config",
]


# Supported Hadamard transform types (also used by HadamardTransform registry).
HADAMARD_TYPES: frozenset[str] = frozenset({"hadamard", "random_hadamard", "quarot_hadamard"})
_SUPPORTED_BACKENDS: frozenset[str] = frozenset({"auto", "inplace", "transform"})


class RotationConfig(BaseModel, BaseRotationConfig):
    """Unified configuration for Hadamard rotation/transform applied to a model.

    See the module docstring for a description of the three backends.

    Notes:
        * ``block_size`` is the group/block size for grouped Hadamard.
          For ``backend="inplace"`` it is forwarded as ``group_size``
          (``None`` / ``-1`` means full-dimension Hadamard).
    """

    # Registry key consumed by BaseRotation.from_config (kept for API parity
    # with other BaseRotationConfig subclasses).
    algorithm: str = Field(default="hadamard", frozen=True)

    # ---- shared ----
    backend: str = Field(default="auto")
    block_size: Optional[int] = Field(default=None)
    hadamard_type: str = Field(default="hadamard")

    # ---- inplace-only ----
    fuse_online_to_weight: Optional[bool] = Field(default=None)
    allow_online_rotation: bool = Field(default=True)

    # for random hadamard (transform path)
    random_seed: bool = Field(default=False, exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("backend")
    @classmethod
    def _validate_backend(cls, v: str) -> str:
        if v not in _SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend: {v}. Supported values: {sorted(_SUPPORTED_BACKENDS)}")
        return v

    @field_validator("hadamard_type")
    @classmethod
    def _validate_hadamard_type(cls, v: str) -> str:
        if v not in HADAMARD_TYPES:
            raise ValueError(f"Unsupported hadamard_type: {v!r}. Supported values: {sorted(HADAMARD_TYPES)}")
        return v


# ---------------------------------------------------------------------------
# Helpers (free functions – match the old experimental/utils.py API)
# ---------------------------------------------------------------------------


def to_dict_rotation_config(rotation_config: str | dict | RotationConfig | None) -> dict[str, Any]:
    """Convert any supported config form to a plain ``dict`` (no data-type logic).

    Accepts:
        * ``None``            → ``{}``
        * :class:`RotationConfig` → ``model_dump()``
        * ``dict``            → shallow-copied
        * ``str``             → ``{"hadamard_type": key}`` (``"default"`` ⇒ plain default)
    """
    if rotation_config is None:
        return {}
    if isinstance(rotation_config, str):
        key = rotation_config.strip()
        if not key:
            return {}
        if key == "default":
            return {"hadamard_type": "hadamard"}
        return {"hadamard_type": key}
    if isinstance(rotation_config, RotationConfig):
        return rotation_config.model_dump()
    return dict(rotation_config)


def dump_group_size_to_rotation_config(rotation_config: str | dict | RotationConfig, group_size: int) -> dict[str, Any]:
    """Return *rotation_config* as a dict with ``block_size`` populated from *group_size* (if unset)."""
    rotation_dict = to_dict_rotation_config(rotation_config)
    if rotation_dict.get("block_size", None) is None:
        rotation_dict["block_size"] = group_size
    return rotation_dict


def normalize_rotation_config(
    rotation_config: str | dict | RotationConfig | None,
    data_type: str = "mx_fp",
) -> dict[str, Any]:
    """Normalise *rotation_config* to a validated ``dict`` ready for ``RotationConfig(**)``.

    Behaviour:
        * ``None`` → ``{}``
        * If ``block_size`` is not set:
            - ``mx_fp`` → default 32
            - ``nv_fp`` → default 16
            - other data types → emit a warning (no default)
        * If ``block_size`` mismatches the data-type recommendation, emit a warning.

    Raises:
        ValueError: If the resulting config is invalid.
    """

    def _apply_data_type_block_size(cfg_dict: dict[str, Any], block_size_explicitly_set: bool) -> dict[str, Any]:
        block_size = cfg_dict.get("block_size")

        if not block_size_explicitly_set or block_size is None:
            if is_mx_fp(data_type):
                cfg_dict["block_size"] = 32
            elif is_nv_fp(data_type):
                cfg_dict["block_size"] = 16
                logger.warning("block_size is not set for data_type 'nv_fp'; defaulting to 16.")
            else:
                logger.warning(
                    f"block_size is not set and cannot be inferred for data_type {data_type!r}; "
                    "please set block_size explicitly in rotation_config if needed."
                )
        else:
            if is_mx_fp(data_type) and block_size != 32:
                logger.warning(f"data_type is 'mx_fp' but block_size={block_size}; recommended value is 32.")
            elif is_nv_fp(data_type) and block_size != 16:
                logger.warning(f"data_type is 'nv_fp' but block_size={block_size}; recommended value is 16.")

        return cfg_dict

    if rotation_config is None:
        return {}

    rotation_dict = to_dict_rotation_config(rotation_config)
    block_size_explicitly_set = "block_size" in rotation_dict
    cfg_dict = _apply_data_type_block_size(rotation_dict, block_size_explicitly_set)
    try:
        return RotationConfig.model_validate(cfg_dict).model_dump()
    except Exception as exc:
        raise ValueError(f"Invalid RotationConfig: {exc}") from exc
