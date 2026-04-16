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
from auto_round.compressors.utils import is_mx_fp, is_nv_fp
from auto_round.utils import logger

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
    data_type: str = "mx_fp",
) -> dict[str, Any]:
    """Normalise various input forms to a canonical ``dict`` for :class:`HadamardConfig`.

    Args:
        config: One of:

            * ``None``            → returns ``{}``
            * ``dict``            → validated via :class:`HadamardConfig`
            * :class:`HadamardConfig` → converted to ``dict``
            * ``str`` shorthand  → treated as ``hadamard_type``
              (``"default"`` → default :class:`HadamardConfig`)
        data_type: Quantization data type. Used to infer ``block_size``
            when not explicitly set (mx_fp → 32, nv_fp → 16).

    Returns:
        A validated ``dict`` that can be passed to ``HadamardConfig(**result)``.

    Raises:
        ValueError: If the config is invalid.
        TypeError:  If the config type is not recognised.
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
                    "please set block_size explicitly in hadamard_config if needed."
                )
        else:
            if is_mx_fp(data_type) and block_size != 32:
                logger.warning(f"data_type is 'mx_fp' but block_size={block_size}; recommended value is 32.")
            elif is_nv_fp(data_type) and block_size != 16:
                logger.warning(f"data_type is 'nv_fp' but block_size={block_size}; recommended value is 16.")

        return cfg_dict

    if config is None:
        return {}

    if isinstance(config, HadamardConfig):
        raw_cfg_dict = config.model_dump(exclude_unset=True)
        block_size_explicitly_set = "block_size" in raw_cfg_dict
        cfg_dict = dict(raw_cfg_dict)
        cfg_dict = _apply_data_type_block_size(cfg_dict, block_size_explicitly_set)
        try:
            return HadamardConfig.model_validate(cfg_dict).model_dump()
        except Exception as exc:
            raise ValueError(f"Invalid HadamardConfig: {exc}") from exc

    if isinstance(config, dict):
        block_size_explicitly_set = "block_size" in config
        cfg_dict = dict(config)
        cfg_dict = _apply_data_type_block_size(cfg_dict, block_size_explicitly_set)
        try:
            return HadamardConfig.model_validate(cfg_dict).model_dump()
        except Exception as exc:
            raise ValueError(f"Invalid HadamardConfig dict: {exc}") from exc

    if isinstance(config, str):
        key = config.strip()
        if not key:
            return {}
        if key == "default":
            cfg_dict = {}
            cfg_dict = _apply_data_type_block_size(cfg_dict, block_size_explicitly_set=False)
            try:
                return HadamardConfig.model_validate(cfg_dict).model_dump()
            except Exception as exc:
                raise ValueError(f"Invalid default hadamard_config after data_type adjustment: {exc}") from exc
        if key not in HADAMARD_TYPES:
            raise ValueError(
                f"Unrecognised hadamard config string: {key!r}. "
                f"Expected one of {sorted(HADAMARD_TYPES)} or 'default'."
            )
        cfg_dict = {"hadamard_type": key}
        cfg_dict = _apply_data_type_block_size(cfg_dict, block_size_explicitly_set=False)
        try:
            return HadamardConfig.model_validate(cfg_dict).model_dump()
        except Exception as exc:
            raise ValueError(f"Failed to build HadamardConfig from {key!r}: {exc}") from exc

    raise TypeError("hadamard_config must be None, dict, HadamardConfig, or str " f"(got {type(config).__name__})")
