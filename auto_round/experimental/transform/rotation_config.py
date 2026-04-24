# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

from typing import Optional

from pydantic import BaseModel, Field, field_validator

__all__ = ["RotationConfig"]


class RotationConfig(BaseModel):
    """
    Unified configuration for Hadamard rotation/transform applied to a model.

    Two implementation paths are supported:

    * ``backend="inplace"``  -> ``auto_round.experimental.rotation_inplace``
        QuaRot-style residual-stream / per-layer rotation. Supports any
        weight/activation dtype (incl. INT4/INT8/FPx). Can optionally fuse
        the online Hadamard into weights (``fuse_online_to_weight=True``).
    * ``backend="transform"`` -> ``auto_round.experimental.transform``
        Per-Linear weight + activation Hadamard with a fused triton kernel.
        **Only supports MXFP4 / NVFP4** and **cannot fuse online to weight.**
    * ``backend="auto"`` (default)
        - If ``fuse_online_to_weight=True`` -> inplace (fused).
        - Else if ``data_type`` is MX-FP / NV-FP -> transform.
        - Otherwise -> inplace (unfused).

    Notes:
        * ``block_size`` is the group/block size for grouped Hadamard.
          For ``backend="inplace"`` it is forwarded as ``group_size`` (``None``
          / ``-1`` means full-dimension Hadamard).
    """

    # ---- shared ----
    backend: str = Field(default="auto")
    block_size: Optional[int] = Field(default=None)
    hadamard_type: str = Field(default="hadamard")

    # ---- inplace-only ----
    fuse_online_to_weight: Optional[bool] = Field(default=None)
    allow_online_rotation: bool = Field(default=True)

    # for random hadamard transform (transform path)
    random_seed: bool = Field(default=False, exclude=True)

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        allowed = {"auto", "inplace", "transform"}
        if v not in allowed:
            raise ValueError(f"Unsupported backend: {v}. Supported values: {sorted(allowed)}")
        return v

    @field_validator("hadamard_type")
    @classmethod
    def validate_hadamard_type(cls, v: str) -> str:
        allowed = {"hadamard", "random_hadamard", "quarot_hadamard"}
        if v not in allowed:
            raise ValueError(f"Unsupported hadamard_type: {v}. Supported values: {sorted(allowed)}")
        return v
