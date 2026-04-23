# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

"""Unified entry point for Hadamard rotation/transform.

Two backend implementations exist:

* ``inplace``  – :mod:`auto_round.algorithms.transforms.rotation.inplace`
    QuaRot-style residual-stream rotation. Works for any weight/activation
    dtype. Optionally fuses the online Hadamard into weights
    (``fuse_online_to_weight=True``).
* ``transform`` – :mod:`auto_round.experimental.transform`
    Per-Linear weight + activation Hadamard with a fused triton kernel.
    Only supports MXFP4 / NVFP4 and **cannot** fuse online to weight.

Routing is controlled by :class:`RotationConfig.backend`:

    "inplace"    -> always inplace
    "transform"  -> always transform (validates dtype + no-fuse)
    "auto"       -> if user asked to fuse              -> inplace
                    elif data_type is mx_fp / nv_fp    -> transform
                    else                                -> inplace
"""

from __future__ import annotations

from typing import Any, Union

import torch

import auto_round.envs as envs
from auto_round.algorithms.transforms.rotation.config import RotationConfig, normalize_rotation_config
from auto_round.compressors.utils import is_mx_fp, is_nv_fp
from auto_round.utils import logger

__all__ = ["apply_hadamard_rotation", "resolve_hadamard_backend"]


def _to_config(
    rotation_config: Union[str, dict, RotationConfig, None],
    data_type: str,
) -> RotationConfig:
    """Normalise *rotation_config* and return a :class:`RotationConfig` instance."""
    cfg_dict = normalize_rotation_config(rotation_config, data_type)
    if isinstance(cfg_dict, RotationConfig):
        return cfg_dict
    return RotationConfig.model_validate(cfg_dict or {})


def resolve_hadamard_backend(config: RotationConfig, data_type: str) -> str:
    """Resolve the actual backend (``"inplace"`` / ``"transform"``) from config."""
    requested = config.backend
    fuse_requested = bool(config.fuse_online_to_weight)
    allow_online_rotation: bool = config.allow_online_rotation

    if requested == "inplace":
        return "inplace"

    transform_backend_name = "transform"
    if requested == "transform":
        if fuse_requested:
            raise ValueError(
                f"backend='{transform_backend_name}' does not support fuse_online_to_weight=True. "
                "Use backend='inplace' (or backend='auto' with fuse_online_to_weight=True) instead."
            )
        if not (is_mx_fp(data_type) or is_nv_fp(data_type)):
            raise ValueError(
                f"backend='{transform_backend_name}' only supports MXFP4 / NVFP4 (got data_type={data_type!r}). "
                "Use backend='inplace' or backend='auto' for other dtypes."
            )
        if not allow_online_rotation:
            raise ValueError(f"backend='{transform_backend_name}' only supports `allow_online_rotation`=True")

        return "transform"

    # backend == "auto"
    if fuse_requested:
        return "inplace"
    if is_mx_fp(data_type) or is_nv_fp(data_type):
        return "transform"
    return "inplace"


def apply_hadamard_rotation(
    model: torch.nn.Module,
    rotation_config: Union[str, dict, RotationConfig, None],
    data_type: str,
    compute_device: torch.device | str = None,
) -> (torch.nn.Module, Any):
    """Apply Hadamard rotation/transform to *model*, dispatching by backend.

    Args:
        model: Target model.
        rotation_config: ``str`` / ``dict`` / :class:`RotationConfig` / ``None``.
            See :class:`RotationConfig` for fields.
        data_type: Quantization data type (e.g. ``"mx_fp"``, ``"nv_fp"``,
            ``"int"``, ``"fp"``).
        compute_device: Device for inplace-backend computation. Ignored by
            the transform backend.

    Returns:
        The same model (for chaining); also stored on ``model.rotation_config``.
    """
    config = _to_config(rotation_config, data_type)
    backend = resolve_hadamard_backend(config, data_type)

    # Resolve fuse flag: explicit > env var > default(True)
    fuse_online_to_weight = config.fuse_online_to_weight
    if config.fuse_online_to_weight is not None:
        fuse_online_to_weight = bool(config.fuse_online_to_weight)
    elif envs.AR_FUSE_ONLINE_ROTATION:
        fuse_online_to_weight = bool(envs.AR_FUSE_ONLINE_ROTATION)

    logger.info(
        f"Applying Hadamard (backend={backend}, "
        f"data_type={data_type}, fuse_online_to_weight={fuse_online_to_weight if backend == 'inplace' else False})."
    )

    if backend == "inplace":
        logger.warning("this backend does not support real exporting, please export the model to fake format")
        from auto_round.algorithms.transforms.rotation.inplace import apply_rotation_transform

        # block_size -> group_size (None / -1 / 0 means full-dimension)
        bs = config.block_size
        group_size = bs if (bs is not None and bs > 0) else None

        model, hooks = apply_rotation_transform(
            model,
            group_size=group_size,
            allow_online_rotation=config.allow_online_rotation,
            rotation_matrix=config.hadamard_type,
            fuse_online_to_weight=fuse_online_to_weight,
            compute_device=compute_device,
        )
        # Stash for downstream (export / serialization). Plain dict so JSON
        # serialization (HF save_pretrained -> config.json) round-trips.
        setattr(model, "rotation_config", config.model_dump() if hasattr(config, "model_dump") else config)
        return model, hooks

    elif backend == "transform":
        supported_hadamard_types = ("hadamard", "random_hadamard")
        if config.hadamard_type not in supported_hadamard_types:
            raise ValueError("this backend only supports hadamard or random_hadamard")
        from auto_round.algorithms.transforms.rotation.apply import apply_rotation_transform

        return apply_rotation_transform(model, config, data_type=data_type)
    else:
        raise ValueError(f"Unsupported Hadamard backend {backend!r}")
