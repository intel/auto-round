# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0
"""Backward-compat re-export shim.

The canonical ``RotationConfig`` schema now lives in
:mod:`auto_round.algorithms.transforms.rotation.config`.
"""

from auto_round.algorithms.transforms.rotation.config import RotationConfig  # noqa: F401

__all__ = ["RotationConfig"]
