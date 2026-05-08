# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0
"""Backward-compat re-export shim.

The canonical implementation now lives in
:mod:`auto_round.algorithms.transforms.rotation.apply`.
"""

from auto_round.algorithms.transforms.rotation.apply import (  # noqa: F401
    apply_rotation_transform,
)

__all__ = ["apply_rotation_transform"]
