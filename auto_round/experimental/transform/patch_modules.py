# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0
"""Backward-compat re-export shim.

The canonical implementation now lives in
:mod:`auto_round.algorithms.transforms.rotation.patch`.
"""

from auto_round.algorithms.transforms.rotation.patch import (  # noqa: F401
    patch_quantlinear,
    patch_wrapperlinear_to_apply_transform,
    patch_wrapperwalayer_forward_to_apply_transform,
)

__all__ = [
    "patch_quantlinear",
    "patch_wrapperlinear_to_apply_transform",
    "patch_wrapperwalayer_forward_to_apply_transform",
]
