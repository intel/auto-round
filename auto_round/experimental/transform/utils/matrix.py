# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0
"""Backward-compat re-export shim.

The canonical implementation now lives in
:mod:`auto_round.algorithms.transforms.rotation.utils.matrix`.
"""

from auto_round.algorithms.transforms.rotation.utils.matrix import (  # noqa: F401
    apply_transform_weight,
    multihead_matmul,
)

# Old private name kept for backward compatibility.
_multihead_matmul = multihead_matmul

__all__ = ["apply_transform_weight"]
