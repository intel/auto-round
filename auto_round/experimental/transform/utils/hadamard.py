# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0
"""Backward-compat re-export shim.

The canonical implementation now lives in
:mod:`auto_round.algorithms.transforms.rotation.utils.math`.
"""

from auto_round.algorithms.transforms.rotation.utils.math import (  # noqa: F401
    _fetch_hadamard_divisor,
    _HADAMARD_MATRICES_PATH as REPO_PATH,
    _matmul_hadU,
    deterministic_hadamard_matrix,
    is_pow2,
    random_hadamard_matrix,
)

__all__ = ["random_hadamard_matrix", "deterministic_hadamard_matrix", "is_pow2"]
