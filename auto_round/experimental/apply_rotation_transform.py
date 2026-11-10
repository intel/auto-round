# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0
"""Backward-compat re-export shim.

The canonical implementation now lives in
:mod:`auto_round.algorithms.transforms.hadamard.dispatcher`.
"""

from auto_round.algorithms.transforms.hadamard.dispatcher import (  # noqa: F401
    apply_hadamard_rotation,
    resolve_hadamard_backend,
)
