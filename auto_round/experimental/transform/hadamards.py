# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0
"""Backward-compat re-export shim.

The canonical implementation now lives in
:mod:`auto_round.algorithms.transforms.rotation.transforms`.
"""

from auto_round.algorithms.transforms.rotation.transforms import (  # noqa: F401
    HADAMARDS,
    HadamardTransform,
    RandomHadamardTransform,
    _filter_kwargs as filter_kwarg_dict,
    build_hadamard_transform,
)

__all__ = [
    "HADAMARDS",
    "HadamardTransform",
    "RandomHadamardTransform",
    "build_hadamard_transform",
]
