# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0
"""Inplace (QuaRot-style) Hadamard rotation backend.

Canonical home of the residual-stream Hadamard rotation implementation.
"""

from auto_round.algorithms.transforms.rotation.inplace.apply import apply_rotation_transform  # noqa: F401
from auto_round.algorithms.transforms.rotation.inplace.hooks import clear_random_hadamard_cache  # noqa: F401

__all__ = ["apply_rotation_transform", "clear_random_hadamard_cache"]
