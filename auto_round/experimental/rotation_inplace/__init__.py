# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0
"""Backward-compat re-export shim.

The canonical implementation now lives in
:mod:`auto_round.algorithms.transforms.rotation.inplace`.
"""

from auto_round.algorithms.transforms.rotation.inplace.apply import (  # noqa: F401
    apply_rotation_transform,
)
from auto_round.algorithms.transforms.rotation.inplace.hooks import (  # noqa: F401
    clear_random_hadamard_cache,
)
