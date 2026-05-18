# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""
SpinQuant in-place application sub-package.

Follows the same structure as ``auto_round.algorithms.transforms.rotation.inplace``.
"""

from auto_round.algorithms.transforms.spinquant.inplace.apply import (
    apply_spinquant_in_place,
    register_spinquant_hooks,
    remove_spinquant_hooks,
)

__all__ = [
    "apply_spinquant_in_place",
    "register_spinquant_hooks",
    "remove_spinquant_hooks",
]
