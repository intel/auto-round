# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0
"""Backward-compat re-export shim.

The canonical implementation now lives in
:mod:`auto_round.algorithms.transforms.rotation.utils.triton.mxfp4`.
"""

from auto_round.algorithms.transforms.rotation.utils.triton.mxfp4 import (  # noqa: F401
    mxfp4_forward_kernel,
    mxfp4_forward_kernel_wrapper,
)

__all__ = ["mxfp4_forward_kernel", "mxfp4_forward_kernel_wrapper"]
