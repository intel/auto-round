# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

import torch


def is_triton_available() -> bool:
    """
    Best-effort check for whether Triton kernel path can be used.
    """
    try:
        import triton  # noqa: F401
    except Exception:
        return False

    if not torch.cuda.is_available():
        return False

    try:
        from .mxfp4 import mxfp4_forward_kernel_wrapper  # noqa: F401
    except Exception:
        return False

    return True
