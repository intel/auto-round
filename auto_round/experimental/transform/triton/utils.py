# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

import torch


def is_triton_kernel_available() -> bool:
    """
    Best-effort check for whether Triton kernel path can be used.
    """
    try:
        import triton  # pylint: disable=E0401
    except Exception:
        return False

    if not torch.cuda.is_available():
        return False

    try:
        from auto_round.experimental.transform.triton.mxfp4 import mxfp4_forward_kernel_wrapper  # pylint: disable=E0401
    except Exception:
        return False

    return True
