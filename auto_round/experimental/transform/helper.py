# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch

from auto_round.experimental.transform.hadamard_config import HadamardConfig
from auto_round.experimental.transform.hadamards import HADAMARDS


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


def normalize_hadamard_config(hadamard_config: Any) -> dict[str, Any]:
    """
    Normalize and validate `hadamard_config`.

    Supported input types:
        - None          -> {}
        - dict          -> validated via HadamardConfig
        - HadamardConfig -> validated & converted to dict
        - str           -> shorthand for `transform_type` in TRANSFORMS keys

    On any validation failure, raises ValueError/TypeError.
    """
    # 1) None -> {}
    if hadamard_config is None:
        return {}

    # 2) Already a HadamardConfig instance
    if isinstance(hadamard_config, HadamardConfig):
        # Ensure it passes its own validation and convert to dict
        cfg = HadamardConfig.model_validate(hadamard_config).model_dump()
        return cfg

    # 3) dict -> validate via HadamardConfig
    if isinstance(hadamard_config, dict):
        try:
            cfg = HadamardConfig.model_validate(hadamard_config).model_dump()
        except Exception as e:
            raise ValueError(f"Invalid hadamard_config dict: {e}") from e
        return cfg

    # 4) str -> shorthand for transform_type
    if isinstance(hadamard_config, str):
        key = hadamard_config.strip()
        if not key:
            return {}

        if key not in HADAMARDS:
            raise ValueError(
                f"Invalid hadamard_config string: {key!r}. " f"Expected one of {sorted(HADAMARDS.keys())}."
            )

        cfg_dict = {"transform_type": key}

        try:
            cfg = HadamardConfig.model_validate(cfg_dict).model_dump()
        except Exception as e:
            raise ValueError(f"hadamard_config built from string {key!r} is invalid for HadamardConfig: {e}") from e

        return cfg

    raise TypeError(
        "hadamard_config must be one of: None, dict, HadamardConfig, or str " f"(got {type(hadamard_config).__name__})"
    )
