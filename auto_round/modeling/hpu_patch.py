# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

from auto_round.logger import logger


def patch_finegrained_fp8():
    """Use importlib to replace transformers.integrations.finegrained_fp8 with auto-round's HPU-compatible version."""
    try:
        from auto_round.utils.hpu_utils import is_hpu_available

        if not is_hpu_available():
            return  # No patching needed on non-HPU devices

        import importlib
        import sys

        # Import auto-round's HPU-compatible finegrained_fp8_patch module
        finegrained_fp8_patch = importlib.import_module("auto_round.modeling.finegrained_fp8_patch")

        # Replace transformers.integrations.finegrained_fp8 in sys.modules
        sys.modules["transformers.integrations.finegrained_fp8"] = finegrained_fp8_patch

        logger.info(
            "✓ Replaced transformers.integrations.finegrained_fp8 with auto_round.modeling.finegrained_fp8_patch"
        )

    except Exception as e:
        import warnings

        logger.warning(f"Failed to patch finegrained_fp8: {e}")


# Apply patch on import if HPU is available
patch_finegrained_fp8()
