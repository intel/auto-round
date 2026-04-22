# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

from auto_round.logger import logger


def patch_finegrained_fp8():
    """Use importlib to replace transformers.integrations.finegrained_fp8 with auto-round's HPU-compatible version."""
    try:
        from auto_round.utils import is_hpex_available

        if not is_hpex_available():
            return  # No patching needed on non-HPU devices

        import importlib
        import sys

        # Import auto-round's HPU-compatible finegrained_fp8_patch module
        from auto_round.utils import (
            is_transformers_version_greater_or_equal_4,
            is_transformers_version_greater_or_equal_5,
        )

        if is_transformers_version_greater_or_equal_5():
            patch_file_name = "auto_round.modeling.finegrained_fp8_patch"
        elif is_transformers_version_greater_or_equal_4():
            patch_file_name = "auto_round.modeling.finegrained_fp8_patch_v4"
        else:
            logger.warning(
                (
                    "Transformers version is below 4.0.0, skipping finegrained_fp8 patching.",
                    " Please upgrade to Transformers 4.x or later for HPU support.",
                )
            )
            return

        finegrained_fp8_patch = importlib.import_module(patch_file_name)

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
