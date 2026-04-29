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

        # Patch the upstream transformers module in-place rather than replacing it
        # entirely. Replacing the whole module via sys.modules drops other public
        # symbols that newer transformers (e.g. >=4.57) expects to import from
        # `transformers.integrations.finegrained_fp8` (such as
        # `ALL_FP8_EXPERTS_FUNCTIONS`, `FP8Experts`, ...), causing ImportError.
        try:
            upstream = importlib.import_module("transformers.integrations.finegrained_fp8")
        except Exception as import_err:  # pragma: no cover - defensive
            # Fallback to legacy behavior if the upstream module cannot be imported.
            sys.modules["transformers.integrations.finegrained_fp8"] = finegrained_fp8_patch
            logger.warning(
                "Failed to import upstream transformers.integrations.finegrained_fp8"
                f" ({import_err}); falling back to full module replacement."
            )
            return

        patched_names = []
        for name in dir(finegrained_fp8_patch):
            if name.startswith("_"):
                continue
            setattr(upstream, name, getattr(finegrained_fp8_patch, name))
            patched_names.append(name)

        logger.info(
            "✓ Patched transformers.integrations.finegrained_fp8 with HPU-compatible"
            f" overrides from {patch_file_name} ({len(patched_names)} symbols)"
        )
        logger.debug(
            "Patched symbols for transformers.integrations.finegrained_fp8 from " f"{patch_file_name}: {patched_names}"
        )

    except Exception as e:
        import warnings

        logger.warning(f"Failed to patch finegrained_fp8: {e}")


# Apply patch on import if HPU is available
patch_finegrained_fp8()
