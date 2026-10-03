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


def _make_solve_triangular_wrapper(orig):
    """Build a drop-in replacement for ``torch.linalg.solve_triangular`` that
    probes for a native kernel at runtime.

    Some ``(device, dtype)`` combinations have no native triangular-solve
    kernel: the call falls back to the generic CPU implementation, which
    raises ``NotImplementedError`` (e.g. bf16, observed on HPU).  Instead of
    hard-coding a list of broken devices, the wrapper tries the original
    dtype first and, on ``NotImplementedError`` only, solves in fp32 and
    casts the result back to ``torch.result_type(A, B)``.  The outcome is
    cached per ``(device.type, A.dtype, B.dtype)`` so the probe cost is paid
    exactly once per combination.
    """
    import torch

    float32 = torch.float32
    result_type = torch.result_type
    support_cache = {}

    def solve_triangular_compat(A, B, **kwargs):
        # A caller-provided out= tensor must be written in its own dtype; the
        # fp32 detour cannot satisfy it, so pass through untouched.
        if kwargs.get("out") is not None:
            return orig(A, B, **kwargs)

        key = (A.device.type, A.dtype, B.dtype)
        if support_cache.get(key) is not False:
            try:
                result = orig(A, B, **kwargs)
            except NotImplementedError:
                if key not in support_cache:
                    support_cache[key] = False
                    logger.info(
                        "torch.linalg.solve_triangular has no native kernel for"
                        f" device '{key[0]}' A.dtype={key[1]} B.dtype={key[2]};"
                        " falling back to fp32 computation."
                    )
            else:
                support_cache.setdefault(key, True)
                return result
        # No native kernel: solve in fp32, then cast back to the promoted dtype.
        result = orig(A.to(float32), B.to(float32), **kwargs)
        return result.to(result_type(A, B))

    solve_triangular_compat._ar_solve_triangular_patched = True
    solve_triangular_compat._ar_orig_solve_triangular = orig
    solve_triangular_compat._ar_support_cache = support_cache
    return solve_triangular_compat


def patch_solve_triangular():
    """Install the runtime-probed fp32 fallback for ``torch.linalg.solve_triangular``.

    ``transformers`` calls this API at runtime (e.g. Qwen3.5 linear-attention
    state cleanup), so patching the torch entry point covers all callers and
    survives transformers upgrades.  The wrapper is only installed on hosts
    where HPEX is available; once installed it is process-wide, which also
    keeps CPU fallback calls made from HPU code paths correct.
    """
    try:
        from auto_round.utils import is_hpex_available

        if not is_hpex_available():
            return  # No patching needed on non-HPU devices

        import torch

        current = torch.linalg.solve_triangular
        if getattr(current, "_ar_solve_triangular_patched", False):
            return  # already patched

        torch.linalg.solve_triangular = _make_solve_triangular_wrapper(current)
        logger.info(
            "✓ Patched torch.linalg.solve_triangular with a runtime-probed fp32 fallback"
            " for (device, dtype) pairs lacking a native kernel."
        )
    except Exception as e:
        logger.warning(f"Failed to patch torch.linalg.solve_triangular: {e}")


# Apply patches on import if HPU is available
patch_finegrained_fp8()
patch_solve_triangular()
