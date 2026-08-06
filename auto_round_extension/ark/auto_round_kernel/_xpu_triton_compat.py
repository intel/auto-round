# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""Workarounds for the Intel XPU triton backend on this platform.

Triton 3.7.x enables ``has_predicated_io`` for non-LTS Intel GPU drivers and then
emits the ``SPV_INTEL_predicated_io`` SPIR-V extension in its JIT kernels. Some
level-zero loaders (e.g. 1.13.35563) reject that extension at kernel-load time::

    InvalidModule: Invalid SPIR-V module: input SPIR-V module uses unknown
    extension 'SPV_INTEL_predicated_io'

which aborts any triton kernel launch and kills the process (it is not a
catchable Python exception, so the torch preprocess fallback never runs).

The workaround forces ``has_predicated_io`` off so triton emits standard
(non-extension) predicated loads/stores. This disables a minor codegen
optimization on every XPU driver, but the ark triton preprocess kernels are small
and the trade-off is only safe correctness on the affected drivers.
"""

import logging

logger = logging.getLogger(__name__)

_APPLIED = False


def apply_xpu_triton_workarounds() -> None:
    """Patch triton's Intel compiler so JIT kernels avoid the rejected SPIR-V extension."""
    global _APPLIED
    if _APPLIED:
        return
    try:
        import triton.backends.intel.compiler as _intel_compiler
    except ImportError:
        # No intel triton backend (e.g. stock triton); nothing to patch.
        return

    _orig_parse_target = _intel_compiler.XPUBackend.parse_target

    def _patched_parse_target(self, tgt_prop):
        dev_prop = _orig_parse_target(self, tgt_prop)
        dev_prop["has_predicated_io"] = False
        return dev_prop

    _intel_compiler.XPUBackend.parse_target = _patched_parse_target
    _APPLIED = True
    logger.info("Applied XPU triton workaround: has_predicated_io forced off (SPV_INTEL_predicated_io)")
