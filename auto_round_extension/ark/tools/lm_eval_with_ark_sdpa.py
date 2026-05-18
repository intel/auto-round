#!/usr/bin/env python

# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import runpy

from auto_round_kernel import patch_torch_sdpa


def main() -> None:
    backend = os.environ.get("ARK_TORCH_SDPA_BACKEND", "sdpa")
    quant_block_size = int(os.environ.get("ARK_SAGEV1_QUANT_BLOCK_SIZE", "64"))
    patch_torch_sdpa(
        strict=True,
        backend=backend,
        quant_block_size=quant_block_size,
    )
    runpy.run_module("lm_eval.__main__", run_name="__main__")


if __name__ == "__main__":
    main()
