# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# PYTHONPATH=/home/yliu7/workspace/inc/3rd-party/vllm/vllm/model_executor/layers/quantization/auto_round_vllm_extension/:$PYTHONPATH

import os

VLLM_ENABLE_AR_EXT = os.environ.get("VLLM_ENABLE_AR_EXT", "") in [
    "1",
    "true",
    "True",
]

if VLLM_ENABLE_AR_EXT:
    print("*****************************************************************************")
    print(f"* !!! VLLM_ENABLE_AR_EXT is set to {VLLM_ENABLE_AR_EXT}, applying auto_round_vllm_extension *")
    print("*****************************************************************************")
    from vllm.model_executor.layers.quantization import auto_round_vllm_extension as auto_round_ext

    auto_round_ext.apply()
else:
    print("*****************************************************************************")
    print(
        f"* Sitecustomize is loaded, but VLLM_ENABLE_AR_EXT is set to {VLLM_ENABLE_AR_EXT}, skipping auto_round_vllm_extension *"
    )
    print("*****************************************************************************")
