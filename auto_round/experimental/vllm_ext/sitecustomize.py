# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    from .auto_round_vllm_extension import apply as auto_round_ext_apply

    auto_round_ext_apply()
else:
    print("*****************************************************************************")
    print(
        f"* Sitecustomize is loaded, but VLLM_ENABLE_AR_EXT is set to {VLLM_ENABLE_AR_EXT}, skipping auto_round_vllm_extension *"
    )
    print("*****************************************************************************")
