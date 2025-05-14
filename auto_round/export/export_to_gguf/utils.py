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

QK_K = 256
K_SCALE_SIZE = 12
GGML_QUANT_SIZES = {
    "bf16": (1, 2),
    "q4_0": (32, 2 + 16),
    "q4_1": (32, 2 + 2 + 16),
    "q5_0": (32, 2 + 4 + 16),
    "q5_1": (32, 2 + 2 + 4 + 16),
    "q8_0": (32, 2 + 32),
    "q2_k": (256, 2 + 2 + QK_K//16 + QK_K//4),
    "q3_k": (256, 2 + QK_K // 4 + QK_K // 8 + 12),
    "q4_k": (256, 2 + 2 + QK_K//2 + 12),
    "q5_k": (256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),
    "q6_k": (256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16),
    "q8_k": (256, 4 + QK_K + QK_K // 8)
}