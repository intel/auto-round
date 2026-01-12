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
from enum import IntEnum


class ModelType(IntEnum):
    TEXT = 1
    MMPROJ = 2


GGUF_INNER_CONFIG = {}

GGUF_INNER_CONFIG["gguf:q4_0"] = {
    "bits": 4,
    "act_bits": 16,
    "group_size": 32,
    "sym": True,
    "data_type": "int",
    "embedding": "gguf:q4_0",
    "lm_head": "gguf:q6_k",
    "super_bits": None,
    "super_group_size": None,
}

GGUF_INNER_CONFIG["gguf:q4_1"] = {
    "bits": 4,
    "act_bits": 16,
    "group_size": 32,
    "sym": False,
    "data_type": "int_asym_float_zp",
    "embedding": "gguf:q4_1",
    "lm_head": "gguf:q6_k",
    "super_bits": None,
    "super_group_size": None,
}

GGUF_INNER_CONFIG["gguf:q5_0"] = {
    "bits": 5,
    "act_bits": 16,
    "group_size": 32,
    "sym": True,
    "data_type": "int",
    "embedding": "gguf:q5_0",
    "lm_head": "gguf:q6_k",
    "super_bits": None,
    "super_group_size": None,
}

GGUF_INNER_CONFIG["gguf:q5_1"] = {
    "bits": 5,
    "act_bits": 16,
    "group_size": 32,
    "sym": False,
    "data_type": "int_asym_float_zp",
    "embedding": "gguf:q5_1",
    "lm_head": "gguf:q6_k",
    "super_bits": None,
    "super_group_size": None,
}

GGUF_INNER_CONFIG["gguf:q8_0"] = {
    "bits": 8,
    "act_bits": 16,
    "group_size": 32,
    "sym": True,
    "data_type": "int",
    "embedding": "gguf:q8_0",
    "lm_head": "gguf:q8_0",
    "super_bits": None,
    "super_group_size": None,
}

GGUF_INNER_CONFIG["gguf:q2_k"] = {
    "bits": 2,
    "act_bits": 16,
    "super_group_size": 16,
    "super_bits": 4,
    "group_size": 16,
    "sym": False,
    "data_type": "int_asym_dq",
    "embedding": "gguf:q2_k",
    "lm_head": "gguf:q6_k",
}

GGUF_INNER_CONFIG["gguf:q3_k"] = {
    "bits": 3,
    "act_bits": 16,
    "super_group_size": 16,
    "super_bits": 6,
    "group_size": 16,
    "sym": True,
    "data_type": "int_sym_dq",
    "embedding": "gguf:q3_k",
    "lm_head": "gguf:q6_k",
}

GGUF_INNER_CONFIG["gguf:q4_k"] = {
    "bits": 4,
    "act_bits": 16,
    "super_group_size": 8,
    "super_bits": 6,
    "group_size": 32,
    "sym": False,
    "data_type": "int_asym_dq",
    "embedding": "gguf:q4_k",
    "lm_head": "gguf:q6_k",
}

GGUF_INNER_CONFIG["gguf:q5_k"] = {
    "bits": 5,
    "act_bits": 16,
    "super_group_size": 8,
    "super_bits": 6,
    "group_size": 32,
    "sym": False,
    "data_type": "int_asym_dq",
    "embedding": "gguf:q5_k",
    "lm_head": "gguf:q6_k",
}

GGUF_INNER_CONFIG["gguf:q6_k"] = {
    "bits": 6,
    "act_bits": 16,
    "super_group_size": 16,
    "super_bits": 8,
    "group_size": 16,
    "sym": True,
    "data_type": "int_sym_dq",
    "embedding": "gguf:q6_k",
    "lm_head": "gguf:q6_k",
}

GGUF_INNER_CONFIG["gguf:bf16"] = GGUF_INNER_CONFIG["gguf:fp16"] = {
    "bits": 16,
    "act_bits": 16,
    "super_group_size": None,
    "super_bits": None,
    "group_size": None,
    "sym": True,
    "data_type": "int_sym_dq",
    "embedding": None,
    "lm_head": None,
}


GGUF_CONFIG = {}
GGUF_CONFIG["gguf:q4_0"] = GGUF_INNER_CONFIG["gguf:q4_0"]
GGUF_CONFIG["gguf:q4_0"]["mostly"] = "gguf:q4_0"
GGUF_CONFIG["gguf:q4_1"] = GGUF_INNER_CONFIG["gguf:q4_1"]
GGUF_CONFIG["gguf:q4_1"]["mostly"] = "gguf:q4_1"
GGUF_CONFIG["gguf:q5_0"] = GGUF_INNER_CONFIG["gguf:q5_0"]
GGUF_CONFIG["gguf:q5_0"]["mostly"] = "gguf:q5_0"
GGUF_CONFIG["gguf:q5_1"] = GGUF_INNER_CONFIG["gguf:q5_1"]
GGUF_CONFIG["gguf:q5_1"]["mostly"] = "gguf:q5_1"
GGUF_CONFIG["gguf:q2_k_s"] = GGUF_INNER_CONFIG["gguf:q2_k"]
GGUF_CONFIG["gguf:q2_k_s"]["mostly"] = "gguf:q2_k"
# GGUF_CONFIG["gguf:q3_k"] = GGUF_INNER_CONFIG["gguf:q3_k"]
# GGUF_CONFIG["gguf:q3_k"]["mostly"] = "gguf:q3_k"
GGUF_CONFIG["gguf:q3_k_s"] = GGUF_INNER_CONFIG["gguf:q3_k"]
GGUF_CONFIG["gguf:q3_k_s"]["mostly"] = "gguf:q3_k"
GGUF_CONFIG["gguf:q3_k_m"] = GGUF_INNER_CONFIG["gguf:q3_k"]
GGUF_CONFIG["gguf:q3_k_m"]["mostly"] = "gguf:q3_k"
GGUF_CONFIG["gguf:q3_k_l"] = GGUF_INNER_CONFIG["gguf:q3_k"]
GGUF_CONFIG["gguf:q3_k_l"]["mostly"] = "gguf:q3_k"
# GGUF_CONFIG["gguf:q4_k"] = GGUF_INNER_CONFIG["gguf:q4_k"]
# GGUF_CONFIG["gguf:q4_k"]["mostly"]= "gguf:q4_k"
GGUF_CONFIG["gguf:q4_k_s"] = GGUF_INNER_CONFIG["gguf:q4_k"]
GGUF_CONFIG["gguf:q4_k_s"]["mostly"] = "gguf:q4_k"
GGUF_CONFIG["gguf:q4_k_m"] = GGUF_INNER_CONFIG["gguf:q4_k"]
GGUF_CONFIG["gguf:q4_k_m"]["mostly"] = "gguf:q4_k"
# GGUF_CONFIG["gguf:q5_k"] = GGUF_INNER_CONFIG["gguf:q5_k"]
# GGUF_CONFIG["gguf:q5_k"]["mostly"]= "gguf:q5_k"
GGUF_CONFIG["gguf:q5_k_s"] = GGUF_INNER_CONFIG["gguf:q5_k"]
GGUF_CONFIG["gguf:q5_k_s"]["mostly"] = "gguf:q5_k"
GGUF_CONFIG["gguf:q5_k_m"] = GGUF_INNER_CONFIG["gguf:q5_k"]
GGUF_CONFIG["gguf:q5_k_m"]["mostly"] = "gguf:q5_k"
GGUF_CONFIG["gguf:q6_k"] = GGUF_INNER_CONFIG["gguf:q6_k"]
GGUF_CONFIG["gguf:q6_k"]["mostly"] = "gguf:q6_k"
GGUF_CONFIG["gguf:q8_0"] = GGUF_INNER_CONFIG["gguf:q8_0"]
GGUF_CONFIG["gguf:q8_0"]["mostly"] = "gguf:q8_0"
# GGUF_CONFIG["gguf:fp16"] = GGUF_INNER_CONFIG["gguf:fp16"]
# GGUF_CONFIG["gguf:fp16"]["mostly"]= "gguf:fp16"
# GGUF_CONFIG["gguf:bf16"] = GGUF_INNER_CONFIG["gguf:fp16"]
# GGUF_CONFIG["gguf:bf16"]["mostly"]= "gguf:bf16"
GGUF_CONFIG["gguf:q2_k_mixed"] = GGUF_INNER_CONFIG["gguf:q2_k"]


QK_K = 256
K_SCALE_SIZE = 12
GGML_QUANT_SIZES = {
    "bf16": (1, 2),
    "q4_0": (32, 2 + 16),
    "q4_1": (32, 2 + 2 + 16),
    "q5_0": (32, 2 + 4 + 16),
    "q5_1": (32, 2 + 2 + 4 + 16),
    "q8_0": (32, 2 + 32),
    "q2_k": (256, 2 + 2 + QK_K // 16 + QK_K // 4),
    "q3_k": (256, 2 + QK_K // 4 + QK_K // 8 + 12),
    "q4_k": (256, 2 + 2 + QK_K // 2 + 12),
    "q5_k": (256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),
    "q6_k": (256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16),
    "q8_k": (256, 4 + QK_K + QK_K // 8),
}
