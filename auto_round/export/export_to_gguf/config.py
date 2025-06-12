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

GGUF_CONFIG = {}

GGUF_CONFIG["gguf:q4_0"] = {
    "bits": 4,
    "act_bits": 16,
    "group_size": 32,
    "sym": True,
    "data_type": "int",
    "embedding": "gguf:q4_0",
    "lm_head": "gguf:q6_k_s",
    "super_bits": None,
    "super_group_size": None,
}

GGUF_CONFIG["gguf:q4_1"] = {
    "bits": 4,
    "act_bits": 16,
    "group_size": 32,
    "sym": False,
    "data_type": "int_asym_float_zp",
    "embedding": "gguf:q4_1",
    "lm_head": "gguf:q6_k_s",
    "super_bits": None,
    "super_group_size": None,
}

GGUF_CONFIG["gguf:q5_0"] = {
    "bits": 5,
    "act_bits": 16,
    "group_size": 32,
    "sym": True,
    "data_type": "int",
    "embedding": "gguf:q5_0",
    "lm_head": "gguf:q6_k_s",
    "super_bits": None,
    "super_group_size": None,
}

GGUF_CONFIG["gguf:q5_1"] = {
    "bits": 5,
    "act_bits": 16,
    "group_size": 32,
    "sym": False,
    "data_type": "int_asym_float_zp",
    "embedding": "gguf:q5_1",
    "lm_head": "gguf:q6_k_s",
    "super_bits": None,
    "super_group_size": None,
}

GGUF_CONFIG["gguf:q8_0"] = {
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

GGUF_CONFIG["gguf:q2_k_s"] = {
    "bits": 2,
    "act_bits": 16,
    "super_group_size": 16,
    "super_bits": 4,
    "group_size": 16,
    "sym": False,
    "data_type": "int_asym_dq",
    "embedding": "gguf:q2_k_s",
    "lm_head": "gguf:q6_k_s"
}

GGUF_CONFIG["gguf:q3_k_s"] = {
    "bits": 3,
    "act_bits": 16,
    "super_group_size": 16,
    "super_bits": 6,
    "group_size": 16,
    "sym": True,
    "data_type": "int_sym_dq",
    "embedding": "gguf:q3_k_s",
    "lm_head": "gguf:q6_k_s"
}

GGUF_CONFIG["gguf:q4_k_s"] = GGUF_CONFIG["gguf:q4_k_m"] = {
    "bits": 4,
    "act_bits": 16,
    "super_group_size": 8,
    "super_bits": 6,
    "group_size": 32,
    "sym": False,
    "data_type": "int_asym_dq",
    "embedding": "gguf:q4_k_s",
    "lm_head": "gguf:q6_k_s"
}

GGUF_CONFIG["gguf:q5_k_s"] = {
    "bits": 5,
    "act_bits": 16,
    "super_group_size": 8,
    "super_bits": 6,
    "group_size": 32,
    "sym": False,
    "data_type": "int_asym_dq",
    "embedding": "gguf:q5_k_s",
    "lm_head": "gguf:q6_k_s"
}

GGUF_CONFIG["gguf:q6_k"] = GGUF_CONFIG["gguf:q6_k_s"] = {
    "bits": 6,
    "act_bits": 16,
    "super_group_size": 16,
    "super_bits": 8,
    "group_size": 16,
    "sym": True,
    "data_type": "int_sym_dq",
    "embedding": "gguf:q6_k_s",
    "lm_head": "gguf:q6_k_s"
}
