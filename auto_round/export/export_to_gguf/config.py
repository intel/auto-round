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
import copy
from enum import IntEnum

from auto_round.schemes import GGUF_PRESET_ALIASES, GGUF_SCHEME_FACTS


class ModelType(IntEnum):
    TEXT = 1
    MMPROJ = 2


_GGUF_FORMAT_FIELDS = {
    "gguf:q4_0": {"embedding": "gguf:q4_0", "lm_head": "gguf:q6_k"},
    "gguf:q4_1": {"embedding": "gguf:q4_1", "lm_head": "gguf:q6_k"},
    "gguf:q5_0": {"embedding": "gguf:q5_0", "lm_head": "gguf:q6_k"},
    "gguf:q5_1": {"embedding": "gguf:q5_1", "lm_head": "gguf:q6_k"},
    "gguf:q8_0": {"embedding": "gguf:q8_0", "lm_head": "gguf:q8_0"},
    "gguf:q2_k": {"embedding": "gguf:q2_k", "lm_head": "gguf:q6_k"},
    "gguf:q3_k": {"embedding": "gguf:q3_k", "lm_head": "gguf:q6_k"},
    "gguf:q4_k": {"embedding": "gguf:q4_k", "lm_head": "gguf:q6_k"},
    "gguf:q5_k": {"embedding": "gguf:q5_k", "lm_head": "gguf:q6_k"},
    "gguf:q6_k": {"embedding": "gguf:q6_k", "lm_head": "gguf:q6_k"},
    "gguf:bf16": {"embedding": None, "lm_head": None},
}
_GGUF_FORMAT_FIELDS["gguf:fp16"] = _GGUF_FORMAT_FIELDS["gguf:bf16"]

GGUF_INNER_CONFIG = {}
for key, facts in GGUF_SCHEME_FACTS.items():
    GGUF_INNER_CONFIG[key] = {**copy.deepcopy(facts), **_GGUF_FORMAT_FIELDS[key]}


GGUF_CONFIG = {}
for alias_key, facts_key in GGUF_PRESET_ALIASES.items():
    GGUF_CONFIG[alias_key] = GGUF_INNER_CONFIG[facts_key]
    if alias_key != "gguf:q2_k_mixed":
        GGUF_CONFIG[alias_key]["mostly"] = facts_key


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
