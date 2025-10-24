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
import importlib

import torch
import transformers
from packaging import version

from auto_round.export.export_to_gguf.config import GGUF_CONFIG


def compare_versions(v1, v2):
    return version.parse(v1) >= version.parse(v2)


def torch_version_at_least(version_string):
    return compare_versions(torch.__version__, version_string)


TORCH_VERSION_AT_LEAST_2_6_PRE_RELEASE = torch_version_at_least("2.5.99")
TORCH_VERSION_AT_LEAST_2_6 = torch_version_at_least("2.6.0")
TORCH_VERSION_AT_LEAST_2_5 = torch_version_at_least("2.5.0")
TORCH_VERSION_AT_LEAST_2_4 = torch_version_at_least("2.4.0")


class SupportedFormats:

    def __init__(self):
        self._support_format = (
            "auto_round",
            "auto_gptq",
            "auto_awq",
            "auto_round:auto_gptq",
            "auto_round:gptqmodel",
            "auto_round:auto_awq",
            "auto_round:llm_compressor",
            "itrex",
            "itrex_xpu",
            "fake",
            "llm_compressor",
        )
        self._gguf_format = tuple(sorted(GGUF_CONFIG.keys()))
        self._support_list = self._support_format + self._gguf_format

    def __contains__(self, key):
        return True if key in self._support_list else False

    def __str__(self):
        # Return "(%s)" % ', '.join(self._support_format + ("gguf:q*_0", "gguf:q*_1", "gguf:q*_k_s"))
        return "(%s)" % ", ".join(self._support_list)

    def __getitem__(self, key):
        return self._support_list[key]


SHARED_CACHE_KEYS = ("position_ids", "cache_position", "position_embeddings")

deepspeed_exists = False
if importlib.util.find_spec("deepspeed"):  # check if deepspeed is installed
    deepspeed_exists = True

SUPPORTED_DTYPES = ("int", "mx_fp", "fp", "nv_fp")
SUPPORTED_FORMATS = SupportedFormats()
SUPPORTED_LAYER_TYPES = (torch.nn.Linear, transformers.pytorch_utils.Conv1D)
# Changed to str as it relies on triton or others lib to load this
INNER_SUPPORTED_LAYER_TYPES = ("FP8Linear",)
# transformers.integrations.finegrained_fp8.FP8Linear
if deepspeed_exists:
    from deepspeed.module_inject import LinearAllreduce, LinearLayer

    SUPPORTED_LAYER_TYPES = SUPPORTED_LAYER_TYPES + (LinearLayer, LinearAllreduce)
