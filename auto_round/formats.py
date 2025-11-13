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

from __future__ import annotations

from typing import Callable, Union

from auto_round.schemes import PRESET_SCHEMES, QuantizationScheme


class OutputFormat:
    support_schemes: list = []
    _format_list: dict[str, OutputFormat] = {}

    def __init__(self, format):
        self.output_format = format.split(":")[0]
        self.backend = format.split(":")[1] if ":" in format else None

    @classmethod
    def register(cls, *names: str) -> Callable[[OutputFormat], OutputFormat]:
        assert names

        def func(output_format: OutputFormat) -> OutputFormat:
            for name in names:
                cls._format_list[name] = output_format
            return output_format

        return func

    @classmethod
    def get_support_matrix(cls: OutputFormat) -> str:
        output_str = ""
        for k, v in cls._format_list.items():
            support_scheme = ", ".join(v.support_schemes).rstrip(",")
            output_str += f"\x1b[31;1m{k}\x1b[0m support scheme:\n\t{support_scheme}\n"
        return output_str

    @classmethod
    def is_support_scheme(cls: OutputFormat, scheme: Union[str, QuantizationScheme]) -> bool:
        if scheme in cls.support_schemes:
            return True
        if isinstance(scheme, QuantizationScheme):
            for key in cls.support_schemes:
                if scheme == PRESET_SCHEMES[key]:
                    return True
        return False


@OutputFormat.register("fake")
class FakeFormat(OutputFormat):
    support_schemes = [
        "W4A16",
        "W2A16",
        "W3A16",
        "W8A16",
        "MXFP4",
        "MXFP8",
        "NVFP4",
        "FPW8A16",
        "W2A16G64",
        "W2A16G32",
        "FP8_STATIC",
        "BF16",
        "GGUF:Q4_0",
        "GGUF:Q4_1",
        "GGUF:Q5_0",
        "GGUF:Q5_1",
        "GGUF:Q2_K_S",
        "GGUF:Q3_K_S",
        "GGUF:Q3_K_M",
        "GGUF:Q3_K_L",
        "GGUF:Q4_K_S",
        "GGUF:Q4_K_M",
        "GGUF:Q5_K_S",
        "GGUF:Q5_K_M",
        "GGUF:Q6_K",
        "GGUF:Q8_0",
    ]


@OutputFormat.register("llm_compressor")
class LLMCompressorFormat(OutputFormat):
    support_schemes = ["MXFP4", "MXFP8", "NVFP4", "FPW8A16", "FP8_STATIC"]


@OutputFormat.register("auto_gptq")
class AutoGPTQFormat(OutputFormat):
    support_schemes = ["W4A16", "W2A16", "W3A16", "W8A16", "BF16", "W2A16G64", "W2A16G32"]


@OutputFormat.register("auto_awq")
class AutoAWQFormat(OutputFormat):
    support_schemes = ["W4A16", "W2A16", "W3A16", "W8A16", "BF16", "W2A16G64", "W2A16G32"]


@OutputFormat.register("itrex")
@OutputFormat.register("itrex_xpu")
class ITREXFormat(OutputFormat):
    support_schemes = ["W4A16", "W2A16", "W3A16", "W8A16", "BF16", "W2A16G64", "W2A16G32"]


@OutputFormat.register("gguf")
class GGUFFormat(OutputFormat):
    support_schemes = [
        "GGUF:Q4_0",
        "GGUF:Q4_1",
        "GGUF:Q5_0",
        "GGUF:Q5_1",
        "GGUF:Q2_K_S",
        "GGUF:Q3_K_S",
        "GGUF:Q3_K_M",
        "GGUF:Q3_K_L",
        "GGUF:Q4_K_S",
        "GGUF:Q4_K_M",
        "GGUF:Q5_K_S",
        "GGUF:Q5_K_M",
        "GGUF:Q6_K",
        "GGUF:Q8_0",
    ]


@OutputFormat.register("auto_round")
class AutoRoundFormat(OutputFormat):
    support_schemes = [
        "W4A16",
        "W2A16",
        "W3A16",
        "W8A16",
        "MXFP4",
        "MXFP8",
        "NVFP4",
        "FPW8A16",
        "W2A16G64",
        "W2A16G32",
        "FP8_STATIC",
        "BF16",
    ]

    def __init__(self, format):
        self.output_format = format.split(":")[0]
        self.backend = format.split(":")[1] if ":" in format else None

        if self.backend == "llm_compressor":
            self.support_schemes = ["MXFP4", "MXFP8", "NVFP4", "FPW8A16", "FP8_STATIC"]
        elif self.backend == "auto_gptq" or "gptqmodel":
            self.support_schemes = ["W4A16", "W2A16", "W3A16", "W8A16", "BF16", "W2A16G64", "W2A16G32"]
        elif self.backend == "auto_awq":
            self.support_schemes = ["W4A16", "W2A16", "W3A16", "W8A16", "BF16", "W2A16G64", "W2A16G32"]
