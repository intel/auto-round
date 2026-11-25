# Copyright (c) 2026 Intel Corporation
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

from typing import Any

from auto_round.utils import to_standard_regex

GPTQ_REQUIRED_CONFIG_KEYS = ("bits", "group_size", "sym")

BLOCK_PATTERNS = ["transformer.h", "model.decoder.layers", "gpt_neox.layers", "model.layers"]


def convert_to_autogptq_dynamic(regex_config: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Convert AutoRound-style regex_config into AutoGPTQ-style QuantizerConfig.dynamic.

    Rules:
    - bits < 16 -> quantize -> positive match `+:regex`
    - bits == 16 -> skip quantize -> negative match `-:regex`
    """
    converted = {}
    for name, cfg in regex_config.items():
        bits = cfg.get("bits")
        regex = to_standard_regex(name)

        if bits is None:
            continue  # ignore invalid entries
        elif bits < 16:
            converted[f"+:{regex}"] = {"bits": bits}
            for key in GPTQ_REQUIRED_CONFIG_KEYS:  # only save keys gptq supported
                converted[f"+:{regex}"][key] = cfg[key]
        else:
            # skip quantization
            converted[f"-:{regex}"] = {}
    return converted


def convert_from_autogptq_dynamic(dynamic_config: dict) -> dict:
    """
    Convert AutoGPTQ-style QuantizerConfig.dynamic into AutoRound-style extra_config.

    Rules:
    - '+:regex' => quantize => keep bits and other quantization keys
    - '-:regex' => skip quantize => set bits to 16 (FP16 passthrough)
    """
    converted = {}
    for name, cfg in dynamic_config.items():
        # Strip the +: or -:
        if name.startswith("+:"):
            regex = name[2:]
            # keep all config fields (bits, group_size, sym, etc.)
            converted[regex] = dict(cfg)
        elif name.startswith("-:"):
            regex = name[2:]
            # mark skipped layers with bits=16
            converted[regex] = {"bits": 16, "act_bits": 16}
    return converted
