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

from typing import Dict, List

from auto_round.utils import matches_any_regex, to_standard_regex


def generate_ignore_regex_list(dynamic_config: Dict[str, Dict], layer_config: Dict[str, Dict]) -> List[str]:
    """
    Generate ignore regex list for llm_compressor based on dynamic_config and layer_config.

    Rules:
    1. Any layer in dynamic_config with bits >= 16 is ignored.
    2. Any layer in layer_config with bits >= 16 is ignored if not already included.
    3. Output regex patterns are normalized for llm_compressor ('re:...' style).

    Args:
        dynamic_config (Dict[str, Dict]): dynamic quantization config
        layer_config (Dict[str, Dict]): layer-wise quantization config

    Returns:
        List[str]: List of regex patterns to ignore during quantization.
    """
    prefix = "re:"
    ignore_regex: List[str] = []

    # Step 1: Add dynamic_config keys with bits >= 16
    for key, cfg in dynamic_config.items():
        bits = cfg.get("bits")
        if bits > 8:
            ignore_regex.append(prefix + to_standard_regex(key))

    # Step 2: Add layer_config keys if bits >= 16 and not already included
    for key, cfg in layer_config.items():
        bits = cfg.get("bits")

        if not matches_any_regex(key, ignore_regex, prefix):
            ignore_regex.append(key)

    return ignore_regex
