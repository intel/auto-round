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
import re

from auto_round.utils import SUPPORTED_LAYER_TYPES


def _expand_regex_config(regex_config, base_config, layer_names, model):
    """
    Expand regex-based layer configs to full layer names.

    Args:
        regex_config (dict): regex-based config (dynamic_config or part of extra_config)
        base_config (dict): extra_config to write into
        layer_names (list): known quantization layer names
        model (nn.Module): target model

    Returns:
        dict: expanded base_config
    """
    if not regex_config:
        return base_config

    # Collect all supported layer names in model
    all_supported_layer_names = [n for n, m in model.named_modules() if isinstance(m, SUPPORTED_LAYER_TYPES)]

    # Identify which keys are regex patterns (not exact layer names)
    regex_keys = [k for k in regex_config.keys() if k not in all_supported_layer_names]

    for regex_key in regex_keys:
        try:
            pattern = re.compile(regex_key)
        except re.error:
            # invalid regex, skip silently
            continue

        # Prefer matches within layer_names first
        matched_layers = [ln for ln in layer_names if re.search(pattern, ln)]
        if not matched_layers:
            matched_layers = [ln for ln in all_supported_layer_names if re.search(pattern, ln)]

        if matched_layers:
            cfg = regex_config[regex_key]
            if cfg == {}:
                continue
            for ln in matched_layers:
                # do not overwrite explicit layer config
                if ln not in base_config:
                    base_config[ln] = cfg

    return base_config
