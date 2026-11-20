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
"""DeepSeek V4-specific checkpoint aliases and calibration replay helpers."""

import copy
import re

import torch


def is_deepseek_v4(model):
    return getattr(getattr(model, "config", None), "model_type", None) == "deepseek_v4"


def expand_layer_config(layer_config):
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping
    from transformers.core_model_loading import WeightRenaming

    result = copy.deepcopy(layer_config)
    rules = [rule for rule in get_checkpoint_conversion_mapping("deepseek_v4") if isinstance(rule, WeightRenaming)]
    for rule in rules:
        if len(rule.source_patterns) != 1 or len(rule.target_patterns) != 1:
            raise ValueError("Expected scalar DeepSeek V4 rename rule")
    for original, config in layer_config.items():
        name = original
        for rule in rules:
            name = re.sub(rule.source_patterns[0], rule.target_patterns[0], name)
        name = re.sub(
            r"(\.experts\.\d+)\.w([123])(?=\.|$)",
            lambda match: match[1] + "." + {"1": "gate_proj", "2": "down_proj", "3": "up_proj"}[match[2]],
            name,
        )
        result.setdefault(name, copy.deepcopy(config))
    return result


def select_shared_rope(value, indices, sample_count, batch_dim):
    """Select V4's nested RoPE from per-collection-batch cache entries."""
    entries = value if isinstance(value, list) else [value]
    per_entry = max(1, -(-sample_count // len(entries)))

    def row(entry, index):
        if isinstance(entry, dict):
            return {key: row(item, index) for key, item in entry.items()}
        if isinstance(entry, tuple):
            return tuple(row(item, index) for item in entry)
        if isinstance(entry, torch.Tensor):
            if entry.shape[batch_dim] == 1:
                return entry
            return entry.narrow(batch_dim, index, 1)
        return entry

    def combine(picks):
        first = picks[0]
        if isinstance(first, dict):
            return {key: combine([pick[key] for pick in picks]) for key in first}
        if isinstance(first, tuple):
            return tuple(combine([pick[i] for pick in picks]) for i in range(len(first)))
        if isinstance(first, torch.Tensor):
            return torch.cat([pick.to(first.device) for pick in picks], dim=batch_dim)
        return first

    return combine([row(entries[i // per_entry], i % per_entry) for i in indices])
