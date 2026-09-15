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

"""Projection grouping for SDXL BasicTransformerBlock SVDQuant."""

from __future__ import annotations

import torch

from auto_round.algorithms.transforms.svdquant.smooth_adapters.base import (
    SmoothSearchGroup,
    TargetPredicate,
    generic_linear_groups,
    module_global_name,
    resolve_module,
)

_SELF_ATTENTION_QKV = ("attn1.to_q", "attn1.to_k", "attn1.to_v")


def supports_sdxl_block(block: torch.nn.Module) -> bool:
    return block.__class__.__name__ == "BasicTransformerBlock"


def discover_sdxl_groups(block: torch.nn.Module, is_target: TargetPredicate) -> list[SmoothSearchGroup]:
    selected = []
    for path in _SELF_ATTENTION_QKV:
        module = resolve_module(block, path)
        if isinstance(module, torch.nn.Linear) and is_target(path, module):
            selected.append((path, module))

    groups = []
    consumed = set()
    if len(selected) == len(_SELF_ATTENTION_QKV):
        attention = resolve_module(block, "attn1")
        if attention is None:
            raise ValueError(f"SDXL SVDQuant group {module_global_name(block, 'attn1.qkv')!r} has no attention module")
        names = tuple(module_global_name(block, path) for path, _ in selected)
        projections = tuple(module for _, module in selected)
        groups.append(
            SmoothSearchGroup(
                key=module_global_name(block, "attn1.qkv"),
                projection_names=names,
                projections=projections,
                projection_input_key=names[0],
                projection_input_module=projections[0],
                evaluation_input_key=module_global_name(block, "attn1"),
                evaluation_module=attention,
            )
        )
        consumed.update(id(module) for module in projections)
    groups.extend(generic_linear_groups(block, is_target, consumed=consumed))
    return groups


__all__ = ["discover_sdxl_groups", "supports_sdxl_block"]
