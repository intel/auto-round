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

"""Shared-input QKV grouping for vLLM-Omni's fused SVDQuant linear layers."""

import torch

from auto_round.algorithms.transforms.svdquant.smooth_adapters.base import (
    SmoothSearchGroup,
    TargetPredicate,
    generic_linear_groups,
    module_global_name,
)

_QKV_NAMES = (("to_q", "to_k", "to_v"), ("q_proj", "k_proj", "v_proj"))


def discover_omni_groups(block: torch.nn.Module, is_target: TargetPredicate) -> list[SmoothSearchGroup]:
    """Group explicit self-attention QKV without model-class-specific layouts."""
    groups = []
    consumed = set()
    for local_name, attention in block.named_modules():
        if getattr(attention, "is_cross_attention", None) is not False:
            continue
        projection_names = next((names for names in _QKV_NAMES if all(hasattr(attention, name) for name in names)), ())
        if not projection_names:
            continue
        paths = tuple(f"{local_name}.{name}" if local_name else name for name in projection_names)
        projections = tuple(getattr(attention, name) for name in projection_names)
        if not all(
            isinstance(module, torch.nn.Linear) and id(module) not in consumed and is_target(path, module)
            for path, module in zip(paths, projections)
        ):
            continue
        if len({module.in_features for module in projections}) != 1 or len({id(module) for module in projections}) != 3:
            continue
        names = tuple(module_global_name(block, path) for path in paths)
        groups.append(
            SmoothSearchGroup(
                key=module_global_name(block, f"{local_name}.qkv" if local_name else "qkv"),
                projection_names=names,
                projections=projections,
                projection_input_key=names[0],
                projection_input_module=projections[0],
                evaluation_input_key=module_global_name(block, local_name),
                evaluation_module=attention,
            )
        )
        consumed.update(id(module) for module in projections)
    groups.extend(generic_linear_groups(block, is_target, consumed=consumed))
    return groups
