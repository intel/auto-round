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

import torch
import torch.nn as nn


def _update_parameter(
    module: torch.nn.Module,
    name: str,
    data: torch.Tensor,
) -> None:
    old_param = getattr(module, name)
    new_param = nn.Parameter(data, requires_grad=old_param.requires_grad)
    setattr(module, name, new_param)


def is_fused_layout(original: torch.nn.Module) -> bool:
    """Check if MoE experts use fused gate_up_proj/down_proj layout."""
    return hasattr(original, "gate_up_proj") and hasattr(original, "down_proj")


def is_linearized_layout(original: torch.nn.Module) -> bool:
    """Check if MoE experts use linearized layout with individual gate_proj/up_proj/down_proj."""
    if not isinstance(original, torch.nn.ModuleList) or len(original) == 0:
        return False
    first_expert = original[0]
    return all(hasattr(first_expert, attr) for attr in ("gate_proj", "up_proj", "down_proj"))


def get_num_experts(original: torch.nn.Module) -> int:
    """Get the number of experts from either fused or linearized layout."""
    if is_fused_layout(original):
        return original.gate_up_proj.shape[0]
    if is_linearized_layout(original):
        # Count only numeric keys (expert modules), exclude 'act_fn' etc.
        if hasattr(original, "_modules"):
            numeric_keys = [k for k in original._modules.keys() if k.isdigit()]
            return len(numeric_keys)
        return len(original)
    raise AttributeError(
        "Unsupported MoE experts layout: expected fused gate_up_proj/down_proj "
        "or linearized gate_proj/up_proj/down_proj experts"
    )
