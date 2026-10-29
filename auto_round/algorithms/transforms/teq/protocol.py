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

"""Protocol for transforms that must compose with trainable TEQ scales.

Algorithms can publish matching ``weight`` and ``input`` callables on a Linear
layer.  TEQ and its downstream tuning wrapper then use the common equivalent
order: ``QDQ(weight_transform(W * alpha))`` and
``QDQ(input_transform(x / alpha))``.  This is intentionally independent from
Hadamard, allowing another equivalent transform to compose with TEQ safely.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

__all__ = [
    "TEQ_EQUIVALENT_TRANSFORMS_ATTR",
    "apply_teq_equivalent_input_transform",
    "apply_teq_equivalent_weight_transform",
    "get_teq_equivalent_transforms",
    "install_teq_equivalent_input_hook",
    "is_teq_equivalent_input_hook",
    "set_teq_equivalent_transforms",
]


TEQ_EQUIVALENT_TRANSFORMS_ATTR = "_autoround_teq_transforms"


def set_teq_equivalent_transforms(
    module: torch.nn.Module,
    *,
    weight: Callable[[torch.Tensor], torch.Tensor],
    input: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """Publish a matched transform pair without registering state-dict modules."""
    module.__dict__[TEQ_EQUIVALENT_TRANSFORMS_ATTR] = {
        "weight": weight,
        "input": input,
    }


def get_teq_equivalent_transforms(module: torch.nn.Module) -> dict:
    """Return a layer's published transform pair, or an empty mapping."""
    transforms = module.__dict__.get(TEQ_EQUIVALENT_TRANSFORMS_ATTR, {})
    return transforms if isinstance(transforms, dict) else {}


def apply_teq_equivalent_weight_transform(module: torch.nn.Module, weight: torch.Tensor) -> torch.Tensor:
    """Apply the published weight-side transform, when the layer has one."""
    transform = get_teq_equivalent_transforms(module).get("weight")
    return transform(weight) if transform is not None else weight


def apply_teq_equivalent_input_transform(module: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
    """Apply the published input-side transform, when the layer has one."""
    transform = get_teq_equivalent_transforms(module).get("input")
    return transform(value) if transform is not None else value


def is_teq_equivalent_input_hook(hook) -> bool:
    """Return whether ``hook`` is the committed input transform hook below."""
    return bool(getattr(hook, "_autoround_teq_equivalent_input_hook", False))


def install_teq_equivalent_input_hook(module: torch.nn.Module) -> None:
    """Install the input-side transform after the rotated weight is committed.

    The transform backend intentionally keeps a FP model unchanged until its
    weight reaches a quantization wrapper.  Once that wrapper commits the
    rotated QDQ weight, regular block forwards (including quantized-input
    propagation to the next block) must apply the matching activation-side
    transform too.  Hooks are runtime-only and are rebuilt from saved rotation
    metadata on a reloaded model, so they must not be registered twice.
    """
    if get_teq_equivalent_transforms(module).get("input") is None:
        return
    if getattr(module, "_autoround_teq_equivalent_input_hook_installed", False):
        return

    def _input_hook(layer, args):
        if not args:
            return None
        return (apply_teq_equivalent_input_transform(layer, args[0]), *args[1:])

    _input_hook._autoround_teq_equivalent_input_hook = True
    module.register_forward_pre_hook(_input_hook, prepend=True)
    module.__dict__["_autoround_teq_equivalent_input_hook_installed"] = True
