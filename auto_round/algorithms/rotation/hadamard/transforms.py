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
"""Concrete ``torch.nn.Module`` implementations of Hadamard transforms.

:class:`HadamardTransform` – block-diagonal Hadamard (deterministic).
:class:`RandomHadamardTransform` – randomly signed Hadamard.
:func:`build_hadamard_transform` – factory that selects the right class.
"""
from __future__ import annotations

import inspect
import math
from typing import Any, Callable, Dict

import torch
import torch.nn as nn

from auto_round.algorithms.rotation.hadamard.utils.math import (
    deterministic_hadamard_matrix,
    random_hadamard_matrix,
)
from auto_round.algorithms.rotation.hadamard.utils.matrix import apply_transform_weight

__all__ = [
    "HadamardTransform",
    "RandomHadamardTransform",
    "HADAMARDS",
    "build_hadamard_transform",
]


def _filter_kwargs(fn: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the keyword arguments accepted by *fn*."""
    accepted = inspect.signature(fn).parameters.keys()
    return {k: v for k, v in kwargs.items() if k in accepted}


class HadamardTransform(nn.Module):
    """Block-diagonal deterministic Hadamard rotation.

    The rotation matrix ``W`` (stored as a frozen ``nn.Parameter``) is
    constructed once from :func:`deterministic_hadamard_matrix` and
    normalised by ``1 / sqrt(block_size)``.

    Args:
        block_size:   Size of each Hadamard block (must be a power of 2).
        device:       Device to place the weight on.
        precision:    Dtype for the weight tensor.
        location:     ``"weight"`` (default) or ``"input"`` – controls the
                      orientation of the multiplication in :meth:`forward`.
        module_type:  ``type(module)`` passed to
                      :func:`~utils.matrix.apply_transform_weight`.
        inverse:      If ``True``, use transposed orientation (for activation
                      transforms that are the inverse of the weight transform).
    """

    def __init__(
        self,
        block_size: int = 32,
        device: torch.device | None = None,
        precision: torch.dtype | None = None,
        location: str = "weight",
        module_type: type[nn.Module] = nn.Linear,
        inverse: bool = False,
    ) -> None:
        super().__init__()
        self.size = block_size
        self.scale = 1.0 / math.sqrt(self.size)
        self.location = location
        self.module_type = module_type
        self.inverse = inverse
        self.weight = self._build_weight(self.size, device, precision)

    def _build_weight(
        self,
        size: int,
        device: torch.device | None,
        precision: torch.dtype | None,
    ) -> nn.Parameter:
        data = deterministic_hadamard_matrix(size, precision, device) * self.scale
        return nn.Parameter(data, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ori_shape = x.shape
        x = x.view(-1, self.size)
        out = apply_transform_weight(
            self.weight.to(x.device),
            x.to(dtype=self.weight.dtype),
            self.location,
            self.module_type,
        )
        return out.to(x.dtype).view(ori_shape)


class RandomHadamardTransform(HadamardTransform):
    """Randomly signed Hadamard rotation.

    Extends :class:`HadamardTransform` with a seeded random diagonal so the
    same seed always produces the same rotation matrix.

    Args:
        seed:      Integer seed for the internal ``torch.Generator``.
        generator: Pre-built ``torch.Generator`` (overrides *seed* if given).
        *args, **kwargs: Forwarded to :class:`HadamardTransform`.
    """

    def __init__(
        self,
        *args: Any,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> None:
        if generator is not None:
            self.generator = generator
        else:
            self.generator = torch.Generator()
            if seed is not None:
                self.generator.manual_seed(seed)
        super().__init__(*args, **kwargs)

    def _build_weight(
        self,
        size: int,
        device: torch.device | None,
        precision: torch.dtype | None,
    ) -> nn.Parameter:
        data = random_hadamard_matrix(size, precision, device, self.generator) * self.scale
        if self.inverse:
            data = data.T
        return nn.Parameter(data, requires_grad=False)


# ---------------------------------------------------------------------------
# Registry and factory
# ---------------------------------------------------------------------------

#: Maps ``hadamard_type`` strings to their transform classes.
HADAMARDS: dict[str, type[HadamardTransform]] = {
    "hadamard": HadamardTransform,
    "random_hadamard": RandomHadamardTransform,
}


def build_hadamard_transform(hadamard_type: str, **kwargs: Any) -> HadamardTransform:
    """Instantiate the correct :class:`HadamardTransform` subclass.

    Args:
        hadamard_type: Key into :data:`HADAMARDS` (``"hadamard"`` or
                       ``"random_hadamard"``).
        **kwargs:      Forwarded to the transform constructor after filtering
                       out unsupported keys.

    Returns:
        A new :class:`HadamardTransform` instance.
    """
    if hadamard_type not in HADAMARDS:
        raise ValueError(f"Unknown hadamard_type: {hadamard_type!r}. " f"Available: {sorted(HADAMARDS)}")
    cls = HADAMARDS[hadamard_type]
    return cls(**_filter_kwargs(cls.__init__, kwargs))
