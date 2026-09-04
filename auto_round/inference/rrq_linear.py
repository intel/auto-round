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
"""RRQ inference module: multi-plane INT2 linear layer with dynamic precision.

Phase 1 implementation (packed INT2):

Each plane is a *stock* W2A16 ``QuantLinear`` (``qweight`` int32 + ``scales`` +
``qzeros``), so dequantization reuses the existing INT2 AutoRound code path
(:meth:`QuantLinear.forward`) verbatim -- no custom packing/unpacking logic is
needed here.

A layer is composed of::

    base (plane 0) + residual planes 1..K-1

``forward`` computes the base result first, then -- if residual planes are
active -- computes each residual's result in turn and **accumulates** all of
them::

    - active_bits=2: base only
    - active_bits=4: base + 1 residual
    - active_bits=6: base + 2 residuals
    - active_bits=8: base + 3 residuals (all planes)

The bias is added only once (to the base result).  This is a correctness
reference implementation, not a fused kernel.
"""

import torch
import torch.nn as nn

from auto_round.logger import logger

__all__ = ["RRQLinear", "set_rrq_bits"]


class RRQLinear(nn.Module):
    """Multi-plane INT2 quantized linear layer with dynamic precision switching.

    The ``base`` submodule and each ``planes[1:]`` submodule are standard
    W2A16 ``QuantLinear`` modules.  ``forward`` computes the base output first
    and then accumulates the output of the first ``active`` residual planes::

        out = base(x) + sum_{k=1}^{active} planes[k](x) + bias

    where ``active`` is ``active_bits // 2`` (number of *residual* planes; the
    base plane always contributes).

    Submodules:
        ``base``          -- plane 0 (``QuantLinear``).
        ``planes.rrq_1``  -- residual plane 1 (``QuantLinear``).
        ``planes.rrq_2``  -- residual plane 2 (``QuantLinear``).
        ``planes.rrq_3``  -- residual plane 3 (``QuantLinear``).  (K-1 total.)

    Attributes:
        ``active_bits``  -- 2/4/6/8; controls how many residual planes are used.
        ``bias``         -- bias tensor (``None`` if the layer has no bias).
    """

    def __init__(self, base: nn.Module, residual_planes, bias: torch.Tensor = None) -> None:
        super().__init__()
        self.base = base
        self.num_planes = 1 + len(list(residual_planes))  # base + residual
        self.in_features = base.infeatures
        self.out_features = base.outfeatures
        self.bits = base.bits

        planes = nn.ModuleDict()
        for i, plane in enumerate(residual_planes, start=1):
            planes[f"rrq_{i}"] = plane
        self.planes = planes

        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.active_bits = self.bits * self.num_planes  # default: all planes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic plane selection.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        # The base plane is plane 0 and contributes ``bits`` on its own; each
        # additional residual plane adds another ``bits``.  So the number of
        # *residual* planes to include is ``active_bits // bits - 1``.
        active = self.active_bits // self.bits - 1  # number of residual planes

        out = self.base(x)
        if active > 0:
            for i in range(1, active + 1):
                out = out + self.planes[f"rrq_{i}"](x)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out

    def set_active_bits(self, bits: int) -> None:
        """Set the effective bit-width (2/4/6/8)."""
        if bits not in (2, 4, 6, 8) or bits % self.bits != 0:
            raise ValueError(f"active_bits must be one of 2/4/6/8, got {bits}")
        if bits > self.bits * self.num_planes:
            raise ValueError(
                f"Requested {bits}-bit precision ({bits // self.bits} planes) "
                f"but this layer only has {self.num_planes} planes "
                f"(max {self.bits * self.num_planes} bits)."
            )
        self.active_bits = bits

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"num_planes={self.num_planes}, active_bits={self.active_bits}"
        )


def set_rrq_bits(model: nn.Module, bits: int) -> None:
    """Set the active precision for all RRQLinear layers in a model.

    Args:
        model: Model containing ``RRQLinear`` modules.
        bits: Target effective bit-width (2, 4, 6, or 8).

    Raises:
        ValueError: If ``bits`` is not in {2, 4, 6, 8} or no RRQ layers exist.
    """
    valid_bits = {2, 4, 6, 8}
    if bits not in valid_bits:
        raise ValueError(f"RRQ supports only bits in {valid_bits}, got {bits}")

    found = False
    for module in model.modules():
        if isinstance(module, RRQLinear):
            module.set_active_bits(bits)
            found = True

    if not found:
        logger.warning("No RRQLinear modules found in model; set_rrq_bits had no effect.")
