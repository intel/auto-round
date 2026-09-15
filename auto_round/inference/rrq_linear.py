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

Phase-1 reference implementation.  Each plane is stored as its *dequantized*
float weight of shape ``(out_features, in_features)``.  ``forward`` accumulates
the first ``active_planes`` planes and runs a single linear op:

- active_planes=1: 2-bit (base only)
- active_planes=2: 4-bit (base + 1 residual)
- active_planes=3: 6-bit (base + 2 residuals)
- active_planes=4: 8-bit (base + 3 residuals)

This is a correctness reference implementation, not a performance kernel.
Packed-INT2 storage and fused kernels are a follow-up concern (Phase 2+).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from auto_round.utils import logger

__all__ = ["RRQLinear", "set_rrq_bits", "set_rrq_random_residual"]


class RRQLinear(nn.Module):
    """Multi-plane INT2 quantized linear layer with dynamic precision switching.

    Each plane is stored as its *dequantized* float weight of shape
    ``(out_features, in_features)``.  ``forward`` accumulates the first
    ``active_planes`` planes and runs a single linear op::

        - active_planes=1: 2-bit (base only)
        - active_planes=2: 4-bit (base + 1 residual)
        - active_planes=3: 6-bit (base + 2 residuals)
        - active_planes=4: 8-bit (base + 3 residuals)

    Buffers:
        ``rrq_qweight_k``  -- dequantized weight of plane ``k`` (float),
                             shape ``[out_features, in_features]``.
        ``rrq_scales_k``  : per-group scale of plane ``k`` (informational).
        ``rrq_zp_k``      : zero-point of plane ``k`` (asymmetric only), if any.
    """

    def __init__(
        self,
        in_features: int = None,
        out_features: int = None,
        num_planes: int = 4,
        bits: int = 2,
        bias: bool = True,
        base: nn.Module = None,
        residual_planes=None,
    ) -> None:
        super().__init__()

        if base is not None or residual_planes is not None:
            residual_planes = list(residual_planes or [])
            self._packed = True
            self.base = base
            self.num_planes = 1 + len(residual_planes)
            self.in_features = base.infeatures
            self.out_features = base.outfeatures
            self.bits = base.bits
            self.planes = nn.ModuleDict({f"rrq_{index}": plane for index, plane in enumerate(residual_planes, start=1)})
            self.active_planes = self.num_planes
            if isinstance(bias, torch.Tensor):
                self.register_buffer("bias", bias)
            else:
                self.register_parameter("bias", None)
            self._packed_weight = None
            self._packed_weight_planes = None
            return

        self._packed = False
        self.in_features = in_features
        self.out_features = out_features
        self.num_planes = num_planes
        self.bits = bits
        self.active_planes = num_planes  # default: use all planes

        # Dequantized weight of each plane, shape (out_features, in_features).
        for k in range(num_planes):
            self.register_buffer(
                f"rrq_qweight_{k}",
                torch.zeros((out_features, in_features), dtype=torch.float16),
            )

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic plane selection.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        if self._packed:
            # Dequant each active plane once, accumulate them into a single
            # weight, then run one matmul + bias. This replaces four separate
            # GEMMs (one per plane) with a single GEMM.
            weight = self._get_packed_weight(self.active_planes)
            out = torch.matmul(x, weight.to(x.dtype))
            out = out.to(x.dtype)
            # The base plane may own the bias; add it if present.
            if self.base.bias is not None:
                out = out + self.base.bias.to(out.dtype)
            if self.bias is not None:
                out = out + self.bias.to(out.dtype)
            return out

        weight = self._dequantize(self.active_planes)
        out = F.linear(x.to(weight.dtype), weight)
        if self.bias is not None:
            out = out + self.bias.to(x.dtype)
        return out

    # -- Packed (production) path helpers -----------------------------------------

    def _get_packed_weight(self, num_planes: int) -> torch.Tensor:
        """Build (and cache) the dequantized accumulated weight for ``num_planes``.

        Each plane is dequantized once via :meth:`QuantLinear._dequantize`, and
        the resulting ``(in, out)`` weights are summed in float32 to avoid
        precision loss.  The cache is invalidated when the underlying packed
        tensors change (the cache key is ``num_planes``).
        """
        if self._packed_weight is not None and self._packed_weight_planes == num_planes:
            return self._packed_weight

        weight = self.base._dequantize().float()
        for index in range(1, num_planes):
            weight = weight + self.planes[f"rrq_{index}"]._dequantize().to(weight.dtype)

        self._packed_weight = weight
        self._packed_weight_planes = num_planes
        return weight

    def reset_packed_weight_cache(self) -> None:
        """Invalidate the cached weight (call after in-place weight changes)."""
        self._packed_weight = None
        self._packed_weight_planes = None

    # -- Non-packed (reference) path helpers --------------------------------------

    def _dequantize(self, num_planes: int) -> torch.Tensor:
        """Accumulate the first ``num_planes`` planes into a full weight tensor.

        Args:
            num_planes: Number of planes to accumulate (1 <= num_planes <= num_planes).

        Returns:
            Reconstructed weight tensor of shape ``(out_features, in_features)``.
        """
        assert 1 <= num_planes <= self.num_planes, f"active_planes must be in [1, {self.num_planes}], got {num_planes}"

        total = torch.zeros(
            self.out_features,
            self.in_features,
            dtype=torch.float32,
            device=self.rrq_qweight_0.device,
        )
        for k in range(num_planes):
            plane = getattr(self, f"rrq_qweight_{k}").to(torch.float32)
            total += plane

        return total

    def set_active_planes(self, num_planes: int) -> None:
        """Set how many planes are used (1..num_planes)."""
        if not 1 <= num_planes <= self.num_planes:
            raise ValueError(f"active_planes must be in [1, {self.num_planes}], got {num_planes}")
        self.active_planes = num_planes

    def set_active_bits(self, bits: int) -> None:
        """Compatibility wrapper for callers that select precision in bits."""
        if bits not in (2, 4, 6, 8) or bits % self.bits != 0:
            raise ValueError(f"active_bits must be one of 2/4/6/8, got {bits}")
        self.set_active_planes(bits // self.bits)

    @property
    def active_bits(self) -> int:
        return self.active_planes * self.bits

    @active_bits.setter
    def active_bits(self, bits: int) -> None:
        self.set_active_bits(bits)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, num_planes={self.num_planes}, "
            f"active_planes={self.active_planes}"
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

    num_planes = bits // 2  # each plane is 2 bits
    found = False
    for module in model.modules():
        if isinstance(module, RRQLinear):
            if num_planes > module.num_planes:
                raise ValueError(
                    f"Requested {bits}-bit precision ({num_planes} planes) but "
                    f"{module} only has {module.num_planes} planes."
                )
            module.active_planes = num_planes
            found = True

    if not found:
        logger.warning("No RRQLinear modules found in model; set_rrq_bits had no effect.")


def set_rrq_random_residual(
    model: nn.Module,
    fraction: float = 0.5,
    seed: int = 0,
    high_bits: int = 4,
    low_bits: int = 2,
) -> int:
    """Randomly give a fraction of RRQLinear layers the higher precision.

    A reproducible (seeded) uniform sample of ``fraction`` of the RRQLinear
    layers is set to ``high_bits`` effective bits; the rest use ``low_bits``.
    This yields a mixed operating point between the two levels (e.g. ``0.5`` of
    the layers at 4-bit and the rest at 2-bit is an effective ~3-bit model).

    Args:
        model: Model containing ``RRQLinear`` modules.
        fraction: Fraction of layers assigned ``high_bits`` (0.0 <= f <= 1.0).
        seed: RNG seed for the layer selection (reproducible).
        high_bits: Effective bit-width for the selected layers (2/4/6/8).
        low_bits: Effective bit-width for the rest (2/4/6/8).

    Returns:
        The number of layers assigned ``high_bits``.

    Raises:
        ValueError: If ``fraction`` is out of range, bit-widths are invalid,
            or no RRQLinear modules exist.
    """
    import random

    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0.0, 1.0], got {fraction}")
    valid_bits = {2, 4, 6, 8}
    if high_bits not in valid_bits or low_bits not in valid_bits:
        raise ValueError(f"high_bits/low_bits must be in {valid_bits}, got {high_bits}/{low_bits}")

    layers = [(name, m) for name, m in model.named_modules() if isinstance(m, RRQLinear)]
    if not layers:
        raise ValueError("No RRQLinear modules found in model.")

    high_planes = high_bits // 2
    low_planes = low_bits // 2
    for name, m in layers:
        if max(high_planes, low_planes) > m.num_planes:
            raise ValueError(
                f"Requested {max(high_bits, low_bits)}-bit but {name!r} only has " f"{m.num_planes} planes."
            )

    names = [name for name, _ in layers]
    n_high = round(fraction * len(names))
    keep = set(random.Random(seed).sample(names, n_high))
    for name, m in layers:
        m.active_planes = high_planes if name in keep else low_planes

    logger.info(
        f"RRQ random residual: {n_high}/{len(names)} layers at {high_bits}-bit, "
        f"rest at {low_bits}-bit (fraction={fraction}, seed={seed})."
    )
    return n_high
