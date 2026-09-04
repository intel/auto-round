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
"""Configuration for Recurrent Residual Quantization (RRQ).

RRQ quantizes each weight tensor into K sequential INT2 planes (1 base +
K-1 residual planes).  The total effective bit-width is K * plane_bits.
In Phase 1 the algorithm is pure RTN (no sign-SGD tuning).
"""

from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.algorithms.registry import register_algorithm


class RRQConfig(RTNConfig):
    """Configuration for Recurrent Residual Quantization (RTN-based, Phase 1).

    Fixed in this version:
        - bits = 2 (per plane)
        - data_type = "int"
        - act_bits = 16 (weight-only)
        - num_residual_planes = 3 (total planes = 4, effective bits 2/4/6/8)

    The user only controls ``group_size`` and ``sym``.
    """

    #: Number of INT2 residual planes (after the base plane).
    #: Total planes = 1 (base) + num_residual_planes.
    num_residual_planes: int = 3

    def __init__(self, **kwargs):
        # Enforce fixed values for Phase 1
        if "bits" in kwargs and kwargs["bits"] != 2:
            raise ValueError(f"RRQ only supports bits=2 per plane, got {kwargs['bits']}")
        if "data_type" in kwargs and kwargs["data_type"] != "int":
            raise ValueError(f"RRQ only supports data_type='int', got {kwargs['data_type']!r}")
        if "act_bits" in kwargs and kwargs["act_bits"] != 16:
            raise ValueError(f"RRQ is weight-only; act_bits must be 16, got {kwargs['act_bits']}")

        # Inject fixed values
        kwargs.setdefault("bits", 2)
        kwargs.setdefault("data_type", "int")
        kwargs.setdefault("act_bits", 16)

        # Extract num_residual_planes before super (it's not a scheme field)
        self._num_residual_planes = kwargs.pop("num_residual_planes", 3)
        if self._num_residual_planes != 3:
            raise ValueError(
                f"RRQ Phase 1 only supports num_residual_planes=3, got {self._num_residual_planes}"
            )
        if self._num_residual_planes <= 0:
            raise ValueError("num_residual_planes must be positive")

        super().__init__(**kwargs)

    @property
    def total_planes(self) -> int:
        """Total number of INT2 planes (base + residual)."""
        return 1 + self._num_residual_planes

    @property
    def total_bits(self) -> int:
        """Total effective bit-width of the full representation."""
        return self.bits * self.total_planes

    @classmethod
    def register_args(cls, registry):
        """Register CLI args for RRQ (no extra args in Phase 1)."""
        super().register_args(registry)

    def check_config(self) -> None:
        super().check_config()
        # RRQ-specific validation
        assert self.bits == 2, "RRQ Phase 1 requires bits=2"
        assert self.data_type == "int", "RRQ Phase 1 only supports int data_type"
        assert self.act_bits == 16, "RRQ Phase 1 is weight-only"


register_algorithm(
    "rrq",
    aliases=("rrq", "rrq_rtn"),
    config_factory=RRQConfig,
    summary="Recurrent Residual Quantization: INT2 base + 3 residual planes (2/4/6/8-bit).",
)
