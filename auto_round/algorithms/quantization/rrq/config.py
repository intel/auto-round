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

Phase 1 quantizes each plane with RTN (``iters=0``).  Phase 3 adds a
per-plane sign-SGD tuning pass (``iters > 0``) so every plane is optimized
against the current residual using the block's calibration loss, while the
already-quantized prefix is held fixed.
"""

from auto_round.algorithms.config import AlgorithmParameterRegistry
from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.algorithms.registry import register_algorithm


class RRQConfig(RTNConfig):
    """Configuration for Recurrent Residual Quantization.

    Fixed in this version:
        - bits = 2 (per plane)
        - data_type = "int"
        - act_bits = 16 (weight-only)
        - num_residual_planes = 3 (total planes = 4, effective bits 2/4/6/8)

    With ``iters=0`` (the default) every plane is quantized by RTN only
    (Phase 1 behaviour, no calibration needed).  Setting ``iters > 0``
    enables per-plane sign-SGD tuning with the block calibration loss
    (Phase 3), which requires a calibration dataset.
    """

    #: Number of INT2 residual planes (after the base plane).
    #: Total planes = 1 (base) + num_residual_planes.
    num_residual_planes: int = 3

    def __init__(self, **kwargs):
        # Enforce fixed values
        if "bits" in kwargs and kwargs["bits"] != 2:
            raise ValueError(f"RRQ only supports bits=2 per plane, got {kwargs['bits']}")
        if "data_type" in kwargs and kwargs["data_type"] != "int":
            raise ValueError(f"RRQ only supports data_type='int', got {kwargs['data_type']!r}")
        if "act_bits" in kwargs and kwargs["act_bits"] != 16:
            raise ValueError(f"RRQ is weight-only; act_bits must be 16, got {kwargs['act_bits']}")

        # Extract tunable fields before super (they are not scheme fields)
        self._rrq_iters = kwargs.pop("iters", 0)
        self._rrq_lr = kwargs.pop("lr", None)
        self._rrq_minmax_lr = kwargs.pop("minmax_lr", None)
        self._rrq_momentum = kwargs.pop("momentum", 0.0)
        self._rrq_lr_scheduler = kwargs.pop("lr_scheduler", None)
        self._rrq_enable_minmax_tuning = kwargs.pop("enable_minmax_tuning", True)

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

        # ``disable_opt_rtn`` forces plain RTN semantics for the weight, which
        # is always the case for the per-plane RRQ quantization.
        kwargs.setdefault("disable_opt_rtn", True)

        super().__init__(**kwargs)

        self.iters = int(self._rrq_iters or 0)
        self.lr = self._rrq_lr
        self.minmax_lr = self._rrq_minmax_lr
        self.momentum = self._rrq_momentum
        self.lr_scheduler = self._rrq_lr_scheduler
        self.enable_minmax_tuning = self._rrq_enable_minmax_tuning
        # Sign-SGD tuning needs the block calibration data; RTN-only does not.
        self.need_calib = self.iters > 0

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
        """Register CLI args for RRQ (Phase 1 common args + optional tuning)."""
        super().register_args(registry)
        registry.add_argument(
            "--iters",
            field="iters",
            default=0,
            type=int,
            help="Iterations of per-plane sign-SGD tuning. 0 (default) keeps "
            "pure RTN; >0 enables Phase-3 tuning and requires a calib dataset.",
        )
        registry.add_argument(
            "--lr", field="lr", default=None, type=float, help="Learning rate for the RRQ tuning."
        )
        registry.add_argument(
            "--minmax_lr", field="minmax_lr", default=None, type=float, help="Learning rate for min-max tuning."
        )
        registry.add_argument(
            "--momentum", field="momentum", default=0.0, type=float, help="Momentum for the RRQ optimizer."
        )

    def _lr_for_bits(self, bits):
        """Auto lr heuristic for sign-SGD tuning (mirrors SignRound)."""
        if self.iters <= 0:
            return None
        return 1.0 / self.iters

    def compute_lr(self, bits):
        """Resolve the rounding lr for a layer bit-width."""
        if self.lr is not None:
            return self.lr
        return self._lr_for_bits(bits)

    def compute_minmax_lr(self, bits):
        """Resolve the min-max tuning lr for a layer bit-width."""
        if self.minmax_lr is not None:
            return self.minmax_lr
        return self.compute_lr(bits)

    def check_config(self) -> None:
        super().check_config()
        # RRQ-specific validation
        assert self.bits == 2, "RRQ requires bits=2"
        assert self.data_type == "int", "RRQ only supports int data_type"
        assert self.act_bits == 16, "RRQ is weight-only"
        if self.iters < 0:
            raise ValueError("`iters` must be non-negative")


register_algorithm(
    "rrq",
    aliases=("rrq", "rrq_rtn"),
    config_factory=RRQConfig,
    summary="Recurrent Residual Quantization: INT2 base + 3 residual planes (2/4/6/8-bit).",
)
