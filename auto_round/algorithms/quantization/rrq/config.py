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

By default RRQ follows AutoRound's default SignRound optimization settings
(``iters=200``, automatic learning rates, zero momentum, and min/max tuning).
Passing ``iters=0`` explicitly selects the RTN-only mode. Every plane uses the
same optimization settings; residual planes differ only in their frozen prefix
and target residual.
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

    With the default ``iters=200``, every plane uses the same SignRound
    optimization parameters as ordinary AutoRound. Passing ``iters=0``
    explicitly selects RTN-only quantization without calibration.
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
        # Match SignRoundConfig's AutoRound default. RTN remains available via
        # an explicit ``iters=0``.
        self._rrq_iters = kwargs.pop("iters", 200)
        self._rrq_lr = kwargs.pop("lr", None)
        self._rrq_minmax_lr = kwargs.pop("minmax_lr", None)
        self._rrq_momentum = kwargs.pop("momentum", 0.0)
        self._rrq_lr_scheduler = kwargs.pop("lr_scheduler", None)
        self._rrq_enable_minmax_tuning = kwargs.pop("enable_minmax_tuning", True)
        self._rrq_gradient_accumulate_steps = kwargs.pop("gradient_accumulate_steps", 1)
        self._rrq_enable_quanted_input = kwargs.pop("enable_quanted_input", True)
        self._rrq_not_use_best_mse = kwargs.pop("not_use_best_mse", False)
        self._rrq_dynamic_max_gap = kwargs.pop("dynamic_max_gap", -1)
        self._rrq_enable_lfq = kwargs.pop("enable_lfq", False)

        # Inject fixed values
        kwargs.setdefault("bits", 2)
        kwargs.setdefault("data_type", "int")
        kwargs.setdefault("act_bits", 16)

        # Extract num_residual_planes before super (it's not a scheme field)
        self._num_residual_planes = kwargs.pop("num_residual_planes", 3)
        if self._num_residual_planes not in (1, 3):
            raise ValueError(
                f"RRQ supports num_residual_planes=1 or 3, got {self._num_residual_planes}"
            )
        if self._num_residual_planes <= 0:
            raise ValueError("num_residual_planes must be positive")

        # ``disable_opt_rtn`` stays True at the config level purely to keep RRQ
        # routed to its own quantizer: an RTNConfig subclass with
        # ``disable_opt_rtn=False`` is silently coerced to OptimizedRTNConfig by
        # the AutoRound entry, which would drop every residual plane. The
        # per-plane RTN quality is matched to standard AutoRound (opt-RTN)
        # inside the quantizer instead (see RRQRTNQuantizer).
        kwargs.setdefault("disable_opt_rtn", True)

        super().__init__(**kwargs)

        self.iters = int(self._rrq_iters or 0)
        self.lr = self._rrq_lr
        self.minmax_lr = self._rrq_minmax_lr
        self.momentum = self._rrq_momentum
        self.lr_scheduler = self._rrq_lr_scheduler
        self.enable_minmax_tuning = self._rrq_enable_minmax_tuning
        self.gradient_accumulate_steps = self._rrq_gradient_accumulate_steps
        self.enable_quanted_input = self._rrq_enable_quanted_input
        self.not_use_best_mse = self._rrq_not_use_best_mse
        self.dynamic_max_gap = self._rrq_dynamic_max_gap
        self.enable_lfq = self._rrq_enable_lfq
        # Both paths use block calibration data: SignRound needs it for tuning,
        # and the RTN path needs it to collect the imatrix for imatrix-weighted
        # opt-RTN (matching an ordinary AutoRound W2A16 OptimizedRTN base).
        self.need_calib = True

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
            default=200,
            type=int,
            help="Iterations of per-plane sign-SGD tuning. 0 explicitly selects "
            "pure RTN; the default 200 matches AutoRound SignRound.",
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
        # Match SignRoundConfig._lr_for_bits: low-bit layers get a higher lr
        # when the iteration budget is large.
        if self.iters >= 1000 and bits is not None and bits <= 3:
            return 2.0 / self.iters
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
