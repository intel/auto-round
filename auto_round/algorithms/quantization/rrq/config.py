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
    explicitly selects RTN-only quantization (no sign-SGD tuning).
    """

    #: Number of INT2 residual planes (after the base plane).
    #: Total planes = 1 (base) + num_residual_planes.
    num_residual_planes: int = 3

    def __init__(
        self,
        *,
        iters: int = 200,
        lr: float | None = None,
        minmax_lr: float | None = None,
        momentum: float = 0.0,
        lr_scheduler=None,
        enable_minmax_tuning: bool = True,
        gradient_accumulate_steps: int = 1,
        enable_quanted_input: bool = True,
        not_use_best_mse: bool = False,
        dynamic_max_gap: int = -1,
        enable_lfq: bool = False,
        num_residual_planes: int = 3,
        disable_opt_rtn: bool | None = True,
        **kwargs,
    ):
        # Enforce fixed values (None means "not provided", use default)
        if kwargs.get("bits") is not None and kwargs["bits"] != 2:
            raise ValueError(f"RRQ only supports bits=2 per plane, got {kwargs['bits']}")
        if kwargs.get("data_type") is not None and kwargs["data_type"] != "int":
            raise ValueError(f"RRQ only supports data_type='int', got {kwargs['data_type']!r}")
        if kwargs.get("act_bits") is not None and kwargs["act_bits"] != 16:
            raise ValueError(f"RRQ is weight-only; act_bits must be 16, got {kwargs['act_bits']}")

        # RRQ-tunable fields (not scheme fields). Declared as named parameters so
        # the CLI field-acceptance contract (see test_cli_usage) holds; stored
        # before super() so the inherited RTN/Quantization __init__ doesn't drop
        # them. Match SignRoundConfig's AutoRound default. RTN remains available
        # via an explicit ``iters=0``.
        self._rrq_iters = iters
        self._rrq_lr = lr
        self._rrq_minmax_lr = minmax_lr
        self._rrq_momentum = momentum
        self._rrq_lr_scheduler = lr_scheduler
        self._rrq_enable_minmax_tuning = enable_minmax_tuning
        self._rrq_gradient_accumulate_steps = gradient_accumulate_steps
        self._rrq_enable_quanted_input = enable_quanted_input
        self._rrq_not_use_best_mse = not_use_best_mse
        self._rrq_dynamic_max_gap = dynamic_max_gap
        self._rrq_enable_lfq = enable_lfq

        # Inject fixed values
        kwargs.setdefault("bits", 2)
        kwargs.setdefault("data_type", "int")
        kwargs.setdefault("act_bits", 16)

        # num_residual_planes is not a scheme field
        if num_residual_planes not in (1, 3):
            raise ValueError(f"RRQ supports num_residual_planes=1 or 3, got {num_residual_planes}")
        if num_residual_planes <= 0:
            raise ValueError("num_residual_planes must be positive")
        self._num_residual_planes = num_residual_planes

        # ``disable_opt_rtn`` must stay True: if it were False, the entry
        # would coerce this config to OptimizedRTNConfig (dropping every
        # residual plane).  The per-plane RTN quality is matched to standard
        # AutoRound (opt-RTN) inside the quantizer instead (see
        # RRQRTNQuantizer).  ``check_config()`` enforces this invariant.
        # CLI may pass None (user didn't specify the flag), so force it to True.
        if disable_opt_rtn is None:
            disable_opt_rtn = True
        super().__init__(disable_opt_rtn=disable_opt_rtn, **kwargs)

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
        registry.add_argument("--lr", field="lr", default=None, type=float, help="Learning rate for the RRQ tuning.")
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
        if not self.disable_opt_rtn:
            raise ValueError(
                "RRQ requires disable_opt_rtn=True. "
                "Setting disable_opt_rtn=False would cause the entry to "
                "coerce this config to OptimizedRTNConfig, silently dropping "
                "all RRQ residual planes. The per-plane RTN quality is "
                "already matched to standard AutoRound inside the RRQ "
                "quantizer (see RRQRTNQuantizer)."
            )


register_algorithm(
    "rrq",
    aliases=("rrq", "rrq_rtn"),
    config_factory=RRQConfig,
    summary="Recurrent Residual Quantization: INT2 base + 3 residual planes (2/4/6/8-bit).",
    hidden=True,
)
