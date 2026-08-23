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
from typing import Callable

from auto_round.algorithms.quantization.config import QuantizationConfig
from auto_round.logger import logger


class SignRoundConfig(QuantizationConfig):
    """Configuration for SignRound-style block quantization."""

    def __init__(
        self,
        *,
        iters: int = 200,
        lr: float | None = None,
        minmax_lr: float | None = None,
        lr_scheduler: Callable | None = None,
        momentum: float = 0.0,
        nblocks: int = 1,
        enable_minmax_tuning: bool = True,
        enable_norm_bias_tuning: bool = False,
        gradient_accumulate_steps: int = 1,
        enable_alg_ext: bool = False,
        not_use_best_mse: bool = False,
        dynamic_max_gap: int = -1,
        enable_quanted_input: bool = True,
        optimizer: str | None = None,  # TODO later wenhuach delete this
        enable_adam: bool = False,  # TODO later  wenhuach delete this
        enable_lfq: bool = False,
        **kwargs,
    ) -> None:
        """Initialize a SignRound configuration.

        Args:
            iters: Number of optimization iterations for each quantized
                block.
            lr: Learning rate used by the main rounding optimization.
                If None, a heuristic based on ``iters`` is used.
            minmax_lr: Learning rate used by min-max tuning. If None, it
                falls back to ``lr``.
            lr_scheduler: Optional learning-rate scheduler name or
                scheduler object used by the optimizer.
            momentum: Momentum factor used by the optimizer.
            nblocks: Number of blocks to optimize together.
            enable_minmax_tuning: Whether to tune weight min/max ranges.
            enable_norm_bias_tuning: Whether to tune normalization and
                bias terms.
            gradient_accumulate_steps: Number of gradient accumulation
                steps used per optimization update.
            enable_alg_ext: Whether to enable the experimental SignRound
                extension implementation.
            not_use_best_mse: Whether to skip restoring the best-MSE
                checkpoint during tuning.
            dynamic_max_gap: Maximum dynamic gap used by adaptive tuning
                logic.
            enable_quanted_input: Whether each block should consume the
                quantized output of previous blocks during calibration.
            optimizer: Optional optimizer name override.
            enable_adam: Whether to use the Adam-based SignRound variant.
            **kwargs: Common quantization arguments forwarded to
                QuantizationConfig, such as bits, group_size, sym,
                data_type, and activation quantization fields.
        """
        super().__init__(**kwargs)
        self.gradient_accumulate_steps = gradient_accumulate_steps
        self.iters = iters
        if self.iters < 0:
            logger.warning("`iters` must be non-negative, reset it to 200")
            self.iters = 200

        # lr/minmax_lr depend on `bits`, which may still be unresolved here
        # (e.g. only `scheme=` was given) -- finalize_scheme() fills them in.
        self.lr = lr
        self.minmax_lr = minmax_lr
        # Whether lr/minmax_lr are auto-derived; set properly in finalize_scheme().
        # Used to enable per-layer (mixed-bit) lr selection during tuning.
        self.lr_is_auto = lr is None
        self.minmax_lr_is_auto = minmax_lr is None
        self.lr_scheduler = lr_scheduler

        self.nblocks = nblocks
        self.momentum = momentum
        self.enable_alg_ext = enable_alg_ext

        self.enable_minmax_tuning = enable_minmax_tuning
        self.enable_norm_bias_tuning = enable_norm_bias_tuning
        if self.enable_norm_bias_tuning:
            logger.warning("the `enable_norm_bias_tuning` feature is experimental and currently has limited support.")
        self.not_use_best_mse = not_use_best_mse
        self.dynamic_max_gap = dynamic_max_gap
        self.enable_quanted_input = enable_quanted_input
        self.optimizer = optimizer
        self.enable_adam = enable_adam
        self.enable_lfq = enable_lfq
        if self.enable_lfq:
            logger.warning("the `enable_lfq` feature is experimental and currently has limited model support.")

    def _lr_for_bits(self, bits: int | None) -> float | None:
        """Return the auto lr heuristic for a given bit-width.

        Low-bit schemes (<=3 bits) use a higher lr for better accuracy.
        Returns None when lr cannot be derived (e.g. ``iters == 0``).
        """
        if self.iters <= 0:
            return None
        # TODO need to check 4 bits lr setting for auto-round-best, 3bits only validate on small models
        if self.iters >= 1000 and bits is not None and bits <= 3:
            return 2.0 / self.iters
        return 1.0 / self.iters

    def compute_lr(self, bits: int | None) -> float | None:
        """Resolve the rounding lr for a specific layer bit-width.

        When the user supplied an explicit ``lr`` it is used for every layer;
        otherwise the auto heuristic is applied per-layer so that mixed-bit
        configs (e.g. a 4-bit model with a few 2-bit layers) get an
        appropriate lr for each layer.
        """
        if not self.lr_is_auto:
            return self.lr
        return self._lr_for_bits(bits)

    def compute_minmax_lr(self, bits: int | None) -> float | None:
        """Resolve the min-max tuning lr for a specific layer bit-width."""
        if not self.minmax_lr_is_auto:
            return self.minmax_lr
        # minmax_lr falls back to lr when not explicitly set
        return self.compute_lr(bits)

    def finalize_scheme(self) -> None:
        """Resolve lr/minmax_lr once `bits` is known (low-bit schemes use a higher lr).

        The scalar ``lr``/``minmax_lr`` values resolved here are derived from the
        global ``bits`` and serve as defaults/fallbacks. For mixed-bit configs the
        per-layer values are computed later via ``compute_lr``/``compute_minmax_lr``.
        """
        # Track whether the values are auto-derived so per-layer bit widths can
        # be honored during optimization (see compute_lr/compute_minmax_lr).
        self.lr_is_auto = self.lr is None and self.iters > 0
        self.minmax_lr_is_auto = self.minmax_lr is None
        if self.lr_is_auto:
            self.lr = self._lr_for_bits(self.bits)
        self.minmax_lr = self.minmax_lr or self.lr

    def check_configs(self) -> None:
        """Checks if the configurations are valid.

        Raises:
        ValueError, TypeError: If any of the configurations are invalid.
        """
        super().check_config()

        if self.iters < 0:
            raise ValueError("`iters` must be non-negative")
        if self.nblocks <= 0:
            raise ValueError("`nblocks` must be positive")
        if self.gradient_accumulate_steps <= 0:
            raise ValueError("`gradient_accumulate_steps` must be positive")


class AdamRoundConfig(SignRoundConfig):
    pass


class SignRoundV2Config(SignRoundConfig):
    pass
