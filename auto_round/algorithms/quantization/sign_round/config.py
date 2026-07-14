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
from typing import Any, Callable, Union

from auto_round.algorithms.quantization.config import QuantizationConfig
from auto_round.logger import logger


class SignRoundConfig(QuantizationConfig):
    """Configuration for SignRound-style block quantization."""

    def __init__(
        self,
        *,
        iters: int = 200,
        lr: float = None,
        minmax_lr: float = None,
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
        optimizer: str = None,
        enable_adam: bool = False,
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
        self.lr_scheduler = lr_scheduler

        self.nblocks = nblocks
        self.momentum = momentum
        self.enable_alg_ext = enable_alg_ext

        # Some helpers
        self.infer_bs_coeff = 1

        self.enable_minmax_tuning = enable_minmax_tuning
        self.enable_norm_bias_tuning = enable_norm_bias_tuning
        if self.enable_norm_bias_tuning:
            logger.warning("the `enable_norm_bias_tuning` feature is experimental and currently has limited support.")
        self.not_use_best_mse = not_use_best_mse
        self.dynamic_max_gap = dynamic_max_gap
        self.enable_quanted_input = enable_quanted_input
        self.optimizer = optimizer
        self.enable_adam = enable_adam

    def finalize_scheme(self) -> None:
        """Resolve lr/minmax_lr once `bits` is known (low-bit schemes use a higher lr)."""
        if self.lr is None:
            # TODO need to check 4 bits lr setting for auto-round-best, 3bits only validate on small models
            if self.iters >= 1000 and self.bits is not None and self.bits <= 3:
                self.lr = 2.0 / self.iters
                logger.info("set the lr to 2.0/iters for better accuracy")
            else:
                self.lr = 1.0 / self.iters
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
