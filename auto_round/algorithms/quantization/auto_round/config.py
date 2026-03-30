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
from typing import Union

from auto_round.algorithms.quantization.config import QuantizationConfig
from auto_round.logger import logger


class AutoRoundConfig(QuantizationConfig):
    """

    Args:
    iters (int): Number of iterations (default is 200).
        lr (float): The learning rate (default is 0.005).
        minmax_lr (float): The learning rate for min-max tuning (default is None).
        lr_scheduler: The learning rate scheduler to be used.
        batch_size (int): Batch size for training (default is 8).
        enable_minmax_tuning (bool): Whether to enable min-max tuning (default is True).
        enable_norm_bias_tuning (bool): Whether to enable fast norm/layer_bias tuning
    """

    _alg_cls = "ARQuantizer"

    def __init__(
        self,
        layer_config: dict[str, Union[str, dict]] = None,
        *,
        iters: int = 200,
        lr: float = None,
        minmax_lr: float = None,
        lr_scheduler=None,
        seqlen: int = 2048,
        nsamples: int = 128,
        momentum: float = 0.0,
        batch_size: int = 8,
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
    ):
        super().__init__(layer_config=layer_config, **kwargs)
        self.iters = iters
        if self.iters < 0:
            logger.warning("`iters` must be non-negative, reset it to 200")
            self.iters = 200

        if not lr:
            # TODO need to check 4 bits lr setting for auto-round-best, 3bits only validate on small models
            if self.iters >= 1000 and self.bits is not None and self.bits <= 3:
                self.lr = 2.0 / self.iters
                logger.info("set the lr to 2.0/iters for better accuracy")
            else:
                self.lr = 1.0 / self.iters
        else:
            self.lr = lr
        self.minmax_lr = minmax_lr or self.lr
        self.lr_scheduler = lr_scheduler

        self.seqlen = seqlen
        self.nsamples = nsamples
        self.batch_size, self.gradient_accumulate_steps = batch_size, gradient_accumulate_steps
        self.nblocks = nblocks
        self.momentum = momentum
        self.enable_alg_ext = enable_alg_ext

        # Some helpers
        self.infer_bs_coeff = 1
        self.batch_dim = None

        self.enable_minmax_tuning = enable_minmax_tuning
        self.enable_norm_bias_tuning = enable_norm_bias_tuning
        if self.enable_norm_bias_tuning:
            logger.warning("the `enable_norm_bias_tuning` feature is experimental and currently has limited support.")
        self.not_use_best_mse = not_use_best_mse
        self.dynamic_max_gap = dynamic_max_gap
        self.enable_quanted_input = enable_quanted_input
        self.optimizer = optimizer
        self.enable_adam = enable_adam

        if self.enable_adam:
            self._alg_cls = "ARAdamQuantizer"

    def check_configs(self) -> None:
        """Checks if the configurations are valid.

        Raises:
        ValueError, TypeError: If any of the configurations are invalid.
        """
        super().check_config()

        if self.batch_size <= 0:
            raise ValueError("`batch_size` must be positive")
        if self.iters < 0:
            raise ValueError("`iters` must be non-negative")
        if self.seqlen <= 0:
            raise ValueError("`seqlen` must be positive")
        if self.nblocks <= 0:
            raise ValueError("`nblocks` must be positive")
        if self.gradient_accumulate_steps <= 0:
            raise ValueError("`gradient_accumulate_steps` must be positive")

        if self.nsamples < self.gradient_accumulate_steps * self.batch_size:
            if self.batch_size > self.nsamples:
                if self.iters > 0:  # GGUF should log this warning, but we don't know the format here
                    logger.warning(
                        f"reset `batch_size` to {self.nsamples} as `nsamples`({self.nsamples})"
                        f" is smaller than batch_size({self.batch_size})"
                    )
                self.batch_size = self.nsamples
            if self.gradient_accumulate_steps > self.nsamples // self.batch_size:
                self.gradient_accumulate_steps = self.nsamples // self.batch_size
                logger.warning(
                    f"reset `gradient_accumulate_steps` to {self.gradient_accumulate_steps}"
                    f" as nsamples must equal or greater"
                    f" than gradient_accumulate_steps * batch_size"
                )
