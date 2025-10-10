# Copyright (c) 2025 Intel Corporation
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
from dataclasses import asdict
from typing import Iterable

import torch

from auto_round import AutoScheme
from auto_round.auto_schemes import AUTO_SCHEMES_METHODS
from auto_round.auto_schemes.utils import compute_avg_bits_for_scheme
from auto_round.logger import logger


class GenScheme:
    """Generate and validate quantization schemes for model layers."""

    def __init__(
        self,
        auto_scheme: AutoScheme,  # TODO support shared layer
        model: torch.nn.Module,
        quant_layer_names: Iterable[str],
        fixed_layer_scheme: dict[str, dict],
        dataset: str = "pile-10k",  # TODO use auto-round dataset
        tokenizer=None,
    ):
        self.auto_scheme = auto_scheme
        self.model = model
        self.tokenizer = tokenizer
        self.quant_layer_names = quant_layer_names
        self.fixed_layer_scheme = fixed_layer_scheme
        self.dataset = dataset

        self._check_configs()

    def _check_configs(self) -> None:
        """Validate auto_scheme configuration and ensure avg_bits target is valid."""
        if isinstance(self.model, torch.nn.Module) and self.tokenizer is None:
            raise ValueError("tokenizer must not be None if model is nn.Module")

        if not isinstance(self.dataset, str):
            raise TypeError("`dataset` must be a string, got {type(self.dataset).__name__}.")

        min_avg_bit, max_avg_bit = self.compute_avg_bit_range()
        target = self.auto_scheme.avg_bits

        logger.info("Average bits range: [%.3f, %.3f], target = %.3f", min_avg_bit, max_avg_bit, target)

        if not (min_avg_bit <= target <= max_avg_bit):
            raise ValueError(
                f"Target avg_bits={target:.3f} is outside the valid range " f"[{min_avg_bit:.3f}, {max_avg_bit:.3f}]."
            )

    def get_layer_config(self):
        method_name = self.auto_scheme.method
        method_func = AUTO_SCHEMES_METHODS[method_name]
        layer_config = method_func(
            self.auto_scheme, self.model, self.quant_layer_names, self.fixed_layer_scheme, self.dataset, self.tokenizer
        )
        return layer_config

    def compute_avg_bit_range(self) -> tuple[float, float]:
        """Compute the min and max average bitwidths among candidate quantization options."""
        avg_bits = [
            compute_avg_bits_for_scheme(
                self.model,
                self.quant_layer_names,
                self.fixed_layer_scheme,
                option,
                self.auto_scheme.ignore_scale_zp_bits,
            )[0]
            for option in self.auto_scheme.options
        ]
        return min(avg_bits), max(avg_bits)
