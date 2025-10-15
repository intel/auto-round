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
import math
from dataclasses import asdict
from typing import Iterable, Union

import torch

from auto_round import AutoScheme
from auto_round.auto_schemes import AUTO_SCHEMES_METHODS
from auto_round.auto_schemes.utils import compute_avg_bits_for_scheme
from auto_round.export.export_to_gguf.config import GGUF_INNER_CONFIG
from auto_round.logger import logger
from auto_round.utils import _gguf_type_fallback, get_layer_features, get_module


class GenScheme:
    """Generate and validate quantization schemes for model layers."""

    def __init__(
        self,
        auto_scheme: AutoScheme,  # TODO support shared layer
        model: torch.nn.Module,
        quant_layer_names: Iterable[str],
        fixed_layer_scheme: dict[str, dict],
        dataset: str = "pile-10k",  # TODO use auto-round dataset
        device_map: Union[str, torch.device, int, dict, None] = None,
        tokenizer=None,
    ):
        self.auto_scheme = auto_scheme
        self.model = model
        self.tokenizer = tokenizer
        self.quant_layer_names = quant_layer_names
        self.fixed_layer_scheme = fixed_layer_scheme
        self.dataset = dataset
        self.device_map = device_map
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
        if abs(target - min_avg_bit) < 1e-3 or abs(target - max_avg_bit) < 1e-3:
            if abs(target - min_avg_bit) < 1e-3:
                target = min_avg_bit
            else:
                target = max_avg_bit
            self.auto_scheme.avg_bits = target

        if not (min_avg_bit <= target <= max_avg_bit):
            raise ValueError(
                f"Target avg_bits={target:.3f} is outside the valid range " f"[{min_avg_bit:.3f}, {max_avg_bit:.3f}]."
            )

    def get_layer_config(self) -> dict[str, dict]:
        method_name = self.auto_scheme.method
        method_func = AUTO_SCHEMES_METHODS[method_name]
        layer_config = method_func(
            self.auto_scheme,
            self.model,
            self.quant_layer_names,
            self.fixed_layer_scheme,
            self.dataset,
            self.tokenizer,
            device_map=self.device_map,
        )
        layer_config = self.fallback_gguf_layer_config(layer_config)
        return layer_config

    def fallback_gguf_layer_config(self, layer_config: dict[str, dict]) -> dict[str, dict]:
        """
        Apply fallback configurations for GGUF quantized layers when the current
        layer configuration is incompatible with input feature alignment.

        Args:
            layer_config (dict[str, dict]): Mapping from layer name to its quantization scheme.

        Returns:
            dict[str, dict]: Updated layer configuration with applied fallbacks if necessary.
        """
        for name, scheme in layer_config.items():  # TODO: add unit test (wenhua), the code is a little tricky
            if scheme.get("super_bits") is None:
                continue  # Skip non-GGUF k-quant layers

            layer = get_module(self.model, name)
            input_features = get_layer_features(layer)

            # Determine fallback quantization type
            if input_features % 256 != 0 and input_features % 32 != 0:
                new_type = "gguf:bf16"
            else:
                bits = scheme["bits"]
                new_type = f"gguf:q{bits}_0"

                if new_type not in GGUF_INNER_CONFIG:
                    new_type = f"gguf:q{bits}_1"

                    if new_type not in GGUF_INNER_CONFIG:
                        current_type = f"gguf:q{bits}_k"
                        new_type = _gguf_type_fallback(current_type)

            # Apply fallback configuration
            target_config = GGUF_INNER_CONFIG[new_type]
            scheme.update(target_config)

            logger.warning(f"Fallback applied: {name} → {new_type}")

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
