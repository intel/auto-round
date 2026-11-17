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

"""Shared lifecycle for full-model SVDQuant output formats."""

from __future__ import annotations

from typing import Any

import torch

from auto_round.export.base import OutputFormat
from auto_round.schemes import PRESET_SCHEMES, QuantizationScheme


class SVDQuantFormatBase(OutputFormat):
    """Defer packing until save and validate overrides with the runtime's scheme rules."""

    def __init__(self, format: str, scheme: QuantizationScheme, ctx: Any):
        self.output_format = format
        self.backend = None
        self.mllm = ctx.mllm
        self.check_scheme_args(scheme)
        self._resolved_scheme = scheme.copy()

    def is_supported_immediate_packing(self) -> bool:
        return False

    def is_supported_immediate_saving(self) -> bool:
        return False

    def _validate_svd_layer_overrides(self, model: torch.nn.Module, layer_config: dict | None) -> None:
        if model is None or not layer_config:
            return
        from auto_round.algorithms.transforms.svdquant.wrapper import SVDQuantLinear

        for name, module in model.named_modules():
            if not isinstance(module, SVDQuantLinear):
                continue
            for layer_name in (name, f"{name}.residual_linear"):
                if layer_name not in layer_config:
                    continue
                override = layer_config[layer_name]
                if isinstance(override, str):
                    scheme = PRESET_SCHEMES[override.upper()].copy()
                elif isinstance(override, QuantizationScheme):
                    scheme = override.copy()
                elif isinstance(override, dict):
                    values = dict(override)
                    preset_name = values.pop("scheme", None)
                    if preset_name is None:
                        scheme = self._resolved_scheme.copy()
                    else:
                        scheme = PRESET_SCHEMES[preset_name.upper()].copy()
                    scheme.update_from_dict(values)
                else:
                    raise TypeError(
                        f"Unsupported layer_config value for SVDQuant residual {layer_name!r}: {type(override)}"
                    )
                try:
                    self.check_scheme_args(scheme)
                except ValueError as exc:
                    raise ValueError(
                        f"{self.format_name} layer {layer_name!r} has an incompatible residual scheme: {exc}"
                    ) from exc

    def pack_layer(self, layer_name, model, device=None, **kwargs):
        return None
