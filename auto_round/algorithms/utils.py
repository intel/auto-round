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
"""Shared utilities for the algorithms package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from auto_round.algorithms.base import BaseOrchestrator


def _is_nvfp4_value(value: Any) -> bool:
    """Return True when a raw config value indicates NVFP4."""
    if not isinstance(value, str):
        return False
    value = value.lower()
    return "nv_fp" in value or "nvfp4" in value


def _has_nvfp4_layer(orchestrator: "BaseOrchestrator") -> bool:
    """Whether global or per-layer config enables any NVFP4 quantization."""
    if _is_nvfp4_value(getattr(orchestrator, "data_type", "")):
        return True

    layer_config = getattr(orchestrator, "layer_config", None)
    if not isinstance(layer_config, dict):
        return False

    for config in layer_config.values():
        if not isinstance(config, dict):
            continue
        if _is_nvfp4_value(config.get("data_type")) or _is_nvfp4_value(config.get("act_data_type")):
            return True
        if _is_nvfp4_value(config.get("scheme")):
            return True
    return False


