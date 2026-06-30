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

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch


@dataclass(frozen=True)
class AdapterSelection:
    module_name: str
    adapter_name: str


class BaseVLLMModuleAdapter:
    name = "base"

    @classmethod
    def supports(cls, module: torch.nn.Module) -> bool:
        return False

    @classmethod
    def get_weight(cls, module: torch.nn.Module) -> Optional[torch.Tensor]:
        """Get the weight tensor for this module."""
        return None

    @classmethod
    def get_canonical_name(cls, module: torch.nn.Module, module_name: str) -> str:
        """Get the canonical name for this module (for export naming)."""
        return module_name


class VLLMLinearAdapter(BaseVLLMModuleAdapter):
    name = "linear"

    @classmethod
    def supports(cls, module: torch.nn.Module) -> bool:
        return isinstance(module, torch.nn.Linear)

    @classmethod
    def get_weight(cls, module: torch.nn.Module) -> Optional[torch.Tensor]:
        if isinstance(module, torch.nn.Linear):
            return module.weight
        return None

    @classmethod
    def get_canonical_name(cls, module: torch.nn.Module, module_name: str) -> str:
        return module_name


class VLLMFusedMoEAdapter(BaseVLLMModuleAdapter):
    name = "fused_moe"

    @classmethod
    def supports(cls, module: torch.nn.Module) -> bool:
        class_name = module.__class__.__name__.lower()
        has_fused_weights = hasattr(module, "w13_weight") or hasattr(module, "w2_weight")
        return ("fusedmoe" in class_name or "routedexperts" in class_name) or has_fused_weights

    @classmethod
    def get_weight(cls, module: torch.nn.Module) -> Optional[torch.Tensor]:
        """Get fused w13 weight if available."""
        if hasattr(module, "w13_weight"):
            return getattr(module, "w13_weight")
        if hasattr(module, "w1_weight") and hasattr(module, "w3_weight"):
            w1 = getattr(module, "w1_weight", None)
            w3 = getattr(module, "w3_weight", None)
            if w1 is not None and w3 is not None:
                return torch.cat([w1, w3], dim=0)
        return None

    @classmethod
    def get_canonical_name(cls, module: torch.nn.Module, module_name: str) -> str:
        return module_name


class ModuleAdapterRegistry:
    """Registry that maps runtime modules to vLLM-specific adapters."""

    def __init__(self, adapters: Optional[list[type[BaseVLLMModuleAdapter]]] = None) -> None:
        self.adapters = adapters or [VLLMFusedMoEAdapter, VLLMLinearAdapter]

    def select(self, module: torch.nn.Module) -> Optional[type[BaseVLLMModuleAdapter]]:
        for adapter in self.adapters:
            if adapter.supports(module):
                return adapter
        return None

    def discover(self, model: torch.nn.Module) -> list[AdapterSelection]:
        found: list[AdapterSelection] = []
        for module_name, module in model.named_modules():
            adapter = self.select(module)
            if adapter is None:
                continue
            found.append(AdapterSelection(module_name=module_name, adapter_name=adapter.name))
        return found

    def iter_supported_modules(self, model: torch.nn.Module) -> Iterable[tuple[str, torch.nn.Module, str]]:
        for module_name, module in model.named_modules():
            adapter = self.select(module)
            if adapter is None:
                continue
            yield module_name, module, adapter.name

    def get_module_weights(self, model: torch.nn.Module) -> dict[str, Optional[torch.Tensor]]:
        """Get weights for all supported modules keyed by canonical name."""
        weights = {}
        for module_name, module, _ in self.iter_supported_modules(model):
            adapter = self.select(module)
            if adapter is None:
                continue
            weight = adapter.get_weight(module)
            if weight is not None:
                canonical_name = adapter.get_canonical_name(module, module_name)
                weights[canonical_name] = weight
        return weights
