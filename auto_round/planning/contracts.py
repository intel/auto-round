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

import copy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

from auto_round.schemes import QuantizationScheme

LayerConfig = Mapping[str, Mapping[str, Any]]
BlockGroups = Tuple[Tuple[str, ...], ...]


def freeze_mapping(value: Optional[LayerConfig]) -> LayerConfig:
    """Return an isolated, read-only snapshot of a layer configuration mapping."""
    frozen = {name: MappingProxyType(copy.deepcopy(dict(config))) for name, config in dict(value or {}).items()}
    return MappingProxyType(frozen)


def freeze_block_groups(value: Optional[Tuple[Tuple[str, ...], ...]]) -> Optional[BlockGroups]:
    """Freeze the two-dimensional block grouping used by model traversal."""
    if value is None:
        return None
    return tuple(tuple(group) for group in value)


@dataclass(frozen=True)
class CompressionIntent:
    format: Optional[str] = None
    layer_config: LayerConfig = field(default_factory=lambda: MappingProxyType({}))
    scale_dtype: Any = None
    quant_block_list: Optional[BlockGroups] = None
    mllm: bool = False
    iters: int = 0
    enable_alg_ext: bool = False
    quant_nontext_module: bool = False
    platform: Optional[str] = None
    is_auto_scheme: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_config", freeze_mapping(self.layer_config))
        object.__setattr__(self, "quant_block_list", freeze_block_groups(self.quant_block_list))


@dataclass(frozen=True)
class ResolvedScheme:
    _value: QuantizationScheme
    preset_name: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "_value", copy.deepcopy(self._value))

    @property
    def value(self) -> QuantizationScheme:
        """Return an isolated scheme value so callers cannot mutate this DTO."""
        return copy.deepcopy(self._value)

    @classmethod
    def from_scheme(cls, value: QuantizationScheme, preset_name: Optional[str] = None) -> "ResolvedScheme":
        return cls(_value=value, preset_name=preset_name)


@dataclass(frozen=True)
class FormatResolution:
    formats: Tuple[Any, ...]
    scheme: ResolvedScheme
    layer_policy: Any = None
    layer_config_patch: LayerConfig = field(default_factory=lambda: MappingProxyType({}))
    scale_dtype: Any = None
    quant_block_list: Optional[BlockGroups] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "formats", tuple(self.formats))
        object.__setattr__(self, "layer_config_patch", freeze_mapping(self.layer_config_patch))
        object.__setattr__(self, "quant_block_list", freeze_block_groups(self.quant_block_list))


@dataclass(frozen=True)
class CompressionPlan:
    scheme: ResolvedScheme
    formats: Tuple[Any, ...]
    layer_config: LayerConfig
    regex_config: LayerConfig = field(default_factory=lambda: MappingProxyType({}))
    has_qlayer_outside_block: bool = False
    scale_dtype: Any = None
    quant_block_list: Optional[BlockGroups] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "formats", tuple(self.formats))
        object.__setattr__(self, "layer_config", freeze_mapping(self.layer_config))
        object.__setattr__(self, "regex_config", freeze_mapping(self.regex_config))
        object.__setattr__(self, "quant_block_list", freeze_block_groups(self.quant_block_list))
