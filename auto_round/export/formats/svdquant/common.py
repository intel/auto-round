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

"""Logical SVDQuant records and adapter contracts shared by runtime exporters.

This module has no dependency on runtime packing or checkpoint layouts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

import torch

from auto_round.algorithms.transforms.svdquant.wrapper import SVDQuantLinear
from auto_round.wrapper import WrapperWALayer


@dataclass(frozen=True)
class SVDQuantLinearScheme:
    """AutoRound-selected weight and activation scheme values."""

    data_type: str | None
    bits: int | None
    group_size: int | tuple[int, int] | None
    sym: bool | None
    act_data_type: str | None
    act_bits: int | None
    act_group_size: int | tuple[int, int] | None
    act_sym: bool | None
    act_dynamic: bool | None


@dataclass(frozen=True)
class SourceLinearRecord:
    """Logical tensors from one source ``SVDQuantLinear`` before adapter mapping."""

    name: str
    residual_weight: torch.Tensor
    lora_down: torch.Tensor
    lora_up: torch.Tensor
    smooth: torch.Tensor
    smooth_orig: torch.Tensor
    bias: torch.Tensor | None
    scheme: SVDQuantLinearScheme


@dataclass(frozen=True)
class SVDQuantExportRecord:
    """Adapter-selected logical tensors and their exported key mapping.

    ``sources`` tracks the original projections used to build this record.
    Adapters may rename, fuse, or split projections, but must preserve their rank.
    Projections with shared input factors can be fused without another SVD;
    otherwise the adapter must recompose and decompose at the original rank.
    """

    prefix: str
    residual_weight: torch.Tensor
    lora_down: torch.Tensor
    lora_up: torch.Tensor
    smooth: torch.Tensor
    smooth_orig: torch.Tensor
    bias: torch.Tensor | None
    scheme: SVDQuantLinearScheme
    sources: tuple[SourceLinearRecord, ...]
    key_mapping: Mapping[str, str] = field(default_factory=dict)
    omitted_suffixes: frozenset[str] = field(default_factory=frozenset)


class SVDQuantModelAdapter(Protocol):
    """Boundary for model-level mapping and runtime metadata.

    ``map_modules`` receives every logical source together so architecture adapters
    can recompose effective weights before sibling fusion or splitting. Its output
    must retain source provenance and use the configured source rank.
    """

    def map_modules(
        self, model: torch.nn.Module, records: Iterable[SourceLinearRecord]
    ) -> Iterable[SVDQuantExportRecord]: ...

    def metadata(self, model: torch.nn.Module, rank: int) -> Mapping[str, str]: ...

    def extra_tensors(self, model: torch.nn.Module) -> Mapping[str, torch.Tensor]: ...

    def validate_records(
        self, sources: tuple[SourceLinearRecord, ...], records: tuple[SVDQuantExportRecord, ...]
    ) -> None: ...

    def validate(self, tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, str]) -> None: ...


class IdentitySVDQuantModelAdapter:
    """Map modules unchanged and explicitly identify a generic intermediate."""

    def map_modules(
        self, model: torch.nn.Module, records: Iterable[SourceLinearRecord]
    ) -> Iterable[SVDQuantExportRecord]:
        return (
            SVDQuantExportRecord(
                prefix=record.name or "model",
                residual_weight=record.residual_weight,
                lora_down=record.lora_down,
                lora_up=record.lora_up,
                smooth=record.smooth,
                smooth_orig=record.smooth_orig,
                bias=record.bias,
                scheme=record.scheme,
                sources=(record,),
            )
            for record in records
        )

    def metadata(self, model: torch.nn.Module, rank: int) -> Mapping[str, str]:
        return {"artifact_type": "generic_intermediate"}

    def extra_tensors(self, model: torch.nn.Module) -> Mapping[str, torch.Tensor]:
        return {}

    def validate_records(
        self, sources: tuple[SourceLinearRecord, ...], records: tuple[SVDQuantExportRecord, ...]
    ) -> None:
        return

    def validate(self, tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, str]) -> None:
        if metadata.get("artifact_type") != "generic_intermediate":
            raise ValueError("identity adapter output must be marked as a generic intermediate")


def collect_source_records(model: torch.nn.Module) -> tuple[SourceLinearRecord, ...]:
    """Read logical tensors and schemes from every SVDQuantLinear in the model."""
    records = []
    for name, module in model.named_modules():
        if not isinstance(module, SVDQuantLinear):
            continue
        residual_linear = module.residual_linear
        while isinstance(residual_linear, WrapperWALayer):
            residual_linear = residual_linear.orig_layer
        records.append(
            SourceLinearRecord(
                name=name,
                residual_weight=residual_linear.weight.detach(),
                lora_down=module.lora_down.weight.detach(),
                lora_up=module.lora_up.weight.detach(),
                smooth=module.smooth.detach(),
                smooth_orig=getattr(module, "smooth_orig", module.smooth).detach(),
                bias=None if residual_linear.bias is None else residual_linear.bias.detach(),
                scheme=SVDQuantLinearScheme(
                    data_type=getattr(residual_linear, "data_type", None),
                    bits=getattr(residual_linear, "bits", None),
                    group_size=getattr(residual_linear, "group_size", None),
                    sym=getattr(residual_linear, "sym", None),
                    act_data_type=getattr(residual_linear, "act_data_type", None),
                    act_bits=getattr(residual_linear, "act_bits", None),
                    act_group_size=getattr(residual_linear, "act_group_size", None),
                    act_sym=getattr(residual_linear, "act_sym", None),
                    act_dynamic=getattr(residual_linear, "act_dynamic", None),
                ),
            )
        )
    if not records:
        raise ValueError("No SVDQuantLinear modules found to export.")
    return tuple(records)


def export_key(record: SVDQuantExportRecord, suffix: str) -> str:
    mapped_suffix = record.key_mapping.get(suffix, suffix)
    if not isinstance(mapped_suffix, str) or not mapped_suffix:
        raise ValueError(f"{record.prefix} key mapping for {suffix!r} must be a non-empty string")
    return f"{record.prefix}.{mapped_suffix}"


def validate_adapter_provenance(
    sources: tuple[SourceLinearRecord, ...], records: tuple[SVDQuantExportRecord, ...]
) -> None:
    """Check that mapped records use known sources, keep their rank, and omit none."""

    source_ids = {id(source) for source in sources}
    referenced_ids: set[int] = set()
    for record in records:
        output_rank = record.lora_down.shape[0]
        for source in record.sources:
            if id(source) not in source_ids:
                raise ValueError(f"{record.prefix} adapter record references a foreign logical source")
            if source.lora_down.shape[0] != output_rank:
                raise ValueError(
                    f"{record.prefix} source rank={source.lora_down.shape[0]} must equal output rank={output_rank}; "
                    "the Nunchaku schema stores one configured rank, so adapters must recompose effective source "
                    "weights and decompose them at that configured rank; exact rank-sum fusion is unsupported"
                )
            referenced_ids.add(id(source))
    missing = [source.name or "<root>" for source in sources if id(source) not in referenced_ids]
    if missing:
        raise ValueError(f"model adapter dropped logical sources: {missing}")
