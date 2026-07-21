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

"""Helpers for mapping and independently packing restored GGUF MoE outputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gguf
import numpy as np
import torch

_PROJECTION_ALIASES = {
    gguf.MODEL_TENSOR.FFN_GATE_EXP: ("gate_proj", "w1", "linear"),
    gguf.MODEL_TENSOR.FFN_DOWN_EXP: ("down_proj", "w2", "linear_1"),
    gguf.MODEL_TENSOR.FFN_UP_EXP: ("up_proj", "w3", "linear_v"),
}


@dataclass(frozen=True)
class GGUFMoEOutput:
    """The source projection and expert names for one GGUF MoE output."""

    projection: str
    hf_names: tuple[str, ...]


def resolve_moe_output(cls, restored: Any, new_name: str, bid: int | None) -> GGUFMoEOutput | None:
    """Resolve one GGUF expert output to exactly one restored source projection."""

    sources = restored.moe_sources
    if not sources:
        return None

    candidates = []
    for tensor_type, aliases in _PROJECTION_ALIASES.items():
        if cls.match_model_tensor_name(new_name, tensor_type, bid):
            candidates.extend(source for source in sources if source.projection in aliases)
    if len(candidates) != 1:
        raise ValueError(f"{new_name}: cannot map GGUF MoE output to checkpoint source projections")

    source = candidates[0]
    return GGUFMoEOutput(projection=source.projection, hf_names=source.hf_names)


def _expert_context(context: str, projection: str, expert_index: int) -> str:
    return f"{context}, projection {projection}, expert {expert_index}"


def pack_moe_output(
    data_torch: torch.Tensor,
    data_qtype: gguf.GGMLQuantizationType,
    output: GGUFMoEOutput,
    quantize_fn: Callable[[torch.Tensor, str], tuple[np.ndarray, gguf.GGMLQuantizationType]],
    context: str,
) -> tuple[np.ndarray, gguf.GGMLQuantizationType]:
    """Pack each expert independently and stack byte-identical callback results."""

    if data_torch.ndim != 3:
        error_context = _expert_context(context, output.projection, 0)
        raise ValueError(f"{error_context}: expected a three-dimensional MoE tensor, got {tuple(data_torch.shape)}")
    if data_torch.shape[0] != len(output.hf_names):
        expert_index = min(data_torch.shape[0], len(output.hf_names))
        error_context = _expert_context(context, output.projection, expert_index)
        raise ValueError(
            f"{error_context}: expert count mismatch: tensor has {data_torch.shape[0]}, "
            f"source lineage has {len(output.hf_names)}"
        )

    packed = []
    packed_layout = None
    for expert_index, (expert_tensor, hf_name) in enumerate(zip(data_torch, output.hf_names)):
        expert_packed, expert_qtype = quantize_fn(expert_tensor, hf_name)
        error_context = _expert_context(context, output.projection, expert_index)
        if not isinstance(expert_packed, np.ndarray):
            raise ValueError(
                f"{error_context}: packed result must be a NumPy ndarray, got {type(expert_packed).__name__}"
            )
        if expert_qtype != data_qtype:
            raise ValueError(f"{error_context}: packed qtype mismatch: expected {data_qtype}, got {expert_qtype}")
        expert_layout = (expert_packed.dtype, expert_packed.shape, expert_packed.nbytes)
        if packed_layout is None:
            packed_layout = expert_layout
        elif expert_layout != packed_layout:
            raise ValueError(
                f"{error_context}: packed byte-layout mismatch: expected dtype {packed_layout[0]}, "
                f"shape {packed_layout[1]}, and {packed_layout[2]} bytes; got dtype {expert_layout[0]}, "
                f"shape {expert_layout[1]}, and {expert_layout[2]} bytes"
            )
        packed.append(expert_packed)

    return np.stack(packed, axis=0), data_qtype


def validate_moe_source_qtypes(
    hf_names: tuple[str, ...],
    fallback_qtype: gguf.GGMLQuantizationType,
    resolve_explicit_qtype: Callable[[str], gguf.GGMLQuantizationType | None],
    context: str,
) -> gguf.GGMLQuantizationType:
    """Select one qtype only when all expert sources agree."""

    explicit = [resolve_explicit_qtype(name) for name in hf_names]
    present = [qtype for qtype in explicit if qtype is not None]
    if not present:
        return fallback_qtype
    if len(present) != len(explicit) or len(set(present)) != 1:
        raise ValueError(f"{context}: MoE source qtypes are missing or inconsistent: {explicit}")
    return present[0]


def validate_moe_imatrices(
    hf_names: tuple[str, ...],
    resolve_module: Callable[[str], Any],
    context: str,
) -> None:
    """Validate the all-absent or all-present per-source imatrix contract."""

    modules = [resolve_module(name) for name in hf_names]
    imatrices = [getattr(module, "imatrix", None) for module in modules]
    if not any(imatrix is not None for imatrix in imatrices):
        return

    for expert_index, (module, imatrix) in enumerate(zip(modules, imatrices)):
        error_context = f"{context}, expert {expert_index}"
        if imatrix is None:
            raise ValueError(f"{error_context}: MoE source imatrix is missing")
        expected_length = module.weight.shape[-1]
        if not isinstance(imatrix, torch.Tensor) or imatrix.ndim != 1 or imatrix.shape[0] != expected_length:
            shape = tuple(imatrix.shape) if isinstance(imatrix, torch.Tensor) else type(imatrix).__name__
            raise ValueError(
                f"{error_context}: MoE source imatrix must be one-dimensional with length "
                f"{expected_length}, got {shape}"
            )
