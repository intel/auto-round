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

"""Format-neutral descriptions and lazy views for reversible MoE fusion."""

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

_MOE_FUSION_SPEC_ATTRIBUTE = "_auto_round_moe_fusion_spec"


@dataclass(frozen=True)
class ProjectionFusionSpec:
    """Describe how live expert projections form one checkpoint projection."""

    checkpoint_projection: str
    source_projections: tuple[str, ...]
    concat_dim: int | None
    checkpoint_transposed: bool = False
    checkpoint_bias: str | None = None


@dataclass(frozen=True)
class MoEFusionSpec:
    """Describe all reversible projections for one MoE experts module."""

    num_experts: int
    projections: tuple[ProjectionFusionSpec, ...]


@dataclass(frozen=True)
class MoETensorSource:
    """Identify the live tensors that contribute to a lazy checkpoint view."""

    projection: str
    hf_names: tuple[str, ...]
    parameter: str = "weight"


@dataclass(frozen=True)
class MoEFusionView:
    """Expose one lazily reconstructed checkpoint tensor and its live sources."""

    checkpoint_name: str
    tensor_fn: Callable[[], torch.Tensor]
    sources: tuple[MoETensorSource, ...]


def register_moe_fusion_spec(module: nn.Module, spec: MoEFusionSpec) -> None:
    """Attach a fusion specification as plain module metadata."""

    object.__setattr__(module, _MOE_FUSION_SPEC_ATTRIBUTE, spec)


def get_moe_fusion_spec(module: nn.Module) -> MoEFusionSpec | None:
    """Return fusion metadata previously registered on ``module``."""

    return getattr(module, _MOE_FUSION_SPEC_ATTRIBUTE, None)


def build_standard_moe_fusion_spec(
    detected_projections: Mapping[str, Mapping[str, Any]],
    num_experts: int,
    checkpoint_transposed: bool,
    module: nn.Module,
) -> MoEFusionSpec:
    """Build fusion metadata from the standard fused-MoE projection description."""

    projections = []
    for checkpoint_projection, config in detected_projections.items():
        source_projections = tuple(config.get("split_into") or (checkpoint_projection,))
        checkpoint_bias_name = f"{checkpoint_projection}_bias"
        checkpoint_bias = checkpoint_bias_name if getattr(module, checkpoint_bias_name, None) is not None else None
        projections.append(
            ProjectionFusionSpec(
                checkpoint_projection=checkpoint_projection,
                source_projections=source_projections,
                concat_dim=1 if len(source_projections) > 1 else None,
                checkpoint_transposed=checkpoint_transposed,
                checkpoint_bias=checkpoint_bias,
            )
        )
    return MoEFusionSpec(num_experts=num_experts, projections=tuple(projections))


def _qualified_name(module_path: str, name: str) -> str:
    return f"{module_path}.{name}" if module_path else name


def _get_module(module: nn.Module, module_path: str) -> nn.Module | None:
    try:
        return module.get_submodule(module_path)
    except (AttributeError, KeyError):
        return None


def _tensor_sources(
    module_path: str,
    projection_spec: ProjectionFusionSpec,
    num_experts: int,
    parameter: str = "weight",
) -> tuple[MoETensorSource, ...]:
    return tuple(
        MoETensorSource(
            projection=projection,
            hf_names=tuple(
                _qualified_name(module_path, f"{expert_index}.{projection}.{parameter}")
                for expert_index in range(num_experts)
            ),
            parameter=parameter,
        )
        for projection in projection_spec.source_projections
    )


def _context(module_path: str, projection: str, expert_index: int) -> str:
    return f"module '{module_path or '<root>'}', projection '{projection}', expert {expert_index}"


def _stack_source(
    model: nn.Module,
    module_path: str,
    source: MoETensorSource,
) -> torch.Tensor:
    tensors = []
    expected_shape = None
    for expert_index, hf_name in enumerate(source.hf_names):
        module_name = hf_name.rsplit(".", 1)[0]
        source_module = _get_module(model, module_name)
        tensor = getattr(source_module, source.parameter, None) if source_module is not None else None
        context = _context(module_path, source.projection, expert_index)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Missing {source.parameter} for {context}")
        expected_ndim = 2 if source.parameter == "weight" else 1
        if tensor.ndim != expected_ndim:
            raise ValueError(
                f"Incompatible {source.parameter} shape for {context}: "
                f"expected {expected_ndim} dimensions, got {tuple(tensor.shape)}"
            )
        if expected_shape is None:
            expected_shape = tensor.shape
        elif tensor.shape != expected_shape:
            raise ValueError(
                f"Incompatible {source.parameter} shape for {context}: "
                f"expected {tuple(expected_shape)}, got {tuple(tensor.shape)}"
            )
        tensors.append(tensor)
    return torch.stack(tensors, dim=0)


def _concatenate_sources(
    tensors: list[torch.Tensor],
    sources: tuple[MoETensorSource, ...],
    module_path: str,
    concat_dim: int | None,
) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0]
    if concat_dim is None:
        context = _context(module_path, sources[1].projection, 0)
        raise ValueError(f"Incompatible shape for {context}: multiple sources require concat_dim")

    reference = tensors[0]
    if not -reference.ndim <= concat_dim < reference.ndim:
        raise ValueError(
            f"Incompatible shape for {_context(module_path, sources[0].projection, 0)}: "
            f"concat dimension {concat_dim} is invalid for {tuple(reference.shape)}"
        )
    concat_dim %= reference.ndim
    for source, tensor in zip(sources[1:], tensors[1:]):
        if tensor.shape != reference.shape:
            raise ValueError(
                f"Incompatible shape for {_context(module_path, source.projection, 0)}: "
                f"expected {tuple(reference.shape)}, got {tuple(tensor.shape)}"
            )
    return torch.cat(tensors, dim=concat_dim)


def _build_tensor(
    model: nn.Module,
    module_path: str,
    projection_spec: ProjectionFusionSpec,
    sources: tuple[MoETensorSource, ...],
    parameter: str = "weight",
) -> torch.Tensor:
    tensors = [_stack_source(model, module_path, source) for source in sources]
    concat_dim = 1 if parameter == "bias" and len(tensors) > 1 else projection_spec.concat_dim
    tensor = _concatenate_sources(tensors, sources, module_path, concat_dim)
    if parameter == "weight" and projection_spec.checkpoint_transposed:
        if tensor.ndim != 3:
            raise ValueError(
                f"Incompatible shape for {_context(module_path, sources[0].projection, 0)}: "
                f"expected a three-dimensional weight, got {tuple(tensor.shape)}"
            )
        tensor = tensor.transpose(1, 2)
    return tensor


def iter_moe_fusion_views(model: nn.Module) -> Iterator[MoEFusionView]:
    """Yield lazy checkpoint tensor views for every registered MoE module."""

    named_modules = list(model.named_modules())
    spec_paths = {module_path for module_path, module in named_modules if get_moe_fusion_spec(module) is not None}
    for module_path, module in named_modules:
        if not getattr(module, "_auto_round_replaced_fused_moe", False):
            continue
        descendant_prefix = f"{module_path}." if module_path else ""
        has_spec = module_path in spec_paths or any(
            spec_path.startswith(descendant_prefix) for spec_path in spec_paths if spec_path
        )
        if not has_spec:
            raise ValueError(f"{module_path or '<root>'}: linearized fused MoE replacement has no MoEFusionSpec")

    for module_path, module in named_modules:
        fusion_spec = get_moe_fusion_spec(module)
        if fusion_spec is None:
            continue
        for projection_spec in fusion_spec.projections:
            weight_sources = _tensor_sources(module_path, projection_spec, fusion_spec.num_experts)
            yield MoEFusionView(
                checkpoint_name=_qualified_name(module_path, projection_spec.checkpoint_projection),
                tensor_fn=lambda module_path=module_path, projection_spec=projection_spec, sources=weight_sources: (
                    _build_tensor(model, module_path, projection_spec, sources)
                ),
                sources=weight_sources,
            )

            if projection_spec.checkpoint_bias is not None:
                bias_sources = _tensor_sources(module_path, projection_spec, fusion_spec.num_experts, "bias")
                yield MoEFusionView(
                    checkpoint_name=_qualified_name(module_path, projection_spec.checkpoint_bias),
                    tensor_fn=lambda module_path=module_path, projection_spec=projection_spec, sources=bias_sources: (
                        _build_tensor(model, module_path, projection_spec, sources, "bias")
                    ),
                    sources=bias_sources,
                )
