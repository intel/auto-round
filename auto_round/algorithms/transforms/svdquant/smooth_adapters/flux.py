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

import torch

from auto_round.algorithms.transforms.svdquant.smooth_adapters.base import (
    SmoothSearchGroup,
    TargetPredicate,
    generic_linear_groups,
    module_global_name,
    resolve_module,
)
from auto_round.logger import logger

_VERIFIED_FLUX_MODEL_IDS = ("black-forest-labs/flux.1-dev", "flux.1-dev")


@dataclass(frozen=True)
class FluxSmoothGroupSpec:
    """Declarative FLUX projection grouping and output-error evaluation route."""

    key: str
    projection_paths: tuple[str, ...]
    evaluation_path: str
    output_indices: tuple[int, ...] | None


_DOUBLE_GROUPS = (
    FluxSmoothGroupSpec("attn.qkv", ("attn.to_q", "attn.to_k", "attn.to_v"), "attn", (0,)),
    FluxSmoothGroupSpec("attn.add_qkv", ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"), "attn", (1,)),
    FluxSmoothGroupSpec("attn.to_out.0", ("attn.to_out.0",), "attn.to_out.0", None),
    FluxSmoothGroupSpec("attn.to_add_out", ("attn.to_add_out",), "attn.to_add_out", None),
    FluxSmoothGroupSpec("ff.net.0.proj", ("ff.net.0.proj",), "ff.net.0.proj", None),
    FluxSmoothGroupSpec("ff.net.2", ("ff.net.2",), "ff.net.2", None),
    FluxSmoothGroupSpec("ff_context.net.0.proj", ("ff_context.net.0.proj",), "ff_context.net.0.proj", None),
    FluxSmoothGroupSpec("ff_context.net.2", ("ff_context.net.2",), "ff_context.net.2", None),
)

_SINGLE_GROUPS = (
    FluxSmoothGroupSpec("parallel_qkv_mlp", ("attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp"), "", None),
    FluxSmoothGroupSpec("proj_out", ("proj_out",), "proj_out", None),
)


def supports_flux_block(block: torch.nn.Module) -> bool:
    return block.__class__.__name__ in {"FluxTransformerBlock", "FluxSingleTransformerBlock"}


def warn_if_unverified_flux_model(model: torch.nn.Module) -> None:
    """Warn when FLUX grouping is used outside the model validated by this adapter."""
    config = getattr(model, "config", None)
    if hasattr(config, "get"):
        model_id = config.get("_name_or_path", "")
    else:
        model_id = getattr(config, "_name_or_path", "")
    normalized = str(model_id).lower()
    if not any(verified_id in normalized for verified_id in _VERIFIED_FLUX_MODEL_IDS):
        logger.warning_once(
            "SVDQuant FLUX grouping has been validated with FLUX.1-dev only; "
            "model %r is unverified and may require a dedicated adapter.",
            model_id or type(model).__name__,
        )


def _make_group(
    block: torch.nn.Module,
    specification: FluxSmoothGroupSpec,
    is_target: TargetPredicate,
) -> SmoothSearchGroup | None:
    selected = []
    for path in specification.projection_paths:
        module = resolve_module(block, path)
        if isinstance(module, torch.nn.Linear) and is_target(path, module):
            selected.append((path, module))
    if not selected:
        return None

    evaluation_module = (
        block if not specification.evaluation_path else resolve_module(block, specification.evaluation_path)
    )
    if evaluation_module is None:
        raise ValueError(
            f"Flux SVDQuant group {module_global_name(block, specification.key)!r} has no evaluation module."
        )
    names = tuple(module_global_name(block, path) for path, _ in selected)
    projections = tuple(module for _, module in selected)
    return SmoothSearchGroup(
        key=module_global_name(block, specification.key),
        projection_names=names,
        projections=projections,
        projection_input_key=names[0],
        projection_input_module=projections[0],
        evaluation_input_key=module_global_name(block, specification.evaluation_path),
        evaluation_module=evaluation_module,
        output_indices=specification.output_indices,
    )


def discover_flux_groups(block: torch.nn.Module, is_target: TargetPredicate) -> list[SmoothSearchGroup]:
    specifications = _SINGLE_GROUPS if block.__class__.__name__ == "FluxSingleTransformerBlock" else _DOUBLE_GROUPS

    groups = []
    for specification in specifications:
        group = _make_group(block, specification, is_target)
        if group is not None:
            groups.append(group)
    consumed = {id(projection) for group in groups for projection in group.projections}
    groups.extend(generic_linear_groups(block, is_target, consumed=consumed))
    return groups
