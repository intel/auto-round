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

import torch

from auto_round.algorithms.transforms.svdquant.smooth_adapters.base import (
    SmoothSearchGroup,
    TargetPredicate,
    generic_linear_groups,
    module_global_name,
    resolve_module,
)

_DOUBLE_GROUPS = (
    ("attn.qkv", ("attn.to_q", "attn.to_k", "attn.to_v"), "attn", (0,), (3,)),
    (
        "attn.add_qkv",
        ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"),
        "attn",
        (1,),
        (3,),
    ),
    ("attn.to_out.0", ("attn.to_out.0",), "attn.to_out.0", None, (1,)),
    ("attn.to_add_out", ("attn.to_add_out",), "attn.to_add_out", None, (1,)),
    ("ff.net.0.proj", ("ff.net.0.proj",), "ff.net.0.proj", None, (1,)),
    ("ff.net.2", ("ff.net.2",), "ff.net.2", None, (1,)),
    ("ff_context.net.0.proj", ("ff_context.net.0.proj",), "ff_context.net.0.proj", None, (1,)),
    ("ff_context.net.2", ("ff_context.net.2",), "ff_context.net.2", None, (1,)),
)


def supports_flux_block(block: torch.nn.Module) -> bool:
    return block.__class__.__name__ in {"FluxTransformerBlock", "FluxSingleTransformerBlock"}


def _make_group(
    block: torch.nn.Module,
    key: str,
    projection_paths: tuple[str, ...],
    evaluation_path: str,
    output_indices: tuple[int, ...] | None,
    output_splits: tuple[int, ...],
    is_target: TargetPredicate,
) -> SmoothSearchGroup | None:
    selected: list[tuple[str, torch.nn.Linear]] = []
    for path in projection_paths:
        module = resolve_module(block, path)
        if isinstance(module, torch.nn.Linear) and is_target(path, module):
            selected.append((path, module))
    if not selected:
        return None

    widths = {module.in_features for _, module in selected}
    if len(widths) != 1:
        raise ValueError(f"Flux smooth-search group {module_global_name(block, key)!r} has mixed input widths")
    evaluation_module = block if not evaluation_path else resolve_module(block, evaluation_path)
    if evaluation_module is None:
        raise ValueError(f"Flux smooth-search group {module_global_name(block, key)!r} has no evaluation module")
    names = tuple(module_global_name(block, path) for path, _ in selected)
    projections = tuple(module for _, module in selected)
    splits = output_splits if sum(output_splits) == len(projections) else (len(projections),)
    return SmoothSearchGroup(
        key=module_global_name(block, key),
        projection_names=names,
        projections=projections,
        projection_input_key=names[0],
        projection_input_module=projections[0],
        evaluation_input_key=module_global_name(block, evaluation_path),
        evaluation_module=evaluation_module,
        output_indices=output_indices,
        output_splits=splits,
    )


def discover_flux_groups(block: torch.nn.Module, is_target: TargetPredicate) -> list[SmoothSearchGroup]:
    groups: list[SmoothSearchGroup] = []
    if block.__class__.__name__ == "FluxSingleTransformerBlock":
        specifications = (
            (
                "parallel_qkv_mlp",
                ("attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp"),
                "",
                None,
                (3, 1),
            ),
            ("proj_out", ("proj_out",), "proj_out", None, (1,)),
        )
    else:
        specifications = _DOUBLE_GROUPS

    for specification in specifications:
        group = _make_group(block, *specification, is_target)
        if group is not None:
            groups.append(group)

    consumed = {id(projection) for group in groups for projection in group.projections}
    groups.extend(generic_linear_groups(block, is_target, consumed=consumed))
    return groups
