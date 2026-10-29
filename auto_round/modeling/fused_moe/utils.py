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

import torch
import torch.nn as nn

from auto_round import envs


def _update_parameter(
    module: torch.nn.Module,
    name: str,
    data: torch.Tensor,
) -> None:
    old_param = getattr(module, name)
    new_param = nn.Parameter(data, requires_grad=old_param.requires_grad)
    setattr(module, name, new_param)


def is_fused_layout(original: torch.nn.Module) -> bool:
    """Check if MoE experts use fused gate_up_proj/down_proj layout."""
    return hasattr(original, "gate_up_proj") and hasattr(original, "down_proj")


def is_linearized_layout(original: torch.nn.Module) -> bool:
    """Check if MoE experts use linearized layout with individual gate_proj/up_proj/down_proj."""
    if not isinstance(original, torch.nn.ModuleList) or len(original) == 0:
        return False
    first_expert = original[0]
    return all(hasattr(first_expert, attr) for attr in ("gate_proj", "up_proj", "down_proj"))


def _prepare_moe_route_batches(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    """Sort token-expert pairs once and return contiguous batches for active experts."""
    num_tokens = hidden_states.size(0)
    top_k = top_k_index.size(-1)
    token_indices = torch.arange(num_tokens, device=hidden_states.device).unsqueeze(1).expand(-1, top_k).reshape(-1)
    expert_ids = top_k_index.reshape(-1)
    sample_weights = top_k_weights.reshape(-1).to(hidden_states.dtype)

    sorted_expert_ids, permutation = torch.sort(expert_ids)
    if sorted_expert_ids.numel() == 0:
        return (
            hidden_states.new_empty((0, hidden_states.size(-1))),
            sample_weights,
            permutation,
            [],
        )

    active_experts, counts = torch.unique_consecutive(sorted_expert_ids, return_counts=True)
    route_plan = torch.stack((active_experts.to(torch.int64), counts.to(torch.int64)), dim=1).tolist()
    sorted_token_indices = token_indices.index_select(0, permutation)
    sorted_hidden_states = hidden_states.index_select(0, sorted_token_indices)
    sorted_weights = sample_weights.index_select(0, permutation)
    return sorted_hidden_states, sorted_weights, permutation, [(expert, count) for expert, count in route_plan]


def _restore_moe_route_order(
    sorted_outputs: torch.Tensor,
    permutation: torch.Tensor,
    num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    """Restore original top-k ordering before reducing routed expert outputs."""
    outputs = torch.zeros_like(sorted_outputs)
    outputs.index_copy_(0, permutation, sorted_outputs)
    return outputs.view(num_tokens, top_k, outputs.size(-1)).sum(dim=1)


def sequential_moe_forward(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    experts,
    num_experts: int,
) -> torch.Tensor:
    """Run each active expert on its contiguous batch of routed token-expert pairs.

    Sorting and counting routes once avoids scanning every route for every expert
    and avoids per-expert dynamic-shape ``nonzero``/gather operations. The route
    plan incurs one host synchronization to obtain active expert IDs and counts;
    expert inputs are contiguous slices, and outputs are scattered back once to
    preserve the original top-k reduction order.

    Args:
        hidden_states: (num_tokens, hidden_dim) input tensor.
        top_k_index: (num_tokens, top_k) selected expert indices.
        top_k_weights: (num_tokens, top_k) routing weights.
        experts: Indexable collection of per-expert callables (e.g. nn.ModuleList),
            each taking (num_samples, hidden_dim) and returning (num_samples, hidden_dim).
        num_experts: Total number of experts.

    Returns:
        final_hidden_states: (num_tokens, hidden_dim) output tensor.
    """
    hidden_dim = hidden_states.size(-1)
    num_tokens = hidden_states.size(0)
    top_k = top_k_index.size(-1)
    sorted_hidden_states, sorted_weights, permutation, route_plan = _prepare_moe_route_batches(
        hidden_states, top_k_index, top_k_weights
    )
    output_batches = []
    offset = 0
    for expert_idx, count in route_plan:
        expert_input = sorted_hidden_states[offset : offset + count]
        if 0 <= expert_idx < num_experts:
            expert_output = experts[expert_idx](expert_input).to(hidden_states.dtype)
        else:
            expert_output = torch.zeros_like(expert_input)
        output_batches.append(expert_output)
        offset += count

    if output_batches:
        sorted_outputs = torch.cat(output_batches, dim=0) * sorted_weights.unsqueeze(-1)
    else:
        sorted_outputs = sorted_hidden_states
    return _restore_moe_route_order(sorted_outputs, permutation, num_tokens, top_k)


def grouped_or_sequential_moe_forward(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    experts,
    num_experts: int,
) -> torch.Tensor:
    """Fast grouped-GEMM MoE forward on CUDA, single-sort sequential loop everywhere else.

    Custom fused-MoE replacements (Qwen3.5, etc.) split the fused 3D experts into a
    ``ModuleList`` of per-expert ``gate_proj``/``up_proj``/``down_proj`` linears -- exactly
    the layout :func:`grouped_linear_experts_forward` consumes. Routing through it batches
    the routed token/expert pairs into sorted grouped GEMMs (one launch for all experts,
    with batched fake-quant), which is markedly faster for tuning fwd+bwd on many-expert
    MoEs than the per-expert loop; it self-checks the layer and falls back to the loop when
    a layer is ineligible.

    Only taken on CUDA: the grouped/loop paths issue the per-expert gather kernels that
    trigger a driver-level bug on some XPU builds, which :func:`sequential_moe_forward`
    (a single sort, no per-expert gather) is written to avoid. It also requires the experts
    container to expose ``act_fn`` / ``_apply_gate`` (the grouped path applies gating at the
    container level), so containers without one keep the sequential path.

    Gated by ``AR_MOE_EXPERTS_IMPL``: setting it to ``linear_loop`` forces the sequential
    path even on CUDA; ``auto`` / ``linear_grouped`` / ``linear_grouped_sliced`` take grouped.
    """
    from auto_round.modeling.fused_moe.moe_experts_interface import (
        LINEAR_LOOP_IMPL,
        resolve_experts_implementation,
    )

    use_grouped = (
        hidden_states.device.type == "cuda"
        and (hasattr(experts, "act_fn") or hasattr(experts, "_apply_gate"))
        and resolve_experts_implementation() != LINEAR_LOOP_IMPL
    )
    if use_grouped:
        from auto_round.modeling.fused_moe.grouped_experts import grouped_linear_experts_forward

        return grouped_linear_experts_forward(experts, hidden_states, top_k_index, top_k_weights)
    return sequential_moe_forward(hidden_states, top_k_index, top_k_weights, experts, num_experts)


def get_num_experts(original: torch.nn.Module) -> int:
    """Get the number of experts from either fused or linearized layout."""
    if is_fused_layout(original):
        return original.gate_up_proj.shape[0]
    if is_linearized_layout(original):
        # Count only numeric keys (expert modules), exclude 'act_fn' etc.
        if hasattr(original, "_modules"):
            numeric_keys = [k for k in original._modules.keys() if k.isdigit()]
            return len(numeric_keys)
        return len(original)
    raise AttributeError(
        "Unsupported MoE experts layout: expected fused gate_up_proj/down_proj "
        "or linearized gate_proj/up_proj/down_proj experts"
    )


def force_all_experts_routing_enabled() -> bool:
    """Return whether MoE wrappers should override router top-k for coverage."""
    return envs.AR_FORCE_MOE_ROUTING_ALL_EXPERTS


def build_forced_expert_indices(
    *,
    module: torch.nn.Module,
    num_tokens: int,
    top_k: int,
    num_experts: int,
    device: torch.device,
) -> torch.Tensor:
    """Build deterministic rotating expert indices to cover all experts over time."""
    if num_tokens <= 0:
        raise ValueError("num_tokens must be > 0")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if num_experts <= 0:
        raise ValueError("num_experts must be > 0")

    top_k = min(top_k, num_experts)
    total_pairs = num_tokens * top_k
    offset = int(getattr(module, "_auto_round_forced_route_offset", 0))
    forced = (torch.arange(total_pairs, device=device) + offset) % num_experts
    forced = forced.view(num_tokens, top_k)
    module._auto_round_forced_route_offset = (offset + total_pairs) % num_experts
    return forced


def build_forced_routing(
    *,
    module: torch.nn.Module,
    routing_scores: torch.Tensor | None,
    top_k: int,
    num_experts: int,
    dtype: torch.dtype,
    num_tokens: int | None = None,
    device: torch.device | None = None,
    normalize: bool = True,
    scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create forced top-k indices and matching routing weights.

    When scores are unavailable, fall back to uniform weights.
    """
    if routing_scores is None:
        if num_tokens is None or device is None:
            raise ValueError("num_tokens and device are required when routing_scores is None")
    else:
        num_tokens = int(routing_scores.shape[0])
        device = routing_scores.device

    assert num_tokens is not None
    assert device is not None
    forced_indices = build_forced_expert_indices(
        module=module,
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_experts,
        device=device,
    )

    if routing_scores is not None:
        weights = routing_scores.gather(1, forced_indices)
    else:
        denom = float(min(top_k, num_experts))
        weights = torch.full(
            (num_tokens, min(top_k, num_experts)),
            1.0 / denom,
            device=forced_indices.device,
            dtype=torch.float32,
        )

    if normalize:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    if scaling_factor != 1.0:
        weights = weights * scaling_factor

    return forced_indices, weights.to(dtype)
