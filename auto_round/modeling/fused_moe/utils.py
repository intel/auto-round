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


def sequential_moe_forward(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    experts,
    num_experts: int,
) -> torch.Tensor:
    """Sequential per-expert MoE forward using a single sort to group tokens by expert.

    Repeatedly issuing dynamic-shape gather kernels (``nonzero()``,
    ``index_select()`` or boolean-mask indexing) once per expert in a Python
    loop reliably triggers a driver-level bug on some XPU builds - either a
    "vectorized gather kernel index out of bounds" device-side assertion or a
    hard ``UR_RESULT_ERROR_DEVICE_LOST`` - even when every index is valid.
    Sorting once needs only two dynamic gather/scatter calls in total
    (regardless of ``num_experts``): one ``index_select`` to group tokens by
    expert, and one ``index_copy_`` to scatter results back to their original
    positions. Each expert's slice ``permuted[start:end]`` is a plain view (no
    gather kernel involved), so this is also efficient.

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
    device = hidden_states.device

    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, top_k).reshape(-1)  # (S,)
    sample_weights = top_k_weights.reshape(-1).to(hidden_states.dtype)  # (S,)
    expert_ids = top_k_index.reshape(-1)  # (S,)

    selected_hidden_states = hidden_states[token_idx]  # (S, hidden_dim)

    sort_order = torch.argsort(expert_ids)
    permuted_hidden_states = selected_hidden_states.index_select(0, sort_order)  # (S, hidden_dim)

    # Per-expert token counts, computed once on host to drive static slicing.
    counts = torch.bincount(expert_ids, minlength=num_experts).tolist()

    out_permuted = torch.zeros_like(permuted_hidden_states)
    start = 0
    for expert_idx, count in enumerate(counts):
        if count == 0:
            continue
        end = start + count
        expert_input = permuted_hidden_states[start:end]  # static slice/view, no gather kernel
        out_permuted[start:end] = experts[expert_idx](expert_input).to(out_permuted.dtype)
        start = end

    # Scatter results back to their original (pre-sort) token-expert order.
    out_per_sample = torch.empty_like(out_permuted)
    out_per_sample.index_copy_(0, sort_order, out_permuted)

    # Apply routing weights
    out_per_sample = out_per_sample * sample_weights.unsqueeze(-1)  # (S, hidden_dim)

    # Accumulate results using deterministic reshape+sum instead of index_add_
    # (index_add_ with duplicate indices is non-deterministic on CUDA due to atomicAdd)
    final_hidden_states = out_per_sample.view(num_tokens, top_k, hidden_dim).sum(dim=1)

    return final_hidden_states


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
