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

"""
Custom experts implementation for transformers' MOE integration.

This module provides a `linear_loop` experts implementation that uses
individual nn.Linear layers per expert instead of fused 3D Parameters.
This enables proper quantization of MOE expert weights.

The implementation integrates with transformers' `use_experts_implementation`
decorator and `ALL_EXPERTS_FUNCTIONS` registry.

Usage:

    from auto_round.modeling.fused_moe.moe_experts_interface import prepare_model_for_moe_quantization
    # Before quantization
    prepare_model_for_moe_quantization(model)

    # Now the model uses linear_loop forward which supports quantized nn.Linear layers
"""

from typing import Callable

import torch
from torch import nn

from auto_round.utils import clear_memory, logger

try:
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

    HAS_EXPERTS_INTERFACE = True
except ImportError:
    HAS_EXPERTS_FUNCTIONS = False
    ALL_EXPERTS_FUNCTIONS = None

# Track if we've logged linear_loop usage (to avoid spamming logs)
_linear_loop_logged = False


def linear_loop_experts_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Forward using individual nn.Linear layers per expert.

    This implementation loops over experts and uses self.gate_up_proj[i] and
    self.down_proj[i] as nn.Linear layers (or quantized equivalents), enabling
    proper quantization support.

    Expected module attributes:
        - gate_up_proj: nn.ModuleList of nn.Linear (in_features=hidden_dim, out_features=2*intermediate_dim)
        - down_proj: nn.ModuleList of nn.Linear (in_features=intermediate_dim, out_features=hidden_dim)
        - act_fn: activation function
        - num_experts: number of experts
        - _apply_gate: optional custom gating function

    Args:
        self: The experts module
        hidden_states: Input tensor of shape (num_tokens, hidden_dim)
        top_k_index: Selected expert indices of shape (num_tokens, top_k)
        top_k_weights: Expert weights of shape (num_tokens, top_k)

    Returns:
        final_hidden_states: Output tensor of shape (num_tokens, hidden_dim)
    """
    global _linear_loop_logged
    if not _linear_loop_logged:
        logger.info(f"Using linear_loop experts forward for {self.__class__.__name__}")
        _linear_loop_logged = True

    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)
    num_experts = self.num_experts

    # Reshape for easier indexing
    # S is the number of selected token-expert pairs (S = num_tokens * num_top_k)
    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1)  # (S,)
    sample_weights = top_k_weights.reshape(-1)  # (S,)
    expert_ids = top_k_index.reshape(-1)  # (S,)

    # Get current hidden states for selected samples
    selected_hidden_states = hidden_states[token_idx]  # (S, hidden_dim)

    # Allocate output tensor
    out_per_sample = torch.zeros(token_idx.size(0), hidden_dim, device=device, dtype=hidden_states.dtype)

    # Process each expert
    for expert_idx in range(num_experts):
        # Find samples routed to this expert
        mask = expert_ids == expert_idx
        if not mask.any():
            continue

        expert_input = selected_hidden_states[mask]  # (num_samples_for_expert, hidden_dim)

        # Use nn.Linear layers for this expert
        gate_up_out = self.gate_up_proj[expert_idx](expert_input)  # (num_samples, 2*intermediate_dim)

        # Apply gating
        if hasattr(self, "_apply_gate"):
            gated_out = self._apply_gate(gate_up_out)  # (num_samples, intermediate_dim)
        else:
            gate, up = gate_up_out.chunk(2, dim=-1)
            gated_out = self.act_fn(gate) * up

        # Down projection
        expert_out = self.down_proj[expert_idx](gated_out)  # (num_samples, hidden_dim)

        # Store results
        out_per_sample[mask] = expert_out.to(out_per_sample.dtype)

    # Apply routing weights
    out_per_sample = out_per_sample * sample_weights.unsqueeze(-1)  # (S, hidden_dim)

    # Accumulate results using deterministic reshape+sum instead of index_add_
    # (index_add_ with duplicate indices is non-deterministic on CUDA due to atomicAdd)
    final_hidden_states = out_per_sample.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)

    return final_hidden_states.to(hidden_states.dtype)


def register_linear_loop_experts() -> bool:
    """Register the linear_loop experts implementation with transformers.

    Returns:
        True if registration was successful, False otherwise.
    """
    if not HAS_EXPERTS_INTERFACE:
        logger.warning(
            "transformers.integrations.moe.ALL_EXPERTS_FUNCTIONS not available. "
            "linear_loop experts implementation not registered. "
            "Requires transformers >= 5.0.0"
        )
        return False

    if "linear_loop" not in ALL_EXPERTS_FUNCTIONS._global_mapping:
        ALL_EXPERTS_FUNCTIONS._global_mapping["linear_loop"] = linear_loop_experts_forward
        logger.debug("Registered 'linear_loop' experts implementation")

    return True


def _experts_supports_decorator(module: nn.Module) -> bool:
    """Check if experts module supports @use_experts_implementation decorator.

    Only experts classes decorated with @use_experts_implementation will use
    our linear_loop forward. Others need full module replacement.
    """
    forward_method = getattr(module.__class__, "forward", None)
    if forward_method is None:
        return False
    # @use_experts_implementation sets __wrapped__ on the decorated method
    return hasattr(forward_method, "__wrapped__")


def _unfuse_experts_weights_inplace(module: nn.Module, check_decorator: bool = True) -> bool:
    """Convert fused 3D expert weights to nn.ModuleList of nn.Linear layers.

    This function modifies the module in-place, replacing:
    - gate_up_proj: nn.Parameter(num_experts, ...) -> nn.ModuleList[nn.Linear]
    - down_proj: nn.Parameter(num_experts, ...) -> nn.ModuleList[nn.Linear]

    Args:
        module: The experts module to unfuse
        check_decorator: If True, only unfuse if the module supports
            @use_experts_implementation decorator. Default is True.

    Returns:
        True if unfusing was successful, False if module doesn't match pattern
    """
    # Check if this is a fused experts module
    if not hasattr(module, "gate_up_proj") or not isinstance(module.gate_up_proj, nn.Parameter):
        return False
    if not hasattr(module, "down_proj") or not isinstance(module.down_proj, nn.Parameter):
        return False
    if module.gate_up_proj.dim() != 3 or module.down_proj.dim() != 3:
        return False

    # Only unfuse if the module supports the decorator (unless check_decorator is False)
    # Modules that don't support the decorator (like Llama4TextExperts) should be
    # handled by full module replacement instead
    if check_decorator and not _experts_supports_decorator(module):
        logger.debug(f"Skipping unfuse for {module.__class__.__name__}: does not support @use_experts_implementation")
        return False

    gate_up_proj = module.gate_up_proj
    down_proj = module.down_proj
    num_experts = gate_up_proj.shape[0]

    # Detect if transposed (from decorator attributes or shape analysis)
    is_transposed = getattr(module, "is_transposed", None)
    has_bias = getattr(module, "has_bias", False)

    if is_transposed is None:
        # Infer from shape: gate_up has 2*intermediate in one dimension
        dim1, dim2 = gate_up_proj.shape[1], gate_up_proj.shape[2]
        is_transposed = dim1 < dim2  # transposed: (num_experts, hidden, 2*intermediate)

    if is_transposed:
        # Transposed: gate_up_proj(num_experts, hidden, 2*intermediate)
        hidden_dim = gate_up_proj.shape[1]
        intermediate_dim = gate_up_proj.shape[2] // 2
    else:
        # Not transposed: gate_up_proj(num_experts, 2*intermediate, hidden)
        intermediate_dim = gate_up_proj.shape[1] // 2
        hidden_dim = gate_up_proj.shape[2]

    # Get bias if present
    gate_up_bias = getattr(module, "gate_up_proj_bias", None)
    down_bias = getattr(module, "down_proj_bias", None)

    # Create nn.ModuleList of nn.Linear layers
    gate_up_linears = nn.ModuleList()
    down_linears = nn.ModuleList()

    dtype = gate_up_proj.dtype
    device = gate_up_proj.device if gate_up_proj.device.type != "meta" else "cpu"

    for i in range(num_experts):
        # Create gate_up linear (hidden_dim -> 2*intermediate_dim)
        gate_up_linear = nn.Linear(
            hidden_dim, 2 * intermediate_dim, bias=has_bias and gate_up_bias is not None, dtype=dtype, device=device
        )

        # Create down linear (intermediate_dim -> hidden_dim)
        down_linear = nn.Linear(
            intermediate_dim, hidden_dim, bias=has_bias and down_bias is not None, dtype=dtype, device=device
        )

        # Copy weights if not on meta device
        if gate_up_proj.device.type != "meta":
            if is_transposed:
                # gate_up: (hidden, 2*intermediate) -> need transpose for Linear (out, in)
                gate_up_linear.weight.data.copy_(gate_up_proj[i].t())
                # down: (intermediate, hidden) -> need transpose for Linear (out, in)
                down_linear.weight.data.copy_(down_proj[i].t())
            else:
                # gate_up: (2*intermediate, hidden) -> already (out, in) format
                gate_up_linear.weight.data.copy_(gate_up_proj[i])
                # down: (hidden, intermediate) -> already (out, in) format
                down_linear.weight.data.copy_(down_proj[i])

            if has_bias and gate_up_bias is not None:
                gate_up_linear.bias.data.copy_(gate_up_bias[i])
                down_linear.bias.data.copy_(down_bias[i])

        gate_up_linears.append(gate_up_linear)
        down_linears.append(down_linear)

    # Replace the fused parameters with ModuleLists
    # First, remove the old parameters
    del module.gate_up_proj
    del module.down_proj

    # Set the new ModuleLists
    module.gate_up_proj = gate_up_linears
    module.down_proj = down_linears

    # Ensure num_experts is set
    if not hasattr(module, "num_experts"):
        module.num_experts = num_experts

    return True


def prepare_model_for_moe_quantization(model: nn.Module, implementation: str = "linear_loop") -> list[str]:
    """Prepare a model for MOE quantization using transformers' experts interface.

    This function:
    1. Registers the linear_loop experts implementation with transformers
    2. Sets model.config._experts_implementation = implementation
    3. Unfuses all fused MOE expert weights into nn.ModuleList[nn.Linear]

    After calling this function, the model's forward pass will use individual
    nn.Linear layers per expert, which can be quantized normally.

    Args:
        model: The model to prepare
        implementation: The experts implementation to use (default: "linear_loop")

    Returns:
        List of module names that were unfused
    """
    # Register our custom implementation
    if not register_linear_loop_experts():
        raise RuntimeError(
            "Failed to register linear_loop experts implementation. "
            "This requires transformers >= 5.0.0 with MOE integration support."
        )

    # Unfuse all fused experts modules (only those supporting @use_experts_implementation)
    unfused_modules = []
    for name, module in model.named_modules():
        if _unfuse_experts_weights_inplace(module):
            unfused_modules.append(name)
            logger.debug(f"Unfused expert weights in: {name}")

    # Only set config if we actually unfused something
    # Models that don't support the decorator (like Llama4) won't have anything unfused
    # and should use full module replacement instead
    if unfused_modules:
        logger.info(f"Unfused {len(unfused_modules)} MOE experts modules for quantization")
        clear_memory()

        # Set config for linear_loop forward
        if hasattr(model, "config"):
            saved_impl = getattr(model.config, "experts_implementation", None)
            impl_to_set = saved_impl if saved_impl else implementation
            model.config._experts_implementation = impl_to_set
            logger.info(f"Set model.config._experts_implementation = '{impl_to_set}'")

    return unfused_modules


def is_linear_loop_available() -> bool:
    """Check if linear_loop experts implementation is available."""
    return HAS_EXPERTS_INTERFACE
