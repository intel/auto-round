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

import torch
from torch import nn

from auto_round.utils import clear_memory, logger

try:
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

    HAS_EXPERTS_INTERFACE = True
except ImportError:
    HAS_EXPERTS_INTERFACE = False
    ALL_EXPERTS_FUNCTIONS = None

# Expert implementation name - change this if transformers want to use a different name
LINEAR_LOOP_IMPL = "linear_loop"

# Known expert projection patterns for reference
# These are used as hints when auto-detection needs to infer projection properties
# Format: proj_name -> {"is_input_proj": bool, "output_multiplier": int}
#   is_input_proj: True if takes hidden_dim as input, False if takes intermediate_dim
#   output_multiplier: output dimension multiplier (e.g., 2 for fused gate+up projection)
KNOWN_PROJECTION_PATTERNS = {
    # Transformers 5.0+ standard (Qwen3-MoE, etc.)
    "gate_up_proj": {"is_input_proj": True, "output_multiplier": 2},  # hidden -> 2*intermediate
    "down_proj": {"is_input_proj": False, "output_multiplier": 1},    # intermediate -> hidden
    # Mixtral-style
    "w1": {"is_input_proj": True, "output_multiplier": 1},   # gate: hidden -> intermediate
    "w2": {"is_input_proj": False, "output_multiplier": 1},  # down: intermediate -> hidden
    "w3": {"is_input_proj": True, "output_multiplier": 1},   # up: hidden -> intermediate
    # DBRX-style
    "v1": {"is_input_proj": True, "output_multiplier": 1},
    "w1_proj": {"is_input_proj": True, "output_multiplier": 1},
    "w2_proj": {"is_input_proj": False, "output_multiplier": 1},
}


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
    logger.debug(f"Using {LINEAR_LOOP_IMPL} experts forward for {self.__class__.__name__}")

    # Handle [batch_size, seq_len, hidden_dim] input format
    if hidden_states.dim() == 3:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)  # [bs * seq_len, hidden_dim]
        top_k_index = top_k_index.view(-1, top_k_index.size(-1))  # [bs * seq_len, top_k]
        top_k_weights = top_k_weights.view(-1, top_k_weights.size(-1))  # [bs * seq_len, top_k]
    else:
        batch_size, seq_len = None, None
        hidden_dim = hidden_states.size(-1)

    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    num_experts = self.num_experts

    # Reshape for easier indexing
    # S is the number of selected token-expert pairs (S = num_tokens * num_top_k)
    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1)  # (S,)
    sample_weights = top_k_weights.reshape(-1).to(hidden_states.dtype)  # (S,)
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

    # Reshape back to original format if input was [batch_size, seq_len, hidden_dim]
    if batch_size is not None:
        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)

    return final_hidden_states


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

    if LINEAR_LOOP_IMPL not in ALL_EXPERTS_FUNCTIONS._global_mapping:
        ALL_EXPERTS_FUNCTIONS._global_mapping[LINEAR_LOOP_IMPL] = linear_loop_experts_forward
        logger.debug(f"Registered '{LINEAR_LOOP_IMPL}' experts implementation")

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


def _detect_expert_projections(module: nn.Module) -> dict[str, dict]:
    """Detect which expert projections exist in the module.

    This function scans the module for any 3D nn.Parameter attributes.
    It first checks known projection names, then discovers any unknown 3D parameters.

    Returns:
        Dict mapping projection names to their config, only for projections that exist
        as 3D nn.Parameter in the module.
    """
    detected = {}

    # First, check known projection patterns
    for proj_name, config in KNOWN_PROJECTION_PATTERNS.items():
        param = getattr(module, proj_name, None)
        if param is not None and isinstance(param, nn.Parameter) and param.dim() == 3:
            detected[proj_name] = config

    # If no known patterns found, scan for any 3D Parameter (future-proofing)
    if not detected:
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
            param = getattr(module, attr_name, None)
            if param is not None and isinstance(param, nn.Parameter) and param.dim() == 3:
                # Use default config for unknown projections
                logger.debug(f"Discovered unknown 3D projection: {attr_name}")
                detected[attr_name] = {"is_input_proj": True, "output_multiplier": 1}

    return detected


def _infer_dimensions(param: nn.Parameter, config: dict, is_transposed: bool) -> tuple[int, int]:
    """Infer input and output dimensions for a projection.

    Args:
        param: The 3D parameter (num_experts, dim1, dim2)
        config: Projection config with is_input_proj and output_multiplier
        is_transposed: Whether weights are stored transposed

    Returns:
        (in_features, out_features) for the Linear layer
    """
    dim1, dim2 = param.shape[1], param.shape[2]
    multiplier = config.get("output_multiplier", 1)

    if is_transposed:
        # transposed: (num_experts, in_features, out_features)
        in_features, out_features = dim1, dim2
    else:
        # not transposed: (num_experts, out_features, in_features)
        out_features, in_features = dim1, dim2

    # Adjust for multiplier (e.g., gate_up has 2x intermediate)
    if multiplier > 1:
        out_features = out_features // multiplier * multiplier  # ensure divisible

    return in_features, out_features


def _unfuse_single_projection(
    module: nn.Module,
    proj_name: str,
    num_experts: int,
    is_transposed: bool,
    dtype: torch.dtype,
    target_device: torch.device,
) -> nn.ModuleList | None:
    """Unfuse a single projection from 3D Parameter to ModuleList of Linear layers.

    Args:
        module: The experts module
        proj_name: Name of the projection attribute
        num_experts: Number of experts
        is_transposed: Whether weights are stored transposed
        dtype: Data type for the Linear layers
        target_device: Device for the Linear layers

    Returns:
        ModuleList of Linear layers, or None if projection doesn't exist
    """
    param = getattr(module, proj_name, None)
    if param is None or not isinstance(param, nn.Parameter) or param.dim() != 3:
        return None

    # Get projection config
    config = KNOWN_PROJECTION_PATTERNS.get(proj_name, {"is_input_proj": True, "output_multiplier": 1})

    # Infer dimensions
    in_features, out_features = _infer_dimensions(param, config, is_transposed)

    # Check for bias
    bias_name = f"{proj_name}_bias"
    bias_param = getattr(module, bias_name, None)
    has_bias = bias_param is not None

    # Create ModuleList
    linears = nn.ModuleList()
    source_device = param.device

    for i in range(num_experts):
        linear = nn.Linear(in_features, out_features, bias=has_bias, dtype=dtype, device=target_device)

        # Copy weights if not on meta device
        if source_device.type != "meta":
            if is_transposed:
                linear.weight.data.copy_(param[i].t())
            else:
                linear.weight.data.copy_(param[i])

            if has_bias:
                linear.bias.data.copy_(bias_param[i])

        linears.append(linear)

    # Release original parameter memory using to_empty
    if source_device.type != "meta":
        try:
            param.data = param.data.to_empty(device="meta")
            logger.debug(f"Released memory for {proj_name} using to_empty(device='meta')")
        except Exception:
            # Fallback: just delete
            pass

    return linears


def _unfuse_experts_weights_inplace(
    module: nn.Module,
    check_decorator: bool = True,
    projection_names: list[str] | None = None,
) -> bool:
    """Convert fused 3D expert weights to nn.ModuleList of nn.Linear layers.

    This function modifies the module in-place, replacing fused 3D Parameters
    with nn.ModuleList[nn.Linear] for each detected projection.

    Args:
        module: The experts module to unfuse
        check_decorator: If True, only unfuse if the module supports
            @use_experts_implementation decorator. Default is True.
        projection_names: Optional list of projection names to unfuse.
            If None, auto-detects from KNOWN_PROJECTION_PATTERNS.

    Returns:
        True if unfusing was successful, False if module doesn't match pattern
    """
    # Detect available projections
    if projection_names:
        detected_projections = {
            name: config
            for name, config in KNOWN_PROJECTION_PATTERNS.items()
            if name in projection_names
        }
    else:
        detected_projections = _detect_expert_projections(module)

    if not detected_projections:
        return False

    # Only unfuse if the module supports the decorator (unless check_decorator is False)
    if check_decorator and not _experts_supports_decorator(module):
        logger.debug(f"Skipping unfuse for {module.__class__.__name__}: does not support @use_experts_implementation")
        return False

    # Get first projection to determine num_experts and layout
    first_proj_name = next(iter(detected_projections))
    first_param = getattr(module, first_proj_name)
    num_experts = first_param.shape[0]

    # Detect if transposed
    is_transposed = getattr(module, "is_transposed", None)
    if is_transposed is None:
        # Infer from shape: typically hidden_dim < intermediate_dim
        dim1, dim2 = first_param.shape[1], first_param.shape[2]
        is_transposed = dim1 < dim2

    dtype = first_param.dtype
    target_device = first_param.device if first_param.device.type != "meta" else "cpu"

    # Unfuse each projection
    unfused_count = 0
    for proj_name in detected_projections:
        linears = _unfuse_single_projection(
            module, proj_name, num_experts, is_transposed, dtype, target_device
        )
        if linears is not None:
            # Delete original parameter and set new ModuleList
            delattr(module, proj_name)
            setattr(module, proj_name, linears)
            unfused_count += 1
            logger.debug(f"Unfused {proj_name}: {num_experts} experts")


    # Ensure num_experts is set
    if not hasattr(module, "num_experts"):
        module.num_experts = num_experts

    return unfused_count > 0


def prepare_model_for_moe_quantization(model: nn.Module, implementation: str = LINEAR_LOOP_IMPL) -> list[str]:
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
            logger.debug(f"Set model.config._experts_implementation = '{impl_to_set}'")

    return unfused_modules


def is_linear_loop_available() -> bool:
    """Check if linear_loop experts implementation is available."""
    return HAS_EXPERTS_INTERFACE
