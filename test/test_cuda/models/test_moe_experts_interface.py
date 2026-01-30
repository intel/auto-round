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
Test script for moe_experts_interface.py - linear_loop experts implementation.

This verifies that:
1. linear_loop is registered with transformers 'ALL_EXPERTS_FUNCTIONS'
2. Fused expert weights are correctly unfused to nn.Linear layers
3. The forward pass produces correct results
"""

import torch
from torch import nn


def test_linear_loop_registration():
    """Test that linear_loop is registered with transformers."""
    from auto_round.modeling.fused_moe.moe_experts_interface import (
        register_linear_loop_experts,
        is_linear_loop_available,
        register_linear_loop_experts,
    )

    if not is_linear_loop_available():
        print("SKIP: transformers MOE integration not available")
        return
    success = register_linear_loop_experts()
    assert success, "Failed to register linear_loop"

    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

    assert "linear_loop" in ALL_EXPERTS_FUNCTIONS._global_mapping
    print("✓ linear_loop registered with transformers")


def test_unfuse_experts_weights():
    """Test unfusing fused expert weights to nn.Linear layers."""
    from auto_round.modeling.fused_moe.moe_experts_interface import _unfuse_experts_weights_inplace

    # Create a mock fused experts module (Mixtral style - not transposed)
    num_experts = 4
    hidden_dim = 64
    intermediate_dim = 128

    class MockFusedExperts(nn.Module):
        def __init__(self):
            super().__init__()
            # Not transposed: (num_experts, 2*intermediate, hidden)
            self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * intermediate_dim, hidden_dim))
            # (num_experts, hidden, intermediate)
            self.down_proj = nn.Parameter(torch.randn(num_experts, hidden_dim, intermediate_dim))
            self.act_fn = nn.SiLU()
            self.num_experts = num_experts
            self.has_bias = False
            self.is_transposed = False

    module = MockFusedExperts()

    # Store original weights for comparison
    original_gate_up = module.gate_up_proj.data.clone()
    original_down = module.down_proj.data.clone()

    # Unfuse (check_decorator=False since mock module doesn't have the decorator)
    success = _unfuse_experts_weights_inplace(module, check_decorator=False)
    assert success, "Failed to unfuse weights"

    # Verify structure
    assert isinstance(module.gate_up_proj, nn.ModuleList)
    assert isinstance(module.down_proj, nn.ModuleList)
    assert len(module.gate_up_proj) == num_experts
    assert len(module.down_proj) == num_experts

    # Verify weights are preserved
    for i in range(num_experts):
        # gate_up: original (2*intermediate, hidden), linear.weight should be same
        assert torch.allclose(
            module.gate_up_proj[i].weight.data, original_gate_up[i], atol=1e-6
        ), f"gate_up weight mismatch for expert {i}"

        # down: original (hidden, intermediate), linear.weight should be same
        assert torch.allclose(
            module.down_proj[i].weight.data, original_down[i], atol=1e-6
        ), f"down weight mismatch for expert {i}"

    print("✓ Unfused weights correctly (Mixtral style)")


def test_unfuse_experts_weights_transposed():
    """Test unfusing transposed expert weights (Llama4/GptOss style)."""
    from auto_round.modeling.fused_moe.moe_experts_interface import _unfuse_experts_weights_inplace

    num_experts = 4
    hidden_dim = 64
    intermediate_dim = 128

    class MockFusedExpertsTransposed(nn.Module):
        def __init__(self):
            super().__init__()
            # Transposed: (num_experts, hidden, 2*intermediate)
            self.gate_up_proj = nn.Parameter(torch.randn(num_experts, hidden_dim, 2 * intermediate_dim))
            # Transposed: (num_experts, intermediate, hidden)
            self.down_proj = nn.Parameter(torch.randn(num_experts, intermediate_dim, hidden_dim))
            self.act_fn = nn.SiLU()
            self.num_experts = num_experts
            self.is_transposed = True
            self.has_bias = False

    module = MockFusedExpertsTransposed()

    # Store original weights for comparison
    original_gate_up = module.gate_up_proj.data.clone()
    original_down = module.down_proj.data.clone()

    # Unfuse (check_decorator=False since mock module doesn't have the decorator)
    success = _unfuse_experts_weights_inplace(module, check_decorator=False)
    assert success, "Failed to unfuse transposed weights"

    # Verify structure
    assert isinstance(module.gate_up_proj, nn.ModuleList)
    assert isinstance(module.down_proj, nn.ModuleList)

    # Verify weights are correctly transposed
    for i in range(num_experts):
        # gate_up: original (hidden, 2*intermediate), should be transposed to (2*intermediate, hidden)
        assert torch.allclose(
            module.gate_up_proj[i].weight.data, original_gate_up[i].t(), atol=1e-6
        ), f"gate_up weight mismatch for expert {i}"

        # down: original (intermediate, hidden), should be transposed to (hidden, intermediate)
        assert torch.allclose(
            module.down_proj[i].weight.data, original_down[i].t(), atol=1e-6
        ), f"down weight mismatch for expert {i}"

    print("✓ Unfused weights correctly (transposed style)")


def test_linear_loop_forward():
    """Test that linear_loop forward produces correct results."""
    from auto_round.modeling.fused_moe.moe_experts_interface import (
        linear_loop_experts_forward,
        _unfuse_experts_weights_inplace,
        linear_loop_experts_forward,
    )

    num_experts = 4
    hidden_dim = 64
    intermediate_dim = 128
    num_tokens = 10
    top_k = 2

    # Create module with unfused weights
    class MockExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * intermediate_dim, hidden_dim))
            self.down_proj = nn.Parameter(torch.randn(num_experts, hidden_dim, intermediate_dim))
            self.act_fn = nn.SiLU()
            self.num_experts = num_experts
            self.has_bias = False
            self.is_transposed = False

    module = MockExperts()

    # Unfuse weights (check_decorator=False since mock module doesn't have the decorator)
    _unfuse_experts_weights_inplace(module, check_decorator=False)

    # Create inputs
    hidden_states = torch.randn(num_tokens, hidden_dim)
    top_k_index = torch.randint(0, num_experts, (num_tokens, top_k))
    top_k_weights = torch.softmax(torch.randn(num_tokens, top_k), dim=-1)

    # Run forward
    output = linear_loop_experts_forward(module, hidden_states, top_k_index, top_k_weights)

    # Verify output shape
    assert output.shape == hidden_states.shape, f"Output shape mismatch: {output.shape} vs {hidden_states.shape}"

    # Verify output is not all zeros (sanity check)
    assert not torch.allclose(output, torch.zeros_like(output)), "Output is all zeros"

    print("✓ linear_loop forward works correctly")


def test_prepare_model_for_moe_quantization():
    """Test the full prepare_model_for_moe_quantization flow."""
    from auto_round.modeling.fused_moe.moe_experts_interface import (
        prepare_model_for_moe_quantization,
        is_linear_loop_available,
        prepare_model_for_moe_quantization,
    )

    if not is_linear_loop_available():
        print("SKIP: transformers MOE integration not available")
        return

    num_experts = 4
    hidden_dim = 64
    intermediate_dim = 128

    # Create a mock model with fused experts
    class MockConfig:
        def __init__(self):
            self._experts_implementation = "eager"

    class MockExpertsModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * intermediate_dim, hidden_dim))
            self.down_proj = nn.Parameter(torch.randn(num_experts, hidden_dim, intermediate_dim))
            self.act_fn = nn.SiLU()
            self.num_experts = num_experts
            self.has_bias = False
            self.is_transposed = False
        
        def forward(self, x):
            pass

    # Add __wrapped__ to simulate @use_experts_implementation decorator
    MockExpertsModule.forward.__wrapped__ = True

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = MockConfig()
            self.layer = nn.ModuleDict({"experts": MockExpertsModule()})

    model = MockModel()

    # Prepare for quantization
    unfused = prepare_model_for_moe_quantization(model)

    # Verify
    assert model.config._experts_implementation == "linear_loop"
    assert len(unfused) == 1
    assert isinstance(model.layer["experts"].gate_up_proj, nn.ModuleList)

    print("✓ prepare_model_for_moe_quantization works correctly")


if __name__ == "__main__":
    print("Testing moe_experts_interface.py...\n")

    test_unfuse_experts_weights()
    test_unfuse_experts_weights_transposed()
    test_linear_loop_forward()
    test_linear_loop_registration()
    test_prepare_model_for_moe_quantization()

    print("\n✓ All tests passed!")
