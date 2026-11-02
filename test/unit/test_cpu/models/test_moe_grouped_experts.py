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

"""Grouped-GEMM MoE experts forward must match the per-expert loop backend."""

from unittest.mock import patch

import pytest
import torch
from torch import nn

from auto_round.modeling.fused_moe.grouped_experts import grouped_linear_experts_forward
from auto_round.modeling.fused_moe.moe_experts_interface import (
    GROUPED_LINEAR_IMPL,
    linear_loop_experts_forward,
    resolve_experts_implementation,
)

NUM_EXPERTS = 6
HIDDEN = 16
INTERMEDIATE = 24
TOP_K = 2
NUM_TOKENS = 9


class _Container(nn.Module):
    pass


class _Experts(nn.Module):
    """Mimics the unfused layout produced by ``_unfuse_experts_weights_inplace``."""

    def __init__(self, num_experts=NUM_EXPERTS, bias=False):
        super().__init__()
        self.num_experts = num_experts
        self.act_fn = nn.SiLU()
        for i in range(num_experts):
            container = _Container()
            container.gate_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=bias)
            container.up_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=bias)
            container.down_proj = nn.Linear(INTERMEDIATE, HIDDEN, bias=bias)
            self.add_module(str(i), container)


def _make_routing(num_tokens=NUM_TOKENS, num_experts=NUM_EXPERTS, top_k=TOP_K):
    scores = torch.rand(num_tokens, num_experts)
    weights, index = torch.topk(scores, top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return index, weights


def _mark_quantizable(layer, act_bits=16, act_dynamic=True):
    layer.bits = 4
    layer.group_size = 8
    layer.sym = True
    layer.data_type = "int"
    layer.act_bits = act_bits
    layer.act_group_size = -1
    layer.act_sym = True
    layer.act_data_type = "int"
    layer.act_dynamic = act_dynamic
    layer.scale_dtype = torch.float32
    return layer


def _wrap_experts(experts, wrapper_cls=None, **mark_kwargs):
    from auto_round.wrapper import WrapperLinear

    wrapper_cls = wrapper_cls or WrapperLinear
    for i in range(experts.num_experts):
        container = getattr(experts, str(i))
        for name in ("gate_proj", "up_proj", "down_proj"):
            layer = _mark_quantizable(getattr(container, name), **mark_kwargs)
            setattr(
                container,
                name,
                wrapper_cls(
                    layer,
                    enable_minmax_tuning=True,
                    enable_norm_bias_tuning=False,
                    enable_torch_compile=False,
                    device="cpu",
                ),
            )
    return experts


@pytest.mark.parametrize("bias", [False, True])
def test_grouped_matches_loop_fp(bias):
    torch.manual_seed(0)
    experts = _Experts(bias=bias)
    hidden_states = torch.randn(NUM_TOKENS, HIDDEN)
    index, weights = _make_routing()

    expected = linear_loop_experts_forward(experts, hidden_states, index, weights)
    got = grouped_linear_experts_forward(experts, hidden_states, index, weights)

    assert got.shape == expected.shape
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


def test_grouped_matches_loop_3d_input():
    torch.manual_seed(1)
    experts = _Experts()
    hidden_states = torch.randn(2, 5, HIDDEN)
    index, weights = _make_routing(num_tokens=10)
    index = index.view(2, 5, TOP_K)
    weights = weights.view(2, 5, TOP_K)

    expected = linear_loop_experts_forward(experts, hidden_states, index, weights)
    got = grouped_linear_experts_forward(experts, hidden_states, index, weights)

    assert got.shape == hidden_states.shape
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


def test_grouped_matches_loop_with_inactive_experts():
    """Experts that receive no token must be skipped, not silently indexed incorrectly."""
    torch.manual_seed(2)
    experts = _Experts()
    hidden_states = torch.randn(NUM_TOKENS, HIDDEN)
    # Route everything to experts 1 and 4 only.
    index = torch.tensor([[1, 4]] * NUM_TOKENS)
    weights = torch.full((NUM_TOKENS, TOP_K), 0.5)

    expected = linear_loop_experts_forward(experts, hidden_states, index, weights)
    got = grouped_linear_experts_forward(experts, hidden_states, index, weights)

    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


def test_grouped_matches_loop_when_wrapped_for_tuning():
    torch.manual_seed(3)
    experts = _wrap_experts(_Experts())
    hidden_states = torch.randn(NUM_TOKENS, HIDDEN)
    index, weights = _make_routing()

    expected = linear_loop_experts_forward(experts, hidden_states, index, weights)
    got = grouped_linear_experts_forward(experts, hidden_states, index, weights)

    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


def test_grouped_propagates_gradients_to_tuning_params():
    torch.manual_seed(4)
    experts = _wrap_experts(_Experts())
    hidden_states = torch.randn(NUM_TOKENS, HIDDEN)
    index, weights = _make_routing()

    out = grouped_linear_experts_forward(experts, hidden_states, index, weights)
    out.abs().mean().backward()

    touched = set(index.reshape(-1).tolist())
    grads = [
        getattr(experts, str(i)).down_proj.value.grad
        for i in touched
        if getattr(experts, str(i)).down_proj.value.grad is not None
    ]
    assert grads, "no gradient reached the rounding parameters of the routed experts"
    assert any(grad.abs().sum() > 0 for grad in grads)


def test_grouped_matches_loop_with_activation_quant():
    torch.manual_seed(5)
    experts = _wrap_experts(_Experts(), act_bits=8)
    hidden_states = torch.randn(NUM_TOKENS, HIDDEN)
    index, weights = _make_routing()

    expected = linear_loop_experts_forward(experts, hidden_states, index, weights)
    got = grouped_linear_experts_forward(experts, hidden_states, index, weights)

    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


def test_grouped_falls_back_when_layer_has_forward_hook():
    """Online-rotation style hooks must run per layer, so the loop backend takes over."""
    torch.manual_seed(6)
    experts = _wrap_experts(_Experts())
    scale = torch.full((HIDDEN,), 0.5)
    getattr(experts, "0").up_proj.orig_layer.register_forward_pre_hook(lambda module, args: (args[0] * scale,))

    hidden_states = torch.randn(NUM_TOKENS, HIDDEN)
    index, weights = _make_routing()

    expected = linear_loop_experts_forward(experts, hidden_states, index, weights)
    got = grouped_linear_experts_forward(experts, hidden_states, index, weights)

    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


def _observe_grouped_forward(experts, hidden_states, index, weights):
    """Observe actual slot execution, not a removed planning-only API.

    An empty call list means eligibility checks fell back before grouped GEMM;
    each recorded boolean says whether that slot used batched weight QDQ.
    """
    from auto_round.modeling.fused_moe import grouped_experts as ge

    with patch.object(ge, "_slot_weights", wraps=ge._slot_weights) as slots:
        output = grouped_linear_experts_forward(experts, hidden_states, index, weights)
    return output, [call.args[1] for call in slots.call_args_list]


def test_grouped_disabled_for_mixed_bits_across_experts():
    """Mixed-bit MoE must not be batched: the loop backend keeps ownership."""
    torch.manual_seed(7)
    experts = _wrap_experts(_Experts())
    # Re-wrap expert 2 with a different bit-width.
    from auto_round.wrapper import WrapperLinear

    container = getattr(experts, "2")
    for name in ("gate_proj", "up_proj", "down_proj"):
        layer = getattr(container, name).orig_layer
        layer.bits = 8
        setattr(
            container,
            name,
            WrapperLinear(layer, enable_minmax_tuning=True, enable_torch_compile=False, device="cpu"),
        )

    hidden_states = torch.randn(NUM_TOKENS, HIDDEN)
    index = torch.tensor([[1, 2]] * NUM_TOKENS)
    weights = torch.full((NUM_TOKENS, TOP_K), 0.5)

    expected = linear_loop_experts_forward(experts, hidden_states, index, weights)
    got, batched = _observe_grouped_forward(experts, hidden_states, index, weights)
    assert not batched, "grouped backend must refuse mixed-bit experts"
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


def test_grouped_disabled_when_some_experts_stay_unquantized():
    """A 16-bit (unwrapped) expert next to quantized ones also disables the grouped path."""
    torch.manual_seed(8)
    experts = _wrap_experts(_Experts())
    container = getattr(experts, "3")
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(container, name, getattr(container, name).orig_layer)

    hidden_states = torch.randn(NUM_TOKENS, HIDDEN)
    index = torch.tensor([[0, 3]] * NUM_TOKENS)
    weights = torch.full((NUM_TOKENS, TOP_K), 0.5)

    expected = linear_loop_experts_forward(experts, hidden_states, index, weights)
    got, batched = _observe_grouped_forward(experts, hidden_states, index, weights)
    assert not batched
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


def test_grouped_still_used_when_mixed_expert_is_not_routed():
    """Only the *active* experts are checked, so an unrouted mixed expert is harmless."""
    torch.manual_seed(9)
    experts = _wrap_experts(_Experts())
    container = getattr(experts, "5")
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(container, name, getattr(container, name).orig_layer)

    index = torch.tensor([[0, 1]] * NUM_TOKENS)
    hidden_states = torch.randn(NUM_TOKENS, HIDDEN)
    weights = torch.full((NUM_TOKENS, TOP_K), 0.5)
    expected = linear_loop_experts_forward(experts, hidden_states, index, weights)
    got, batched = _observe_grouped_forward(experts, hidden_states, index, weights)
    assert batched and all(batched)
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------------------
# Fused fake-quantization (one weight_quant_func call for all experts of a projection)
# --------------------------------------------------------------------------------------


def _slot_layers(experts, slot, expert_ids):
    return [getattr(getattr(experts, str(i)), slot) for i in expert_ids]


def test_batched_qdq_matches_per_expert_qdq():
    """Fusing the weights + tuning params must be bit-identical to per-expert qdq."""
    from auto_round.modeling.fused_moe.grouped_experts import _batched_qdq_weights

    torch.manual_seed(10)
    experts = _wrap_experts(_Experts())
    layers = _slot_layers(experts, "down_proj", range(NUM_EXPERTS))
    # Give every expert distinct tuning state so a misaligned concat would show up.
    for i, layer in enumerate(layers):
        with torch.no_grad():
            layer.value.copy_(torch.randn_like(layer.value) * 0.1)
            layer.min_scale.copy_(torch.rand_like(layer.min_scale))
            layer.max_scale.copy_(torch.rand_like(layer.max_scale))

    expected = [layer._qdq_weight(layer.value, layer.min_scale, layer.max_scale)[0] for layer in layers]
    stacked, got = _batched_qdq_weights(layers)

    assert stacked is not None and stacked.shape == (NUM_EXPERTS, HIDDEN, INTERMEDIATE)
    for reference, actual in zip(expected, got):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_batched_qdq_gradients_land_on_the_right_expert():
    """Each row-block's gradient must flow back to its own expert's parameters."""
    from auto_round.modeling.fused_moe.grouped_experts import _batched_qdq_weights

    torch.manual_seed(11)
    experts = _wrap_experts(_Experts())
    layers = _slot_layers(experts, "up_proj", range(NUM_EXPERTS))

    _, weights = _batched_qdq_weights(layers)
    # Only expert 2 contributes to the loss.
    weights[2].sum().backward()

    assert layers[2].value.grad is not None and layers[2].value.grad.abs().sum() > 0
    for i, layer in enumerate(layers):
        if i != 2:
            assert layer.value.grad is None or layer.value.grad.abs().sum() == 0


def test_grouped_forward_enables_batched_qdq():
    torch.manual_seed(12)
    experts = _wrap_experts(_Experts())
    hidden_states = torch.randn(NUM_TOKENS, HIDDEN)
    index, weights = _make_routing()
    _, batched = _observe_grouped_forward(experts, hidden_states, index, weights)
    assert batched == [True, True, True]


def _make_imatrix_experts(shared: bool, group_size=8, data_type="opt_rtn_mx_fp"):
    """Experts carrying an importance matrix, shared or per-expert.

    imatrix is only consumed by the ``opt_rtn_*`` quant funcs, which are pure RTN (they do
    not accept a per-element ``v``), so the wrappers are built the way ``iters=0`` builds
    them: no round tuning, no min/max tuning. Weights are bf16 like a real model --
    ``quant_mx_opt_rtn`` casts to fp32 and then works in place, so an fp32 weight would be
    mutated by the reference call and invalidate the comparison.
    """
    experts = _Experts().to(torch.bfloat16)
    shared_imatrix = torch.rand(HIDDEN) + 0.1
    for i in range(NUM_EXPERTS):
        for name in ("gate_proj", "up_proj", "down_proj"):
            layer = _mark_quantizable(getattr(getattr(experts, str(i)), name))
            layer.group_size = group_size
            layer.data_type = data_type
            layer.bits = 4
            in_features = layer.weight.shape[1]
            if shared and in_features == HIDDEN:
                layer.imatrix = shared_imatrix
            else:
                layer.imatrix = torch.rand(in_features) + 0.1
    return _wrap_experts_preserving_marks(experts, enable_minmax_tuning=False, enable_round_tuning=False)


def test_batched_qdq_supports_shared_imatrix():
    """When every expert points at the same imatrix it is passed through untouched."""
    from auto_round.modeling.fused_moe.grouped_experts import _batched_qdq_weights, _supports_batched_qdq

    torch.manual_seed(13)
    experts = _make_imatrix_experts(shared=True)
    layers = _slot_layers(experts, "up_proj", range(NUM_EXPERTS))
    assert _supports_batched_qdq(layers[0])

    with torch.no_grad():  # opt_rtn_* is the iters=0 path, which runs without autograd
        expected = [layer._qdq_weight(layer.value, layer.min_scale, layer.max_scale)[0] for layer in layers]
        _, got = _batched_qdq_weights(layers)
    for reference, actual in zip(expected, got):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_batched_qdq_supports_per_expert_imatrix():
    """Distinct per-expert importance must survive fusion, not collapse onto expert 0."""
    from auto_round.modeling.fused_moe.grouped_experts import _batched_qdq_weights, _supports_batched_qdq

    torch.manual_seed(14)
    experts = _make_imatrix_experts(shared=False)
    layers = _slot_layers(experts, "down_proj", range(NUM_EXPERTS))
    assert _supports_batched_qdq(layers[0])
    imatrices = [layer.orig_layer.imatrix for layer in layers]
    assert not torch.allclose(imatrices[0], imatrices[1]), "test needs genuinely different imatrices"

    with torch.no_grad():
        expected = [layer._qdq_weight(layer.value, layer.min_scale, layer.max_scale)[0] for layer in layers]
        _, got = _batched_qdq_weights(layers)
    for i, (reference, actual) in enumerate(zip(expected, got)):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0, msg=f"expert {i} mismatch")


def test_fused_imatrix_is_expert_major():
    """The fused importance tensor must map row r to expert ``r // out_features``.

    This is the layout the quant funcs reconstruct via
    ``reshape(1, -1).expand(1, -1).reshape(tensor.shape)``; getting the expert order or
    the repeat factor wrong would silently apply incorrect weights to every expert.
    """
    from auto_round.modeling.fused_moe.grouped_experts import _fused_imatrix

    torch.manual_seed(15)
    experts = _make_imatrix_experts(shared=False)
    layers = _slot_layers(experts, "down_proj", range(NUM_EXPERTS))
    out_features, in_features = layers[0].weight.shape

    fused = _fused_imatrix(layers, out_features, torch.device("cpu"))
    assert fused.shape == (NUM_EXPERTS * out_features, in_features)
    for expert_id, layer in enumerate(layers):
        block = fused[expert_id * out_features : (expert_id + 1) * out_features]
        assert torch.equal(block, layer.orig_layer.imatrix.expand(out_features, -1))


def test_fused_imatrix_passes_shared_tensor_through():
    """A shared imatrix needs no materialization: the quant func's own repeat is correct."""
    from auto_round.modeling.fused_moe.grouped_experts import _fused_imatrix

    torch.manual_seed(16)
    experts = _make_imatrix_experts(shared=True)
    layers = _slot_layers(experts, "up_proj", range(NUM_EXPERTS))
    fused = _fused_imatrix(layers, layers[0].weight.shape[0], torch.device("cpu"))
    assert fused is layers[0].orig_layer.imatrix


def test_batched_qdq_disabled_for_unaligned_imatrix():
    """in_features not a multiple of group_size: the flattened imatrix padding would shift."""
    from auto_round.modeling.fused_moe.grouped_experts import _supports_batched_qdq

    torch.manual_seed(17)
    # INTERMEDIATE = 24 is not a multiple of 16, so down_proj's input is unaligned.
    experts = _make_imatrix_experts(shared=False, group_size=16)
    assert not _supports_batched_qdq(_slot_layers(experts, "down_proj", [0])[0])


def _rebuild_with_group_size(group_size):
    experts = _Experts()
    for i in range(NUM_EXPERTS):
        for name in ("gate_proj", "up_proj", "down_proj"):
            _mark_quantizable(getattr(getattr(experts, str(i)), name)).group_size = group_size
    return _wrap_experts_preserving_marks(experts)


def _wrap_experts_preserving_marks(experts, **wrapper_kwargs):
    from auto_round.wrapper import WrapperLinear

    wrapper_kwargs.setdefault("enable_minmax_tuning", True)
    for i in range(experts.num_experts):
        container = getattr(experts, str(i))
        for name in ("gate_proj", "up_proj", "down_proj"):
            layer = getattr(container, name)
            setattr(
                container,
                name,
                WrapperLinear(layer, enable_torch_compile=False, device="cpu", **wrapper_kwargs),
            )
    return experts


def test_batched_qdq_supports_per_tensor_group_size():
    """group_size == 0 means one scale per expert; fusing as one row per expert matches."""
    from auto_round.modeling.fused_moe.grouped_experts import _batched_qdq_weights

    torch.manual_seed(16)
    experts = _rebuild_with_group_size(0)
    layers = _slot_layers(experts, "down_proj", range(NUM_EXPERTS))
    for layer in layers:
        with torch.no_grad():
            layer.value.copy_(torch.randn_like(layer.value) * 0.1)
            layer.max_scale.copy_(torch.rand_like(layer.max_scale))

    hidden_states = torch.randn(NUM_TOKENS, HIDDEN)
    index, weights = _make_routing()
    _, batched = _observe_grouped_forward(experts, hidden_states, index, weights)
    assert batched == [True, True, True]

    expected = [layer._qdq_weight(layer.value, layer.min_scale, layer.max_scale)[0] for layer in layers]
    _, got = _batched_qdq_weights(layers)
    for reference, actual in zip(expected, got):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_grouped_matches_loop_with_per_tensor_group_size():
    torch.manual_seed(17)
    experts = _rebuild_with_group_size(0)
    hidden_states = torch.randn(NUM_TOKENS, HIDDEN)
    index, weights = _make_routing()

    expected = linear_loop_experts_forward(experts, hidden_states, index, weights)
    got = grouped_linear_experts_forward(experts, hidden_states, index, weights)
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


def test_batched_qdq_supports_aligned_2d_blocks():
    """2-D block grouping fuses when each expert holds a whole number of block rows."""
    from auto_round.modeling.fused_moe.grouped_experts import _batched_qdq_weights, _supports_batched_qdq

    torch.manual_seed(18)
    experts = _Experts()
    for i in range(NUM_EXPERTS):
        for name in ("gate_proj", "up_proj", "down_proj"):
            layer = _mark_quantizable(getattr(getattr(experts, str(i)), name))
            layer.group_size = (8, 8)  # HIDDEN=16 and INTERMEDIATE=24 are both multiples of 8
            layer.data_type = "block_fp8_sym"
            layer.bits = 8
    experts = _wrap_experts_preserving_marks(experts)

    layers = _slot_layers(experts, "up_proj", range(NUM_EXPERTS))
    assert _supports_batched_qdq(layers[0])
    expected = [layer._qdq_weight(layer.value, layer.min_scale, layer.max_scale)[0] for layer in layers]
    _, got = _batched_qdq_weights(layers)
    for reference, actual in zip(expected, got):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_batched_qdq_disabled_for_unaligned_2d_blocks():
    """out_features not a multiple of the block height: per-expert padding would differ."""
    from auto_round.modeling.fused_moe.grouped_experts import _supports_batched_qdq

    torch.manual_seed(19)
    experts = _Experts()
    for i in range(NUM_EXPERTS):
        for name in ("gate_proj", "up_proj", "down_proj"):
            layer = _mark_quantizable(getattr(getattr(experts, str(i)), name))
            layer.group_size = (16, 8)  # INTERMEDIATE=24 is not a multiple of 16
            layer.data_type = "block_fp8_sym"
            layer.bits = 8
    experts = _wrap_experts_preserving_marks(experts)

    assert not _supports_batched_qdq(_slot_layers(experts, "up_proj", [0])[0])


def test_batched_qdq_supports_nvfp_global_scale():
    """Each expert keeps its own NVFP scalar via a per-row global_scale column."""
    from auto_round.modeling.fused_moe.grouped_experts import _batched_qdq_weights, _supports_batched_qdq

    torch.manual_seed(20)
    experts = _Experts()
    for i in range(NUM_EXPERTS):
        for name in ("gate_proj", "up_proj", "down_proj"):
            layer = _mark_quantizable(getattr(getattr(experts, str(i)), name))
            layer.group_size = 16
            layer.data_type = "nv_fp4"
            layer.bits = 4
    experts = _wrap_experts_preserving_marks(experts)

    layers = _slot_layers(experts, "up_proj", range(NUM_EXPERTS))
    assert layers[0].weight_global_scale is not None
    assert _supports_batched_qdq(layers[0])
    expected = [layer._qdq_weight(layer.value, layer.min_scale, layer.max_scale)[0] for layer in layers]
    _, got = _batched_qdq_weights(layers)
    for reference, actual in zip(expected, got):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_grouped_matches_loop_with_nvfp():
    torch.manual_seed(21)
    experts = _Experts()
    for i in range(NUM_EXPERTS):
        for name in ("gate_proj", "up_proj", "down_proj"):
            layer = _mark_quantizable(getattr(getattr(experts, str(i)), name))
            layer.group_size = 16
            layer.data_type = "nv_fp4"
    experts = _wrap_experts_preserving_marks(experts)

    hidden_states = torch.randn(NUM_TOKENS, HIDDEN)
    index, weights = _make_routing()
    expected = linear_loop_experts_forward(experts, hidden_states, index, weights)
    got = grouped_linear_experts_forward(experts, hidden_states, index, weights)
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


def test_batched_qdq_disabled_for_custom_qdq_wrapper():
    """A wrapper overriding _qdq_weight (GGUF double-quant, AutoScheme) keeps per-expert qdq."""
    from auto_round.modeling.fused_moe.grouped_experts import _supports_batched_qdq
    from auto_round.wrapper import WrapperLinear

    class _CustomQdqWrapper(WrapperLinear):
        def _qdq_weight(self, value, min_scale, max_scale):
            return super()._qdq_weight(value, min_scale, max_scale)

    torch.manual_seed(14)
    experts = _wrap_experts(_Experts(), wrapper_cls=_CustomQdqWrapper)
    layer = getattr(experts, "0").up_proj
    assert not _supports_batched_qdq(layer)

    hidden_states = torch.randn(NUM_TOKENS, HIDDEN)
    index, weights = _make_routing()
    expected = linear_loop_experts_forward(experts, hidden_states, index, weights)
    got, batched = _observe_grouped_forward(experts, hidden_states, index, weights)
    assert batched == [False, False, False], "grouped GEMM must still apply, only fused qdq is skipped"
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


def test_grouped_matches_loop_with_signround_v2_wrapper():
    """SignRoundV2's optimized wrapper adds init_scale; it must be concatenated too."""
    from auto_round.algorithms.quantization.sign_roundv2.quantizer import SignRoundOptimizedWrapperLinear

    torch.manual_seed(15)
    experts = _wrap_experts(_Experts(), wrapper_cls=SignRoundOptimizedWrapperLinear)
    hidden_states = torch.randn(NUM_TOKENS, HIDDEN)
    index, weights = _make_routing()

    expected = linear_loop_experts_forward(experts, hidden_states, index, weights)
    got = grouped_linear_experts_forward(experts, hidden_states, index, weights)

    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------------------
# Grouped GEMM backend
# --------------------------------------------------------------------------------------


def test_native_grouped_mm_is_used_when_the_policy_allows_it(monkeypatch):
    """The native kernel must actually be reachable, not silently skipped by the gate.

    Native grouped_mm is preferred unless linear_grouped_sliced is requested;
    actual hardware eligibility is still checked independently.
    """
    from auto_round.modeling.fused_moe import grouped_experts as ge

    monkeypatch.setenv("AR_MOE_EXPERTS_IMPL", "linear_grouped")
    monkeypatch.setattr(ge, "_NATIVE_GROUPED_MM_DISABLED", False)

    torch.manual_seed(30)
    experts = _Experts()
    hidden_states = torch.randn(NUM_TOKENS, HIDDEN)
    index, weights = _make_routing()

    probe_weight = torch.empty(2, HIDDEN, INTERMEDIATE)
    probe_offsets = torch.tensor([1, 2], dtype=torch.int32)
    if not ge._native_grouped_mm_usable(hidden_states, probe_weight, probe_offsets):
        pytest.skip("torch reports grouped_mm as unusable on this build/device")

    calls = []
    original = ge._native_grouped_mm

    def _spy(x, weight, offsets):
        calls.append(tuple(weight.shape))
        return original(x, weight, offsets)

    monkeypatch.setattr(ge, "_native_grouped_mm", _spy)
    got = grouped_linear_experts_forward(experts, hidden_states, index, weights)

    expected = linear_loop_experts_forward(experts, hidden_states, index, weights)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
    assert calls, "grouped_mm was reported usable but never called"
    # (E, in, out): grouped_mm wants the transposed layout.
    assert calls[0][1] == HIDDEN and calls[0][2] == INTERMEDIATE


@pytest.mark.parametrize(
    "setting,preferred", [(None, True), ("linear_grouped", True), ("linear_grouped_sliced", False)]
)
def test_native_grouped_mm_policy(monkeypatch, setting, preferred):
    """Policy preference is distinct from hardware/kernel availability."""
    from auto_round.modeling.fused_moe import grouped_experts as ge

    if setting is None:
        monkeypatch.delenv("AR_MOE_EXPERTS_IMPL", raising=False)
    else:
        monkeypatch.setenv("AR_MOE_EXPERTS_IMPL", setting)
    assert ge._native_grouped_mm_preferred(torch.device("cpu")) is preferred
    assert ge._native_grouped_mm_preferred(torch.device("cuda")) is preferred
    if not preferred:
        assert not ge._native_grouped_mm_available()


def test_sliced_fallback_matches_native_kernel(monkeypatch):
    """The fallback loop and the native kernel must agree, gradients included."""
    from auto_round.modeling.fused_moe import grouped_experts as ge

    monkeypatch.setenv("AR_MOE_EXPERTS_IMPL", "linear_grouped")
    monkeypatch.setattr(ge, "_NATIVE_GROUPED_MM_DISABLED", False)

    torch.manual_seed(31)
    experts = _wrap_experts(_Experts())
    hidden_states = torch.randn(NUM_TOKENS, HIDDEN)
    index, weights = _make_routing()

    probe_weight = torch.empty(2, HIDDEN, INTERMEDIATE)
    probe_offsets = torch.tensor([1, 2], dtype=torch.int32)
    if not ge._native_grouped_mm_usable(hidden_states, probe_weight, probe_offsets):
        pytest.skip("torch reports grouped_mm as unusable on this build/device")

    def run():
        for param in experts.parameters():
            param.grad = None
        out = grouped_linear_experts_forward(experts, hidden_states, index, weights)
        out.square().mean().backward()
        grads = [getattr(experts, str(i)).down_proj.value.grad for i in range(NUM_EXPERTS)]
        return out.detach(), [None if g is None else g.clone() for g in grads]

    native_out, native_grads = run()
    monkeypatch.setattr(ge, "_NATIVE_GROUPED_MM_DISABLED", True)
    sliced_out, sliced_grads = run()

    torch.testing.assert_close(sliced_out, native_out, rtol=1e-5, atol=1e-5)
    for native, sliced in zip(native_grads, sliced_grads):
        assert (native is None) == (sliced is None)
        if native is not None:
            torch.testing.assert_close(sliced, native, rtol=1e-5, atol=1e-5)


def test_qdq_chunking_is_bit_exact(monkeypatch):
    """Chunking the fused fake-quant only bounds the temporary, never the result."""
    torch.manual_seed(32)
    experts = _wrap_experts(_Experts())
    hidden_states = torch.randn(NUM_TOKENS, HIDDEN)
    index, weights = _make_routing()

    monkeypatch.delenv("AR_MOE_CHUNK", raising=False)
    full = grouped_linear_experts_forward(experts, hidden_states, index, weights)

    for chunk in ("1", "2", "3", str(NUM_EXPERTS), str(NUM_EXPERTS * 2)):
        monkeypatch.setenv("AR_MOE_CHUNK", chunk)
        chunked = grouped_linear_experts_forward(experts, hidden_states, index, weights)
        torch.testing.assert_close(chunked, full, rtol=0, atol=0, msg=f"chunk={chunk} diverged")


def test_qdq_chunk_size_parsing(monkeypatch):
    from auto_round.modeling.fused_moe import grouped_experts as ge

    experts = _wrap_experts(_Experts())
    layers = _slot_layers(experts, "up_proj", range(NUM_EXPERTS))
    weight = layers[0].weight
    expected_auto = max(
        ge._DEFAULT_QDQ_CHUNK,
        ge._QDQ_FUSED_WORKINGSET_BYTES // (weight.numel() * (weight.element_size() + 4)),
    )

    monkeypatch.delenv("AR_MOE_CHUNK", raising=False)
    assert ge._qdq_chunk_size(layers) == expected_auto
    monkeypatch.setenv("AR_MOE_CHUNK", "auto")
    assert ge._qdq_chunk_size(layers) == expected_auto
    monkeypatch.setenv("AR_MOE_CHUNK", "4")
    assert ge._qdq_chunk_size(layers) == 4
    monkeypatch.setenv("AR_MOE_CHUNK", "0")
    assert ge._qdq_chunk_size(layers) == 0  # 0 == fuse everything
    monkeypatch.setenv("AR_MOE_CHUNK", "garbage")
    assert ge._qdq_chunk_size(layers) == expected_auto

    # The configured size must NOT be clamped to the active count: clamping would let a
    # short active list silently shrink the group and make the fused shape follow routing.
    monkeypatch.setenv("AR_MOE_CHUNK", "16")
    assert ge._qdq_chunk_size(layers[:3]) == 16


@pytest.mark.parametrize("allow_compiled", [False, True])
def test_fused_qdq_selects_quant_func_by_shape_policy(allow_compiled):
    """Fixed groups may compile; router-sized groups must use the eager callable."""
    from auto_round.modeling.fused_moe.grouped_experts import _batched_qdq_weights

    torch.manual_seed(34)
    experts = _wrap_experts(_Experts())
    layers = _slot_layers(experts, "up_proj", range(NUM_EXPERTS))

    compiled_calls, eager_calls = [], []
    for layer in layers:
        eager = layer.weight_quant_func

        def _compiled(*a, _f=eager, **kw):
            compiled_calls.append(1)
            return _f(*a, **kw)

        def _eager(*a, _f=eager, **kw):
            eager_calls.append(1)
            return _f(*a, **kw)

        layer.weight_quant_func = _compiled
        layer.weight_quant_func_eager = _eager

    _batched_qdq_weights(layers, allow_compiled=allow_compiled)
    assert bool(compiled_calls) is allow_compiled
    assert bool(eager_calls) is not allow_compiled


def test_wrapper_preserves_eager_quant_func_before_compile():
    from auto_round.modeling.fused_moe.grouped_experts import _batched_qdq_weights
    from auto_round.wrapper import WrapperLinear

    with patch("auto_round.wrapper.compile_func", side_effect=lambda func, device: lambda *a, **kw: func(*a, **kw)):
        layer = WrapperLinear(_mark_quantizable(nn.Linear(HIDDEN, INTERMEDIATE)), enable_torch_compile=True)
    assert layer.weight_quant_func_eager is not layer.weight_quant_func
    expected = layer._qdq_weight(layer.value, layer.min_scale, layer.max_scale)[0]
    with patch.object(layer, "weight_quant_func", side_effect=AssertionError("dynamic tail must remain eager")):
        _, got = _batched_qdq_weights([layer], allow_compiled=False)
    torch.testing.assert_close(got[0], expected, rtol=0, atol=0)


def test_optimized_wrapper_preserves_its_own_eager_quant_func():
    from auto_round.algorithms.quantization.sign_roundv2.quantizer import SignRoundOptimizedWrapperLinear
    from auto_round.modeling.fused_moe.grouped_experts import _batched_qdq_weights

    torch.manual_seed(36)
    experts = _wrap_experts(_Experts(), wrapper_cls=SignRoundOptimizedWrapperLinear)
    layers = _slot_layers(experts, "up_proj", range(NUM_EXPERTS))
    assert all(layer.weight_quant_func_eager is layer.weight_quant_func for layer in layers)
    expected = [layer._qdq_weight(layer.value, layer.min_scale, layer.max_scale)[0] for layer in layers]
    with patch.object(layers[0], "weight_quant_func", side_effect=AssertionError("tail must use optimized eager QDQ")):
        _, got = _batched_qdq_weights(layers, allow_compiled=False)
    for reference, actual in zip(expected, got):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_fused_qdq_falls_back_when_no_eager_func_is_stashed():
    """Wrappers built before this existed must still work."""
    from auto_round.modeling.fused_moe.grouped_experts import _batched_qdq_weights

    torch.manual_seed(35)
    experts = _wrap_experts(_Experts())
    layers = _slot_layers(experts, "up_proj", range(NUM_EXPERTS))
    expected = [layer._qdq_weight(layer.value, layer.min_scale, layer.max_scale)[0] for layer in layers]
    for layer in layers:
        del layer.weight_quant_func_eager

    _, got = _batched_qdq_weights(layers, allow_compiled=False)
    for reference, actual in zip(expected, got):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_qdq_chunking_is_the_default(monkeypatch):
    """A layer with more experts than the chunk really does get split."""
    from auto_round.modeling.fused_moe import grouped_experts as ge

    monkeypatch.delenv("AR_MOE_CHUNK", raising=False)
    monkeypatch.setattr(ge, "_DEFAULT_QDQ_CHUNK", 2)
    monkeypatch.setattr(ge, "_QDQ_FUSED_WORKINGSET_BYTES", 1)

    experts = _wrap_experts(_Experts())
    layers = _slot_layers(experts, "up_proj", range(NUM_EXPERTS))
    assert ge._qdq_chunk_size(layers) == 2

    calls = []
    original = ge._batched_qdq_weights
    monkeypatch.setattr(
        ge, "_batched_qdq_weights", lambda group, **kw: (calls.append(len(group)), original(group, **kw))[1]
    )

    hidden_states = torch.randn(NUM_TOKENS, HIDDEN)
    index, weights = _make_routing()
    got = grouped_linear_experts_forward(experts, hidden_states, index, weights)

    assert calls and max(calls) == 2, f"expected groups of 2, got {calls}"
    expected = linear_loop_experts_forward(experts, hidden_states, index, weights)
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


def test_fused_shape_is_static_under_torch_compile(monkeypatch):
    """The fused group size must not depend on how many experts got routed.

    ``torch.compile``'s dynamo cache is keyed by code object, shared across every
    WrapperLinear, so a group whose size follows the router would add a recompile per
    distinct active-expert count -- once per calibration sample.
    """
    from auto_round.modeling.fused_moe import grouped_experts as ge

    monkeypatch.delenv("AR_MOE_CHUNK", raising=False)
    monkeypatch.setattr(ge, "_DEFAULT_QDQ_CHUNK", 2)
    monkeypatch.setattr(ge, "_QDQ_FUSED_WORKINGSET_BYTES", 1)
    experts = _wrap_experts(_Experts())
    for i in range(NUM_EXPERTS):
        for name in ("gate_proj", "up_proj", "down_proj"):
            getattr(getattr(experts, str(i)), name).enable_torch_compile = True

    layers = _slot_layers(experts, "up_proj", range(NUM_EXPERTS))
    seen_group_sizes = set()
    for active in range(1, NUM_EXPERTS + 1):
        groups, tail, allow_compiled = ge._plan_qdq_groups(layers[:active])
        assert allow_compiled
        seen_group_sizes.update(len(group) for group in groups)
        assert len(groups) * 2 + len(tail) == active, f"lost experts at active={active}"

    assert seen_group_sizes <= {2}, f"fused shape varies with the routing: {seen_group_sizes}"


def test_partial_groups_are_fused_without_torch_compile(monkeypatch):
    """No compile, no shape constraint: the leftover experts get fused too."""
    from auto_round.modeling.fused_moe import grouped_experts as ge

    monkeypatch.delenv("AR_MOE_CHUNK", raising=False)
    monkeypatch.setattr(ge, "_DEFAULT_QDQ_CHUNK", 4)
    monkeypatch.setattr(ge, "_QDQ_FUSED_WORKINGSET_BYTES", 1)
    experts = _wrap_experts(_Experts())  # enable_torch_compile=False
    layers = _slot_layers(experts, "up_proj", range(NUM_EXPERTS))

    groups, tail, allow_compiled = ge._plan_qdq_groups(layers[:6])
    assert allow_compiled
    assert [len(g) for g in groups] == [4, 2]
    assert tail == []


@pytest.mark.parametrize("optimized", [False, True])
def test_grouped_matches_loop_with_torch_compile_flag(monkeypatch, optimized):
    """The static-shape path must still produce the loop's answer, tail included."""
    from auto_round.modeling.fused_moe import grouped_experts as ge

    monkeypatch.setenv("AR_MOE_CHUNK", "4")  # NUM_EXPERTS=6 -> one group + tail of 2
    torch.manual_seed(33)
    wrapper_cls = None
    if optimized:
        from auto_round.algorithms.quantization.sign_roundv2.quantizer import SignRoundOptimizedWrapperLinear

        wrapper_cls = SignRoundOptimizedWrapperLinear
    experts = _wrap_experts(_Experts(), wrapper_cls=wrapper_cls)
    for i in range(NUM_EXPERTS):
        for name in ("gate_proj", "up_proj", "down_proj"):
            getattr(getattr(experts, str(i)), name).enable_torch_compile = True

    hidden_states = torch.randn(NUM_TOKENS, HIDDEN)
    index = torch.tensor([[i % NUM_EXPERTS, (i + 1) % NUM_EXPERTS] for i in range(NUM_TOKENS)])
    weights = torch.full((NUM_TOKENS, 2), 0.5)

    expected = linear_loop_experts_forward(experts, hidden_states, index, weights)
    expected.square().mean().backward()
    expected_grads = {name: None if p.grad is None else p.grad.clone() for name, p in experts.named_parameters()}
    for parameter in experts.parameters():
        parameter.grad = None
    with patch.object(ge, "_batched_qdq_weights", wraps=ge._batched_qdq_weights) as batched:
        got = grouped_linear_experts_forward(experts, hidden_states, index, weights)
    assert [(len(call.args[0]), call.kwargs["allow_compiled"]) for call in batched.call_args_list] == [
        (4, True),
        (2, False),
    ] * 3
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)
    got.square().mean().backward()
    for name, parameter in experts.named_parameters():
        reference = expected_grads[name]
        assert (parameter.grad is None) == (reference is None)
        if reference is not None:
            torch.testing.assert_close(parameter.grad, reference, rtol=1e-4, atol=1e-6)


def test_resolve_experts_implementation(monkeypatch):
    monkeypatch.delenv("AR_MOE_EXPERTS_IMPL", raising=False)
    assert resolve_experts_implementation() == GROUPED_LINEAR_IMPL
    # Explicit loop remains available; sliced selects the grouped forward with
    # a different GEMM policy, rather than a different experts implementation.
    monkeypatch.setenv("AR_MOE_EXPERTS_IMPL", "linear_loop")
    assert resolve_experts_implementation() == "linear_loop"
    monkeypatch.setenv("AR_MOE_EXPERTS_IMPL", "linear_grouped_sliced")
    assert resolve_experts_implementation() == GROUPED_LINEAR_IMPL
