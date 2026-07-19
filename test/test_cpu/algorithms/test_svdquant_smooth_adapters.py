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

from auto_round.algorithms.transforms.svdquant.smooth_adapters import discover_smooth_search_groups


class _Projection(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(in_features, out_features)


class _FeedForward(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = torch.nn.ModuleList(
            [_Projection(width, width * 2), torch.nn.GELU(), torch.nn.Linear(width * 2, width)]
        )


class _Attention(torch.nn.Module):
    def __init__(self, width: int, *, added: bool) -> None:
        super().__init__()
        self.to_q = torch.nn.Linear(width, width)
        self.to_k = torch.nn.Linear(width, width)
        self.to_v = torch.nn.Linear(width, width)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(width, width)])
        if added:
            self.add_q_proj = torch.nn.Linear(width, width)
            self.add_k_proj = torch.nn.Linear(width, width)
            self.add_v_proj = torch.nn.Linear(width, width)
            self.to_add_out = torch.nn.Linear(width, width)

    def forward(self, hidden_states, encoder_hidden_states=None, **_kwargs):
        hidden_output = self.to_q(hidden_states) + self.to_k(hidden_states) + self.to_v(hidden_states)
        if encoder_hidden_states is None:
            return hidden_output
        encoder_output = (
            self.add_q_proj(encoder_hidden_states)
            + self.add_k_proj(encoder_hidden_states)
            + self.add_v_proj(encoder_hidden_states)
        )
        return hidden_output, encoder_output


class FluxTransformerBlock(torch.nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.attn = _Attention(width, added=True)
        self.ff = _FeedForward(width)
        self.ff_context = _FeedForward(width)


class FluxSingleTransformerBlock(torch.nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.attn = _Attention(width, added=False)
        self.proj_mlp = torch.nn.Linear(width, width * 2)
        self.proj_out = torch.nn.Linear(width * 3, width)


class UnknownBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Linear(4, 5)
        self.nested = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(5, 3))


def _all_targets(_name: str, module: torch.nn.Module) -> bool:
    return isinstance(module, torch.nn.Linear)


def test_flux_double_block_discovers_expected_groups():
    block = FluxTransformerBlock()
    block.global_name = "transformer_blocks.7"

    groups = discover_smooth_search_groups(block, _all_targets)
    by_key = {group.key: group for group in groups}

    assert set(by_key) == {
        "transformer_blocks.7.attn.qkv",
        "transformer_blocks.7.attn.add_qkv",
        "transformer_blocks.7.attn.to_out.0",
        "transformer_blocks.7.attn.to_add_out",
        "transformer_blocks.7.ff.net.0.proj",
        "transformer_blocks.7.ff.net.2",
        "transformer_blocks.7.ff_context.net.0.proj",
        "transformer_blocks.7.ff_context.net.2",
    }
    assert by_key["transformer_blocks.7.attn.qkv"].projection_names == (
        "transformer_blocks.7.attn.to_q",
        "transformer_blocks.7.attn.to_k",
        "transformer_blocks.7.attn.to_v",
    )
    assert by_key["transformer_blocks.7.attn.qkv"].evaluation_module is block.attn
    assert by_key["transformer_blocks.7.attn.qkv"].output_indices == (0,)
    assert by_key["transformer_blocks.7.attn.add_qkv"].output_indices == (1,)


def test_flux_single_block_groups_parallel_qkv_and_mlp_up_projection():
    block = FluxSingleTransformerBlock()
    block.global_name = "single_transformer_blocks.3"

    groups = discover_smooth_search_groups(block, _all_targets)
    by_key = {group.key: group for group in groups}

    parallel = by_key["single_transformer_blocks.3.parallel_qkv_mlp"]
    assert parallel.projection_names == (
        "single_transformer_blocks.3.attn.to_q",
        "single_transformer_blocks.3.attn.to_k",
        "single_transformer_blocks.3.attn.to_v",
        "single_transformer_blocks.3.proj_mlp",
    )
    assert parallel.evaluation_module is block
    assert parallel.output_splits == (3, 1)
    assert by_key["single_transformer_blocks.3.proj_out"].projections == (block.proj_out,)


def test_discovery_filters_targets_and_assigns_each_linear_once():
    block = FluxTransformerBlock()
    block.global_name = "transformer_blocks.0"

    groups = discover_smooth_search_groups(
        block,
        lambda name, module: isinstance(module, torch.nn.Linear) and "ff_context" not in name,
    )
    projections = [projection for group in groups for projection in group.projections]

    assert len(projections) == len({id(projection) for projection in projections})
    assert all("ff_context" not in name for group in groups for name in group.projection_names)
    assert all(len({projection.in_features for projection in group.projections}) == 1 for group in groups)


def test_unknown_block_falls_back_to_one_linear_per_group():
    block = UnknownBlock()
    block.global_name = "unknown.2"

    groups = discover_smooth_search_groups(block, _all_targets)

    assert [group.key for group in groups] == ["unknown.2.first", "unknown.2.nested.1"]
    assert all(len(group.projections) == 1 for group in groups)
    assert all(group.evaluation_module is group.projections[0] for group in groups)
    assert all(group.projection_input_key == group.evaluation_input_key for group in groups)


def test_output_normalizer_selects_parent_tuple_outputs():
    block = FluxTransformerBlock()
    block.global_name = "transformer_blocks.0"
    by_key = {group.key: group for group in discover_smooth_search_groups(block, _all_targets)}
    outputs = (torch.tensor([1.0]), torch.tensor([2.0]))

    qkv = by_key["transformer_blocks.0.attn.qkv"]
    added_qkv = by_key["transformer_blocks.0.attn.add_qkv"]

    assert qkv.normalize_output(outputs) == (outputs[0],)
    assert added_qkv.normalize_output(outputs) == (outputs[1],)


def test_transform_captures_bounded_group_inputs_and_shared_parent_call():
    from auto_round.algorithms.transforms.svdquant.apply import SVDQuantTransform
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig

    block = FluxTransformerBlock()
    block.global_name = "transformer_blocks.0"
    transform = SVDQuantTransform(SVDQuantConfig(smooth_enabled=True))
    ctx = type("Context", (), {"block": block, "block_name": block.global_name})()
    hidden_states = torch.randn(2, 3, 4)
    encoder_hidden_states = torch.randn(2, 2, 4)

    with transform.block_forward_hooks(ctx) as handles:
        assert len(handles) == 9
        block.attn(hidden_states, encoder_hidden_states=encoder_hidden_states, ignored=True)
        block.attn(hidden_states + 1, encoder_hidden_states=encoder_hidden_states + 1)

    qkv = transform._smooth_calibration["transformer_blocks.0.attn.qkv"]
    added_qkv = transform._smooth_calibration["transformer_blocks.0.attn.add_qkv"]
    assert len(qkv.projection_inputs) == 1
    assert len(qkv.evaluation_calls) == 1
    assert len(added_qkv.projection_inputs) == 1
    assert qkv.evaluation_calls[0] is added_qkv.evaluation_calls[0]
    assert qkv.projection_inputs[0].device.type == "cpu"
    assert qkv.evaluation_calls[0].args[0].device.type == "cpu"
    assert not block.attn._forward_hooks
    assert not block.attn.to_q._forward_hooks

    transform._clear_smooth_calibration()
    assert transform._smooth_calibration == {}


def test_flux_qkv_search_uses_parent_output_and_final_shared_down(monkeypatch):
    import auto_round.algorithms.transforms.svdquant.residual as residual_module
    from auto_round.algorithms.transforms.svdquant.apply import SVDQuantTransform
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig
    from auto_round.algorithms.transforms.svdquant.smooth import build_alpha_beta_candidates, build_smooth_scale

    block = FluxTransformerBlock(width=2)
    block.global_name = "transformer_blocks.0"
    qkv = (block.attn.to_q, block.attn.to_k, block.attn.to_v)
    weights = (
        torch.tensor([[3.0, 0.5], [-1.0, 2.0]]),
        torch.tensor([[-2.0, 1.0], [0.25, 1.5]]),
        torch.tensor([[1.0, -2.0], [2.5, 0.75]]),
    )
    for name, projection, weight in zip(("to_q", "to_k", "to_v"), qkv, weights):
        projection.weight.data.copy_(weight)
        projection.bias.data.zero_()
        projection.data_type = "int"
        projection.bits = 4
        projection.group_size = 2
        projection.sym = True
        projection.global_name = f"transformer_blocks.0.attn.{name}"

    hidden_states = torch.tensor([[[1.0, -2.0], [0.5, 3.0]]])
    encoder_hidden_states = torch.zeros(1, 1, 2)
    reference_output = sum(hidden_states @ weight.T for weight in weights)
    x_span = hidden_states.abs().reshape(-1, 2).amax(dim=0)
    w_span = torch.cat(weights, dim=0).abs().amax(dim=0)
    references = []
    for alpha, beta in build_alpha_beta_candidates(4):
        scale = build_smooth_scale(x_span, w_span, alpha, beta)
        stacked = torch.cat([weight * scale for weight in weights], dim=0)
        u, singular_values, vh = torch.linalg.svd(stacked, full_matrices=False)
        low_rank = (u[:, :1] * singular_values[:1].reshape(1, -1)) @ vh[:1]
        deployed = torch.round(stacked - low_rank) + low_rank
        actual = sum((hidden_states / scale) @ part.T for part in deployed.split(2, dim=0))
        references.append((scale, torch.sum((actual - reference_output).square()).item()))
    expected_scale = None
    best_error = float("inf")
    for scale, error in references:
        if error <= best_error:
            expected_scale = scale
            best_error = error

    monkeypatch.setattr(residual_module, "rtn_qdq_residual", lambda residual, _scheme: torch.round(residual))
    transform = SVDQuantTransform(
        SVDQuantConfig(
            rank=1,
            smooth_enabled=True,
            smooth_num_grids=4,
            low_rank_dtype="float32",
            target_modules=["attn.to_q", "attn.to_k", "attn.to_v"],
        )
    )
    ctx = type("Context", (), {"block": block, "block_name": block.global_name})()

    with transform.block_forward_hooks(ctx):
        block.attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
    transform.pre_quantize_block(ctx)

    assert expected_scale is not None
    wrappers = (block.attn.to_q, block.attn.to_k, block.attn.to_v)
    for wrapper in wrappers:
        torch.testing.assert_close(wrapper.smooth, expected_scale.reciprocal(), rtol=0, atol=1e-6)
        torch.testing.assert_close(wrapper.lora_down.weight, wrappers[0].lora_down.weight, rtol=0, atol=0)
    actual_output = block.attn(hidden_states, encoder_hidden_states=encoder_hidden_states)[0]
    torch.testing.assert_close(actual_output, reference_output, rtol=0, atol=1e-5)
