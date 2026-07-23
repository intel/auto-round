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

import pytest
import torch
from torch import nn

from auto_round.modeling.fused_moe.fusion_spec import (
    MoEFusionSpec,
    MoETensorSource,
    ProjectionFusionSpec,
    build_standard_moe_fusion_spec,
    get_moe_fusion_spec,
    iter_moe_fusion_views,
    register_moe_fusion_spec,
)


class _Expert(nn.Module):
    def __init__(self, expert_index: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(3, 2, bias=bias)
        self.up_proj = nn.Linear(3, 2, bias=bias)
        self.down_proj = nn.Linear(2, 3, bias=bias)

        offset = expert_index * 100
        with torch.no_grad():
            self.gate_proj.weight.copy_(torch.arange(6).reshape(2, 3) + offset)
            self.up_proj.weight.copy_(torch.arange(6).reshape(2, 3) + offset + 10)
            self.down_proj.weight.copy_(torch.arange(6).reshape(3, 2) + offset + 20)
            if bias:
                self.gate_proj.bias.copy_(torch.arange(2) + offset + 30)
                self.up_proj.bias.copy_(torch.arange(2) + offset + 40)
                self.down_proj.bias.copy_(torch.arange(3) + offset + 50)


class _Experts(nn.ModuleList):
    pass


class _Layer(nn.Module):
    def __init__(self, bias: bool = False):
        super().__init__()
        self.experts = _Experts([_Expert(0, bias=bias), _Expert(1, bias=bias)])


class _Model(nn.Module):
    def __init__(self, bias: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(bias=bias)])


def _standard_spec(model: _Model, checkpoint_transposed: bool = False):
    detected_projections = {
        "gate_up_proj": {"split_into": ["gate_proj", "up_proj"]},
        "down_proj": {},
    }
    return build_standard_moe_fusion_spec(
        detected_projections=detected_projections,
        num_experts=2,
        checkpoint_transposed=checkpoint_transposed,
        module=model.layers[0].experts,
    )


def test_non_transposed_views_reconstruct_checkpoint_without_mutating_model():
    model = _Model()
    experts = model.layers[0].experts
    spec = _standard_spec(model)
    register_moe_fusion_spec(experts, spec)

    gate_up = torch.stack(
        [torch.cat([experts[i].gate_proj.weight, experts[i].up_proj.weight], dim=0) for i in range(2)], dim=0
    )
    down = torch.stack([experts[i].down_proj.weight for i in range(2)], dim=0)

    views = {view.checkpoint_name: view for view in iter_moe_fusion_views(model)}

    assert torch.equal(views["layers.0.experts.gate_up_proj"].tensor_fn(), gate_up)
    assert torch.equal(views["layers.0.experts.down_proj"].tensor_fn(), down)
    assert views["layers.0.experts.gate_up_proj"].sources == (
        MoETensorSource(
            projection="gate_proj",
            hf_names=(
                "layers.0.experts.0.gate_proj.weight",
                "layers.0.experts.1.gate_proj.weight",
            ),
        ),
        MoETensorSource(
            projection="up_proj",
            hf_names=(
                "layers.0.experts.0.up_proj.weight",
                "layers.0.experts.1.up_proj.weight",
            ),
        ),
    )
    assert tuple(experts._modules) == ("0", "1")
    assert "gate_up_proj" not in experts._parameters
    assert "_auto_round_moe_fusion_spec" not in experts._modules
    assert "_auto_round_moe_fusion_spec" not in experts._parameters
    assert "_auto_round_moe_fusion_spec" not in experts._buffers
    assert get_moe_fusion_spec(experts) is spec


def test_transposed_view_reconstructs_checkpoint_weight():
    model = _Model()
    experts = model.layers[0].experts
    register_moe_fusion_spec(experts, _standard_spec(model, checkpoint_transposed=True))
    gate_up = torch.stack(
        [torch.cat([experts[i].gate_proj.weight, experts[i].up_proj.weight], dim=0) for i in range(2)], dim=0
    )

    views = {view.checkpoint_name: view for view in iter_moe_fusion_views(model)}

    assert torch.equal(views["layers.0.experts.gate_up_proj"].tensor_fn(), gate_up.transpose(1, 2))


def test_bias_views_reconstruct_checkpoint_without_weight_transpose():
    model = _Model(bias=True)
    experts = model.layers[0].experts
    spec = MoEFusionSpec(
        num_experts=2,
        projections=(
            ProjectionFusionSpec(
                checkpoint_projection="gate_up_proj",
                source_projections=("gate_proj", "up_proj"),
                concat_dim=1,
                checkpoint_transposed=True,
                checkpoint_bias="gate_up_proj_bias",
            ),
            ProjectionFusionSpec(
                checkpoint_projection="down_proj",
                source_projections=("down_proj",),
                concat_dim=None,
                checkpoint_transposed=True,
                checkpoint_bias="down_proj_bias",
            ),
        ),
    )
    register_moe_fusion_spec(experts, spec)
    gate_up_bias = torch.stack(
        [torch.cat([experts[i].gate_proj.bias, experts[i].up_proj.bias], dim=0) for i in range(2)], dim=0
    )
    down_bias = torch.stack([experts[i].down_proj.bias for i in range(2)], dim=0)

    views = {view.checkpoint_name: view for view in iter_moe_fusion_views(model)}

    assert torch.equal(views["layers.0.experts.gate_up_proj_bias"].tensor_fn(), gate_up_bias)
    assert torch.equal(views["layers.0.experts.down_proj_bias"].tensor_fn(), down_bias)
    assert all(source.parameter == "bias" for source in views["layers.0.experts.gate_up_proj_bias"].sources)


def test_missing_source_has_contextual_error():
    model = _Model()
    experts = model.layers[0].experts
    register_moe_fusion_spec(experts, _standard_spec(model))
    del experts[1].up_proj

    views = {view.checkpoint_name: view for view in iter_moe_fusion_views(model)}

    with pytest.raises(ValueError, match=r"layers\.0\.experts.*up_proj.*expert 1"):
        views["layers.0.experts.gate_up_proj"].tensor_fn()


def test_marked_linearized_replacement_without_fusion_spec_is_rejected():
    model = nn.Module()
    model.block = nn.Module()
    model.block._auto_round_replaced_fused_moe = True

    with pytest.raises(ValueError, match=r"^block: linearized fused MoE replacement has no MoEFusionSpec$"):
        list(iter_moe_fusion_views(model))


def test_qwen3_5_replacement_registers_descendant_fusion_spec():
    from auto_round.modeling.fused_moe.qwen3_5_moe import LinearQwen3_5MoeSparseMoeBlock

    hidden_size = 3
    intermediate_size = 2
    num_experts = 2

    class TinyExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.zeros(num_experts, 2 * intermediate_size, hidden_size))
            self.down_proj = nn.Parameter(torch.zeros(num_experts, hidden_size, intermediate_size))

    class TinyOriginal(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Identity()
            self.shared_expert = nn.Identity()
            self.experts = TinyExperts()
            self.shared_expert_gate = nn.Identity()

    text_config = type(
        "TextConfig",
        (),
        {
            "hidden_size": hidden_size,
            "hidden_act": "silu",
            "moe_intermediate_size": intermediate_size,
            "num_experts": num_experts,
        },
    )()
    config = type("Config", (), {"get_text_config": lambda self: text_config})()

    replacement = LinearQwen3_5MoeSparseMoeBlock(TinyOriginal(), config)

    assert replacement._auto_round_replaced_fused_moe is True
    assert get_moe_fusion_spec(replacement.experts) is not None
    assert [view.checkpoint_name for view in iter_moe_fusion_views(replacement)] == [
        "experts.gate_up_proj",
        "experts.down_proj",
    ]
