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

from auto_round.algorithms.transforms.svdquant.smooth_adapters import discover_svdquant_groups


class Attention(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.to_q = torch.nn.Linear(width, width)
        self.to_k = torch.nn.Linear(width, width)
        self.to_v = torch.nn.Linear(width, width)
        self.add_q_proj = torch.nn.Linear(width, width)
        self.add_k_proj = torch.nn.Linear(width, width)
        self.add_v_proj = torch.nn.Linear(width, width)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(width, width)])
        self.to_add_out = torch.nn.Linear(width, width)


class Projection(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.proj = torch.nn.Linear(in_features, out_features)


class FeedForward(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = torch.nn.ModuleList(
            [Projection(width, width * 2), torch.nn.GELU(), torch.nn.Linear(width * 2, width)]
        )


class FluxTransformerBlock(torch.nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.attn = Attention(width)
        self.ff = FeedForward(width)
        self.ff_context = FeedForward(width)


def test_flux_adapter_discovers_fused_projection_groups_without_duplicates():
    block = FluxTransformerBlock()
    block.global_name = "transformer_blocks.7"

    groups = discover_svdquant_groups(block, lambda _name, module: isinstance(module, torch.nn.Linear))
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
    assert len(by_key["transformer_blocks.7.attn.qkv"].projections) == 3
    assert len(by_key["transformer_blocks.7.attn.add_qkv"].projections) == 3
    assert len({id(projection) for group in groups for projection in group.projections}) == 12
