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

from auto_round.algorithms.transforms.awq.base import AWQTransform
from auto_round.algorithms.transforms.awq.config import AWQConfig
from auto_round.algorithms.transforms.awq.mappings import ResolvedMapping
from auto_round.algorithms.transforms.smoothing.search import search_candidates


class Parent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(8)
        self.q = torch.nn.Linear(8, 8, bias=False)
        self.k = torch.nn.Linear(8, 8, bias=False)

    def forward(self, inputs):
        hidden = self.norm(inputs)
        output = self.q(hidden) + self.k(hidden)
        # AWQ evaluates only the first tuple output.
        return output, output.square() * 100


@pytest.mark.parametrize("duo_scaling", [False, True, "both"])
@pytest.mark.parametrize("parent_forward", [False, True])
@torch.no_grad()
def test_shared_search_preserves_awq_candidates_and_loss(duo_scaling, parent_forward):
    torch.manual_seed(41)
    parent = Parent()
    parent.q.global_name = "block.q"
    parent.k.global_name = "block.k"
    mapping = ResolvedMapping(
        smooth_name="block.norm",
        smooth_layer=parent.norm,
        balance_names=["block.q", "block.k"],
        balance_layers=[parent.q, parent.k],
        parent_name="block",
        parent=parent,
    )
    transform = AWQTransform(
        AWQConfig(
            bits=4,
            group_size=8,
            sym=True,
            data_type="int",
            duo_scaling=duo_scaling,
            n_grid=6,
            smooth_batch_size=1,
        )
    )
    calls = [((torch.randn(batch, 3, 8),), {}) for batch in (2, 1)]
    if parent_forward:
        transform._parent_args_cache[parent] = calls
    x_mean = torch.cat([parent.norm(args[0]).reshape(-1, 8) for args, _ in calls]).abs().mean(0)
    original = {layer: layer.weight.detach().clone() for layer in mapping.balance_layers}
    expected = transform._grid_search_scales(mapping, x_mean)
    assert expected is not None
    params = {layer: transform._qdq_tool.resolve_params(layer) for layer in mapping.balance_layers}
    funcs = {layer: transform._qdq_tool.resolve_quant_funcs(params[layer]) for layer in mapping.balance_layers}
    w_mean = transform._compute_layer_means(mapping.balance_layers, 8)
    reference = transform._run_parent_samples(parent, calls, offload_to_cpu=True) if parent_forward else None
    candidates = []
    for ratio, use_duo in transform._get_grid_search_params():
        if use_duo:
            scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
        else:
            scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)
        scales = scales / (scales.max() * scales.min()).sqrt()
        scales[torch.isinf(scales)] = 1
        scales[torch.isnan(scales)] = 1
        candidates.append(scales)

    def score(scales):
        error = 0.0
        try:
            for layer in mapping.balance_layers:
                quant_func, opt_quant_func = funcs[layer]
                quantized = transform._qdq_tool.qdq(
                    original[layer] * scales.view(1, -1),
                    params[layer],
                    quant_func=quant_func,
                    opt_quant_func=opt_quant_func,
                    imatrix=getattr(layer, "imatrix", None),
                ) / scales.view(1, -1)
                if parent_forward:
                    layer.weight.copy_(quantized)
                else:
                    error += (original[layer] - quantized).pow(2).sum().item()
            if parent_forward:
                return transform._compute_parent_loss(parent, calls, reference)
            return error
        finally:
            for layer, weight in original.items():
                layer.weight.copy_(weight)

    selected, _ = search_candidates(
        candidates,
        score,
        module_name=mapping.smooth_name,
    )
    torch.testing.assert_close(selected, expected, rtol=0, atol=0)
    for layer, weight in original.items():
        torch.testing.assert_close(layer.weight, weight, rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["qdq", "replay"])
def test_awq_failed_trial_restores_weights_and_clears_cache(monkeypatch, failure):
    from types import SimpleNamespace

    parent = Parent()
    parent.q.global_name, parent.k.global_name = "block.q", "block.k"
    mapping = ResolvedMapping(
        smooth_name="block.norm",
        smooth_layer=parent.norm,
        balance_names=["block.q", "block.k"],
        balance_layers=[parent.q, parent.k],
        parent_name="block",
        parent=parent,
    )
    transform = AWQTransform(AWQConfig(bits=4, group_size=8, sym=True, data_type="int", n_grid=2))
    transform._block_mappings = {"block": [mapping]}
    transform._activation_stats[mapping.smooth_name] = [torch.ones(8), 1]
    transform._parent_args_cache[parent] = [((torch.randn(1, 3, 8),), {})]
    transform._clip_input_feat[mapping.smooth_name] = torch.ones(1, 8)
    originals = {layer: layer.weight.detach().clone() for layer in mapping.balance_layers}
    calls = 0

    def qdq(weight, *args, **kwargs):
        nonlocal calls
        calls += 1
        if failure == "qdq" and calls == 2:
            raise RuntimeError("trial failure")
        return torch.zeros_like(weight)

    def replay(*args, **kwargs):
        raise RuntimeError("trial failure")

    monkeypatch.setattr(transform._qdq_tool, "qdq", qdq)
    monkeypatch.setattr(transform, "_compute_parent_loss", replay)
    with pytest.raises(RuntimeError, match="trial failure"):
        transform.pre_quantize_block(SimpleNamespace(block_names=["block"], block_name="block"))
    for layer, weight in originals.items():
        torch.testing.assert_close(layer.weight, weight, rtol=0, atol=0)
    assert not transform._activation_stats
    assert not transform._parent_args_cache
    assert not transform._clip_input_feat
