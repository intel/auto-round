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

from types import SimpleNamespace

import torch

from auto_round.algorithms.composer import AlgorithmComposer, BlockContext
from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.algorithms.transforms.svdquant import SVDQuantConfig, SVDQuantLinear
from auto_round.algorithms.transforms.svdquant.apply import SVDQuantTransform


class FluxAttention(torch.nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.to_q = torch.nn.Linear(width, 4, bias=False)
        self.to_k = torch.nn.Linear(width, 4, bias=False)
        self.to_v = torch.nn.Linear(width, 4, bias=False)


class FluxTransformerBlock(torch.nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.attn = FluxAttention(width)
        self.norm1 = torch.nn.Module()
        self.norm1.linear = torch.nn.Linear(width, width, bias=False)


class TinyFlux(torch.nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([FluxTransformerBlock(width)])


def _mark_modules(model):
    for name, module in model.named_modules():
        module.global_name = name
        if isinstance(module, torch.nn.Linear):
            module.bits = 4
            module.group_size = 32
            module.sym = True
            module.data_type = "mx_fp4e2m1"
            module.act_bits = 16


def test_svdquant_composes_before_rtn():
    composer = AlgorithmComposer([SVDQuantConfig(rank=2), RTNConfig(disable_opt_rtn=True)])

    assert len(composer.preprocessors) == 1
    assert isinstance(composer.preprocessors[0], SVDQuantTransform)


def test_explicit_flux_adapter_resolves_targets_before_prepare_run():
    transform = SVDQuantTransform(SVDQuantConfig(rank=2, model_adapter="flux"))
    projection = torch.nn.Linear(8, 8)
    projection.global_name = "transformer_blocks.0.attn.to_q"
    adanorm = torch.nn.Linear(8, 8)
    adanorm.global_name = "transformer_blocks.0.norm1.linear"

    assert transform._is_target("attn.to_q", projection)
    assert not transform._is_target("norm1.linear", adanorm)


def test_no_smooth_flux_qkv_share_one_down_factor():
    model = TinyFlux()
    _mark_modules(model)
    block_name = "transformer_blocks.0"
    inputs = torch.randn(3, 8)
    expected = tuple(
        projection(inputs)
        for projection in (
            model.transformer_blocks[0].attn.to_q,
            model.transformer_blocks[0].attn.to_k,
            model.transformer_blocks[0].attn.to_v,
        )
    )
    transform = SVDQuantTransform(SVDQuantConfig(rank=2, residual_iters=1, low_rank_dtype="fp32"))
    orchestrator = SimpleNamespace(
        model_context=SimpleNamespace(model=model),
        compress_context=None,
        calibration_context=None,
        scheme_context=None,
        scale_dtype=None,
        nblocks=1,
        quant_block_list=[[block_name]],
    )
    transform.bind(orchestrator)
    transform.prepare_run()

    transform.pre_quantize_block(
        BlockContext(model=model, block_names=[block_name], block_name=block_name, block_index=0)
    )

    q = model.transformer_blocks[0].attn.to_q
    k = model.transformer_blocks[0].attn.to_k
    v = model.transformer_blocks[0].attn.to_v
    assert all(isinstance(module, SVDQuantLinear) for module in (q, k, v))
    torch.testing.assert_close(q.lora_down.weight, k.lora_down.weight)
    torch.testing.assert_close(q.lora_down.weight, v.lora_down.weight)
    assert q.residual_linear.data_type == "mx_fp4e2m1_rceil"
    assert q.lora_down.bits == 16
    assert q.lora_up.bits == 16
    for actual, reference in zip((q(inputs), k(inputs), v(inputs)), expected):
        torch.testing.assert_close(actual, reference)


def test_flux_adapter_default_targets_only_runtime_supported_projections():
    model = TinyFlux()
    model.config = {"_class_name": "FluxTransformer2DModel"}
    _mark_modules(model)
    block_name = "transformer_blocks.0"
    transform = SVDQuantTransform(SVDQuantConfig(rank=2, model_adapter="auto", low_rank_dtype="fp32"))
    orchestrator = SimpleNamespace(
        model_context=SimpleNamespace(model=model),
        compress_context=None,
        calibration_context=None,
        scheme_context=None,
        scale_dtype=None,
        nblocks=1,
        quant_block_list=[[block_name]],
    )
    transform.bind(orchestrator)
    transform.prepare_run()

    transform.pre_quantize_block(
        BlockContext(model=model, block_names=[block_name], block_name=block_name, block_index=0)
    )

    assert isinstance(model.transformer_blocks[0].attn.to_q, SVDQuantLinear)
    assert isinstance(model.transformer_blocks[0].norm1.linear, torch.nn.Linear)


def test_no_smooth_grouped_residual_iteration_and_cleanup():
    model = TinyFlux(width=32)
    _mark_modules(model)
    block_name = "transformer_blocks.0"
    transform = SVDQuantTransform(
        SVDQuantConfig(rank=2, residual_iters=2, residual_early_stop=True, low_rank_dtype="fp32")
    )
    orchestrator = SimpleNamespace(
        model_context=SimpleNamespace(model=model),
        compress_context=None,
        calibration_context=None,
        scheme_context=None,
        scale_dtype=None,
        nblocks=1,
        quant_block_list=[[block_name]],
    )
    transform.bind(orchestrator)
    transform.prepare_run()
    ctx = BlockContext(model=model, block_names=[block_name], block_name=block_name, block_index=0)

    transform.pre_quantize_block(ctx)

    assert isinstance(model.transformer_blocks[0].attn.to_q, SVDQuantLinear)
    assert block_name in transform._block_groups
    transform.post_quantize_block(ctx)
    assert block_name not in transform._block_groups
