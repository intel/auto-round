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

import pytest
import torch

from auto_round.algorithms.composer import AlgorithmComposer, BlockContext
from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.algorithms.transforms.svdquant import SVDQuantConfig, SVDQuantLinear
from auto_round.schemes import PRESET_SCHEMES
from auto_round.utils.device_manager import device_manager


class DummyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(32, 16)

    def forward(self, hidden_states):
        return self.proj(hidden_states)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([DummyBlock()])


class FluxTransformerBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.to_q = torch.nn.Linear(32, 16)
        self.norm1 = torch.nn.Module()
        self.norm1.linear = torch.nn.Linear(32, 16)


class BasicTransformerBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = torch.nn.Module()
        self.attn1.to_q = torch.nn.Linear(32, 16)
        self.attn2 = torch.nn.Module()
        self.attn2.to_k = torch.nn.Linear(32, 16)


def _prepare_model():
    model = DummyModel()
    for name, module in model.named_modules():
        module.global_name = name
        if isinstance(module, torch.nn.Linear):
            module.bits = 4
            module.group_size = 32
            module.sym = True
            module.data_type = "mx_fp4e2m1"
            module.act_bits = 4
            module.act_group_size = 32
            module.act_sym = True
            module.act_data_type = "mx_fp4e2m1"
            module.act_dynamic = True
            module.scale_dtype = torch.float32
    return model


@pytest.mark.parametrize("smooth", [False, True], ids=["nosmooth", "smooth"])
@pytest.mark.parametrize("terminal", ["rtn", "signround"])
def test_svdquant_pipeline_smoke(monkeypatch, smooth, terminal):
    monkeypatch.setattr(device_manager, "_device_map", "cpu")
    monkeypatch.setattr(device_manager, "_device_list", ["cpu"])
    monkeypatch.setattr(device_manager, "_major_device", "cpu")
    model = _prepare_model()
    block_name = "blocks.0"
    svd_config = SVDQuantConfig(
        rank=2,
        smooth_enabled=smooth,
        smooth_num_grids=2,
        smooth_max_calibration_calls=1,
        residual_iters=1,
        low_rank_dtype="fp32",
        target_modules=["proj"],
    )
    terminal_config = (
        RTNConfig(disable_opt_rtn=True)
        if terminal == "rtn"
        else SignRoundConfig(
            iters=1,
            lr=1.0,
            minmax_lr=1.0,
            enable_minmax_tuning=False,
            enable_quanted_input=False,
        )
    )
    orchestrator = SimpleNamespace(
        model_context=SimpleNamespace(
            model=model,
            amp=True,
            amp_dtype=torch.bfloat16,
            is_diffusion=False,
            is_moe_model=False,
            output_config=None,
        ),
        compress_context=SimpleNamespace(
            enable_torch_compile=False,
            low_gpu_mem_usage=False,
            cache_device=torch.device("cpu"),
            clear_memory=lambda: None,
        ),
        calibration_context=SimpleNamespace(batch_size=1, batch_dim=0),
        scheme_context=PRESET_SCHEMES["MXFP4"],
        scale_dtype=None,
        nblocks=1,
        quant_block_list=[[block_name]],
        data_type="mx_fp4e2m1",
        batch_dim=0,
        batch_size=1,
        cache_device="cpu",
        amp=True,
        amp_dtype=torch.bfloat16,
        shared_cache_keys=(),
    )
    composer = AlgorithmComposer([svd_config, terminal_config], orchestrator=orchestrator)
    composer.prepare_run()
    block = model.blocks[0]
    inputs = torch.randn(1, 2, 32)
    ctx = BlockContext(
        model=model,
        block_names=[block_name],
        block_name=block_name,
        block_index=0,
        block_cnt=1,
    )

    composer.compress_block(block, [inputs], {}, ctx)

    assert isinstance(block.proj, SVDQuantLinear)
    assert torch.isfinite(block(inputs)).all()
    transform = composer.preprocessors[0]
    assert not transform._smooth_calibration
    assert block_name not in transform._block_groups


def test_flux_targets_resolve_when_model_is_attached_after_prepare_run():
    model = torch.nn.Module()
    model.config = {"_class_name": "FluxTransformer2DModel"}
    model.blocks = torch.nn.ModuleList([FluxTransformerBlock()])
    for name, module in model.named_modules():
        module.global_name = name
        if isinstance(module, torch.nn.Linear):
            for key, value in {
                "bits": 4,
                "group_size": 32,
                "sym": True,
                "data_type": "mx_fp4e2m1",
                "act_bits": 4,
                "act_group_size": 32,
                "act_sym": True,
                "act_data_type": "mx_fp4e2m1",
                "act_dynamic": True,
            }.items():
                setattr(module, key, value)

    model_context = SimpleNamespace(model=None)
    transform = AlgorithmComposer([SVDQuantConfig(rank=2), RTNConfig(disable_opt_rtn=True)]).preprocessors[0]
    transform.bind(
        SimpleNamespace(
            model_context=model_context,
            compress_context=None,
            calibration_context=None,
            scheme_context=PRESET_SCHEMES["MXFP4"],
            scale_dtype=None,
            nblocks=1,
            quant_block_list=[["blocks.0"]],
        )
    )
    transform.prepare_run()
    model_context.model = model
    ctx = BlockContext(model=model, block_names=["blocks.0"], block_name="blocks.0", block_index=0)

    transform.pre_quantize_block(ctx)

    assert isinstance(model.blocks[0].attn.to_q, SVDQuantLinear)
    assert isinstance(model.blocks[0].norm1.linear, torch.nn.Linear)


def test_sdxl_targets_resolve_when_model_is_attached_after_prepare_run():
    model = torch.nn.Module()
    model.config = {
        "_class_name": "UNet2DConditionModel",
        "addition_embed_type": "text_time",
        "cross_attention_dim": 2048,
        "projection_class_embeddings_input_dim": 2816,
    }
    model.blocks = torch.nn.ModuleList([BasicTransformerBlock()])
    for name, module in model.named_modules():
        module.global_name = name
        if isinstance(module, torch.nn.Linear):
            for key, value in {
                "bits": 4,
                "group_size": 32,
                "sym": True,
                "data_type": "mx_fp4e2m1",
                "act_bits": 4,
                "act_group_size": 32,
                "act_sym": True,
                "act_data_type": "mx_fp4e2m1",
                "act_dynamic": True,
            }.items():
                setattr(module, key, value)

    model_context = SimpleNamespace(model=None)
    transform = AlgorithmComposer([SVDQuantConfig(rank=2), RTNConfig(disable_opt_rtn=True)]).preprocessors[0]
    transform.bind(
        SimpleNamespace(
            model_context=model_context,
            compress_context=None,
            calibration_context=None,
            scheme_context=PRESET_SCHEMES["MXFP4"],
            scale_dtype=None,
            nblocks=1,
            quant_block_list=[["blocks.0"]],
        )
    )
    transform.prepare_run()
    model_context.model = model
    ctx = BlockContext(model=model, block_names=["blocks.0"], block_name="blocks.0", block_index=0)

    transform.pre_quantize_block(ctx)

    assert isinstance(model.blocks[0].attn1.to_q, SVDQuantLinear)
    assert isinstance(model.blocks[0].attn2.to_k, torch.nn.Linear)
