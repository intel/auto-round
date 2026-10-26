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

"""Coverage for the Omni-specific tensor layout, independent of model architecture."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from auto_round.algorithms.composer import AlgorithmComposer, BlockContext
from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.algorithms.transforms.svdquant import SVDQuantConfig
from auto_round.algorithms.transforms.svdquant.wrapper import SVDQuantLinear
from auto_round.data_type.nvfp import nv_fp4
from auto_round.data_type.utils import update_block_global_scale_if_needed
from auto_round.export.svdquant_nunchaku import IdentitySVDQuantModelAdapter
from auto_round.export.svdquant_omni import save_svdquant_omni
from auto_round.formats import get_formats
from auto_round.schemes import PRESET_SCHEMES
from auto_round.utils.device_manager import device_manager
from auto_round.wrapper import WrapperWALayer


def test_omni_tensor_layout_and_adapter_mapping(tmp_path):
    model = torch.nn.Module()
    model.config = {"_name_or_path": tmp_path}
    residual = torch.nn.Linear(32, 16, dtype=torch.bfloat16)
    for name, value in PRESET_SCHEMES["NVFP4"].to_dict().items():
        setattr(residual, name, value)
    down = torch.nn.Linear(32, 8, bias=False, dtype=torch.bfloat16)
    up = torch.nn.Linear(8, 16, bias=False, dtype=torch.bfloat16)
    model.projection = SVDQuantLinear(residual, down, up, torch.full((32,), 2, dtype=torch.bfloat16))
    model.unquantized = torch.nn.Linear(32, 16)

    class RenameAdapter(IdentitySVDQuantModelAdapter):
        def map_modules(self, model, records):
            return [replace(record, prefix="runtime.projection") for record in super().map_modules(model, records)]

    for adapter, prefix in ((None, "projection"), (RenameAdapter(), "runtime.projection")):
        if adapter is not None:
            # Also exercise export of tuned residuals through the same ABI.
            residual.data_type, residual.bits, residual.group_size = "nv_fp4", 4, 16
            residual.weight_global_scale = torch.tensor([4.0])
            quantized, residual.scale, _ = nv_fp4(residual.weight.float(), global_scale=residual.weight_global_scale)
            residual.weight.data = quantized.bfloat16()
        output = tmp_path / prefix
        output_format = get_formats("svdquant_omni", SimpleNamespace(scheme="NVFP4"))[0]
        assert output_format.save_quantized(output, model=model, adapter=adapter) is model
        tensors = load_file(output / "diffusion_pytorch_model.safetensors")
        assert set(tensors) == {
            *(
                f"{prefix}.{suffix}"
                for suffix in (
                    "qweight",
                    "wscales",
                    "wtscale",
                    "wcscales",
                    "proj_down",
                    "proj_up",
                    "smooth_factor",
                    "bias",
                )
            ),
            "unquantized.weight",
            "unquantized.bias",
        }
        assert tensors[f"{prefix}.qweight"].shape == (16, 16)
        assert tensors[f"{prefix}.qweight"].dtype == torch.int8
        assert tensors[f"{prefix}.wscales"].shape == (2, 16)
        assert tensors[f"{prefix}.wscales"].dtype == torch.float8_e4m3fn
        assert tensors[f"{prefix}.wtscale"].shape == (1,)
        assert tensors[f"{prefix}.wtscale"].dtype == torch.bfloat16
        if adapter is not None:
            torch.testing.assert_close(tensors[f"{prefix}.wtscale"], torch.tensor([0.25], dtype=torch.bfloat16))
            torch.testing.assert_close(tensors[f"{prefix}.wscales"].float(), residual.scale.reshape(16, 2).T)
        torch.testing.assert_close(tensors[f"{prefix}.wcscales"], torch.ones(16, dtype=torch.bfloat16))
        torch.testing.assert_close(tensors[f"{prefix}.proj_down"], (down.weight.float() * 2).T.bfloat16())
        torch.testing.assert_close(tensors[f"{prefix}.proj_up"], up.weight)
        torch.testing.assert_close(tensors[f"{prefix}.smooth_factor"], torch.full((32,), 0.5, dtype=torch.bfloat16))
        torch.testing.assert_close(tensors[f"{prefix}.bias"], residual.bias)
        torch.testing.assert_close(tensors["unquantized.weight"], model.unquantized.weight)
        config = json.loads((output / "config.json").read_text())
        quantization = json.loads((output / "quantization_config.json").read_text())
        assert config["quantization_config"] == quantization
        assert quantization == {
            "quant_method": "svdquant",
            "precision": "nvfp4",
            "rank": 8,
            "act_unsigned": False,
            "modules_to_not_convert": ["unquantized"],
        }


def test_omni_format_rejects_incompatible_schemes(tmp_path, monkeypatch):
    output_format = get_formats("svdquant_omni", SimpleNamespace(scheme="NVFP4"))[0]
    assert output_format.output_format == "svdquant_omni"
    assert not output_format.is_supported_immediate_packing()
    assert not output_format.is_supported_immediate_saving()
    for preset in ("MXFP4", "W4A16", "NVFP4_E5M3"):
        with pytest.raises(ValueError, match="NVFP4 W4A4 group16"):
            get_formats("svdquant_omni", SimpleNamespace(scheme=preset))
    for field, value in (("bits", 8), ("group_size", 32), ("act_bits", 16), ("act_dynamic", False)):
        scheme = PRESET_SCHEMES["NVFP4"].copy()
        setattr(scheme, field, value)
        with pytest.raises(ValueError, match=field):
            output_format.check_scheme_args(scheme)
    model = torch.nn.Module()
    residual = torch.nn.Linear(32, 16)
    for name, value in PRESET_SCHEMES["MXFP4"].to_dict().items():
        setattr(residual, name, value)
    model.projection = SVDQuantLinear(
        residual, torch.nn.Linear(32, 8, bias=False), torch.nn.Linear(8, 16, bias=False), torch.ones(32)
    )
    with pytest.raises(ValueError, match="NVFP4 W4A4 group16"):
        save_svdquant_omni(model, tmp_path / "invalid")
    with pytest.raises(ValueError, match="incompatible residual scheme"):
        output_format._validate_svd_layer_overrides(model, {"projection": "MXFP4"})

    from auto_round import AutoRound
    from auto_round.algorithms.quantization.rtn.config import RTNConfig
    from auto_round.algorithms.transforms.svdquant import SVDQuantConfig

    created = {}

    class FakeCompressor:
        def __init__(self, config, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr("auto_round.autoround._build_model_type_ctor_kwargs", lambda *args, **kwargs: ("llm", {}))
    monkeypatch.setattr("auto_round.autoround._get_compressor_class", lambda model_type, base_cls: FakeCompressor)
    for format in ("svdquant_omni", "svdquant_omni,fake"):
        AutoRound(
            "dummy-model",
            scheme="NVFP4",
            format=format,
            alg_configs=[SVDQuantConfig(smooth_enabled=False), RTNConfig(disable_opt_rtn=True)],
        )
        assert created["format"] == format


def test_omni_zero_shot_rtn_fuses_qkv_with_shared_scale(tmp_path, monkeypatch):
    monkeypatch.setattr(device_manager, "_device_map", "cpu")
    monkeypatch.setattr(device_manager, "_device_list", ["cpu"])
    monkeypatch.setattr(device_manager, "_major_device", "cpu")

    class SelfAttention(torch.nn.Module):
        is_cross_attention = False

        def __init__(self):
            super().__init__()
            self.to_q = torch.nn.Linear(32, 32, bias=False)
            self.to_k = torch.nn.Linear(32, 32, bias=False)
            self.to_v = torch.nn.Linear(32, 32, bias=False)

    class WanBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = SelfAttention()

    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([WanBlock()])
    model.config = {"_name_or_path": str(tmp_path)}
    scheme = PRESET_SCHEMES["NVFP4"].copy()
    for name, module in model.named_modules():
        module.global_name = name
        if isinstance(module, torch.nn.Linear):
            for key, value in scheme.to_dict().items():
                setattr(module, key, value)
            module.scale_dtype = torch.float32

    block_name = "blocks.0"
    orchestrator = SimpleNamespace(
        model_context=SimpleNamespace(
            model=model,
            amp=False,
            amp_dtype=torch.float32,
            is_diffusion=True,
            is_moe_model=False,
            output_config=None,
        ),
        compress_context=SimpleNamespace(
            enable_torch_compile=False,
            low_gpu_mem_usage=False,
            cache_device=torch.device("cpu"),
            clear_memory=lambda: None,
            formats=[SimpleNamespace(format_name="svdquant_omni")],
        ),
        calibration_context=SimpleNamespace(batch_size=1, batch_dim=0),
        scheme_context=scheme,
        scale_dtype=torch.float32,
        nblocks=1,
        quant_block_list=[[block_name]],
        data_type=scheme.data_type,
        batch_dim=0,
        batch_size=1,
        cache_device="cpu",
        amp=False,
        amp_dtype=torch.float32,
        shared_cache_keys=(),
    )
    composer = AlgorithmComposer(
        [
            SVDQuantConfig(rank=2, smooth_enabled=False, residual_iters=1, low_rank_dtype="fp32"),
            RTNConfig(disable_opt_rtn=True),
        ],
        orchestrator=orchestrator,
    )
    composer.prepare_run()

    block = model.blocks[0]
    # Match the Orchestrator's initial NVFP4 scale setup before SVD replacement.
    update_block_global_scale_if_needed(block, scheme.data_type, scheme.group_size)
    ctx = BlockContext(
        model=model,
        block_names=[block_name],
        block_name=block_name,
        block_index=0,
        block_cnt=1,
    )
    composer.compress_block(block, fp_inputs=None, input_others={}, block_ctx=ctx)

    residual_scales = []
    for name in ("to_q", "to_k", "to_v"):
        residual = getattr(block.attn, name).residual_linear
        while isinstance(residual, WrapperWALayer):
            residual = residual.orig_layer
        residual_scales.append(residual.weight_global_scale)
    torch.testing.assert_close(residual_scales[0], residual_scales[1])
    torch.testing.assert_close(residual_scales[0], residual_scales[2])

    output_dir = tmp_path / "svdquant_omni"
    save_svdquant_omni(model, output_dir)
    tensors = load_file(output_dir / "diffusion_pytorch_model.safetensors")
    assert "blocks.0.attn.to_qkv.wtscale" in tensors
