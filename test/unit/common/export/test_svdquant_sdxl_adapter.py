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

import json

import pytest
import torch

from auto_round.export.svdquant_adapters import (
    SDXL_SVDQUANT_TARGET_MODULES,
    detect_svdquant_model_adapter,
)
from auto_round.export.svdquant_adapters.sdxl import SDXLSVDQuantNunchakuAdapter
from auto_round.export.svdquant_nunchaku import (
    SourceLinearRecord,
    SVDQuantExportConfig,
    SVDQuantLinearScheme,
    collect_svdquant_tensors,
    save_svdquant_nunchaku_safetensors,
)

SCHEME = SVDQuantLinearScheme("mx_fp4", 4, 32, True, "mx_fp4", 4, 32, True, True)


class ConfiguredModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


class BasicTransformerBlock(torch.nn.Module):
    pass


def _install(root, path, module):
    current = root
    parts = path.split(".")
    for part in parts[:-1]:
        if not hasattr(current, part):
            current.add_module(part, torch.nn.Module())
        current = getattr(current, part)
    current.add_module(parts[-1], module)


def _sdxl_config():
    return {
        "_class_name": "UNet2DConditionModel",
        "addition_embed_type": "text_time",
        "cross_attention_dim": 2048,
        "projection_class_embeddings_input_dim": 2816,
    }


def _source(name, out_features=8, in_features=8, rank=2, seed=0, bias=True):
    generator = torch.Generator().manual_seed(seed)
    return SourceLinearRecord(
        name=name,
        residual_weight=torch.randn(out_features, in_features, generator=generator),
        lora_down=torch.randn(rank, in_features, generator=generator),
        lora_up=torch.randn(out_features, rank, generator=generator),
        smooth=torch.linspace(0.5, 1.5, in_features),
        smooth_orig=torch.linspace(0.75, 1.75, in_features),
        bias=torch.randn(out_features, generator=generator) if bias else None,
        scheme=SCHEME,
    )


def _effective(source):
    return (source.residual_weight + source.lora_up @ source.lora_down) * source.smooth


def _wrapped_linear(in_features=32, out_features=8, rank=2):
    from auto_round.algorithms.transforms.svdquant.wrapper import SVDQuantLinear

    residual = torch.nn.Linear(in_features, out_features)
    residual.data_type, residual.bits, residual.group_size, residual.sym = "mx_fp4", 4, 32, True
    residual.act_data_type, residual.act_bits, residual.act_group_size = "mx_fp4", 4, 32
    residual.act_sym, residual.act_dynamic = True, True
    return SVDQuantLinear(
        residual,
        torch.nn.Linear(in_features, rank, bias=False),
        torch.nn.Linear(rank, out_features, bias=False),
        torch.linspace(0.5, 1.5, in_features),
    )


def test_detects_sdxl_unet_from_runtime_relevant_config():
    model = ConfiguredModel(_sdxl_config())

    assert detect_svdquant_model_adapter(model) == "sdxl"


def test_does_not_treat_stable_diffusion_v1_unet_as_sdxl():
    model = ConfiguredModel(
        {
            "_class_name": "UNet2DConditionModel",
            "cross_attention_dim": 768,
        }
    )

    assert detect_svdquant_model_adapter(model) == "identity"


def test_sdxl_target_allowlist_matches_nunchaku_patched_linears():
    assert set(SDXL_SVDQUANT_TARGET_MODULES) == {
        "attn1.to_q",
        "attn1.to_k",
        "attn1.to_v",
        "attn1.to_out.0",
        "attn2.to_q",
        "attn2.to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    }


def test_sdxl_maps_direct_linears_and_fuses_self_attention_qkv():
    prefix = "down_blocks.1.attentions.0.transformer_blocks.0"
    qkv = tuple(_source(f"{prefix}.attn1.to_{name}", seed=index + 1) for index, name in enumerate("qkv"))
    direct = _source(f"{prefix}.attn2.to_q", seed=9)
    adapter = SDXLSVDQuantNunchakuAdapter(require_complete_model=False)

    records = tuple(adapter.map_modules(ConfiguredModel(_sdxl_config()), (*qkv, direct)))

    assert [record.prefix for record in records] == [f"{prefix}.attn1.to_qkv", f"{prefix}.attn2.to_q"]
    fused = records[0]
    assert fused.sources == qkv
    assert fused.lora_down.shape[0] == 2
    expected = torch.cat([_effective(source) for source in qkv])
    actual = (fused.residual_weight + fused.lora_up @ fused.lora_down) * fused.smooth
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(fused.bias, torch.cat([source.bias for source in qkv]))
    assert records[1].sources == (direct,)


def test_sdxl_preserves_shared_qkv_decomposition_without_recomposing():
    prefix = "mid_block.attentions.0.transformer_blocks.0"
    shared_down = torch.randn(2, 8)
    shared_smooth = torch.linspace(0.5, 1.5, 8)
    shared_smooth_orig = torch.linspace(0.75, 1.75, 8)
    sources = tuple(_source(f"{prefix}.attn1.to_{name}", seed=index + 11) for index, name in enumerate("qkv"))
    sources = tuple(
        SourceLinearRecord(
            name=source.name,
            residual_weight=source.residual_weight,
            lora_down=shared_down.clone(),
            lora_up=source.lora_up,
            smooth=shared_smooth.clone(),
            smooth_orig=shared_smooth_orig.clone(),
            bias=source.bias,
            scheme=source.scheme,
        )
        for source in sources
    )

    (record,) = tuple(
        SDXLSVDQuantNunchakuAdapter(require_complete_model=False).map_modules(ConfiguredModel(_sdxl_config()), sources)
    )

    torch.testing.assert_close(record.residual_weight, torch.cat([source.residual_weight for source in sources]))
    torch.testing.assert_close(record.lora_down, shared_down)
    torch.testing.assert_close(record.lora_up, torch.cat([source.lora_up for source in sources]))
    torch.testing.assert_close(record.smooth, shared_smooth)
    torch.testing.assert_close(record.smooth_orig, shared_smooth_orig)


def test_sdxl_metadata_names_runtime_model_and_serializes_config():
    model = ConfiguredModel(_sdxl_config())

    metadata = SDXLSVDQuantNunchakuAdapter(require_complete_model=False).metadata(model, rank=32)

    assert metadata["model_class"] == "NunchakuSDXLUNet2DConditionModel"
    assert metadata["format"] == "pt"
    assert metadata["comfy_config"] == "{}"
    assert json.loads(metadata["config"]) == _sdxl_config()


def test_complete_sdxl_mapping_rejects_missing_runtime_projection():
    prefix = "down_blocks.1.attentions.0.transformer_blocks.0"
    sources = tuple(_source(f"{prefix}.attn1.to_{name}", seed=index + 1) for index, name in enumerate("qkv"))
    model = ConfiguredModel(_sdxl_config())
    _install(model, prefix, BasicTransformerBlock())

    with pytest.raises(ValueError, match="coverage.*missing"):
        tuple(SDXLSVDQuantNunchakuAdapter(require_complete_model=True).map_modules(model, sources))


def test_sdxl_export_preserves_float_unet_state_without_wrapper_internal_keys(tmp_path):
    from safetensors import safe_open

    prefix = "down_blocks.1.attentions.0.transformer_blocks.0"
    model = ConfiguredModel(_sdxl_config())
    _install(model, f"{prefix}.attn2.to_q", _wrapped_linear())
    _install(model, f"{prefix}.attn2.to_k", torch.nn.Linear(32, 8))
    _install(model, "conv_in", torch.nn.Conv2d(4, 8, kernel_size=3, padding=1))
    adapter = SDXLSVDQuantNunchakuAdapter(require_complete_model=False)
    config = SVDQuantExportConfig(runtime_loadable=True)

    tensors = collect_svdquant_tensors(model, adapter=adapter, config=config)
    path = tmp_path / "sdxl.safetensors"
    save_svdquant_nunchaku_safetensors(model, str(path), adapter=adapter, config=config)

    assert f"{prefix}.attn2.to_k.weight" in tensors
    assert f"{prefix}.attn2.to_k.bias" in tensors
    assert "conv_in.weight" in tensors
    assert "conv_in.bias" in tensors
    assert not any(".residual_linear." in key or ".lora_down." in key or ".lora_up." in key for key in tensors)
    assert all(tensor.device.type == "cpu" and tensor.is_contiguous() for tensor in tensors.values())
    with safe_open(path, framework="pt") as handle:
        assert set(handle.keys()) == set(tensors)
        assert handle.metadata()["model_class"] == "NunchakuSDXLUNet2DConditionModel"
        assert handle.get_tensor("conv_in.weight").dtype == torch.bfloat16
