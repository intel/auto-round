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
import os

import pytest
import torch
from safetensors import safe_open

from auto_round import AutoRound
from auto_round.data_type.nvfp import FLOAT4_E2M1_MAX, FLOAT8_UE5M3_MAX

PREFIX = "model.decoder.layers.0.self_attn"


def _read_tensors(model_dir, names):
    tensors = {}
    for file_name in os.listdir(model_dir):
        if not file_name.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(model_dir, file_name), framework="pt") as f:
            for name in names:
                if name in f.keys():
                    tensors[name] = f.get_tensor(name)
    return tensors


def _quantize(model_path, dataloader, output_dir, fmt, iters=0):
    autoround = AutoRound(
        model_path,
        scheme="NVFP4_E5M3_GS",
        iters=iters,
        seqlen=8,
        nsamples=2,
        dataset=dataloader,
    )
    _, quantized_model_path = autoround.quantize_and_save(output_dir, format=fmt)
    return quantized_model_path


@pytest.mark.parametrize("iters", [0, 1])
def test_nvfp4_e5m3_gs_auto_round_export_and_inference(tiny_opt_model_path, dataloader, tmp_path, iters):
    from transformers import AutoModelForCausalLM

    from auto_round.experimental import qmodules as ar_qmodules

    model_dir = _quantize(tiny_opt_model_path, dataloader, str(tmp_path / "out"), "auto_round", iters=iters)

    with open(os.path.join(model_dir, "config.json")) as f:
        quantization_config = json.load(f)["quantization_config"]
    assert quantization_config["packing_format"] == "auto_round:llm_compressor_nvfp4_e5m3_gs"
    assert quantization_config["data_type"] == "nvfp4_v2_gs"
    assert quantization_config["act_data_type"] == "nvfp4_v2_with_static_gs"

    names = [
        f"{PREFIX}.{proj}.{suffix}"
        for proj in ("q_proj", "k_proj", "v_proj")
        for suffix in ("weight_scale", "weight_global_scale", "input_global_scale")
    ]
    tensors = _read_tensors(model_dir, names)
    assert set(tensors) == set(names)
    for proj in ("q_proj", "k_proj", "v_proj"):
        assert tensors[f"{PREFIX}.{proj}.weight_scale"].dtype == torch.uint8
        assert tensors[f"{PREFIX}.{proj}.weight_global_scale"].dtype == torch.float32
        assert tensors[f"{PREFIX}.{proj}.input_global_scale"].item() != 1.0
    # q/k/v share one weight and one input global scale for fused kernels.
    for suffix in ("weight_global_scale", "input_global_scale"):
        assert torch.equal(tensors[f"{PREFIX}.q_proj.{suffix}"], tensors[f"{PREFIX}.k_proj.{suffix}"])
        assert torch.equal(tensors[f"{PREFIX}.q_proj.{suffix}"], tensors[f"{PREFIX}.v_proj.{suffix}"])

    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cpu")
    layer = model.model.decoder.layers[0].self_attn.q_proj
    assert isinstance(layer, ar_qmodules.NVFP4E5M3GSQuantLinear)
    assert torch.equal(layer.weight_global_scale.cpu(), tensors[f"{PREFIX}.q_proj.weight_global_scale"])
    with torch.no_grad():
        logits = model(torch.tensor([[2, 100, 200, 300]])).logits
    assert torch.isfinite(logits).all()


def test_nvfp4_e5m3_gs_llm_compressor_export(tiny_opt_model_path, dataloader, tmp_path):
    model_dir = _quantize(tiny_opt_model_path, dataloader, str(tmp_path / "out"), "llm_compressor")

    with open(os.path.join(model_dir, "config.json")) as f:
        quantization_config = json.load(f)["quantization_config"]
    assert quantization_config["format"] == "nvfp4-e5m3-gs-pack-quantized"
    assert quantization_config["quant_method"] == "compressed-tensors"

    names = [f"{PREFIX}.q_proj.weight_global_scale", f"{PREFIX}.q_proj.input_global_scale"]
    tensors = _read_tensors(model_dir, names)
    assert set(tensors) == set(names)


def test_nvfp4_e5m3_gs_weight_global_scale_uses_ue5m3_range(tiny_opt_model_path, dataloader):
    from transformers import AutoModelForCausalLM

    orig_weight = AutoModelForCausalLM.from_pretrained(tiny_opt_model_path).model.decoder.layers[0].fc2.weight
    autoround = AutoRound(
        tiny_opt_model_path, scheme="NVFP4_E5M3_GS", iters=0, seqlen=8, nsamples=2, dataset=dataloader
    )
    model, _ = autoround.quantize()
    layer = model.model.decoder.layers[0].fc2
    layer = getattr(layer, "orig_layer", layer)
    expected = FLOAT8_UE5M3_MAX * FLOAT4_E2M1_MAX / orig_weight.float().abs().max()
    torch.testing.assert_close(layer.weight_global_scale.float().reshape(()), expected, rtol=1e-6, atol=0)
