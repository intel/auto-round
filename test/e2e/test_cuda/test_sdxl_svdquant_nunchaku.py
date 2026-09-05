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
"""End-to-end SDXL Base 1.0 SVDQuant MXFP4 export and Nunchaku inference test."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from safetensors import safe_open

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="SDXL Nunchaku inference requires CUDA"),
]


def test_sdxl_base_svdquant_mxfp4_loads_and_generates(tmp_path):
    """Quantize the real SDXL Base 1.0 UNet, load it with Nunchaku, and generate one image."""
    pytest.importorskip("diffusers")
    pytest.importorskip("nunchaku")
    from test.helpers import get_model_path

    from diffusers import StableDiffusionXLPipeline
    from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel

    from auto_round import AutoRound
    from auto_round.algorithms.quantization.rtn.config import RTNConfig
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig

    model_path = get_model_path("stabilityai/stable-diffusion-xl-base-1.0")
    output_dir = tmp_path / "sdxl-base-svdquant-mxfp4"
    prompt = "A cinematic photograph of a red panda in a bamboo forest"
    seed = 12345
    num_inference_steps = 20
    guidance_scale = 5.0
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda:0")
    reference_image = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator(device="cuda:0").manual_seed(seed),
        height=512,
        width=512,
    ).images[0]
    reference_pixels = np.asarray(reference_image, dtype=np.float32) / 255.0
    compressor = AutoRound(
        model=pipe,
        tokenizer=None,
        scheme="MXFP4",
        alg_configs=[
            SVDQuantConfig(rank=32, smooth_enabled=False, model_adapter="sdxl"),
            RTNConfig(disable_opt_rtn=True),
        ],
        device_map=0,
        low_cpu_mem_usage=True,
        num_inference_steps=1,
    )
    compressor.quantize_and_save(output_dir, format="svdquant_nunchaku")
    del compressor, pipe

    weight_path = output_dir / "unet" / "diffusion_pytorch_model.safetensors"
    assert weight_path.is_file()
    with safe_open(weight_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    quantization_config = json.loads(metadata["quantization_config"])
    assert metadata["model_class"] == "NunchakuSDXLUNet2DConditionModel"
    assert quantization_config["rank"] == 32
    assert quantization_config["weight"]["dtype"] == "fp4_e2m1_all"
    assert quantization_config["weight"]["group_size"] == 32
    assert quantization_config["activation"]["dtype"] == "fp4_e2m1_all"
    assert quantization_config["activation"]["group_size"] == 32

    unet = NunchakuSDXLUNet2DConditionModel.from_pretrained(
        weight_path,
        torch_dtype=torch.bfloat16,
        device="cuda:0",
    )
    quantized_pipe = StableDiffusionXLPipeline.from_pretrained(
        output_dir,
        unet=unet,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda:0")
    try:
        image = quantized_pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cuda:0").manual_seed(seed),
            height=512,
            width=512,
        ).images[0]
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        assert pixels.shape == (512, 512, 3)
        assert np.isfinite(pixels).all()
        mean_absolute_error = np.abs(reference_pixels - pixels).mean()
        reference_centered = reference_pixels - reference_pixels.mean()
        pixels_centered = pixels - pixels.mean()
        centered_cosine = np.sum(reference_centered * pixels_centered) / np.sqrt(
            np.sum(reference_centered**2) * np.sum(pixels_centered**2)
        )
        assert mean_absolute_error < 0.15
        assert centered_cosine > 0.70
    finally:
        del quantized_pipe, unet
        torch.cuda.empty_cache()
