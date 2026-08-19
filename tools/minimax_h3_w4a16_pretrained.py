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
"""Export MiniMax-H3 weights to packed W4A16 with a native Omni FL2VA partition."""

from __future__ import annotations

import os
from pathlib import Path

import torch

try:
    _TORCHVISION_LIB = torch.library.Library("torchvision", "DEF")
    _TORCHVISION_LIB.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
except RuntimeError:
    pass

from minimax_h3_native_vae_layout import repack_native_vae

from auto_round import AutoRound

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(os.environ.get("MINIMAX_H3_MODEL", ROOT / "artifacts/minimax_h3_real_pretrained"))
PROMPT_TSV = Path(os.environ.get("MINIMAX_H3_PROMPT_TSV", ROOT / "tools/minimax_h3_fp8_prompt.tsv"))
OUTPUT = Path(os.environ.get("MINIMAX_H3_W4A16_OUTPUT", ROOT / "artifacts/minimax_h3_w4a16_packed_pretrained"))
OFFICIAL_FL2VA = os.environ.get("MINIMAX_H3_OFFICIAL_FL2VA")


def main() -> None:
    requested_gpu = os.environ.get("MINIMAX_H3_GPU", "7")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != requested_gpu:
        raise RuntimeError(f"Set CUDA_VISIBLE_DEVICES={requested_gpu} for this job.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"GPU {requested_gpu} is not available under CUDA_VISIBLE_DEVICES={requested_gpu}: "
            f"is_available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}"
        )
    if not (MODEL / "modular_model_index.json").is_file():
        raise FileNotFoundError(f"Expected converted pretrained Modular Diffusers repo: {MODEL}")
    if not PROMPT_TSV.is_file():
        raise FileNotFoundError(f"Expected calibration prompt TSV: {PROMPT_TSV}")
    if not OFFICIAL_FL2VA or not Path(OFFICIAL_FL2VA).is_dir():
        raise FileNotFoundError(
            "Set MINIMAX_H3_OFFICIAL_FL2VA to the original release FL2VA directory so the exported "
            "checkpoint keeps Omni's native VAE contract."
        )

    autoround = AutoRound(
        str(MODEL),
        scheme="W4A16",
        dataset=str(PROMPT_TSV),
        nsamples=1,
        batch_size=1,
        iters=0,
        device_map=0,
        model_dtype="bf16",
        num_inference_steps=50,
        low_gpu_mem_usage=True,
        enable_torch_compile=False,
        disable_opt_rtn=True,
    )
    autoround.quantize_and_save(str(OUTPUT), format="auto_round:auto_gptq", inplace=True)
    partition = repack_native_vae(OUTPUT, OFFICIAL_FL2VA)
    print(f"Wrote packed W4A16 checkpoint to {OUTPUT}; Omni partition: {partition}")


if __name__ == "__main__":
    main()
