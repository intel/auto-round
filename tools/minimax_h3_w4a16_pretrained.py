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

import json
import os
from pathlib import Path

import torch

try:
    _TORCHVISION_LIB = torch.library.Library("torchvision", "DEF")
    _TORCHVISION_LIB.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
except RuntimeError:
    pass

from auto_round import AutoRound

from minimax_h3_native_vae_layout import repack_native_vae


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(os.environ.get("MINIMAX_H3_MODEL", ROOT / "artifacts/minimax_h3_real_pretrained"))
PROMPT_TSV = Path(os.environ.get("MINIMAX_H3_PROMPT_TSV", ROOT / "tools/minimax_h3_w4a16_prompt.tsv"))
OUTPUT = Path(
    os.environ.get("MINIMAX_H3_W4A16_OUTPUT", ROOT / "artifacts/minimax_h3_w4a16_packed_pretrained")
)
OFFICIAL_FL2VA = os.environ.get("MINIMAX_H3_OFFICIAL_FL2VA")


def _native_block_name(name: str) -> str:
    """Map Diffusers H3 block selectors to Omni's native module names."""
    replacements = (
        ("transformer_blocks", "blocks"),
        ("token_refiner.refiner_blocks", "token_refiner.blocks"),
    )
    for source, target in replacements:
        if name == source or name.startswith(source + "."):
            return target + name[len(source) :]
    return name


def _rewrite_omni_quant_metadata(output: Path) -> None:
    """Write runtime-native block selectors into the Omni transformer metadata.

    AutoRound quantizes the Diffusers module names, while the native Omni
    transformer is built with ``blocks`` and ``token_refiner.blocks``. The
    packed tensors remain Diffusers-named and are translated by Omni's
    checkpoint adapter; only the quantization selectors need native names.
    """
    transformer_dir = output / "transformer"
    config_paths = (transformer_dir / "quantization_config.json", transformer_dir / "config.json")
    for path in config_paths:
        if not path.is_file():
            continue
        config = json.loads(path.read_text())
        quant_config = config if "quant_method" in config else config.get("quantization_config")
        if not isinstance(quant_config, dict) or quant_config.get("quant_method") != "auto-round":
            continue
        names = quant_config.get("block_name_to_quantize")
        if isinstance(names, str):
            names = [item.strip() for item in names.split(",") if item.strip()]
        if isinstance(names, list):
            quant_config["block_name_to_quantize"] = [_native_block_name(item) for item in names]
        extra_config = quant_config.get("extra_config")
        if isinstance(extra_config, dict):
            quant_config["extra_config"] = {
                _native_block_name(key): value for key, value in extra_config.items()
            }
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def main() -> None:
    requested_gpu = os.environ.get("MINIMAX_H3_GPU", "3")
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
        disable_model_free=True,
    )
    autoround.quantize_and_save(str(OUTPUT), format="auto_round:auto_gptq", inplace=True)
    _rewrite_omni_quant_metadata(OUTPUT)
    partition = repack_native_vae(OUTPUT, OFFICIAL_FL2VA)
    print(f"Wrote packed W4A16 checkpoint to {OUTPUT}; Omni partition: {partition}")


if __name__ == "__main__":
    main()
