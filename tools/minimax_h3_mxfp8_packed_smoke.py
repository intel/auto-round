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
"""Smoke-test MXFP8 packed weights through AutoRound's quantized linear module."""

from __future__ import annotations

import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from safetensors import safe_open

# Register the local torchvision compatibility op before importing Diffusers.
try:
    _TORCHVISION_LIB = torch.library.Library("torchvision", "DEF")
    _TORCHVISION_LIB.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
except RuntimeError:
    pass

import minimax_h3_diffusers_smoke as smoke
from diffusers import ComponentsManager, ModularPipeline

from auto_round.experimental.qmodules import MXFP4QuantLinear, MXFP8QuantLinear
from auto_round.schemes import preset_name_to_scheme

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(os.environ.get("MINIMAX_H3_MODEL", ROOT / "artifacts/minimax_h3_real_pretrained"))
SCHEME = os.environ.get("MINIMAX_H3_MXFP_SCHEME", "MXFP8").upper()
if SCHEME not in {"MXFP4", "MXFP8"}:
    raise ValueError(f"Unsupported MXFP smoke scheme: {SCHEME}")
PACKED = Path(
    os.environ.get(
        "MINIMAX_H3_MXFP_PACKED",
        ROOT / f"artifacts/minimax_h3_{SCHEME.lower()}_packed_pretrained",
    )
)
OUTPUT = Path(
    os.environ.get("MINIMAX_H3_MXFP_SMOKE_OUTPUT", ROOT / f"artifacts/minimax_h3_{SCHEME.lower()}_packed_smoke")
)
STEPS = int(os.environ.get("MINIMAX_H3_SMOKE_STEPS", "50"))
QUANT_CLASS = MXFP4QuantLinear if SCHEME == "MXFP4" else MXFP8QuantLinear
WEIGHT_SUFFIX = ".weight_packed" if SCHEME == "MXFP4" else ".weight"


def _get_module(root: torch.nn.Module, name: str) -> torch.nn.Module:
    module = root
    for part in name.split("."):
        module = getattr(module, part)
    return module


def _replace_linears(transformer: torch.nn.Module, names: list[str]) -> dict[str, torch.nn.Module]:
    config = preset_name_to_scheme("MXFP8")
    replaced = {}
    for name in names:
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = _get_module(transformer, parent_name)
        else:
            parent = transformer
            child_name = name
        original = getattr(parent, child_name)
        if not isinstance(original, torch.nn.Linear):
            raise TypeError(f"Expected Linear at transformer.{name}, got {type(original).__name__}")
        quantized = QUANT_CLASS.from_original(config, original)
        setattr(parent, child_name, quantized)
        replaced[name] = quantized
    return replaced


def _load_packed_weights(transformer: torch.nn.Module, modules: dict[str, torch.nn.Module]) -> None:
    index_path = PACKED / "transformer/model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    shard_names = sorted(set(index["weight_map"].values()))
    for shard_name in shard_names:
        shard_path = PACKED / "transformer" / shard_name
        with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
            for key in shard.keys():
                if not key.endswith(".weight_scale"):
                    continue
                layer_name = key[: -len(".weight_scale")]
                if layer_name not in modules:
                    raise KeyError(f"Packed MXFP8 layer is not present in the transformer: {layer_name}")
                module = modules[layer_name]
                packed_weight = shard.get_tensor(f"{layer_name}{WEIGHT_SUFFIX}")
                if SCHEME == "MXFP4":
                    module.weight_packed.copy_(packed_weight)
                else:
                    module.weight.copy_(packed_weight)
                module.weight_scale.copy_(shard.get_tensor(key).view(torch.uint8))
        print(f"Loaded packed {SCHEME} shard {shard_name}", flush=True)


def _save_video(video: np.ndarray) -> str:
    frames = video[0] if video.ndim == 5 else video
    if frames.shape[1] in (1, 3, 4):
        frames = np.moveaxis(frames, 1, -1)
    frames = (np.clip(frames[..., :3], 0.0, 1.0) * 255).round().astype(np.uint8)
    path = OUTPUT / f"minimax_h3_{SCHEME.lower()}_50steps.mp4"
    imageio.mimwrite(str(path), frames, fps=24, codec="libx264", quality=8, macro_block_size=None)
    return str(path)


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "7":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=7 so the job uses the required physical GPU.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "GPU 7 is not available under CUDA_VISIBLE_DEVICES=7: "
            f"is_available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}"
        )
    index_path = PACKED / "transformer/model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Expected packed MXFP8 index: {index_path}")

    components_manager = ComponentsManager()
    pipe = ModularPipeline.from_pretrained(
        str(MODEL),
        workflow="t2va",
        local_files_only=True,
        components_manager=components_manager,
    )
    pipe.load_components(local_files_only=True, dtype=torch.bfloat16)
    index = json.loads(index_path.read_text())
    layer_names = sorted(key[: -len(".weight_scale")] for key in index["weight_map"] if key.endswith(".weight_scale"))
    modules = _replace_linears(pipe.transformer, layer_names)
    _load_packed_weights(pipe.transformer, modules)
    print(f"Installed {len(modules)} packed {SCHEME} linear modules", flush=True)
    components_manager.enable_auto_cpu_offload(device="cuda:0", memory_reserve_margin="8GB")

    smoke.OUTPUT = OUTPUT
    smoke.PROMPT_TSV = ROOT / "tools/minimax_h3_fp8_prompt.tsv"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(0)
    result = pipe(
        prompt=smoke._load_prompt(),
        height=384,
        width=672,
        num_frames=124,
        num_inference_steps=STEPS,
        generator=generator,
        output_type="pt",
    )
    video = smoke._to_numpy(result.get("videos"))
    audio = smoke._to_numpy(result.get("audios", result.get("audio")))
    if video is None or not torch.isfinite(torch.from_numpy(video)).all():
        raise RuntimeError("MXFP8 packed smoke produced missing or non-finite video output")
    frame_paths = smoke._save_video_frames(video)
    if audio is not None:
        if not torch.isfinite(torch.from_numpy(audio)).all():
            raise RuntimeError("MXFP8 packed smoke produced non-finite audio output")
        np.save(OUTPUT / "audio.npy", audio)
    video_path = _save_video(video)
    metadata = {
        "model": str(MODEL),
        "packed_model": str(PACKED),
        "scheme": SCHEME,
        "steps": STEPS,
        "height": 384,
        "width": 672,
        "num_frames": 124,
        "quantized_linear_modules": len(modules),
        "video_shape": list(video.shape),
        "video_min": float(video.min()),
        "video_max": float(video.max()),
        "video_mean": float(video.mean()),
        "audio_shape": None if audio is None else list(audio.shape),
        "video_path": video_path,
        "frame_paths": frame_paths,
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
