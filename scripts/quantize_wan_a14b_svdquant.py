# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Calibrated Smooth + SignRound MXFP4 export of both Wan2.2 T2V A14B experts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

PROFILES = {
    "smoke": dict(nsamples=1, calib_steps=2, iters=1, residual_iters=1, smooth_grids=2, smooth_calls=1),
    "fast": dict(nsamples=2, calib_steps=2, iters=20, residual_iters=1, smooth_grids=6, smooth_calls=4),
    "quality": dict(nsamples=8, calib_steps=4, iters=200, residual_iters=3, smooth_grids=20, smooth_calls=16),
}
SMOKE_PROMPTS = [
    "An orange cat walks through a sunlit garden, flowers swaying in the breeze, detailed fur, smooth motion.",
    "A small sailboat crosses a calm blue lake, ripples spreading behind it, mountains in the distance.",
]
MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
MODEL_REVISION = "5be7df9619b54f4e2667b2755bc6a756675b5cd7"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, help="Official Hugging Face ID or local Diffusers directory")
    parser.add_argument("--revision", default=MODEL_REVISION, help="Revision used for Hub loading")
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory; existing paths are never overwritten"
    )
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    data = parser.add_mutually_exclusive_group()
    data.add_argument("--prompts-file", type=Path, help="UTF-8 text, one calibration prompt per nonempty line")
    data.add_argument("--dataset", help="AutoRound diffusion dataset name or local caption TSV")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=9)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0, help="Logical CUDA index after CUDA_VISIBLE_DEVICES")
    parser.add_argument("--low-gpu-mem-usage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale-2", type=float, default=3.0)
    for key in PROFILES["smoke"]:
        parser.add_argument("--" + key.replace("_", "-"), type=int)
    args = parser.parse_args(argv)
    for key, value in PROFILES[args.profile].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.height < 16 or args.width < 16 or args.height % 16 or args.width % 16:
        parser.error("height and width must be positive multiples of 16")
    if args.num_frames < 1 or (args.num_frames - 1) % 4:
        parser.error("num-frames must be 4k+1")
    if args.rank <= 0 or args.rank % 16:
        parser.error("rank must be a positive multiple of 16")
    if min(getattr(args, key) for key in PROFILES["smoke"]) < 1 or args.calib_steps < 2:
        parser.error("profile parameters must be positive; calib-steps must be at least 2")
    if args.profile == "quality" and not (args.prompts_file or args.dataset):
        parser.error("quality requires representative --prompts-file or --dataset")
    return args


def calibration_data(args):
    if args.dataset:
        return args.dataset
    prompts = (
        [line.strip() for line in args.prompts_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if args.prompts_file
        else SMOKE_PROMPTS
    )
    if len(prompts) < args.nsamples:
        raise ValueError(f"Need {args.nsamples} prompts, received {len(prompts)}")
    return [([index], [prompt]) for index, prompt in enumerate(prompts[: args.nsamples])]


def validate_source(pipe):
    if type(pipe).__name__ != "WanPipeline" or pipe.config.boundary_ratio != 0.875:
        raise ValueError("Expected the official Wan2.2-T2V-A14B WanPipeline with boundary_ratio=0.875")
    for name in ("transformer", "transformer_2"):
        model = getattr(pipe, name, None)
        config = getattr(model, "config", None)
        expected = dict(num_layers=40, num_attention_heads=40, attention_head_dim=128, in_channels=16, out_channels=16)
        if config is None or any(getattr(config, key, None) != value for key, value in expected.items()):
            raise ValueError(f"{name} is not a Wan2.2 T2V A14B expert")


def audit_export(output):
    import torch
    from safetensors import safe_open

    index = json.loads((output / "model_index.json").read_text())
    report = {}
    for name in ("transformer", "transformer_2"):
        if index[name] != ["nunchaku", "NunchakuWanTransformer3DModel"]:
            raise ValueError(f"Incorrect runtime class for {name}: {index[name]}")
        checkpoint = output / name / "diffusion_pytorch_model.safetensors"
        with safe_open(checkpoint, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            count = sum(key.endswith(".qweight") for key in keys)
            if count != 400:
                raise ValueError(f"{name}: expected 400 quantized projections, found {count}")
            for key in keys:
                tensor = handle.get_tensor(key)
                if tensor.is_floating_point() and not torch.isfinite(tensor.float()).all().item():
                    raise ValueError(f"Non-finite exported tensor: {name}/{key}")
            quantization = json.loads(handle.metadata()["quantization_config"])
        report[name] = dict(quantized_linears=count, bytes=checkpoint.stat().st_size, quantization=quantization)
    return report


def main(argv=None):
    args = parse_args(argv)
    dataset = calibration_data(args)
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")

    import diffusers
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline

    from auto_round import AutoRound
    from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
    from auto_round.algorithms.transforms.svdquant import SVDQuantConfig

    if not torch.cuda.is_available():
        raise RuntimeError("This runner requires a CUDA GPU for SVD and SignRound")
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    source = dict(revision=args.revision) if not Path(args.model).is_dir() else dict(local_files_only=True)
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(
        torch=torch.__version__, diffusers=diffusers.__version__, gpu=torch.cuda.get_device_name(args.device)
    )
    (args.output / "quantization-run.json").write_text(json.dumps(settings, indent=2) + "\n")
    vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32, **source)
    # DiffusionMixin respects declared FP32 modules when aligning a preloaded pipeline.
    # An empty substring covers the entire VAE, including buffers.
    vae._keep_in_fp32_modules = [""]
    pipe = WanPipeline.from_pretrained(args.model, vae=vae, torch_dtype=torch.bfloat16, **source)
    validate_source(pipe)
    svd = SVDQuantConfig(
        rank=args.rank,
        smooth_enabled=True,
        smooth_num_grids=args.smooth_grids,
        smooth_max_calibration_calls=args.smooth_calls,
        residual_iters=args.residual_iters,
        residual_early_stop=True,
        low_rank_dtype="bf16",
        model_adapter="wan",
    )
    terminal = SignRoundConfig(iters=args.iters, nblocks=1, enable_quanted_input=True)
    compressor = AutoRound(
        pipe,
        scheme="MXFP4",
        alg_configs=[svd, terminal],
        dataset=dataset,
        nsamples=args.nsamples,
        batch_size=1,
        low_gpu_mem_usage=args.low_gpu_mem_usage,
        low_cpu_mem_usage=False,
        device_map=args.device,
        model_dtype="bf16",
        calib_num_inference_steps=args.calib_steps,
        pipeline_call_kwargs=dict(
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            output_type="latent",
            guidance_scale_2=args.guidance_scale_2,
        ),
        guidance_scale=args.guidance_scale,
        generator_seed=args.seed,
        seed=args.seed,
        format="svdquant_nunchaku",
    )
    compressor.quantize()
    compressor.save_quantized(str(args.output), format="svdquant_nunchaku")
    report = audit_export(args.output)
    report["seconds"] = time.monotonic() - start
    report["peak_cuda_allocated_gib"] = torch.cuda.max_memory_allocated(args.device) / 2**30
    (args.output / "export-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
