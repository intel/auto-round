#!/usr/bin/env python3
"""Quantize Wan2.2-TI2V-5B with AutoRound.

This script quantizes the TI2V diffusion transformer from the diffusers-format
Wan2.2 checkpoint layout and calibrates the transformer with text prompts.

Note:
    This diffusion path supports both 'fake' and 'auto_round' exports.
"""

import argparse

from auto_round import AutoRound


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize Wan2.2-TI2V-5B with AutoRound.")
    parser.add_argument(
        "--model",
        default="/mnt/disk0/lvl/Wan2.2-TI2V-5B-Diffusers",
        help="Path to the diffusers-format Wan2.2-TI2V-5B checkpoint directory.",
    )
    parser.add_argument(
        "--output",
        default="tmp_wan22_ti2v_w4a16",
        help="Output directory for the quantized model.",
    )
    parser.add_argument(
        "--format",
        default="fake",
        help="Export format. Supported values: 'fake' and 'auto_round'.",
    )
    parser.add_argument("--scheme", default="W4A16", help="Quantization scheme.")
    parser.add_argument("--iters", type=int, default=200, help="Number of optimization iterations.")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate.")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration samples.")
    parser.add_argument("--batch-size", type=int, default=1, help="Calibration batch size.")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=10,
        help="Number of denoising steps used during calibration.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale used during calibration.",
    )
    parser.add_argument(
        "--generator-seed",
        type=int,
        default=None,
        help="Seed for the initial noise generator.",
    )
    parser.add_argument(
        "--dataset",
        default="coco2014",
        help="Calibration dataset. Use 'coco2014' or a local .tsv caption file.",
    )
    parser.add_argument(
        "--device-map",
        default=None,
        help="Device map for quantization, e.g. '0' or '0,1'. Defaults to GPU 0.",
    )
    parser.add_argument(
        "--low-gpu-mem-usage",
        dest="low_gpu_mem_usage",
        action="store_true",
        default=True,
        help="Cache calibration inputs on CPU to reduce GPU memory usage. Enabled by default for diffusion.",
    )
    parser.add_argument(
        "--no-low-gpu-mem-usage",
        dest="low_gpu_mem_usage",
        action="store_false",
        help="Cache calibration inputs on GPU instead of CPU.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.format not in {"fake", "auto_round"}:
        raise ValueError("Wan2.2-TI2V quantization currently supports format='fake' or 'auto_round'.")

    device_map = args.device_map if args.device_map is not None else 0
    ar = AutoRound(
        args.model,
        scheme=args.scheme,
        lr=args.lr,
        iters=args.iters,
        nsamples=args.nsamples,
        batch_size=args.batch_size,
        dataset=args.dataset,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator_seed=args.generator_seed,
        low_gpu_mem_usage=args.low_gpu_mem_usage,
        device_map=device_map,
    )
    ar.quantize_and_save(args.output, format=args.format, inplace=True)
    print(f"Quantized model saved to {args.output}")


if __name__ == "__main__":
    main()