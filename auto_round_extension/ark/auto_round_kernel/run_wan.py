import os

import torch
import numpy as np
from diffusers import WanPipeline, AutoencoderKLWan, WanTransformer3DModel, UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image

from wan_sparse_patch import patch_wan_sparse_attention_from_env

dtype = torch.bfloat16
device = "xpu"

model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
model_id = "/home/yiliu7/.cache/huggingface/Wan-AI/Wan2.2-TI2V-5B-Diffusers"
model_id = "/home/yiliu7/.cache/huggingface/Wan-AI/Wan2.1-T2V-1.3B/"
model_id = "/home/yiliu7/.cache/huggingface/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype)
pipe.to(device)
pipe.enable_model_cpu_offload()
height = int(os.getenv("WAN_HEIGHT", "480"))
width = int(os.getenv("WAN_WIDTH", "832"))
num_frames = int(os.getenv("WAN_NUM_FRAMES", "81"))
num_inference_steps = int(os.getenv("WAN_STEPS", "50"))
guidance_scale = float(os.getenv("WAN_GUIDANCE_SCALE", "5.0"))
use_sparse = os.getenv("WAN_USE_SPARSE", "0").lower() not in {"0", "false", "no", "off"}
output_path_file = f"wan_output_{num_frames}f_{num_inference_steps}steps_{guidance_scale}gs_sparse{os.getenv('WAN_SPARSE_TOPK', '0.75')}.mp4"
output_path = os.getenv("WAN_OUTPUT", output_path_file)


prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

if use_sparse:
    print(
        f"[wan_sparse] enabled sparse patch: topk={os.getenv('WAN_SPARSE_TOPK', '0.75')} "
        f"smooth_k={os.getenv('WAN_SPARSE_SMOOTH_K', '1')}"
        f" cross={os.getenv('WAN_SPARSE_ENABLE_CROSS_ATTN', '0')}"
    )
    with patch_wan_sparse_attention_from_env(pipe.transformer) as sparse_stats:
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ).frames[0]
    print(
        "[wan_sparse] stats:"
        f" self_attn_total={sparse_stats.self_attn_total}"
        f" cross_attn_total={sparse_stats.cross_attn_total}"
        f" sparse_self_attn_calls={sparse_stats.sparse_self_attn_calls}"
        f" sparse_cross_attn_calls={sparse_stats.sparse_cross_attn_calls}"
        f" cross_attn_fallbacks={sparse_stats.cross_attn_fallbacks}"
        f" cross_attn_policy_fallbacks={sparse_stats.cross_attn_policy_fallbacks}"
        f" cross_attn_unsupported_fallbacks={sparse_stats.cross_attn_unsupported_fallbacks}"
        f" cross_attn_runtime_fallbacks={sparse_stats.cross_attn_runtime_fallbacks}"
        f" unsupported_fallbacks={sparse_stats.unsupported_fallbacks}"
        f" runtime_fallbacks={sparse_stats.runtime_fallbacks}"
        f" avg_sparsity={sparse_stats.avg_sparsity:.4f}"
    )
else:
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).frames[0]

export_to_video(output, output_path, fps=24)
