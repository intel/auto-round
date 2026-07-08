import os

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

from wan_sparse_patch import patch_wan_sparse_attention_from_env

dtype = torch.bfloat16
device = "xpu"


def env_flag(name, default="0"):
    return os.getenv(name, default).lower() not in {"0", "false", "no", "off"}


def env_value(name):
    value = os.getenv(name, "").strip()
    return value or None


model_id = os.getenv("WAN_MODEL", "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
device_map = env_value("WAN_DEVICE_MAP")
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe_kwargs = {"vae": vae, "torch_dtype": dtype}
if device_map is not None:
    pipe_kwargs["device_map"] = device_map
pipe = WanPipeline.from_pretrained(model_id, **pipe_kwargs)
cpu_offload_enabled = env_flag("WAN_ENABLE_CPU_OFFLOAD", "1")
cpu_offload_mode = os.getenv("WAN_CPU_OFFLOAD_MODE", "model").strip().lower()
if device_map is not None:
    if cpu_offload_enabled:
        raise ValueError("WAN_DEVICE_MAP cannot be combined with WAN_ENABLE_CPU_OFFLOAD=1")
    print(f"[wan_sparse] using device_map={device_map} hf_device_map={pipe.hf_device_map}")
elif cpu_offload_enabled:
    if cpu_offload_mode == "model":
        pipe.enable_model_cpu_offload(device=device)
    elif cpu_offload_mode == "sequential":
        pipe.enable_sequential_cpu_offload(device=device)
    else:
        raise ValueError("WAN_CPU_OFFLOAD_MODE must be one of: model, sequential")
else:
    pipe.to(device)
# Match the Wan2.2 T2V-A14B diffusers model-card example by default.
height = int(os.getenv("WAN_HEIGHT", "720"))
width = int(os.getenv("WAN_WIDTH", "1280"))
num_frames = int(os.getenv("WAN_NUM_FRAMES", "81"))
num_inference_steps = int(os.getenv("WAN_STEPS", "40"))
guidance_scale = float(os.getenv("WAN_GUIDANCE_SCALE", "4.0"))
guidance_scale_2 = float(os.getenv("WAN_GUIDANCE_SCALE_2", "3.0"))
use_sparse = os.getenv("WAN_USE_SPARSE", "0").lower() not in {"0", "false", "no", "off"}
if use_sparse:
    output_path_file = (
        f"wan_sparse_topk{os.getenv('WAN_SPARSE_TOPK', '0.75')}_"
        f"{num_frames}f_{num_inference_steps}steps_{guidance_scale}gs.mp4"
    )
else:
    output_path_file = f"wan_dense_{num_frames}f_{num_inference_steps}steps_{guidance_scale}gs.mp4"
output_path = os.getenv("WAN_OUTPUT", output_path_file)


prompt = os.getenv(
    "WAN_PROMPT",
    "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
)
negative_prompt = os.getenv(
    "WAN_NEGATIVE_PROMPT",
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
)
fps = int(os.getenv("WAN_FPS", "16"))

if use_sparse:
    print(
        f"[wan_sparse] enabled sparse patch: topk={os.getenv('WAN_SPARSE_TOPK', '0.75')} "
        f"smooth_k={os.getenv('WAN_SPARSE_SMOOTH_K', '1')}"
        f" cross={os.getenv('WAN_SPARSE_ENABLE_CROSS_ATTN', '0')}"
        f" q_tile={os.getenv('WAN_SPARSE_Q_TILE_OVERRIDE', '0')}"
        f" q_block={os.getenv('WAN_SPARSE_Q_BLOCK_TOKENS', 'default')}"
        f" k_block={os.getenv('WAN_SPARSE_K_BLOCK_TOKENS', 'default')}"
    )
    with patch_wan_sparse_attention_from_env(pipe.transformer) as sparse_stats:
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_2,
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
        guidance_scale_2=guidance_scale_2,
        num_inference_steps=num_inference_steps,
    ).frames[0]

export_to_video(output, output_path, fps=fps)
print(f"[wan_sparse] wrote {output_path}")
