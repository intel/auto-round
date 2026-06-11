import os

import torch
from diffusers import FluxPipeline

from flux_sparse_patch import patch_flux_sparse_attention_from_env


dtype = torch.bfloat16
device = "xpu"

model_id = "/home/yiliu7/.cache/huggingface/black-forest-labs/FLUX.1-dev"
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
pipe.to(device)
pipe.enable_model_cpu_offload()

height = int(os.getenv("FLUX_HEIGHT", "1024"))
width = int(os.getenv("FLUX_WIDTH", "1024"))
num_inference_steps = int(os.getenv("FLUX_STEPS", "50"))
guidance_scale = float(os.getenv("FLUX_GUIDANCE_SCALE", "3.5"))
max_sequence_length = int(os.getenv("FLUX_MAX_SEQUENCE_LENGTH", "512"))
seed = int(os.getenv("FLUX_SEED", "0"))
use_sparse = os.getenv("FLUX_USE_SPARSE", "0").lower() not in {"0", "false", "no", "off"}

output_file = (
    f"flux_output_{height}x{width}_{num_inference_steps}steps_"
    f"{guidance_scale}gs_sparse{os.getenv('FLUX_SPARSE_TOPK', '0.75')}.png"
)
output_path = os.getenv("FLUX_OUTPUT", output_file)

prompt = os.getenv("FLUX_PROMPT", "A cat holding a sign that says hello world")

if use_sparse:
    print(
        f"[flux_sparse] enabled attention sparse patch: topk={os.getenv('FLUX_SPARSE_TOPK', '0.75')} "
        f"smooth_k={os.getenv('FLUX_SPARSE_SMOOTH_K', '1')}"
    )
    with patch_flux_sparse_attention_from_env(pipe.transformer) as sparse_stats:
        image = pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]
    print(
        "[flux_sparse] stats:"
        f" total_calls={sparse_stats.total_calls}"
        f" single_stream_calls={sparse_stats.single_stream_calls}"
        f" joint_stream_calls={sparse_stats.joint_stream_calls}"
        f" sparse_calls={sparse_stats.sparse_calls}"
        f" unsupported_fallbacks={sparse_stats.unsupported_fallbacks}"
        f" runtime_fallbacks={sparse_stats.runtime_fallbacks}"
        f" avg_sparsity={sparse_stats.avg_sparsity:.4f}"
    )
else:
    image = pipe(
        prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=max_sequence_length,
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0]

image.save(output_path)
