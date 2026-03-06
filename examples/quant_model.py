# model_name = "/dataset/meta-llama/Meta-Llama-3-8B/"
# model_name = "/data5/yliu7/HF_HOME/DeepSeek-R1-bf16-layer4"
# model_name = "/models/Qwen3-8B-FP8/"
# model_name = "/data5/yliu7/HF_HOME/DeepSeek-R1-bf16-layer4"
# model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# model_name="/models/Qwen3-235B-A22B/"
model_name = "/mnt/disk5/unsloth/DeepSeek-R1-BF16"
model_name = "/models/Qwen3-8B-FP8/"
# model_name = "/mnt/disk8/Qwen/Qwen3-8B-FP8"
# model_name = "/mnt/disk5/Qwen3-30B-A3B-FP8"
# model_name = "/models/DeepSeek-V2-Lite-Chat/"
# model_name = "/mnt/disk8/deepseek-ai/DeepSeek-V2-Lite-Chat"
model_name = "/mnt/disk8/Qwen/Qwen3-30B-A3B"
from auto_round import AutoRound


def fix_everything(seed):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


def main(args):
    model_name = args.model
    scheme = "FP8_STATIC"
    autoround = AutoRound(
        model_name,
        scheme=scheme,
        enable_torch_compile=True,
        iters=0,
        low_gpu_mem_usage=True,
        low_cpu_mem_usage=True,
        disable_opt_rtn=True,
        # disable_trust_remote_code=True,
        # static_kv_dtype="fp8",
    )
    model_base_name = model_name.rstrip("/").split("/")[-1]
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = "/mnt/disk5/hf_models/" + model_base_name + "-" + scheme + "-fp8-kv-2-test"
    print(f"Output dir: {output_dir}")

    model, save_folder = autoround.quantize_and_save(
        output_dir=output_dir,
        format="llm_compressor",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auto-round quantization script.")
    parser.add_argument(
        "-m",
        "--model",
        help="Path to the model.",
        type=str,
        default=model_name,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Path to the output directory.",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    main(args)
