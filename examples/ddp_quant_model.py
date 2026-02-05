"""
python ddp_quant_model.py --ddp --nsamples 128 --iters 100

DDP-enabled AutoRound Quantization Script

This script demonstrates how to use DDP (DistributedDataParallel) for model quantization
across multiple GPUs using AutoRound.

Usage:
    # Single GPU (default behavior)
    python ddp_quant_model.py

    # Multiple GPUs using mp.spawn
    python ddp_quant_model.py --ddp

    # Using torchrun
    torchrun --nproc_per_node=2 --master_port=29501 ddp_quant_model.py
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import transformers
from torch import Tensor
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

model_name = "Kimi-K2-Instruct-BF16"
model_name = "/models/Qwen3-30B-A3B"
model_name = "facebook/opt-125m"
model_name = "/data5/yliu7/HF_HOME/meta-llama/Llama-3.2-1B-Instruct/"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

model_name = "/data5/yliu7/HF_HOME/unsloth/gpt-oss-20b-BF16"
model_name = "/data4/yliu/unsloth/gpt-oss-120b-BF16"
model_name = "/storage/yiliu7/unsloth/gpt-oss-20b-BF16/"
model_name = "/storage/yiliu7/unsloth/gpt-oss-120b-BF16"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model_name = "/storage/yiliu7/Qwen/Qwen2-VL-7B-Instruct"
model_name = "/models/DeepSeek-V2-Lite-Chat/"
model_name = "/storage/yiliu7/deepseek-ai/DeepSeek-V2-Lite-Chat/"
model_name = "/storage/yiliu7/tflsxyy/DeepSeek-V3-bf16-4layers"
model_name = "/storage/yiliu7/deepseek-ai/DeepSeek-R1"
model_name = "Qwen/Qwen3-Embedding-4B"
# model_name = "/storage/yiliu7/Qwen/Qwen3-VL-30B-A3B-Instruct"
model_name = "/storage/jenkins/huggingface/Llama-4-Scout-17B-16E-Instruct"
# model_name = "/storage/yiliu7/meta-llama/Llama-4-Scout-17B-16E"
# model_name = "/storage/yiliu7/unsloth/gpt-oss-20b-BF16/"
model_name = "/storage/yiliu7/Qwen/Qwen3-VL-30B-A3B-Instruct"
model_name = "/storage/yiliu7/Qwen/Qwen3-8B/"
model_name = "/data5/yiliu4/Qwen/Qwen2-0.5B"
model_name = "Qwen/Qwen2-0.5B"
model_name = "Qwen/Qwen3-8B"
# from transformers import Qwen2VLForConditionalGeneration
# tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_name,device_map="cpu", torch_dtype="auto",trust_remote_code=True)
# from sentence_transformers import SentenceTransformer

# Load the model
# # model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")
# tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-4B', padding_side='left')
# model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-4B')

# block = model.model.layers
device_map = {}


from auto_round import AutoRound
from auto_round import schemes as ar_schemes

scheme = "MXFP8"
scheme = ar_schemes.FP8_STATIC
scheme = ar_schemes.MXFP8
scheme = ar_schemes.MXFP4
# scheme = "MXFP4"
scheme = ar_schemes.FP8_STATIC
# scheme = ar_schemes.NVFP4
scheme = "FP8_STATIC"
# scheme = "MXFP4"
scheme = "W4A16"
# scheme = "W4A16"
# "re:.*lm_head",
# "re:.*self_attn",
# "re:.*attn",
# "re:.*attention.*",
# "re:.*router",

# from mem_patch import *


# from transformers.conversion_mapping import register_checkpoint_conversion_mapping

# register_checkpoint_conversion_mapping("deepseek_v3", [], overwrite=True)
# model.eval()


def setup_ddp(rank, world_size):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12356")

    # Initialize process group
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def quantize_model(rank, world_size, model_name, scheme, iters=4, nsamples=32):
    """
    Quantize model on a specific GPU rank.

    Args:
        rank: GPU rank for this process
        world_size: Total number of GPUs
        model_name: Model name or path
        scheme: Quantization scheme
        iters: Number of iterations
        nsamples: Number of samples
    """
    print(f"[Rank {rank}/{world_size}] Starting quantization")

    # Setup DDP if using multiple GPUs
    if world_size > 1:
        setup_ddp(rank, world_size)

    # Set device for this process
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    # try:
    # Initialize AutoRound
    autoround = AutoRound(
        model_name,
        scheme=scheme,
        iters=iters,
        nsamples=nsamples,
        # low_gpu_mem_usage=False,
        low_gpu_mem_usage=True,
        # device=f"cuda:{rank}",
        device_map=rank,
        enable_torch_compile=True,
    )
    SAVE_DIR = model_name.rstrip("/").split("/")[-1] + f"-{scheme}"
    # Only rank 0 saves the model
    if rank == 0:
        print(f"[Rank {rank}] Quantizing and saving to {SAVE_DIR}")
        model, _ = autoround.quantize_and_save(format="auto_round", output_dir=SAVE_DIR)
        print(f"[Rank {rank}] Quantized model saved to {SAVE_DIR}")
    else:
        # Other ranks just run quantization without saving
        print(f"[Rank {rank}] Running quantization (not saving)")
        model, _ = autoround.quantize_and_save(format="auto_round", output_dir=f"{SAVE_DIR}_rank{rank}")

    # Synchronize all processes
    if world_size > 1:
        dist.barrier()

    print(f"[Rank {rank}] Quantization completed")

    # except Exception as e:
    #     print(f"[Rank {rank}] Error during quantization: {e}")
    #     raise

    # finally:
    #     # Cleanup DDP
    if world_size > 1:
        cleanup_ddp()


def main_single_gpu(model_name, scheme, iters, nsamples):
    """Run quantization on a single GPU (original behavior)."""
    print("Running single GPU quantization")
    autoround = AutoRound(
        model_name,
        scheme=scheme,
        iters=iters,
        nsamples=nsamples,
        low_gpu_mem_usage=False,
    )

    SAVE_DIR = model_name.rstrip("/").split("/")[-1] + f"-{scheme}"
    model, _ = autoround.quantize_and_save(format="auto_round", output_dir=SAVE_DIR)
    print(f"Quantized model saved to {SAVE_DIR}")
    return model


def main_spawn(model_name, scheme, iters, nsamples):
    """Main function using mp.spawn for multi-GPU quantization."""
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    if world_size < 2:
        print("Warning: Only 1 GPU detected. Running single GPU mode.")
        return main_single_gpu(model_name, scheme, iters, nsamples)

    print(f"Starting DDP quantization with {world_size} GPUs")

    mp.spawn(quantize_model, args=(world_size, model_name, scheme, iters, nsamples), nprocs=world_size, join=True)

    print("Quantization completed!")


def main_torchrun(model_name, scheme, iters, nsamples):
    """Main function for torchrun-based execution."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"Torchrun mode: rank={rank}, local_rank={local_rank}, world_size={world_size}")

    quantize_model(local_rank, world_size, model_name, scheme, iters, nsamples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoRound Quantization with DDP support")
    parser.add_argument("--model_name", type=str, default=model_name, help="Model name or path")
    parser.add_argument(
        "--scheme", type=str, default="FP8_STATIC", help="Quantization scheme (FP8_STATIC, MXFP8, MXFP4, etc.)"
    )
    parser.add_argument("--iters", type=int, default=4, help="Number of iterations")
    parser.add_argument("--nsamples", type=int, default=32, help="Number of samples")
    parser.add_argument("--ddp", action="store_true", help="Enable DDP multi-GPU mode")

    args = parser.parse_args()

    # For backward compatibility with existing hardcoded values
    model_name = args.model_name

    # Parse scheme from string if needed
    from auto_round import schemes as ar_schemes

    scheme_map = {
        "FP8_STATIC": ar_schemes.FP8_STATIC,
        "MXFP8": ar_schemes.MXFP8,
        "MXFP4": ar_schemes.MXFP4,
    }
    # scheme = scheme_map.get(args.scheme, args.scheme)

    # Check if running with torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        print("Detected torchrun environment")
        main_torchrun(model_name, scheme, args.iters, args.nsamples)
    elif args.ddp:
        print("Using mp.spawn mode for multi-GPU quantization")
        main_spawn(model_name, scheme, args.iters, args.nsamples)
    else:
        print("Using single GPU mode")
        main_single_gpu(model_name, scheme, args.iters, args.nsamples)

# with torch.no_grad(), torch.device("cuda"):
#     model = AutoModelForCausalLM.from_pretrained(SAVE_DIR, device_map="auto", torch_dtype="auto",trust_remote_code=True)
#     model.eval()
#     input_text = "Explain the theory of relativity in simple terms."
#     inputs = autoround.tokenizer(input_text, return_tensors="pt").to(model.device)
#     outputs = model.generate(**inputs, max_new_tokens=20)
#     decoded_output = autoround.tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print("Generated Output:")
#     print(decoded_output)
# with torch.no_grad():
#     model, _  = autoround.quantize_and_save(format="auto_round", output_dir=SAVE_DIR)
#     print(f"Quantized model saved to {SAVE_DIR}")
