#!/usr/bin/env python
# Copyright (c) 2025 Intel Corporation
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
"""Full model verification: quantize Qwen3-30B-A3B with activation checkpointing,
save, reload, and check generation quality.

Usage:
    /home/yiliu7/workspace/ar-local/bin/python tasks/act_cp/verify_full_model.py
"""

import gc
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound

MODEL_NAME = "/storage/yiliu7/Qwen/Qwen3-30B-A3B"
SCHEME = "MXFP8"
ITERS = 10
DEVICE = 0
SAVE_DIR = "/storage/yiliu7/Qwen/Qwen3-30B-A3B-MXFP8-AR-ACT-CKPT"

TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a Python function to check if a number is prime.",
    "Translate 'Hello, how are you?' to Chinese.",
]


def quantize_model():
    """Quantize the full model with activation checkpointing."""
    print(f"\n{'='*80}")
    print(f"  STEP 1: Quantize {MODEL_NAME}")
    print(f"  Scheme: {SCHEME}, iters: {ITERS}")
    print("  enable_activation_checkpointing=True")
    print(f"{'='*80}\n")

    torch.cuda.set_device(DEVICE)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    t0 = time.time()
    autoround = AutoRound(
        MODEL_NAME,
        scheme=SCHEME,
        iters=ITERS,
        low_gpu_mem_usage=True,
        enable_activation_checkpointing=True,
        device_map=DEVICE,
    )

    res = autoround.quantize_and_save(format="auto_round", output_dir=SAVE_DIR)
    elapsed = time.time() - t0

    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    peak_gb = peak_mb / 1024

    print("\n  Quantization complete!")
    print(f"  Save dir: {SAVE_DIR}")
    print(f"  Peak GPU memory: {peak_mb:.1f} MB = {peak_gb:.2f} GB")
    print(f"  Wall time: {elapsed:.1f} s ({elapsed/60:.1f} min)")

    del autoround
    torch.cuda.empty_cache()
    gc.collect()

    return peak_gb, elapsed


def test_generation():
    """Reload the quantized model and test generation quality."""
    print(f"\n{'='*80}")
    print("  STEP 2: Reload and test generation")
    print(f"  Loading from: {SAVE_DIR}")
    print(f"{'='*80}\n")

    tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        SAVE_DIR,
        device_map=DEVICE,
        trust_remote_code=True,
    )

    print(f"  Model loaded: {model.__class__.__name__}")
    print(f"  Device: {next(model.parameters()).device}")
    print()

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"  --- Prompt {i+1}: {prompt}")

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=1.0,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        print(f"  Response: {generated[:300]}")
        print()

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()


def main():
    peak_gb, elapsed = quantize_model()
    test_generation()

    print(f"\n{'='*80}")
    print("  FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Scheme: {SCHEME}, iters: {ITERS}")
    print(f"  Peak GPU memory: {peak_gb:.2f} GB")
    print(f"  Total quantization time: {elapsed:.1f} s ({elapsed/60:.1f} min)")
    print(f"  Output dir: {SAVE_DIR}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
