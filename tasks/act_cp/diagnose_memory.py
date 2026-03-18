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
"""Diagnose CUDA memory breakdown during tuning to understand where the peak comes from.

Instruments _quantize_block at key points to separate:
  - Block weights on GPU
  - Quant params (value, min_scale, max_scale)
  - Forward activations (autograd saved tensors)
  - Gradients
  - Optimizer state
"""

import gc
import sys
import time

import torch

from auto_round import AutoRound


def mb(x):
    return x / (1024**2)


def gb(x):
    return x / (1024**3)


def mem_snapshot(label=""):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(
        f"  [{label:>35s}]  alloc={mb(alloc):>10.1f} MB  peak={mb(peak):>10.1f} MB  reserved={mb(reserved):>10.1f} MB"
    )
    return alloc, peak


def count_params_memory(module):
    """Count memory of all parameters in a module."""
    total = 0
    for p in module.parameters():
        total += p.nelement() * p.element_size()
    return total


def count_quant_params(block):
    """Count memory of WrapperLinear quant params (value, min_scale, max_scale)."""
    total = 0
    count = 0
    for n, m in block.named_modules():
        if hasattr(m, "orig_layer") and hasattr(m, "params"):
            for key, p in m.params.items():
                total += p.nelement() * p.element_size()
                count += 1
    return total, count


def count_grad_memory(block):
    """Count memory of all gradients in a module."""
    total = 0
    for p in block.parameters():
        if p.grad is not None:
            total += p.grad.nelement() * p.grad.element_size()
    return total


def main():
    model_name = "/storage/yiliu7/Qwen/Qwen3-30B-A3B-L2"
    scheme = "MXFP8"
    iters = 3
    device = 0

    torch.cuda.set_device(device)

    for mode in ["baseline", "checkpointed"]:
        enable_ckpt = mode == "checkpointed"

        print(f"\n{'#'*80}")
        print(f"  MODE: {mode.upper()}")
        print(f"  enable_activation_checkpointing={enable_ckpt}")
        print(f"{'#'*80}")

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()

        autoround = AutoRound(
            model_name,
            scheme=scheme,
            iters=iters,
            low_gpu_mem_usage=True,
            enable_activation_checkpointing=enable_ckpt,
            device_map=device,
        )

        # Manually run quantize with instrumentation
        # We patch _quantize_block to add memory snapshots
        original_quantize_block = autoround._quantize_block

        def instrumented_quantize_block(block, input_ids, input_others, q_input=None, device="cpu", auto_offload=True):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            gc.collect()

            mem_snapshot("before block.to(device)")

            # Call original - but we need to instrument at a finer level
            # Instead, let's hook into the key internal stages
            # We'll use the standard flow and capture snapshots at forward/backward boundaries

            # Hook into the block's forward to capture activation memory
            forward_mem = {}

            def pre_forward_hook(mod, inputs):
                torch.cuda.synchronize()
                forward_mem["pre_fwd_alloc"] = torch.cuda.memory_allocated()

            def post_forward_hook(mod, inputs, outputs):
                torch.cuda.synchronize()
                forward_mem["post_fwd_alloc"] = torch.cuda.memory_allocated()
                forward_mem["post_fwd_peak"] = torch.cuda.max_memory_allocated()

            h1 = block.register_forward_pre_hook(pre_forward_hook)
            h2 = block.register_forward_hook(post_forward_hook)

            result = original_quantize_block(
                block, input_ids, input_others, q_input=q_input, device=device, auto_offload=auto_offload
            )

            h1.remove()
            h2.remove()

            _, final_peak = mem_snapshot("after block quantization")

            if forward_mem:
                fwd_growth = forward_mem.get("post_fwd_alloc", 0) - forward_mem.get("pre_fwd_alloc", 0)
                print(f"  [{'forward activation growth':>35s}]  {mb(fwd_growth):>10.1f} MB")
                print(f"  [{'forward peak (torch)':>35s}]  {mb(forward_mem.get('post_fwd_peak', 0)):>10.1f} MB")

            # Check quant params
            qp_mem, qp_count = count_quant_params(block)
            print(f"  [{'quant params':>35s}]  {mb(qp_mem):>10.1f} MB  ({qp_count} tensors)")

            # Check gradients
            grad_mem = count_grad_memory(block)
            print(f"  [{'gradients':>35s}]  {mb(grad_mem):>10.1f} MB")

            # Block weights
            w_mem = count_params_memory(block)
            print(f"  [{'block params (weights)':>35s}]  {mb(w_mem):>10.1f} MB")

            print(f"  [{'BLOCK PEAK':>35s}]  {mb(final_peak):>10.1f} MB = {gb(final_peak):.2f} GB")
            print()

            return result

        autoround._quantize_block = instrumented_quantize_block

        t0 = time.time()
        autoround.quantize()
        elapsed = time.time() - t0

        overall_peak = torch.cuda.max_memory_allocated()
        print(f"\n  OVERALL PEAK: {mb(overall_peak):.1f} MB = {gb(overall_peak):.2f} GB")
        print(f"  Wall time: {elapsed:.1f} s")

        del autoround
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
