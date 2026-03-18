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
"""Micro-benchmark: measure exactly where GPU memory goes during one tuning iteration.

Instruments the actual tuning loop to snapshot memory at key points:
1. After block.to(device) but before wrapper
2. After wrapper_block (WrapperLinear params created)
3. After forward pass (autograd saved tensors)
4. After backward (gradients computed, saved tensors freed)
5. After optimizer step

Usage:
    /home/yiliu7/workspace/ar-local/bin/python tasks/act_cp/micro_benchmark.py
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


def mem_snap(label=""):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    print(f"  [{label:>45s}]  alloc={mb(alloc):>10.1f} MB  peak={mb(peak):>10.1f} MB")
    return alloc


def main():
    model_name = "/storage/yiliu7/Qwen/Qwen3-30B-A3B-L2"
    scheme = "MXFP8"
    device = 0

    torch.cuda.set_device(device)

    for mode_name, enable_ckpt in [("BASELINE", False), ("BLOCK_CHECKPOINT", True)]:
        print(f"\n{'#'*80}")
        print(f"  MODE: {mode_name}")
        print(f"{'#'*80}\n")

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()

        autoround = AutoRound(
            model_name,
            scheme=scheme,
            iters=2,  # Just 2 iterations for measurement
            low_gpu_mem_usage=True,
            enable_activation_checkpointing=enable_ckpt,
            device_map=device,
        )

        # Manually run 1 block to measure memory at each stage
        # Get the first block
        block_names = autoround.quant_block_list[0]  # noqa: F821
        block_name = block_names[0]
        print(f"  Block: {block_name}")
        print(f"  block_forward: {autoround.block_forward.__name__}")  # noqa: F821

        # Get block
        parts = block_name.split(".")
        block = autoround.model  # noqa: F821
        for p in parts:
            block = getattr(block, p)

        # Get inputs - run calibration first
        # We need to actually run quantize but intercept it
        # Instead, let's patch _quantize_block

        original_quantize_block = autoround._quantize_block  # noqa: F821

        def instrumented_quantize_block(block, input_ids, input_others, q_input=None, device="cpu", auto_offload=True):
            from auto_round.compressors.base import convert_module_to_hp_if_necessary, materialize_model_
            from auto_round.wrapper import wrapper_block

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            gc.collect()

            mem_snap("start (empty GPU)")

            # Move block to GPU
            materialize_model_(block)
            convert_module_to_hp_if_necessary(block, autoround.amp_dtype, device)  # noqa: F821
            block = block.to(device)

            m1 = mem_snap("block on GPU (orig weights only)")

            # Get calibration outputs
            output = autoround._get_block_outputs(  # noqa: F821
                block,
                input_ids,
                input_others,
                autoround.batch_size * autoround.infer_bs_coeff,  # noqa: F821
                device,
                autoround.cache_device,  # noqa: F821
            )

            mem_snap("after calibration outputs")

            # Wrapper block - creates WrapperLinear with value, min_scale, max_scale params
            quantized_layers, unquantized_layers = wrapper_block(
                block,
                autoround.enable_minmax_tuning,  # noqa: F821
                autoround.enable_norm_bias_tuning,  # noqa: F821
                enable_torch_compile=autoround.enable_torch_compile,  # noqa: F821
                device=device,
            )

            m2 = mem_snap("after wrapper_block (quant params created)")

            # Count params
            total_value_bytes = 0
            total_scale_bytes = 0
            n_wrapped = 0
            for n, m in block.named_modules():
                if hasattr(m, "orig_layer") and hasattr(m, "params"):
                    n_wrapped += 1
                    for key, p in m.params.items():
                        b = p.nelement() * p.element_size()
                        if "value" in key:
                            total_value_bytes += b
                        else:
                            total_scale_bytes += b
            print(f"    Wrapped layers: {n_wrapped}")
            print(f"    value params: {mb(total_value_bytes):.1f} MB")
            print(f"    scale params: {mb(total_scale_bytes):.1f} MB")
            print(f"    quant param growth: {mb(m2-m1):.1f} MB")

            # Setup optimizer (minimal)
            round_params = []
            minmax_params = []
            for n, m in block.named_modules():
                if hasattr(m, "orig_layer"):
                    for key in m.params.keys():
                        if "min" in key or "max" in key:
                            minmax_params.append(m.params[key])
                        else:
                            round_params.append(m.params[key])

            from auto_round.compressors.base import SignSGD

            optimizer = SignSGD(round_params, lr=0.005, weight_decay=0)
            mse_loss = torch.nn.MSELoss().to(device)

            # Now run ONE forward pass and measure
            torch.cuda.reset_peak_memory_stats()
            m_before_fwd = mem_snap("before forward pass")

            # Sample one batch
            indices = torch.arange(min(autoround.batch_size, len(input_ids)))  # noqa: F821
            current_input_ids, current_input_others = autoround._sampling_inputs(  # noqa: F821
                input_ids,
                input_others,
                indices,
                seqlen=autoround.seqlen,  # noqa: F821
                batch_dim=autoround.batch_dim,  # noqa: F821
                share_cache_keys=autoround.shared_cache_keys,  # noqa: F821
            )

            # Forward
            output_q = autoround.block_forward(  # noqa: F821
                block, current_input_ids, current_input_others, autoround.amp, autoround.amp_dtype, device  # noqa: F821
            )

            m_after_fwd = mem_snap("after forward (autograd tensors saved)")
            fwd_growth = m_after_fwd - m_before_fwd
            print(f"    >>> FORWARD MEMORY GROWTH: {mb(fwd_growth):.1f} MB <<<")
            print("    >>> (This is what checkpointing should eliminate) <<<")

            # Get reference output for loss
            current_output = output[0] if isinstance(output, (list, tuple)) else output
            if isinstance(current_output, list):
                current_output = torch.cat([current_output[i] for i in indices], dim=0)
            else:
                current_output = current_output

            # Loss
            loss = mse_loss(
                output_q.to(torch.float32), current_output[: output_q.shape[0]].to(device).to(torch.float32)
            )

            mem_snap("after loss computation")

            # Backward
            (loss * 1000).backward()

            m_after_bwd = mem_snap("after backward (grads computed)")
            print(f"    >>> POST-BACKWARD MEMORY: {mb(m_after_bwd):.1f} MB <<<")

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

            m_after_step = mem_snap("after optimizer step + zero_grad")

            # Peak
            peak = torch.cuda.max_memory_allocated()
            print(f"\n    >>> PEAK DURING FWD+BWD: {mb(peak):.1f} MB = {gb(peak):.2f} GB <<<")

            # Clean up
            from auto_round.wrapper import unwrapper_block

            unwrapper_block(block, {})
            from auto_round.utils.model import mv_module_from_gpu

            mv_module_from_gpu(block)

            return output, output

        autoround._quantize_block = instrumented_quantize_block  # noqa: F821

        try:
            autoround.quantize()  # noqa: F821
        except Exception as e:
            print(f"  Error (expected after first block): {e}")

        del autoround
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
