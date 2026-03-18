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
"""Micro-benchmark v2: verify per-WrapperLinear checkpointing reduces peak memory.

Usage:
    /home/yiliu7/workspace/ar-local/bin/python tasks/act_cp/micro_benchmark_v2.py
"""

import gc
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


def run_one_block(model_name, scheme, device, enable_ckpt, mode_name):
    """Run quantization and measure per-block memory."""
    print(f"\n{'#'*80}")
    print(f"  MODE: {mode_name}")
    print(f"{'#'*80}\n")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    autoround = AutoRound(
        model_name,
        scheme=scheme,
        iters=2,
        low_gpu_mem_usage=True,
        enable_activation_checkpointing=enable_ckpt,
        device_map=device,
    )

    print(f"  block_forward: {autoround.block_forward.__name__}")

    # Patch to instrument
    original_quantize_block = autoround._quantize_block

    results = {}

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
        mem_snap("block on GPU")

        # Get calibration outputs
        output = autoround._get_block_outputs(  # noqa: F821
            block,
            input_ids,
            input_others,
            autoround.batch_size * autoround.infer_bs_coeff,  # noqa: F821
            device,
            autoround.cache_device,  # noqa: F821
        )

        # Wrapper block
        quantized_layers, unquantized_layers = wrapper_block(
            block,
            autoround.enable_minmax_tuning,  # noqa: F821
            autoround.enable_norm_bias_tuning,  # noqa: F821
            enable_torch_compile=autoround.enable_torch_compile,  # noqa: F821
            device=device,
            enable_activation_checkpointing=enable_ckpt,
        )
        mem_snap("after wrapper_block")

        # Check if checkpointing is enabled in WrapperLinear
        ckpt_count = 0
        for n, m in block.named_modules():
            if hasattr(m, "enable_activation_checkpointing") and m.enable_activation_checkpointing:
                ckpt_count += 1
        print(f"    WrapperLinear layers with checkpointing: {ckpt_count}")

        # Setup optimizer
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

        # Run ONE iteration
        torch.cuda.reset_peak_memory_stats()
        m_before = mem_snap("before iteration")

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
        t0 = time.time()
        output_q = autoround.block_forward(  # noqa: F821
            block, current_input_ids, current_input_others, autoround.amp, autoround.amp_dtype, device  # noqa: F821
        )
        m_fwd = mem_snap("after forward")

        # Loss
        current_output = output[0] if isinstance(output, (list, tuple)) else output
        if isinstance(current_output, list):
            current_output = torch.cat([current_output[i] for i in indices], dim=0)
        loss = mse_loss(output_q.to(torch.float32), current_output[: output_q.shape[0]].to(device).to(torch.float32))

        # Backward
        (loss * 1000).backward()
        t1 = time.time()

        m_bwd = mem_snap("after backward")

        optimizer.step()
        optimizer.zero_grad()
        mem_snap("after step + zero_grad")

        peak = torch.cuda.max_memory_allocated()

        print(f"\n    >>> FORWARD GROWTH: {mb(m_fwd - m_before):.1f} MB <<<")
        print(f"    >>> PEAK: {mb(peak):.1f} MB = {gb(peak):.2f} GB <<<")
        print(f"    >>> ITERATION TIME: {t1-t0:.2f} s <<<")

        results["peak_mb"] = mb(peak)
        results["fwd_growth_mb"] = mb(m_fwd - m_before)
        results["time_s"] = t1 - t0

        # Clean up
        from auto_round.wrapper import unwrapper_block

        unwrapper_block(block, {})
        from auto_round.utils.model import mv_module_from_gpu

        mv_module_from_gpu(block)

        return output, output

    autoround._quantize_block = instrumented_quantize_block

    try:
        autoround.quantize()  # noqa: F821
    except Exception:
        pass

    del autoround
    torch.cuda.empty_cache()
    gc.collect()

    return results


def main():
    model_name = "/storage/yiliu7/Qwen/Qwen3-30B-A3B-L2"
    scheme = "MXFP8"
    device = 0

    torch.cuda.set_device(device)

    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    baseline = run_one_block(model_name, scheme, device, False, "BASELINE (no checkpointing)")
    ckpt = run_one_block(model_name, scheme, device, True, "PER-WRAPPERLAYER CHECKPOINTING")

    if baseline and ckpt:
        saved = baseline["peak_mb"] - ckpt["peak_mb"]
        pct = saved / baseline["peak_mb"] * 100 if baseline["peak_mb"] > 0 else 0
        slowdown = (ckpt["time_s"] - baseline["time_s"]) / baseline["time_s"] * 100

        print(f"\n{'='*80}")
        print("  COMPARISON")
        print(f"{'='*80}")
        print(f"  Baseline peak:       {baseline['peak_mb']:.1f} MB = {baseline['peak_mb']/1024:.2f} GB")
        print(f"  Checkpointed peak:   {ckpt['peak_mb']:.1f} MB = {ckpt['peak_mb']/1024:.2f} GB")
        print(f"  Memory SAVED:        {saved:.1f} MB = {saved/1024:.2f} GB ({pct:.1f}%)")
        print(f"  Baseline time:       {baseline['time_s']:.2f} s")
        print(f"  Checkpointed time:   {ckpt['time_s']:.2f} s")
        print(f"  Time overhead:       {slowdown:+.1f}%")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
