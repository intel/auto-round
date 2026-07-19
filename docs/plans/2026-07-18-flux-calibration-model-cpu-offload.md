# FLUX Calibration Model CPU Offload Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make single-GPU FLUX calibration execute on CUDA through Diffusers model CPU offload instead of stalling in the group-offload path.

**Architecture:** Keep the existing single-device calibration helper and its cleanup contract. Change only the low-GPU-memory strategy from block-level group offload to component-level model CPU offload; leave full pipeline placement unchanged when low-GPU-memory mode is disabled.

**Tech Stack:** Python, PyTorch, Diffusers, pytest

---

### Task 1: Select model CPU offload

**Files:**
- Modify: `test/test_cpu/models/test_diffusion.py`
- Modify: `auto_round/compressors/diffusion_mixin.py`

**Step 1: Write the failing test**

Replace the group-offload assertion with a fake pipeline exposing
`enable_model_cpu_offload(device=...)`. Assert that the helper returns
`"model"`, selects `cuda:0`, and never calls `pipe.to()`.

**Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest test/test_cpu/models/test_diffusion.py::test_single_device_low_gpu_memory_uses_model_cpu_offload -q
```

Expected: failure because the helper still selects or requires group offload.

**Step 3: Write the minimal implementation**

In `_prepare_single_device_pipeline_for_calibration()`, remove the group-offload
preference and call:

```python
pipe.enable_model_cpu_offload(device=target_device)
return "model"
```

Keep the explicit unsupported-pipeline error and the full-pipeline path for
`low_gpu_mem_usage=False`.

**Step 4: Run focused tests**

Run:

```bash
python -m pytest test/test_cpu/models/test_diffusion.py -q
```

Expected: all focused diffusion tests pass.

### Task 2: Verify the real FLUX calibration path

**Files:**
- Use: `/root/rivermind-data/user2/xixi/run_autoround_flux_mxfp4_smooth_r32_r2_early_rtn_gpu0.sh`

**Step 1: Start the existing command in a persistent tmux session**

Use GPU 0 and the existing COCO2017 TSV with `nsamples=128`.

**Step 2: Verify accelerator placement**

Check `nvidia-smi` for the AutoRound Python PID and inspect the main log for
progress beyond the first calibration sample.

**Step 3: Leave the validated run active**

If the process advances without an exception or OOM, keep it running in tmux
and report the session name, PID, GPU memory, and log path.
