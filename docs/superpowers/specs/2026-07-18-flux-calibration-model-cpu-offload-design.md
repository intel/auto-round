# FLUX Calibration Model CPU Offload Design

## Problem

Single-device diffusion calibration with `low_gpu_mem_usage=True` currently
prefers Diffusers group offload. In the FLUX calibration path, the first sample
remains on CPU and does not reach the block-input capture hook in a practical
amount of time.

## Design

Use Diffusers `enable_model_cpu_offload()` for single-device, low-GPU-memory
calibration. Components such as text encoders and the transformer are moved to
the selected accelerator only while active. Full-pipeline placement remains the
behavior when low-GPU-memory mode is disabled.

The helper returns `"model"` so the existing post-calibration cleanup removes
Accelerate hooks. Pipelines without model CPU offload support fail explicitly.

## Validation

1. A focused unit test verifies that low-GPU-memory calibration calls
   `enable_model_cpu_offload(device="cuda:0")` and never moves the full pipeline.
2. The existing diffusion CPU tests pass.
3. A real FLUX calibration run must show the AutoRound PID on GPU 0 and advance
   beyond the first calibration sample.
