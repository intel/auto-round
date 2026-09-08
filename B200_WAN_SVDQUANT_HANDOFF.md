# B200 Wan2.2 SVDQuant MXFP4 Handoff

## Goal

Quantize the dual-transformer Diffusers `Wan2.2-T2V-A14B` pipeline with:

- AutoRound SVDQuant rank 32
- MXFP4 E2M1 W4A4, group size 32, UE8M0 scales
- Smooth enabled
- SignRound as the final residual-linear optimizer
- BF16 low-rank branch
- Nunchaku-compatible export

After export, validate a short latent trajectory before decoding or generating a full video.

## Branch

Use:

```bash
git switch wangchang/wan-svdquant-nunchaku
```

Relevant commits:

- `f2537f04`: Wan adapter, dual-expert calibration routing, exporter and tests
- `e9c7f95a`: CPU calibration cache for B200
- `f972f69c`: single-B200 default

The branch intentionally excludes the failed RTX 5090 OOM experiments: MXFP chunking, gradient checkpointing, cross-GPU loss placement, forced post-calibration CPU moves, and altered GPU memory ratios.

## Required assets

Set the following paths for the B200 host:

- `MODEL`: Diffusers `Wan2.2-T2V-A14B-Diffusers`
- `DATASET`: COCO caption TSV used for calibration
- `PY`: Python 3.12 environment executable
- `ROOT`: parent directory for outputs, logs and caches
- `REPO`: this AutoRound checkout; normally inferred automatically

The original host used:

- model: `/mnt/disk3/changwa1/Wan2.2-T2V-A14B-Diffusers`
- dataset: `/mnt/disk3/changwa1/coco2017-captions.tsv`

Do not copy the old virtual environment blindly. Create a B200-compatible environment and install this checkout with:

```bash
python -m pip install --no-build-isolation -e .
```

## Nunchaku requirement

Export depends on the Nunchaku Wan runtime and MXFP4 extension. The source host used a package tree containing:

- `nunchaku/models/transformers/transformer_wan.py`
- `NunchakuWanTransformer3DModel`
- `SVDQW4A4Linear`

The B200 machine must have a CUDA/PyTorch/B200-compatible Nunchaku build. Do not assume the RTX 5090 compiled extension is binary-compatible. Verify imports first:

```bash
PYTHONPATH=/path/to/nunchaku python -c "from nunchaku.models.transformers.transformer_wan import NunchakuWanTransformer3DModel; from nunchaku.models.linear import SVDQW4A4Linear; print('ok')"
```

The launcher currently derives the Nunchaku path from `$ROOT/nunchaku-torch-cu130-ubuntu22/nunchaku`. If the B200 location differs, either preserve this layout or include the correct Nunchaku parent in `PYTHONPATH` before launching.

## B200 memory strategy

The default is one visible 98 GB B200, physical GPU 0, with `--low_gpu_mem_usage`.

For a single visible GPU, AutoRound uses Diffusers model CPU offload. Pipeline components move between CPU and GPU as needed. Calibration inputs and reference outputs stay on CPU and only active data moves to GPU. Smooth SVD and SignRound temporaries use the B200.

The launcher also passes `--disable_low_cpu_mem_usage`. This disables disk-based model-weight offload and keeps working weights in host RAM. It is separate from GPU model CPU offload. Ensure the host has enough RAM.

Multi-GPU is not required by default. Current Smooth and SignRound processing is sequential per Block and mainly executes on the first visible GPU; three GPUs do not provide 3× SignRound speedup.

## Launcher

Use:

```bash
scripts/launch_wan_svdquant_mxfp4_b200.sh [smoke|fast|quality] [GPU_LIST]
```

Default GPU list is `0`.

Example with target paths:

```bash
ROOT=/data/wan \
MODEL=/data/wan/Wan2.2-T2V-A14B-Diffusers \
DATASET=/data/wan/coco2017-captions.tsv \
PY=/data/venvs/autoround/bin/python \
./scripts/launch_wan_svdquant_mxfp4_b200.sh smoke 0
```

Profiles:

| Profile | Residual iters | Samples | Calibration diffusion steps | SignRound iters | Smooth grids | Smooth calls |
|---|---:|---:|---:|---:|---:|---:|
| smoke | 1 | 1 | 2 | 1 | 2 | 1 |
| fast | 1 | 2 | 2 | 20 | 6 | 4 |
| quality | 3 | 8 | 4 | 200 | 20 | 16 |

Override any setting using `RESIDUAL_ITERS`, `NSAMPLES`, `STEPS`, `SIGNROUND_ITERS`, `SMOOTH_GRIDS`, `SMOOTH_MAX_CALLS`, `RANK`, `OUT`, or `LOG`.

Keep `LOW_GPU_MEM_USAGE=1` initially. Only set it to `0` after measuring sufficient memory headroom.

## Required order

1. Verify B200 driver, CUDA, PyTorch, free VRAM and host RAM.
2. Checkout this branch and install it editable.
3. Verify the Nunchaku Wan import and extension compatibility.
4. Run focused tests:

```bash
python -m pytest \
  test/test_cpu/export/test_svdquant_wan_adapter.py \
  test/test_cpu/test_quantize_wan_script.py \
  test/test_cpu/models/test_vlm_ram_reduction.py -q
```

5. Run `smoke 0`.
6. Require both Transformer experts to finish all 40 Blocks and export.
7. Validate each expert has exactly 400 MXFP4 quantized linears and finite tensors.
8. Run a 256×256, 9-frame, 4-step latent-only check with seed 0.
9. Require finite values at every step, especially after switching to `transformer_2`.
10. If Smoke passes, run `fast 0`.
11. Run the latent-only check again, then a very short video.
12. Only then consider `quality` and the final video.

## Export acceptance criteria

Output must include:

- `model_index.json`
- `transformer/config.json`
- `transformer/diffusion_pytorch_model.safetensors`
- `transformer_2/config.json`
- `transformer_2/diffusion_pytorch_model.safetensors`
- scheduler, tokenizer, text encoder and VAE assets

Each expert must have:

- 400 quantized Linear modules
- precision `mxfp4`
- rank 32
- finite floating-point tensors

Both Transformer entries in `model_index.json` must reference `NunchakuWanTransformer3DModel`.

## Quantization semantics

SVDQuant residual iteration always uses RTN QDQ internally. It does not make the final optimizer RTN. The final optimizer is SignRound because the launcher passes `--algorithm signround,svdquant`. `--calib_num_inference_steps` controls calibration pipeline steps independently from `--num_inference_steps`, which is reserved for evaluation generation.

## Existing checkpoint findings

The data-free No-smooth + RTN checkpoint with residual iteration 1 ran but generated nearly uniform blurred output.

Residual iteration 10 improved sampled effective-weight error only modestly:

- average relative RMSE: about 10.73% to 10.30%
- sampled layers improved 12/12

However, its 4-step latent trajectory became all NaN immediately after entering the low-noise expert. Do not spend time generating a full video from that checkpoint. The B200 objective is a genuinely data-driven Smooth + SignRound export.

## Known pitfalls

- The Wan pipeline has two 40-layer Transformer experts; calibration must route all primary-pass timesteps to the primary expert and all secondary-pass timesteps to the secondary expert. This branch implements that routing.
- Do not hard-link mutable `model_index.json` between source and output models.
- A technically valid MP4 or successful MXFP4 kernel dispatch does not prove acceptable diffusion quality.
- Single-layer cosine near 0.99 can still accumulate into trajectory collapse across hundreds of W4A4 linears.
- Do not start a long Quality job before Smoke export and latent validation pass.
- Do not restore the rejected persistent queue across all 400 projections.

## Final video target

Only after all short checks pass:

- prompt: `Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.`
- 1280×720
- 81 frames
- 40 inference steps
- boundary ratio 0.875
- flow shift 5.0
- guidance 4.0 / low-noise guidance 3.0
- 16 FPS
- FP32 VAE
