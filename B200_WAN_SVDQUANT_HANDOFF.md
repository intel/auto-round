# Wan2.2 T2V A14B: calibrated SVDQuant export and Nunchaku inference

## Scope and hardware

Target: `Wan-AI/Wan2.2-T2V-A14B-Diffusers`, pinned revision `5be7df9619b54f4e2667b2755bc6a756675b5cd7`.
Both 40-block experts are quantized: `transformer` and `transformer_2`, each with 400 MXFP4 projections.
The pipeline retains the source scheduler, VAE and boundary ratio. The pinned scheduler has `flow_shift=3.0`, and `boundary_ratio=0.875`.

**AutoRound export and Nunchaku inference have different hardware requirements.** AutoRound uses PyTorch SVD, QDQ and packing; this export does not require installing Nunchaku or executing its CUDA kernels. Use a CUDA GPU with enough VRAM and host RAM for A14B calibration/tuning. Start with one visible GPU and CPU model offload. This is not a measured A14B memory guarantee.

The Nunchaku branch's native MXFP4 kernel supports SM120/SM121 (for example RTX 5090). It does **not** support B200/SM100 or H100/SM90. Its `mma.sync` block-scale instruction fails offline compilation for SM100; B200 needs a separate `tcgen05` implementation. Adding SM100 to `setup.py` cannot fix this. A B200 machine can prepare the export using AutoRound; this branch cannot yet validate native Nunchaku inference on that B200.

No A14B model weights were downloaded or executed on the development host. Local validation includes a real 5B Smooth + SignRound export and recognizable 33-frame videos with the repaired Nunchaku RoPE loader, plus focused dual-expert regression checks. These do not establish A14B visual quality or B200 runtime support.

## Get the branches

```bash
git clone --branch wangchang/wan-svdquant-nunchaku https://github.com/changwangss/auto-round.git auto-round-wan
git clone --branch wangchang/wan-mxfp4-runtime https://github.com/changwangss/nunchaku.git nunchaku-wan
```

Use a Python 3.12 environment with a working CUDA PyTorch installation appropriate to the target GPU. Development checks used PyTorch `2.13.0+cu130`, Diffusers `0.39.0`, Transformers `5.12.1`; the local CUDA extension is not portable between arbitrary environments.

```bash
python -m pip install --no-build-isolation -e ./auto-round-wan
python -m pip install 'diffusers==0.39.0' 'transformers==5.12.1' accelerate sentencepiece protobuf safetensors
python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'
```

## Download on the target machine only

Choose a volume with room for the source, a complete exported pipeline, and temporary serialization files. Working BF16 weights for both experts remain in host RAM (`low_cpu_mem_usage=False`); CPU offload saves VRAM but does not reduce this host RAM requirement. Each run keeps a separate output; do not overwrite a verified export.

```bash
export MODEL=/data/wan/Wan2.2-T2V-A14B-Diffusers
hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --revision 5be7df9619b54f4e2667b2755bc6a756675b5cd7 --local-dir "$MODEL"
```

A pre-existing local Diffusers snapshot may be passed directly. The runner never replaces its scheduler or downloads the native non-Diffusers checkpoint.

## Quantize both experts with calibration

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
python -u auto-round-wan/scripts/quantize_wan_a14b_svdquant.py \
  --model "$MODEL" --output /data/wan/a14b-svdquant-smoke \
  --profile smoke --height 256 --width 256 --num-frames 9
```

This uses the public AutoRound API with `SVDQuantConfig(smooth_enabled=True)` followed by `SignRoundConfig`, MXFP4 W4A4 group 32, rank 32 and BF16 low-rank weights. Both experts receive their own calibration/tuning pass. Calibration temporarily routes all steps to the active expert, then restores the original boundary. The older `quantize_wan_svdquant_nunchaku.py` is a **data-free** export tool; it is not the calibrated workflow above.

| Profile | Samples | Calibration steps | SignRound iters | Residual iters | Smooth grids | Smooth calls |
|---|---:|---:|---:|---:|---:|---:|
| smoke | 1 | 2 | 1 | 1 | 2 | 1 |
| fast | 2 | 2 | 20 | 1 | 6 | 4 |
| quality | 8 | 4 | 200 | 3 | 20 | 16 |

Smoke uses a built-in prompt and tests the complete machinery; one SignRound iteration is not a quality recommendation. Fast also has built-in prompts. Quality requires your representative `--prompts-file` (one UTF-8 prompt per nonempty line) or `--dataset` (AutoRound dataset name or caption TSV). Dimensions remain explicit and independent of the profile.

After smoke export **and target runtime validation** pass, a larger calibration example is:

```bash
python -u auto-round-wan/scripts/quantize_wan_a14b_svdquant.py \
  --model "$MODEL" --output /data/wan/a14b-svdquant-quality \
  --profile quality --prompts-file /data/wan/calibration-prompts.txt \
  --height 480 --width 832 --num-frames 33
```

This larger setting has not been measured on A14B; reduce dimensions, frames or samples if needed. The runner accepts individual overrides (`--nsamples`, `--calib-steps`, `--iters`, `--rank`, `--residual-iters`, `--smooth-grids`, `--smooth-calls`). `--device` is a logical CUDA index; the launcher uses one GPU. The shell compatibility entrypoint is now portable:

```bash
MODEL="$MODEL" OUT=/data/wan/a14b-other-smoke PY=python \
  bash auto-round-wan/scripts/launch_wan_svdquant_mxfp4_b200.sh smoke 0
```

## Export acceptance

The output contains both expert directories with `config.json` and `diffusion_pytorch_model.safetensors`, plus scheduler, tokenizer, text encoder, VAE and `model_index.json`. Both Transformer entries point to `nunchaku.NunchakuWanTransformer3DModel`. The source pipeline index is preserved.

`quantization-run.json` records settings and package/GPU versions. `export-audit.json` is written only after both experts have 400 packed projections and all exported floating-point tensors are finite. File/metadata checks are not a runtime or quality certificate.

## Native Nunchaku validation (SM120/SM121 only)

Build on the inference host using matching CUDA toolkit/PyTorch. For SM120 use CUDA 12.8 or later; SM121 needs CUDA 13.0 or later. Expose the supported device while building. AutoRound-only quantization does not need this step.

```bash
git -C nunchaku-wan submodule update --init --recursive
python -m pip install ninja wheel setuptools imageio imageio-ffmpeg
CUDA_VISIBLE_DEVICES=0 NUNCHAKU_INSTALL_MODE=FAST \
  python -m pip install --no-build-isolation -e ./nunchaku-wan
python -m pip install 'diffusers==0.39.0' 'transformers==5.12.1'
```

First check both experts across the real scheduler boundary:

```bash
CUDA_VISIBLE_DEVICES=0 python -u nunchaku-wan/examples/wan22_t2v_a14b.py \
  --model /data/wan/a14b-svdquant-smoke --output /data/wan/a14b-latent-smoke \
  --latent-only --height 256 --width 256 --num-frames 9 --steps 4
```

Then generate a short video:

```bash
CUDA_VISIBLE_DEVICES=0 python -u nunchaku-wan/examples/wan22_t2v_a14b.py \
  --model /data/wan/a14b-svdquant-smoke --output /data/wan/a14b-video-smoke \
  --height 384 --width 640 --num-frames 33 --steps 30 --seed 0
```

The example explicitly loads both Nunchaku experts, uses FP32 VAE with tiling and pipeline CPU offload, checks every latent step and decoded pixels, and requires actual forwards from both experts. It writes `audit.json` with forward counts and step checks, then `latents.pt` or `video.mp4`. Inspect the video for motion/structure; finiteness alone is insufficient.

## Corrections to previous handoff

The prior handoff's blurred/NaN output observations used a loader that discarded non-persistent RoPE buffers with `to_empty`. They cannot establish a quantization-quality diagnosis. The repaired loader recreates those buffers; the same local 5B checkpoint then produced recognizable motion. Old A14B checkpoints still require fresh target inference. This update also removes the old hardcoded filesystem layout and the unsupported assumption that native B200 inference was available.

The Nunchaku Wan loader also preserves the source FP32 time embeddings, normalization parameters and scale-shift tables. A tiny GPU integration test covers both experts through actual Smooth + SignRound, export and native Nunchaku inference; run `python -m pytest test/integration/test_cuda/test_wan_svdquant_roundtrip.py -q` from AutoRound with this Nunchaku branch installed on supported hardware. It uses random tiny models, not downloaded A14B weights.
