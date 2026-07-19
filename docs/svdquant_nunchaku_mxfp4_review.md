# SVDQuant RTN MXFP4 and Nunchaku Export

This document records the implemented behavior and the FLUX.1-dev validation of the
`wangchang/svdquant` branch. AutoRound production code and tests do not import
DeepCompressor or Nunchaku. Those projects are used only as external reference and
runtime validation tools.

## Implemented Architecture

The implementation separates model-independent quantization from runtime-specific
serialization:

- `SVDQuantTransform` rewrites selected `torch.nn.Linear` modules into a residual
  linear plus BF16 low-rank branch.
- Residual QDQ reuses AutoRound's registered `mx_fp` RTN quantization function.
  There is no second MXFP4 numerical implementation in the transform.
- The MXFP4 codec packs E2M1 values and UE8M0 scales in AutoRound-owned code.
- `SVDQuantModelAdapter` keeps model naming, projection fusion, and extra tensors
  outside the generic exporter.
- `FluxSVDQuantNunchakuAdapter` is the first model adapter. It emits the exact FLUX
  onefile schema consumed by Nunchaku.

The reconstructed linear is:

```text
W_effective = (W_residual + W_up @ W_down) * smooth
```

With smoothing disabled, `smooth` and `smooth_orig` are exact identity vectors.

## Residual Iteration

`SVDQuantConfig.residual_iters` defaults to `1`.

- `residual_iters=1`: one SVD decomposition. The residual is quantized by the normal
  downstream AutoRound flow or during export.
- `residual_iters>1`: each additional outer iteration applies AutoRound RTN QDQ to
  the current residual, computes the remaining error, and updates a fixed-rank SVD
  branch.

Residual outer iteration uses RTN QDQ by design for both RTN and SignRound pipelines.
`--algorithm` independently selects the final downstream residual quantizer: RTN or
SignRound. SignRound is not an outer-iteration method.

## Smoothing and Calibration

`SVDQuantConfig.smooth_enabled` and the CLI both default to `False`. The CLI enables
smoothing only when `--enable_svdquant_smooth` is present.

The enabled mode is a low-rank-aware output grid search, not a fixed-alpha formula.
`--svdquant_smooth_num_grids 20` evaluates 39 ordered `(alpha, beta)` candidates per
search group. Each candidate uses one shared truncated SVD, deployment-dtype low-rank
rounding, residual RTN QDQ, and the registered parent module's output MSE. Flux Q/K/V
projections share one input scale and one low-rank down factor; single-stream Q/K/V and
`proj_mlp` are searched together because they consume the same normalized input.

The search captures all samples already scheduled by the current AutoRound block
calibration context and replays the existing block batches. Its buffers live only for
the current block and are released after scale selection or on failure. The selected
runtime tensor is `smooth = 1 / scale`. Flux export preserves shared QKV down/smooth
tensors directly, and splits single-stream `proj_out` residual/down/smooth tensors by
input columns without replacing the selected scale with identity.

`smooth_enabled=False` is a strict no-smoothing mode:

- no activation calibration hook is installed;
- no activation maxima are required;
- smooth tensors are identity;
- the FLUX command below does not need a calibration dataset or cache.

This is the mode used for the validated rank-32 artifact.

For a quality-oriented smooth run, add:

```text
--nsamples 32
--batch_size 1
--dataset coco2014
--num_inference_steps 10
--enable_svdquant_smooth
--svdquant_smooth_num_grids 20
```

For diffusion calibration this schedules roughly `nsamples * num_inference_steps`
block inputs, so the command above evaluates about 320 inputs. Use a smaller grid,
sample count, and denoising-step count only for smoke validation. Smooth search performs
`2 * num_grids - 1` temporary grouped decompositions in addition to the final
`svdquant_residual_iters` decomposition loop, so it is substantially slower than the
no-smooth command.

## FLUX Export Command

The standard CLI loads the complete Diffusers pipeline, quantizes only its Transformer,
and exports a self-contained hybrid pipeline. Auxiliary components remain BF16 and the
Transformer is a Nunchaku MXFP4 onefile:

```bash
cd /home/user2/data/xixi/auto-round-svdquant
source /home/user2/data/xixi/torch213-cu130-env/.venv/bin/activate
export UV_CACHE_DIR=/home/user2/data/xixi/torch213-cu130-env/.uv-cache
export CUDA_VISIBLE_DEVICES=0

python -u -m auto_round \
  --model /home/user2/data/xixi/FLUX.1-dev \
  --model_dtype bf16 \
  --scheme MXFP4 \
  --algorithm rtn \
  --iters 0 \
  --disable_opt_rtn \
  --enable_svdquant \
  --svdquant_rank 32 \
  --svdquant_residual_iters 1 \
  --svdquant_residual_quant_method rtn \
  --svdquant_model_adapter flux \
  --format svdquant_nunchaku \
  --device 0 \
  --disable_low_cpu_mem_usage \
  --output_dir /home/user2/data/xixi/autoround-cli-flux-mxfp4-r32-nosmooth-pipeline \
  2>&1 | tee /home/user2/data/xixi/autoround-cli-flux-mxfp4-r32-nosmooth-pipeline.log
```

The output is a standard Diffusers component tree. Its packed Transformer is written to:

```text
/home/user2/data/xixi/autoround-cli-flux-mxfp4-r32-nosmooth-pipeline/transformer/diffusion_pytorch_model.safetensors
```

The existing AutoRound diffusion backend loads the pipeline and selects only
`pipe.transformer` for quantization. Smoothing is disabled by default; omit
`--enable_svdquant_smooth` for a strictly data-free run, and do not pass a calibration
dataset or cache. The exporter saves VAE, text encoders, tokenizers, and scheduler through
their normal `save_pretrained` paths, but never saves BF16 Transformer weights. On hosts with
enough RAM, `--disable_low_cpu_mem_usage` keeps the working model in memory and avoids
creating a large disk-offload workspace. The validated run used about 64 GB peak RSS,
4 GB peak VRAM, and took about 51 minutes including FLUX fusion/export on one RTX 5090D.

For exporter debugging, `scripts/quantize_flux_svdquant_nunchaku.py` remains available
as a lower-level blockwise runner. New reproductions should use the standard CLI.

## Nunchaku Onefile Layout

The standard FLUX.1-dev artifact contains 2,604 tensors:

- eight W4A4 groups per double-stream block;
- four W4A4 groups per single-stream block;
- W4A16 AdaNorm tensors;
- RMSNorm weights;
- exactly 20 top-level BF16 tensors.

Representative shapes are:

```text
transformer_blocks.0.qkv_proj.qweight        int8   [9216, 1536]
transformer_blocks.0.norm1.linear.qweight    int32  [4608, 1536]
single_transformer_blocks.0.out_proj.qweight int8   [3072, 1536]
x_embedder.weight                            bf16   [3072, 64]
```

Metadata includes:

```json
{
  "model_class": "NunchakuFluxTransformer2dModel",
  "format": "pt",
  "quantization_config": {
    "method": "svdquant",
    "rank": 32,
    "weight": {"dtype": "fp4_e2m1_all", "scale_dtype": "ue8m0", "group_size": 32},
    "activation": {"dtype": "fp4_e2m1_all", "scale_dtype": "ue8m0", "group_size": 32}
  }
}
```

QKV and single-block `proj_out` fusion reconstruct the effective source weights first,
then compute a new configured-rank SVD. It does not concatenate LoRA ranks.

## Static Artifact Verification

The following transformer-only artifact was generated and validated before the workspace
cleanup; the path is retained here as a historical reproducibility record and is no longer
present in the current workspace:

```text
/home/user2/data/xixi/autoround-cli-flux-mxfp4-r32-nosmooth/transformer/diffusion_pytorch_model.safetensors
size: 6.4 GiB
keys: 2604
sha256: 445b481e934679ea128bf0a78ebf83b45da19e9579fc289928884a1edbe76ea3
```

It was compared with the historical known-good DeepCompressor artifact:

```text
/home/user2/data/xixi/flux.1-dev-mxfp4-lowrank32-nosmooth.safetensors
```

the key sets, tensor shapes, and tensor dtypes match exactly. The AutoRound artifact
contains no unpacked `residual.weight` or float residual copy.

## Kernel Versus QDQ Gate

After a new pipeline export, the external validation script requires the local Nunchaku
and DeepCompressor source trees only to decode and run the reference kernel:

```bash
export PYTHONPATH=/home/user2/data/xixi/deepcompressor:/home/user2/data/xixi/nunchaku
export MODEL=/home/user2/data/xixi/autoround-cli-flux-mxfp4-r32-nosmooth-pipeline/transformer/diffusion_pytorch_model.safetensors
export CUDA_VISIBLE_DEVICES=0

PREFIX=transformer_blocks.0.mlp_fc1 \
  python /home/user2/data/xixi/check_mxfp4_kernel_vs_qdq.py
PREFIX=transformer_blocks.0.qkv_proj \
  python /home/user2/data/xixi/check_mxfp4_kernel_vs_qdq.py
PREFIX=single_transformer_blocks.0.out_proj \
  python /home/user2/data/xixi/check_mxfp4_kernel_vs_qdq.py
```

Results on RTX 5090D, SM120:

| Layer | Relative MAE | Max relative error | Correlation |
| --- | ---: | ---: | ---: |
| `transformer_blocks.0.mlp_fc1` (CLI, M=16) | 0.110075 | 0.112626 | 0.993992 |
| `transformer_blocks.0.qkv_proj` | 0.131848 | 0.109775 | 0.989319 |
| `single_transformer_blocks.0.out_proj` | 0.113137 | 0.110838 | 0.993579 |

These values are in the same range as the known-good MXFP4 artifact and show no
packing, scale, or metadata layout mismatch.

## Nunchaku Load and Image Gate

Load a newly exported self-contained pipeline directly. No BF16 base pipeline or manual
Transformer replacement is needed:

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "/home/user2/data/xixi/autoround-cli-flux-mxfp4-r32-nosmooth-pipeline",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
image = pipe(
    "A cat holding a sign that says Hello world",
    num_inference_steps=20,
    generator=torch.Generator(device="cpu").manual_seed(0),
).images[0]
image.save("flux-mxfp4.png")
```

The root `model_index.json` entry is
`["nunchaku", "NunchakuFluxTransformer2dModel"]`. Diffusers therefore calls Nunchaku
with the `transformer/` directory, and Nunchaku resolves its
`diffusion_pytorch_model.safetensors` onefile.

The following direct onefile path remains useful for isolated kernel debugging and records
the previously validated transformer-only artifact:

```python
import torch
from nunchaku import NunchakuFluxTransformer2dModel

transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    "/home/user2/data/xixi/autoround-cli-flux-mxfp4-r32-nosmooth/transformer/diffusion_pytorch_model.safetensors",
    torch_dtype=torch.bfloat16,
    precision="mxfp4",
    device="cuda:0",
)
```

Generate the validated 20-step image:

```bash
export PYTHONPATH=/home/user2/data/xixi/nunchaku
export MXFP4_MODEL=/home/user2/data/xixi/autoround-cli-flux-mxfp4-r32-nosmooth/transformer/diffusion_pytorch_model.safetensors
export CUDA_VISIBLE_DEVICES=0

python /home/user2/data/xixi/generate_flux_compare.py \
  --mode mxfp4 \
  --steps 20 \
  --height 512 \
  --width 512 \
  --offload \
  --out /home/user2/data/xixi/flux-dev-autoround-cli-mxfp4-r32-nosmooth-torch213-cu130-20steps.png
```

The historical generated image was coherent and not noise. It contained a clear cat
holding a readable "HeLLo world" sign, validating the transformer-only CLI onefile with
Nunchaku on the torch 2.13 + cu130 branch. The image has since been removed. SHA256:
`f98c58dae0f7fc985fc894f12d6fb64b50590aca7bb4e21957ae922b38245ce4`.

For the residual-iters=2 full-pipeline run, use:

```bash
bash /home/user2/data/xixi/run_autoround_flux_mxfp4_r32_r2_rtn.sh
```

That runner quantizes the complete source pipeline and then calls
`torch213-cu130-env/generate_flux_mxfp4_pipeline.py`. The smoke script loads only the
export directory through `FluxPipeline.from_pretrained`; it does not load the original
BF16 pipeline or inject a Transformer manually.

## Test Coverage and Dependency Boundary

The focused CPU suite covers configuration, residual iteration, codec behavior,
export validation, format registration, W4A16 packing, and FLUX mapping. Runtime
Nunchaku checks remain external so that AutoRound does not acquire an inference
runtime dependency.

Final focused results:

```text
278 SVDQuant/core/export tests passed, 4 warnings
17 targeted CLI tests passed, 2 warnings
```

Audit command:

```bash
rg -n "(^| )import (deepcompressor|nunchaku)|from (deepcompressor|nunchaku)" \
  auto_round/algorithms/transforms/svdquant \
  auto_round/export/svdquant_*
```

Expected result: no matches.

## Limitations

- The CLI loader is generic for standalone Diffusers transformer classes. FLUX naming
  and projection fusion remain isolated in the explicit `flux` model adapter.
- Residual outer iteration is fixed to RTN QDQ by design; RTN and SignRound remain
  available as downstream final quantizers.
- The tested Nunchaku MXFP4 kernel branch targets SM120/SM121. B200 is SM100 and
  requires separate Nunchaku build/kernel support before runtime validation.
- The artifact path name does not contain Nunchaku's expected precision token, so
  Nunchaku prints a filename heuristic warning when `precision="mxfp4"` is explicit.
  Loading and inference are unaffected.
