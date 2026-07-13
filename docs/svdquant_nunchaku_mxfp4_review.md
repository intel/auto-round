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

The outer loop currently supports RTN only. SignRound remains usable as the normal
downstream residual quantizer, but SignRound-aware outer iteration is not implemented.

## Smoothing and Calibration

`smooth_enabled=False` is a strict no-smoothing mode:

- no activation calibration hook is installed;
- no activation maxima are required;
- smooth tensors are identity;
- the FLUX command below does not need a calibration dataset or cache.

This is the mode used for the validated rank-32 artifact.

## FLUX Export Command

The repository contains a reproducible blockwise runner. It keeps only one FLUX block
on the decomposition GPU and moves completed blocks back to CPU.

```bash
cd /home/user2/data/xixi/auto-round-svdquant
source /home/user2/data/xixi/.venv/bin/activate
export UV_CACHE_DIR=/home/user2/data/xixi/.cache/uv
export CUDA_VISIBLE_DEVICES=0

python -u scripts/quantize_flux_svdquant_nunchaku.py \
  --model /home/user2/data/xixi/FLUX.1-dev/transformer \
  --output /home/user2/data/xixi/flux.1-dev-autoround-mxfp4-r32-nosmooth.safetensors \
  --rank 32 \
  --device cuda:0 2>&1 | tee /home/user2/data/xixi/flux-autoround-full.log
```

The output is written to a temporary safetensors file and atomically renamed only
after a complete export. A failure does not leave a partial model at the final path.

The high-level language-model CLI is not used here. A standalone Diffusers
`FluxTransformer2DModel` is otherwise classified through the language-model path and
requests a tokenizer/text calibration dataset. The blockwise runner is the validated
no-calibration path for a transformer-only FLUX checkpoint.

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

The validated AutoRound artifact is:

```text
/home/user2/data/xixi/flux.1-dev-autoround-mxfp4-r32-nosmooth.safetensors
size: 6.4 GiB
keys: 2604
sha256: 075452388daa609777ce2b27abbce45aa5fa4a15b398119477e040c96286df52
```

Compared with the known-good DeepCompressor artifact:

```text
/home/user2/data/xixi/flux.1-dev-mxfp4-lowrank32-nosmooth.safetensors
```

the key sets, tensor shapes, and tensor dtypes match exactly. The AutoRound artifact
contains no unpacked `residual.weight` or float residual copy.

## Kernel Versus QDQ Gate

The external validation script requires the local Nunchaku and DeepCompressor source
trees only to decode and run the reference kernel:

```bash
export PYTHONPATH=/home/user2/data/xixi/deepcompressor:/home/user2/data/xixi/nunchaku
export MODEL=/home/user2/data/xixi/flux.1-dev-autoround-mxfp4-r32-nosmooth.safetensors
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
| `transformer_blocks.0.mlp_fc1` | 0.109740 | 0.112090 | 0.994051 |
| `transformer_blocks.0.qkv_proj` | 0.131848 | 0.109775 | 0.989319 |
| `single_transformer_blocks.0.out_proj` | 0.113137 | 0.110838 | 0.993579 |

These values are in the same range as the known-good MXFP4 artifact and show no
packing, scale, or metadata layout mismatch.

## Nunchaku Load and Image Gate

Load the transformer directly:

```python
import torch
from nunchaku import NunchakuFluxTransformer2dModel

transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    "/home/user2/data/xixi/flux.1-dev-autoround-mxfp4-r32-nosmooth.safetensors",
    torch_dtype=torch.bfloat16,
    precision="mxfp4",
    device="cuda:0",
)
```

Generate the validated 20-step image:

```bash
export PYTHONPATH=/home/user2/data/xixi/nunchaku
export MXFP4_MODEL=/home/user2/data/xixi/flux.1-dev-autoround-mxfp4-r32-nosmooth.safetensors
export CUDA_VISIBLE_DEVICES=0

python /home/user2/data/xixi/generate_flux_compare.py \
  --mode mxfp4 \
  --steps 20 \
  --height 512 \
  --width 512 \
  --offload \
  --out /home/user2/data/xixi/flux-dev-autoround-mxfp4-r32-nosmooth-20steps.png
```

The generated image is coherent and not noise. It contains a clear cat holding a
readable "hello world" sign. Its SHA256 is
`329053d804dea15a8bdb13198bd6653553016cfa53e877fcd55c174c7ff66edf`.

## Test Coverage and Dependency Boundary

The focused CPU suite covers configuration, residual iteration, codec behavior,
export validation, format registration, W4A16 packing, and FLUX mapping. Runtime
Nunchaku checks remain external so that AutoRound does not acquire an inference
runtime dependency.

Final focused result:

```text
272 passed, 4 warnings in 11.19s
```

Audit command:

```bash
rg -n "(^| )import (deepcompressor|nunchaku)|from (deepcompressor|nunchaku)" \
  auto_round/algorithms/transforms/svdquant \
  auto_round/export/svdquant_*
```

Expected result: no matches.

## Limitations

- The validated no-smoothing runner is transformer-specific at the orchestration
  layer; the SVDQuant core, codec, exporter, and adapter interface remain generic.
- RTN is the only implemented residual outer-loop method.
- The tested Nunchaku MXFP4 kernel branch targets SM120/SM121. B200 is SM100 and
  requires separate Nunchaku build/kernel support before runtime validation.
- The artifact path name does not contain Nunchaku's expected precision token, so
  Nunchaku prints a filename heuristic warning when `precision="mxfp4"` is explicit.
  Loading and inference are unaffected.
