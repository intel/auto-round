# SVDQuant (Experimental)

SVDQuant splits a linear weight into a quantized residual branch and a small
floating-point low-rank branch:

```text
W ~= Q(R) + U @ V
Linear(x) ~= QuantizedLinear(x, Q(R)) + Linear(Linear(x, V), U)
```

This feature follows the decomposition proposed in [SVDQuant: Absorbing
Outliers by Low-Rank Components for 4-Bit Diffusion
Models](https://arxiv.org/abs/2411.05007). AutoRound can combine the transform
with RTN or SignRound and export a Nunchaku-loadable MXFP4 FLUX pipeline.

> The end-to-end export path has been validated with FLUX.1-dev only. Other
> models using compatible Diffusers FLUX block classes are experimental and
> produce a warning.

## Quick start

Install AutoRound and its diffusion dependencies in an isolated environment:

```bash
pip install --no-build-isolation -e .
pip install -r test/test_cuda/requirements_diffusion.txt
```

### Default RTN workflow

The recommended starting point uses rank 32, one residual iteration, no
smooth search, and plain RTN:

```bash
CUDA_VISIBLE_DEVICES=0 auto-round-rtn \
  --model /path/to/FLUX.1-dev \
  --model_dtype bf16 \
  --scheme MXFP4 \
  --algorithm svdquant \
  --format svdquant_nunchaku \
  --device 0 \
  --output_dir ./flux-dev-mxfp4-svdquant-rtn
```

When `svdquant` is selected and `--format` is omitted, the CLI also defaults to
`svdquant_nunchaku`.

### Default SignRound workflow

Use SignRound as the terminal quantizer by selecting `auto_round`:

```bash
CUDA_VISIBLE_DEVICES=0 auto-round \
  --model /path/to/FLUX.1-dev \
  --model_dtype bf16 \
  --scheme MXFP4 \
  --algorithm svdquant,auto_round \
  --format svdquant_nunchaku \
  --dataset /path/to/coco2017-captions.tsv \
  --batch_size 1 \
  --device 0 \
  --low_gpu_mem_usage \
  --output_dir ./flux-dev-mxfp4-svdquant-signround
```

The caption dataset is a tab-separated file with `id` and `caption` columns:

```text
id	caption
0	A photo of a cat
1	A city street at night
```

The COCO2017 calibration captions are available from
[`changwangss/coco2017`](https://huggingface.co/datasets/changwangss/coco2017).

### Optional quality settings

Smooth search and additional residual iterations are opt-in:

```bash
--enable-svdquant-smooth \
--svdquant-smooth-num-grids 20 \
--svdquant-smooth-max-calibration-calls 128 \
--svdquant-residual-iters 20 \
--enable-svdquant-residual-early-stop
```

For diffusion calibration, `--nsamples` controls prompts and
`--calib_num_inference_steps` controls the number of inference steps requested when the scheduler builds its
native short calibration schedule. Higher-order schedulers may expand these into more internal timesteps.
Start small before running a full quality configuration.

## Processing flow

SVDQuant is a structural preprocessor and must run before one terminal
quantizer:

```text
Original linear weights
  -> optional activation-aware smooth search
  -> group related projections
  -> low-rank decomposition and residual iteration
  -> replace nn.Linear with SVDQuantLinear
       -> quantized residual_linear
       -> floating-point lora_down
       -> floating-point lora_up
       -> optional input smooth factor
  -> RTN or SignRound
  -> Nunchaku export
```

FLUX Q/K/V projections are grouped so they share one low-rank down factor and
retain separate up factors, matching the runtime layout.

## Residual iteration

With the default `residual_iters=1`:

1. Compute a rank-limited decomposition of the grouped weight `W`.
2. Materialize `U` and `V` in `low_rank_dtype`.
3. Form `R = W - U @ V`.
4. Let the terminal RTN or SignRound stage quantize the residual linear.

With `residual_iters > 1`, each outer iteration alternates between fitting the
low-rank branch to `W - Q(R_previous)` and applying deployment-compatible RTN
QDQ to the new residual. Early stop retains the best candidate and stops when
the objective becomes worse.

The objective is weight reconstruction error without smooth search and
calibration output error with smooth search. Residual iterations are separate
from SignRound `--iters`.

## Smooth search

For `G = --svdquant-smooth-num-grids`, the candidates are:

```text
(alpha, beta) = (0, 0)
(alpha, 0) for alpha in {1/G, ..., (G-1)/G}
(alpha, 1-alpha) for the same alpha values
```

The default `G=20` produces 39 candidates. Each candidate constructs
`x_span**alpha / w_span**beta` and is scored against floating-point group
outputs. The lowest finite output error wins.

`--svdquant-smooth-max-calibration-calls` uses reservoir sampling to bound the
retained calls per group. Calls that are not selected are not copied into the
smooth CPU cache. This limit does not replace SignRound calibration samples.

## Configuration

| Python field | CLI option | Default | Meaning |
|---|---|---:|---|
| `rank` | `--svdquant-rank` | `32` | Low-rank branch rank. |
| `smooth_enabled` | `--enable-svdquant-smooth` | `False` | Enable Alpha/Beta search. |
| `smooth_num_grids` | `--svdquant-smooth-num-grids` | `20` | Candidate grid resolution. |
| `smooth_max_calibration_calls` | `--svdquant-smooth-max-calibration-calls` | `128` | Retained calls per group. |
| `residual_iters` | `--svdquant-residual-iters` | `1` | Residual outer iterations. |
| `residual_early_stop` | `--enable-svdquant-residual-early-stop` | `False` | Stop after the objective worsens. |
| `low_rank_dtype` | `--svdquant-low-rank-dtype` | `bf16` | BF16, FP16, or FP32 factors. |
| `target_modules` | `--svdquant-target-modules` | `None` | Included module-name substrings. |
| `exclude_modules` | `--svdquant-exclude-modules` | `None` | Excluded module-name substrings. |
| `model_adapter` | `--svdquant-model-adapter` | `auto` | Runtime grouping/export adapter. |

## Python API

```python
from auto_round import AutoRound
from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.algorithms.transforms.svdquant import SVDQuantConfig

autoround = AutoRound(
    "/path/to/FLUX.1-dev",
    scheme="MXFP4",
    model_dtype="bf16",
    alg_configs=[
        SVDQuantConfig(rank=32, model_adapter="flux"),
        RTNConfig(disable_opt_rtn=True),
    ],
    device_map=0,
    low_gpu_mem_usage=True,
)
autoround.quantize_and_save(
    "./flux-dev-mxfp4-svdquant-rtn",
    format="svdquant_nunchaku",
)
```

## Export and inference

The `svdquant_nunchaku` format currently requires MXFP4 E2M1 W4A4, group size
32, symmetric weights/activations, dynamic activation quantization, and a
supported runtime adapter. Nunchaku is not imported during quantization; it is
required only for inference.

Pre-quantized FLUX.1-dev checkpoints are available for
[smooth SVDQuant + SignRound](https://huggingface.co/changwangss/smooth_svdquant_signround)
and
[no-smooth SVDQuant + SignRound](https://huggingface.co/changwangss/nosmooth_svdquant_signround).

The output is a self-contained Diffusers pipeline. Load its Nunchaku transformer
explicitly, then pass it to Diffusers:

```python
import torch
from diffusers import FluxPipeline
from nunchaku import NunchakuFluxTransformer2dModel

model_dir = "./flux-dev-mxfp4-svdquant-signround"
transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    f"{model_dir}/transformer/diffusion_pytorch_model.safetensors",
    torch_dtype=torch.bfloat16,
    precision="mxfp4",
    device="cuda:0",
)
pipe = FluxPipeline.from_pretrained(
    model_dir,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
pipe.enable_model_cpu_offload()

image = pipe(
    "A cat holding a sign that says Hello world",
    num_inference_steps=20,
    guidance_scale=3.5,
    generator=torch.Generator(device="cuda").manual_seed(12345),
    height=512,
    width=512,
).images[0]
image.save("flux-svdquant-mxfp4.png")
```

## Limitations

- Runtime-loadable export is currently validated only for FLUX.1-dev.
- SVDQuant requires `nblocks=1`.
- Smooth search replays every retained call for every Alpha/Beta candidate.
- More residual iterations repeat decomposition and QDQ work.
- A smoke image validates loading and numerical stability, not dataset-level
  generation quality.
