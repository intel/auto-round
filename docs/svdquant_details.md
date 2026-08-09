# SVDQuant — Details (Experimental)

> **Experimental feature.** The currently validated end-to-end path targets
> MXFP4 quantization of Diffusers FLUX transformers. The core SVDQuant
> preprocessor is model-agnostic, but the `svdquant_nunchaku` format is
> runtime-loadable only for supported model adapters; FLUX is the first
> supported adapter.

This document describes AutoRound's SVDQuant preprocessing, its interaction
with RTN and SignRound, activation-aware smoothing, and Nunchaku export.

SVDQuant decomposes a linear weight into a low-rank branch and a quantized
residual branch:

```text
W ~= Q(R) + U @ V

Linear(x) ~= QuantizedLinear(x, Q(R)) + Linear(Linear(x, V), U)
```

The low-rank branch uses a configured BF16, FP16, or FP32 dtype during
quantization, while the larger residual branch is quantized to MXFP4. The
Nunchaku export materializes the low-rank tensors as BF16 by default. This
keeps a small, high-precision correction for weight directions that are
expensive to represent with four-bit values.

---

## Prerequisites

Use an isolated environment with a CUDA-enabled PyTorch build appropriate for
the target GPU. When working from the AutoRound source tree, install AutoRound
and the diffusion dependencies before quantization:

```bash
pip install --no-build-isolation -e .
pip install -r test/test_cuda/requirements_diffusion.txt
```

Nunchaku is not needed to generate the quantized artifact. It is needed only
for inference and must be built or installed with MXFP4 support against a
compatible PyTorch/CUDA ABI. Verify the inference environment before loading
an exported pipeline:

```bash
python -c "import torch, nunchaku; print(torch.__version__, torch.version.cuda, nunchaku.__file__)"
```

The local diffusion-caption dataset is a tab-separated file with `id` and
`caption` columns:

```text
id\tcaption
0\tA photo of a cat
1\tA city street at night
```

---

## 1. Algorithm composition

SVDQuant is a structural preprocessor, not a terminal quantizer. It must be
followed by exactly one block quantizer:

```text
SVDQuantTransform -> RTN or SignRound -> output format
```

The CLI expresses this as an ordered algorithm list:

| CLI | Pipeline | Calibration |
|-----|----------|-------------|
| `--algorithm svdquant,rtn` | SVDQuant + RTN | Data-free when smooth is disabled |
| `--algorithm svdquant,auto_round` | SVDQuant + SignRound | Uses SignRound calibration |

The SVDQuant residual outer iteration always uses deployment-compatible RTN
QDQ. This is independent of the terminal quantizer: selecting SignRound means
SignRound optimizes the residual linears after SVDQuant has created them; it
does not replace RTN inside the residual decomposition loop.

### 1.1 Processing flow

```text
Original projection weights
        |
        +-- optional smooth search using bounded calibration calls
        |
        +-- group related FLUX projections
        |     (for example Q/K/V share one down factor)
        |
        +-- truncated SVD and residual outer iteration
        |
        +-- replace nn.Linear with SVDQuantLinear
        |       +-- MXFP4 residual_linear
        |       +-- BF16/FP16 lora_down
        |       +-- BF16/FP16 lora_up
        |       +-- optional smooth factor
        |
        +-- terminal RTN or SignRound quantization
        |
        +-- optional Nunchaku FLUX export
```

Related projections are decomposed as one stacked matrix where required by
the runtime format. For example, FLUX Q/K/V projections share one low-rank
down factor and retain separate up factors.

---

## 2. Residual decomposition

With `residual_iters=1`, one grouped weight matrix `W` is processed as follows:

1. Compute a rank-`r` truncated SVD of `W`.
2. Materialize the low-rank factors in `low_rank_dtype`.
3. Form the residual `R = W - U @ V`.
4. Wrap `R`, `U`, and `V` in `SVDQuantLinear`.
5. Let the terminal RTN or SignRound stage quantize the residual linear.

With `residual_iters > 1`, SVDQuant additionally runs a deployment-compatible
RTN QDQ outer loop before the terminal quantizer. Each iteration computes a
new low-rank decomposition from `W - Q(R_previous)`, forms and QDQs the new
residual, and evaluates `Q(R) + U @ V`. This alternates low-rank fitting and
residual quantization. `--enable-svdquant-residual-early-stop` stops after the
error first becomes worse than the best accepted candidate.

The selection metric depends on smooth mode:

- **Smooth disabled:** weight reconstruction squared error.
- **Smooth enabled:** calibration output squared error for the projection
  group, including deployment-compatible residual and activation QDQ.

Increasing `residual_iters` increases repeated SVD and QDQ work. It does not
change the terminal SignRound `--iters`; the two iteration counts control
different optimization loops.

---

## 3. Activation-aware smooth search

Smooth search is disabled by default. Enable it with:

```bash
--enable-svdquant-smooth
```

For each FLUX projection group, AutoRound collects activation channel spans
and weight channel spans, then evaluates Alpha/Beta smooth factors. For
`G = --svdquant-smooth-num-grids`, the candidate set is:

```text
(alpha, beta) = (0, 0)
(alpha, 0) for alpha in {1/G, ..., (G-1)/G}
(alpha, 1-alpha) for the same alpha values
```

The total candidate count is `1 + 2 * (G - 1)`. The default `G=20` evaluates
39 candidates per smooth group. Every candidate is scored against cached
floating-point outputs, and the lowest finite output error is selected.

`--svdquant-smooth-max-calibration-calls` bounds the retained calls per smooth
group using reservoir sampling. Calls that are not retained are not copied to
the CPU cache. This bounds smooth-search memory, but the terminal SignRound
calibration flow still has its own sample and block-input requirements.

Selected factor statistics are logged:

```text
scale_min, scale_max, scale_ratio, below_1e-3, above_20
```

Extreme-factor warnings are diagnostic. They do not modify or clamp the
selected factor.

---

## 4. Configuration reference

`SVDQuantConfig` is defined in
`auto_round/algorithms/transforms/svdquant/config.py`.

| Python field | CLI option | Default | Description |
|--------------|------------|---------|-------------|
| `rank` | `--svdquant-rank` | `32` | Rank of the high-precision correction branch; Nunchaku export requires a positive rank. |
| `smooth_enabled` | `--enable-svdquant-smooth` | `False` | Enable activation-aware Alpha/Beta search. |
| `smooth_num_grids` | `--svdquant-smooth-num-grids` | `20` | Grid resolution; produces `1 + 2 * (G - 1)` candidates. |
| `smooth_max_calibration_calls` | `--svdquant-smooth-max-calibration-calls` | `128` | Maximum retained smooth calibration calls per group. |
| `smooth_eps` | Python API only | `1e-6` | Positive floor used while constructing factors. |
| `residual_iters` | `--svdquant-residual-iters` | `1` | Alternating low-rank/residual iterations. |
| `residual_early_stop` | `--enable-svdquant-residual-early-stop` | `False` | Stop when the selected error no longer improves. |
| `low_rank_dtype` | `--svdquant-low-rank-dtype` | `"bf16"` | Low-rank factor dtype: BF16, FP16, or FP32 aliases. |
| `target_modules` | `--svdquant-target-modules` | `None` | Comma-separated module-name substrings to transform. |
| `exclude_modules` | `--svdquant-exclude-modules` | `None` | Comma-separated module-name substrings to keep untransformed. |
| `model_adapter` | `--svdquant-model-adapter` | `None` in Python; `"auto"` in CLI | Export mapping: `auto`, `flux`, or `identity`; `None` is resolved as auto. |

All SVDQuant CLI options use hyphens. Existing shared AutoRound options retain
their existing spelling.

---

## 5. CLI usage

### 5.1 No-smooth SVDQuant + RTN

This is the fastest data-free path:

```bash
CUDA_VISIBLE_DEVICES=0 auto-round \
  --model /path/to/FLUX.1-dev \
  --model_dtype bf16 \
  --scheme MXFP4 \
  --algorithm svdquant,rtn \
  --disable_opt_rtn \
  --iters 0 \
  --nblocks 1 \
  --svdquant-rank 32 \
  --svdquant-residual-iters 20 \
  --enable-svdquant-residual-early-stop \
  --svdquant-low-rank-dtype bf16 \
  --svdquant-model-adapter flux \
  --format svdquant_nunchaku \
  --device 0 \
  --low_gpu_mem_usage \
  --disable_low_cpu_mem_usage \
  --output_dir ./flux-dev-mxfp4-svdquant-rtn
```

### 5.2 Smooth SVDQuant + SignRound

This path performs smooth search first, then runs SignRound on the residual
linears:

```bash
CUDA_VISIBLE_DEVICES=0 auto-round \
  --model /path/to/FLUX.1-dev \
  --model_dtype bf16 \
  --scheme MXFP4 \
  --algorithm svdquant,auto_round \
  --iters 200 \
  --nblocks 1 \
  --nsamples 128 \
  --batch_size 1 \
  --num_inference_steps 50 \
  --dataset /path/to/coco2017-captions.tsv \
  --svdquant-rank 32 \
  --enable-svdquant-smooth \
  --svdquant-smooth-num-grids 20 \
  --svdquant-smooth-max-calibration-calls 128 \
  --svdquant-residual-iters 20 \
  --enable-svdquant-residual-early-stop \
  --svdquant-low-rank-dtype bf16 \
  --svdquant-model-adapter flux \
  --format svdquant_nunchaku \
  --device 0 \
  --low_gpu_mem_usage \
  --disable_low_cpu_mem_usage \
  --output_dir ./flux-dev-mxfp4-svdquant-signround
```

This is a quality-oriented example, not a universal memory-safe preset.
Reduce `nsamples`, diffusion steps, smooth calls, or SignRound iterations for
workflow validation before running a full calibration.

---

## 6. Python API

Pass SVDQuant and one terminal quantizer through `alg_configs`:

```python
from auto_round import AutoRound
from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.algorithms.transforms.svdquant import SVDQuantConfig

autoround = AutoRound(
    "/path/to/FLUX.1-dev",
    scheme="MXFP4",
    model_dtype="bf16",
    alg_configs=[
        SVDQuantConfig(
            rank=32,
            smooth_enabled=False,
            residual_iters=20,
            residual_early_stop=True,
            model_adapter="flux",
        ),
        RTNConfig(disable_opt_rtn=True),
    ],
    device_map=0,
    low_gpu_mem_usage=True,
    low_cpu_mem_usage=False,
)

autoround.quantize_and_save(
    "./flux-dev-mxfp4-svdquant-rtn",
    format="svdquant_nunchaku",
)
```

The example above covers the data-free RTN path. Use the CLI workflow in
section 5.2 for smooth SignRound calibration so the caption dataset, batch
size, and diffusion-step arguments are explicit.

---

## 7. FLUX mapping and Nunchaku export

`--format svdquant_nunchaku` currently requires:

- `scheme=MXFP4`
- E2M1 weight and activation data
- group size 32
- symmetric weight and activation quantization
- dynamic activation quantization
- a supported runtime model adapter (`flux` or auto-detected FLUX)

The FLUX adapter transforms only runtime-supported projections. AdaNorm
linears are exported as W4A16 group-64 tensors, while RMSNorm weights and
top-level transformer tensors are exported as BF16. Non-transformer pipeline
components are saved through Diffusers in the dtype in which they were loaded;
the CLI examples load them with `--model_dtype bf16`.

The output is a self-contained Diffusers pipeline:

```text
output/
  model_index.json
  scheduler/
  tokenizer/
  tokenizer_2/
  text_encoder/
  text_encoder_2/
  vae/
  transformer/
    config.json
    diffusion_pytorch_model.safetensors
```

The transformer safetensors metadata contains `config`,
`quantization_config`, `model_class`, `comfy_config`, and `format`.
AutoRound's exporter does not import Nunchaku; Nunchaku is required only when
loading the exported model for inference.

Export may take several minutes after quantization because fused FLUX records
are recomposed and decomposed to the configured common rank on CPU.

---

## 8. Nunchaku inference

After the prerequisite Nunchaku environment check succeeds, load the pipeline
directly:

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "./flux-dev-mxfp4-svdquant-signround",
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
pipe.enable_model_cpu_offload()

generator = torch.Generator(device="cuda").manual_seed(12345)
image = pipe(
    "A cat holding a sign that says Hello world",
    num_inference_steps=20,
    guidance_scale=3.5,
    generator=generator,
    height=512,
    width=512,
).images[0]
image.save("flux-svdquant-mxfp4.png")
```

`model_index.json` points the transformer entry to
`nunchaku.NunchakuFluxTransformer2dModel`, so loading fails if a compatible
Nunchaku package is not installed.

---

## 9. Memory and performance guidance

- Start with a small end-to-end smoke configuration before a full calibration.
- `--low_gpu_mem_usage` enables component-level CPU offload for diffusion
  calibration rather than placing the complete pipeline on one GPU.
- `--svdquant-smooth-max-calibration-calls` bounds only the retained smooth
  evaluation pool; it does not replace `--nsamples` or SignRound calibration.
- More smooth candidates increase group replay time approximately linearly.
- More residual iterations repeat SVD and QDQ; early stop can reduce this work
  only after the error becomes worse.
- More SignRound `--iters` increases terminal optimization time and is separate
  from smooth and residual iteration costs.
- CPU memory, GPU memory, and runtime depend strongly on calibration samples,
  diffusion steps, image size, and model placement.

---

## 10. Current limitations

- Runtime-loadable export currently supports FLUX through the FLUX adapter.
- The Nunchaku format is restricted to deployable MXFP4 E2M1 group-32 schemes.
- SVDQuant currently requires `nblocks=1`.
- Smooth calibration is output-aware and can be expensive because every
  Alpha/Beta candidate replays retained group calls.
- Full-quality settings are hardware- and dataset-dependent; the examples are
  starting points rather than guaranteed optimal presets.
- Export uses one common low-rank rank in Nunchaku metadata, so fused records
  are recomputed at that rank instead of preserving a sum of source ranks.
- Generated-image quality must be evaluated over a representative prompt set;
  a single smoke image validates loading and numerical stability only.
