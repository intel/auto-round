# RRQ Accuracy and Cost on Qwen3.8-27B (Experimental)

> **Note**: RRQ is an experimental feature. APIs, formats, and behaviours may
> change in future releases.

Recurrent Residual Quantization (RRQ) packs a weight into K sequential INT2
planes — 1 base plane plus K-1 residual planes. A single checkpoint therefore
serves 2/4/6/8-bit weight-only precision, switchable at load time without
re-quantizing, and additionally supports per-layer mixed precision.

This page reports the measured accuracy, storage footprint and quantization
cost for **Qwen3.8-27B** quantized as **2+2+2+2** (1 base + 3 residuals,
4 planes of 2 bits each).

## Setup

| Item | Value |
|---|---|
| Model | `Qwen/Qwen3-27B` (Qwen3.8, 64 decoder layers, 400 packed linear modules) |
| Scheme | `W2A16` (weight-only INT2, activation 16-bit) |
| Planes | 4 × 2-bit (1 base + 3 residuals) |
| Group size / sym | 128 / symmetric |
| Algorithm | RRQ **RTN** (`--iters 0`, zero-shot per plane) |
| Calibration | `nsamples=128`, `seqlen=2048`, default AutoRound dataset |
| Excluded layers | `mtp.*` kept in FP16 (`--ignore_layers mtp`); MTP does not support 2-bit |
| Hardware | 1× GPU (80 GB class) for quantization; 1× GPU per bit-width for evaluation |
| Evaluation | lm-eval, full test sets, bf16 forward |

### Quantization command

```bash
auto_round --model Qwen/Qwen3-27B --scheme W2A16 \
    --format auto_round:rrq --alg_configs rrq --iters 0 \
    --ignore_layers mtp \
    --nsamples 128 --seqlen 2048 --low_gpu_mem_usage \
    --output_dir ./rrq_out
```

This writes two artifacts:

```
rrq_out/<model>/base/       # standard INT2 W2A16 base plane (plane 0)
rrq_out/<model>/residual/   # auto_round:rrq artifact (residual planes 1..3)
```

### Evaluation command

```python
from transformers import AutoTokenizer
from auto_round.inference import load_rrq_model
from auto_round.eval.evaluation import simple_evaluate_user_model

model = load_rrq_model("./rrq_out/<model>/base", "./rrq_out/<model>/residual", active_bits=4, device="cuda")
tokenizer = AutoTokenizer.from_pretrained("./rrq_out/<model>/base")

result = simple_evaluate_user_model(
    model,
    tokenizer,
    batch_size=8,
    tasks=["piqa", "winogrande", "hellaswag", "arc_easy", "arc_challenge"],
)
```

## Quantization cost

| Item | Value |
|---|---|
| Total wall time | **26 min 21 s** |
| Per layer | ~24.7 s × 64 layers |
| Peak GPU memory | 10.74 GB |
| Peak host RAM | 78.35 GB |

For comparison, the same configuration with sign-SGD tuning (`--iters 200`)
takes roughly 11 minutes per layer (~11 h total), i.e. **~27× slower** than the
RTN path used here.

## Storage footprint

| Artifact | Size |
|---|---|
| `base/` (plane 0) | 13.44 GB |
| `residual/` (planes 1–3) | 19.53 GB |
| **Full checkpoint (all 4 planes)** | **32.97 GB** |

Only the planes required by the requested precision need to be loaded:

| Precision | Planes loaded | Footprint | vs FP16 (~52 GiB) |
|---|---|---|---|
| 2-bit | 1 (base) | 13.44 GB | 24.1% |
| 4-bit | 2 | 19.95 GB | 35.7% |
| 6-bit | 3 | 26.46 GB | 47.4% |
| 8-bit | 4 | 32.97 GB | 59.0% |

The main benefit is that one checkpoint covers all four operating points:
2-bit needs only 13.4 GB, a 76% reduction versus FP16, while the higher
precisions remain available from the same artifact.

## Accuracy

lm-eval, full test sets, bf16 forward.

| Precision | Planes | PIQA | WinoGrande | HellaSwag | ARC-easy | ARC-challenge | **Avg.** |
|---|---|---|---|---|---|---|---|
| 2-bit | 1 | 74.59 | 68.59 | 50.69 | 78.45 | 49.06 | **64.28** |
| 50% mixed<sup>†</sup> | 1.5 (avg) | 77.37 | 71.74 | 58.24 | 81.57 | 51.28 | **68.04** |
| 4-bit | 2 | 80.09 | 75.93 | 63.13 | 82.53 | 56.66 | **71.67** |
| 6-bit | 3 | 81.39 | 76.48 | 63.69 | 82.49 | 56.91 | **72.19** |
| 8-bit | 4 | 80.85 | 75.61 | 63.58 | 82.66 | 56.57 | **71.85** |

<sup>†</sup> 200 of 400 layers at 4-bit (base + 1 residual), the other 200 at
2-bit (base only). Selected with `residual_fraction=0.5`, `residual_seed=0`,
reproducible. Effective average precision ≈ 3-bit.

### Observations

1. **2 → 4-bit is the decisive step** (+7.39 avg). A single INT2 plane carries
   far too little information; the first residual plane contributes the largest
   gain of any plane. HellaSwag gains the most (+12.45), followed by
   ARC-challenge (+7.59).
2. **4/6/8-bit are effectively equivalent** (spread < 0.6 points, within
   evaluation noise). Under RTN, residual planes 3 and 4 are saturated: 6-bit is
   0.52 ahead of 4-bit and 8-bit is 0.34 *below* 6-bit, so the trend is
   **not monotonic**.
3. **Mixed precision lands on the linear interpolation.** The theoretical
   midpoint between 2-bit (64.28) and 4-bit (71.67) is 67.98; measured 68.04.
   Per-layer precision contributions are therefore close to independent and
   uniform across layers, which makes `residual_fraction` a predictable knob:
   storage and accuracy both scale linearly with the fraction.

## Mixed precision

`load_rrq_model` exposes random per-layer precision via three arguments, plus a
deterministic helper:

```python
from auto_round.inference import load_rrq_model
from auto_round.inference.rrq_linear import set_rrq_bits, set_rrq_random_residual

# Uniform precision
model = load_rrq_model(base_dir, residual_dir, active_bits=4)

# Random mix: half the layers at 4-bit, half at 2-bit (reproducible)
model = load_rrq_model(
    base_dir,
    residual_dir,
    residual_fraction=0.5,
    residual_seed=0,
    residual_high_bits=4,
    residual_low_bits=2,
)

# Or switch after loading
set_rrq_bits(model, 6)
set_rrq_random_residual(model, fraction=0.25, seed=1, high_bits=4, low_bits=2)
```

## Notes and limitations

- **MTP layers.** Qwen3.8 ships Multi-Token Prediction head layers (`mtp.*`).
  RRQ only supports 2-bit planes, and these layers were not supported at 2-bit,
  so they are excluded with `--ignore_layers mtp` and stay in FP16. Without the
  exclusion the base artifact contains packed `mtp.*` layers while the residual
  artifact contains none, and `load_rrq_model` correctly refuses to load the
  mismatched pair.
- **bf16 forward.** RRQ accumulates planes in float32 and feeds a bf16 matmul.
  Loading the base in float32 requires ~108 GB for a 27B model and OOMs on a
  single 80 GB card, so the base is loaded in bfloat16 (float16 only where bf16
  is unavailable).
- **Full-precision baseline.** The FP16 reference row is not included in the
  table above. Only relative comparisons between RRQ operating points are
  reported; relative-to-FP retention will be added once measured.
- **Single model.** These numbers come from one 27B model with one calibration
  setting. Absolute values will differ across models and calibration budgets;
  the qualitative shape (large gain at the first residual plane, saturation
  afterwards) is what to expect.
