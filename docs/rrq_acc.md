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

### Generating residuals for an existing INT2 base

If you already have a quantized INT2 model, `generate_rrq_residual` builds the
residual planes without re-quantizing the base:

```python
from auto_round.export import generate_rrq_residual

generate_rrq_residual(
    "./rrq_out/<model>/base",
    raw_model="Qwen/Qwen3-27B",
    output_dir="./rrq_out/<model>/residual",
    group_size=128,
    sym=True,
    device="cuda",
    num_workers=32,
)
```

The layers are independent, so this parallelizes perfectly. `num_workers`
spreads them over processes and `device` decides where the quantization math
runs; a bare `"cuda"` expands to every visible GPU and jobs are assigned
round-robin. For Qwen3.8-27B (400 layers, 8 GPUs, `num_workers=32`) this takes
**5 min 10 s**, versus many hours serially on CPU. Each worker pins itself to
one GPU, and packing stays on CPU because the packing kernel is CPU-only.

```bash
# multi-GPU without writing Python
python -c "
from auto_round.export import generate_rrq_residual
generate_rrq_residual('base/', raw_model='Qwen/Qwen3-27B',
                      output_dir='residual/', group_size=128, sym=True,
                      device='cuda', num_workers=32)"
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

## Base plane quality: RTN vs. AutoRound

The numbers above use RTN for every plane (`--iters 0`). The base plane is by
far the most important one, so it is worth spending sign-SGD tuning on it alone
and generating the residual planes incrementally:

1. Quantize the base plane with full AutoRound tuning (`--iters 1000`), optionally
   sharded across GPUs.
2. Feed that INT2 base to `generate_rrq_residual` to obtain the residual planes.

That split is exactly what `generate_rrq_residual` exists for. Note that the
residual planes still use RTN — the same optimized RTN scale search as the
one-pass path — so the only difference between the two configurations is how
the base plane was produced.

| Configuration | Base plane | Residual planes | Base cost | Residual cost |
|---|---|---|---|---|
| **A** | RTN (`iters=0`) | RTN | incl. below | — |
| **B** | AutoRound (`iters=1000`) | RTN | 6 h 09 m (4 GPUs) | 5 m 10 s (8 GPUs) |

Configuration A takes 26 min 21 s in total; B costs an additional 6 h because of
the sign-SGD tuning of the base plane.

### Accuracy comparison

lm-eval, full test sets, bf16 forward, identical evaluation harness.

| Precision | A: all RTN | B: AutoRound base + RTN residual | **Δ** |
|---|---|---|---|
| 2-bit | 64.28 | **68.02** | **+3.74** |
| 4-bit | **71.67** | 72.02 | +0.35 |
| 6-bit | **72.19** | 72.03 | −0.16 |
| 8-bit | **71.85** | 71.79 | −0.06 |

Configuration B detail:

| Precision | PIQA | WinoGrande | HellaSwag | ARC-easy | ARC-challenge |
|---|---|---|---|---|---|
| 2-bit | 77.97 | 71.35 | 55.60 | 81.86 | 53.33 |
| 4-bit | 80.47 | 76.16 | 63.38 | 82.91 | 57.17 |
| 6-bit | 81.18 | 74.90 | 63.67 | 83.16 | 57.25 |
| 8-bit | 81.07 | 75.37 | 63.75 | 82.20 | 56.57 |

Per-task delta (B − A):

| Precision | PIQA | WinoGrande | HellaSwag | ARC-easy | ARC-challenge | Avg. |
|---|---|---|---|---|---|---|
| 2-bit | +3.38 | +2.76 | **+4.91** | +3.41 | +4.27 | **+3.74** |
| 4-bit | +0.38 | +0.23 | +0.25 | +0.38 | +0.51 | +0.35 |
| 6-bit | −0.21 | −1.58 | −0.02 | +0.67 | +0.34 | −0.16 |
| 8-bit | +0.22 | −0.24 | +0.17 | −0.46 | −0.00 | −0.06 |

At 2-bit the gain is spread across every task, and the two weakest 2-bit tasks
gain the most (HellaSwag +4.91, ARC-challenge +4.27). From 4-bit up the deltas
are small and sign-inconsistent, i.e. noise.

Configuration B precision ladder:

| Step | Avg. delta |
|---|---|
| 2 → 4-bit | **+4.00** (decisive) |
| 4 → 6-bit | +0.01 (saturated) |
| 6 → 8-bit | −0.24 (not monotonic) |

B saturates one step earlier than A: a better base plane means the first residual
plane closes almost the whole gap, so planes 3 and 4 are pure noise.

### Observations

1. **Tuning the base plane only pays off at 2-bit** (+3.74), where it buys
   ~3.7 accuracy points for 6 extra hours of quantization time. HellaSwag gains
   the most (+4.91), then ARC-challenge (+4.27).
2. **At 4-bit and above the two configurations are equivalent** (spread ≤ 0.35,
   i.e. evaluation noise). Once one residual plane is present the base plane is
   no longer the bottleneck, so the expensive tuning is wasted: the extra
   residual planes absorb the error the base leaves behind.
3. **A tuned 2-bit base is competitive with a 3-bit mixed-precision checkpoint.**
   Compare B at 2-bit (68.02, 13.44 GB) with A at `residual_fraction=0.5`
   (68.04, 19.95 GB): the averages are identical while B uses 6.5 GB less, at
   the cost of a different capability profile — HellaSwag is 2.64 lower but
   ARC-challenge is 2.05 higher.
4. **Practical guidance.** If the target deployment is 2-bit, tune the base with
   AutoRound and generate the residuals incrementally. If the target is 4-bit or
   higher — or if the checkpoint must serve several widths — quantize everything
   with RTN (`--iters 0`) and get all four operating points in half an hour.

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

## Inference performance

`RRQLinear` keeps the packed INT2 planes resident and dequantizes them on every
forward. That is what makes a single checkpoint switchable between
2/4/6/8-bit, but it is slow: each layer rebuilds its weight tensor for every
token.

Calling `materialize_rrq(model)` once converts every RRQ layer into a dense
`nn.Linear`, paying the dequantization a single time and then running ordinary
dense GEMMs. The trade-off is memory and flexibility: the dense copy needs about
2 bytes per parameter (~54 GB for a 27B model in bf16) and the precision is
frozen at whatever was active when it was called.

```python
from auto_round import load_rrq_model, materialize_rrq

model = load_rrq_model(base_dir, residual_dir, active_bits=4, device="cuda")
# ... validate accuracy at 2/4/6/8-bit if you like ...
materialize_rrq(model)  # freeze the active precision, dense inference
```

Measured on Qwen3.8-27B (8-bit, single 80 GB card):

| Path | Speed | Peak VRAM |
|---|---|---|
| `load_rrq_model` only (dequant per forward) | 0.31 tok/s | 30.9 GB |
| after `materialize_rrq` | **15.08 tok/s** | 51.1 GB |

That is a **48x** speedup, numerically equivalent (same float32 accumulation,
rounded to the layer dtype).

`load_rrq_model` also caches dequantized weights automatically when the whole
model fits: it enables the cache only if one dense copy of *every* layer fits in
the free device memory, and otherwise logs how many GiB it would need. A
partial cache cannot help because each forward visits every layer, so the
uncached ones dominate.

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
