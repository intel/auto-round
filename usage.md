# Experimental FP4/INT4 BKM

This note records the reproducible Llama-3.1-8B-Instruct HellaSwag comparison for the experimental group-size-16 schemes.

## Schemes

| CLI scheme | Format | Description |
| --- | --- | --- |
| `exp-fp4` | FP4 | Uses the `NVFP4_E5M3` preset |
| `exp-nvsint4` | INT4 | Signed INT4 with E4M3 scale |
| `exp-nvint4` | INT4 | Signed INT4 without zero-point, with E5M3 scale |
| `exp-nvint4-zp` | INT4-ZP | Legacy asymmetric INT4 with per-group zero-points and E5M3 scale |
| `EXP-FP4-BF16S` / `EXP-FP4-FP16S` | FP4 | FP4 with BF16 / FP16 weight scales |
| `EXP-INT4-BF16S` / `EXP-INT4-FP16S` | INT4 | Symmetric, zero-point-free INT4 (`[-7, 7]`) with BF16 / FP16 weight scales |
| `EXP-INT4-ZP-BF16S` / `EXP-INT4-ZP-FP16S` | INT4-ZP | Asymmetric INT4 with zero-point and BF16 / FP16 weight scales; preserves results from the earlier EXP-INT4 implementation |
| `EXP-SINT4-BF16S` / `EXP-SINT4-FP16S` | INT4 | Signed INT4 with BF16 / FP16 weight scales |

The six scale variants use independent quantizers; only weight scale precision changes. They are intended for fake-format experiments and model accuracy comparisons.
Use the lowercase `exp-*` names above for the existing FP8-scale baselines. They are distinct from the older `exp-sint4` and `exp-int4` names in earlier notes.

## Run

Set `SCHEME` to one of the schemes above and `MODE` to `rtn`, `opt-rtn`, or `tuned`. Select an idle GPU from 0-6; GPU 7 is reserved. With `CUDA_VISIBLE_DEVICES` set, `--device_map 0` refers to the selected visible GPU.

```bash
MODEL_PATH=/models/Llama-3.1-8B-Instruct
GPU_ID=0
SCHEME=exp-nvint4
MODE=tuned
RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUTPUT_ROOT="/data0/xinhe/autoround-exp-hellaswag/runs/${RUN_ID}"

case "${MODE}" in
  rtn)
    MODE_ARGS=(--iters 0 --disable_opt_rtn)
    ;;
  opt-rtn)
    MODE_ARGS=(--iters 0)
    ;;
  tuned)
    MODE_ARGS=(--iters 200 --enable_minmax_tuning)
    ;;
  *)
    printf 'Unknown MODE: %s\n' "${MODE}" >&2
    exit 2
    ;;
esac

CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m auto_round "${MODEL_PATH}" \
  --scheme "${SCHEME}" \
  --format fake \
  --tasks hellaswag \
  --eval_bs 32 \
  --nsamples 128 \
  --seqlen 2048 \
  --dataset NeelNanda/pile-10k \
  --seed 42 \
  --device_map 0 \
  --enable_deterministic_algorithms \
  --output_dir "${OUTPUT_ROOT}/${SCHEME}-${MODE}/model" \
  "${MODE_ARGS[@]}"
```

The modes differ as follows:

| Mode | Arguments | Behavior |
| --- | --- | --- |
| RTN | `--iters 0 --disable_opt_rtn` | Round-to-nearest baseline; disables the optimized RTN pass |
| opt-RTN | `--iters 0` | Uses the optimized RTN pass without iterative tuning |
| tuned | `--iters 200 --enable_minmax_tuning` | Runs 200 tuning iterations with min/max tuning enabled |

The recorded run used calibration data `NeelNanda/pile-10k` (128 samples, sequence length 2048), HellaSwag evaluation batch size 32, and seed 42. Each row below reports `acc` and `acc_norm`; the approximate 95% confidence intervals from the evaluation logs are ±0.49 and ±0.41 percentage points, respectively.

## Results

Llama-3.1-8B-Instruct, HellaSwag:

| Scheme | RTN acc / acc_norm | opt-RTN acc / acc_norm | tuned acc / acc_norm |
| --- | ---: | ---: | ---: |
| `exp-fp4` | 58.48% / 77.67% | 58.70% / 78.13% | 58.11% / 77.33% |
| `exp-nvsint4` | 58.40% / 77.79% | 58.44% / 77.56% | 58.33% / 77.81% |
| `exp-nvint4-zp` | 59.34% / 78.99% | 59.31% / 78.90% | 59.23% / 78.92% |

In this run, tuned did not consistently outperform RTN or opt-RTN. The small differences should be interpreted in light of the evaluation uncertainty; this single run is not sufficient to establish a reliable ranking.

## Recommended Validation Matrix

The following models and tasks are recommended for broader validation. The Llama-3.1-8B-Instruct/HellaSwag cell is covered by the results above; the other cells are proposed follow-up runs, not completed measurements.

| Model | GSM8K | MMLU | HellaSwag | PIQA |
| --- | --- | --- | --- | --- |
| Phi-4 | Recommended | Recommended | Recommended | Recommended |
| Llama-3.1-8B-Instruct | Recommended | Recommended | Completed above | Recommended |
| Qwen2.5-7B-Instruct | Recommended | Recommended | Recommended | Recommended |
| Qwen3-32B | Recommended | Recommended | Recommended | Recommended |

For `--tasks`, use the corresponding lm-eval task names (`gsm8k`, `mmlu`, `hellaswag`, and `piqa`) supported by the installed lm-eval version. Before launching a full matrix, verify task availability with that version and choose an evaluation batch size that fits the selected GPU. Keep model revision, task configuration, quantization mode, calibration data, and seed fixed when comparing schemes.

## Reference

BKM code: [intel/auto-round, exp-int4 branch](https://github.com/intel/auto-round/tree/exp-int4)