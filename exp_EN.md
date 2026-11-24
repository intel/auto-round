# FP4/INT4 RTN Quantization Accuracy Comparison

## Summary

This experiment compares four models using a BF16 baseline and group-size-16 quantization schemes. All quantized results use RTN and **do not use calibration data**. The `EXP-UINT4-ZP-BF16/FP16` results originated from an earlier matrix run whose directories and logs used `EXP-INT4-ZP-BF16S/FP16S`; the names are corrected here to reflect the actual unsigned codes and zero-point. Legacy names remain accepted by the CLI and resolve to the canonical names.

A macro-average is computed with equal weight for GSM8K strict-match, MMLU acc, HellaSwag acc_norm, and PIQA acc_norm. `EXP-UINT4-ZP-UE5M3` has the highest average among the quantization schemes in this experiment (80.09), close to BF16 (80.44). Relative performance varies across models, so this does not imply it is best for every model or task.

| Scheme | Weight / activation | Four-model average |
| --- | --- | ---: |
| BF16 | BF16 / BF16 | 80.44 |
| `EXP-UINT4-ZP-UE5M3`† | W4 / A16 | 80.09 |
| `EXP-UINT4-ZP-BF16`† | W4 / A16 | 79.92 |
| `EXP-INT4-BF16`† | W4 / A16 | 79.80 |
| `EXP-UINT4-ZP-FP16`† | W4 / A16 | 79.75 |
| `EXP-INT4-FP16`† | W4 / A16 | 79.72 |
| `EXP-INT4-UBF16`† | W4 / A16 | 79.33 |
| `EXP-INT4-UFP16`† | W4 / A16 | 79.25 |
| `EXP-INT4-E4M3`† | W4 / A16 | 79.13 |
| `EXP-INT4-UE5M3`†‡ | W4 / A16 | 78.91 |
| `EXP-FP4-UE5M3` | W4 / A4 | 78.72 |
| `EXP-FP4-BF16`† | W4 / A4 | 78.35 |
| `EXP-FP4-FP16`† | W4 / A4 | 78.35 |

Note: the table above contains historical results. At the time, INT4 schemes were weight-only quantized (WOQ, W4A16), while FP4 schemes used their historical configuration; these scores do not represent the current all-scheme W4A4 run, and the two families should not be treated as having the same activation setup. `†` marks results whose activation setup differs from the current preset; `‡` marks results that also used the fixed `[-8, 7]` clipping logic. See "Affected Historical Results". Results under the current definitions are in "Full Results · W4A4" below.

BF16 and FP16 scale variants have nearly identical average results, with no consistent dtype advantage. All results are from a single evaluation run; small differences should not be interpreted as statistically significant.

### Affected Historical Results

Two issues have been fixed: INT4 schemes previously quantized weights only (`act_bits=16`), and the BF16/FP16-scale FP4 variants quantized activations with `nvfp4_v2` instead of the weight format; `EXP-INT4-UE5M3` previously used `[-8, 7]` and clipped a positive maximum to 7. All EXP schemes now quantize activations with the same format as the weights (W4A4), and `EXP-INT4-UE5M3` uses symmetric `[-7, 7]` again. Markers in the historical tables mean:

| Scheme | Historical configuration | Current definition | Marker |
| --- | --- | --- | --- |
| `EXP-INT4-E4M3`, `EXP-UINT4-ZP-UE5M3`, `EXP-INT4-UBF16/UFP16`, `EXP-UINT4-ZP-BF16/FP16`, `EXP-INT4-BF16/FP16` | W4A16 | W4A4, activations use the weight format | † |
| `EXP-INT4-UE5M3` | W4A16; codes `[-8, 7]`, positive maximum clipped to 7 | W4A4; codes `[-7, 7]` | †‡ |
| `EXP-FP4-BF16/FP16` | W4A4; activations use `nvfp4_v2` (UE5M3 scale) | W4A4; activations use the same BF16/FP16-scale FP4 as weights | † |
| `EXP-FP4-UE5M3` | W4A4; activations use `nvfp4_v2` | Unchanged | None |

Marked historical scores do not represent the current presets and are kept for traceability only. Use "Full Results · W4A4", or re-evaluate, to compare current definitions.

### Historical Four-Model Average (Including Historical Llama Configuration)

The two tables below summarize the historical multi-model results above; they are not four-model results for the current unified W4A4 setup. The Llama rows also use historical configurations: INT4 is WOQ (W4A16), while FP4 uses its historical setup. These are distinct from the current-preset Llama scores in "Full Results · W4A4". The average is the arithmetic mean of GSM8K strict-match, MMLU acc, HellaSwag acc_norm, and PIQA acc_norm. We first calculate it for each model, then take an equal-weight average over the four models. This summary does not imply equal task difficulty or statistical significance.

#### By Model

| Model | BF16 | `EXP-FP4-UE5M3` | `EXP-INT4-E4M3`† | `EXP-UINT4-ZP-UE5M3`† | `EXP-INT4-UE5M3`†‡ | `EXP-FP4-BF16`† | `EXP-FP4-FP16`† | `EXP-UINT4-ZP-BF16`† | `EXP-UINT4-ZP-FP16`† | `EXP-INT4-UBF16`† | `EXP-INT4-UFP16`† | `EXP-INT4-BF16`† | `EXP-INT4-FP16`† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-3.1-8B-Instruct | 76.24 | 72.00 | 72.78 | 75.51 | 72.88 | 71.30 | 71.08 | 75.43 | 75.12 | 74.53 | 74.66 | 74.59 | 74.51 |
| Phi-4 | 82.67 | 81.91 | 81.98 | 82.33 | 81.68 | 81.60 | 81.55 | 82.25 | 82.21 | 82.04 | 81.98 | 82.21 | 82.10 |
| Qwen2.5-7B-Instruct | 78.53 | 77.66 | 78.09 | 78.99 | 77.48 | 76.96 | 77.34 | 78.30 | 78.03 | 77.96 | 77.74 | 79.18 | 79.09 |
| Qwen3-32B | 84.32 | 83.31 | 83.69 | 83.54 | 83.61 | 83.55 | 83.43 | 83.69 | 83.64 | 82.77 | 82.63 | 83.22 | 83.19 |

#### By Scheme

| Scheme | Llama-3.1-8B-Instruct | Phi-4 | Qwen2.5-7B-Instruct | Qwen3-32B | Four-model average |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 76.24 | 82.67 | 78.53 | 84.32 | 80.44 |
| `EXP-FP4-UE5M3` | 72.00 | 81.91 | 77.66 | 83.31 | 78.72 |
| `EXP-INT4-E4M3`† | 72.78 | 81.98 | 78.09 | 83.69 | 79.13 |
| `EXP-UINT4-ZP-UE5M3`† | 75.51 | 82.33 | 78.99 | 83.54 | 80.09 |
| `EXP-INT4-UE5M3`†‡ | 72.88 | 81.68 | 77.48 | 83.61 | 78.91 |
| `EXP-FP4-BF16`† | 71.30 | 81.60 | 76.96 | 83.55 | 78.35 |
| `EXP-FP4-FP16`† | 71.08 | 81.55 | 77.34 | 83.43 | 78.35 |
| `EXP-UINT4-ZP-BF16`† | 75.43 | 82.25 | 78.30 | 83.69 | 79.92 |
| `EXP-UINT4-ZP-FP16`† | 75.12 | 82.21 | 78.03 | 83.64 | 79.75 |
| `EXP-INT4-UBF16`† | 74.53 | 82.04 | 77.96 | 82.77 | 79.33 |
| `EXP-INT4-UFP16`† | 74.66 | 81.98 | 77.74 | 82.63 | 79.25 |
| `EXP-INT4-BF16`† | 74.59 | 82.21 | 79.18 | 83.22 | 79.80 |
| `EXP-INT4-FP16`† | 74.51 | 82.10 | 79.09 | 83.19 | 79.72 |

## Details

### Experimental Setup

The models are Llama-3.1-8B-Instruct, Phi-4, Qwen2.5-7B-Instruct, and Qwen3-32B. Tasks are GSM8K, MMLU, HellaSwag, and PIQA. GSM8K uses 5-shot greedy decoding with a 1024-token generation limit. Evaluation dtype is BF16. Throughout this document, results use GSM8K strict-match, MMLU acc, HellaSwag acc_norm, and PIQA acc_norm.

| CLI scheme | Format | Quantization method (current definition; activations use the weight format, W4A4) | Theoretical weight BPW |
| --- | --- | --- | ---: |
| BF16 | BF16 | Unquantized baseline | 16 |
| `EXP-FP4-UE5M3` | FP4 | E2M1, per-group UE5M3 scale, no global scale | 4.5 |
| `EXP-INT4-E4M3` | INT4 | Codes `[-8, 7]`, per-group signed E4M3 scale; the scale may be negative, so the larger-magnitude side maps to -8 | 4.5 |
| `EXP-INT4-UE5M3` | INT4 | Codes `[-7, 7]`, per-group UE5M3 scale; the scale is nonnegative and ±7 is used symmetrically | 4.5 |
| `EXP-UINT4-ZP-UE5M3` | UINT4-ZP | Unsigned codes `[0, 15]`, per-group UE5M3 scale and zero-point | 4.75* |
| `EXP-FP4-BF16` / `EXP-FP4-FP16` | FP4 | E2M1, per-group BF16 / FP16 scale | 5.0 |
| `EXP-INT4-UBF16` / `EXP-INT4-UFP16` | INT4 | Codes `[-7, 7]`, per-group nonnegative BF16 / FP16 scale | 5.0 |
| `EXP-UINT4-ZP-BF16` / `EXP-UINT4-ZP-FP16` | UINT4-ZP | Unsigned codes `[0, 15]`, per-group BF16 / FP16 scale and zero-point | 5.25* |
| `EXP-INT4-BF16` / `EXP-INT4-FP16` | INT4 | Codes `[-8, 7]`, per-group signed BF16 / FP16 scale; the larger-magnitude side maps to -8 | 5.0 |

BPW assumes group size 16 and compact metadata: `BPW = weight_bits + scale_bits / 16 + zero_point_bits / 16`. E4M3/UE5M3 scales are counted as 8 bits and BF16/FP16 scales as 16 bits. Schemes marked `*` assume zero-points in `[0, 15]` are compactly stored in 4 bits. Alignment, padding, and other container overhead are excluded. These are theoretical values; actual fake-format artifact sizes may differ.

`EXP-UINT4-ZP-UE5M3` uses unsigned codes `[0, 15]` and a zero-point; the dequantized weights may still be negative. `EXP-FP4-UE5M3` is the `NVFP4_E5M3` preset, unlike `NVFP4`, which uses E4M3 scales and an FP32 global scale. All EXP schemes now quantize activations with the same format as the weights (W4A4, group size 16); see "Affected Historical Results" for how historical runs differ. BF16/FP16 identify the floating-point scale dtype. The `U` in `UBF16/UFP16` indicates nonnegative scale values for the symmetric INT4 variant; it does not mean BF16/FP16 is an unsigned floating-point dtype. These are CLI label changes only; quantization behavior and serialized internal dtype identifiers are unchanged.

Legacy-name mapping: `EXP-FP4` -> `EXP-FP4-UE5M3`; `EXP-NVSINT4` -> `EXP-INT4-E4M3`; `EXP-NVINT4` -> `EXP-INT4-UE5M3`; `EXP-NVINT4-ZP` -> `EXP-UINT4-ZP-UE5M3`; `EXP-FP4-BF16S/FP16S` -> `EXP-FP4-BF16/FP16`; `EXP-SINT4-BF16S/FP16S` -> `EXP-INT4-BF16/FP16`; symmetric `[-7, 7]` `EXP-INT4-BF16S/FP16S` -> `EXP-INT4-UBF16/UFP16`; and `EXP-INT4-ZP-BF16S/FP16S` -> `EXP-UINT4-ZP-BF16/FP16`. Historical logs are not rewritten; use these mappings when inspecting old run directories.

All quantization uses RTN (`--iters 0 --disable_opt_rtn`) and does not read or consume calibration data. Some reproduction scripts retain arguments such as `--dataset` and `--nsamples`, but the RTN path does not use this data. Random seed: 42.

### Reproduction

Set `MODEL_PATH` to the model directory. For the BF16 baseline, point `EVAL_MODEL` to the original model. For a quantized scheme, run the quantization command first and point `EVAL_MODEL` to its output. `SCHEME` can be any quantized scheme in the table above.

```bash
MODEL_PATH=/path/to/model
SCHEME=EXP-INT4-UE5M3
OUTPUT_DIR=/path/to/output/${SCHEME}

python -m auto_round "$MODEL_PATH" \
  --scheme "$SCHEME" \
  --format fake \
  --nsamples 128 \
  --seqlen 2048 \
  --batch_size 8 \
  --dataset NeelNanda/pile-10k \
  --seed 42 \
  --device_map 0 \
  --enable_deterministic_algorithms \
  --disable_opt_rtn \
  --iters 0 \
  --output_dir "$OUTPUT_DIR"
EVAL_MODEL="$OUTPUT_DIR"
EVAL_BS=8
# Use EVAL_BS=32 for Llama-3.1-8B-Instruct.
auto-round eval "$EVAL_MODEL" \
  --tasks mmlu,hellaswag,piqa \
  --device_map 0 \
  --seed 42 \
  --eval_bs "$EVAL_BS" \
  --eval_model_dtype bfloat16 \
  --eval_task_by_task
auto-round eval "$EVAL_MODEL" \
  --tasks gsm8k \
  --device_map 0 \
  --seed 42 \
  --eval_bs 8 \
  --num_fewshot 5 \
  --eval_model_dtype bfloat16 \
  --eval_gen_kwargs max_gen_toks=1024,do_sample=False
```

### Full Results

Scores are percentages. The reported metrics are GSM8K strict-match, MMLU acc, HellaSwag acc_norm, and PIQA acc_norm. Both tables share the same BF16 baseline. All results are from a single evaluation run.

#### WOQ (W4A16)

Historical INT4 results with weight-only quantization (`act_bits=16`); these correspond to the `†` INT4 scores in the Summary and the historical averages. `‡` marks results that also used the fixed `[-8, 7]` clipping logic.

| Model | Scheme | GSM8K strict-match | MMLU acc | HellaSwag acc_norm | PIQA acc_norm |
| --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B-Instruct | BF16 | 76.72 | 68.07 | 79.17 | 81.01 |
| Llama-3.1-8B-Instruct | `EXP-INT4-E4M3` | 68.99 | 64.19 | 77.78 | 80.14 |
| Llama-3.1-8B-Instruct | `EXP-UINT4-ZP-UE5M3` | 75.82 | 66.42 | 79.00 | 80.79 |
| Llama-3.1-8B-Instruct | `EXP-INT4-UE5M3`‡ | 68.31 | 64.88 | 78.09 | 80.25 |
| Llama-3.1-8B-Instruct | `EXP-INT4-UBF16` | 71.49 | 66.56 | 79.39 | 80.69 |
| Llama-3.1-8B-Instruct | `EXP-INT4-UFP16` | 71.72 | 66.74 | 79.29 | 80.90 |
| Llama-3.1-8B-Instruct | `EXP-UINT4-ZP-BF16` | 74.91 | 67.11 | 79.08 | 80.63 |
| Llama-3.1-8B-Instruct | `EXP-UINT4-ZP-FP16` | 74.15 | 67.17 | 78.97 | 80.20 |
| Llama-3.1-8B-Instruct | `EXP-INT4-BF16` | 71.72 | 66.91 | 78.97 | 80.74 |
| Llama-3.1-8B-Instruct | `EXP-INT4-FP16` | 71.49 | 66.85 | 79.00 | 80.69 |
| Phi-4 | BF16 | 90.67 | 76.86 | 81.98 | 81.18 |
| Phi-4 | `EXP-INT4-E4M3` | 89.39 | 75.89 | 81.57 | 81.07 |
| Phi-4 | `EXP-UINT4-ZP-UE5M3` | 89.99 | 75.98 | 81.59 | 81.77 |
| Phi-4 | `EXP-INT4-UE5M3`‡ | 88.63 | 75.67 | 81.36 | 81.07 |
| Phi-4 | `EXP-INT4-UBF16` | 89.23 | 76.46 | 81.47 | 81.01 |
| Phi-4 | `EXP-INT4-UFP16` | 88.86 | 76.46 | 81.47 | 81.12 |
| Phi-4 | `EXP-UINT4-ZP-BF16` | 89.54 | 76.24 | 81.61 | 81.61 |
| Phi-4 | `EXP-UINT4-ZP-FP16` | 89.92 | 76.10 | 81.54 | 81.28 |
| Phi-4 | `EXP-INT4-BF16` | 89.69 | 76.79 | 81.49 | 80.85 |
| Phi-4 | `EXP-INT4-FP16` | 89.31 | 76.73 | 81.45 | 80.90 |
| Qwen2.5-7B-Instruct | BF16 | 81.58 | 71.64 | 80.52 | 80.36 |
| Qwen2.5-7B-Instruct | `EXP-INT4-E4M3` | 83.17 | 70.85 | 79.22 | 79.11 |
| Qwen2.5-7B-Instruct | `EXP-UINT4-ZP-UE5M3` | 84.91 | 71.26 | 80.03 | 79.76 |
| Qwen2.5-7B-Instruct | `EXP-INT4-UE5M3`‡ | 81.05 | 70.44 | 79.54 | 78.89 |
| Qwen2.5-7B-Instruct | `EXP-INT4-UBF16` | 81.65 | 71.15 | 79.82 | 79.22 |
| Qwen2.5-7B-Instruct | `EXP-INT4-UFP16` | 80.89 | 71.09 | 79.92 | 79.05 |
| Qwen2.5-7B-Instruct | `EXP-UINT4-ZP-BF16` | 81.73 | 71.66 | 80.27 | 79.54 |
| Qwen2.5-7B-Instruct | `EXP-UINT4-ZP-FP16` | 80.59 | 71.82 | 80.22 | 79.49 |
| Qwen2.5-7B-Instruct | `EXP-INT4-BF16` | 86.35 | 71.36 | 80.10 | 78.89 |
| Qwen2.5-7B-Instruct | `EXP-INT4-FP16` | 86.20 | 71.29 | 80.08 | 78.78 |
| Qwen3-32B | BF16 | 91.89 | 80.79 | 82.60 | 82.00 |
| Qwen3-32B | `EXP-INT4-E4M3` | 90.67 | 80.35 | 82.28 | 81.45 |
| Qwen3-32B | `EXP-UINT4-ZP-UE5M3` | 90.37 | 80.24 | 82.27 | 81.28 |
| Qwen3-32B | `EXP-INT4-UE5M3`‡ | 90.90 | 80.01 | 82.29 | 81.23 |
| Qwen3-32B | `EXP-INT4-UBF16` | 88.48 | 80.42 | 81.43 | 80.74 |
| Qwen3-32B | `EXP-INT4-UFP16` | 87.79 | 80.36 | 81.45 | 80.90 |
| Qwen3-32B | `EXP-UINT4-ZP-BF16` | 90.60 | 80.63 | 82.07 | 81.45 |
| Qwen3-32B | `EXP-UINT4-ZP-FP16` | 90.07 | 80.54 | 82.18 | 81.77 |
| Qwen3-32B | `EXP-INT4-BF16` | 89.01 | 80.56 | 82.12 | 81.18 |
| Qwen3-32B | `EXP-INT4-FP16` | 88.78 | 80.58 | 82.15 | 81.23 |

#### W4A4

All 12 schemes under the current presets, with activations quantized in the same format as the weights (W4A4, group size 16, RTN, seed 42, fake quantization). The Qwen3-32B GSM8K runs for `EXP-FP4-UE5M3`, `EXP-INT4-UE5M3`, and `EXP-UINT4-ZP-UE5M3` used inference-time compiled activation QDQ, verified to be bit-identical to eager.

| Model | Scheme | GSM8K strict-match | MMLU acc | HellaSwag acc_norm | PIQA acc_norm |
| --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B-Instruct | BF16 | 76.72 | 68.07 | 79.17 | 81.01 |
| Llama-3.1-8B-Instruct | `EXP-FP4-UE5M3` | 63.84 | 63.67 | 77.49 | 79.00 |
| Llama-3.1-8B-Instruct | `EXP-INT4-E4M3` | 65.13 | 62.04 | 76.71 | 78.56 |
| Llama-3.1-8B-Instruct | `EXP-INT4-UE5M3` | 64.97 | 64.39 | 78.32 | 79.27 |
| Llama-3.1-8B-Instruct | `EXP-UINT4-ZP-UE5M3` | 71.57 | 65.48 | 78.72 | 80.20 |
| Llama-3.1-8B-Instruct | `EXP-FP4-BF16` | 63.46 | 64.00 | 78.22 | 78.89 |
| Llama-3.1-8B-Instruct | `EXP-FP4-FP16` | 62.32 | 63.78 | 78.05 | 79.76 |
| Llama-3.1-8B-Instruct | `EXP-INT4-UBF16` | 67.10 | 64.76 | 78.34 | 79.38 |
| Llama-3.1-8B-Instruct | `EXP-INT4-UFP16` | 68.16 | 64.29 | 78.47 | 79.54 |
| Llama-3.1-8B-Instruct | `EXP-UINT4-ZP-BF16` | 75.44 | 66.00 | 78.52 | 80.58 |
| Llama-3.1-8B-Instruct | `EXP-UINT4-ZP-FP16` | 73.54 | 65.95 | 78.54 | 79.92 |
| Llama-3.1-8B-Instruct | `EXP-INT4-BF16` | 70.05 | 65.04 | 78.52 | 79.38 |
| Llama-3.1-8B-Instruct | `EXP-INT4-FP16` | 68.08 | 65.38 | 78.54 | 79.65 |
| Phi-4 | BF16 | 90.67 | 76.86 | 81.98 | 81.18 |
| Phi-4 | `EXP-FP4-UE5M3` | 88.86 | 75.67 | 81.00 | 80.52 |
| Phi-4 | `EXP-INT4-E4M3` | 88.55 | 75.42 | 81.14 | 80.79 |
| Phi-4 | `EXP-INT4-UE5M3` | 85.82 | 76.11 | 81.00 | 81.18 |
| Phi-4 | `EXP-UINT4-ZP-UE5M3` | 89.23 | 75.99 | 81.26 | 81.28 |
| Phi-4 | `EXP-FP4-BF16` | 88.48 | 75.92 | 81.02 | 81.01 |
| Phi-4 | `EXP-FP4-FP16` | 88.63 | 76.06 | 80.96 | 80.36 |
| Phi-4 | `EXP-INT4-UBF16` | 88.02 | 75.77 | 81.03 | 80.69 |
| Phi-4 | `EXP-INT4-UFP16` | 89.08 | 76.09 | 81.06 | 80.58 |
| Phi-4 | `EXP-UINT4-ZP-BF16` | 90.22 | 75.92 | 81.36 | 81.18 |
| Phi-4 | `EXP-UINT4-ZP-FP16` | 89.76 | 75.78 | 81.30 | 81.07 |
| Phi-4 | `EXP-INT4-BF16` | 88.78 | 76.47 | 81.20 | 80.79 |
| Phi-4 | `EXP-INT4-FP16` | 88.78 | 76.47 | 81.20 | 80.79 |
| Qwen2.5-7B-Instruct | BF16 | 81.58 | 71.64 | 80.52 | 80.36 |
| Qwen2.5-7B-Instruct | `EXP-FP4-UE5M3` | 83.24 | 70.57 | 79.04 | 78.07 |
| Qwen2.5-7B-Instruct | `EXP-INT4-E4M3` | 81.96 | 69.69 | 78.44 | 77.64 |
| Qwen2.5-7B-Instruct | `EXP-INT4-UE5M3` | 75.59 | 70.11 | 78.55 | 78.84 |
| Qwen2.5-7B-Instruct | `EXP-UINT4-ZP-UE5M3` | 81.20 | 70.67 | 79.24 | 79.05 |
| Qwen2.5-7B-Instruct | `EXP-FP4-BF16` | 83.17 | 70.25 | 79.07 | 78.51 |
| Qwen2.5-7B-Instruct | `EXP-FP4-FP16` | 82.64 | 70.35 | 79.20 | 77.91 |
| Qwen2.5-7B-Instruct | `EXP-INT4-UBF16` | 76.88 | 70.05 | 78.86 | 77.64 |
| Qwen2.5-7B-Instruct | `EXP-INT4-UFP16` | 77.03 | 70.11 | 78.76 | 78.29 |
| Qwen2.5-7B-Instruct | `EXP-UINT4-ZP-BF16` | 81.27 | 71.41 | 79.57 | 79.49 |
| Qwen2.5-7B-Instruct | `EXP-UINT4-ZP-FP16` | 80.44 | 71.14 | 79.70 | 78.13 |
| Qwen2.5-7B-Instruct | `EXP-INT4-BF16` | 83.55 | 70.36 | 79.11 | 78.40 |
| Qwen2.5-7B-Instruct | `EXP-INT4-FP16` | 83.55 | 70.36 | 79.11 | 78.40 |
| Qwen3-32B | BF16 | 91.89 | 80.79 | 82.60 | 82.00 |
| Qwen3-32B | `EXP-FP4-UE5M3` | 91.96 | 79.57 | 81.34 | 81.18 |
| Qwen3-32B | `EXP-INT4-E4M3` | 90.67 | 79.48 | 81.76 | 80.14 |
| Qwen3-32B | `EXP-INT4-UE5M3` | 87.19 | 79.28 | 80.50 | 80.14 |
| Qwen3-32B | `EXP-UINT4-ZP-UE5M3` | 91.05 | 79.60 | 82.05 | 81.01 |
| Qwen3-32B | `EXP-FP4-BF16` | 90.90 | 79.66 | 81.74 | 80.36 |
| Qwen3-32B | `EXP-FP4-FP16` | 91.96 | 79.42 | 81.63 | 81.23 |
| Qwen3-32B | `EXP-INT4-UBF16` | 86.88 | 79.69 | 80.27 | 79.76 |
| Qwen3-32B | `EXP-INT4-UFP16` | 87.87 | 79.01 | 80.64 | 80.20 |
| Qwen3-32B | `EXP-UINT4-ZP-BF16` | 91.96 | 79.86 | 81.83 | 80.96 |
| Qwen3-32B | `EXP-UINT4-ZP-FP16` | 90.37 | 79.78 | 82.06 | 80.90 |
| Qwen3-32B | `EXP-INT4-BF16` | 89.23 | 79.70 | 81.60 | 80.09 |
| Qwen3-32B | `EXP-INT4-FP16` | 89.23 | 79.69 | 81.62 | 80.09 |
