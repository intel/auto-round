# W4A4 Quantization Scheme Evaluation Report

**Models:** Llama-3.1-8B-Instruct, Phi-4, Qwen2.5-7B-Instruct, Qwen3-32B  
**Evaluation status:** Complete for all 12 schemes on all 4 models (48/48)

## Executive Summary

Twelve group-size-16 W4A4 schemes were evaluated on four models with GSM8K, MMLU, HellaSwag, and PIQA. Averaged over the four models, the BF16 baseline scores **80.44%** and the W4A4 schemes range from 77.58% to 79.72%, i.e. 0.72 to 2.86 points below BF16.

`EXP-UINT4-ZP-BF16` has the highest four-model mean at **79.72%** (−0.72 vs BF16) and ranks first on every model, but it also has the highest storage cost (5.25 BPW). Taking bits per weight into account, `EXP-UINT4-ZP-UE5M3` is the best trade-off: at 4.75 BPW it reaches 79.23%, higher than every 5.0-BPW scheme. At the minimum 4.5 BPW, `EXP-FP4-UE5M3` is best (78.44%). Within INT4, `[-8, 7]` codes beat symmetric `[-7, 7]` codes at every matching BPW and scale dtype; the `[-7, 7]` family is the weakest overall (77.58%–78.07%), mainly because of lower GSM8K scores.

The four-metric mean is a compact summary, not a substitute for task-specific selection. Results are from one evaluation run and should be treated as point estimates.

## Experimental Method

- **Models:** Llama-3.1-8B-Instruct, Phi-4, Qwen2.5-7B-Instruct, Qwen3-32B.
- **Quantization:** 4-bit weights and 4-bit activations (W4A4) with activations in the same format as the weights, group size 16, round-to-nearest (RTN), fake-quantization evaluation.
- **Calibration:** The RTN path does not consume calibration data.
- **Random seed:** 42.
- **Likelihood tasks:** MMLU accuracy, HellaSwag accuracy and normalized accuracy, PIQA accuracy and normalized accuracy; evaluation dtype BF16. Batch size varied across models and runs.
- **Generation task:** GSM8K, 5-shot greedy decoding, BF16 evaluation dtype, batch size 8, maximum generation length 1024 tokens.
- **Reporting:** All values below are percentages. The four-metric mean is the arithmetic mean of GSM8K strict-match, MMLU accuracy, HellaSwag normalized accuracy, and PIQA normalized accuracy. The four-model mean weights each model equally. BF16 is the unquantized baseline.

## Scheme Definitions

All schemes below quantize weights and activations at 4 bits with group size 16. The scale dtype and code mapping distinguish the variants. Schemes are grouped by code family: FP4, INT4 `[-8, 7]`, INT4 `[-7, 7]`, and UINT4 with zero-point.

| Scheme | Weight/activation code representation | Per-group scale and offset | Weight BPW |
| --- | --- | --- | ---: |
| `EXP-FP4-UE5M3` | Signed E2M1 FP4; representable magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6 | Nonnegative UE5M3 scale; no zero-point | 4.5 |
| `EXP-FP4-BF16` | Signed E2M1 FP4 | Nonnegative BF16 scale; no zero-point | 5.0 |
| `EXP-FP4-FP16` | Signed E2M1 FP4 | Nonnegative FP16 scale; no zero-point | 5.0 |
| `EXP-INT4-E4M3` | Signed integer codes `[-8, 7]` | Signed E4M3 scale; scale sign selects which side uses the larger magnitude endpoint | 4.5 |
| `EXP-INT4-BF16` | Signed integer codes `[-8, 7]` | Signed BF16 scale; scale sign selects which side uses the larger magnitude endpoint | 5.0 |
| `EXP-INT4-FP16` | Signed integer codes `[-8, 7]` | Signed FP16 scale; scale sign selects which side uses the larger magnitude endpoint | 5.0 |
| `EXP-INT4-UE5M3` | Symmetric signed integer codes `[-7, 7]` | Nonnegative UE5M3 scale | 4.5 |
| `EXP-INT4-UBF16` | Symmetric signed integer codes `[-7, 7]` | Nonnegative BF16 scale | 5.0 |
| `EXP-INT4-UFP16` | Symmetric signed integer codes `[-7, 7]` | Nonnegative FP16 scale | 5.0 |
| `EXP-UINT4-ZP-UE5M3` | Unsigned integer codes `[0, 15]` | Nonnegative UE5M3 scale and per-group zero-point | 4.75* |
| `EXP-UINT4-ZP-BF16` | Unsigned integer codes `[0, 15]` | Nonnegative BF16 scale and per-group zero-point | 5.25* |
| `EXP-UINT4-ZP-FP16` | Unsigned integer codes `[0, 15]` | Nonnegative FP16 scale and per-group zero-point | 5.25* |

For zero-point schemes, unsigned storage codes represent values around a per-group zero-point; the reconstructed values may be negative. `U` in `UE5M3`, `UBF16`, and `UFP16` identifies a nonnegative scale, not an unsigned floating-point encoding.

Weight BPW (bits per weight) is the theoretical storage cost with group size 16: `BPW = 4 + scale_bits / 16 + zero_point_bits / 16`, with 8-bit E4M3/UE5M3 scales and 16-bit BF16/FP16 scales. `*` assumes the zero-point is packed in 4 bits. Container alignment and padding are excluded.

## Accuracy Results

### Four-Metric Mean by Scheme

Each model column is the four-metric mean for that model. Rows are sorted by the four-model mean; bold marks the best quantized scheme in each column.

| Scheme | BPW | Llama-3.1-8B-Instruct | Phi-4 | Qwen2.5-7B-Instruct | Qwen3-32B | Four-model mean | Δ vs BF16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 16 | 76.24 | 82.67 | 78.53 | 84.32 | 80.44 | — |
| `EXP-UINT4-ZP-BF16` | 5.25* | **75.14** | **82.17** | **77.94** | **83.65** | **79.72** | -0.72 |
| `EXP-UINT4-ZP-FP16` | 5.25* | 74.49 | 81.98 | 77.35 | 83.28 | 79.27 | -1.17 |
| `EXP-UINT4-ZP-UE5M3` | 4.75* | 73.99 | 81.94 | 77.54 | 83.43 | 79.23 | -1.21 |
| `EXP-INT4-BF16` | 5.0 | 73.25 | 81.81 | 77.86 | 82.66 | 78.89 | -1.55 |
| `EXP-INT4-FP16` | 5.0 | 72.91 | 81.81 | 77.86 | 82.66 | 78.81 | -1.63 |
| `EXP-FP4-UE5M3` | 4.5 | 71.00 | 81.51 | 77.73 | 83.51 | 78.44 | -2.00 |
| `EXP-FP4-BF16` | 5.0 | 71.14 | 81.61 | 77.75 | 83.17 | 78.42 | -2.02 |
| `EXP-FP4-FP16` | 5.0 | 70.98 | 81.50 | 77.53 | 83.56 | 78.39 | -2.05 |
| `EXP-INT4-UFP16` | 5.0 | 72.62 | 81.70 | 76.05 | 81.93 | 78.07 | -2.37 |
| `EXP-INT4-E4M3` | 4.5 | 70.61 | 81.48 | 76.93 | 83.01 | 78.01 | -2.43 |
| `EXP-INT4-UBF16` | 5.0 | 72.40 | 81.38 | 75.86 | 81.65 | 77.82 | -2.62 |
| `EXP-INT4-UE5M3` | 4.5 | 71.74 | 81.03 | 75.77 | 81.78 | 77.58 | -2.86 |

### Per-Task Four-Model Mean

| Scheme | GSM8K strict-match | MMLU acc | HellaSwag acc_norm | PIQA acc_norm |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 85.22 | 74.34 | 81.07 | 81.14 |
| `EXP-UINT4-ZP-BF16` | **84.72** | **73.30** | 80.32 | **80.55** |
| `EXP-UINT4-ZP-FP16` | 83.53 | 73.16 | **80.40** | 80.01 |
| `EXP-UINT4-ZP-UE5M3` | 83.26 | 72.94 | 80.32 | 80.39 |
| `EXP-INT4-BF16` | 82.90 | 72.89 | 80.11 | 79.67 |
| `EXP-INT4-FP16` | 82.41 | 72.98 | 80.12 | 79.73 |
| `EXP-FP4-UE5M3` | 81.98 | 72.37 | 79.72 | 79.69 |
| `EXP-FP4-BF16` | 81.50 | 72.46 | 80.01 | 79.69 |
| `EXP-FP4-FP16` | 81.39 | 72.40 | 79.96 | 79.82 |
| `EXP-INT4-UFP16` | 80.54 | 72.38 | 79.73 | 79.65 |
| `EXP-INT4-E4M3` | 81.58 | 71.66 | 79.51 | 79.28 |
| `EXP-INT4-UBF16` | 79.72 | 72.57 | 79.63 | 79.37 |
| `EXP-INT4-UE5M3` | 78.39 | 72.47 | 79.59 | 79.86 |

## Findings and Conclusion

1. All twelve W4A4 schemes stay within 2.86 points of BF16 on the four-model mean (77.58%–79.72% vs 80.44%), at 4.5–5.25 BPW.
2. Only three schemes are on the accuracy–BPW Pareto front: `EXP-FP4-UE5M3` (4.5 BPW, 78.44%), `EXP-UINT4-ZP-UE5M3` (4.75 BPW, 79.23%), and `EXP-UINT4-ZP-BF16` (5.25 BPW, 79.72%). Every other scheme is matched or beaten by one of them at equal or lower BPW.
3. `EXP-UINT4-ZP-BF16` is the most accurate scheme: highest four-model mean, first on each of the four models, and best four-model average on GSM8K (84.72%), MMLU (73.30%), and PIQA (80.55%). Its +0.49-point gain over `EXP-UINT4-ZP-UE5M3` costs an extra 0.5 BPW.
4. `EXP-UINT4-ZP-UE5M3` is the best accuracy-per-bit choice. At 4.75 BPW it is 0.34 points above the best 5.0-BPW scheme (`EXP-INT4-BF16`, 78.89%) and only 1.21 points below BF16.
5. Within INT4, the `[-8, 7]` code range is consistently better than symmetric `[-7, 7]` at the same BPW: +0.43 (E4M3 vs UE5M3, 4.5 BPW), +1.07 (BF16 vs UBF16), and +0.74 (FP16 vs UFP16) points, driven mainly by GSM8K (+1.9 to +3.2 points). The extra negative code and signed scale are effectively free in storage.
6. The cost of an 8-bit scale depends on the code family. Going from 16-bit to 8-bit scales (−0.5 BPW) costs nothing for FP4 (78.44% vs 78.39%–78.42%), 0.04–0.49 points for zero-point INT4, 0.24–0.49 points for `[-7, 7]` INT4, and 0.80–0.88 points for `[-8, 7]` INT4 (E4M3). BF16 and FP16 scales differ by at most 0.45 points with no consistent ordering.
7. Sensitivity differs by model. Llama-3.1-8B-Instruct shows the largest drop (1.10–5.63 points from BF16 76.24%); Phi-4 shows the smallest (0.50–1.64 points from 82.67%).
8. Scheme selection should follow the storage budget and target workload. Use the four-metric mean and BPW for a broad comparison, but prioritize the corresponding task metric when deployment is task-specific.

These findings apply to these four models, this configuration, and a single evaluation run. They do not establish statistical significance or guarantee the same ranking on other models or inference backends.

## Details

### Full Results

Scores are percentages.

| Model | Scheme | GSM8K strict-match | MMLU acc | HellaSwag acc_norm | PIQA acc_norm |
| --- | --- | ---: | ---: | ---: | ---: |
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

### Llama-3.1-8B-Instruct Extended Metrics

GSM8K flexible-extract and HellaSwag/PIQA unnormalized accuracy are additionally reported for Llama-3.1-8B-Instruct. Bold marks the best scheme for that metric.

| Scheme | GSM8K flexible / strict | MMLU acc | HellaSwag acc / acc_norm | PIQA acc / acc_norm | Four-metric mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `EXP-FP4-UE5M3` | 70.36 / 63.84 | 63.67 | 58.43 / 77.49 | 77.86 / 79.00 | 71.00 |
| `EXP-INT4-E4M3` | 65.20 / 65.13 | 62.04 | 57.66 / 76.71 | 77.75 / 78.56 | 70.61 |
| `EXP-INT4-UE5M3` | 67.40 / 64.97 | 64.39 | 58.56 / 78.32 | 78.45 / 79.27 | 71.74 |
| `EXP-UINT4-ZP-UE5M3` | 73.31 / 71.57 | 65.48 | 58.80 / 78.72 | 79.22 / 80.20 | 73.99 |
| `EXP-FP4-BF16` | 69.37 / 63.46 | 64.00 | 58.50 / 78.22 | 78.29 / 78.89 | 71.14 |
| `EXP-FP4-FP16` | 69.67 / 62.32 | 63.78 | 58.46 / 78.05 | 78.35 / 79.76 | 70.98 |
| `EXP-INT4-UBF16` | 68.99 / 67.10 | 64.76 | 58.54 / 78.34 | 78.62 / 79.38 | 72.40 |
| `EXP-INT4-UFP16` | 70.28 / 68.16 | 64.29 | 58.49 / 78.47 | 78.56 / 79.54 | 72.62 |
| `EXP-UINT4-ZP-BF16` | 75.97 / 75.44 | 66.00 | 58.82 / 78.52 | 79.60 / 80.58 | **75.14** |
| `EXP-UINT4-ZP-FP16` | 74.75 / 73.54 | 65.95 | **59.16** / 78.54 | 78.89 / 79.92 | 74.49 |
| `EXP-INT4-BF16` | 72.25 / 70.05 | 65.04 | 58.78 / 78.52 | 78.78 / 79.38 | 73.25 |
| `EXP-INT4-FP16` | 70.96 / 68.08 | 65.38 | 58.74 / 78.54 | 78.73 / 79.65 | 72.91 |
