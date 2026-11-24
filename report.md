# W4A4 Quantization Scheme Evaluation Report

**Model:** Llama-3.1-8B-Instruct  
**Evaluation date:** 2026-11-19  
**Evaluation status:** Complete for all 12 schemes

## Executive Summary

Twelve group-size-16 W4A4 schemes were evaluated on GSM8K, MMLU, HellaSwag, and PIQA. `EXP-UINT4-ZP-BF16` achieved the highest four-metric mean at **75.14%**, with the best GSM8K flexible/strict scores (75.97% / 75.44%), MMLU accuracy (66.00%), and PIQA accuracy/normalized accuracy (79.60% / 80.58%). `EXP-UINT4-ZP-FP16` led HellaSwag accuracy at 59.16%; `EXP-UINT4-ZP-UE5M3` led HellaSwag normalized accuracy at 78.72%.

The four-metric mean is a compact summary, not a substitute for task-specific selection. Results are from one evaluation run and should be treated as point estimates.

## Experimental Method

- **Model:** Llama-3.1-8B-Instruct, snapshot `0e9e39f249a16976918f6564b8830bc894c89659`.
- **Quantization:** 4-bit weights and 4-bit activations (W4A4), group size 16, round-to-nearest (RTN), fake-quantization evaluation.
- **Calibration:** The RTN path does not consume calibration data. The command retained a 128-sample dataset argument; it is not used to fit quantization parameters in this RTN configuration.
- **Random seed:** 42.
- **Likelihood tasks:** MMLU accuracy, HellaSwag accuracy and normalized accuracy, PIQA accuracy and normalized accuracy; evaluation dtype BF16. Batch size was 32 for the first eight schemes and 16 for the four remaining schemes.
- **Generation task:** GSM8K, 5-shot greedy decoding, BF16 evaluation dtype, batch size 8, maximum generation length 1024 tokens.
- **Reporting:** All values below are percentages. The four-metric mean is the arithmetic mean of GSM8K strict-match, MMLU accuracy, HellaSwag normalized accuracy, and PIQA normalized accuracy.

## Scheme Definitions

All schemes below quantize weights and activations at 4 bits with group size 16. The scale dtype and code mapping distinguish the variants.

| Scheme | Weight/activation code representation | Per-group scale and offset |
| --- | --- | --- |
| `EXP-FP4-UE5M3` | Signed E2M1 FP4; representable magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6 | Nonnegative UE5M3 scale; no zero-point |
| `EXP-FP4-BF16` | Signed E2M1 FP4 | Nonnegative BF16 scale; no zero-point |
| `EXP-FP4-FP16` | Signed E2M1 FP4 | Nonnegative FP16 scale; no zero-point |
| `EXP-INT4-E4M3` | Signed integer codes `[-8, 7]` | Signed E4M3 scale; scale sign selects which side uses the larger magnitude endpoint |
| `EXP-INT4-UE5M3` | Symmetric signed integer codes `[-7, 7]` | Nonnegative UE5M3 scale |
| `EXP-INT4-UBF16` | Symmetric signed integer codes `[-7, 7]` | Nonnegative BF16 scale |
| `EXP-INT4-UFP16` | Symmetric signed integer codes `[-7, 7]` | Nonnegative FP16 scale |
| `EXP-INT4-BF16` | Signed integer codes `[-8, 7]` | Signed BF16 scale; scale sign selects which side uses the larger magnitude endpoint |
| `EXP-INT4-FP16` | Signed integer codes `[-8, 7]` | Signed FP16 scale; scale sign selects which side uses the larger magnitude endpoint |
| `EXP-UINT4-ZP-UE5M3` | Unsigned integer codes `[0, 15]` | Nonnegative UE5M3 scale and per-group zero-point |
| `EXP-UINT4-ZP-BF16` | Unsigned integer codes `[0, 15]` | Nonnegative BF16 scale and per-group zero-point |
| `EXP-UINT4-ZP-FP16` | Unsigned integer codes `[0, 15]` | Nonnegative FP16 scale and per-group zero-point |

For zero-point schemes, unsigned storage codes represent values around a per-group zero-point; the reconstructed values may be negative. `U` in `UE5M3`, `UBF16`, and `UFP16` identifies a nonnegative scale, not an unsigned floating-point encoding.

## Accuracy Results

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

## Findings and Conclusion

1. `EXP-UINT4-ZP-BF16` ranks first by the specified four-metric mean (75.14%) and leads on GSM8K flexible and strict match, MMLU accuracy, and PIQA accuracy and normalized accuracy.
2. Zero-point schemes are the strongest group in this model run: their four-metric means range from 73.99% to 75.14%, above the means of the non-zero-point FP4 and INT4 variants in this table.
3. Scale dtype effects are task-dependent. BF16-scale and FP16-scale variants do not exhibit a universal ordering; for example, `EXP-UINT4-ZP-FP16` has the highest HellaSwag accuracy, while `EXP-UINT4-ZP-UE5M3` has the highest HellaSwag normalized accuracy.
4. Scheme selection should follow the target workload. Use the four-metric mean for a broad summary, but prioritize the corresponding task metric when deployment is task-specific.

These findings apply to this model, configuration, and single evaluation run. They do not establish statistical significance or guarantee the same ranking on other models or inference backends.
