## Calibration Dataset Study (WIP)

**Current Recommendation**: For 4-bit quantization, current results suggest **75% UltraChat + 25% OpenCode (Mix2 recipe)**. This recipe can be applied using `--dataset 'HuggingFaceH4/ultrachat_200k:apply_chat_template=true:num=96,nvidia/OpenCodeInstruct:num=32'`

---

### Qwen3.8-27B-W4A16

| Calibration Dataset | AIME26 | GPQA Diamond | IFBench Strict | MMLU-Pro | LiveCodeBench |
|---|---:|---:|---:|---:|---:|
| cci3hq | 96.67 | 90.40 | 74.92 | 85.03 | 91.18 |
| OpenCode | 98.00 | 89.27 | 75.23 | 85.26 | 92.51 |
| pile10k | 97.33 | 89.09 | 73.67 | 85.00 | 91.37 |
| UltraChat | 98.00 | **90.78** | 74.75 | 85.20 | 91.75 |
| UltraFineWeb | 97.33 | 89.90 | 71.63 | 84.96 | 91.75 |
| Mix1 | **99.34** | 90.10 | 73.65 | 85.40 | **92.70** |
| Mix2 | 98.02 | 90.40 | **76.00** | **85.50** | 92.20 |
| Mix2 (Chat Temp) | 99.18 | 89.53 | - | - | - |
| Mix3 | 96.67 | 88.40 | 73.00 | - | - |


### Gemma-4-12B-it-W4A16

| Calibration Dataset | AIME26 | GPQA Diamond | IFBench Strict |
|---|---:|---:|---:|
| cci3hq | 49.34 | 38.70 | 58.70 |
| OpenCode | 66.00 | 52.00 | 64.25 |
| pile10k | 58.66 |  | 65.95 |
| UltraChat | 68.00 | 62.60 | 68.75 |
| UltraFineWeb | 57.34 |  | 67.80 |
| Mix1 | 56.70 | 53.50 | 67.15 |
| Mix2 | **76.70** | 58.60 | 67.95 |
| Mix2 (Chat Temp) | 70.00 | **64.32** | **69.05** |
| Mix3 | 66.68 | 62.82 | 68.75 |

---

#### Detailed Dataset Definitions

| Name | Composition |
|---|---|
| Mix1 | 50% UltraChat + 25% OpenCode + 12.5% cci3hq + 12.5% pile10k |
| Mix2 | 75% UltraChat + 25% OpenCode |
| Mix2 (Chat Temp) | 75% UltraChat + 25% OpenCode (Chat Template Variant) |
| Mix3 | 87.5% UltraChat + 12.5% OpenCode |


<details>
<summary><strong>Qwen3.8-27B Multi-Run Statistics</strong></summary>


| AIME26-Dataset | Run1 | Run2 | Run3 | Run4 | Run5 | Avg |
|---|---:|---:|---:|---:|---:|---:|
| cci3hq | 96.67 | 90.00 | 100.00 | 100.00 | 96.67 | 96.67 |
| OpenCode | 96.67 | 96.67 | 100.00 | 100.00 | 96.67 | 98.00 |
| pile10k | 100.00 | 93.33 | 100.00 | 93.33 | 100.00 | 97.33 |
| UltraChat | 100.00 | 93.33 | 100.00 | 100.00 | 96.67 | 98.00 |
| UltraFineWeb | 96.67 | 100.00 | 100.00 | 96.67 | 93.33 | 97.33 |
| Mix1 | 100.00 | 100.00 | 100.00 | 96.70 | 100.00 | 99.34 |
| Mix2 | 96.70 | 100.00 | 96.70 | 100.00 | 96.70 | 98.02 |
| Mix2 (Chat Temp) | 100.00 | 100.00 | 96.70 | 100.00 | - | 99.18 |
| Mix3 | 93.30 | 100.00 | 96.70 | - | - | 96.67 |

| GPQA Diamond-Dataset | Run1 | Run2 | Run3 | Run4 | Run5 | Avg |
|---|---:|---:|---:|---:|---:|---:|
| cci3hq | 91.92 | 90.40 | 88.89 | 91.41 | 89.39 | 90.40 |
| OpenCode | 89.39 | 87.88 | 90.40 | 89.39 | - | 89.27 |
| pile10k | 87.88 | 88.38 | 88.89 | 89.90 | 90.40 | 89.09 |
| UltraChat | 90.91 | 91.41 | 90.40 | 90.40 | - | 90.78 |
| UltraFineWeb | 89.39 | 90.40 | 90.40 | 90.40 | 88.89 | 89.90 |
| Mix1 | 90.90 | 89.40 | 90.90 | 90.40 | 88.90 | 90.10 |
| Mix2 | 87.90 | 90.90 | 89.90 | 91.90 | 91.40 | 90.40 |
| Mix2 (Chat Temp) | 87.90 | 90.40 | 89.40 | 90.40 | - | 89.53 |
| Mix3 | 88.40 | 90.40 | 86.40 | - | - | 88.40 |

</details>

<details>
<summary><strong>Gemma-4-12B-it Multi-Run Statistics</strong></summary>

| AIME26-Dataset | Run1 | Run2 | Run3 | Run4 | Run5 | Avg |
|---|---:|---:|---:|---:|---:|---:|
| cci3hq | 46.70 | 46.70 | 50.00 | 50.00 | 53.30 | 49.34 |
| OpenCode | 63.30 | 76.70 | 60.00 | 60.00 | 70.00 | 66.00 |
| pile10k | 53.30 | 53.30 | 66.70 | 56.70 | 63.30 | 58.66 |
| UltraChat | 66.70 | 63.30 | 66.70 | 70.00 | 73.30 | 68.00 |
| UltraFineWeb | 56.70 | 60.00 | 60.00 | 56.70 | 53.30 | 57.34 |
| Mix1 | 56.70 | - | - | - | - | 56.70 |
| Mix2 | 76.70 | - | - | - | - | 76.70 |
| Mix2 (Chat Temp) | 73.30 | 66.70 | 76.70 | 80.00 | 53.30 | 70.00 |
| Mix3 | 66.70 | 66.70 | 66.70 | 63.30 | 70.00 | 66.68 |

| GPQA Diamond-Dataset | Run1 | Run2 | Run3 | Run4 | Run5 | Avg |
|---|---:|---:|---:|---:|---:|---:|
| cci3hq | 42.40 | 35.40 | 36.90 | 40.40 | 38.40 | 38.70 |
| OpenCode | 51.50 | 50.50 | 54.00 | - | - | 52.00 |
| pile10k | - | - | - | - | - | - |
| UltraChat | 61.60 | 61.10 | 65.10 | - | - | 62.60 |
| UltraFineWeb | - | - | - | - | - | - |
| Mix1 | 53.50 | - | - | - | - | 53.50 |
| Mix2 | 58.60 | - | - | - | - | 58.60 |
| Mix2 (Chat Temp) | 65.10 | 63.60 | 65.70 | 62.10 | 65.10 | 64.32 |
| Mix3 | 60.10 | 61.60 | 66.70 | 64.60 | 61.10 | 62.82 |


</details>
