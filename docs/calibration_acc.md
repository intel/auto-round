## Calibration Dataset Study (WIP)

**Current Recommendation**: For 4-bit quantization, current results suggest **75% UltraChat + 25% OpenCode (Mix2 recipe)**. With the default `nsamples=128`, this recipe can be applied using `--dataset 'HuggingFaceH4/ultrachat_200k:apply_chat_template=true:num=96,nvidia/OpenCodeInstruct:num=32'`.

### Qwen3.8-27B-W4A16

| Calibration Dataset | AIME26 | GPQA Diamond | IFBench Strict | MMLU-Pro | LiveCodeBench | SWE-mini | Terminal_Bench |
|---|---:|---:|---:|---:|---:|---:|---:|
| cci3hq | 96.67, 90, 100, 100, 96.67=96.67 | 91.92, 90.4, 88.89, 91.41, 89.39=90.40 | 73.67, 76.16=74.92 | 85.03 | 91.18 | - | - |
| OpenCode | 96.67, 96.67, 100, 100, 96.67=98.00 | 89.39, 87.88, 90.4, 89.39, -=89.27 | 74, 76.45=75.23 | 85.26 | 92.51 | 58.00 | 49.4 |
| pile10k | 100, 93.33, 100, 93.33, 100=97.33 | 87.88, 88.38, 88.89, 89.9, 90.4=89.09 | 72.33, 75=73.67 | 85.00 | 91.37 | - | - |
| UltraChat | 100, 93.33, 100, 100, 96.67=98.00 | 90.91, 91.41, 90.4, 90.4, -=**90.78** | 73.33, 76.16=74.75 | 85.20 | 91.75 | - | - |
| UltraFineWeb | 96.67, 100, 100, 96.67, 93.33=97.33 | 89.39, 90.4, 90.4, 90.4, 88.89=89.90 | 70, 73.26=71.63 | 84.96 | 91.75 | - | - |
| mix_a (ultrachat 50% + OpenCode 25% + cci3hq 12.5% + pile10k 12.5%) | 100,100,100,96.7,100=**99.34** | 90.9,89.4,90.9,90.4,88.9=90.10 | 72.3, 75=73.65 | 85.40 | **92.70** | 60	| - |
| mix_b (ultrachat 75% + OpenCode 25%) | 96.7, 100, 96.7, 100, 96.7=98.02 | 87.9, 90.9, 89.9, 91.9, 91.4=90.4 | 74.7, 77.3=**76.00** | **85.50** | 92.20 | 56.9 | - |
| mix_b (ultrachat 75% (apply_chat_template=true) + OpenCode 25%) | 100, 100, 96.7, 100, -=99.18 | 87.9, 90.4, 89.4, 90.4,  -=89.53 | - | - | - | - | - |
| mix_c (ultrachat 87.5% + OpenCode 12.5%) | 93.3, 100, 96.7, -, -=96.67 | 88.4, 90.4, 86.4, -, -=88.40 | 71.3, 74.7=73.00 | - | - | - | - |


### Gemma-4-12B-it-W4A16

| Calibration Dataset | AIME26 | GPQA Diamond | IFBench Strict |
|---|---:|---:|---:|
| cci3hq | 46.7, 46.7, 50, 50, 53.3=49.34 | 42.4, 35.4, 36.9, 40.4, 38.4=38.70 | 59, 58.4=58.70 |
| OpenCode | 63.3, 76.7, 60, 60, 70=66.00 | 51.5, 50.5, 54, -, -=52.00 | 63.7, 64.8=64.25 |
| pile10k | 53.3, 53.3, 66.7, 56.7, 63.3=58.66 | - | 65.3, 66.6=65.95 |
| UltraChat | 66.7, 63.3, 66.7, 70, 73.3=68.00 | 61.6, 61.1, 65.1, -, -=62.60 | 67.7, 69.8=68.75 |
| UltraFineWeb | 56.7, 60, 60, 56.7, 53.3=57.34 | - | 66.7, 68.9=67.80|
| mix_a (ultrachat 50% + OpenCode 25% + cci3hq 12.5% + pile10k 12.5%) | 56.70 | 53.50 | 66.3, 68=67.15 |
| mix_b (ultrachat 75% + OpenCode 25%) | **76.70** | 58.60 | 67.95 |
| mix_b (ultrachat 75% (apply_chat_template=true) + OpenCode 25%) | 73.3,  66.7, 76.7, 80, 53.3=70.00 | 65.1, 63.6, 65.7, 62.1, 65.1=**64.32** | 68, 70.1=**69.05** |
| mix_c (ultrachat 87.5% + OpenCode 12.5%) | 66.7, 66.7, 66.7, 63.3, 70=66.68 | 60.1, 61.6, 66.7, 64.6, 61.1=62.82 | 67.7, 69.8=68.75 |


### Reproduction Setup
**(1) Environment**
- Quantization & vLLM: Python 3.11.17, AutoRound 0.16.0, PyTorch 2.13.0+cu130, vLLM 0.29.0
- Standard Evaluation: Python 3.11.17, EvalScope 1.12.0
- Agent Evaluation: Python 3.12.15, EvalScope 1.12.0

**(2) Model Configurations**
**Qwen**
- Hardware: H200 / B200
- Quantization: W4A16g128
- Generation: temperature=1, top_p=1, top_k=20, min_p=0
- Max tokens: 131,072
- Evaluation: batch_size=32, timeout=14,400s, seed=42
- Thinking: enabled, preserve_thinking=true
- Streaming: disabled

**Gemma**
- Model: /models/gemma-4-12B-it
- Served model name: gemma4_bf16
- Hardware: A100
- Dtype: bfloat16
- Thinking: enabled
- Reasoning parser: gemma4
- Generation: temperature=1, top_p=0.95, top_k=64, min_p=0
- Repetition penalty: 1
- Max tokens: 32,768
- Evaluation: batch_size=16, timeout=7,200s
- Enforce eager: enabled
- Max model length: not explicitly set

**Benchmark Datasets**

| Benchmark | Samples |
|---|---:|
| AIME26 | 30 |
| GPQA Diamond | 198 |
| IFBench (Prompts & Instructions) Strict Prompt Strict, Instruction Strict=avg) | 300 & 344 |
| LiveCodeBench (release_v6) | 1,055 |
| MMLU-Pro | 12,032 |
