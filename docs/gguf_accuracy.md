1 We evaluate all models using the `fake` format, as lm-eval reports inaccurate accuracy for real GGUF format 


lm-eval 0.48

```bash
 lm-eval --model hf --model_args pretrained="./"   --tasks mmlu,leaderboard_ifeval,leaderboard_mmlu_pro,gsm8k 
 --batch_size 16
```

2 `lm-head` and `embedding` layers are not quantized in any of the following models.

| Q4_K_S                    | Avg.       | mmlu   | mmlu_pro | ifeval  | gsm8k  |
|---------------------------|------------|--------|----------|---------|--------|
| Qwen2.5-7B-GGUF           | 0.6366     | 0.7097 | 0.4385   | 0.61115 | 0.7870 |
| Qwen2.5-7B-AutoRound      | **0.6529** | 0.7137 | 0.4471   | 0.6373  | 0.8135 |
| Llama-3.1-8B-GGUF         | 0.5589     | 0.6609 | 0.3610   | 0.4949  | 0.7187 |
| Llama-3.1-8B-AutoRound    | **0.5666** | 0.6627 | 0.3648   | 0.49965 | 0.7392 |
| Falcon3-7B-GGUF           | 0.5179     | 0.6649 | 0.3607   | 0.3251  | 0.7210 |
| Falcon3-7B-GGUF-AutoRound | **0.5261** | 0.6706 | 0.3841   | 0.31445 | 0.7354 |
| phi-4-GGUF                | **0.5623** | 0.7648 | 0.5292   | 0.0590  | 0.8961 |
| phi-4-AutoRound           | 0.5588     | 0.7673 | 0.5239   | 0.05175 | 0.8923 |

| Q2_K_S               | Avg.       | mmlu   | mmlu_pro | if_eval  | gsm8k  |
|----------------------|------------|--------|----------|----------|--------|
| Qwen2.5-7B-GGUF      | 0.3942     | 0.5750 | 0.2701   | 0.4071   | 0.3245 |
| Qwen2.5-7B-AutoRound | **0.5133** | 0.6384 | 0.3383   | 0.4714   | 0.6050 |
| Falcon3-7B-GGUF      | 0.1936     | 0.3491 | 0.1521   | 0.21615  | 0.0569 |
| Falcon3-7B-AutoRound | **0.3817** | 0.5607 | 0.2625   | 0.28955  | 0.4139 |
| phi-4-GGUF           | 0.4438     | 0.6715 | 0.3807   | 0.0802   | 0.6429 |
| phi-4-AutoRound      | **0.5113** | 0.7107 | 0.4383   | 0.086750 | 0.8097 |

