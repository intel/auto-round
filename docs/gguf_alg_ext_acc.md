We use **lm-eval** for evaluation. For LLaMA, we enabled `add_bos_token` and
`removed @use_kernel_forward_from_hub("RMSNorm")`
in [modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L52C1-L52C40)
to stabilize accuracy during evaluation. All other settings follow the default configurations of AutoRound and lm-eval.

*Average accuracy across `lambada_openai`, `hellaswag`, `piqa`, `winogrande`, `truthfulqa_mc1`, `openbookqa`, `boolq`, `arc_easy`,	`arc_challenge` and `mmlu`.*

|method|scheme|Llama-3.1-8B|Qwen2.5-7B-Instruct|Qwen3-8b|Qwen3-30B-A3B-Instruct-2507|
|:-----|:-----|:-----------|:------------------|:-------|:--------------------------|
|**BF16**  | -    |0.6295(100%)|0.6571(100%)       |0.6322(100%)|0.6746(100%)           |
| **original** | q2_k_s | 0.5535(87.92%)| 0.6266(95.35%)|0.5901(93.35%)|0.6386(94.66%)|
| **enable_alg_ext** |q2_k_s|0.5740(91.18%)|0.6349(96.62%)|0.5962(94.31%)|0.6460(95.77%)|
| **original**  | q3_k_s | 0.6040(95.95%)|0.6382(97.12%)|0.6128(96.94%)|0.6598(97.82%)|
| **enable_alg_ext** |q3_k_s|0.6081(96.59%)|0.6503(98.97%)|0.6252(98.89%)|0.6622(98.17%)|
| **original**  | q4_k_s | 0.6228(98.94%)|0.6560(99.83%)|0.6303(99.70%)|0.6762(100.24%)|
| **enable_alg_ext** |q4_k_s|0.6239(99.11%)|0.6605(100.51%)|0.6320(99.98%)|0.6777(100.46%)|