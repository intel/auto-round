Below accuracy results are got by lm-eval.

cmd:

```bash
# model quantization
auto-round --model model_name_or_path --scheme FP8_BLOCK --iters 0 --format fp8 # RTN
auto-round --model model_name_or_path --scheme FP8_BLOCK --format fp8 # Tuning


# accuracy evaluation
# --apply_chat_template --fewshot_as_multiturn are required for gsm8k_llama,mmlu_llama,mmlu_pro_llama tasks
# `add_bos_token=true` is only required in model_args for llama evaluation
lm_eval --model vllm --model_args pretrained=model_path,add_bos_token=true,tensor_parallel_size=1,gpu_memory_utilization=0.8,max_model_len=8192 --tasks tasks --batch_size 128
```

## LLaMa-3-8B-Instruct

|Scheme          |arc_challenge|arc_easy|boolq |hellaswag|lambada_openai|openbookqa|piqa  |truthfulqa_mc1|winogrande|gsm8k_llama|mmlu_llama|mmlu_pro_llama|Avg.  |
|:---------------|:------------|:-------|:-----|:--------|:-------------|:---------|:-----|:-------------|:---------|:----------|:---------|:-------------|:-----|
|BF16            |0.5341       |0.8232  |0.8428|0.5805   |0.7258        |0.3540    |0.7758|0.3709        |0.7151    |0.6111     |0.6737    |0.3778        |0.6145|
|FP8_BLOCK RTN   |0.5239       |0.8182  |0.8483|0.5780   |0.7235        |0.3520    |0.7840|0.3647        |0.7040    |0.6171     |0.6681    |0.3738        |0.6130|
|FP8_BLOCK Tuning|0.5213       |0.8047  |0.8376|0.5834   |0.7099        |0.3480    |0.7764|0.3684        |0.7245    |0.6224     |0.6654    |0.3690        |0.6109|


## Qwen3-8B

|Scheme          |arc_challenge|arc_easy|boolq |hellaswag|lambada_openai|openbookqa|piqa  |truthfulqa_mc1|winogrande|gsm8k |mmlu  |mmlu_pro|Avg.  |
|:---------------|:------------|:-------|:-----|:--------|:-------------|:---------|:-----|:-------------|:---------|:-----|:-----|:-------|:-----|
|BF16            |0.5589       |0.8329  |0.8661|0.5719   |0.6501        |0.3140    |0.7688|0.3647        |0.6772    |0.8726|0.7297|0.6218  |0.6524|
|FP8_BLOCK RTN   |0.5444       |0.8350  |0.8679|0.5704   |0.6466        |0.3060    |0.7644|0.3611        |0.6914    |0.8741|0.7268|0.6242  |0.6510|
|FP8_BLOCK Tuning|0.5486       |0.8308  |0.8639|0.5691   |0.6513        |0.3080    |0.7622|0.3586        |0.6882    |0.8673|0.7272|0.6139  |0.6491|


