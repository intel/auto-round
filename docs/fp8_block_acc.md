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
|BF16            |0.5358       |0.8232  |0.8459|0.5795   |0.7266        |0.3540    |0.7786|0.3721        |0.7135    |0.7369     |0.6741    |0.4334        |0.6311|
|FP8_BLOCK RTN   |0.5316       |0.8173  |0.8468|0.5787   |0.7256        |0.3560    |0.7824|0.3684        |0.7103    |0.7369     |0.6661    |0.4358        |0.6297|
|FP8_BLOCK Tuning|0.5401       |0.8228  |0.8459|0.5786   |0.7275        |0.3480    |0.7769|0.3684        |0.7103    |0.7437     |0.6727    |0.4315        |0.6305|


## Qwen3-8B

|Scheme          |arc_challenge|arc_easy|boolq |hellaswag|lambada_openai|openbookqa|piqa  |truthfulqa_mc1|winogrande|gsm8k |mmlu  |mmlu_pro|Avg.  |
|:---------------|:------------|:-------|:-----|:--------|:-------------|:---------|:-----|:-------------|:---------|:-----|:-----|:-------|:-----|
|BF16            |0.5580       |0.8342  |0.8667|0.5712   |0.6517        |0.3120    |0.7688|0.3647        |0.6788    |0.8726|0.7292|0.6214  |0.6524|
|FP8_BLOCK RTN   |0.5538       |0.8350  |0.8691|0.5713   |0.6468        |0.3140    |0.7661|0.3721        |0.6780    |0.8704|0.7269|0.6204  |0.6520|
|FP8_BLOCK Tuning|0.5520       |0.8384  |0.8703|0.5710   |0.6519        |0.3260    |0.7671|0.3635        |0.6725    |0.8681|0.7302|0.6168  |0.6523|


