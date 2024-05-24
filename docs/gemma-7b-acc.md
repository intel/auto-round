Install [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness.git) from source,  and the  git id 96d185fa6232a5ab685ba7c43e45d1dbb3bb906d, Install the latest [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) from source first

Please note that there is a discrepancy between the baseline result and the official data, which is a known issue within the official model card community.

```bash
lm_eval --model hf --model_args pretrained="Intel/gemma-7b-int4-inc",autogptq=True,gptq_use_triton=True --device cuda:0 --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,rte,arc_easy,arc_challenge,mmlu --batch_size 32
```

| Metric         | BF16   | int4   |
| -------------- |--------| ------ |
| Avg.           | 0.6239 | 0.6307 |
| mmlu           | 0.6162 | 0.6147 |
| lambada_openai | 0.6751 | 0.7204 |
| hellaswag      | 0.6047 | 0.5903 |
| winogrande     | 0.7324 | 0.7514 |
| piqa           | 0.7943 | 0.7949 |
| truthfulqa_mc1 | 0.3097 | 0.3011 |
| openbookqa     | 0.3320 | 0.3400 |
| boolq          | 0.8278 | 0.8269 |
| rte            | 0.6534 | 0.7076 |
| arc_easy       | 0.8178 | 0.7959 |
| arc_challenge  | 0.4991 | 0.4940 |
