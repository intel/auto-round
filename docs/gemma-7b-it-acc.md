Install [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness.git) from source,  and the  git id 96d185fa6232a5ab685ba7c43e45d1dbb3bb906d, Install the latest [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) from source first

Please note that there is a discrepancy between the baseline result and the official data, which is a known issue within the official model card community.

```bash
lm_eval --model hf --model_args pretrained="Intel/gemma-7b-it-int4-inc",autogptq=True,gptq_use_triton=True --device cuda:0 --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,rte,arc_easy,arc_challenge,mmlu --batch_size 32
```

| Metric         | FP16   | int4   |
| -------------- | ------ | ------ |
| Avg.           | 0.6022 | 0.6017 |
| mmlu           | 0.5029 | 0.4993 |
| lambada_openai | 0.6035 | 0.6286 |
| hellaswag      | 0.5620 | 0.5564 |
| winogrande     | 0.6796 | 0.6788 |
| piqa           | 0.7709 | 0.7731 |
| truthfulqa_mc1 | 0.3048 | 0.3035 |
| openbookqa     | 0.3740 | 0.3700 |
| boolq          | 0.8138 | 0.8144 |
| rte            | 0.7870 | 0.7870 |
| arc_easy       | 0.7525 | 0.7508 |
| arc_challenge  | 0.4727 | 0.4573 |
