Install [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness.git) from source, we used the git id 96d185fa6232a5ab685ba7c43e45d1dbb3bb906d
##pip install auto-gptq[triton] 
##pip install triton==2.2.0
```bash
lm_eval --model hf --model_args pretrained="./",autogptq=True,gptq_use_triton=True --device cuda:0 --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,rte,arc_easy,arc_challenge,mmlu --batch_size 32
```



| Metric         | FP16   | INT4   |
| -------------- | ------ | ------ |
| Avg.           | 0.4722 | 0.4757 |
| mmlu           | 0.2568 | 0.2636 |
| lambada_openai | 0.6359 | 0.6487 |
| hellaswag      | 0.4585 | 0.4519 |
| winogrande     | 0.6077 | 0.5967 |
| piqa           | 0.7367 | 0.7410 |
| truthfulqa_mc1 | 0.2240 | 0.2338 |
| openbookqa     | 0.2500 | 0.2380 |
| boolq          | 0.6046 | 0.6505 |
| rte            | 0.5451 | 0.5379 |
| arc_easy       | 0.6077 | 0.6035 |
| arc_challenge  | 0.2679 | 0.2671 |
