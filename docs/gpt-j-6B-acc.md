Install [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness.git) from source, we used the git id 96d185fa6232a5ab685ba7c43e45d1dbb3bb906d
##pip install auto-gptq[triton] 
##pip install triton==2.2.0
```bash
lm_eval --model hf --model_args pretrained="./",autogptq=True,gptq_use_triton=True --device cuda:0 --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,rte,arc_easy,arc_challenge,mmlu --batch_size 32
```



| Metric         | FP16   | INT4   |
| -------------- | ------ | ------ |
| Avg.           | 0.5039 | 0.5034 |
| mmlu           | 0.2694 | 0.2793 |
| lambada_openai | 0.6831 | 0.6790 |
| hellaswag      | 0.4953 | 0.4902 |
| winogrande     | 0.6409 | 0.6401 |
| piqa           | 0.7541 | 0.7465 |
| truthfulqa_mc1 | 0.2020 | 0.2179 |
| openbookqa     | 0.2900 | 0.2900 |
| boolq          | 0.6544 | 0.6554 |
| rte            | 0.5451 | 0.5271 |
| arc_easy       | 0.6692 | 0.6734 |
| arc_challenge  | 0.3396 | 0.3387 |
