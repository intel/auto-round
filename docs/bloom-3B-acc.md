Install [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness.git) from source, we used the git id 96d185fa6232a5ab685ba7c43e45d1dbb3bb906d
##pip install auto-gptq[triton] 
##pip install triton==2.2.0
```bash
lm_eval --model hf --model_args pretrained="./",autogptq=True,gptq_use_triton=True --device cuda:0 --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,rte,arc_easy,arc_challenge,mmlu --batch_size 32
```



| Metric         | FP16   | INT4   |
| -------------- | ------ | ------ |
| Avg.           | 0.4532 | 0.4514 |
| mmlu           | 0.2592 | 0.2537 |
| lambada_openai | 0.5176 | 0.5135 |
| hellaswag      | 0.4136 | 0.4093 |
| winogrande     | 0.5864 | 0.5856 |
| piqa           | 0.7062 | 0.7095 |
| truthfulqa_mc1 | 0.2326 | 0.2264 |
| openbookqa     | 0.2160 | 0.2140 |
| boolq          | 0.6156 | 0.6199 |
| rte            | 0.5632 | 0.5632 |
| arc_easy       | 0.5947 | 0.5888 |
| arc_challenge  | 0.2799 | 0.2816 |
