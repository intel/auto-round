Install [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness.git) from source, we used the git id 96d185fa6232a5ab685ba7c43e45d1dbb3bb906d
##pip install auto-gptq[triton] 
##pip install triton==2.2.0
```bash
lm_eval --model hf --model_args pretrained="Intel/Mistral-7B-Instruct-v0.2-int4-inc",autogptq=True,gptq_use_triton=True --device cuda:0 --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,rte,arc_easy,arc_challenge,mmlu --batch_size 32
```



| Metric         | BF16   | INT4   |
| -------------- | ------ | ------ |
| Avg.           | 0.6647 | 0.6621 |
| mmlu           | 0.5906 | 0.5872 |
| lambada_openai | 0.7141 | 0.7141 |
| hellaswag      | 0.6602 | 0.6557 |
| winogrande     | 0.7395 | 0.7364 |
| piqa           | 0.8052 | 0.8047 |
| truthfulqa_mc1 | 0.5251 | 0.5153 |
| openbookqa     | 0.3600 | 0.3420 |
| boolq          | 0.8535 | 0.8541 |
| rte            | 0.7040 | 0.7148 |
| arc_easy       | 0.8161 | 0.8165 |
| arc_challenge  | 0.5435 | 0.5435 |
