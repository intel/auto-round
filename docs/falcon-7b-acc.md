Install [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness.git) from source,  and the git id 96d185fa6232a5ab685ba7c43e45d1dbb3bb906d, 

pip install auto-gptq[triton]  

pip install triton==2.2.0

Since we encountered an issue evaluating this model with lm-eval, we opted to evaluate the qdq model instead. In our assessment, we found that its accuracy closely matches that of the real quantized model in most cases except for some small models like opt-125m. The batch size 32 is used.

| Metric         | FP16   | int4 qdq |
| -------------- | ------ | -------- |
| Avg.           | 0.5521 | 0.5507   |
| mmlu           | 0.2495 | 0.2427   |
| lambada_openai | 0.7452 | 0.7487   |
| hellaswag      | 0.5771 | 0.5731   |
| winogrande     | 0.6725 | 0.6756   |
| piqa           | 0.7949 | 0.7943   |
| truthfulqa_mc1 | 0.2252 | 0.2142   |
| openbookqa     | 0.3060 | 0.3060   |
| boolq          | 0.7364 | 0.7382   |
| rte            | 0.6173 | 0.6245   |
| arc_easy       | 0.7479 | 0.7433   |
| arc_challenge  | 0.4019 | 0.3968   |
