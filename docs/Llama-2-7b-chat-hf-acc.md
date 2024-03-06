Due to licensing restrictions, we are unable to release the model.

Install [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness.git) from source, and the git id 96d185fa6232a5ab685ba7c43e45d1dbb3bb906d.

Since we encountered an issue evaluating this model with lm-eval, we opted to evaluate the qdq model instead. In our assessment, we found that its accuracy closely matches that of the real quantized model in most cases except for some small models like opt-125m.


| Metric         | FP16   | int4 qdq |
| -------------- | ------ | -------- |
| Avg.           | 0.5901 | 0.5897   |
| mmlu           | 0.4640 | 0.4545   |
| lambada_openai | 0.7105 | 0.7037   |
| hellaswag      | 0.5780 | 0.5706   |
| winogrande     | 0.6638 | 0.6614   |
| piqa           | 0.7639 | 0.7633   |
| truthfulqa_mc1 | 0.3023 | 0.3035   |
| openbookqa     | 0.3340 | 0.3260   |
| boolq          | 0.7976 | 0.8064   |
| rte            | 0.6968 | 0.7292   |
| arc_easy       | 0.7382 | 0.7336   |
| arc_challenge  | 0.4420 | 0.4352   |
