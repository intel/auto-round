Install [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness.git) from source,  and the  git id f3b7917091afba325af3980a35d8a6dcba03dc3f is used

Download the model from hf(coming soon) or follow examples/language-modeling/scripts/phi-2.sh to generate the model

Since we encountered an issue evaluating this model with lm-eval, we opted to evaluate the qdq model instead. In our assessment, we found that its accuracy closely matches that of the real quantized model in most cases except for some small models like opt-125m.



| Metric         | FP16   | INT4 qdq |
| -------------- | ------ | -------- |
| Avg.           | 0.6155 | 0.6163   |
| mmlu           | 0.5448 | 0.5417   |
| lambada_openai | 0.6268 | 0.6225   |
| hellaswag      | 0.5585 | 0.5498   |
| winogrande     | 0.7530 | 0.7545   |
| piqa           | 0.7867 | 0.7824   |
| truthfulqa_mc1 | 0.3133 | 0.3060   |
| openbookqa     | 0.4000 | 0.4100   |
| boolq          | 0.8339 | 0.8327   |
| rte            | 0.6245 | 0.6643   |
| arc_easy       | 0.7997 | 0.7955   |
| arc_challenge  | 0.5290 | 0.5196   |