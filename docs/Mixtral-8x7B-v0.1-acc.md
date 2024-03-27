### Evaluate the model
Install [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness.git) from source, we used the git id f3b7917091afba325af3980a35d8a6dcba03dc3f

Download the model from hf(coming soon) or follow examples/language-modeling/scripts/Mixtral-8x7B-v0.1.sh to generate the model

~~~bash
lm_eval --model hf --model_args pretrained="Intel/Mixtral-8x7B-v0.1-int4-inc",autogptq=True,gptq_use_triton=True --device cuda:0 --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,rte,arc_easy,arc_challenge,mmlu --batch_size 32
~~~

| Metric         | BF16   | INT4   |
| -------------- |--------| ------ |
| Avg.           | 0.6698 | 0.6633 |
| mmlu           | 0.6802 | 0.6693 |
| lambada_openai | 0.7827 | 0.7825 |
| hellaswag      | 0.6490 | 0.6459 |
| winogrande     | 0.7648 | 0.7514 |
| piqa           | 0.8248 | 0.8210 |
| truthfulqa_mc1 | 0.3427 | 0.3219 |
| openbookqa     | 0.3540 | 0.3560 |
| boolq          | 0.8523 | 0.8474 |
| rte            | 0.7076 | 0.6931 |
| arc_easy       | 0.8430 | 0.8430 |
| arc_challenge  | 0.5666 | 0.5648 |
