Install [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness.git) from source, we used the git id f3b7917091afba325af3980a35d8a6dcba03dc3f

~~~bash
lm_eval  --model hf --model_args pretrained="Intel/neural-chat-v3-1-int4-inc",autogptq=True,gptq_use_triton=True --device cuda:0 --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,rte,arc_easy,arc_challenge,mmlu  --batch_size 128
~~~

| Metric         | FP16   | INT4   |
| -------------- | ------ | ------ |
| Avg.           | 0.6769 | 0.6721 |
| mmlu           | 0.5919 | 0.5862 |
| lambada_openai | 0.7394 | 0.7337 |
| hellaswag      | 0.6323 | 0.6272 |
| winogrande     | 0.7687 | 0.7577 |
| piqa           | 0.8161 | 0.8150 |
| truthfulqa_mc1 | 0.4431 | 0.4394 |
| openbookqa     | 0.3760 | 0.3700 |
| boolq          | 0.8783 | 0.8743 |
| rte            | 0.7690 | 0.7726 |
| arc_easy       | 0.8413 | 0.8384 |
| arc_challenge  | 0.5896 | 0.5785 |
