Install [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness.git) from source, we used the git id f3b7917091afba325af3980a35d8a6dcba03dc3f

~~~bash
lm_eval  --model hf --model_args pretrained="Intel/neural-chat-v3-3-int4-inc",autogptq=True,gptq_use_triton=True --device cuda:0 --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,rte,arc_easy,arc_challenge,mmlu  --batch_size 128
~~~

| Metric         | FP16   | INT4   |
| -------------- | ------ | ------ |
| Avg.           | 0.6778 | 0.6748 |
| mmlu           | 0.5993 | 0.5926 |
| lambada_openai | 0.7303 | 0.7370 |
| hellaswag      | 0.6639 | 0.6559 |
| winogrande     | 0.7632 | 0.7735 |
| piqa           | 0.8101 | 0.8074 |
| truthfulqa_mc1 | 0.4737 | 0.4737 |
| openbookqa     | 0.3880 | 0.3680 |
| boolq          | 0.8694 | 0.8694 |
| rte            | 0.7581 | 0.7509 |
| arc_easy       | 0.8266 | 0.8249 |
| arc_challenge  | 0.5734 | 0.5691 |