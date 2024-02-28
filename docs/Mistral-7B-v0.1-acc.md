Install [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness.git) from source, we used the git id f3b7917091afba325af3980a35d8a6dcba03dc3f

```bash
lm_eval --model hf --model_args pretrained="Intel/Mistral-7B-v0.1-int4-inc",autogptq=True,gptq_use_triton=True --device cuda:0 --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,rte,arc_easy,arc_challenge,mmlu --batch_size 32
```



| Metric         | FP16   | INT4   |
| -------------- | ------ | ------ |
| Avg.           | 0.6306 | 0.6308 |
| mmlu           | 0.5961 | 0.5880 |
| lambada_openai | 0.7561 | 0.7551 |
| hellaswag      | 0.6128 | 0.6079 |
| winogrande     | 0.7443 | 0.7451 |
| piqa           | 0.8079 | 0.8014 |
| truthfulqa_mc1 | 0.2803 | 0.2889 |
| openbookqa     | 0.3280 | 0.3300 |
| boolq          | 0.8373 | 0.8278 |
| rte            | 0.6643 | 0.6968 |
| arc_easy       | 0.8085 | 0.8060 |
| arc_challenge  | 0.5009 | 0.4915 |

