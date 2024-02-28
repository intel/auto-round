Install [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness.git) from source,  and the  git id f3b7917091afba325af3980a35d8a6dcba03dc3f is used

Download the model from hf(coming soon) or follow examples/language-modeling/scripts/Mixtral-8x7B-Instruct-v0.1.sh to generate the model

```bash
lm_eval --model hf --model_args pretrained="Intel/Mixtral-8x7B-Instruct-v0.1-int4-inc",autogptq=True,gptq_use_triton=True --device cuda:0 --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,rte,arc_easy,arc_challenge,mmlu --batch_size 32
```

| Metric         | FP16   | INT4   |
| -------------- | ------ | ------ |
| Avg.           | 0.7000 | 0.6977 |
| mmlu           | 0.6885 | 0.6824 |
| lambada_openai | 0.7718 | 0.7790 |
| hellaswag      | 0.6767 | 0.6745 |
| winogrande     | 0.7687 | 0.7719 |
| piqa           | 0.8351 | 0.8335 |
| truthfulqa_mc1 | 0.4969 | 0.4884 |
| openbookqa     | 0.3680 | 0.3720 |
| boolq          | 0.8850 | 0.8783 |
| rte            | 0.7184 | 0.7004 |
| arc_easy       | 0.8699 | 0.8712 |
| arc_challenge  | 0.6220 | 0.6229 |