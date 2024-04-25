### Evaluate the model 

Install [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness.git) from source,  and the git id we used is 96d185fa6232a5ab685ba7c43e45d1dbb3bb906d

pip install auto-gptq[triton] 
pip install triton==2.2.0

Please note that there is a discrepancy between the baseline result and the official data, which is a known issue within the official model card community.

```bash
lm_eval --model hf --model_args pretrained="Intel/gemma-2b-int4-inc",autogptq=True,gptq_use_triton=True --device cuda:0 --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,rte,arc_easy,arc_challenge,mmlu --batch_size 16
```

| Metric         | FP16   | INT4   |
| -------------- | ------ |--------|
| Avg.           | 0.5383 | 0.5338 |
| mmlu           | 0.3337 | 0.3276 |
| lambada_openai | 0.6398 | 0.6319 |
| hellaswag      | 0.5271 | 0.5161 |
| winogrande     | 0.6472 | 0.6472 |
| piqa           | 0.7699 | 0.7622 |
| truthfulqa_mc1 | 0.2203 | 0.2191 |
| openbookqa     | 0.3020 | 0.2980 |
| boolq          | 0.6939 | 0.6939 |
| rte            | 0.6426 | 0.6498 |
| arc_easy       | 0.7424 | 0.7348 |
| arc_challenge  | 0.4019 | 0.3908 |
