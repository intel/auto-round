### Evaluate the model 

Install [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness.git) from source, lm-eval 0.4.2 is used

pip install auto-gptq


Please note that there is a discrepancy between the baseline result and the official data, which is a known issue within the official model card community.
Given that the Gemma model family exhibits inconsistent results between FP16 and BF16 on lm-eval, we recommend converting to FP16 for both tuning and evaluation.

```bash
lm_eval --model hf --model_args pretrained="Intel/gemma-2b-int4-inc",autogptq=True,gptq_use_triton=True,dtype=float16 --device cuda:0 --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,rte,arc_easy,arc_challenge,mmlu --batch_size 16
```



| Metric         | BF16 | FP16   | AutoRound v0.1 | AutoRound v0.2 |
| -------------- | ---- | ------ |----------------|----------------|
| Avg.| 0.5263 | 0.5277 | 0.5235         | 0.5248         |
| mmlu           | 0.3287 | 0.3287 | 0.3297         | 0.3309         |
| lambada_openai | 0.6344 | 0.6375 | 0.6307         | 0.6379         |
| hellaswag      | 0.5273 | 0.5281 | 0.5159         | 0.5184         |
| winogrande     | 0.6504 | 0.6488 | 0.6543         | 0.6575         |
| piqa           | 0.7671 | 0.7720 | 0.7612         | 0.7606         |
| truthfulqa_mc1 | 0.2203 | 0.2203 | 0.2203         | 0.2191         |
| openbookqa     | 0.2980 | 0.3020 | 0.3000         | 0.3060         |
| boolq          | 0.6927 | 0.6936 | 0.6939         | 0.6966         |
| arc_easy       | 0.7420 | 0.7403 | 0.7353         | 0.7357         |
| arc_challenge  | 0.4019 | 0.4061 | 0.3933         | 0.3857         |
