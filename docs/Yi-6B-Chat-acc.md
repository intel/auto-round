Due to licensing restrictions, we are unable to release the model. Install [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness.git) from source, and the git id 96d185fa6232a5ab685ba7c43e45d1dbb3bb906d.

We used the following command for evaluation.
For reference, the results of official AWQ-INT4 release are listed.

~~~bash
lm_eval --model hf  --model_args pretrained="./",autogptq=True,gptq_use_triton=True,trust_remote_code=True --device cuda:0 --tasks ceval-valid,cmmlu,mmlu,gsm8k --batch_size 16 --num_fewshot 0
~~~

| Metric | BF16   |[01-ai/Yi-6B-Chat-4bits](https://huggingface.co/01-ai/Yi-6B-Chat-4bits)| INT4   |
|--------|--------|----------------------|--------|
| Avg.   | 0.6043 | 0.5867               | 0.5939 |
| mmlu   | 0.6163 | 0.6133               | 0.6119 |
| cmmlu  | 0.7431 | 0.7312               | 0.7314 |
| ceval  | 0.7355 | 0.7155               | 0.7281 |
| gsm8k  | 0.3222 | 0.2866               | 0.3040 |
