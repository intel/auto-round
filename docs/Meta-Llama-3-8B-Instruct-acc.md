Due to licensing restrictions, we are unable to release the model.

lm-eval 0.4.2 is used

For evaluating w4g128 without quantized lm-head, 
```bash
lm_eval --model hf --model_args pretrained="./",autogptq=True,gptq_use_triton=True --device cuda:0 --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,rte,arc_easy,arc_challenge,mmlu --batch_size 16
```

For evaluating w4g128 with quantized lm-head, we opted to evaluate the qdq model instead,since we encountered an issue evaluating the model with lm-head quantized model. In our assessment, we found that its accuracy closely matches that of the real quantized model in most cases except for some small models like opt-125m.


| Metric           | **BF16** | w4g128 w/o lm-head | w4g128 with lm-head qdq |
| ---------------- | :------- |--------------------|---------------------------------|
| Avg.             | 0.6352   | 0.6312             | 0.6303                          |
| mmlu             | 0.6386   | 0.6306             | 0.6318                          |
| winogrande       | 0.7143   | 0.7238             | 0.7269                          |
| truthfulqa_mc1   | 0.3623   | 0.3537             | 0.3525                          |
| rte              | 0.6751   | 0.6859             | 0.6679                          |
| piqa             | 0.7867   | 0.7797             | 0.7802                          |
| openbookqa       | 0.3400   | 0.3300             | 0.3320                          |
| lambada_openai   | 0.7182   | 0.7200             | 0.7173                          |
| hellaswag        | 0.5769   | 0.5699             | 0.5701                          |
| boolq            | 0.8297   | 0.8309             | 0.8284                          |
| arc_easy         | 0.8152   | 0.8089             | 0.8106                          |
| arc_challenge    | 0.5299   | 0.5102             |  0.5154                              |
