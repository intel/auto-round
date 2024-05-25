Due to licensing restrictions, we are unable to release the model. lm-eval 0.4.2 is used

For evaluating w4g128 without quantized lm-head, 
```bash
lm_eval --model hf --model_args pretrained="./",autogptq=True,gptq_use_triton=True --device cuda:0 --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,rte,arc_easy,arc_challenge,mmlu --batch_size 16
```

for evaluation with quantized lm-head
```bash
git clone https://github.com/intel/auto-round
cd auto-round/examples/language-modeling
python3 eval_042/evluation.py --model_name "./" --eval_bs 16
```

| Metric           | **BF16** | w4g128 w/o lm-head | w4g128 with lm-head |
| ---------------- | :------- |--------------------|-----------------------------|
| Avg.             | 0.6352   | 0.6312             | 0.6303                      |
| mmlu             | 0.6386   | 0.6306             | 0.6243                     |
| winogrande       | 0.7143   | 0.7238             | 0.7261                      |
| truthfulqa_mc1   | 0.3623   | 0.3537             | 0.3574                     |
| rte              | 0.6751   | 0.6859             | 0.6715                      |
| piqa             | 0.7867   | 0.7797             | 0.7775                     |
| openbookqa       | 0.3400   | 0.3300             | 0.3340                      |
| lambada_openai   | 0.7182   | 0.7200             | 0.7118                      |
| hellaswag        | 0.5769   | 0.5699             | 0.5686                     |
| boolq            | 0.8297   | 0.8309             | 0.8266                     |
| arc_easy         | 0.8152   | 0.8089             | 0.8123                      |
| arc_challenge    | 0.5299   | 0.5102             |  0.5111                          |
