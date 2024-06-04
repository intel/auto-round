pip install lm-eval==0.4.2
pip install auto-gptq

Please note that there is a discrepancy between the baseline result and the official data, which is a known issue within the official model card community.

Given that the Gemma model family exhibits inconsistent results between FP16 and BF16 on lm-eval, we recommend converting to FP16 for both tuning and evaluation.
```bash
lm_eval --model hf --model_args pretrained="Intel/gemma-7b-int4-inc",autogptq=True,gptq_use_triton=True,dtype=float16 --device cuda:0 --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,rte,arc_easy,arc_challenge,mmlu --batch_size 32
```
| Metric         | BF16 | FP16   | AutoRound v0.1 | AutoRound V0.2 |
| -------------- | ---- | ------ |----------------|----------------|
| Avg. | 0.6208 | 0.6302 | 0.6242         | 0.6254         |
| mmlu           | 0.6126 | 0.6189 | 0.6085         | 0.6147         |
| lambada_openai | 0.6707 | 0.7308 | 0.7165         | 0.7270         |
| hellaswag      | 0.6039 | 0.6063 | 0.6017         | 0.6017         |
| winogrande     | 0.7356 | 0.7506 | 0.7482         | 0.7490         |
| piqa           | 0.8014 | 0.8025 | 0.7976         | 0.7982         |
| truthfulqa_mc1 | 0.3121 | 0.3121 | 0.3060         | 0.2840         |
| openbookqa     | 0.3300 | 0.3220 | 0.3340         | 0.3240         |
| boolq          | 0.8254 | 0.8324 | 0.8300         | 0.8407         |
| rte            | 0.6643 | 0.6859 | 0.6787         | 0.6968         |
| arc_easy       | 0.8068 | 0.8262 | 0.8089         | 0.8194         |
| arc_challenge  | 0.5043 | 0.5000 | 0.4915         | 0.4949         |
