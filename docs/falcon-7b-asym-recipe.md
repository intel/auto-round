**This recipe is outdated, we recommend using symmetric quantization.** You can remove --asym from the command.
```bash
auto-round \
--model tiiuae/falcon-7b   \
--device 0 \
--group_size 64 \
--bits 4 \
--iters 1000 \
--nsamples 512 \
--asym \
--format 'auto_gptq,auto_round' \
--output_dir "./tmp_autoround"
```


We generate the model with group_size 64 as there is an issue when evaluating with group_size 128.
Evaluate the model
pip3 install lm-eval==0.4.2

```bash
lm_eval --model hf --model_args pretrained="Intel/falcon-7b-int4-inc",autogptq=True,gptq_use_triton=True --device cuda:0 --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,arc_easy,arc_challenge,mmlu --batch_size 16
```

| Metric         | BF16   | int4   |
| -------------- | ------ | ------ |
| Avg.           | 0.5462 | 0.5454 |
| mmlu           | 0.2546 | 0.2562 |
| lambada_openai | 0.7450 | 0.7485 |
| hellaswag      | 0.5773 | 0.5719 |
| winogrande     | 0.6740 | 0.6835 |
| piqa           | 0.7943 | 0.7905 |
| truthfulqa_mc1 | 0.2228 | 0.2166 |
| openbookqa     | 0.3080 | 0.3100 |
| boolq          | 0.7361 | 0.7431 |
| arc_easy       | 0.7475 | 0.7424 |
| arc_challenge  | 0.4027 | 0.3908 |

