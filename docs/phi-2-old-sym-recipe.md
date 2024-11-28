 **This recipe is outdated, we recommend using the latest full range symmetric quantization.** You can remove --asym from the command.

A sample command to generate an INT4 model.
```bash
auto-round \
--model   facebook/opt-2.7b \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--nsamples 512 \
--format 'auto_gptq,auto_round' \
--output_dir "./tmp_autoround"
```


pip install lm-eval==0.4.2

Due to the significant accuracy drop with the asymmetric kernel for this model, we opted to use symmetric quantization.

```bash
lm_eval --model hf --model_args pretrained="Intel/phi-2-int4-inc" --device cuda:0 --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,arc_easy,arc_challenge,mmlu --batch_size 16
```

| Metric         | FP16   | INT4  |
| -------------- | ------ | -------- |
| Avg.           | 0.6155 | 0.6163   |
| mmlu           | 0.5448 | 0.5417   |
| lambada_openai | 0.6268 | 0.6225   |
| hellaswag      | 0.5585 | 0.5498   |
| winogrande     | 0.7530 | 0.7545   |
| piqa           | 0.7867 | 0.7824   |
| truthfulqa_mc1 | 0.3133 | 0.3060   |
| openbookqa     | 0.4000 | 0.4100   |
| boolq          | 0.8339 | 0.8327   |
| rte            | 0.6245 | 0.6643   |
| arc_easy       | 0.7997 | 0.7955   |
| arc_challenge  | 0.5290 | 0.5196   |
