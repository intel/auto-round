**This recipe is outdated, we recommend using symmetric quantization.** You can remove --asym from the command.
```bash
auto-round \
--model baichuan-inc/Baichuan2-7B-Chat  \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--nsamples 512 \
--asym \
--minmax_lr 2e-3 \
--format 'auto_gptq,auto_round' \
--output_dir "./tmp_autoround"
```



Due to licensing restrictions, we are unable to release the model. Install [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness.git) from source, and the git id 96d185fa6232a5ab685ba7c43e45d1dbb3bb906d.

We used the following command for evaluation.

~~~bash
lm_eval --model hf  --model_args pretrained="./",autogptq=True,gptq_use_triton=True,trust_remote_code=True --device cuda:0 --tasks ceval-valid,cmmlu,mmlu,gsm8k --batch_size 16 --num_fewshot 0
~~~

| Metric | BF16   | INT4   |
|--------|--------|--------|
| Avg.   | 0.4504 | 0.4470 |
| mmlu   | 0.5096 | 0.5053 |
| cmmlu  | 0.5486 | 0.5426 |
| ceval  | 0.5394 | 0.5223 |
| gsm8k  | 0.2039 | 0.2176 |
