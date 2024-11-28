 **This recipe is outdated, we recommend using symmetric quantization.** You can remove --asym from the command.
 

A sample command to generate an INT4 model.
```bash
auto-round \
--model  mistralai/Mistral-7B-v0.1 \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--nsamples 512 \
--asym \
--format 'auto_gptq,auto_round' \
--output_dir "./tmp_autoround"
```

quant_lm_head

```bash
auto-round \
--model  mistralai/Mistral-7B-v0.1 \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--nsamples 512 \
--asym \
--quant_lm_head \
--format 'auto_gptq,auto_round' \
--output_dir "./tmp_autoround"
```

lm-eval 0.4.2 is used

| Metric         | BF16   | [INT4-lmhead](https://huggingface.co/Intel/Mistral-7B-v0.1-int4-inc-lmhead) | [INT4](https://huggingface.co/Intel/Mistral-7B-v0.1-int4-inc) |
| -------------- | ------ |-----------------| ------------------------------------------------------------ |
| Avg.           | 0.6260 | 0.6228          | 0.6218                                                       |
| mmlu           | 0.5868 | 0.5760          | 0.5772                                                       |
| lambada_openai | 0.7555 | 0.7539          | 0.7543                                                       |
| hellaswag      | 0.6125 | 0.6055          | 0.6072                                                       |
| winogrande     | 0.7395 | 0.7380          | 0.7388                                                       |
| piqa           | 0.8069 | 0.8009          | 0.8030                                                       |
| truthfulqa_mc1 | 0.2803 | 0.2876          | 0.2864                                                       |
| openbookqa     | 0.3280 | 0.3300          | 0.3260                                                       |
| boolq          | 0.8379 | 0.8291          | 0.8281                                                       |
| arc_easy       | 0.8089 | 0.8043          | 0.8035                                                       |
| arc_challenge  | 0.5034 | 0.5026          | 0.4932                                                       |
