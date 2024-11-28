**This recipe is outdated, we recommend using symmetric quantization.** You can remove --asym from the command.

A sample command to generate an INT4 model. 
```bash
auto-round \
--model  mistralai/Mistral-7B-Instruct-v0.2 \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--nsamples 512 \
--asym \
--format 'auto_gptq,auto_round' \
--output_dir "./tmp_autoround"
```

| Metric         | BF16   | INT4   |
| -------------- | ------ | ------ |
| Avg.           | 0.6647 | 0.6621 |
| mmlu           | 0.5906 | 0.5872 |
| lambada_openai | 0.7141 | 0.7141 |
| hellaswag      | 0.6602 | 0.6557 |
| winogrande     | 0.7395 | 0.7364 |
| piqa           | 0.8052 | 0.8047 |
| truthfulqa_mc1 | 0.5251 | 0.5153 |
| openbookqa     | 0.3600 | 0.3420 |
| boolq          | 0.8535 | 0.8541 |
| rte            | 0.7040 | 0.7148 |
| arc_easy       | 0.8161 | 0.8165 |
| arc_challenge  | 0.5435 | 0.5435 |

