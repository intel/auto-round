Install [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness.git) from source, we used 0.4.2

```bash
lm_eval --model hf --model_args pretrained="Intel/Mistral-7B-v0.1-int4-inc",autogptq=True,gptq_use_triton=True --device cuda:0 --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,rte,arc_easy,arc_challenge,mmlu --batch_size 32
```




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
