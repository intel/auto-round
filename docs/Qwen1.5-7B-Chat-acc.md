Due to licensing restrictions, we are unable to release the model.

Install [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness.git) from source, and the git id 96d185fa6232a5ab685ba7c43e45d1dbb3bb906d.



| Metric         | FP16   | [INT8-gptq](https://huggingface.co/Qwen/Qwen1.5-7B-Chat-GPTQ-Int8) | [INT4-gptq](https://huggingface.co/Qwen/Qwen1.5-7B-Chat-GPTQ-Int4) |   INT4 recipe1   |   INT4 recipe2   |
| -------------- | ------ | --------  | --------- | -----------------| -----------------|
| Avg.           | 0.6220 |  0.6218   |  0.6064   |     0.6207       |     0.6185       |
| ceval          | 0.6988 |  0.6956   |  0.6763   |     0.6855       |     0.6913       |
| cmmlu          | 0.6879 |  0.6876   |  0.6729   |     0.6783       |     0.6776       |
| mmlu           | 0.6016 |  0.6013   |  0.5902   |     0.5966       |     0.5970       |
| gsm8k          | 0.4996 |  0.5027   |  0.4860   |     0.5224       |     0.5080       |

