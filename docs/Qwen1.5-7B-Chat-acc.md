Due to licensing restrictions, we are unable to release the model.

Install [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness.git) from source, and the git id 96d185fa6232a5ab685ba7c43e45d1dbb3bb906d.

We used the following command for evaluation.

~~~bash
'lm_eval --model hf \
    --model_args pretrained="gpu_model_path",autogptq=True,gptq_use_triton=True,trust_remote_code=True \
    --device cuda:0 --tasks ceval-valid,cmmlu,mmlu,gsm8k --batch_size 16 --num_fewshot 0'
~~~


| Metric         | BF16   | INT4 sym recipe | INT4 asym recipe |   
| -------------- | ------ |-----------------|------------------| 
| Avg.           | 0.6231 | 0.6205          | 0.6186           |  
| ceval          | 0.6887 | 0.6761          | 0.6820           |
| cmmlu          | 0.6959 | 0.6870          | 0.6884           |
| mmlu           | 0.6020 | 0.5974          | 0.5946           |
| gsm8k          | 0.5057 | 0.5216          | 0.5095           |
