Step-by-Step
============

This document presents step-by-step instructions for auto-round.

# Prerequisite


## 1. Prepare Calibration Dataset

### Default Dataset
The [NeelNanda/pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k) in huggingface is adopted as the default calibration data and  will be downloaded automatically from the datasets Hub. To customize a dataset, please kindly follow our dataset code.
See more about loading [huggingface dataset](https://huggingface.co/docs/datasets/main/en/quickstart)

### Customized Dataset
- Option 1: Pass a local json file path to dataset argument
- Option 2: Register your dataset following the [code](../../auto_round/calib_dataset.py) and pass the new dataset and split args to initialize AutoRound object, e.g. autoround=Autoround(dataset="NeelNanda/pile-10k:train", ...)
- Option 3: pass list of string or list of input_ids to dataset.
~~~python
def customized_data():
    ##Important Notice!!! Autoround will drop data < args.seqlen and truncate data to args.seqlen
    data = ["AutoRound is an advanced weight-only quantization algorithm for low-bits LLM inference" * 240]
    data.append("AutoRound is an advanced weight-only quantization algorithm for low-bits LLM inference")
    return data


def customized_data_with_tokenizer(tokenizer, seqlen=2048):
    ##Import notice!!! Autoround will drop data < args.seqlen
    data = ["AutoRound is an advanced weight-only quantization algorithm for low-bits LLM inference" * 240]
    data.append("AutoRound is an advanced weight-only quantization algorithm for low-bits LLM inference")
    tokens = []
    for d in data:
        token = tokenizer(d, truncation=True, max_length=seqlen, return_tensors="pt").data
        tokens.append(token)
    return tokens
~~~

Combination of different datasets has been supported, --dataset "./tmp.json,NeelNanda/pile-10k:train, mbpp:train+validation+test". Please note that samples with sequence length < args.seqlen will be dropped.

<br />

## 2. Run Examples
Enter into the examples folder and install lm-eval to run the evaluation
```bash
pip install -r requirements.txt
```

- **Default Settings:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --model_name facebook/opt-125m  --bits 4 --group_size 128
```
- **Reduced GPU Memory Usage:**

set "--train_bs 1 --gradient_accumulate_steps 8" (more tuning cost)

reduce the train bs to 4(potential accuracy drop) 
- **Speedup the tuning:**

disable_low_gpu_mem_usage(more gpu memory)

reduce the train bs to 4(little accuracy drop) 

reduce the seqlen to 512(potential large accuracy drop)

or combine them

- **Enable quantized lm-head:**

Currently only support in Intel xpu,however, we found the fake tuning could improve the accuracy is some scenarios. --disable_low_gpu_mem_usage is strongly recommended if the whole model could be loaded to the device, otherwise it will be quite slow to cache the inputs of lm-head. Another way is reducing n_samples,e.g. 128, to alleviate the issue.
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --model_name facebook/opt-125m  --bits 4 --group_size 128 --quant_lm_head --disable_low_gpu_mem_usage
```

- **Utilizing the AdamW Optimizer:**

Include the flag `--adam`. Note that AdamW is less effective than sign gradient descent in many scenarios we tested.

- **Running the Original SignRound:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --model_name facebook/opt-125m  --bits 4 --group_size 128 --iters 400 --lr 0.0025 --disable_minmax_tuning --disable_quanted_input
```

- **Code generation LLM:**

We utilized mbpp for calibration, but your own training dataset is highly recommended. Please note that samples with seqlen < args.seqlen will be dropped in current version.
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --model_name Salesforce/codegen25-7b-multi --bits 4 --group_size 128 --dataset "mbpp" --seqlen 128 "
```
- **Running on Intel Gaudi2**
```bash
bash run_autoround_on_gaudi.sh 
```



## 3. Evaluation
The example supports evaluation for various tasks in lm_eval. Moreover, it facilitates separate evaluation through the 'evaluation.py' script, which extends support to three additional tasks (ptb, c4, and wikitext2) beyond the capabilities of the official lm_eval. Additionally, evaluation results will be neatly organized into an Excel file for ease of demonstration.

For large models, GPU memory may be insufficient. Enable multi-GPU evaluation by setting 'CUDA_VISIBLE_DEVICES'.

Due to the large size of the model, the quantization and evaluation processes may be time-consuming. To provide flexibility in the process, two options are offered:

- You can set up multi-GPU cards for the quantization example, which will only use the first card for quantization and then evaluate with all GPU cards.
```bash
CUDA_VISIBLE_DEVICES=1,2 python3 main.py --model_name facebook/opt-125m --amp --bits 4 --group_size -1 --deployment_device fake,cpu --output_dir /save_model_path/ 
```

- Enable 'disable_eval' for the quantization example, save the qdq model by setting 'deployment_device=fake', and then set up multi-GPU cards for the evaluation script.
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --model_name facebook/opt-125m --amp --bits 4 --group_size -1 --disable_eval --deployment_device fake --output_dir /save_model_path/ 

CUDA_VISIBLE_DEVICES=1,2 python3 eval/evaluation.py --model_name /save_model_path/ --eval_bs 8 --tasks mmlu,lambada_openai,ptb --excel_path /result_excel/save_path/
```

You can also utilize the official lm_eval [link](https://github.com/EleutherAI/lm-evaluation-harness/tree/main?tab=readme-ov-file#basic-usage).

## 4. Known Issues
* Random issues in tuning Qwen models
* ChatGlm-V1 is not supported


## 5. Environment

PyTorch 1.8 or higher version is needed
The transformers version required varies across different types of models. Here, the transformers version used for running models during experiments is provided as a reference.

| Model | Transformers version |
|  :----: | :----: |
| EleutherAI/gpt-j-6b | 4.28/4.30/4.34/4.36 |
| huggyllama/llama-7b | 4.28/4.30/4.34/4.36 |
| meta-llama/Llama-2-7b-hf | 4.30/4.34/4.36 |
| facebook/opt-6.7b | 4.28/4.30/4.34/4.36 |
| tiiuae/falcon-7b | 4.28/4.30/4.34/4.36 |
| mosaicml/mpt-7b | 4.28/4.30/4.34/4.36 |
| mosaicml/mpt-7b-chat | 4.34 |
| bigscience/bloom-7b1 | 4.28/4.30/4.34/4.36 |
| baichuan-inc/Baichuan2-7B-Chat | 4.36 |
| Qwen/Qwen-7B | 4.28/4.30/4.34/4.36 |
| Qwen/Qwen1.5-7B-Chat | 4.38/4.40 |
| THUDM/chatglm3-6b | 4.34/4.36 |
| mistralai/Mistral-7B-v0.1 | 4.34/4.36 |
| MBZUAI/LaMini-GPT-124M | 4.34/4.36 |
| EleutherAI/gpt-neo-125m | 4.34 |
| databricks/dolly-v2-3b | 4.34 |
| stabilityai/stablelm-base-alpha-3b | 4.34 |
| Intel/neural-chat-7b-v3 | 4.34/4.36 |
| rinna/bilingual-gpt-neox-4b | 4.36 |
| microsoft/phi-2 | 4.36 |
| google/gemma-7b | 4.38/4.40 |
| Salesforce/codegen25-7b-multi | 4.33.2|


## Reference
If you find SignRound useful for your research, please cite our paper:
```bash
@article{cheng2023optimize,
  title={Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs},
  author={Cheng, Wenhua and Zhang, Weiwei and Shen, Haihao and Cai, Yiyang and He, Xin and Lv, Kaokao},
  journal={arXiv preprint arXiv:2309.05516},
  year={2023}
}
```







