Step-by-Step
============

This document presents step-by-step instructions for auto-round.

# Prerequisite

## 1. Environment

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
| bigscience/bloom-7b1 | 4.28/4.30/4.34/4.36 |
| baichuan-inc/Baichuan-7B | 4.28/4.30 |
| Qwen/Qwen-7B | 4.28/4.30/4.34/4.36 |
| THUDM/chatglm3-6b | 4.34/4.36 |
| mistralai/Mistral-7B-v0.1 | 4.34/4.36 |
| MBZUAI/LaMini-GPT-124M | 4.34/4.36 |
| EleutherAI/gpt-neo-125m | 4.34 |
| databricks/dolly-v2-3b | 4.34 |
| stabilityai/stablelm-base-alpha-3b | 4.34 |
| Intel/neural-chat-7b-v3 | 4.34/4.36 |


## 2. Prepare Dataset

The NeelNanda/pile-10k in huggingface is adopted as the default calibration data and  will be downloaded automatically from the datasets Hub. To customize a dataset, please kindly follow our dataset code.
See more about loading [huggingface dataset](https://huggingface.co/docs/datasets/loading_datasets.html)

<br />

## 3. Run Examples
Enter into the examples folder and install lm-eval to run the evaluation
```bash
pip install -r requirements.txt
```

- **Default Settings:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --model_name facebook/opt-125m --amp --bits 4 --group_size -1 --enable_minmax_tuning --use_quant_input
```
- **Reduced GPU Memory Usage and Adjusted Training Batch Size:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --model_name facebook/opt-125m --amp --bits 4 --group_size -1 --low_gpu_mem_usage --train_bs 1 --gradient_accumulate_steps 8
```
- **Utilizing the AdamW Optimizer:**
Include the flag `--adam`. Note that AdamW is less effective than Sign gradient descent in many scenarios we tested.

- **Running the Original SignRound:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --model_name facebook/opt-125m --amp --bits 4 --group_size -1 --iters 400 --lr 0.0025 --minmax_lr 0.0025
```

- **Running on Intel Gaudi2**
```bash
bash run_autoround_on_gaudi.sh 
```
 `--enable_minmax_tuning` is strongly recommended 


## 4. Known Issues
* Random issues in tuning Qwen models
* ChatGlm-V1 is not supported


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



