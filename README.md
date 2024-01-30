<div align="center">

AutoRound
===========================
<h3> Advanced Weight-Only Quantization Algorithm for LLMs</h3>

[![python](https://img.shields.io/badge/python-3.8%2B-blue)](https://github.com/intel/auto-round)
[![version](https://img.shields.io/badge/release-0.1-green)](https://github.com/intel/auto-round)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/intel/auto-round/blob/main/LICENSE)
---
<div align="left">

AutoRound is an advanced weight-only quantization algorithm, based on SignRound. It's tailored for a wide range of models and consistently delivers noticeable improvements, often significantly outperforming SignRound with the cost of more tuning time for quantization.

## Prerequisites
- Python 3.9 or higher

## Installation
### Build from Source
```bash
pip install -r requirements.txt
python setup.py install
```
## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=True, torch_dtype="auto", trust_remote_code=True
        )
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
bits, group_size, scheme = 4, 128, "asym"
autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, scheme=scheme)
autoround.quantize()

# Intel CPU Inference, Currently, llama, bloom, and mistral are supported.
output_dir = "/path/to/quantized_model"
autoround.export(output_dir)
# then follow ITREX to load the model and inference

```

<details>
  <summary>Detailed Hyperparameters</summary>

- `model`: The PyTorch model to be quantized.
            
- `tokenizer`: An optional tokenizer for processing input data. If none is provided, a dataloader must be supplied.
  
- `bits (int)`: Number of bits for quantization (default is 4).
  
- `group_size (int)`: Size of the quantization group (default is 128).

- `scheme (str)`: The quantization scheme (symmetric/asymmetric) to be used (default is "asym").
  
- `use_quant_input (bool)`: Whether to use the output of the previous quantized block as the input for the current block (default is True).
  
- `enable_minmax_tuning (bool)`: Whether to enable weight min-max tuning (default is True).
  
- `iters (int)`: Number of tuning iterations (default is 200).
  
- `lr (float)`: The learning rate for rounding value (default is None, it will be set to 1.0/iters automatically).
  
- `minmax_lr (float)`: The learning rate for min-max tuning (default is None, it will be set to lr automatically).
  
- `n_samples (int)`: Number of samples for tuning (default is 512).
  
- `seqlen (int)`: Data length of the sequence for tuning (default is 2048).
  
- `bs (int)`: Batch size for training (default is 8).
  
- `amp (bool)`: Whether to use automatic mixed precision (default is True).
  
- `n_blocks (int)`: Packing several blocks as one for tuning together (default is 1).
  
- `gradient_accumulate_steps (int)`: Number of gradient accumulation steps (default is 1).
  
- `low_gpu_mem_usage (bool)`: Whether to save GPU memory at the cost of a little tuning time (default is True).
  
- `dataset_name (str)`: The default dataset name for tuning (default is "NeelNanda/pile-10k").
  
- `dataset_split (str)`: The split of the dataset to be used for tuning (default is "train").
  
- `dataloader`: The dataloader for tuning data.
  
- `weight_config (dict)`: Configuration for weight quantization (default is an empty dictionary), mainly for mixed bits or mixed precision.
  
- `device`: The device to be used for tuning (default is "cuda:0").

</details>


We provide a comparative analysis with other methods [link](docs/README.md) in our accuracy data section. Notably, our approach has outperformed GPTQ with a score of 30/32 and AWQ with a score of 27/32 across llamv1/llamav2/mistral-7b on W4G-1, W4G128, W3G128, W2G128.  And the tuning costs are comparable.

## Tips
1 Consider increasing tuning steps to achieve better results, albeit with increased tuning time. 

2 Leverage AutoGPTQ to evaluate the model on GPU

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=True, torch_dtype="auto", trust_remote_code=True
        )
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

autoround = AutoRound(model, tokenizer, bits=4, group_size=128, scheme="asym")
autoround.quantize()

## export to autogptq
# please install auto-gptq https://github.com/AutoGPTQ/
output_dir = "/path/to/quantized_model"
autoround.export(output_dir, target="auto_gptq", use_triton=True)
# then follow auto-gptq to load the model and inference
```

## Known Issues
* Random issues in tuning Qwen models
* ChatGlm-V1 is not supported
  
## Examples
The following various models are currently supported, and quantitative validation results for some mainstream models have been provided. Please refer to the [example readme](examples/README.md) for details.

Validated models and the corresponding versions of transformers for reference:
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



