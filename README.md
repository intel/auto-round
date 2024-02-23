<div align="center">

AutoRound
===========================
<h3> Advanced Weight-Only Quantization Algorithm for LLMs</h3>

[![python](https://img.shields.io/badge/python-3.8%2B-blue)](https://github.com/intel/auto-round)
[![version](https://img.shields.io/badge/release-0.1-green)](https://github.com/intel/auto-round)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/intel/auto-round/blob/main/LICENSE)
---
<div align="left">

AutoRound is an advanced weight-only quantization algorithm for low-bits LLM inference. It's tailored for a wide range of models and consistently delivers noticeable improvements, often significantly outperforming SignRound with the cost of more tuning time for quantization.

## Prerequisites
- Python 3.9 or higher

## Installation
### Build from Source
```bash
pip install -r requirements.txt
python setup.py install
```
## Usage of Tuning

### On CPU/GPU
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
bits, group_size, scheme = 4, 128, "asym"
tuning_device = "cuda:0" ## or "cpu"
autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, scheme=scheme, device=tuning_device)
autoround.quantize()

output_dir = "./tmp_autoround"
deployment_device = "cpu" ## or gpu
if deployment_device=="cpu":
    autoround.save_quantized(output_dir, format="itrex") ## export to itrex format
else:
    autoround.save_quantized(output_dir, format="auto_gptq", use_triton=True) ##export to autogptq format
```


### On Intel Gaudi2

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
bits, group_size, scheme = 4, 128, "asym"

# need to load model first, then import
from auto_round import AutoRound
autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, scheme=scheme,
                      device="hpu", amp=False)
autoround.quantize()
```


<details>
  <summary>Detailed Hyperparameters</summary>

- `model`: The PyTorch model to be quantized.
            
- `tokenizer`: An optional tokenizer for processing input data. If none is provided, a dataloader must be supplied.
  
- `bits (int)`: Number of bits for quantization (default is 4).
  
- `group_size (int)`: Size of the quantization group (default is 128).

- `scheme (str)`: The quantization scheme (sym/asym) to be used (default is "asym").
  
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

## Model inference
Please run the tuning code first



### Intel CPU
```python
# save_quantized to itrex format first
# Please read ITREX(https://github.com/intel/intel-extension-for-transformers/tree/main/intel_extension_for_transformers/llm/runtime/neural_speed) to understand the details
# currently please install neural-speed (https://github.com/intel/neural-speed) from source
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig
from transformers import AutoTokenizer
quantized_model_path = "./tmp_autoround"
woq_config = WeightOnlyQuantConfig(group_size=group_size, scheme=scheme, 
                                   use_autoround=True)  ##only supports 4 bits currently
prompt = "There is a girl who likes adventure,"
tokenizer = AutoTokenizer.from_pretrained(quantized_model_path, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
model = AutoModelForCausalLM.from_pretrained(quantized_model_path, quantization_config=woq_config, 
                                             trust_remote_code=True, device="cpu")
outputs = model.generate(inputs, max_new_tokens=50)
```
### GPU
```python
# save_quantized to autogptq format first and then follow transformers or auto-gptq to load the model and inference
from transformers import AutoModelForCausalLM, AutoTokenizer
quantized_model_path = "./tmp_autoround"
model = AutoModelForCausalLM.from_pretrained(quantized_model_path,
                                             device_map="auto",
                                             trust_remote_code=True
                                             )
tokenizer = AutoTokenizer.from_pretrained(quantized_model_path, use_fast=True)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
```
## Huggingface Model cards
We fine-tuned the hyperparameters for each model with an iteration of 1K and successfully achieved near-lossless quantized models in the majority of scenarios. Some of these models has been uploaded to the Huggingface Hub.

### AutoGPTQ format
[Intel/neural-chat-7b-v3-3-int4-inc](https://huggingface.co/Intel/neural-chat-7b-v3-3-int4-inc)

[Intel/neural-chat-7b-v3-1-int4-inc](https://huggingface.co/Intel/neural-chat-7b-v3-1-int4-inc)

[Intel/Mistral-7B-v0.1-int4-inc](https://huggingface.co/Intel/Mistral-7B-v0.1-int4-inc)

[Intel/Mixtral-8x7B-Instruct-v0.1-int4-inc](https://huggingface.co/Intel/Mixtral-8x7B-Instruct-v0.1-int4-inc) coming soon

[Intel/Mixtral-8x7B-v0.1-int4-inc](https://huggingface.co/Intel/Mixtral-8x7B-v0.1-int4-inc) coming soon

[Intel/phi-2-int4-inc](https://huggingface.co/Intel/phi-2-int4-inc) coming soon

### Itrex format

Please stay tuned

## Comparison with other methods

We provide a [comprehensive analysis](docs/README.md) with other methods in our accuracy data section. Notably, our approach has outperformed GPTQ with a score of 30/32 and AWQ with a score of 27/32 across llamv1/llamav2/mistral-7b on W4G-1, W4G128, W3G128, W2G128.  And the tuning costs are comparable.

## Tips
1 Consider increasing tuning steps to achieve better results, albeit with increased tuning time. 

2 Setting 'use_quant_input' to False has been observed to occasionally yield improved results.

3 Setting 'minmax_lr' to 2.0/iters has been observed to occasionally yield improved results.

  
## Examples
Quantization has been enabled for various large language models. Please refer to the [example readme](examples/README.md) for details.


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



