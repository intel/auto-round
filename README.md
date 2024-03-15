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
### Install from pypi
```bash
pip install auto-round
```
## Usage of Tuning

### On CPU/ Gaudi2/ GPU

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tuning_device = "cuda:0"  ## or "cpu", "hpu"
dtype = "auto" if tuning_device != "hpu" else torch.bfloat16
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

from auto_round import AutoRound

bits, group_size, sym = 4, 128, False
autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym, device=tuning_device)
autoround.quantize()
output_dir = "./tmp_autoround"
autoround.save_quantized(output_dir)
```



## Model inference
Please run the tuning code first



### Intel CPU
```python
# Please save the quantized model in 'itrex' format first, then refer to the ITREX tutorial for more details on inference with the INT4 model.
# (https://github.com/intel/intel-extension-for-transformers/tree/main/intel_extension_for_transformers/llm/runtime/neural_speed)
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig
from transformers import AutoTokenizer

quantized_model_path = "./tmp_autoround"
scheme = "sym" if sym else "asym"
woq_config = WeightOnlyQuantConfig(
    group_size=group_size, scheme=scheme, use_autoround=True
)  ##only supports 4 bits currently
prompt = "There is a girl who likes adventure,"
tokenizer = AutoTokenizer.from_pretrained(quantized_model_path, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
model = AutoModelForCausalLM.from_pretrained(
    quantized_model_path, quantization_config=woq_config, trust_remote_code=True, device="cpu"
)
outputs = model.generate(inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```


### GPU
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

quantized_model_path = "./tmp_autoround"
model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(quantized_model_path, use_fast=True)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
```

<details>
  <summary>Detailed Hyperparameters</summary>

- `model`: The PyTorch model to be quantized.
            
- `tokenizer`: An optional tokenizer for processing input data. If none is provided, a dataloader must be supplied.
  
- `bits (int)`: Number of bits for quantization (default is 4).
  
- `group_size (int)`: Size of the quantization group (default is 128).

- `sym (bool)`: Whether to use symmetric quantization.
  
- `use_quant_input (bool)`: Whether to use the output of the previous quantized block as the input for the current block (default is True).
  
- `enable_minmax_tuning (bool)`: Whether to enable weight min-max tuning (default is True).
  
- `iters (int)`: Number of tuning iterations (default is 200).
  
- `lr (float)`: The learning rate for rounding value (default is None, it will be set to 1.0/iters automatically).
  
- `minmax_lr (float)`: The learning rate for min-max tuning (default is None, it will be set to lr automatically).
  
- `n_samples (int)`: Number of samples for tuning (default is 512).
  
- `seqlen (int)`: Data length of the sequence for tuning (default is 2048).
  
- `batch_size (int)`: Batch size for training (default is 8).

- `scale_dtype (str)`: The data type of quantization scale to be used (default is "float32"), different kernels have different choices.
  
- `amp (bool)`: Whether to use automatic mixed precision (default is True).
  
- `n_blocks (int)`: Packing several blocks as one for tuning together (default is 1).
  
- `gradient_accumulate_steps (int)`: Number of gradient accumulation steps (default is 1).
  
- `low_gpu_mem_usage (bool)`: Whether to save GPU memory at the cost of a little tuning time (default is True).
  
- `dataset (str)`: The default dataset name for tuning (default is "NeelNanda/pile-10k").
  
- `dataset_split (str)`: The split of the dataset to be used for tuning (default is "train").
  
- `dataloader`: The dataloader for tuning data.
  
- `weight_config (dict)`: Configuration for weight quantization (default is an empty dictionary), mainly for mixed bits or mixed precision.
  
- `device`: The device to be used for tuning. The default is set to 'auto', allowing for automatic detection.

</details>


## Support List

| Model                    | Supported                                                                                                                                                                                                                                                          |
|--------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Intel/neural-chat-7b-v3-3 | [HF-int4-model](https://huggingface.co/Intel/neural-chat-7b-v3-3-int4-inc), [accuracy](./docs/neural-chat-7b-v3-3-acc.md), [recipe](./examples/language-modeling/scripts/neural-chat-7b-v3-3.sh), [example](./examples/language-modeling/)                         |
| Intel/neural-chat-7b-v3-1 | [HF-int4-model](https://huggingface.co/Intel/neural-chat-7b-v3-1-int4-inc), [accuracy](./docs/neural-chat-7b-v3-1-acc.md), [recipe](./examples/language-modeling/scripts/neural-chat-7b-v3-1.sh), [example](./examples/language-modeling/)                         |
| mistralai/Mistral-7B-v0.1 | [HF-int4-model](https://huggingface.co/Intel/Mistral-7B-v0.1-int4-inc), [accuracy](./docs/Mistral-7B-v0.1-acc.md), [recipe](./examples/language-modeling/scripts/Mistral-7B-v0.1.sh), [example](./examples/language-modeling/)                                     |
| google/gemma-7b          | [HF-int4-model](https://huggingface.co/Intel/gemma-7b-int4-inc) under review, [accuracy](./docs/gemma-7b-acc.md), [recipe](./examples/language-modeling/scripts/gemma-7b.sh),  [example](./examples/language-modeling/)                                            |
| google/gemma-7b-it       | [HF-int4-model](https://huggingface.co/Intel/gemma-7b-it-int4-inc) under review, [accuracy](./docs/gemma-7b-it-acc.md), [recipe](./examples/language-modeling/scripts/gemma-7b-it.sh), [example](./examples/language-modeling/)                                    |                                            |
  mistralai/Mixtral-8x7B-Instruct-v0.1 | [HF-int4-model](https://huggingface.co/Intel/Mistral-7B-v0.1-int4-inc) under review, [accuracy](./docs/Mixtral-8x7B-Instruct-v0.1-acc.md), [recipe](./examples/language-modeling/scripts/Mixtral-8x7B-Instruct-v0.1.sh),  [example](./examples/language-modeling/) |
| mistralai/Mixtral-8x7B-v0.1 | [HF-int4-model](https://huggingface.co/Intel/Mixtral-8x7B-v0.1-int4-inc) under review, [accuracy](./docs/Mixtral-8x7B-v0.1-acc.md), [recipe](./examples/language-modeling/scripts/Mixtral-8x7B-v0.1.sh), [example](./examples/language-modeling/)                  |
| microsoft/phi-2          | [HF-int4-model](https://huggingface.co/Intel/phi-2-int4-inc) under review, [accuracy](./docs/phi-2-acc.md), [recipe](./examples/language-modeling/scripts/phi-2.sh), [example](./examples/language-modeling/)                                                      |
| meta-llama/Llama-2-7b-chat-hf | [accuracy](./docs/Llama-2-7b-chat-hf-acc.md), [recipe](./examples/language-modeling/scripts/Llama-2-7b-chat-hf.sh), [example](./examples/language-modeling/)                                                                                                                    |
| Salesforce/codegen25-7b-multi | [example](./examples/code-generation)                                                                                                                                                                                                                              |
| EleutherAI/gpt-j-6b | [example](./examples/language-modeling/)                                                                                                                                                                                                                           |
| huggyllama/llama-7b | [example](./examples/language-modeling/)                                                                                                                                                                                                                           |
| meta-llama/Llama-2-7b-hf | [example](./examples/language-modeling/)                                                                                                                                                                                                                           |
| facebook/opt-6.7b | [example](./examples/language-modeling/)                                                                                                                                                                                                                           |
| tiiuae/falcon-7b | [example](./examples/language-modeling/)                                                                                                                                                                                                                           |
| mosaicml/mpt-7b | [example](./examples/language-modeling/)                                                                                                                                                                                                                           |
| bigscience/bloom-7b1 | [example](./examples/language-modeling/)                                                                                                                                                                                                                           |
| baichuan-inc/Baichuan-7B | [example](./examples/language-modeling/)                                                                                                                                                                                                                           |
| Qwen/Qwen-7B | [example](./examples/language-modeling/)                                                                                                                                                                                                                           |
| THUDM/chatglm3-6b | [example](./examples/language-modeling/)                                                                                                                                                                                                                           |
| MBZUAI/LaMini-GPT-124M | [example](./examples/language-modeling/)                                                                                                                                                                                                                           |
| EleutherAI/gpt-neo-125m | [example](./examples/language-modeling/)                                                                                                                                                                                                                           |
| databricks/dolly-v2-3b | [example](./examples/language-modeling/)                                                                                                                                                                                                                           |
| stabilityai/stablelm-base-alpha-3b | [example](./examples/language-modeling/)




## Comparison with other methods

We provide a [comprehensive analysis](docs/acc.md) with other methods in our accuracy data section. Notably, our approach has outperformed GPTQ with a score of 30/32 and AWQ with a score of 27/32 across llamv1/llamav2/mistral-7b on W4G-1, W4G128, W3G128, W2G128.  And the tuning costs are comparable.

## Tips
1 Consider increasing tuning steps to achieve better results, albeit with increased tuning time. 

2 Setting 'use_quant_input' to False has been observed to occasionally yield improved results.

3 Setting 'minmax_lr' to 2.0/iters has been observed to occasionally yield improved results.

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
