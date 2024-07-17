<div align="center">

AutoRound
===========================
<h3> Advanced Quantization Algorithm for LLMs</h3>

[![python](https://img.shields.io/badge/python-3.8%2B-blue)](https://github.com/intel/auto-round)
[![version](https://img.shields.io/badge/release-0.2-green)](https://github.com/intel/auto-round)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/intel/auto-round/blob/main/LICENSE)
---
<div align="left">

AutoRound is an advanced quantization algorithm for low-bits LLM inference. It's tailored for a wide range
of models. Our method adopts sign gradient descent to fine-tune rounding values and minmax values of weights in just 200 steps,
which competes impressively against recent methods without introducing any additional inference overhead. The below
image presents an overview of AutoRound.  Check out our updated paper on [arxiv](https://arxiv.org/pdf/2309.05516v4)

<div align="center">

![](docs/imgs/autoround_overview.png)

<div align="left">

## What's New
* [2024/07] Important change: the default value of nsamples has been changed from 512 to 128 to reduce the  memory usages, which may cause a slight accuracy drop in some scenarios
* [2024/06] AutoRound format supports mixed bit-widths and group sizes for inference, resolving the significant performance drop issue with the asymmetric kernel
* [2024/05] AutoRound supports lm-head quantization, saving 0.7G for LLaMA3-8B at W4G128.
* [2024/05] AutoRound performs well
  in [low_bit_open_llm_leaderboard](https://huggingface.co/spaces/Intel/low_bit_open_llm_leaderboard)

## Prerequisites

- Python 3.9 or higher

## Installation

### Build from Source

```bash
pip install -vvv --no-build-isolation -e .
or
pip install -r requirements.txt
python setup.py install


```

### Install from pypi

```bash
pip install auto-round
```

## Model quantization

### Gaudi2/ CPU/ GPU

We found a significant accuracy discrepancy with the qdq model using the AutoGPTQ GPU backend with asymmetric
quantization in some scenarios, especially at lower bits,like 2. Please save quantized model to AuoRound format to fix this issue.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

from auto_round import AutoRound

bits, group_size, sym = 4, 128, False
autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym)
autoround.quantize()
output_dir = "./tmp_autoround"
autoround.save_quantized(output_dir) ##save_quantized(output_dir,format="auto_round")
```

<details>
  <summary>Detailed Hyperparameters</summary>

- `model`: The PyTorch model to be quantized.

- `tokenizer`: An optional tokenizer for processing input data. If none, a dataset must be provided.

- `bits (int)`: Number of bits for quantization (default is 4).

- `group_size (int)`: Size of the quantization group (default is 128).

- `sym (bool)`: Whether to use symmetric quantization (default is False).

- `enable_quanted_input (bool)`: Whether to use the output of the previous quantized block as the input for the current
  block for tuning (default is True).

- `enable_minmax_tuning (bool)`: Whether to enable weight min-max tuning (default is True).

- `iters (int)`: Number of tuning iterations (default is 200).

- `lr (float)`: The learning rate for rounding value (default is None, it will be set to 1.0/iters automatically).

- `minmax_lr (float)`: The learning rate for min-max tuning (default is None, it will be set to lr automatically).

- `nsamples (int)`: Number of samples for tuning (default is 128).

- `seqlen (int)`: Data length of the sequence for tuning (default is 2048).

- `batch_size (int)`: Batch size for training (default is 8).

- `scale_dtype (str)`: The data type of quantization scale to be used (default is "float16"), different kernels have
  different choices.

- `amp (bool)`: Whether to use automatic mixed precision (default is True).

- `nblocks (int)`: Packing several blocks as one for tuning together (default is 1).

- `gradient_accumulate_steps (int)`: Number of gradient accumulation steps (default is 1).

- `low_gpu_mem_usage (bool)`: Whether to save GPU memory at the cost of ~20% more tuning time (default is False).

- `dataset Union[str, list, tuple, torch.utils.data.DataLoader]`: The dataset name for tuning (default is "
  NeelNanda/pile-10k"). Local json file and combination of datasets have been supported, e.g. "
  ./tmp.json,NeelNanda/pile-10k:train, mbpp:train+validation+test"

- `layer_config (dict)`: Configuration for weight quantization (default is an empty dictionary), mainly for mixed bits
  or mixed precision.

- `device`: The device to be used for tuning. The default is set to 'auto', allowing for automatic detection.

</details>

### Tips

1 Consider increasing 'iters' (e.g. 1000) to achieve better results, albeit with increased tuning time.

2 Consider increasing 'nsamples' (e.g. 512) to achieve better results, albeit with more memory(~20G).

3 Setting 'minmax_lr' to 2.0/iters has been observed to occasionally yield improved results.

## Model inference

Please run the quantization code first.

### CPU

```python
##pip install intel-extension-for-transformers
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

quantized_model_path = "./tmp_autoround"
model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
```

### GPU

```python
##pip install auto-gptq
from transformers import AutoModelForCausalLM, AutoTokenizer
##from auto_round.auto_quantizer import AutoHfQuantizer ## uncomment it for models with auto_round format

quantized_model_path = "./tmp_autoround"
model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
```

### Intel Gaudi-2

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round.auto_quantizer import AutoHfQuantizer
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
quantized_model_path = "./tmp_autoround"
model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto").to('hpu').to(torch.float32)
tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
```


## Support List

| Model                                | Supported                                                                                                                                                                                                                                                                                                           |
|--------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Intel/neural-chat-7b-v3-3            | [HF-int4-model](https://huggingface.co/Intel/neural-chat-7b-v3-3-int4-inc), [accuracy](./docs/neural-chat-7b-v3-3-acc.md), [recipe](./examples/language-modeling/scripts/neural-chat-7b-v3-3.sh), [example](./examples/language-modeling/)                                                                          |
| Intel/neural-chat-7b-v3-1            | [HF-int4-model](https://huggingface.co/Intel/neural-chat-7b-v3-1-int4-inc), [accuracy](./docs/neural-chat-7b-v3-1-acc.md), [recipe](./examples/language-modeling/scripts/neural-chat-7b-v3-1.sh), [example](./examples/language-modeling/)                                                                          |
| mistralai/Mistral-7B-v0.1            | [HF-int4-model-lmhead](https://huggingface.co/Intel/Mistral-7B-v0.1-int4-inc-lmhead),[HF-int4-model](https://huggingface.co/Intel/Mistral-7B-v0.1-int4-inc), [accuracy](./docs/Mistral-7B-v0.1-acc.md), [recipe](./examples/language-modeling/scripts/Mistral-7B-v0.1.sh), [example](./examples/language-modeling/) |
| microsoft/phi-2                      | [HF-int4-sym-model](https://huggingface.co/Intel/phi-2-int4-inc), [accuracy](./docs/phi-2-acc.md), [recipe](./examples/language-modeling/scripts/phi-2.sh), [example](./examples/language-modeling/)                                                                                                                
| google/gemma-2b                      | [HF-int4-model](https://huggingface.co/Intel/gemma-2b-int4-inc), [accuracy](./docs/gemma-2b-acc.md), [recipe](./examples/language-modeling/scripts/gemma-2b.sh),  [example](./examples/language-modeling/)                                                                                                          
| tiiuae/falcon-7b                     | [HF-int4-model-G64](https://huggingface.co/Intel/falcon-7b-int4-inc), [accuracy](./docs/falcon-7b-acc.md), [recipe](./examples/language-modeling/scripts/falcon-7b.sh), [example](./examples/language-modeling/)                                                                                                    |
| mistralai/Mistral-7B-Instruct-v0.2   | [HF-int4-model](https://huggingface.co/Intel/Mistral-7B-Instruct-v0.2-int4-inc) (under review), [accuracy](./docs/Mistral-7B-Instruct-v0.2-acc.md), [recipe](./examples/language-modeling/scripts/Mistral-7B-Instruct-v0.2.sh),  [example](./examples/language-modeling/)                                           |
| mistralai/Mixtral-8x7B-Instruct-v0.1 | [HF-int4-model](https://huggingface.co/Intel/Mixtral-8x7B-Instruct-v0.1-int4-inc) (under review), [accuracy](./docs/Mixtral-8x7B-Instruct-v0.1-acc.md), [recipe](./examples/language-modeling/scripts/Mixtral-8x7B-Instruct-v0.1.sh),  [example](./examples/language-modeling/)                                     |
| mistralai/Mixtral-8x7B-v0.1          | [HF-int4-model](https://huggingface.co/Intel/Mixtral-8x7B-v0.1-int4-inc) (under review), [accuracy](./docs/Mixtral-8x7B-v0.1-acc.md), [recipe](./examples/language-modeling/scripts/Mixtral-8x7B-v0.1.sh), [example](./examples/language-modeling/)                                                                 |
| meta-llama/Meta-Llama-3-8B-Instruct  | [accuracy](./docs/Meta-Llama-3-8B-Instruct-acc.md), [recipe](./examples/language-modeling/scripts/Meta-Llama-3-8B-Instruct.sh), [example](./examples/language-modeling/)                                                                                                                                            |
| google/gemma-7b                      | [accuracy](./docs/gemma-7b-acc.md), [recipe](./examples/language-modeling/scripts/gemma-7b.sh),  [example](./examples/language-modeling/)                                                                                           |
| meta-llama/Llama-2-7b-chat-hf        | [accuracy](./docs/Llama-2-7b-chat-hf-acc.md), [recipe](./examples/language-modeling/scripts/Llama-2-7b-chat-hf.sh), [example](./examples/language-modeling/)                                                                                                                                                        |
| Qwen/Qwen1.5-7B-Chat                 | [accuracy](./docs/Qwen1.5-7B-Chat-acc.md), [sym recipe](./examples/language-modeling/scripts/Qwen1.5-7B-Chat-sym.sh), [asym recipe ](./examples/language-modeling/scripts/Qwen1.5-7B-Chat-asym.sh), [example](./examples/language-modeling/)                                                                        |
| baichuan-inc/Baichuan2-7B-Chat       | [accuracy](./docs/baichuan2-7b-chat-acc.md), [recipe](./examples/language-modeling/scripts/baichuan2-7b-chat.sh), [example](./examples/language-modeling/)                                                                                                                                                          |
| 01-ai/Yi-6B-Chat                     | [accuracy](./docs/Yi-6B-Chat-acc.md), [recipe](./examples/language-modeling/scripts/Yi-6B-Chat.sh), [example](./examples/language-modeling/)                                                                                                                                                                        |
| facebook/opt-2.7b                    | [accuracy](./docs/opt-2.7b-acc.md), [recipe](./examples/language-modeling/scripts/opt-2.7b.sh), [example](./examples/language-modeling/)                                                                                                                                                                            |
| bigscience/bloom-3b                  | [accuracy](./docs/bloom-3B-acc.md), [recipe](./examples/language-modeling/scripts/bloom-3b.sh), [example](./examples/language-modeling/)                                                                                                                                                                            |
| EleutherAI/gpt-j-6b                  | [accuracy](./docs/gpt-j-6B-acc.md), [recipe](./examples/language-modeling/scripts/gpt-j-6b.sh), [example](./examples/language-modeling/)                                                                                                                                                                            |
| Salesforce/codegen25-7b-multi        | [example](./examples/language-modeling/)                                                                                                                                                                                                                                                                            |
| huggyllama/llama-7b                  | [example](./examples/language-modeling/)                                                                                                                                                                                                                                                                            |
| mosaicml/mpt-7b                      | [example](./examples/language-modeling/)                                                                                                                                                                                                                                                                            |
| THUDM/chatglm3-6b                    | [example](./examples/language-modeling/)                                                                                                                                                                                                                                                                            |
| MBZUAI/LaMini-GPT-124M               | [example](./examples/language-modeling/)                                                                                                                                                                                                                                                                            |
| EleutherAI/gpt-neo-125m              | [example](./examples/language-modeling/)                                                                                                                                                                                                                                                                            |
| databricks/dolly-v2-3b               | [example](./examples/language-modeling/)                                                                                                                                                                                                                                                                            |
| stabilityai/stablelm-base-alpha-3b   | [example](./examples/language-modeling/)                                                                                                                                                                                                                                                                            

## Comparison with other methods

We provide a [comprehensive analysis](docs/acc.md) with other methods in our accuracy data section. In summary, our
approach achieved superior performance compared to GPTQ, scoring 30/32, AWQ with 27/32, HQQ with 15/16, and OmniQuant
with a perfect score of 16/16 across llamv1/llamav2/mistral-7b on W4G-1, W4G128, W3G128, and W2G128, based on the
average accuracies of 11 zero-shot tasks.



## Reference

If you find SignRound useful for your research, please cite our paper:

```bash
@article{cheng2023optimize,
  title={Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs},
  author={Cheng, Wenhua and Zhang, Weiwei and Shen, Haihao and Cai, Yiyang and He, Xin and Lv, Kaokao and Liu, Yi},
  journal={arXiv preprint arXiv:2309.05516},
  year={2023}
}
```
