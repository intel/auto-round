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
of models. Our method adopts sign gradient descent to fine-tune rounding values and minmax values of weights in just 200
steps,
which competes impressively against recent methods without introducing any additional inference overhead and keeping low tuning cost. The below
image presents an overview of AutoRound. Check out our paper on [arxiv](https://arxiv.org/pdf/2309.05516v4) for more details and visit [low_bit_open_llm_leaderboard](https://huggingface.co/spaces/Intel/low_bit_open_llm_leaderboard) for more accuracy data across various models.

<div align="center">

![](docs/imgs/autoround_overview.png)

<div align="left">

## What's New

* [2024/08] AutoRound format supports Intel Gaudi2 devices. For an example, please refer to [Intel/Qwen2-7B-int4-inc](https://huggingface.co/Intel/Qwen2-7B-int4-inc).
* [2024/08] AutoRound includes several experimental features, e.g., activation quantization, mx_fp data type, and fast tuning of norm/bias parameters.
* [2024/07] Important change: the default value of nsamples has been changed from 512 to 128 to reduce the  memory usages, which may cause a slight accuracy drop in some scenarios
* [2024/06] AutoRound format supports mixed bit-widths and group sizes for inference, resolving the significant performance drop issue with the asymmetric kernel
* [2024/05] AutoRound supports lm-head quantization, saving 0.7G for LLaMA3-8B at W4G128.


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

0.3.0+
### Gaudi2/ CPU/ GPU

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

from auto_round import AutoRound

bits, group_size, sym = 4, 128, False
autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym)
autoround.quantize()
output_dir = "./tmp_autoround"
##format= 'auto_round', 'auto_gptq', 'auto_awq'
autoround.save_quantized(output_dir, format='auto_round') 
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

#### Formats
**AutoRound format**： This format is  well-suited for CPU and HPU devices and mixed precision inference. It addresses the asymmetric quantization kernel issues present in the AutoGPTQ format. Besides, lm-head quantization and mixed precision is supported. However, it is not yet widely adopted by the community. For CUDA support, installing from the source is required.

**AutoGPTQ Format**: This format is well-suited for symmetric quantization on CUDA devices and is widely adopted by the community. It also benefits from the Marlin kernel, which can boost inference performance notably. However, the asymmetric kernel has issues that can cause considerable accuracy drops, particularly at 2-bit quantization. Additionally, symmetric quantization tends to perform poorly at 2-bit precision.

**AutoAWQ format**: This format is well-suited for asymmetric 4-bit quantization on CUDA devices and is widely adopted within the community. It features specialized layer fusion tailored for Llama models. However, it supports only 4-bit asymmetric quantization and is not compatible with some models, such as Phi. Additionally, we have not extensively tested exporting to this format, so there may be potential bugs or issues with the export process.


#### Tips

1 Consider increasing 'iters' (e.g. 1000) to achieve better results, albeit with increased tuning time.

2 Consider increasing 'nsamples' (e.g. 512) to achieve better results, albeit with more memory(~20G).

3 Setting 'minmax_lr' to 2.0/iters has been observed to occasionally yield improved results.

## Model inference

Please run the quantization code first

### AutoGPTQ/AutoAWQ format
Refer to their repositories to infer the model.


### AutoRound format

**cuda**: git clone https://github.com/intel/auto-round.git && cd auto-round && pip install -vvv --no-build-isolation
-e .

**cpu**:

* option 1: pip install auto-round && pip install intel-extension-for-transformers
* option 2: git clone https://github.com/intel/auto-round.git && cd auto-round && pip install -vvv --no-build-isolation
  -e .

**hpu**: docker image with Gaudi Software Stack is recommended. More details can be found
in [Gaudi Guide](https://docs.habana.ai/en/latest/).

#### Gaudi2/ CPU/ GPU on 0.3.0+

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRoundConfig

device = "auto"  ##cpu, hpu, cuda
quantization_config = AutoRoundConfig(
    backend=device
)
quantized_model_path = "./tmp_autoround"
model = AutoModelForCausalLM.from_pretrained(quantized_model_path,
                                             device_map=device, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
```

#### Gaudi2/ CPU/ GPU on 0.3.0

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round.auto_quantizer import AutoHfQuantizer ## must import

quantized_model_path = "./tmp_autoround"
model = AutoModelForCausalLM.from_pretrained(quantized_model_path,
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
```




## Support List

AutoRound supports basically all the major large language models.

Two main model export formats are provided: 'autoround' and 'autogptq'. The AutoRound format supports a wider range of devices, while the autogptq format is highly compatible and enjoys strong support within the community but may have accuracy issue for asym configuration. 

Please note that an asterisk (*) indicates third-party quantized models, which may lack accuracy data and use a different recipe. We greatly appreciate their efforts and encourage more users to share their models, as we cannot release most of the models ourselves.

Model                                | Supported                                                           |
|--------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| meta-llama/Meta-Llama-3.1-70B-Instruct       | [recipe](https://huggingface.co/Intel/Meta-Llama-3.1-70B-Instruct-int4-inc)                       |
| meta-llama/Meta-Llama-3.1-8B-Instruct        | [model-kaitchup-autogptq-int4*](https://huggingface.co/kaitchup/Meta-Llama-3.1-8B-Instruct-autoround-gptq-4bit-asym), [model-kaitchup-autogptq-sym-int4*](https://huggingface.co/kaitchup/Meta-Llama-3.1-8B-Instruct-autoround-gptq-4bit-sym), [recipe](https://huggingface.co/Intel/Meta-Llama-3.1-8B-Instruct-int4-inc)           |
| meta-llama/Meta-Llama-3.1-8B                 | [model-kaitchup-autogptq-sym-int4*](https://huggingface.co/kaitchup/Meta-Llama-3.1-8B-autoround-gptq-4bit-sym)     |
| Qwen/Qwen-VL                          |  [accuracy](./examples/multimodal-modeling/Qwen-VL/README.md), [recipe](./examples/multimodal-modeling/Qwen-VL/run_autoround.sh)
| Qwen/Qwen2-7B                                | [model-autoround-int4](https://huggingface.co/Intel/Qwen2-7B-int4-inc)        |
| Qwen/Qwen2-57B-A14B-Instruct                 | [model-autoround-int4](https://huggingface.co/Intel/Qwen2-57B-A14B-Instruct-int4-inc)     |
| 01-ai/Yi-1.5-9B                      | [model-LnL-AI-autogptq-int4*](https://huggingface.co/LnL-AI/Yi-1.5-9B-4bit-gptq-autoround)         |
| 01-ai/Yi-1.5-9B-Chat                 | [model-LnL-AI-autogptq-int4*](https://huggingface.co/LnL-AI/Yi-1.5-9B-Chat-4bit-gptq-autoround)   |
| Intel/neural-chat-7b-v3-3            | [model-autogptq-int4](https://huggingface.co/Intel/neural-chat-7b-v3-3-int4-inc)          |
| Intel/neural-chat-7b-v3-1            | [model-autogptq-int4](https://huggingface.co/Intel/neural-chat-7b-v3-1-int4-inc)          |
| TinyLlama-1.1B-intermediate   | [model-LnL-AI-autogptq-int4*](https://huggingface.co/LnL-AI/TinyLlama-1.1B-intermediate-step-1341k-3T-autoround-lm_head-symFalse)     |
| mistralai/Mistral-7B-v0.1            | [model-autogptq-lmhead-int4](https://huggingface.co/Intel/Mistral-7B-v0.1-int4-inc-lmhead), [model-autogptq-int4](https://huggingface.co/Intel/Mistral-7B-v0.1-int4-inc)                            |
| google/gemma-2b                      | [model-autogptq-int4](https://huggingface.co/Intel/gemma-2b-int4-inc)                     |
| tiiuae/falcon-7b                     | [model-autogptq-int4-G64](https://huggingface.co/Intel/falcon-7b-int4-inc)                    |
| sapienzanlp/modello-italia-9b   | [model-fbaldassarri-autogptq-int4*](https://huggingface.co/fbaldassarri/modello-italia-9b-autoround-w4g128-cpu)   |
| microsoft/phi-2                      | [model-autogptq-sym-int4](https://huggingface.co/Intel/phi-2-int4-inc)                        |
| microsoft/Phi-3.5-mini-instruct              | [model-kaitchup-autogptq-sym-int4*](https://huggingface.co/kaitchup/Phi-3.5-Mini-instruct-AutoRound-4bit) |
| microsoft/Phi-3-vision-128k-instruct  |  [recipe](./examples/multimodal-modeling/Phi-3-vision/run_autoround.sh)
| mistralai/Mistral-7B-Instruct-v0.2   | [accuracy](./docs/Mistral-7B-Instruct-v0.2-acc.md), [recipe](./examples/language-modeling/scripts/Mistral-7B-Instruct-v0.2.sh),  [example](./examples/language-modeling/)                    |
| mistralai/Mixtral-8x7B-Instruct-v0.1 | [accuracy](./docs/Mixtral-8x7B-Instruct-v0.1-acc.md), [recipe](./examples/language-modeling/scripts/Mixtral-8x7B-Instruct-v0.1.sh),  [example](./examples/language-modeling/)                                     |
| mistralai/Mixtral-8x7B-v0.1          | [accuracy](./docs/Mixtral-8x7B-v0.1-acc.md), [recipe](./examples/language-modeling/scripts/Mixtral-8x7B-v0.1.sh), [example](./examples/language-modeling/)        |
| meta-llama/Meta-Llama-3-8B-Instruct  | [accuracy](./docs/Meta-Llama-3-8B-Instruct-acc.md), [recipe](./examples/language-modeling/scripts/Meta-Llama-3-8B-Instruct.sh), [example](./examples/language-modeling/)         |
| google/gemma-7b                      | [accuracy](./docs/gemma-7b-acc.md), [recipe](./examples/language-modeling/scripts/gemma-7b.sh),  [example](./examples/language-modeling/)                    |
| meta-llama/Llama-2-7b-chat-hf        | [accuracy](./docs/Llama-2-7b-chat-hf-acc.md), [recipe](./examples/language-modeling/scripts/Llama-2-7b-chat-hf.sh), [example](./examples/language-modeling/)           |
| Qwen/Qwen1.5-7B-Chat                 | [accuracy](./docs/Qwen1.5-7B-Chat-acc.md), [sym recipe](./examples/language-modeling/scripts/Qwen1.5-7B-Chat-sym.sh), [asym recipe ](./examples/language-modeling/scripts/Qwen1.5-7B-Chat-asym.sh), [example](./examples/language-modeling/)                    |
| baichuan-inc/Baichuan2-7B-Chat       | [accuracy](./docs/baichuan2-7b-chat-acc.md), [recipe](./examples/language-modeling/scripts/baichuan2-7b-chat.sh), [example](./examples/language-modeling/)                    |         
| 01-ai/Yi-6B-Chat                     | [accuracy](./docs/Yi-6B-Chat-acc.md), [recipe](./examples/language-modeling/scripts/Yi-6B-Chat.sh), [example](./examples/language-modeling/)                    |                                     
| facebook/opt-2.7b                    | [accuracy](./docs/opt-2.7b-acc.md), [recipe](./examples/language-modeling/scripts/opt-2.7b.sh), [example](./examples/language-modeling/)                    |
| bigscience/bloom-3b                  | [accuracy](./docs/bloom-3B-acc.md), [recipe](./examples/language-modeling/scripts/bloom-3b.sh), [example](./examples/language-modeling/)                    |
| EleutherAI/gpt-j-6b                  | [accuracy](./docs/gpt-j-6B-acc.md), [recipe](./examples/language-modeling/scripts/gpt-j-6b.sh), [example](./examples/language-modeling/)                    | 
                                                                                                                    

## Reference

If you find AutoRound useful for your research, please cite our paper:

```bash
@article{cheng2023optimize,
  title={Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs},
  author={Cheng, Wenhua and Zhang, Weiwei and Shen, Haihao and Cai, Yiyang and He, Xin and Lv, Kaokao and Liu, Yi},
  journal={arXiv preprint arXiv:2309.05516},
  year={2023}
}
```

