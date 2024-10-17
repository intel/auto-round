<div align="center">

AutoRound
===========================
<h3> Advanced Quantization Algorithm for LLMs</h3>

[![python](https://img.shields.io/badge/python-3.8%2B-blue)](https://github.com/intel/auto-round)
[![version](https://img.shields.io/badge/release-0.3.1-green)](https://github.com/intel/auto-round)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/intel/auto-round/blob/main/LICENSE)
---
<div align="left">

AutoRound is an advanced quantization algorithm for low-bits LLM inference. It's tailored for a wide range
of models. AutoRound adopts sign gradient descent to fine-tune rounding values and minmax values of weights in just 200
steps,
which competes impressively against recent methods without introducing any additional inference overhead and keeping low
tuning cost. The below
image presents an overview of AutoRound. Check out our paper on [arxiv](https://arxiv.org/pdf/2309.05516) for more
details and visit [low_bit_open_llm_leaderboard](https://huggingface.co/spaces/Intel/low_bit_open_llm_leaderboard) for
more accuracy data and recipes across various models.

<div align="center">

![](docs/imgs/autoround_overview.png)

<div align="left">

## What's New

* [2024/10] Important update: We now support full-range symmetric quantization and have made it the default
  configuration. This approach is typically better or comparable to asymmetric quantization and significantly
  outperforms other symmetric variants, especially at low bit-widths like 2-bit. And,no need to compile from source to run
  AutoRound format anymore.
* [2024/09] AutoRound format supports several LVM models, check out the
  examples [Qwen2-Vl](./examples/multimodal-modeling/Qwen-VL),[Phi-3-vision](./examples/multimodal-modeling/Phi-3-vision), [Llava](./examples/multimodal-modeling/Llava)
* [2024/08] AutoRound format supports Intel Gaudi2 devices. Please refer
  to [Intel/Qwen2-7B-int4-inc](https://huggingface.co/Intel/Qwen2-7B-int4-inc).
* [2024/08] AutoRound introduces several experimental features, including fast tuning of norm/bias parameters (for 2-bit
  and W4A4), activation quantization, and the mx_fp data type.
* [2024/07] Important change: the default value of nsamples has been changed from 512 to 128 to reduce the memory
  usages, which may cause a slight accuracy drop in some scenarios

## Installation

### Build from Source

```bash
pip install -vvv --no-build-isolation -e .
```

### Install from pypi

```bash
pip install auto-round
```

## Model Quantization

### API Usage (Gaudi2/CPU/GPU)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

from auto_round import AutoRound

bits, group_size = 4, 128
autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size)

## the best accuracy, 3X slower, low_gpu_mem_usage could save ~20G but ~30% slower
# autoround = AutoRound(model, tokenizer, nsamples=512, iters=1000, low_gpu_mem_usage=True, bits=bits, group_size=group_size)

## fast and low memory, 2-3X speedup, slight accuracy drop at W4G128
# autoround = AutoRound(model, tokenizer, nsamples=128, iters=200, seqlen=512, batch_size=4, bits=bits, group_size=group_size)

autoround.quantize()
output_dir = "./tmp_autoround"
## format= 'auto_round'(default in version>0.3.0), 'auto_gptq'(default in version<=0.3.0), 'auto_awq'
autoround.save_quantized(output_dir, format='auto_round', inplace=True) 
```

<details>
  <summary>Detailed Hyperparameters</summary>

- `model`: The PyTorch model to be quantized.

- `tokenizer`: An optional tokenizer for processing input data. If none, a dataset must be provided.

- `bits (int)`: Number of bits for quantization (default is 4).

- `group_size (int)`: Size of the quantization group (default is 128).

- `sym (bool)`: Whether to use symmetric quantization (default is True).

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

- `layer_config (dict)`: Configuration for weight quantization (default is None), mainly for mixed bits
  or mixed precision.

- `device`: The device to be used for tuning. The default is set to 'auto', allowing for automatic detection.

</details>

### Basic Usage (version > 0.3.0)

A user guide detailing the full list of supported arguments is provided by calling ```auto_round -h``` on the terminal.
Alternatively, you can use ```auto-round``` instead of ```auto_round```.

```bash
auto_round --model facebook/opt-125m \
    --bits 4 \
    --group_size 128 \
    --format auto_round \
    --disable_eval \
    --output_dir ./tmp_autoround
```

We provide two recipes for best accuracy and fast running speed with low memory. Details as below.
<details>
  <summary>Other Recipes</summary>

  ```bash
## best accuracy, 3X slower, low_gpu_mem_usage could save ~20G but ~30% slower
  auto_round --model facebook/opt-125m \
    --bits 4 \
    --group_size 128 \
    --nsamples 512 \
    --iters 1000 \
    --low_gpu_mem_usage \
    --disable_eval 
  ```

  ```bash
## fast and low memory, 2-3X speedup, slight accuracy drop at W4G128
  auto_round --model facebook/opt-125m \
    --bits 4 \
    --group_size 128 \
    --nsamples 128 \
    --iters 200 \
    --seqlen 512 \
    --batch_size 4 \
    --disable_eval 
  ```

</details>

#### Formats

**AutoRound Format**：This format is well-suited for CPU, HPU devices, 2 bits, as well as mixed-precision
inference. [2,4]
bits are supported. It also benefits
from the Marlin kernel, which can boost inference performance notably.However, it has not yet gained widespread
community adoption. For CUDA support, you will need to
install from the source.

**AutoGPTQ Format**: This format is well-suited for symmetric quantization on CUDA devices and is widely adopted by the
community, [2,3,4,8] bits are supported, for 3 bits, pip install auto-gptq first before quantization. It also benefits
from the Marlin kernel, which can boost inference performance notably. However, **the
asymmetric kernel has issues** that can cause considerable accuracy drops, particularly at 2-bit quantization and small
models.
Additionally, symmetric quantization tends to perform poorly at 2-bit precision.

**AutoAWQ Format**: This format is well-suited for asymmetric 4-bit quantization on CUDA devices and is widely adopted
within the community, only 4-bits quantization is supported. It features
specialized layer fusion tailored for Llama models.

## Model Inference

Please run the quantization code first

### AutoGPTQ/AutoAWQ format

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

quantized_model_path = "./tmp_autoround"
model = AutoModelForCausalLM.from_pretrained(quantized_model_path,
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
```

### AutoRound format

**CPU**: pip install intel-extension-for-transformers

**HPU**: docker image with Gaudi Software Stack is recommended. More details can be found
in [Gaudi Guide](https://docs.habana.ai/en/latest/).

**CUDA**: pip install auto-gptq for sym quantization(tuning needs auto-round 0.30+), for asym quantization, need to install auto-round from source

#### CPU/HPU/CUDA on 0.3.0+

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

#### CPU/HPU/CUDA on 0.3.0

**CUDA**:  need to install auto-round from source

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round.auto_quantizer import AutoHfQuantizer  ## must import

quantized_model_path = "./tmp_autoround"
model = AutoModelForCausalLM.from_pretrained(quantized_model_path,
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
```

<br>
<details>
  <summary>Evaluation</summary>

```bash
## version > 0.3.0
auto_round --model saved_quantized_model \
    --eval \
    --task lambada_openai \
    --eval_bs 1
```

</details>

## Support List

AutoRound supports basically all the major large language models.

Please note that an asterisk (*) indicates third-party quantized models, which may lack accuracy data and use a
different recipe. We greatly appreciate their efforts and encourage more users to share their models, as we cannot
release most of the models ourselves.

 Model                                  | Supported                                                                                                                                                                                                                                                                                                                 |
|----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| meta-llama/Meta-Llama-3.1-70B-Instruct | [recipe](https://huggingface.co/Intel/Meta-Llama-3.1-70B-Instruct-int4-inc)                                                                                                                                                                                                                                               |
| meta-llama/Meta-Llama-3.1-8B-Instruct  | [model-kaitchup-autogptq-int4*](https://huggingface.co/kaitchup/Meta-Llama-3.1-8B-Instruct-autoround-gptq-4bit-asym), [model-kaitchup-autogptq-sym-int4*](https://huggingface.co/kaitchup/Meta-Llama-3.1-8B-Instruct-autoround-gptq-4bit-sym), [recipe](https://huggingface.co/Intel/Meta-Llama-3.1-8B-Instruct-int4-inc) |
| meta-llama/Meta-Llama-3.1-8B           | [model-kaitchup-autogptq-sym-int4*](https://huggingface.co/kaitchup/Meta-Llama-3.1-8B-autoround-gptq-4bit-sym)                                                                                                                                                                                                            |
| Qwen/Qwen-VL                           | [accuracy](./examples/multimodal-modeling/Qwen-VL/README.md), [recipe](./examples/multimodal-modeling/Qwen-VL/run_autoround.sh)                                                                                                                                                                                           
| Qwen/Qwen2-7B                          | [model-autoround-int4](https://huggingface.co/Intel/Qwen2-7B-int4-inc)                                                                                                                                                                                                                                                    |
| Qwen/Qwen2-57B-A14B-Instruct           | [model-autoround-int4](https://huggingface.co/Intel/Qwen2-57B-A14B-Instruct-int4-inc)                                                                                                                                                                                                                                     |
| 01-ai/Yi-1.5-9B                        | [model-LnL-AI-autogptq-int4*](https://huggingface.co/LnL-AI/Yi-1.5-9B-4bit-gptq-autoround)                                                                                                                                                                                                                                |
| 01-ai/Yi-1.5-9B-Chat                   | [model-LnL-AI-autogptq-int4*](https://huggingface.co/LnL-AI/Yi-1.5-9B-Chat-4bit-gptq-autoround)                                                                                                                                                                                                                           |
| Intel/neural-chat-7b-v3-3              | [model-autogptq-int4](https://huggingface.co/Intel/neural-chat-7b-v3-3-int4-inc)                                                                                                                                                                                                                                          |
| Intel/neural-chat-7b-v3-1              | [model-autogptq-int4](https://huggingface.co/Intel/neural-chat-7b-v3-1-int4-inc)                                                                                                                                                                                                                                          |
| TinyLlama-1.1B-intermediate            | [model-LnL-AI-autogptq-int4*](https://huggingface.co/LnL-AI/TinyLlama-1.1B-intermediate-step-1341k-3T-autoround-lm_head-symFalse)                                                                                                                                                                                         |
| mistralai/Mistral-7B-v0.1              | [model-autogptq-lmhead-int4](https://huggingface.co/Intel/Mistral-7B-v0.1-int4-inc-lmhead), [model-autogptq-int4](https://huggingface.co/Intel/Mistral-7B-v0.1-int4-inc)                                                                                                                                                  |
| google/gemma-2b                        | [model-autogptq-int4](https://huggingface.co/Intel/gemma-2b-int4-inc)                                                                                                                                                                                                                                                     |
| tiiuae/falcon-7b                       | [model-autogptq-int4-G64](https://huggingface.co/Intel/falcon-7b-int4-inc)                                                                                                                                                                                                                                                |
| sapienzanlp/modello-italia-9b          | [model-fbaldassarri-autogptq-int4*](https://huggingface.co/fbaldassarri/modello-italia-9b-autoround-w4g128-cpu)                                                                                                                                                                                                           |
| microsoft/phi-2                        | [model-autogptq-sym-int4](https://huggingface.co/Intel/phi-2-int4-inc)                                                                                                                                                                                                                                                    |
| microsoft/Phi-3.5-mini-instruct        | [model-kaitchup-autogptq-sym-int4*](https://huggingface.co/kaitchup/Phi-3.5-Mini-instruct-AutoRound-4bit)                                                                                                                                                                                                                 |
| microsoft/Phi-3-vision-128k-instruct   | [recipe](./examples/multimodal-modeling/Phi-3-vision/run_autoround.sh)                                                                                                                                                                                                                                                    
| mistralai/Mistral-7B-Instruct-v0.2     | [accuracy](./docs/Mistral-7B-Instruct-v0.2-acc.md), [recipe](./examples/language-modeling/scripts/Mistral-7B-Instruct-v0.2.sh),  [example](./examples/language-modeling/)                                                                                                                                                 |
| mistralai/Mixtral-8x7B-Instruct-v0.1   | [accuracy](./docs/Mixtral-8x7B-Instruct-v0.1-acc.md), [recipe](./examples/language-modeling/scripts/Mixtral-8x7B-Instruct-v0.1.sh),  [example](./examples/language-modeling/)                                                                                                                                             |
| mistralai/Mixtral-8x7B-v0.1            | [accuracy](./docs/Mixtral-8x7B-v0.1-acc.md), [recipe](./examples/language-modeling/scripts/Mixtral-8x7B-v0.1.sh), [example](./examples/language-modeling/)                                                                                                                                                                |
| meta-llama/Meta-Llama-3-8B-Instruct    | [accuracy](./docs/Meta-Llama-3-8B-Instruct-acc.md), [recipe](./examples/language-modeling/scripts/Meta-Llama-3-8B-Instruct.sh), [example](./examples/language-modeling/)                                                                                                                                                  |
| google/gemma-7b                        | [accuracy](./docs/gemma-7b-acc.md), [recipe](./examples/language-modeling/scripts/gemma-7b.sh),  [example](./examples/language-modeling/)                                                                                                                                                                                 |
| meta-llama/Llama-2-7b-chat-hf          | [accuracy](./docs/Llama-2-7b-chat-hf-acc.md), [recipe](./examples/language-modeling/scripts/Llama-2-7b-chat-hf.sh), [example](./examples/language-modeling/)                                                                                                                                                              |
| Qwen/Qwen1.5-7B-Chat                   | [accuracy](./docs/Qwen1.5-7B-Chat-acc.md), [sym recipe](./examples/language-modeling/scripts/Qwen1.5-7B-Chat-sym.sh), [asym recipe ](./examples/language-modeling/scripts/Qwen1.5-7B-Chat-asym.sh), [example](./examples/language-modeling/)                                                                              |
| baichuan-inc/Baichuan2-7B-Chat         | [accuracy](./docs/baichuan2-7b-chat-acc.md), [recipe](./examples/language-modeling/scripts/baichuan2-7b-chat.sh), [example](./examples/language-modeling/)                                                                                                                                                                |         
| 01-ai/Yi-6B-Chat                       | [accuracy](./docs/Yi-6B-Chat-acc.md), [recipe](./examples/language-modeling/scripts/Yi-6B-Chat.sh), [example](./examples/language-modeling/)                                                                                                                                                                              |                                     
| facebook/opt-2.7b                      | [accuracy](./docs/opt-2.7b-acc.md), [recipe](./examples/language-modeling/scripts/opt-2.7b.sh), [example](./examples/language-modeling/)                                                                                                                                                                                  |
| bigscience/bloom-3b                    | [accuracy](./docs/bloom-3B-acc.md), [recipe](./examples/language-modeling/scripts/bloom-3b.sh), [example](./examples/language-modeling/)                                                                                                                                                                                  |
| EleutherAI/gpt-j-6b                    | [accuracy](./docs/gpt-j-6B-acc.md), [recipe](./examples/language-modeling/scripts/gpt-j-6b.sh), [example](./examples/language-modeling/)                                                                                                                                                                                  | 

## Integration

AutoRound has been integrated into multiple repositories.

[Intel Neural Compressor](https://github.com/intel/neural-compressor)

[ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel)

[pytorch/ao](https://github.com/pytorch/ao)

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

