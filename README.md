<div align="center">

AutoRound
===========================
<h3> Advanced Quantization Algorithm for LLMs</h3>

[![python](https://img.shields.io/badge/python-3.9%2B-blue)](https://github.com/intel/auto-round)
[![version](https://img.shields.io/badge/release-0.4.4-green)](https://github.com/intel/auto-round)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/intel/auto-round/blob/main/LICENSE)
---
<div align="left">

AutoRound is an advanced quantization algorithm for low-bits LLM/VLM inference. It's tailored for a wide range
of models. AutoRound adopts sign gradient descent to fine-tune rounding values and minmax values of weights in just 200
steps,
which competes impressively against recent methods without introducing any additional inference overhead and keeping low
tuning cost. The below
image presents an overview of AutoRound. Check out our paper on [arxiv](https://arxiv.org/pdf/2309.05516) for more
details and quantized models in several Hugging Face Spaces, e.g. [OPEA](https://huggingface.co/OPEA), [Kaitchup](https://huggingface.co/kaitchup) and [fbaldassarri](https://huggingface.co/fbaldassarri).

<div align="center">

![](docs/imgs/autoround_overview.png)

<div align="left">

## What's New

* [2024/12] Many quantized LLMs/VLMs using AutoRound are released in [OPEA Space](https://huggingface.co/OPEA)
* [2024/11] We provide experimental support for VLM quantization, please check out
  the [README](./auto_round/mllm/README.md)
* [2024/11] We provide some tips and tricks for LLM&VLM quantization, please check
  out [this blog](https://medium.com/@NeuralCompressor/10-tips-for-quantizing-llms-and-vlms-with-autoround-923e733879a7)

[//]: # (* [2024/10] AutoRound has been integrated to [torch/ao]&#40;https://github.com/pytorch/ao&#41;, check out)

[//]: # (  their [release note]&#40;https://github.com/pytorch/ao/releases/tag/v0.6.1&#41;)

[//]: # (* [2024/10] Important update: We now support full-range symmetric quantization and have made it the default)

[//]: # (  configuration. This configuration is typically better or comparable to asymmetric quantization and significantly)

[//]: # (  outperforms other symmetric variants, especially at low bit-widths like 2-bit, check)

[//]: # (  out [some accuracy data]&#40;./docs/full_range_sym.md&#41;.)

[//]: # (* [2024/08] AutoRound format supports Intel Gaudi2 devices. Please refer)

[//]: # (  to [Intel/Qwen2-7B-int4-inc]&#40;https://huggingface.co/Intel/Qwen2-7B-int4-inc&#41;.)

[//]: # (* [2024/08] AutoRound introduces several experimental features, including fast tuning of norm/bias parameters &#40;for 2-bit)

[//]: # (  and W4A4, check out [more details]&#40;./docs/tuning_norm_bias.md&#41;&#41;, activation quantization, and the mx_fp data type.)

## Installation

### Install from pypi

```bash
# GPU
pip install auto-round[gpu]

# CPU
pip install auto-round[cpu]

# HPU
pip install auto-round-lib
```

<details>
  <summary>Build from Source</summary>

  ```bash
  # GPU
  pip install .[gpu]

  # CPU
  pip install .[cpu]

  # HPU
  python setup.py install lib
  ```

</details>

## Model Quantization

### Basic Usage (Gaudi2/CPU/GPU)

A user guide detailing the full list of supported arguments is provided by calling ```auto-round -h``` on the terminal.
Set the format you want in `format` and
multiple formats exporting has been supported. Please check out [step-by-step-instruction](./docs/step_by_step.md) for
more details about calibration dataset or evaluation.

```bash
auto-round \
    --model facebook/opt-125m \
    --bits 4 \
    --group_size 128 \
    --format "auto_gptq,auto_round" \
    --disable_eval \
    --output_dir ./tmp_autoround
```

We provide two recipes for best accuracy and fast running speed with low memory. Details as below.
<details>
  <summary>Other Recipes</summary>

  ```bash
## best accuracy, 3X slower, low_gpu_mem_usage could save ~20G but ~30% slower
auto-round-best \
    --model facebook/opt-125m \
    --bits 4 \
    --group_size 128 \
    --low_gpu_mem_usage \
    --disable_eval 
  ```

  ```bash
## fast and low memory, 2-3X speedup, slight accuracy drop at W4G128
auto-round-fast \
    --model facebook/opt-125m \
    --bits 4 \
    --group_size 128 \
    --disable_eval 
  ```

</details>

### API Usage (Gaudi2/CPU/GPU)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

from auto_round import AutoRound

bits, group_size, sym = 4, 128, True
autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym)

## the best accuracy, 3X slower, low_gpu_mem_usage could save ~20G but ~30% slower
# autoround = AutoRound(model, tokenizer, nsamples=512, iters=1000, low_gpu_mem_usage=True, bits=bits, group_size=group_size, sym=sym)

## fast and low memory, 2-3X speedup, slight accuracy drop at W4G128
# autoround = AutoRound(model, tokenizer, nsamples=128, iters=200, seqlen=512, batch_size=4, bits=bits, group_size=group_size, sym=sym )

autoround.quantize()
output_dir = "./tmp_autoround"
## format= 'auto_round'(default in version>0.3.0), 'auto_gptq', 'auto_awq'
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

### API Usage for VLMs

**This feature is experimental and may be subject to changes**, including potential bug fixes, API modifications, or
adjustments to default hype-parameters

By default, AutoRoundMLLM only quantizes the text module of VLMs and uses `NeelNanda/pile-10k` for calibration. To
quantize the entire model, you can enable `quant_nontext_module` by setting it to True, though support for this feature
is limited. For more information, please refer to the AutoRoundMLLM [readme](./auto_round/mllm/README.md).

```python
from auto_round import AutoRoundMLLM
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer

## load the model
model_name = "Qwen/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

## quantize the model
bits, group_size, sym = 4, 128, True
autoround = AutoRoundMLLM(model, tokenizer, processor,
                          bits=bits, group_size=group_size, sym=sym)
autoround.quantize()

# save the quantized model, set format='auto_gptq' to use AutoGPTQ format
output_dir = "./tmp_autoround"
autoround.save_quantized(output_dir, format='auto_round', inplace=True)
```

### Quantization Costs

Testing was conducted on the Nvidia A100 80G using the nightly version of PyTorch 2.6.0.dev20241029+cu124. Please note
that data
loading and packing costs have been excluded from the evaluation. **We enable torch.compile for Torch 2.6, but not for
2.5
due to encountered issues.**

To optimize GPU memory usage, in addition to activating `low_gpu_mem_usage`, you can set `gradient_accumulate_steps=8`
and a
`batch_size=1`, though this may increase tuning time.

The 3B and 14B models were evaluated on Qwen 2.5, the 8X7B model is Mixtral, while the remaining models utilized LLaMA
3.1.

| Torch version/Config W4G128                                                                 | 3B            | 8B             | 14B            | 70B             | 8X7B           |
|---------------------------------------------------------------------------------------------|---------------|----------------|----------------|-----------------|----------------|
| 2.6  with torch compile                                                                     | 7min<br/>10GB | 12min<br/>18GB | 23min<br/>22GB | 120min<br/>42GB | 28min<br/>46GB |
| 2.6  with torch compile <br/> low_gpu_mem_usage=True                                        | 12min<br/>6GB | 19min<br/>10GB | 33min<br/>11GB | 140min<br/>25GB | 38min<br/>36GB |
| 2.6  with torch compile <br/> low_gpu_mem_usage=True <br/> gradient_accumulate_steps=8,bs=1 | 15min<br/>3GB | 25min<br/>6GB  | 45min<br/>7GB  | 187min<br/>19GB | 75min<br/>36GB |
| 2.5  w/o torch compile                                                                      | 8min<br/>10GB | 16min<br/>20GB | 30min<br/>25GB | 140min<br/>49GB | 50min<br/>49GB |

## Model Inference

Please run the quantization code first

### AutoRound format

**CPU**: pip install intel-extension-for-pytorch(much higher speed on Intel CPU) or pip
install intel-extension-for-transformers,

**HPU**: docker image with Gaudi Software Stack is recommended. More details can be found
in [Gaudi Guide](https://docs.habana.ai/en/latest/).

**CUDA**: no extra operations for sym quantization, for asym quantization, need to install auto-round from source

#### CPU/HPU/CUDA

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRoundConfig

backend = "auto"  ##cpu, hpu, cuda
quantization_config = AutoRoundConfig(
    backend=backend
)
quantized_model_path = "./tmp_autoround"
model = AutoModelForCausalLM.from_pretrained(quantized_model_path,
                                             device_map=backend.split(':')[0],
                                             quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
```

<br>
<details>
  <summary>Evaluation</summary>

```bash
auto-round --model saved_quantized_model \
    --eval \
    --task lambada_openai \
    --eval_bs 1
```

</details>

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

## Support List

AutoRound supports basically all the major large language models.

Please note that an asterisk (*) indicates third-party quantized models, which may lack accuracy data and use a
different recipe. We greatly appreciate their efforts and encourage more users to share their models, as we cannot
release most of the models ourselves.

 Model                                     | Supported                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
|-------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| nvidia/Llama-3.1-Nemotron-70B-Instruct-HF | [model-opea-int4-sym-autoround](https://huggingface.co/OPEA/Llama-3.1-Nemotron-70B-Instruct-HF-int4-sym-inc),  [model-opea-int4-sym-autogptq](https://huggingface.co/OPEA/Llama-3.1-Nemotron-70B-Instruct-HF-int4-sym-inc),                                                                                                                                                                                                                                                                                                        |
| meta-llama/Llama-3.2-90B-Vision-Instruct  | [model-opea-int4-sym-autoround](https://huggingface.co/OPEA/Llama-3.2-90B-Vision-Instruct-int4-sym-inc), [model-opea-int4-sym-autogptq](https://huggingface.co/OPEA/Llama-3.2-90B-Vision-Instruct-int4-sym-inc)                                                                                                                                                                                                                                                                                                                    |
| Qwen/QwQ-32B-Preview                      | [model-opea-int4-sym-autoround-mixed](https://huggingface.co/OPEA/QwQ-32B-Preview-int4-sym-mixed-inc),[model-opea-int4-sym-autoawq-mixed](https://huggingface.co/OPEA/QwQ-32B-Preview-int4-sym-mixed-awq-inc)                                                                                                                                                                                                                                                                                                                      |
| THUDM/cogvlm2-llama3-chat-19B             | [model-opea-int4-sym-autoround](https://huggingface.co/OPEA/cogvlm2-llama3-chat-19B-int4-sym-inc)                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Qwen/Qwen2-VL-Instruct                    | [model-opea-int4-sym-autoround](https://huggingface.co/OPEA/Qwen2-VL-7B-Instruct-int4-sym-inc),[model-opea-int4-sym-autogptq](https://huggingface.co/OPEA/Qwen2-VL-7B-Instruct-int4-sym-inc)                                                                                                                                                                                                                                                                                                                                       |
| meta-llama/Llama-3.2-11B-Vision           | [model-opea-int4-sym-autoround](https://huggingface.co/OPEA/Llama-3.2-11B-Vision-Instruct-int4-sym-inc), [model-opea-int4-sym-autogptq](https://huggingface.co/OPEA/Llama-3.2-11B-Vision-Instruct-int4-sym-inc)                                                                                                                                                                                                                                                                                                                    |
| microsoft/Phi-3.5-vision-instruct         | [model-opea-int4-sym-autoround](https://huggingface.co/OPEA/Phi-3.5-vision-instruct-int4-sym-inc), [model-opea-int4-sym-gptq](https://huggingface.co/OPEA/Phi-3.5-vision-instruct-int4-sym-inc)                                                                                                                                                                                                                                                                                                                                    |
| liuhaotian/llava-v1.5-7b                  | [model-opea-int4-sym-autoround](https://huggingface.co/OPEA/llava-v1.5-7b-int4-sym-inc),[model-opea-int4-sym-autogptq](https://huggingface.co/OPEA/llava-v1.5-7b-int4-sym-inc)                                                                                                                                                                                                                                                                                                                                                     |
| Qwen/Qwen2.5-7B-Instruct                  | [model-opea-int4-sym-autoround](https://huggingface.co/OPEA/Qwen2.5-7B-Instruct-int4-sym-inc),[model-opea-int4-sym-autogptq](https://huggingface.co/OPEA/Qwen2.5-7B-Instruct-int4-sym-inc) [model-kaitchup-autogptq-int4*](https://huggingface.co/kaitchup/Qwen2.5-7B-Instruct-AutoRound-GPTQ-asym-4bit), [recipe](./docs/Qwen2.5-7B-Instruct-sym.md)                                                                                                                                                                    |
| Qwen/Qwen2.5-14B-Instruct                 | [model-opea-int4-sym-autoround](https://huggingface.co/OPEA/Qwen2.5-14B-Instruct-int4-sym-inc),[model-opea-int4-sym-autogptq](https://huggingface.co/OPEA/Qwen2.5-14B-Instruct-int4-sym-inc)                                                                                                                                                                                                                                                                                                                                       |
| Qwen/Qwen2.5-32B-Instruct                 | [model-opea-int4-sym-autoround](https://huggingface.co/OPEA/Qwen2.5-32B-Instruct-int4-sym-inc)                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Qwen/Qwen2.5-Coder-32B-Instruct           | [model-kaitchup-autogptq-int4*](https://huggingface.co/kaitchup/Qwen2.5-Coder-32B-Instruct-AutoRound-GPTQ-4bit)                                                                                                                                                                                                                                                                                                                                                                                                          |
| Qwen/Qwen2.5-72B-Instruct                 | [model-opea-int4-sym-autoround](https://huggingface.co/OPEA/Qwen2.5-72B-Instruct-int4-sym-inc),[model-opea-int4-sym-autogptq](https://huggingface.co/OPEA/Qwen2.5-72B-Instruct-int4-sym-inc), [model-kaitchup-autogptq-int4*](https://huggingface.co/kaitchup/Qwen2.5-72B-Instruct-AutoRound-GPTQ-4bit),  [model-kaitchup-autogptq-int2*](https://huggingface.co/kaitchup/Qwen2.5-72B-Instruct-AutoRound-GPTQ-2bit), [recipe](./docs/Qwen2.5-72B-Instruct-sym.md)                                              |
| meta-llama/Meta-Llama-3.1-70B-Instruct    | [model-opea-int4-sym-autoround](https://huggingface.co/OPEA/Meta-Llama-3.1-70B-Instruct-int4-sym-inc), [model-opea-int4-sym-autogptq](https://huggingface.co/OPEA/Meta-Llama-3.1-70B-Instruct-int4-sym-inc),[model-opea-int4-asym-autoround](https://huggingface.co/OPEA/Meta-Llama-3.1-70B-Instruct-int4-asym-inc)                                                                                                                                                                                                                |
| meta-llama/Meta-Llama-3.1-8B-Instruct     | [model-opea-int4-sym-autoround](https://huggingface.co/OPEA/Meta-Llama-3.1-8B-Instruct-int4-sym-inc),[model-opea-int4-sym-autogptq](https://huggingface.co/OPEA/Meta-Llama-3.1-8B-Instruct-int4-sym-inc),[model-kaitchup-autogptq-int4*](https://huggingface.co/kaitchup/Meta-Llama-3.1-8B-Instruct-autoround-gptq-4bit-asym), [model-kaitchup-autogptq-sym-int4*](https://huggingface.co/kaitchup/Meta-Llama-3.1-8B-Instruct-autoround-gptq-4bit-sym), [recipe](https://huggingface.co/Intel/Meta-Llama-3.1-8B-Instruct-int4-inc) |
| meta-llama/Meta-Llama-3.1-8B              | [model-kaitchup-autogptq-sym-int4*](https://huggingface.co/kaitchup/Meta-Llama-3.1-8B-autoround-gptq-4bit-sym)                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Qwen/Qwen2-7B                             | [model-autoround-sym-int4](https://huggingface.co/Intel/Qwen2-7B-int4-inc), [model-autogptq-sym-int4](https://huggingface.co/Intel/Qwen2-7B-int4-inc)                                                                                                                                                                                                                                                                                                                                                                              |
| THUDM/glm-4-9b-chat                       | [model-opea-int4-sym-autoround](https://huggingface.co/OPEA/glm-4-9b-chat-int4-sym-inc),[model-opea-int4-sym-autogptq](https://huggingface.co/OPEA/glm-4-9b-chat-int4-sym-inc)                                                                                                                                                                                                                                                                                                                                                     |
| Qwen/Qwen2-57B-A14B-Instruct              | [model-autoround-sym-int4](https://huggingface.co/Intel/Qwen2-57B-A14B-Instruct-int4-inc),[model-autogptq-sym-int4](https://huggingface.co/Intel/Qwen2-57B-A14B-Instruct-int4-inc)                                                                                                                                                                                                                                                                                                                                                 |
| 01-ai/Yi-1.5-9B                           | [model-LnL-AI-autogptq-int4*](https://huggingface.co/LnL-AI/Yi-1.5-9B-4bit-gptq-autoround)                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 01-ai/Yi-1.5-9B-Chat                      | [model-LnL-AI-autogptq-int4*](https://huggingface.co/LnL-AI/Yi-1.5-9B-Chat-4bit-gptq-autoround)                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Intel/neural-chat-7b-v3-3                 | [model-autogptq-int4](https://huggingface.co/Intel/neural-chat-7b-v3-3-int4-inc)                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Intel/neural-chat-7b-v3-1                 | [model-autogptq-int4](https://huggingface.co/Intel/neural-chat-7b-v3-1-int4-inc)                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| TinyLlama-1.1B-intermediate               | [model-LnL-AI-autogptq-int4*](https://huggingface.co/LnL-AI/TinyLlama-1.1B-intermediate-step-1341k-3T-autoround-lm_head-symFalse)                                                                                                                                                                                                                                                                                                                                                                                                  |
| mistralai/Mistral-7B-v0.1                 | [model-autogptq-lmhead-int4](https://huggingface.co/Intel/Mistral-7B-v0.1-int4-inc-lmhead), [model-autogptq-int4](https://huggingface.co/Intel/Mistral-7B-v0.1-int4-inc)                                                                                                                                                                                                                                                                                                                                                           |
| google/gemma-2b                           | [model-autogptq-int4](https://huggingface.co/Intel/gemma-2b-int4-inc)                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| tiiuae/falcon-7b                          | [model-autogptq-int4-G64](https://huggingface.co/Intel/falcon-7b-int4-inc)                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| sapienzanlp/modello-italia-9b             | [model-fbaldassarri-autogptq-int4*](https://huggingface.co/fbaldassarri/modello-italia-9b-autoround-w4g128-cpu)                                                                                                                                                                                                                                                                                                                                                                                                                    |
| microsoft/phi-2                           | [model-autoround-sym-int4](https://huggingface.co/Intel/phi-2-int4-inc) [model-autogptq-sym-int4](https://huggingface.co/Intel/phi-2-int4-inc)                                                                                                                                                                                                                                                                                                                                                                                     |
| microsoft/Phi-3.5-mini-instruct           | [model-kaitchup-autogptq-sym-int4*](https://huggingface.co/kaitchup/Phi-3.5-Mini-instruct-AutoRound-4bit)                                                                                                                                                                                                                                                                                                                                                                                                                          |
| mistralai/Mistral-7B-Instruct-v0.2        | [outdated-recipe](./docs/Mistral-7B-Instruct-v0.2-asym-recipe.md)                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| mistralai/Mixtral-8x7B-Instruct-v0.1      | [outdated-recipe](./docs/Mixtral-8x7B-Instruct-v0.1-asym-recipe.md)                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| mistralai/Mixtral-8x7B-v0.1               | [outdated-recipe](./docs/Mixtral-8x7B-v0.1-asym-acc.md)                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| meta-llama/Meta-Llama-3-8B-Instruct       | [outdated-recipe](./docs/Meta-Llama-3-8B-Instruct-asym-recipe.md)                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| google/gemma-7b                           | [outdated-recipe](./docs/gemma-7b-asym-recipe.md)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| meta-llama/Llama-2-7b-chat-hf             | [outdated-recipe](./docs/Llama-2-7b-chat-hf-asym-recipe.md)                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | 
| baichuan-inc/Baichuan2-7B-Chat            | [outdated-recipe](./docs/baichuan2-7b-cha-asym-recipe.md)                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |         
| 01-ai/Yi-6B-Chat                          | [outdated-recipe](./docs/Yi-6B-Chat-asym-recipe.md)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |                                     
| facebook/opt-2.7b                         | [outdated-recipe](./docs/opt-2.7b-asym-recipe.md)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| bigscience/bloom-3b                       | [outdated-recipe](./docs/bloom-3B-asym-recipe.md)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| EleutherAI/gpt-j-6b                       | [outdated-recipe](./docs/gpt-j-6B-asym-recipe.md)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 

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



