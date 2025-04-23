<div align="center">


AutoRound
===========================
<h3> Advanced Quantization Algorithm for LLMs</h3>

[![python](https://img.shields.io/badge/python-3.9%2B-blue)](https://github.com/intel/auto-round)
[![version](https://img.shields.io/badge/release-0.5.1-green)](https://github.com/intel/auto-round)
[![license](https://img.shields.io/badge/license-Apache%202-9C27B0)](https://github.com/intel/auto-round/blob/main/LICENSE)
<a href="https://huggingface.co/OPEA">
<img alt="Model Checkpoints" src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Models-F57C00">
</a>
---
<div align="left">

AutoRound is an advanced quantization algorithm that delivers strong accuracy, even at 2-bit precision. 
It leverages sign gradient descent to fine-tune both rounding values and min-max clipping thresholds in just 200 steps. 
Designed for broad compatibility, it seamlessly supports a wide range of LLMs and is actively expanding to cover more VLMs as well. 
It also supports quantization and inference across multiple hardware platforms, including CPU, XPU, and CUDA. 
AutoRound also offers a variety of useful features, including mixed-bit tuning and inference, lm-head quantization, 
support for exporting to formats like GPTQ/AWQ/GGUF, and flexible tuning recipes. The below
image presents an overview of AutoRound. Check out our paper on [arxiv](https://arxiv.org/pdf/2309.05516) for more
details and quantized models in several Hugging Face Spaces,
e.g. [OPEA](https://huggingface.co/OPEA), [Kaitchup](https://huggingface.co/kaitchup)
and [fbaldassarri](https://huggingface.co/fbaldassarri).

<div align="center">

![](docs/imgs/autoround_overview.png)

<div align="left">

## What's New

* [2024/03] The INT2-mixed R1 model (~200GB) retains 97.9% accuracy. Check
  out [OPEA/DeepSeek-R1-int2-mixed-sym-inc](https://huggingface.co/OPEA/DeepSeek-R1-int2-mixed-sym-inc).
* [2024/01] We provide experimental support for GGUF q4_0 and q4_1 formats.
* [2024/11] We provide experimental support for VLM quantization, please check out
  the [README](./auto_round/mllm/README.md)

## Installation

### Install from pypi

```bash
# GPU
pip install auto-round

# CPU
pip install auto-round[cpu]

# HPU
pip install auto-round-lib
```

<details>
  <summary>Build from Source</summary>

  ```bash
  # GPU
  pip install .

  # CPU
  pip install .[cpu]

  # HPU
  python setup.py install lib
  ```

</details>

## Model Quantization

### Command Line Usage (Gaudi/CPU/XPU/GPU)

A user guide detailing the full list of supported arguments is provided by calling ```auto-round -h``` on the terminal.
Set the format you want in `format` and
multiple formats exporting has been supported. Please check out [step-by-step-instruction](./docs/step_by_step.md) for
more details about calibration dataset or evaluation.

```bash
auto-round \
    --model facebook/opt-125m \
    --bits 4 \
    --group_size 128 \
    --format "auto_gptq,auto_awq,auto_round" \
    --output_dir ./tmp_autoround
```

We offer two configurations, `auto-round-best` and `auto-round-light`, designed for optimal accuracy and improved speed,
respectively. Details are as follows.
<details>
  <summary>Other Recipes</summary>

  ```bash
## best accuracy, 3X slower, low_gpu_mem_usage could save ~20G but ~30% slower
auto-round-best \
    --model facebook/opt-125m \
    --bits 4 \
    --group_size 128 \
    --low_gpu_mem_usage 
  ```

  ```bash
## light accuracy, 2-3X speedup, slight accuracy drop at W4 and larger accuracy drop at W2
auto-round-light \
    --model facebook/opt-125m \
    --bits 4 \
    --group_size 128 \

  ```

  <!-- ```bash
auto-round-fast \
## fast and low memory, 2-3X speedup, slight accuracy drop at W4G128
    --model facebook/opt-125m \
    --bits 4 \
    --group_size 128 \
  ``` -->

</details>

In conclusion, we recommend using **auto-round for INT4 and auto-round-best for INT2**. However, you may adjust the
configuration to suit your specific requirements and available resources.

W4G128 Average Accuracy of 13 tasks and Time Cost Results(Testing was conducted on the Nvidia A100 80G using the version
of PyTorch 2.6.0 with enable_torch_compile):

| Model   | Qwen2.5-0.5B-Instruct | Falcon3-3B      | Qwen2.5-7B-Instruct | Meta-Llama-3.1-8B-Instruct | Falcon3-10B     | Qwen2.5-72B-Instruct |
|---------|-----------------------|-----------------|---------------------|----------------------------|-----------------|----------------------|
| 16bits  | 0.4192                | 0.5203          | 0.6470              | 0.6212                     | 0.6151          | 0.7229               |
| Best    | **0.4137**(7m)        | **0.5142**(23m) | 0.6426(58m)         | **0.6116**(65m)            | **0.6092**(81m) | 0.7242(575m)         |
| Default | 0.4129(2m)            | 0.5133(6m)      | 0.6441(13m)         | 0.6106(13m)                | 0.6080(18m)     | **0.7252**(118m)     |
| Light   | 0.4052(2m)            | 0.5108(3m)      | **0.6453**(5m)      | 0.6104(6m)                 | 0.6063(6m)      | 0.7243(37m)          |

<details>
  <summary>W2G64 results</summary>
W2G64 Average Accuracy of 13 tasks and Time Cost Results(Testing was conducted on the Nvidia A100 80G using the version of PyTorch 2.6.0 with enable_torch_compile). We recommend using higher precision for the head, tail, and non-expert modules to alleviate the significant accuracy drop.

| Model   | Qwen2.5-0.5B-Instruct | Falcon3-3B      | Qwen2.5-7B-Instruct | Falcon3-10B     | Qwen2.5-72B-Instruct |
  |---------|-----------------------|-----------------|---------------------|-----------------|----------------------|
| 16bits  | 0.4192                | 0.5203          | 0.6470              | 0.6151          | 0.7229               |
| Best    | **0.2989**(6m)        | **0.4267**(24m) | **0.5343**(56m)     | **0.5207**(79m) | **0.6715**(564m)     |
| Default | 0.2878(2m)            | 0.4219(6m)      | 0.5209(13m)         | 0.5133(18m)     | 0.6713(122m)         |
| Light   | 0.2760(2m)            | 0.4063(3m)      | 0.4764(5m)          | 0.4810(7m)      | 0.6581(38m)          |

</details>

### API Usage (Gaudi/CPU/XPU/GPU)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

from auto_round import AutoRound

bits, group_size, sym = 4, 128, True
autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym)

## the best accuracy, 4-5X slower, low_gpu_mem_usage could save ~20G but ~30% slower
# autoround = AutoRound(model, tokenizer, nsamples=512, iters=1000, low_gpu_mem_usage=True, bits=bits, group_size=group_size, sym=sym)

## 2-3X speedup, slight accuracy drop at W4G128
# autoround = AutoRound(model, tokenizer, nsamples=128, iters=50, lr=5e-3, bits=bits, group_size=group_size, sym=sym )

output_dir = "./tmp_autoround"
## format= 'auto_round'(default), 'auto_gptq', 'auto_awq'
autoround.quantize_and_save(output_dir, format='auto_round') 
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

If you encounter issues during quantization, try setting iters=0 (to enable RTN) and use group_size=32 for better
results.


<details>
  <summary>Click to expand</summary>

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
    model_name, trust_remote_code=True, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

## quantize the model
bits, group_size, sym = 4, 128, True
autoround = AutoRoundMLLM(model, tokenizer, processor,
                          bits=bits, group_size=group_size, sym=sym)
autoround.quantize()

# save the quantized model, set format='auto_gptq' or 'auto_awq' to use other formats
output_dir = "./tmp_autoround"
autoround.save_quantized(output_dir, format='auto_round', inplace=True)
```

</details>

### Export Formats

**AutoRound Format**: This format is well-suited for CPU, HPU devices, 2 bits, as well as mixed-precision
inference. **[2,3,4,8] bits are supported**. However, it has not yet gained widespread community adoption.

**AutoGPTQ Format**: This format is well-suited for symmetric quantization on CUDA devices and is widely adopted by the
community, **[2,3,4,8] bits are supported**. However, **the
asymmetric kernel has issues** that can cause considerable accuracy drops, particularly at 2-bit quantization and small
models. Besides, recently 3 bits may have some accuracy issues in Transformers.

**AutoAWQ Format**: This format is well-suited for asymmetric 4-bit quantization on CUDA devices and is widely
adopted within the community, **only 4-bits quantization is supported**.

**GGUF** Format: This format is well-suited for CPU devices and is widely adopted by the community, **only q4_0 and
q4_1 (W4G32) is supported in our repo**.

### Quantization Costs

Testing was conducted on the Nvidia A100 80G using the nightly version of PyTorch 2.6.0.dev20241029+cu124. Please note
that data
loading and packing costs have been excluded from the evaluation. **We recommend enabling torch.compile for PyTorch
versions 2.6 and above.**

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

#### Gaudi/CPU/XPU/CUDA

**Please avoid manually moving the quantized model to a different device** (e.g., model.to('cpu')) during inference, as this may cause unexpected exceptions.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRoundConfig  ## must import for auto-round format

quantized_model_path = "./tmp_autoround"
model = AutoModelForCausalLM.from_pretrained(quantized_model_path,
                                             device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
```

### Specify backend

AutoRound automatically selects the best available backend based on the installed libraries and prompts the user to
install additional libraries when a better backend is found. On CUDA, the default priority is Marlin > ExLLaMAV2 >
Triton, but the final choice depends on factors such as bits, group_size, packing format compatibility, etc. And the backend may not always be the most suitable for certain devices. Please refer
to the following table for the details and specify the backend you want.

| Name                                 | Devices | Bits    | Dtypes    | Priority | Packing format  | Requirements                  |
|--------------------------------------|---------|---------|-----------|----------|-----------------|-------------------------------|
| ipex                                 | cpu/xpu | 4       | BF16/FP16 | 5        | gptq_zp+-1/awq  | intel-extension-for-pytorch   |
| itrex                                | cpu     | 2,4,8   | BF16/FP16 | 0        | gptq_zp+-1/awq  | intel-extension-for-transformers |
| marlin                               | cuda    | 4,8     | BF16/FP16 | 6        | gptq/gptq_zp+-1 | gptqmodel                     |
| exllamav2 or<br/>gptqmodel:exllamav2 | cuda    | 4       | BF16/FP16 | 5        | gptq            | gptqmodel                     |
| exllamav2 or<br/>gptq:exllamav2      | cuda    | 4       | FP16      | 5        | gptq_zp+-1      | auto-gptq                     |
| gptq:cuda                            | cuda    | 2,3,4,8 | FP16      | 0        | gptq_zp+-1      | auto-gptq                     |
| triton                               | cuda    | 2,4,8   | BF16/FP16 | 1        | gptq/gptq_zp+-1 | auto-round                    |
| awq                                  | cuda    | 4       | FP16      | 5        | awq             | auto-awq                      |
| hpu                                  | hpu     | 4       | BF16      | 0        | gptq/gptq_zp+-1 | auto-round                    |

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import  AutoRoundConfig

quantized_model_path = "./tmp_autoround"
quantization_config = AutoRoundConfig(backend="auto")
model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto",
                                             torch_dtype="auto",
                                             quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
```

### Convert GPTQ/AWQ format to AutoRound

Most GPTQ/AWQ models can be converted to the AutoRound format for better compatibility and support with Intel devices.
Please note that the quantization config will be changed if the model is serialized.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRoundConfig  ## must import for auto-round format

model_name = "ybelkada/opt-125m-gptq-4bit"
quantization_config = AutoRoundConfig()
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype="auto",
                                             quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
```

#### Evaluation

<details>
  <summary>Click to expand</summary>

```bash
auto-round --model saved_quantized_model \
    --eval \
    --task lambada_openai \
    --eval_bs 1
```

</details>

## Support List

AutoRound supports basically all the major large language models.

<details>
  <summary>Supported Models List</summary>

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
| Qwen/Qwen2.5-7B-Instruct                  | [model-opea-int4-sym-autoround](https://huggingface.co/OPEA/Qwen2.5-7B-Instruct-int4-sym-inc),[model-opea-int4-sym-autogptq](https://huggingface.co/OPEA/Qwen2.5-7B-Instruct-int4-sym-inc) [model-kaitchup-autogptq-int4*](https://huggingface.co/kaitchup/Qwen2.5-7B-Instruct-AutoRound-GPTQ-asym-4bit), [recipe](./docs/Qwen2.5-7B-Instruct-sym.md)                                                                                                                                                                              |
| Qwen/Qwen2.5-14B-Instruct                 | [model-opea-int4-sym-autoround](https://huggingface.co/OPEA/Qwen2.5-14B-Instruct-int4-sym-inc),[model-opea-int4-sym-autogptq](https://huggingface.co/OPEA/Qwen2.5-14B-Instruct-int4-sym-inc)                                                                                                                                                                                                                                                                                                                                       |
| Qwen/Qwen2.5-32B-Instruct                 | [model-opea-int4-sym-autoround](https://huggingface.co/OPEA/Qwen2.5-32B-Instruct-int4-sym-inc)                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Qwen/Qwen2.5-Coder-32B-Instruct           | [model-kaitchup-autogptq-int4*](https://huggingface.co/kaitchup/Qwen2.5-Coder-32B-Instruct-AutoRound-GPTQ-4bit)                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Qwen/Qwen2.5-72B-Instruct                 | [model-opea-int4-sym-autoround](https://huggingface.co/OPEA/Qwen2.5-72B-Instruct-int4-sym-inc),[model-opea-int4-sym-autogptq](https://huggingface.co/OPEA/Qwen2.5-72B-Instruct-int4-sym-inc), [model-kaitchup-autogptq-int4*](https://huggingface.co/kaitchup/Qwen2.5-72B-Instruct-AutoRound-GPTQ-4bit),  [model-kaitchup-autogptq-int2*](https://huggingface.co/kaitchup/Qwen2.5-72B-Instruct-AutoRound-GPTQ-2bit), [recipe](./docs/Qwen2.5-72B-Instruct-sym.md)                                                                  |
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

</details> 

### VLM Support Matrix

For most VLMs, we typically support the default quantization configuration, which involves quantizing only the language
component while excluding the visual component. Besides, we also support quantizing non-text modules of models that
follow the Hugging Face standard, i.e., those with a typical processor, though inference may have some issues due to
model architecture or kernel limitations.

| Model                          | calibration dataset | quant nontext module | Quantized Model Link                                                                                                                                                                                                                                                                                                                                                                                                                                             | 
|--------------------------------|---------------------|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| allenai/Molmo                  | pile                | X                    | [Molmo-7B-D-0924-int4-sym](https://huggingface.co/OPEA/Molmo-7B-D-0924-int4-sym-inc), [Molmo-72B-0924-int4-sym-gptq](https://huggingface.co/OPEA/Molmo-72B-0924-int4-sym-gptq-inc), [Molmo-72B-0924-int4-sym](https://huggingface.co/OPEA/Molmo-72B-0924-int4-sym-inc)                                                                                                                                                                                           |
| deepseek-ai/deepseek-vl2       | pile/llava          | √                    | [deepseek-vl2-int4-sym-gptq](https://huggingface.co/OPEA/deepseek-vl2-int4-sym-gptq-inc)                                                                                                                                                                                                                                                                                                                                                                         |
| google/gemma-3                 | pile/llava          | √                    | [gemma-3-12b-it-AutoRound-gguf-q4-0](https://huggingface.co/OPEA/gemma-3-12b-it-AutoRound-gguf-q4-0), [gemma-3-27b-it-AutoRound-gguf-q4-0](https://huggingface.co/OPEA/gemma-3-27b-it-AutoRound-gguf-q4-0), [gemma-3-12b-it-int4-AutoRound-cpu](https://huggingface.co/OPEA/gemma-3-12b-it-int4-AutoRound-cpu), [gemma-3-27b-it-int4-AutoRound-cpu](https://huggingface.co/OPEA/gemma-3-27b-it-int4-AutoRound-cpu)                                               |
| HuggingFaceTB/SmolVLM          | pile/llava          | √                    | [SmolVLM-Instruct-int4-sym](https://huggingface.co/OPEA/SmolVLM-Instruct-int4-sym-inc)                                                                                                                                                                                                                                                                                                                                                                           |
| ibm-granite/granite-vision-3.2 | pile/llava          | -                    |                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| liuhaotian/Llava-v1.5          | pile/llava          | X                    | [llava-v1.5-7b-int4-sym](https://huggingface.co/OPEA/llava-v1.5-7b-int4-sym-inc)                                                                                                                                                                                                                                                                                                                                                                                 |
| meta-llama/Llama-3.2-Vision    | llava               | √                    | [Llama-3.2V-11B-cot-int4-sym](https://huggingface.co/OPEA/Llama-3.2V-11B-cot-int4-sym-inc), [Llama-3.2-11B-Vision-Instruct-qvision-int4-sym](https://huggingface.co/OPEA/Llama-3.2-11B-Vision-Instruct-qvision-int4-sym-inc), [Llama-3.2-90B-Vision-Instruct-int4-sym](https://huggingface.co/OPEA/Llama-3.2-90B-Vision-Instruct-int4-sym-inc), [Llama-3.2-11B-Vision-Instruct-int4-sym](https://huggingface.co/OPEA/Llama-3.2-11B-Vision-Instruct-int4-sym-inc) |
| microsoft/Phi3.5-Vision        | pile/llava          | √                    | [Phi-3.5-vision-instruct-int4-sym](https://huggingface.co/OPEA/Phi-3.5-vision-instruct-int4-sym-inc), [Phi-3.5-vision-instruct-qvision-int4-sym](https://huggingface.co/OPEA/Phi-3.5-vision-instruct-qvision-int4-sym-inc)                                                                                                                                                                                                                                       |
| mistralai/Mistral-Small-3.1    | pile/llava          | X                    | [Mistral-Small-3.1-24B-Instruct-2503-int4-AutoRound-gptq-sym](https://huggingface.co/OPEA/Mistral-Small-3.1-24B-Instruct-2503-int4-AutoRound-gptq-sym), [Mistral-Small-3.1-24B-Instruct-2503-int4-AutoRound-awq-sym](https://huggingface.co/OPEA/Mistral-Small-3.1-24B-Instruct-2503-int4-AutoRound-awq-sym)                                                                                                                                                     |
| moonshotai/Kimi-VL             | pile/llava          | √                    |                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Qwen/Qwen2-VL                  | pile/llava          | -                    | [Qwen2-VL-7B-Instruct-int4-sym](https://huggingface.co/OPEA/Qwen2-VL-7B-Instruct-int4-sym-inc), [Qwen2-VL-72B-Instruct-int4-sym](https://huggingface.co/OPEA/Qwen2-VL-72B-Instruct-int4-sym-inc), [Qwen2-VL-72B-Instruct-int2-sym](https://huggingface.co/OPEA/Qwen2-VL-72B-Instruct-int2-sym-inc)                                                                                                                                                               |
| rhymes-ai/Aria                 | pile/llava          | √                    |                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| THUDM/CogVLM2                  | pile/llava          | √                    | [cogvlm2-llama3-chat-19B-int4-sym](https://huggingface.co/OPEA/cogvlm2-llama3-chat-19B-int4-sym-inc), [cogvlm2-llama3-chat-19B-qvision-int4-sym](https://huggingface.co/OPEA/cogvlm2-llama3-chat-19B-qvision-int4-sym-inc)                                                                                                                                                                                                                                       |
| THUDM/glm-4v                   | pile                | X                    | [glm-4v-9b-int4-sym](https://huggingface.co/OPEA/glm-4v-9b-int4-sym-inc)                                                                                                                                                                                                                                                                                                                                                                                         |

√ means support, - means support to export but cannot infer, X means not support.

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











