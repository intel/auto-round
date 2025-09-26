Step-by-Step
============

This document presents step-by-step instructions for auto-round llm quantization. For vlms quantization, please refer to [vlms user guide](../auto_round/mllm/README.md)

* [1 Prerequisite](#1-prerequisite)
* [2 Prepare Calibration Dataset](#2-prepare-calibration-dataset)
  + [Default Dataset](#default-dataset)
  + [Customized Dataset](#customized-dataset)
  + [Dataset operations](#dataset-operations)
* [3 Quantization](#3-quantization)
  + [Supported Quantization Configurations](#supported-quantization-configurations)
  + [Hardware Compatibility](#hardware-compatibility)
  + [Supported Export Formats](#supported-export-formats)
  + [Command Line Usage](#command-line-usage)
  + [API usage](#api-usage)
    - [AutoRound API Usage](#autoround-api-usage)
    - [Mixed bits Usage](#mixed-bits-usage)
    - [AutoRoundBest recipe](#autoroundbest-recipe)
    - [AutoRoundLight recipe](#autoroundlight-recipe)
    - [Recipe recommendation](#recipe-recommendation)
  + [RTN mode](#rtn-mode)
  + [GGUF format](#gguf-format)
  + [Quantization Costs](#quantization-costs)
  + [Device/Multi-GPU setting in Quantization](#device-multi-gpu-setting-in-quantization)
    - [Enable multiple gpus calibration in lm_head quantization](#enable-multiple-gpus-calibration-in-lm-head-quantization)
    - [Enable multiple gpus tuning for extremely large model](#enable-multiple-gpus-tuning-for-extremely-large-model)
  + [Adjust Hyperparameters](#adjust-hyperparameters)
* [4 Inference](#4-inference)
  + [CPU](#cpu)
  + [Intel GPU](#intel-gpu)
  + [CUDA](#cuda)
  + [HPU](#hpu)
  + [Specify Inference Backend](#specify-inference-backend)
  + [Convert GPTQ/AWQ to AutoRound](#convert-gptq-awq-to-autoround)
* [5 Evaluation](#5-evaluation)
  + [Combine evaluation with tuning](#combine-evaluation-with-tuning)
  + [Eval the Quantized model](#eval-the-quantized-model)
* [6 Known Issues](#6-known-issues)

## 1 Prerequisite

Install auto-round or install from source

```bash
pip install auto-round
```

## 2 Prepare Calibration Dataset

### Default Dataset

The [NeelNanda/pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k) in huggingface is adopted as the default
calibration data and will be downloaded automatically from the datasets Hub. Other available datasets include:
- `swift/pile-val-backup` from modelscope for addressing HF network issue
- `BAAI/CCI3-HQ` for Chinese
- `codeparrot/github-code-clean` for code
- `HuggingFaceH4/ultrachat_200k` for chat data
- `madao33/new-title-chinese` for Chinese
- `mbpp` for code
- `openbmb/Ultra-FineWeb`

### Customized Dataset

- Option 1: Pass a local json file path to dataset argument
- Option 2: Register your dataset following the [code](../auto_round/calib_dataset.py) and pass the new dataset and
  split args to initialize AutoRound object, e.g. autoround=Autoround(dataset="NeelNanda/pile-10k:train", ...)
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

### Dataset operations

**Dataset combination**:We support combination of different datasets and parametrization of calibration datasets by
using `--dataset ./tmp.json,NeelNanda/pile-10k:num=256,mbpp:num=128`. Both local calibration file
and huggingface dataset are supported. You could specify splits of a dataset by setting `split=split1+split2`.

**Samples concatenation**: An optional setting allows users to concatenate calibration samples
using `--dataset NeelNanda/pile-10k:concat=True`.
All samples will be concatenated first, then split into chunks of seqlen length.

**Apply chat template**: Using `--dataset NeelNanda/pile-10k:apply_chat_template` enables application of a chat template
to the calibration data before tokenization. This is commonly used for instruct-style models during generation. To
customize the system prompt,
use:`--dataset 'NeelNanda/pile-10k:apply_chat_template:system_prompt="You are a helpful assistant."'`

Note: If the concatenation option is not enabled, samples shorter than args.seqlen will be dropped.

Please use ',' to split datasets, ':' to split parameters of a dataset and '+' to add values for one targeted parameter.

## 3 Quantization

### Supported Quantization Schemes

AutoRound supports several Schemes:

- **W4A16**(bits:4,group_size:128,sym:True,act_bits:16)
- **W8A16**(bits:8,group_size:128,sym:True,act_bits:16)
- **W3A16**(bits:3,group_size:128,sym:True,act_bits:16)
- **W2A16**(bits:2,group_size:128,sym:True,act_bits:16)
- **Mixed bits Weight only**
- **NVFP4**(data_type:nvfp4,act_data_type:nvfp4,static_global_scale,group_size 16)
- **MXFP4**(**Research feature,no real kernel**, data_type:mxfp4,act_data_type:mxfp4,rceil,group_size 32)
- **FPW8A16**(**Research feature,no real kernel**, data_type:fp8,act_data_type 16:,group_size 0->per tensor )
- **FP8_STATIC**(**Research feature,no real kernel**, data_type:fp8,act_data_type:fp8,group_size -1 ->per channel, act_group_size=0->per tenosr)

Besides, you could modify the `group_size`, `bits`, `sym` and many other configs you want, though there are maybe no real kernels.

### Supported export Formats

**AutoRound Format**: This format is well-suited for CPU, Intel GPU, CUDA and HPU devices, 2 bits, as well as mixed-precision
inference. **[2,3,4,8] bits are supported**.

**GGUF** Format: Experimental feature. This format is well-suited for CPU devices and is widely adopted by the
community. `q*_k`,`q*_0`,`q*_1` are supported.

**AutoGPTQ Format**: This format is well-suited for symmetric quantization on CUDA devices and is widely adopted by the
community, **[2,3,4,8] bits are supported**. However, **the
asymmetric kernel has issues** that can cause considerable accuracy drops, particularly at 2-bit quantization and small
models. Besides, recently 3 bits may have some accuracy issues in Transformers.

**AutoAWQ Format**: This format is well-suited for asymmetric 4-bit quantization on CUDA devices and is widely
adopted within the community, **only 4-bits quantization is supported**.

**LLM-Compressor Format**:** NVFP4, MXFP(Kernel is WIP), INT8 are supported**.

### Hardware Compatibility

CPU, Intel GPU, HPU and CUDA for both quantization and inference.

### Command Line Usage


- **AutoRound recipe:**

   This setting offers a better trade-off between accuracy and tuning cost, and is recommended in all scenarios.

    ```bash
    auto-round --model facebook/opt-125m  --scheme "W4A16"  --format "auto_gptq,auto_awq,auto_round"
    ```

- **Best Settings:**

  This setting provides the best accuracy in most scenarios but is 4–5× slower than the standard AutoRound recipe. It is especially recommended for 2-bit quantization and is a good choice if sufficient resources are available.
  
  ```bash
  auto-round-best --model facebook/opt-125m  --scheme "W4A16"  --format "auto_gptq,auto_awq,auto_round"
    ```

- **Light Settings:**

    This setting offers the best speed (2-3X faster than AutoRound), but it may cause a significant accuracy drop for small models and 2-bit quantization. It is recommended for 4-bit settings and models larger than 3B
    
    ```bash
    auto-round-light --model facebook/opt-125m  --scheme "W4A16"  --format "auto_gptq,auto_awq,auto_round"
    ```

### API usage
#### AutoRound API Usage
This setting offers a better trade-off between accuracy and tuning cost, and is recommended in all scenarios.

```python
from auto_round import AutoRound

model_name_or_path = "facebook/opt-125m"
ar = AutoRound(
    model_name_or_path,
    scheme="W4A16",
    # enable_torch_compile=True,
)

output_dir = "./tmp_autoround"
# format= 'auto_round'(default), 'auto_gptq', 'auto_awq'
ar.quantize_and_save(output_dir, format="auto_gptq,auto_awq,auto_round")
```

#### Mixed bits Usage
Auto-GPTQ and Auto-AWQ only support a limited set of mixed-bit configurations. If you're unsure about the details, we recommend using the AutoRound format.

vLLM and SGLang fuse MoE and QKV layers, so it's recommended not to assign different bit widths to these layers.

```python
from auto_round import AutoRound

model_name_or_path = "facebook/opt-125m"

layer_config = {  #  Supports both full layer names and fuzzy (partial) matching
    "model.decoder.layers.6.self_attn.out_proj": {"bits": 8, "group_size": 32},
    "model.decoder.layers.*k_proj": {"bits": 2, "group_size": 32},
}
ar = AutoRound(
    model_name_or_path,
    layer_config=layer_config,
)

output_dir = "./tmp_autoround"
ar.quantize_and_save(output_dir, format="auto_round")
```

#### AutoRoundBest recipe
This setting provides the best accuracy in most scenarios but is 4–5× slower than the standard AutoRound recipe. It is especially recommended for 2-bit quantization and is a good choice if sufficient resources are available.
```python
from auto_round import AutoRound

model_name_or_path = "facebook/opt-125m"
ar = AutoRound(model=model_name_or_path, scheme="W4A16", nsamples=512, iters=1000, low_gpu_mem_usage=True)

output_dir = "./tmp_autoround"
ar.quantize_and_save(output_dir, format="auto_round")
```
#### AutoRoundLight recipe
This setting offers the best speed (2 - 3X faster than AutoRound), but it may cause a significant accuracy drop for small models and 2-bit quantization. It is recommended for 4-bit settings and models larger than 3B.

```python
from auto_round import AutoRound

model_name_or_path = "facebook/opt-125m"

ar = AutoRound(
    model=model_name_or_path,
    scheme="W4A16",
    iters=50,
    lr=5e-3,
)

output_dir = "./tmp_autoround"
ar.quantize_and_save(output_dir, format="auto_round")
```
#### Recipe recommendation

In conclusion, we recommend using **auto-round for W4A16 and auto-round-best for W2A16**. However, you may adjust the
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

### RTN mode
AutoRound also supports RTN (Round-To-Nearest) mode for fast, calibration-free baseline quantization. try setting `iters=0` and use `group_size=32` for better results.

For the GGUF format, we have optimized the RTN algorithm inspired by llamacpp. To use the original (pure) RTN algorithm instead, enable the `--disable_opt_rtn` option.
```python
from auto_round import AutoRound

model_name_or_path = "facebook/opt-125m"
ar = AutoRound(
    model=model_name_or_path,
    scheme="W4A16",
    iters=0,
)

output_dir = "./tmp_autoround"
ar.quantize_and_save(output_dir, format="auto_round")
```

### GGUF format
Experimental feature. This format is well-suited for CPU devices and is widely adopted by the community. 
This format is well-suited for CPU devices and is widely adopted by the community.
```python
from auto_round import AutoRound

model_name_or_path = "facebook/opt-125m"
ar = AutoRound(
    model=model_name_or_path,
)
output_dir = "./tmp_autoround"
ar.quantize_and_save(output_dir, format="gguf:q4_k_m")  #  gguf:q*_k_s,gguf:q*_k_0,gguf:q*_k_1,
```


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




### Device/Multi-GPU setting in Quantization
**The tuning device is specified using the `device_map` argument in AutoRound API, _not_ through the `device_map` 
parameter used by Transformers.from_pretrained.**

AutoRound tunes the model in a block-by-block manner. Although the block size is much smaller than the model size, it still requires a significant amount of GPU memory for tuning—typically 10 times the block size. This can lead to out-of-memory (OOM) errors when working with extremely large models.

For strategies to reduce GPU memory usage, please refer to the [Reduced GPU Memory Usage](###Adjust Hyperparameters)
section below, where you  can adjust hyperparameters to optimize memory consumption.

If adjusting hyperparameters does not resolve the issue a, a simple solution is just adding more devices in device_map, for example, 
~~~python
from auto_round import AutoRound

model_name_or_path = "facebook/opt-125m"
ar = AutoRound(
    model=model_name_or_path,
    device_map="0,1,2,3"
)
~~~

or

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3 auto-round --model "facebook/opt-125m" --scheme "W4A16" --device_map "auto"
~~~


There are typically two scenarios that require multi-GPU tuning: one is the calibration phase mainly for lm-head quantization, and the other is quantizing extremely large models (e.g., models larger than 100 GB).

#### Enable multiple gpus calibration in lm_head quantization
For LM head tuning, AutoRound needs to cache the inputs to the lm-head, which requires the entire model to reside on 
  the GPU for efficient calibration. If there is no enough VRAM, some layers will fallback to RTN mode

#### Manually set the device_map

<details>
<summary>Customized device map</summary>
If device_map=auto does not correctly map the model, we also support mapping different layers within a block to 
different devices by setting the `device_map` argument in the AutoRound API. For reference, we provide an example of 
quantizing the DeepSeekV3-BF16 (1.4T) model using five 80GB GPUs.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "opensourcerelease/DeepSeek-R1-bf16"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype="auto")

block = model.model.layers
device_map = {}

for n, m in block.named_modules():
    if type(m) == torch.nn.Linear:
        if "experts" in n and ("shared_experts" not in n) and int(n.split(".")[-2]) < 63:
            device = "cuda:1"
        elif (
            "experts" in n
            and ("shared_experts" not in n)
            and int(n.split(".")[-2]) >= 63
            and int(n.split(".")[-2]) < 128
        ):
            device = "cuda:2"
        elif (
            "experts" in n
            and ("shared_experts" not in n)
            and int(n.split(".")[-2]) >= 128
            and int(n.split(".")[-2]) < 192
        ):
            device = "cuda:3"
        elif "experts" in n and ("shared_experts" not in n) and int(n.split(".")[-2]) >= 192:
            device = "cuda:4"
        else:
            device = "cuda:0"
        n = n[2:]

        device_map.update({n: device})

from auto_round import AutoRound

autoround = AutoRound(
    model=model,
    tokenizer=tokenizer,
    device_map=device_map,
    nsamples=512,
    batch_size=4,
    low_gpu_mem_usage=True,
    seqlen=2048,
)
autoround.quantize()
autoround.save_quantized(format="auto_awq", output_dir="tmp_autoround")
```
</details> 
  

### Adjust Hyperparameters

- **Reduced GPU Memory Usage:**
    
    - set `enable_torch_compile` to True

    - enable `low_gpu_mem_usage`(more tuning cost)

    - set `--bs 1 --gradient_accumulate_steps 8` (more tuning cost)

    - reduce the `bs` to 4(potential accuracy drop)

    - reduce the `seqlen` to 512 (potential accuracy drop)

    - or combine them


- **Reduced CPU Memory Usage :**

    - Trigger immediate packing: Packing will be triggered immediately when using the command-line interface or the
      quantize_and_save API, as long as only one export format is specified.

    - (only available for .bin file currently) set "--low_cpu_mem_mode 1" to use block-wise mode, load the weights from
      disk of each block when tuning and
      release the memory of the block after tuning. (more tuning cost)

    - (only available for .bin file currently) set "--low_cpu_mem_mode 2" to use layer-wise mode, load the weights of
      each layer from disk when tuning, minimum
      memory consumption and also the slowest running speed.


- **Speedup the tuning:**
    - set `enable_torch_compile` to True

    - use `auto-round-light` configuration

    - reduce the seqlen to 512(potential large accuracy drop for some scenarios)

    - reduce the train bs to 4(little accuracy drop. )

    - or combine them


- **Enable quantized lm-head:**

  Currently only support in AutoRound format inference for this config

    ```bash
    auto-round --model_name facebook/opt-125m  --scheme "W4A16" --quant_lm_head --format "auto_round"
    ```


- **Utilize the AdamW Optimizer:**

  Include the flag `--adam`. Note that AdamW is less effective than sign gradient descent in many scenarios we tested.


## 4 Inference

AutoRound automatically selects the best available backend based on the installed libraries and prompts the user to install additional libraries when a better backend is found.

**Please avoid manually moving the quantized model to a different device** (e.g., model.to('cpu')) during inference, as this may cause unexpected exceptions.

###  CPU

Supports 2, 4, and 8 bits. We recommend using intel-extension-for-pytorch (IPEX) for 4 bits inference.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "OPEA/Qwen2.5-1.5B-Instruct-int4-sym-inc"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
```


### Intel GPU

Supports 4 bits only. We recommend using intel-extension-for-pytorch (IPEX) for inference.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "OPEA/Qwen2.5-1.5B-Instruct-int4-sym-inc"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="xpu", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
```

### CUDA

Supports 2, 3, 4, and 8 bits. We recommend using GPTQModel for 4 and 8 bits inference.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "OPEA/Qwen2.5-1.5B-Instruct-int4-sym-inc"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
```


### HPU
docker image with Gaudi Software Stack is recommended. More details can be found
in [Gaudi Guide](https://docs.habana.ai/en/latest/).

```python
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Intel/Qwen2-7B-int4-inc"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("hpu").to(torch.bfloat16)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
```


### Specify Inference Backend

AutoRound automatically selects the  backend for each layer based on compatibility. In general, the priority order is Marlin > ExLLaMAV2 > Triton, but the final choice depends on factors such as group size, bit width, packing format, hardware device, and other implementation details.

The backend may not always be the most suitable for certain devices. 
You can specify your preferred backend such as "ipex" for CPU and Intel GPU, "marlin/exllamav2/triton" for CUDA, according to your needs or hardware compatibility. Please note that additional corresponding libraries may be required.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoRoundConfig

model_name = "OPEA/Qwen2.5-1.5B-Instruct-int4-sym-inc"
quantization_config = AutoRoundConfig(backend="ipex")
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="cpu", quantization_config=quantization_config, torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
```
| Name                                 | Devices  | Bits    | Dtypes    | Priority | Packing format  | Requirements                  |
|--------------------------------------|----------|---------|-----------|----------|-----------------|-------------------------------|
| ipex                                 | cpu/xpu  | 4       | BF16/FP16 | 5        | gptq_zp+-1/awq  | intel-extension-for-pytorch   |
| itrex                                | cpu      | 2,4,8   | BF16/FP16 | 1        | gptq_zp+-1/awq  | <br/>intel-extension-for-transformers |
| marlin                               | cuda     | 4,8     | BF16/FP16 | 6        | gptq/gptq_zp+-1 | gptqmodel                     |
| exllamav2 or<br/>gptqmodel:exllamav2 | cuda     | 4       | BF16/FP16 | 5        | gptq            | gptqmodel                     |
| exllamav2 or<br/>gptq:exllamav2      | cuda     | 4       | FP16      | 5        | gptq_zp+-1      | auto-gptq                     |
| gptq:cuda                            | cuda     | 2,3,4,8 | FP16      | 1        | gptq_zp+-1      | auto-gptq     <br/>                |
| triton                               | xpu/cuda | 2,4,8   | BF16/FP16 | 2        | gptq/gptq_zp+-1 | <br/>auto-round                    |
| awq                                  | cuda     | 4       | FP16      | 5        | awq             | auto-awq                      |
| hpu                                  | hpu      | 4       | BF16      | 0        | gptq/gptq_zp+-1 | auto-round                    |
| torch                                | xpu/cpu/cuda | 2,3,4,8 | BF16/FP16 | 0        | gptq/gptq_zp+-1 | auto-round                    |


### Convert GPTQ/AWQ to AutoRound

Most GPTQ/AWQ models can be converted to the AutoRound format for better compatibility and support with Intel devices. Please note that the quantization config will be changed if the model is serialized.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoRoundConfig

model_name = "ybelkada/opt-125m-gptq-4bit"
quantization_config = AutoRoundConfig()
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="cpu", quantization_config=quantization_config, torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
```


## 5 Evaluation

### Combine evaluation with tuning

- We leverage lm-eval-harnessing for the evaluation. 
If not explicitly specify '--task', the default value will be used (typically covering 10+ common tasks).
  ~~~bash
   auto-round --model facebook/opt-125m  --bits 4 --format "auto_round,auto_gptq" --tasks mmlu
  ~~~
  The last format will be used in evaluation if multiple formats have been exported.

Note: To use the vllm backend, please add `--vllm` into the upper command.

###  Eval the Quantized model

- AutoRound format
  For lm-eval-harness, you could just call
  ~~~bash
  auto-round --model="your_model_path" --eval  --tasks lambada_openai --eval_bs 16
  ~~~
  Multiple gpu evaluation
  ~~~bash
  auto-round --model="your_model_path" --eval  --device 0,1 --tasks lambada_openai --eval_bs 16
  ~~~
  For other evaluation framework, if the framework could support Huggingface models, typically it could support
  AutoRound format, only you need to do is import the following in the beginning of your code
  ~~~python
  from auto_round import AutoRoundConfig
  ~~~  

- AutoGPTQ/AutoAWQ format

  Please refer to their repo and check the evaluation framework's compatibility.
  For lm-eval-harness, you could just call
  ~~~bash
  lm_eval --model hf --model_args pretrained="your_model_path" --device cuda:0 --tasks lambada_openai --batch_size 16
  ~~~
  Multiple gpu evaluation
  ~~~bash
  CUDA_VISIBLE_DEVICES=0,1 lm_eval --model hf --model_args pretrained="your_model_path",parallelize=True --tasks lambada_openai --batch_size 16
  ~~~


## 6 Known Issues

* Random quantization results in tuning some models
* ChatGlm-V1 is not supported


