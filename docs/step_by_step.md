Step-by-Step
============

English | [简体中文](./step_by_step_CN.md)

This document presents step-by-step instructions for auto-round llm quantization. You can refer to [vlms user guide](../auto_round/compressors/mllm/README.md) for vlms quantization and [diffusions user guide](../auto_round/compressors/diffusion/README.md) for diffusions quantization.

* [1 Prerequisite](#1-prerequisite)
* [2 Prepare Calibration Dataset](#2-prepare-calibration-dataset)
  + [Default Dataset](#default-dataset)
  + [Customized Dataset](#customized-dataset)
  + [Dataset operations](#dataset-operations)
* [3 Quantization](#3-quantization)
  + [Supported Quantization Schemes](#supported-quantization-schemes)
  + [Supported Export Formats](#supported-export-formats)
  + [Hardware Compatibility](#hardware-compatibility)
  + [Environment Configuration](#environment-configuration)
  + [Command Line Usage](#command-line-usage)
  + [API usage](#api-usage)
    - [AutoRound API Usage](#autoround-api-usage)
    - [Mixed bits Usage](#mixed-bits-usage)
    - [AutoRoundBest recipe](#autoroundbest-recipe)
    - [AutoRoundLight recipe](#autoroundlight-recipe)
    - [Recipe recommendation](#recipe-recommendation)
  + [AutoScheme](#autoscheme)
    - [CLI Usage](#cli-usage)
    - [API Usage](#api-usage-1)
    - [Hyperparameters in AutoScheme](#hyperparameters-in-autoscheme)
  + [OPT RTN mode](#opt-rtn-mode)
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
  + [Single GPU Evaluation](#single-gpu-evaluation)
  + [Multi-GPU Evaluation](#multi-gpu-evaluation)
  + [Important Notes](#important-notes)
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
        # Important Notice!!! AutoRound will drop data < args.seqlen and truncate data to args.seqlen
        data = ["AutoRound is an advanced quantization algorithm for low-bits LLM inference" * 240]
        return data
    
    
    def customized_data_with_tokenizer(tokenizer, seqlen=2048):
        # Import notice!!! AutoRound will drop data < args.seqlen
        data = ["AutoRound is an advanced quantization algorithm for low-bits LLM inference" * 240]
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
- **GGUF:Q4_K_M**(all Q*_K,Q*_0,Q*_1 provided by llamacpp are supported)
- **Mixed Bits Weight only**
- **NVFP4**(Experimental feature, recommend exporting to `llm_compressor` format.data_type nvfp4,act_data_type nvfp4,static_global_scale,group_size 16)
- **MXFP4**(**Research feature, no real kernel**, Standard MXFP4, data_type mxfp,act_data_type mxfp,bits 4, act_bits 4, group_size 32)
- **MXFP4_RCEIL**(**Research feature,no real kernel**, NVIDIA's variant, data_type mxfp,act_data_type mxfp_rceil,bits 4, act_bits 4, group_size 32)
- **MXFP8**(**Research feature, no real kernel**, data_type mxfp,act_data_type mxfp_rceil,group_size 32)
- **FPW8A16**(**Research feature, no real kernel**, data_type fp8,group_size 0->per tensor )
- **FP8_STATIC**(**Research feature, no real kernel**, data_type:fp8,act_data_type:fp8,group_size -1 ->per channel, act_group_size=0->per tensor)

Besides, you could modify the `group_size`, `bits`, `sym` and many other configs you want, though there are maybe no real kernels.

### Supported Export Formats
You can use command `auto_round list format` to show all supported formats with support scheme.

**AutoRound Format**: This format is well-suited for CPU, Intel GPU, CUDA and HPU devices, 2 bits, as well as mixed-precision
inference. **[2,3,4,8] bits are supported**. Please set `--format auto_round`

**GGUF** Format: Experimental feature. This format is well-suited for CPU devices and is widely adopted by the
community. `q*_k`,`q*_0`,`q*_1` are supported. Please set `--format gguf:q4_k_m`,  `--format gguf:q2_k_s`, etc

**AutoGPTQ Format**: This format is well-suited for symmetric quantization on CUDA devices and is widely adopted by the
community, **[2,3,4,8] bits are supported**. However, **the
asymmetric kernel has issues** that can cause considerable accuracy drops, particularly at 2-bit quantization and small
models. Besides, recently 3 bits may have some accuracy issues in Transformers.  Please set `--format auto_gptq`

**AutoAWQ Format**: This format is well-suited for asymmetric 4-bit quantization on CUDA devices and is widely
adopted within the community, **only 4-bits quantization is supported**. Please set `--format auto_awq`

**LLM-Compressor Format**: **NVFP4, MXFP4(kernel in WIP), MXFP8 are supported**. Please set `--format llm_compressor`

#### Format and scheme support matrix
> Gray indicates the absence of a kernel or the presence of only an inefficient/reference kernel. BF16 is mainly for AutoScheme


| Format | Supported Schemes                                                                                                                                                       |
|:---|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **auto_round** | W4A16, W2A16, W3A16, W8A16, W2A16G64, W2A16G32, `MXFP4`, `MXFP8`, `MXFP4_RCEIL`, `MXFP8_RCEIL`, `NVFP4`, `FPW8A16`, `FP8_STATIC`, `BF16`                                |
| **auto_awq** | W4A16, BF16                                                                                                                                                             |
| **auto_gptq** | W4A16, W2A16, W3A16, W8A16,W2A16G64, W2A16G32, BF16                                                                                                                     |
| **llm_compressor** | NVFP4, `MXFP4`, `MXFP8`, `FPW8A16`, `FP8_STATIC`                                                                                                                        |
| **gguf** | GGUF:Q4_K_M, GGUF:Q2_K_S, GGUF:Q3_K_S, GGUF:Q3_K_M, GGUF:Q3_K_L, GGUF:Q4_K_S, GGUF:Q5_K_S, GGUF:Q5_K_M, GGUF:Q6_K, GGUF:Q4_0, GGUF:Q4_1, GGUF:Q5_0, GGUF:Q5_1,GGUF:Q8_0 |
| **fake** | `all schemes (only for research)`                                                                                                                                       |

### Hardware Compatibility

CPU, Intel GPU, HPU and CUDA for both quantization and inference.

### Environment Configuration

Before starting quantization, you may want to configure AutoRound's environment variables for optimal performance. For detailed information about available environment variables (logging levels, ModelScope integration, workspace settings, etc.), please refer to the [Environment Variables Guide](./environments.md).

### Command Line Usage


- **AutoRound recipe:**

   This setting offers a better trade-off between accuracy and tuning cost, and is recommended in all scenarios.

    ```bash
    auto-round --model Qwen/Qwen3-0.6B  --scheme "W4A16"  --format "auto_gptq,auto_awq,auto_round"
    ```

- **AutoRoundBest recipe:**

  This setting provides the best accuracy in most scenarios but is 4–5× slower than the standard AutoRound recipe. It is especially recommended for 2-bit quantization and is a good choice if sufficient resources are available.
  
  ```bash
  auto-round-best --model Qwen/Qwen3-0.6B  --scheme "W4A16"  --format "auto_gptq,auto_awq,auto_round"
    ```

- **AutoRoundLight Settings:**

    This setting offers the best speed (2-3X faster than AutoRound), but it may cause a significant accuracy drop for small models and 2-bit quantization. It is recommended for 4-bit settings and models larger than 3B
    
    ```bash
    auto-round-light --model Qwen/Qwen3-0.6B  --scheme "W4A16"  --format "auto_gptq,auto_awq,auto_round"
    ```

### API usage
#### AutoRound API Usage
This setting offers a better trade-off between accuracy and tuning cost, and is recommended in all scenarios.

```python
from auto_round import AutoRound

model_name_or_path = "Qwen/Qwen3-0.6B"
ar = AutoRound(
    model_name_or_path,
    scheme="W4A16",
    # enable_torch_compile=True,
)

output_dir = "./tmp_autoround"
# format= 'auto_round'(default), 'auto_gptq', 'auto_awq'
ar.quantize_and_save(output_dir, format="auto_gptq,auto_awq,auto_round")
```

#### Mixed Bits Usage
AutoRound(>0.8) offers auto-scheme to generate mixed bits recipe autocially, please refer to [AutoScheme](#autoscheme) section for more details.

Auto-GPTQ and Auto-AWQ only support a limited set of mixed-bit configurations. If you're unsure about the details, we recommend using the AutoRound format.

vLLM and SGLang fuse MoE and QKV layers, so it's recommended not to assign different bit widths to these layers.

```python
from auto_round import AutoRound

model_name_or_path = "Qwen/Qwen3-0.6B"

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

model_name_or_path = "Qwen/Qwen3-0.6B"
ar = AutoRound(model=model_name_or_path, scheme="W4A16", nsamples=512, iters=1000, low_gpu_mem_usage=True)

output_dir = "./tmp_autoround"
ar.quantize_and_save(output_dir, format="auto_round")
```
#### AutoRoundLight recipe
This setting offers the best speed (2 - 3X faster than AutoRound), but it may cause a significant accuracy drop for small models and 2-bit quantization. It is recommended for 4-bit settings and models larger than 3B.

```python
from auto_round import AutoRound

model_name_or_path = "Qwen/Qwen3-0.6B"

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

<details>
  <summary>Recipe Configuration Details</summary>

| Recipe  | batch_size | iters | seqlen | nsamples | lr    |
|---------|------------|-------|--------|----------|-------|
| default | 8          | 200   | 2048   | 128      | None  |
| best    | 8          | 1000  | 2048   | 512      | None  |
| light   | 8          | 50    | 2048   | 128      | 5e-3  |

</details>

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

### AutoScheme

AutoScheme provides an automatic algorithm to generate adaptive mixed bits/data-type quantization recipes.  For some accuracy result, please refer this doc [here](./auto_scheme_acc.md)

**Please note that mixed data types are supported during tuning, but cannot be exported to real models at this time..**

#### CLI Usage
use `iters=200`for tuning.
~~~bash
auto_round \
  --model_name  $model_name \
  --avg_bits 6 \
  --options "mxfp4,mxfp8" \
  --ignore_scale_zp_bits \
  --iters 0 \
  --format fake 
~~~

#### API Usage
~~~
avg_bits= 3.0
scheme = AutoScheme(avg_bits=avg_bits, options=("W2A16G64“, "W4A16","W8A16"))
ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1)
ar.quantize_and_save()
~~~

#### Hyperparameters in AutoScheme
`avg_bits(float)` Target average bits for the whole model; only layers to be quantized will be counted in the average bits calculation.

`options(Union[str, list[Union[QuantizationScheme, str]])` the options of quantization schemes to choose from. It could be a string like "W4A16", or a list of strings or QuantizationScheme objects.

`ignore_scale_zp_bits(bool)` Whether to ignore the bits of scale and zero point in average bits calculation. Default is False.

`device_map (Optional[str,dict,torch.device])`  only supported in API now, as auto-scheme used more VRAM than auto-round tuning, so you could set a different device_map for it.

`shared_layers (Optional[Iterable[Iterable[str]]])`  only supported in API now

`batch_size (Optional[int])` could be set to 1 to reduce VRAM but increase time cost

`low_gpu_mem_usage(bool=True)` whether to reduce gpu memory usage at the cost of more time cost

In some serving frameworks, certain layers (e.g., QKV or MoE) are fused to accelerate inference. These fused layers may require the same data type and bit configuration. The shared_layers option simplifies this setup by supporting both regex and full-name matching. **Note that regex matching is applied in a block-wise manner.**


```python
from auto_round import AutoRound, AutoScheme

shared_layers = [
    ["*.self_attn.k_proj", "v_proj", "q_proj", "out_proj"],
    ("model.decoder.layers.6.fc1", "model.decoder.layers.6.fc2"),
    ("fc1", "fc2"),
]
target_bits = 5.0
model_name = "Qwen/Qwen3-0.6B"
scheme = AutoScheme(avg_bits=target_bits, options=("W4A16", "MXFP8"), shared_layers=shared_layers)
ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1)
model, layer_config = ar.quantize()
```

Besides, if you want to fix the scheme for some layers, you could set it via `layer_config` in AutoRound API.
```python
from auto_round import AutoRound, AutoScheme

model_name = "Qwen/Qwen3-8B"
avg_bits = 3.0
scheme = AutoScheme(avg_bits=avg_bits, options=("GGUF:Q2_K_S", "GGUF:Q4_K_S"), ignore_scale_zp_bits=True)
layer_config = {"lm_head": "GGUF:Q6_K"}

ar = AutoRound(model=model_name, scheme=scheme, layer_config=layer_config, iters=0)
ar.quantize_and_save()
```

#### AutoScheme Cost

We tested it on Nvidia A100 80G using torch v2.8.

We will try to optimize the RAM usage in the future. The RAM usage is about 1.1-1.5x of the model's BF16 size

| Models        | Scheme                | VRAM Cost | Time Cost             |
| ------------- | --------------------- | --------- | --------------------- |
| Qwen3-8B      | W2A16 / W4A16 / W8A16 | 14G       | 60s * len of options  |
| Qwen3-8B      | MXFP4 / MXFP8         | 18G       | 60s * len of options  |
| Qwen3-8B      | GGUF*                 | 14G       | 80s * len of options  |
| Qwen3-32B     | W2A16 / W4A16 / W8A16 | 29G       | 180s*  len of options |
| Qwen3-32B     | MXFP4 / MXFP8         | 29G       | 180s*  len of options |
| Qwen3-32B     | GGUF*                 | 18G       | 300s * len of options |
| Llama-3.3-70B | W2A16 / W4A16 / W8A16 | 32G       | 420s * len of options |

<details>
<summary>Cost w/o low_gpu_mem_usage </summary>

| Models    | Scheme            | VRAM Cost <br />(torch compile) | Time Cost<br /> torch compile | VRAM Cost <br />wo torch compile | Time Cost<br /> wo torch compile |
| --------- | ----------------- | ------------------------------- | ----------------------------- | -------------------------------- | -------------------------------- |
| Qwen3-8B  | W2A16/W4A16/W8A16 | 34G                             | 30s * len of options          | 61G                              | 40s * len of options             |
| Qwen3-8B  | MXFP4/MXFP8       | 36G                             | 60s * len of options          | 54G                              | 120s * len of options            |
| Qwen3-8B  | GGUF*             | 54G                             | 30s * len of options          | 50G                              | 23S * len of options             |
| Qwen3-32B | W2A16/W4A16/W8A16 | OOM with 240G                   | ---                           | OOM with 240G                    | ---                              |
| Qwen3-32B | MXFP4/MXFP8       | 160G                            | 200s * len of options         | 200G                             | 240s * len of options            |
| Qwen3-32B | GGUF*             | 210G                            | 80s * len of options          | 200G                             | 60s * len of options             |
</details>


#### Limitations
Embedding layer is not supported in AutoScheme, it will use the best scheme in options.


### OPT RTN Mode
AutoRound also supports Optimized RTN (Round-To-Nearest) mode for fast, calibration-free baseline quantization. Setting `iters=0` tp enable it and we recommend using `group_size=32` for better results. Check [accuracy comparison](./opt_rtn.md) between RTN and OPT RTN mode

For the GGUF format, we have optimized the RTN algorithm inspired by llamacpp. To use the original (pure) RTN algorithm instead, enable the `--disable_opt_rtn` option.
```python
from auto_round import AutoRound

model_name_or_path = "Qwen/Qwen3-0.6B"
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

The optimized RTN mode is suggested (--iters 0) for all bits other than 3 bits.

```python
from auto_round import AutoRound

model_name_or_path = "Qwen/Qwen3-0.6B"
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

model_name_or_path = "Qwen/Qwen3-0.6B"
ar = AutoRound(
    model=model_name_or_path,
    device_map="0,1,2,3"
)
~~~

or

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3 auto-round --model "Qwen/Qwen3-0.6B" --scheme "W4A16" --device_map "auto"
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
    - Enable `low_cpu_mem_usage` (experimental): Only one export format is supported. The quantized model is saved immediately after each block is packed, reducing peak CPU memory usage.

    - Trigger immediate packing: Packing will be triggered immediately when using the command-line interface or the
      quantize_and_save API, as long as only one export format is specified.

- **Speedup the tuning:**
    - set `enable_torch_compile` to True

    - use `auto-round-light` configuration

    - reduce the seqlen to 512(potential large accuracy drop for some scenarios)

    - reduce the train bs to 4(little accuracy drop. )

    - or combine them


- **Enable quantized lm-head:**

  Currently only support in AutoRound format inference for this config

    ```bash
    auto-round --model_name Qwen/Qwen3-0.6B  --scheme "W4A16" --quant_lm_head --format "auto_round"
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

AutoRound leverages [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) for evaluation. If `--tasks` is not specified, a set of default tasks (typically 10+ common benchmarks) will be automatically used.

### Single GPU Evaluation

**HF Backend (default):**
```bash
auto-round --model Qwen/Qwen3-0.6B --bits 4 --format "auto_round,auto_gptq" --tasks mmlu
```

**vLLM Backend:**
```bash
auto-round --model Qwen/Qwen3-0.6B --bits 4 --format "auto_round,auto_gptq" --tasks mmlu --eval_backend vllm
```

### Multi-GPU Evaluation

**HF Backend:**
```bash
auto-round --model="your_model_path" --eval --device_map 0,1 --tasks lambada_openai --eval_bs 16
```

**vLLM Backend (Option 1 - using --device_map):**
```bash
auto-round "your_model_path" --eval --device_map 0,1 --tasks lambada_openai --eval_backend vllm
```

**vLLM Backend (Option 2 - manual configuration):**
```bash
CUDA_VISIBLE_DEVICES=0,1 auto-round "your_model_path" --eval --tasks lambada_openai --eval_backend vllm --vllm_args="tensor_parallel_size=2,gpu_memory_utilization=0.8"
```

### Important Notes

- Use the `--eval` flag to evaluate models directly. This supports both original and quantized models.
- The `--eval_task_by_task` option helps handle task failures by evaluating tasks sequentially. This only applies to the HF backend.
- When multiple formats are exported, the last format in the list will be used for evaluation.
- For vLLM backend, you can use `--device 0,1,2` to specify GPU devices. This will automatically set `CUDA_VISIBLE_DEVICES` and configure `tensor_parallel_size` based on the number of devices. Alternatively, you can manually set these via environment variables and `--vllm_args`.


## 6 Known Issues

Randomness in quantization may affect tuning results for some models, set `enable_deterministic_algorithms=True` to ensure reproducibility.


Some VLMs require manual support.


Mamba is not supported.


