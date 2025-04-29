Step-by-Step
============

This document presents step-by-step instructions for auto-round llm quantization.

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
- `madao33/new-title-chinese` for Chinese
- `mbpp` for code

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

### Supported Quantization Configurations

AutoRound supports several quantization configurations:

- **Int8 Weight Only**
- **Int4 Weight Only**
- **Int3 Weight Only**
- **Int2 Weight Only**
- **Mixed bits Weight only**

### Hardware Compatibility

CPU, Intel GPU, HPU,and CUDA for both quantization and inference.

### Command Line Usage


- **AutoRound recipe:**

   This setting offers a better trade-off between accuracy and tuning cost, and is recommended in all scenarios.

    ```bash
    auto-round --model facebook/opt-125m  --bits 4 --group_size 128  --format "auto_gptq,auto_awq,auto_round"
    ```

- **Best Settings:**

  This setting provides the best accuracy in most scenarios but is 4–5× slower than the standard AutoRound recipe. It is especially recommended for 2-bit quantization and is a good choice if sufficient resources are available.
  
- ```bash
    auto-round-best --model facebook/opt-125m  --bits 4 --group_size 128  --format "auto_gptq,auto_awq,auto_round"
    ```

- **Light Settings:**

    This setting offers the best speed (2-3X faster than AutoRound), but it may cause a significant accuracy drop for small models and 2-bit quantization. It is recommended for 4-bit settings and models larger than 3B
    
    ```bash
    auto-round-light --model facebook/opt-125m  --bits 4  --group_size 128 --format "auto_gptq,auto_awq,auto_round"
    ```

### API usage
#### AutoRound API Usage
This setting offers a better trade-off between accuracy and tuning cost, and is recommended in all scenarios.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
bits, group_size, sym = 4, 128, True
autoround = AutoRound(
    model,
    tokenizer,
    bits=bits,
    group_size=group_size,
    sym=sym,
    # enable_torch_compile=True,
)

output_dir = "./tmp_autoround"
# format= 'auto_round'(default), 'auto_gptq', 'auto_awq'
autoround.quantize_and_save(output_dir, format='auto_gptq,auto_awq,auto_round') 
```

#### Mixed bits Usage
Auto-GPTQ and Auto-AWQ only support a limited set of mixed-bit configurations. If you're unsure about the details, we recommend using the AutoRound format.

Also, avoid setting mixed bits to 3 for asymmetric quantization at this time, as models exported with this setting may not be compatible with future versions of the AutoRound format.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
bits, group_size, sym = 4, 128, True
layer_config = {#  Supports both full layer names and fuzzy (partial) matching
  "model.decoder.layers.6.self_attn.out_proj": {"bits": 8, "group_size": 32}, 
  "model.decoder.layers.*k_proj": {"bits": 2, "group_size": 32}
  }
autoround = AutoRound(
    model,
    tokenizer,
    bits=bits,
    group_size=group_size,
    sym=sym,
    layer_config=layer_config,
)

output_dir = "./tmp_autoround"
autoround.quantize_and_save(output_dir, format='auto_round') 
```

#### AutoRoundBest recipe
This setting provides the best accuracy in most scenarios but is 4–5× slower than the standard AutoRound recipe. It is especially recommended for 2-bit quantization and is a good choice if sufficient resources are available.
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
bits, group_size, sym = 4, 128, True
autoround = AutoRound(
    model,
    tokenizer,
    bits=bits,
    group_size=group_size,
    sym=sym,
    nsamples=512,
    iters=1000,
    low_gpu_mem_usage=True
)

output_dir = "./tmp_autoround"
autoround.quantize_and_save(output_dir, format='auto_round') 
```
#### AutoRoundLight recipe
This setting offers the best speed (2 - 3X faster than AutoRound), but it may cause a significant accuracy drop for small models and 2-bit quantization. It is recommended for 4-bit settings and models larger than 3B.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
bits, group_size, sym = 4, 128, True
autoround = AutoRound(
    model,
    tokenizer,
    bits=bits,
    group_size=group_size,
    sym=sym,
    iters=50,
    lr=5e-3,
)

output_dir = "./tmp_autoround"
autoround.quantize_and_save(output_dir, format='auto_round') 
```

### RTN mode
AutoRound also supports RTN (Round-To-Nearest) mode for fast, calibration-free baseline quantization. try setting iters=0 and use group_size=32 for better results.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
bits, group_size, sym = 4, 32, True
autoround = AutoRound(
    model,
    tokenizer,
    bits=bits,
    group_size=group_size,
    sym=sym,
    iters=0,
)

output_dir = "./tmp_autoround"
autoround.quantize_and_save(output_dir, format='auto_round') 
```

### GGUF format

This format is well-suited for CPU devices and is widely adopted by the community, **only q4_0 and q4_1 (W4G32) is supported in our repo**.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
bits, group_size, sym = 4, 32, True
autoround = AutoRound(
    model,
    tokenizer,
    bits=bits,
    group_size=group_size,
    sym=sym,
    iters=50,
    lr=5e-3,
)

output_dir = "./tmp_autoround"
autoround.quantize_and_save(output_dir, format='gguf:q4_0') # gguf:q4_1
```

### Adjust Hyperparameters

- **Reduced GPU Memory Usage:**

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
    - use `auto-round-light` configuration

    - reduce the seqlen to 512(potential large accuracy drop for some scenarios)

    - reduce the train bs to 4(little accuracy drop. )

    - or combine them


- **Enable quantized lm-head:**

  Currently only support in AutoRound format inference for this config

    ```bash
    auto-round --model_name facebook/opt-125m  --bits 4 --group_size 128 --quant_lm_head --format "auto_round"
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
from auto_round import AutoRoundConfig 

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
from auto_round import AutoRoundConfig 

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
from auto_round import AutoRoundConfig 

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
from transformers import AutoModelForCausalLM,AutoTokenizer
from auto_round import AutoRoundConfig

model_name = "Intel/Qwen2-7B-int4-inc"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to('hpu').to(bfloat16)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
```


### Specify Inference Backend

AutoRound automatically selects the  backend for each layer based on compatibility. In general, the priority order is Marlin > ExLLaMAV2 > Triton, but the final choice depends on factors such as group size, bit width, packing format, hardware device, and other implementation details.

The backend may not always be the most suitable for certain devices. 
You can specify your preferred backend such as "ipex" for CPU and Intel GPU, "marlin/exllamav2/triton" for CUDA, according to your needs or hardware compatibility. Please note that additional corresponding libraries may be required.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRoundConfig

model_name = "OPEA/Qwen2.5-1.5B-Instruct-int4-sym-inc"
quantization_config = AutoRoundConfig(backend="ipex")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", quantization_config=quantization_config, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
```
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


### Convert GPTQ/AWQ to AutoRound

Most GPTQ/AWQ models can be converted to the AutoRound format for better compatibility and support with Intel devices. Please note that the quantization config will be changed if the model is serialized.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRoundConfig

model_name = "ybelkada/opt-125m-gptq-4bit"
quantization_config = AutoRoundConfig()
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", quantization_config=quantization_config, torch_dtype="auto")
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
