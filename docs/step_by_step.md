Step-by-Step
============

This document presents step-by-step instructions for auto-round llm quantization.

# 1 Prerequisite

Install auto-round or install from source

```bash
pip install auto-round
```

## 2. Prepare Calibration Dataset

### Default Dataset

The [NeelNanda/pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k) in huggingface is adopted as the default
calibration data and will be downloaded automatically from the datasets Hub. Other available datasets include:

- `swift/pile-val-backup` for addressing HF network issue
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

## 3. Quantization

### Supported Quantization Configurations

AutoRound supports several quantization configurations:

- **Int8 Weight Only**
- **Int4 Weight Only**
- **Int3 Weight Only**
- **Int2 Weight Only**
- **Mixed bits Weight only**

### Hardware Compatibility

CPU, XPU, HPU,and CUDA for both quantization and inference.

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

    This setting offers the best speed (2 - 3X faster than AutoRound), but it may cause a significant accuracy drop for small models and 2-bit quantization. It is recommended for 4-bit settings and models larger than 3B
    
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
# mixed bits config
# layer_config = {"model.decoder.layers.6.self_attn.out_proj": {"bits": 2, "group_size": 32}}
autoround = AutoRound(
    model,
    tokenizer,
    bits=bits,
    group_size=group_size,
    sym=sym,
    # enable_torch_compile=True,
    # layer_config=layer_config,
)

output_dir = "./tmp_autoround"
# format= 'auto_round'(default), 'auto_gptq', 'auto_awq'
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

### GGUF format

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

## 4. Evaluation

### 4.1 Combine evaluation with tuning

- We leverage lm-eval-harnessing for the evaluation
  ~~~bash
   auto-round --model facebook/opt-125m  --bits 4 --format "auto_round,auto_gptq" --tasks mmlu
  ~~~
  The last format will be used in evaluation if multiple formats have been exported.

### 4.2  Eval the Quantized model

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

## Inference

### AutoRound format

**CPU**: **auto_round version >0.3.1**, pip install intel-extension-for-pytorch(much higher speed on Intel CPU) or pip
install intel-extension-for-transformers,

**HPU**: docker image with Gaudi Software Stack is recommended. More details can be found
in [Gaudi Guide](https://docs.habana.ai/en/latest/).

**CUDA**: no extra operations for sym quantization, for asym quantization, need to install auto-round from source

- The following code will automatically detect device, and typically some error message will remind you to install some
  extra libraries
  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer
  from auto_round import AutoRoundConfig ## must import
  
  quantized_model_path = "./tmp_autoround"
  device="cuda"
  model = AutoModelForCausalLM.from_pretrained(quantized_model_path).to(device)
  tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
  text = "There is a girl who likes adventure,"
  inputs = tokenizer(text, return_tensors="pt").to(model.device)
  print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
  ```
- To specify device use a different backend

  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer
  from auto_round import AutoRoundConfig
  
  backend = "auto"  ##cpu, hpu, cuda
  quantization_config = AutoRoundConfig(
      backend=backend
  )
  quantized_model_path = "./tmp_autoround"
  model = AutoModelForCausalLM.from_pretrained(quantized_model_path,
                                               device_map=backend.split(':')[0], quantization_config=quantization_config)
  tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
  text = "There is a girl who likes adventure,"
  inputs = tokenizer(text, return_tensors="pt").to(model.device)
  print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
  ```

## 6. Known Issues

* Random quantization results in tuning some models
* ChatGlm-V1 is not supported