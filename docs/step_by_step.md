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
calibration data and will be downloaded automatically from the datasets Hub. To customize a dataset, please kindly
follow our dataset code.
See more about loading [huggingface dataset](https://huggingface.co/docs/datasets/main/en/quickstart)

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
  **Dataset combination**:We support combination of different datasets and parametrization of calibration datasets by using "--dataset ./tmp.json:
  concat,NeelNanda/pile-10k:split=train+val:num=256,mbpp:concat=True:num=128:apply_chat_template". Both local calibration file
  and huggingface dataset are supported. Through parametrization, users could specify splits of a dataset by setting "
  split=split1+split2".
  
  **Samples concatenation**: A concatenation option could enable users to merge calibration samples. '--dataset NeelNanda/pile-10k:concat=True'
  
  **Apply chat template**: '--dataset NeelNanda/pile-10k:apply_chat_template' would enable users to apply chat_template to calibration
  data before tokenization and is widely used by instruct-models in generation. Please note that samples shorter than
  args.seqlen will be dropped when concatenation option is not enabled.
  
  Please use ',' to split datasets, ':' to split parameters of a dataset and '+' to add values for one targeted parameter.
  

<br />

## 2. Run Quantization

- **Default Settings:**

    ```bash
    auto-round --model facebook/opt-125m  --bits 4 --format "auto_round,auto_gptq" --disble_eval
    ```

- **Reduced GPU Memory Usage:**

    - enable low_gpu_mem_usage(more tuning cost)

    - set "--train_bs 1 --gradient_accumulate_steps 8" (more tuning cost)

    - reduce the train bs to 4(potential accuracy drop)

    - reduce the seqlen to 512 (potential accuracy drop)

    - or combine them


- **Reduced CPU Memory Usage (only available for .bin file currently):**

    - set "--low_cpu_mem_mode 1" to use block-wise mode, load the weights from disk of each block when tuning and
      release the memory of the block after tuning. (more tuning cost)

    - set "--low_cpu_mem_mode 2" to use layer-wise mode, load the weights of each layer from disk when tuning, minimum
      memory consumption and also slowest running speed.


- **Speedup the tuning:**
    - reduce the seqlen to 512(potential large accuracy drop for some scenarios)

    - reduce the train bs to 4(little accuracy drop. )

    - or combine them


- **Enable quantized lm-head:**

    Currently only support in AutoRound format inference for this config

    ```bash
    auto-round --model_name facebook/opt-125m  --bits 4 --group_size 128 --quant_lm_head --format "auto_round"
    ```

- **Enable marlin kernel:**

[//]: # (  - We support inference repacking for auto_round sym quantized models)

[//]: # (  ```python)

[//]: # (  from transformers import AutoModelForCausalLM, AutoTokenizer)

[//]: # (  from auto_round import AutoRoundConfig)

[//]: # (  backend = "cuda_marlin" #supported in auto_round>0.3.1 and 'pip install -v gptqmodel --no-build-isolation'&#41;)

[//]: # (  quantization_config = AutoRoundConfig&#40;backend=backend&#41;)

[//]: # (  quantized_model_path = "./tmp_autoround")

[//]: # (  model = AutoModelForCausalLM.from_pretrained&#40;quantized_model_path,)

[//]: # (                               device_map=backend.split&#40;':'&#41;[0], quantization_config=quantization_config&#41;)

[//]: # (  ```)
  - To leverage auto-gptq marlin kernel, you need to install auto-gptq from source and export the model without sharding.

    ```bash
    auto-round --model facebook/opt-125m  --sym --bits 4 --group_size 128  --format "auto_gptq:marlin"
    ```

- **Utilize the AdamW Optimizer:**

    Include the flag `--adam`. Note that AdamW is less effective than sign gradient descent in many scenarios we tested.


- **Code generation LLM:**

    We utilized mbpp for calibration, but your own training dataset is highly recommended. Please note that samples with
    seqlen < args.seqlen will be dropped in current version.

    ```bash
     auto-round --model Salesforce/codegen25-7b-multi --bits 4 --dataset "mbpp" --seqlen 128 "
    ```
  
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
  For other evaluation framework, if the framework could support Huggingface models, typically it could support AutoRound format, only you need to do is import the following in the beginning of your code
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


- The following code will automatically detect device, and typically some error message will remind you to install some extra libraries
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