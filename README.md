<div align="center">


AutoRound
===========================
<h3> Advanced Quantization Algorithm for LLMs</h3>

[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/intel/auto-round)
[![version](https://img.shields.io/badge/release-0.7.1-green)](https://github.com/intel/auto-round)
[![license](https://img.shields.io/badge/license-Apache%202-9C27B0)](https://github.com/intel/auto-round/blob/main/LICENSE)
<a href="https://huggingface.co/Intel">
<img alt="Model Checkpoints" src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Models-F57C00">
</a>
---
<div align="left">

## 🚀 What is AutoRound?

AutoRound is an advanced quantization library designed for Large Language Models (LLMs) and Vision-Language Models (VLMs). 
It delivers high accuracy at ultra-low bit widths (2–4 bits) with minimal tuning by leveraging sign-gradient descent and offering broad hardware compatibility. 
For more details, see our [paper](https://arxiv.org/pdf/2309.05516) for more details and explore quantized models available on several Hugging Face Spaces, e.g. [Intel](https://huggingface.co/Intel), [OPEA](https://huggingface.co/OPEA),  [Kaitchup](https://huggingface.co/kaitchup)
and [fbaldassarri](https://huggingface.co/fbaldassarri). For usage instructions, please refer to  [User Guide](./docs/step_by_step.md).

<p align="center">
  <img src="docs/imgs/autoround_overview.png" alt="AutoRound Overview" width="80%">
</p>


## 🆕 What's New
[2025/10] We proposed a fast algorithm to generate **mixed bits/datatypes** schemes in minutes. Please
refer to the documentation for accuracy [results](./docs/auto_scheme_acc.md) and [this guide](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#autoscheme) for usage instructions.

[2025/09] AutoRound now includes experimental support for the **mxfp4 and nvfp4 dtypes**. For accuracy results, see the [documentation](./docs/mxnv_acc.md)
. We currently recommend exporting to the LLM-Compressor format.

[2025/08] AutoRound now provides experimental support for **an improved INT2 algorithm** via `--enable_alg_ext`. See this [documentation](./docs/alg_202508.md)
 for some accuracy results. 

[2025/07] AutoRound now offers experimental support for **GGUF** format, and recommends using optimized RTN mode (--iters 0) for
  all bits other than 3 bits. Example
  models: [Intel/Qwen3-235B-A22B-q2ks-mixed-AutoRound](https://huggingface.co/Intel/Qwen3-235B-A22B-q2ks-mixed-AutoRound)
  and [Intel/DeepSeek-R1-0528-q2ks-mixed-AutoRound](https://huggingface.co/Intel/DeepSeek-R1-0528-q2ks-mixed-AutoRound). **A more advanced algorithm** tailored for specific configurations may be available in
  v0.8.1.

[2025/05] AutoRound has been integrated into **vLLM**. You can now run models in the AutoRound format directly with
  vLLM versions later than v0.85.post1.

[2025/04] AutoRound has been integrated into **Transformers**. You can run models in the AutoRound format directly
  with Transformers versions later than 4.51.3.

[2025/03] The INT2-mixed **DeepSeek-R1** model (~200GB) retains 97.9% accuracy. Check
  out [OPEA/DeepSeek-R1-int2-mixed-sym-inc](https://huggingface.co/OPEA/DeepSeek-R1-int2-mixed-sym-inc).


## ✨ Key Features


✅ **Superior Accuracy**
Delivers strong performance even at 2–3 bits [example models](https://huggingface.co/collections/OPEA/2-3-bits-67a5f0bc6b49d73c01b4753b), with leading results at 4 bits [benchmark](https://huggingface.co/spaces/Intel/low_bit_open_llm_leaderboard).

✅ **Ecosystem Integration**
Seamlessly works with **Transformers, vLLM,** and more.

✅ **Multiple Formats Export**
Support **AutoRound, AutoAWQ, AutoGPTQ, and GGUF** for maximum compatibility. Details are shown in [export formats](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#supported-export-formats)

✅ **Affordable Quantization Cost**
Quantize 7B models in about 10 minutes on a single GPU. Details are shown in [quantization costs](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#quantization-costs)

✅ **Fast Mixed Bits/Dtypes Scheme Generation**
Automatically configure in minutes, with about 2X-4X the model’s BF16 VRAM size as overhead. Accuracy [results](./docs/auto_scheme_acc.md) and [user guide](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#autoscheme).

✅ **10+ VLMs Support**
Out-of-the-box quantization for 10+ vision-language models [example models](https://huggingface.co/collections/OPEA/vlms-autoround-675bc712fdd6a55ebaf11bfa), [support matrix](https://github.com/intel/auto-round/tree/main/auto_round/mllm#support-matrix)

✅ **Layerwise Mixed Bits Quantization**
Assign different bits per layer for fine-grained accuracy/performance trade-offs. Details are shown in [mixed bits quantization](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#mixed-bits-usage)

✅ **Optimized Round-to-Nearest Mode**
Use `--iters 0` for fast, calibration-free quantization with some accuracy drop for 4 bits. Details are shown in [opt_rtn mode](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#opt-rtn-mode)

✅ **Multiple Recipes**
Choose from `auto-round-best`, `auto-round`, and `auto-round-light` to suit your needs. Details are shown in [quantization recipes](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#recipe-recommendation)

✅ Advanced Utilities
Includes [multiple gpus quantization](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#devicemulti-gpu-setting-in-quantization), [multiple calibration datasets](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#default-dataset) and support for [10+ runtime backends](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#specify-inference-backend).

✅ Beyond weight only quantization. We are actively expanding support for additional datatypes such as **MXFP**, NVFP, W8A8, and more.


## Installation

### Install from pypi

```bash
# CPU/Intel GPU/CUDA
pip install auto-round

# HPU
pip install auto-round-lib
```

<details>
  <summary>Build from Source</summary>

  ```bash
  # CPU/Intel GPU/CUDA
  pip install .

  # HPU
  python setup.py install lib
  ```

</details>

## Model Quantization (CPU/Intel GPU/Gaudi/CUDA)

### CLI Usage
The full list of supported arguments is provided by calling `auto-round -h` on the terminal.

```bash
auto-round \
    --model Qwen/Qwen3-0.6B \
    --scheme "W4A16" \
    --format "auto_round" \
    --output_dir ./tmp_autoround
```

We offer another two recipes, `auto-round-best` and `auto-round-light`, designed for optimal accuracy and improved speed, respectively. Details are as follows.
<details>
  <summary>Other Recipes</summary>

  ```bash
# Best accuracy, 3X slower, low_gpu_mem_usage could save ~20G but ~30% slower
auto-round-best \
    --model Qwen/Qwen3-0.6B \
    --scheme "W4A16" \
    --low_gpu_mem_usage 
  ```

  ```bash
# 2-3X speedup, slight accuracy drop at W4 and larger accuracy drop at W2
auto-round-light \
    --model Qwen/Qwen3-0.6B \
    --scheme "W4A16" 

  ```

  <!-- ```bash
auto-round-fast \
# Fast and low memory, 2-3X speedup, slight accuracy drop at W4G128
    --model Qwen/Qwen3-0.6B \
    --bits 4 \
    --group_size 128 \
  ``` -->

</details>

In conclusion, we recommend using **auto-round for W4A16 and auto-round-best with `enable_alg_ext` for W2A16**. However, you may adjust the
configuration to suit your specific requirements and available resources.

### API Usage

```python
from auto_round import AutoRound

# Load a model (supports FP8/BF16/FP16/FP32)
model_name_or_path = "Qwen/Qwen3-0.6B"

# Available schemes: "W2A16", "W3A16", "W4A16", "W8A16", "NVFP4", "MXFP4" (no real kernels), "GGUF:Q4_K_M", etc.
ar = AutoRound(model_name_or_path, scheme="W4A16")

# Highest accuracy (4–5× slower).
# `low_gpu_mem_usage=True` saves ~20GB VRAM but runs ~30% slower.
# ar = AutoRound(model_name_or_path, nsamples=512, iters=1000, low_gpu_mem_usage=True)

# Faster quantization (2–3× speedup) with slight accuracy drop at W4G128.
# ar = AutoRound(model_name_or_path, nsamples=128, iters=50, lr=5e-3)

# Supported formats: "auto_round" (default), "auto_gptq", "auto_awq", "llm_compressor", "gguf:q4_k_m", etc.
ar.quantize_and_save(output_dir="./tmp_autoround", format="auto_round")
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

**This feature is experimental and may be subject to changes**.

By default, AutoRoundMLLM only quantize the text module of VLMs and uses `NeelNanda/pile-10k` for calibration. To
quantize the entire model, you can enable `quant_nontext_module` by setting it to True, though support for this feature
is limited. For more information, please refer to the AutoRoundMLLM [readme](./auto_round/mllm/README.md).

```python
from auto_round import AutoRoundMLLM

# Load the model
model_name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"
# Quantize the model
ar = AutoRoundMLLM(model_name_or_path, scheme="W4A16")
output_dir = "./tmp_autoround"
ar.quantize_and_save(output_dir)
```

</details>



## Model Inference

### vLLM (CPU/Intel GPU/CUDA)
Please note that support for the MoE models and visual language models is currently limited.
```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
]
sampling_params = SamplingParams(temperature=0.6, top_p=0.95)
model_name = "Intel/DeepSeek-R1-0528-Qwen3-8B-int4-AutoRound"
llm = LLM(model=model_name)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

### Transformers (CPU/Intel GPU/Gaudi/CUDA)


AutoRound support 10+ backends and automatically selects the best available backend based on the installed libraries and prompts the user to
install additional libraries when a better backend is found.

**Please avoid manually moving the quantized model to a different device** (e.g., model.to('cpu')) during inference, as
this may cause unexpected exceptions.

The support for Gaudi device is limited.
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Intel/DeepSeek-R1-0528-Qwen3-8B-int4-AutoRound"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
```
## Acknowledgement
Special thanks to open-source low precision libraries such as AutoGPTQ, AutoAWQ, GPTQModel, Triton, Marlin, and ExLLaMAV2 for providing low-precision CUDA kernels, which are leveraged in AutoRound.

## 🌟 Support Us
If you find AutoRound helpful, please ⭐ star the repo and share it with your community!






