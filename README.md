<div align="center">


AutoRound
===========================
<h3> Advanced Quantization Algorithm for LLMs</h3>

[![python](https://img.shields.io/badge/python-3.9%2B-blue)](https://github.com/intel/auto-round)
[![version](https://img.shields.io/badge/release-0.6.0-green)](https://github.com/intel/auto-round)
[![license](https://img.shields.io/badge/license-Apache%202-9C27B0)](https://github.com/intel/auto-round/blob/main/LICENSE)
<a href="https://huggingface.co/Intel">
<img alt="Model Checkpoints" src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Models-F57C00">
</a>
---
<div align="left">

## 🚀 What is AutoRound?

AutoRound is an advanced quantization library designed for Large Language Models (LLMs) and Vision-Language Models (VLMs). It delivers high accuracy at ultra-low bit widths (2–4 bits) with minimal tuning by leveraging sign-gradient descent and offering broad hardware compatibility. Check out our paper on [arxiv](https://arxiv.org/pdf/2309.05516) for more details and quantized models in several
Hugging Face Spaces,
e.g. [Intel](https://huggingface.co/Intel), [OPEA](https://huggingface.co/OPEA),  [Kaitchup](https://huggingface.co/kaitchup)
and [fbaldassarri](https://huggingface.co/fbaldassarri).

<p align="center">
  <img src="docs/imgs/autoround_overview.png" alt="AutoRound Overview" width="80%">
</p>

## ✨ Key Features


✅ **Superior Accuracy**
Delivers strong performance even at 2–3 bits [example models](https://huggingface.co/collections/OPEA/2-3-bits-67a5f0bc6b49d73c01b4753b), with leading results at 4 bits [benchmark](https://huggingface.co/spaces/Intel/low_bit_open_llm_leaderboard).

✅ **Ecosystem Integration**
Seamlessly works with Transformers, vLLM, TorchAO, sglang(on going,[pr](https://github.com/sgl-project/sglang/pull/6226)) and more.

✅ **Multiple Formats Export**
Support **AutoRound, AutoAWQ, AutoGPTQ, and GGUF** for maximum compatibility. Details are shown in [export formats](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#supported-export-formats)

✅ **Affordable Quantization Cost**
Quantize 7B models in about 10 minutes on a single GPU. Details are shown in [quantization costs](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#quantization-costs)

✅ **10+ VLMs Support**
Out-of-the-box quantization for 10+ vision-language models [example models](https://huggingface.co/collections/OPEA/vlms-autoround-675bc712fdd6a55ebaf11bfa), [support matrix](https://github.com/intel/auto-round/tree/main/auto_round/mllm#support-matrix)

✅ **Layerwise Mixed Bits Quantization**
Assign different bits per layer for fine-grained accuracy/performance trade-offs. Details are shown in [mixed bits quantization](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#mixed-bits-usage)

✅ **Round-to-Nearest Mode**
Use `--iters 0` for fast, calibration-free quantization with some accuracy drop. Details are shown in [rtn mode](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#rtn-mode)

✅ **Multiple Recipes**
Choose from `auto-round-best`, `auto-round`, and `auto-round-light` to suit your needs. Details are shown in [quantization recipes](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#recipe-recommendation)

✅ Advanced Utilities
Includes [multiple gpus quantization](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#devicemulti-gpu-setting-in-quantization), [multiple calibration datasets](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#default-dataset) and support for [10+ runtime backends](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#specify-inference-backend).

🟨 Beyond weight only quantization. We are actively expanding support for additional datatypes such as **MXFP**, NVFP, W8A8, and more.

## 🆕 What's New

[2025/07] AutoRound now offers experimental support for **GGUF** format, and recommends using optimized RTN mode (--iters 0) for
  all bits other than 3 bits. **A more advanced algorithm** tailored for specific configurations may be available in
  v0.6.1. Example
  models: [Intel/Qwen3-235B-A22B-q2ks-mixed-ar](https://huggingface.co/Intel/Qwen3-235B-A22B-q2ks-ar)
  and [Intel/DeepSeek-R1-0528-q2ks-mixed-ar](https://huggingface.co/Intel/DeepSeek-R1-0528-q2ks-mixed-ar).

[2025/05] AutoRound provides some recipes for **DeepSeek-R1-0528**, please refer
  to [IntelDeepSeek-R1-0528-int2-mixed-ar](https://huggingface.co/Intel/DeepSeek-R1-0528-int2-mixed-ar), [Intel/DeepSeek-R1-0528-int4-ar](https://huggingface.co/Intel/DeepSeek-R1-0528-int4-ar)
  and [Intel/DeepSeek-R1-0528-int4-asym-ar](https://huggingface.co/Intel/DeepSeek-R1-0528-int4-asym-ar) for
  more details.

[2025/05] AutoRound has been integrated into **vLLM**. You can now run models in the AutoRound format directly with
  vLLM versions later than v0.85.post1.

[2025/04] AutoRound has been integrated into **Transformers**. You can run models in the AutoRound format directly
  with Transformers versions later than 4.51.3.

[2025/03] The INT2-mixed **DeepSeek-R1** model (~200GB) retains 97.9% accuracy. Check
  out [OPEA/DeepSeek-R1-int2-mixed-sym-inc](https://huggingface.co/OPEA/DeepSeek-R1-int2-mixed-sym-inc).

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

Please check out [User guide](./docs/step_by_step.md) for more details
### Command Line Usage
Please change to `auto-round-mllm` for visual-language models (VLMs) quantization. The full list of supported arguments is provided by calling `auto-round -h` on the terminal.

```bash
auto-round \
    --model Qwen/Qwen3-0.6B \
    --bits 4 \
    --group_size 128 \
    --format "auto_gptq,auto_awq,auto_round" \
    --output_dir ./tmp_autoround
```

We offer another two configurations, `auto-round-best` and `auto-round-light`, designed for optimal accuracy and improved speed, respectively. Details are as follows.
<details>
  <summary>Other Recipes</summary>

  ```bash
## best accuracy, 3X slower, low_gpu_mem_usage could save ~20G but ~30% slower
auto-round-best \
    --model Qwen/Qwen3-0.6B \
    --bits 4 \
    --group_size 128 \
    --low_gpu_mem_usage 
  ```

  ```bash
## light accuracy, 2-3X speedup, slight accuracy drop at W4 and larger accuracy drop at W2
auto-round-light \
    --model Qwen/Qwen3-0.6B \
    --bits 4 \
    --group_size 128 \

  ```

  <!-- ```bash
auto-round-fast \
## fast and low memory, 2-3X speedup, slight accuracy drop at W4G128
    --model Qwen/Qwen3-0.6B \
    --bits 4 \
    --group_size 128 \
  ``` -->

</details>

In conclusion, we recommend using **auto-round for INT4 and auto-round-best for INT2**. However, you may adjust the
configuration to suit your specific requirements and available resources.

### API Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

bits, group_size, sym = 4, 128, True
autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym)

## the best accuracy, 4-5X slower, low_gpu_mem_usage could save ~20G but ~30% slower
# autoround = AutoRound(model, tokenizer, nsamples=512, iters=1000, low_gpu_mem_usage=True, bits=bits, group_size=group_size, sym=sym)

## 2-3X speedup, slight accuracy drop at W4G128
# autoround = AutoRound(model, tokenizer, nsamples=128, iters=50, lr=5e-3, bits=bits, group_size=group_size, sym=sym )

output_dir = "./tmp_autoround"
## format= 'auto_round'(default), 'auto_gptq', 'auto_awq'
autoround.quantize_and_save(output_dir, format="auto_round")
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
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer

## load the model
model_name = "Qwen/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

## quantize the model
bits, group_size, sym = 4, 128, True
autoround = AutoRoundMLLM(model, tokenizer, processor, bits=bits, group_size=group_size, sym=sym)
autoround.quantize()

# save the quantized model, set format='auto_gptq' or 'auto_awq' to use other formats
output_dir = "./tmp_autoround"
autoround.save_quantized(output_dir, format="auto_round", inplace=True)
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


AutoRound support 10+ backends an automatically selects the best available backend based on the installed libraries and prompts the user to
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





