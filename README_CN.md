<div align="center">



<p align="center">
  <img src="docs/imgs/AutoRound.png" alt="AutoRound Overview" width="20%">
</p>


<h3> 面向 LLM 的高级量化算法</h3>

[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/intel/auto-round)
[![version](https://img.shields.io/badge/release-0.9.5-green)](https://github.com/intel/auto-round)
[![license](https://img.shields.io/badge/license-Apache%202-9C27B0)](https://github.com/intel/auto-round/blob/main/LICENSE)
<a href="https://huggingface.co/Intel">
<img alt="Model Checkpoints" src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Models-F57C00">
</a>

[English](README.md) | 简体中文

---
<div align="left">

## 🚀 AutoRound 是什么？

AutoRound 是面向**大语言模型（LLMs）和视觉-语言模型（VLMs）的高级量化工具。它通过引入符号梯度下降方法（sign-gradient descent）** ，只需进行极少的调参，就能在 **极低精度（2–4 bits）** 下保持较高的准确率，同时也具备较好的硬件兼容性。更多细节请参考论文 [SignRoundV1](https://arxiv.org/pdf/2309.05516) 和 [SignRoundV2](http://arxiv.org/abs/2512.04746)。使用说明请参阅 [用户指南](./docs/step_by_step.md).

<p align="center">
  <img src="docs/imgs/autoround_overview.png" alt="AutoRound Overview" width="80%">
</p>


## 🆕 最新进展

* [2025/12] **SignRoundV2** 论文已发布。开启 `enable_alg_ext` 并使用 **AutoScheme** API 进行混合精度量化即可复现论文实验结果。详见：[*论文*](http://arxiv.org/abs/2512.04746)，[*LLaMA 模型评估说明*](./docs/alg_202508.md)。

* [2025/11]  **LLM-Compressor** 已支持AutoRound算法。详见：[*使用方法*](https://github.com/vllm-project/llm-compressor/tree/main/examples/autoround/README.md)，[*vLLM 博客*](https://blog.vllm.ai/2025/12/09/intel-autoround-llmc.html)，[*RedHat 博客*](https://developers.redhat.com/articles/2025/12/09/advancing-low-bit-quantization-llms-autoround-x-llm-compressor)，[*X 推文*](https://x.com/vllm_project/status/1998710451312771532)，[*Intel 博客*](https://community.intel.com/t5/Blogs/Products-and-Solutions/HPC/Advancing-Low-Bit-Quantization-for-LLMs-AutoRound-x-LLM/post/1729336)，[*LinkedIn*](https://www.linkedin.com/posts/vllm-project_advancing-lowbit-quantization-for-llms-activity-7404478053768441856-ru8f/?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAapNW8BLnAdCAr57GOwSCJXjf76ZvOEOAg)，[*微信*](https://mp.weixin.qq.com/s/l5WA-1_4ipffQN6GOH2Iqg)，[*知乎*](https://zhuanlan.zhihu.com/p/1982167638315664412)。

* [2025/11] 提供了 **增强版 GGUF** 量化算法，开启 `--enable_alg_ext`即可 。[*准确度*](./docs/gguf_alg_ext_acc.md)提供了少量准确率数据。

* [2025/10] AutoRound 已集成至 **SGLang**。详见：[*使用方法*](https://docs.sglang.io/advanced_features/quantization.html#using-auto-round)，[*LMSYS 博客*](https://lmsys.org/blog/2025-11-13-AutoRound/)，[*X 推文*](https://x.com/lmsysorg/status/1991977019220148650?s=20)，[*Intel 博客*](https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/AutoRound-Meets-SGLang-Enabling-Quantized-Model-Inference-with/post/1727196)，[*LinkedIn*](https://www.linkedin.com/feed/update/urn:li:activity:7397742859354857472)。

* [2025/10] 提供 **混合精度** 算法，可在几分钟内自动生成混合bit方案。详见：[*使用方法*](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#autoscheme)，[*准确度*](./docs/auto_scheme_acc.md)。

* [2025/09] 支持 **MXFP4** 和 **NVFP4** 数据类型。详见：[*准确度*](./docs/mxnv_acc.md)。

* [2025/08] ` 提供 **改进版 INT2** 算法, 请开启 `--enable_alg_ext。详见：[*准确度*](./docs/alg_202508.md)。

* [2025/07] 支持 **GGUF** 格式。详见：[*使用方法*](./docs/step_by_step.md#gguf-format)。

* [2025/05] AutoRound 已集成至 **vLLM**。详见：[*使用方法*](https://docs.vllm.ai/en/latest/features/quantization/auto_round/)，[*Medium 博客*](https://medium.com/@NeuralCompressor/accelerating-vllm-and-sglang-deployment-using-autoround-45fdc0b2683e)，[*小红书*](https://www.xiaohongshu.com/explore/69396bc6000000000d03e473?note_flow_source=wechat&xsec_token=CB6G3F_yM99q8XfusvyRlJqm8Db4Es2k0kYIHdIUiSQ9g=)。

* [2025/05] AutoRound 已集成至 **Transformers**。详见：[*博客*](https://huggingface.co/blog/autoround)。

* [2025/03] **DeepSeek-R1** 模型（约 200GB）在量化（使用INT2-混合精度）后仍保持了 97.9% 的准确度。详见：[*模型*](https://huggingface.co/OPEA/DeepSeek-R1-int2-mixed-sym-inc)。


## ✨ 核心特性


✅ **高准确度** 在 2–3 bit 下也能保持较强的性能（[示例模型](https://huggingface.co/collections/OPEA/2-3-bits-67a5f0bc6b49d73c01b4753b)）， 4 bit 量化在[基准](https://huggingface.co/spaces/Intel/low_bit_open_llm_leaderboard)上保持领先水平。

✅ **良好的生态集成** 量化模型已被多个知名库支持，如 **Transformers、vLLM、SGLang** 等。

✅ **多格式导出** 可以导出到​**AutoRound、AutoAWQ、AutoGPTQ、GGUF**​ 格式，。详见：[导出格式](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#supported-export-formats)

✅ **自动混合精度** 可在几分钟内自动生成混合bit策略，但需要模型在 BF16下内存占用量的1.1–1.5倍作为额外开销。详见：[准确度结果](https://github.com/intel/auto-round/blob/main/docs/auto_scheme_acc) 和 [用户指南](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#autoscheme)

✅ **优化的就近取整（RTN）模式** 使用 `--iters 0`​ 可快速完成量化（但在 4 bit 下准确度会有一定降低）。详见：[opt_rtn 模式](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#opt-rtn-mode)

✅ **低量化成本** 单卡 GPU 上量化 7B 模型仅需约 10 分钟。详见：[量化成本](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#quantization-costs)

✅ **支持 10+ VLM 模型**  十余款视觉-语言模型开箱即用式量化。详见：[示例模型](https://huggingface.co/collections/OPEA/vlms-autoround-675bc712fdd6a55ebaf11bfa)，[支持矩阵](https://github.com/intel/auto-round/tree/main/auto_round/mllm#support-matrix)

✅ **多种量化 Recipes** 可选 `auto-round-best`​、`auto-round`​、`auto-round-light`​ 以满足不同需求。详见：[量化Recipes](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#recipe-recommendation)

✅ **高级工具集** 支持[多 GPU 量化](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#devicemulti-gpu-setting-in-quantization)、[多标定数据集](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#default-dataset)以及[十余种推理后端](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#specify-inference-backend)。

✅ **不止于单一权重量化** 正在积极扩展更多数据类型的支持，包括 **MXFP、NVFP、W8A8** 等。


## 安装

### 从 PyPI 安装

```shell
# CPU / Intel GPU / CUDA
pip install auto-round

# HPU
pip install auto-round-lib
```

<details>
  <summary>从源码编译安装</summary>

  ```bash
  # CPU/Intel GPU/CUDA
  pip install .

  # HPU
  python setup.py install lib
  ```

</details>

## 模型量化（CPU / Intel GPU / Gaudi / CUDA）

### CLI 用法

完整的参数列表可通过在终端运行 `auto-round -h` 查看。

> **支持通过 ModelScope 下载模型，只需设置** ​**​`AR_USE_MODELSCOPE=1`​**。

```shell
auto-round \
    --model Qwen/Qwen3-0.6B \
    --scheme "W4A16" \
    --format "auto_round" \
    --output_dir ./tmp_autoround
```

我们还提供另外两种 recipes：`auto-round-best`​（追求最高准确度）和 `auto-round-light`（追求更快速度），具体如下：


<details>
  <summary>其他 Recipes</summary>

  ```bash
# 最佳准确度，速度慢 3 倍，low_gpu_mem_usage 可节省 ~20G 显存，但会慢 ~30%
auto-round-best \
    --model Qwen/Qwen3-0.6B \
    --scheme "W4A16" \
    --low_gpu_mem_usage 
  ```

  ```bash
# 2–3 倍加速，W4 下准确度略降，W2 下准确度下降更明显
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

总的来说，我们建议在 ​**W4A16 场景下使用 auto-round，W2A16 场景下使用 auto-round-best 并启用 ​`enable_alg_ext`​​** 。当然你也可以根据自身需求和手头资源来自行调整配置。

### API 用法

```python
from auto_round import AutoRound

# 加载模型（支持 FP8 / BF16 / FP16 / FP32）
model_name_or_path = "Qwen/Qwen3-0.6B"

# 可用 scheme："W2A16", "W3A16", "W4A16", "W8A16", "NVFP4", "MXFP4"（无真实 kernel）, "GGUF:Q4_K_M" 等
ar = AutoRound(model_name_or_path, scheme="W4A16")

# 最高准确度（慢 4–5 倍）
# `low_gpu_mem_usage=True` 可节省 ~20GB 显存，但会慢 ~30%
# ar = AutoRound(model_name_or_path, nsamples=512, iters=1000, low_gpu_mem_usage=True)

# 更快量化（2–3 倍加速），但在 W4G128 下准确度会略微下降
# ar = AutoRound(model_name_or_path, nsamples=128, iters=50, lr=5e-3)

# 支持格式："auto_round"（默认）, "auto_gptq", "auto_awq", "llm_compressor", "gguf:q4_k_m" 等
ar.quantize_and_save(output_dir="./qmodel", format="auto_round")
```

<details>
<summary>核心超参数说明</summary>

##### 量化方案与配置

- ​**​`scheme`​**​（str | dict | AutoScheme）：预定义量化键，如 `W4A16`​、`MXFP4`​、`NVFP4`​、`GGUF:Q4_K_M`。对于 MXFP4/NVFP4，推荐导出为 LLM-Compressor 格式。
- ​**​`bits`​**​（int）：量化比特数（默认 `None`），非空时会覆盖 scheme 设置。
- ​**​`group_size`​**​（int）：量化分组大小（默认 `None`），非空时会覆盖 scheme 设置。
- ​**​`sym`​**​（bool）：是否使用对称量化（默认 `None`），非空时会覆盖 scheme 设置。
- ​**​`layer_config`​**​（dict）：逐层量化配置（默认 `None`），主要用于自定义混合方案。

##### 算法相关设置

- ​**​`enable_alg_ext`​**​（bool）：[实验性功能] 仅在 `iters > 0`​ 时生效。为特定 scheme（如 MXFP4 / W2A16）启用算法扩展，可能显著提升效果。默认 `False`。
- ​**​`disable_opt_rtn`​**​（bool | None）：对特定 scheme（如 GGUF 和 WOQ）使用纯 RTN 模式。默认 `None`​。若为 None，通常默认为 `False`​ 以提升准确度，但在已知问题下可能设为 `True`。

##### 训练参数

- ​**​`iters`​**​（int）：调参迭代次数（默认 `200`​）。常用取值：0（RTN 模式）、50（推荐 `lr=5e-3`）、1000。迭代次数越多，准确度越高，但速度越慢。
- ​**​`lr`​**​（float）：舍入值学习率（默认 `None`​）。若为 None，则自动设为 `1.0/iters`。
- ​**​`batch_size`​**​（int）：训练 batch size（默认 `8`​），也常用 `4`。
- ​**​`enable_deterministic_algorithms`​**​（bool）：是否启用确定性算法以保证可复现性（默认 `False`）。

##### 标定数据集

- ​**​`dataset`​**​（str | list | tuple | DataLoader）：用于调参的数据集（默认 `"NeelNanda/pile-10k"`​）。支持本地 JSON 文件和数据集组合，如 `"./tmp.json,NeelNanda/pile-10k:train,mbpp:train+validation+test"`。
- ​**​`nsamples`​**​（int）：调参样本数（默认 `128`）。
- ​**​`seqlen`​**​（int）：调参序列长度（默认 `2048`）。

##### 设备 / 速度配置

- ​**​`enable_torch_compile`​**（bool）：若无异常，通常建议开启以获得更快的量化速度和更低资源消耗。
- ​**​`low_gpu_mem_usage`​**​（bool）：是否将中间特征卸载到 CPU，以约 20% 的时间代价节省显存（默认 `False`）。
- ​**​`low_cpu_mem_usage`​**​（bool）：[实验性功能] 是否启用即时保存以减少内存占用（默认 `False`）。
- ​**​`device_map`​**​（str | dict | int）：调参使用的设备，如 `auto`​、`cpu`​、`cuda`​、`0,1,2`​（默认 `0`​）。使用 `auto` 时会尝试利用所有可用 GPU。

</details>

### 支持的量化方案
<details>
<summary>详细说明</summary>
灰色表示无 kernel 或仅有低效/参考实现。BF16 主要用于 AutoScheme。

|格式|支持的方案|
| ------| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|**auto_round**|W4A16（推荐）、W2A16、W3A16、W8A16、W2A16G64、W2A16G32、`MXFP4`​、`MXFP8`​、`MXFP4_RCEIL`​、`MXFP8_RCEIL`​、`NVFP4`​、`FPW8A16`​、`FP8_STATIC`​、`BF16`|
|**auto_awq**|W4A16（推荐）、BF16|
|**auto_gptq**|W4A16（推荐）、W2A16、W3A16、W8A16、W2A16G64、W2A16G32、BF16|
|**llm_compressor**|NVFP4（推荐）、`MXFP4`​、`MXFP8`​、`FPW8A16`​、`FP8_STATIC`|
|**gguf**|GGUF:Q4\_K\_M（推荐）、Auto-RoundGGUF:Q2\_K\_S、GGUF:Q3\_K\_S、GGUF:Q3\_K\_M、GGUF:Q3\_K\_L、GGUF:Q4\_K\_S、GGUF:Q5\_K\_S、GGUF:Q5\_K\_M、GGUF:Q6\_K、GGUF:Q4\_0、GGUF:Q4\_1、GGUF:Q5\_0、GGUF:Q5\_1、GGUF:Q8\_0|
|**fake**|​`所有方案（仅用于研究）`|
</details>

### 自适应量化（AutoScheme）方案（实验性功能）

AutoScheme 内置自动化算法，可生成 **自适应的混合位宽/数据类型** 的量化recipe。关于 AutoScheme 的更多细节可参考[用户指南](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#autoscheme)。

```python
from auto_round import AutoRound, AutoScheme

model_name = "Qwen/Qwen3-8B"
avg_bits = 3.0
scheme = AutoScheme(avg_bits=avg_bits, options=("GGUF:Q2_K_S", "GGUF:Q4_K_S"), ignore_scale_zp_bits=True)
layer_config = {"lm_head": "GGUF:Q6_K"}

# 对于非 GGUF 方案，将 iters 改为 200
ar = AutoRound(model=model_name, scheme=scheme, layer_config=layer_config, iters=0)
ar.quantize_and_save()
```

<details>
<summary>AutoScheme 的重要超参数</summary>

##### AutoScheme 超参数

- ​**​`avg_bits`​**​  **(float)** ：模型整体目标平均位宽，仅将量化层纳入平均位宽的计算范围。
- ​**​`options`​**​  **(str | list[str] | list[QuantizationScheme])** ​：选候的量化方案集合，支持单个用逗号分隔的字符串（例如 `"W4A16,W2A16"`​）、字符串列表（例如 `["W4A16", "W2A16"]`​）或 `QuantizationScheme` 对象列表三种格式。
- ​**​`ignore_scale_zp_bits`​**​  **(bool)** ​：仅支持 API 调用场景，用于决定在计算平均位宽时，是否排除 scale 与 zero-point 的比特数（默认：`False`）。
- ​**​`shared_layers`​**​  **(Iterable[Iterable[str]], optional)** ：仅支持 API 调用场景，用于定义共享同一量化设置的层分组。
- ​**​`batch_size`​**​  **(int, optional)** ​：仅支持 API 调用场景，可设为 `1` 以降低显存占用，但会增加调参时间。

</details>

### 视觉语言模型（VLM）的 API 调用方法

若在量化过程中出现问题，可以尝试设置 `iters=0`​（启用 RTN）和 `group_size=32` 来改善效果。


<details>
  <summary>点击展开</summary>

**该功能为实验性功能，后续可能会有改动。**

默认情况下，AutoRound 仅对 VLM 的文本模块进行量化，且采用 `NeelNanda/pile-10k`​ 作为校准数据集。若要量化整个模型，可通过设置 `quant_nontext_module`​ 为 True 实现（但目前该功能的支持范围有限）。更多信息请参考 AutoRound 的 [readme] (https://github.com/intel/auto-round/blob/main/auto_round/mllm/README%7Creadme%5D%5D%E3%80%82)

```python
from auto_round import AutoRound

# 加载模型
model_name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"
# 量化模型
ar = AutoRound(model_name_or_path, scheme="W4A16")
output_dir = "./qmodel"
ar.quantize_and_save(output_dir)
```

</details>



## 模型推理

### vLLM（CPU / Intel GPU / CUDA）

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

### SGLang（Intel GPU / CUDA）

**注意：当前对混合专家模型（MoE）模型和视觉语言（VLM）模型的支持范围仍然有限。**

```python
import sglang as sgl

llm = sgl.Engine(model_path="Intel/DeepSeek-R1-0528-Qwen3-8B-int4-AutoRound")
prompts = [
    "Hello, my name is",
]
sampling_params = {"temperature": 0.6, "top_p": 0.95}

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```

### Transformers（CPU / Intel GPU / Gaudi / CUDA）

AutoRound 支持十余种推理后端，并会根据已安装的库自动选择最优可用后端；若检测到更优后端但缺少相关依赖时，也会提示用户安装额外库。

​**推理过程中请避免手动将量化后的模型迁移到其他设备**​（例如执行 `model.to('cpu')`），否则可能引发未知异常。

目前对 Gaudi 设备的支持较为有限。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Intel/DeepSeek-R1-0528-Qwen3-8B-int4-AutoRound"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
```

## 研究成果 & 其他活动

[SignRoundV2: Closing the Performance Gap in Extremely Low-Bit Post-Training Quantization for LLMs](https://arxiv.org/abs/2512.04746)（202512 论文）

[Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLM](https://aclanthology.org/2024.findings-emnlp.662/)（202309 论文）

[TEQ: Trainable Equivalent Transformation for Quantization of LLMs](https://arxiv.org/abs/2310.10944)（202310 论文）

[Effective Post-Training Quantization for Large Language Models](https://medium.com/intel-analytics-software/effective-post-training-quantization-for-large-language-models-with-enhanced-smoothquant-approach-93e9d104fb98)（202304 博客）

更多内容请查看 [完整论文列表](./docs/publication_list.md).

## 致谢

特别感谢 AutoGPTQ、AutoAWQ、GPTQModel、Triton、Marlin、ExLLaMAV2 等开源 low-precision 库提供低精度 CUDA kernel，在此基础上 AutoRound 项目作了利用与集成。

## 🌟 支持我们

如果觉得 AutoRound 对你有帮助，欢迎给仓库点个 ⭐，并分享给你的社区！

