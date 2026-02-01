操作指南
============

[English](./step_by_step.md) | 简体中文

本文档介绍了如何用 auto-round 量化大语言模型（LLM）。如需量化视觉大语言模型（VLM），请参阅[视觉大语言模型用户指南](../auto_round/compressors/mllm/README.md)；如需量化扩散模型，请参阅[扩散模型用户指南](../auto_round/compressors/diffusion/README.md)。

* [1 前提条件](#1-前提条件)
* [2 准备标定数据集](#2-准备标定数据集)
  + [默认数据集](#默认数据集)
  + [自定义数据集](#自定义数据集)
  + [数据集操作](#数据集操作)
* [3 量化操作](#3-量化操作)
  + [可选量化配置](#可选量化配置)
  + [支持的导出格式](#支持的导出格式)
  + [硬件兼容性](#硬件兼容性)
  + [环境配置](#环境配置)
  + [命令行使用方法](#命令行用法)
  + [API使用方法](#api使用方法)
    - [AutoRound API 基础用法](#AutoRound-API-基础用法)
    - [混合精度量化方案](#混合精度量化)
    - [AutoRoundBest 配置方案](#AutoRoundBest-高精度配置用法)
    - [AutoRoundLight 配置方案](#AutoRoundLight-高速度配置用法)
    - [配置方案推荐](#配置方案推荐)
  + [AutoScheme 自动方案](#AutoScheme-自动量化方案)
    - [命令行用法](#命令行用法-1)
    - [API 用法](#API-用法)
    - [AutoScheme 中的超参数](#AutoScheme-超参数说明)
  + [OPT RTN 模式](#OPT-RTN-优化舍入模式)
  + [GGUF 格式](#GGUF-格式量化)
  + [量化成本](#量化成本)
  + [设备/多 GPU 量化设置](#设备及多-GPU-量化设置)
    - [lm_head 量化中开启多 GPU 校准](#lm_head-量化中开启多-GPU-校准)
    - [手动配置设备映射](#手动配置设备映射)
  + [超参数调整](#超参数调整)
* [4 推理部署](#4-推理部署)
  + [CPU](#CPU)
  + [英特尔 GPU](#英特尔-GPU)
  + [CUDA](#CUDA)
  + [HPU](#HPU)
  + [指定推理后端](#指定推理后端)
  + [将 GPTQ/AWQ 模型转换为 AutoRound 格式](#将-GPT-或-AWQ-模型转换为autoround格式)
* [5 效果评估](#5-效果评估)
  + [单卡评估](#单-GPU-评估)
  + [多卡评估](#多-GPU-评估)
  + [注意事项](#注意事项)
* [6 已知问题](#6-已知问题)

## 1 前提条件

pip 安装 AutoRound 库（或从源码编译安装）

```bash
pip install auto-round
```

## 2 准备标定数据集

### 默认数据集

默认标定数据集为 Hugging Face 上的 [NeelNanda/pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k) ，该数据集会自动从 Dataset Hub 下载。同时也支持使用以下数据集：
- ModelScope 中的 `swift/pile-val-backup`：用于解决 HF 访问问题
- `BAAI/CCI3-HQ`：用于中文场景
- `codeparrot/github-code-clean`：用于代码场景
- `HuggingFaceH4/ultrachat_200k`：用于对话数据
- `madao33/new-title-chinese`：用于中文场景
- `mbpp`：用于代码场景
- `openbmb/Ultra-FineWeb`

### 自定义数据集

可通过以下方式指定：
- 用法一：向 `dataset` 参数传入本地 JSON 文件路径
- 用法二：参照[示例代码](../auto_round/calib_dataset.py)注册数据集，然后使用新的数据集名称和拆分参数初始化 AutoRound 对象。示例： `autoround=Autoround(dataset="NeelNanda/pile-10k:train", ...)`
- 用法三：向 `dataset` 参数传入字符串列表或者 input_ids 列表

    ~~~python
    def customized_data():
        # 注意！！！AutoRound 会舍弃长度小于 args.seqlen 的数据，并将超过该长度的数据截断至 args.seqlen 长度
        data = ["AutoRound 是面向大语言模型的先进量化算法" * 240]
        return data
    
    
    def customized_data_with_tokenizer(tokenizer, seqlen=2048):
        # 注意！！！AutoRound 会舍弃长度小于 args.seqlen 的数据
        data = ["AutoRound 是面向大语言模型的先进量化算法" * 240]
        tokens = []
        for d in data:
            token = tokenizer(d, truncation=True, max_length=seqlen, return_tensors="pt").data
            tokens.append(token)
        return tokens
    ~~~

### 数据集操作

**数据集组合**：可使用 `--dataset` 参数组合不同数据集并设置其他的参数。示例代码 `--dataset ./tmp.json,NeelNanda/pile-10k:num=256,mbpp:num=128`。此参数同时支持本地标定文件和 Hugging Face 数据集，同时还可用 `split = split1+split2` 指定某个数据集的多个拆分子集。

**样本拼接**：可使用 `--dataset NeelNanda/pile-10k:concat=True` 拼接标定样本。在该模式下，会先将所有样本拼接成完整文本，再按 seqlen 长度切分出来。

**启用对话模板**：可使用 `--dataset NeelNanda/pile-10k:apply_chat_template` 在分词前为标定数据应用对话模板，这在指令式模型的生成任务中比较常用。若需自定义系统提示词，可使用 `--dataset 'NeelNanda/pile-10k:apply_chat_template:system_prompt="你是一个乐于助人的智能助手。"'`

注意：如果没有开启拼接选项，长度小于 args.seqlen 的样本会被舍弃。

数据集之间请用英文逗号`,`分隔；单个数据集的参数请用英文冒号`:`分隔；同一参数的多个取值请用英文加号`+`连接。


## 3 量化操作

### 可选量化配置

AutoRound 支持多种量化配置：
- **W4A16**（bits:4, group_size:128, sym:True, act_bits:16）  # 4位权重，分组大小为128，对称量化，16位激活，
- **W8A16**（bits:8, group_size:128, sym:True, act_bits:16）  
- **W3A16**（bits:3, group_size:128, sym:True, act_bits:16）  
- **W2A16**（bits:2, group_size:128, sym:True, act_bits:16）  
- **GGUF:Q4_K_M**（支持 llamacpp 提供的所有 Q*_K、Q*_0、Q*_1 量化类型）
- **仅权重混合位宽量化**
- **NVFP4**（实验性功能）推荐导出为`llm_compressor`格式，参数：data_type=nvfp4, act_data_type=nvfp4, static_global_scale, group_size=16
- **MXFP4**（研究性功能，暂无实际内核）：标准 MXFP4 量化，参数：data_type=mxfp, act_data_type=mxfp, bits=4, act_bits=4, group_size=32
- **MXFP4_RCEIL**（研究性功能，暂无实际内核）：NVIDIA变体，参数：data_type=mxfp, act_data_type=mxfp_rceil, bits=4, act_bits=4, group_size=32
- **MXFP8**（研究性功能，暂无实际内核），参数：data_type=mxfp, act_data_type=mxfp_rceil, group_size=32
- **FPW8A16**（研究性功能，暂无实际内核），参数：data_type=fp8, group_size=0 -> 每张量
- **FP8_STATIC**（研究性功能，暂无实际内核），参数：data_type:fp8, act_data_type:fp8, group_size=-1（每通道），act_group_size=0（每张量）

你也可以根据需求修改`group_size`（分组大小）、`bits`（精度）、`sym`（是否对称量化）等其他配置，但某些目前可能尚无对应的高性能计算内核支持。


### 支持的导出格式

执行 `auto_round list format` 可查看所有支持的导出格式及其对应的量化方案。

**AutoRound 原生格式**：适用于 CPU、英特尔 GPU、CUDA、HPU 等设备，支持2位宽及混合精度推理，**兼容 [2、3、4、8] bits**。使用时需设置 `--format auto_round`。

**GGUF 格式**：实验性功能，适用于 CPU 设备，是社区主流格式之一，支持 `q*_k`、`q*_0`、`q*_1` 系列的量化。需设置 `--format gguf:q4_k_m`、`--format gguf:q2_k_s`等具体格式。

**AutoGPTQ 格式**：适用于 CUDA 设备的对称量化，在社区中广泛应用，**兼容 [2、3、4、8] bits **（但其**非对称推理核存在问题**，可能导致模型的精度大幅下降，尤其是在 2-bit 量化和小模型的场景；近期 Transformers 框架中 3-bits 量化也存在类似问题）。配置时需设置 `--format auto_gptq`。

**AutoAWQ 格式**：适用于 CUDA 设备的 4 位非对称量化，在社区中也广泛应用。**仅支持 4-bit 量化**。需设置 `--format auto_awq`。

**LLM-Compressor 格式**：**支持 NVFP4、MXFP4（内核开发中）、MXFP8** 等。需设置 `--format llm_compressor`。

#### 格式与方案支持对照表

> 灰色背景的 schemes 表示它没有专门优化的内核，或只有效率极低的参考内核。

| 格式            | 支持的量化方案                                                                                                                                                                                                 |
|:-------------- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **auto_round**  | W4A16、W2A16、W3A16、W8A16、W2A16G64、W2A16G32、`MXFP4`、`MXFP8`、`MXFP4_RCEIL`、`MXFP8_RCEIL`、`NVFP4`、`FPW8A16`、`FP8_STATIC`、`BF16`                                                                      |
| **auto_awq**    | W4A16、BF16                                                                                                                                                                                                   |
| **auto_gptq**   | W4A16、W2A16、W3A16、W8A16、W2A16G64、W2A16G32、BF16                                                                                                                                                           |
| **llm_compressor** | NVFP4、`MXFP4`、`MXFP8`、`FPW8A16`、`FP8_STATIC`                                                                                                                                                              |
| **gguf**        | GGUF:Q4_K_M、GGUF:Q2_K_S、GGUF:Q3_K_S、GGUF:Q3_K_M、GGUF:Q3_K_L、GGUF:Q4_K_S、GGUF:Q5_K_S、GGUF:Q5_K_M、GGUF:Q6_K、GGUF:Q4_0、GGUF:Q4_1、GGUF:Q5_0、GGUF:Q5_1、GGUF:Q8_0                                           |
| **fake**        | `所有方案（仅用于研究场景）`                                                                                                                                                                                   |

### 硬件兼容性

量化和推理均支持 CPU、英特尔 GPU、HPU 和 CUDA。

### 环境配置

为优化运行性能，量化前建议配置 AutoRound 的环境变量。关于日志级别、ModelScope 集成、工作区设置等可用的环境变量等更多细节，可参考[环境变量指南](./environments.md)。

### 命令行用法


- **AutoRound 基础方案**：
  
  该方案很好地兼顾了精度和训练耗时，**推荐在绝大多数场景下使用**。

  ```bash
  auto-round --model Qwen/Qwen3-0.6B  --scheme "W4A16"  --format "auto_gptq,auto_awq,auto_round"
  ```

- **AutoRoundBest 高精度方案**：
  
  绝大多数场景下，该方案能实现最好的模型精度，缺点是训练耗时是基础方案的 4~5 倍；**特别适合 2-bit 量化**，若算力充足，可作为首选。
  
  ```bash
  auto-round-best --model Qwen/Qwen3-0.6B  --scheme "W4A16"  --format "auto_gptq,auto_awq,auto_round"
  ```

- **AutoRoundLight 高速方案**：
  
  该方案训练速度最快（比基础方案快 2~3 倍），但小模型和 2-bit 量化下可能导致模型精度显著下降。所以**推荐在 4-bit 量化或参数量大于 3B 的模型的场景下使用**。
  
  ```bash
  auto-round-light --model Qwen/Qwen3-0.6B  --scheme "W4A16"  --format "auto_gptq,auto_awq,auto_round"
  ```

### API使用方法
#### AutoRound API 基础用法
该方案兼顾精度和训练耗时，**推荐在绝大多数场景下使用**。

```python
from auto_round import AutoRound

model_name_or_path = "Qwen/Qwen3-0.6B"
ar = AutoRound(
    model_name_or_path,
    scheme="W4A16",
    # 可选开启torch编译加速：enable_torch_compile=True,
)

output_dir = "./tmp_autoround"
# 可指定导出格式，支持auto_round（默认）、auto_gptq、auto_awq，多格式用英文逗号分隔
ar.quantize_and_save(output_dir, format="auto_gptq,auto_awq,auto_round")
```

#### 混合精度量化
自 0.8 版本起，AutoRound 提供了 AutoScheme 功能，可自动生成混合精度方案，详情请参阅 [Auto Scheme自动方案](#autoscheme)章节。

Auto-GPTQ 和 Auto-AWQ 仅支持有限的混合精度。如果不熟悉具体细节，建议**使用 AutoRound 原生格式**。

由于 vLLM 和 SGLang 框架会对 MoE 层、QKV 层进行融合以加速推理，所以**不建议给这些层设置不同的 bit **。

```python
from auto_round import AutoRound

model_name_or_path = "Qwen/Qwen3-0.6B"

# 层配置:支持完整层名匹配和模糊（部分）匹配
layer_config = {
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

#### AutoRoundBest 高精度配置用法
绝大多数场景下，该方案能实现最好的模型精度，缺点是训练耗时是基础方案的 4~5 倍；**特别适合 2-bit 量化**，若算力充足，可作为首选。

```python
from auto_round import AutoRound

model_name_or_path = "Qwen/Qwen3-0.6B"
ar = AutoRound(model=model_name_or_path, scheme="W4A16", nsamples=512, iters=1000, low_gpu_mem_usage=True)

output_dir = "./tmp_autoround"
ar.quantize_and_save(output_dir, format="auto_round")
```

#### AutoRoundLight 高速度配置用法
该方案训练速度最快（比基础方案快 2~3 倍），但小模型和 2-bit 量化下可能导致模型精度显著下降。所以**推荐在 4-bit 量化或参数量大于 3B 的模型的场景下使用**。

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

#### 配置方案推荐
综上所述，**4-bits（W4A16）推荐使用基础方案（auto-round），2-bits（W2A16）推荐使用高精度方案（auto-round-best）**；你也可根据实际需求和算力，灵活调整相关的配置。

<details>
  <summary>各配置方案详细参数</summary>

| 配置方案 | 批次大小 | 迭代次数 | 序列长度 | 标定样本数 | 学习率 |
|---------|----------|----------|----------|-------|--------|
| 基础版   | 8        | 200      | 2048     | 128   | 自动适配 |
| 高精度版 | 8        | 1000     | 2048     | 512   | 自动适配 |
| 高速版   | 8        | 50       | 2048     | 128   | 5e-3   |

</details>

W4G128 在 13 个任务上的平均精度与耗时
（测试环境：NVIDIA A100 80G，PyTorch 2.6.0，启用 enable_torch_compile）

| 模型                  | Qwen2.5-0.5B-Instruct | Falcon3-3B | Qwen2.5-7B-Instruct | Meta-Llama-3.1-8B-Instruct | Falcon3-10B | Qwen2.5-72B-Instruct |
|-----------------------|------------------------|------------|----------------------|-----------------------|-------------|-----------------------|
| 16位原精度            | 0.4192                 | 0.5203     | 0.6470               | 0.6212                | 0.6151      | 0.7229                |
| 高精度方案（Best）    | **0.4137**(7分钟)      | **0.5142**(23分钟) | 0.6426(58分钟)      | **0.6116**(65分钟)      | **0.6092**(81分钟) | 0.7242(575分钟)      |
| 基础方案（Default）   | 0.4129(2分钟)          | 0.5133(6分钟)  | **0.6441**(13分钟)  | 0.6106(13分钟)          | 0.6080(18分钟)  | **0.7252**(118分钟)  |
| 高速方案（Light）     | 0.4052(2分钟)          | 0.5108(3分钟)  | 0.6453(5分钟)       | 0.6104(6分钟)           | 0.6063(6分钟)   | 0.7243(37分钟)       |

<details>
  <summary>W2G64测试结果</summary>
W2G64 在 13 个任务上的平均精度与耗时
（测试环境：NVIDIA A100 80G，PyTorch 2.6.0，启用 enable_torch_compile）
为缓解模型精度显著下降，建议对 head、tail 以及非 expert 模块使用更高精度的量化配置。

| 模型                  | Qwen2.5-0.5B-Instruct | Falcon3-3B | Qwen2.5-7B-Instruct | Falcon3-10B | Qwen2.5-72B-Instruct |
|-----------------------|------------------------|------------|----------------------|-------------|-----------------------|
| 16位原精度            | 0.4192                 | 0.5203     | 0.6470               | 0.6151      | 0.7229                |
| 高精度方案（Best）    | **0.2989**(6分钟)      | **0.4267**(24分钟) | **0.5343**(56分钟)  | **0.5207**(79分钟) | **0.6715**(564分钟)  |
| 基础方案（Default）   | 0.2878(2分钟)          | 0.4219(6分钟)  | 0.5209(13分钟)      | 0.5133(18分钟)  | 0.6713(122分钟)      |
| 高速方案（Light）     | 0.2760(2分钟)          | 0.4063(3分钟)  | 0.4764(5分钟)       | 0.4810(7分钟)   | 0.6581(38分钟)       |

</details>

### AutoScheme 自动量化方案
AutoScheme 采用自动化算法，可生成 **自适应的混合精度与数据类型** 的量化方案（mixed bits/data type quantization recipes）。相关测试结果请参考[《AutoScheme精度报告》](./auto_scheme_acc.md)。

**注意**：混合数据类型方案在训练阶段可用，但当前版本还不支持导出至实际模型。

#### 命令行用法
训练时建议设置 `iters=200`
~~~bash
auto_round \
  --model_name  $model_name \
  --avg_bits 6 \
  --options "mxfp4,mxfp8" \
  --ignore_scale_zp_bits \
  --iters 0 \
  --format fake 
~~~

#### API 用法
~~~python
avg_bits= 3.0
scheme = AutoScheme(avg_bits=avg_bits, options=("W2A16G64", "W4A16","W8A16"))
ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1)
ar.quantize_and_save()
~~~


#### AutoScheme 超参数说明
`avg_bits(float)`：模型整体的目标平均 bits；计算时仅计入待量化的层。

`options(Union[str, list[Union[QuantizationScheme, str]]])`：候选量化配置集合。支持以下表示形式：单个用逗号分隔的字符串（例如 `"W4A16,W2A16"`​）、字符串列表（例如 `["W4A16", "W2A16"]`​）和 `QuantizationScheme` 。

`ignore_scale_zp_bits(bool)`：仅支持 API 调用场景。用于决定在计算平均 bit 时，是否忽略 scale 与 zero-point 的位数（默认 `False`）。

`device_map (Optional[str,dict,torch.device])`：仅支持 API 场景。由于 AutoScheme 会比标准 AutoRound 会占用更多显存，故可通过此参数为其指定不同的设备映射。

`shared_layers (Optional[Iterable[Iterable[str]]])`：仅支持 API 场景，用于定义多个层的分组，这些层将共享相同的量化配置。

`batch_size (Optional[int])`：设为 `1` 可以降低显存占用，但会增加训练时间。

`low_gpu_mem_usage(bool=True)`：开启后可减少 GPU 显存占用，但会增加训练时间。默认开启。

为加速推理，在部分框架中，会对特定层（如QKV、MoE）进行融合。这些融合层必须是相同的数据类型和量化配置。`shared_layers` 参数可以简化该配置，**同时支持正则表达式匹配和完整层名匹配**。注意**正则匹配按块匹配规则生效**。

示例代码如下：
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

此外，若需为特定的层固定量化方案，可使用 AutoRound API 中的`layer_config`参数，用法示例如下：
```python
from auto_round import AutoRound, AutoScheme

model_name = "Qwen/Qwen3-8B"
avg_bits = 3.0
scheme = AutoScheme(avg_bits=avg_bits, options=("GGUF:Q2_K_S", "GGUF:Q4_K_S"), ignore_scale_zp_bits=True)
layer_config = {"lm_head": "GGUF:Q6_K"}

ar = AutoRound(model=model_name, scheme=scheme, layer_config=layer_config, iters=0)
ar.quantize_and_save()
```

#### AutoScheme 耗时与显存成本
测试基于 Nvidia A100 80G、PyTorch 2.8。

后续我们会进一步优化显存占用。当前该方案的显存占用约为模型以 BF16 精度加载时的 1.1~1.5 倍。

| 模型            | 量化方案                | 显存占用 | 耗时                  |
|---------------| ----------------------- | -------- | --------------------- |
| Qwen3-8B      | W2A16 / W4A16 / W8A16   | 14G      | 60秒 × 可选方案数量   |
| Qwen3-8B      | MXFP4 / MXFP8           | 18G      | 60秒 × 可选方案数量   |
| Qwen3-8B      | GGUF系列                | 14G      | 80秒 × 可选方案数量   |
| Qwen3-32B     | W2A16 / W4A16 / W8A16   | 29G      | 180秒 × 可选方案数量  |
| Qwen3-32B     | MXFP4 / MXFP8           | 29G      | 180秒 × 可选方案数量  |
| Qwen3-32B     | GGUF系列                | 18G      | 300秒 × 可选方案数量  |
| Llama-3.3-70B | W2A16 / W4A16 / W8A16  | 32G      | 420秒 × 可选方案数量  |

<details>
<summary>关闭 low_gpu_mem_usage 后的结果</summary>

| 模型          | 量化方案              | 显存占用（开启torch compile） | 耗时（开启torch compile） | 显存占用（关闭torch compile） | 耗时（关闭torch compile） |
| ------------- | --------------------- | ----------------------------- | ------------------------- | ------------------------------ | ------------------------- |
| Qwen3-8B  | W2A16/W4A16/W8A16     | 34G                           | 30秒 × 可选方案数量       | 61G                            | 40秒 × 可选方案数量       |
| Qwen3-8B  | MXFP4/MXFP8           | 36G                           | 60秒 × 可选方案数量       | 54G                            | 120秒 × 可选方案数量      |
| Qwen3-8B  | GGUF系列              | 54G                           | 30秒 × 可选方案数量       | 50G                            | 23秒 × 可选方案数量       |
| Qwen3-32B | W2A16/W4A16/W8A16     | 240G显存溢出                   | ——                        | 240G显存溢出                   | ——                        |
| Qwen3-32B | MXFP4/MXFP8           | 160G                          | 200秒 × 可选方案数量      | 200G                           | 240秒 × 可选方案数量      |
| Qwen3-32B | GGUF系列              | 210G                          | 80秒 × 可选方案数量       | 200G                           | 60秒 × 可选方案数量       |

</details>

#### 局限性
AutoScheme 目前还**不支持对嵌入层（Embedding layer）进行自动量化**。该层将直接采用候选方案中精度最高的配置。

### OPT RTN 优化舍入模式
AutoRound 还提供优化版 RTN（Round-To-Nearest，就近舍入）模式，无需标定数据即可实现快速基线量化。**启用方式为 `iters=0`**。同时为获得更好的效果，推荐搭配 `group_size=32` 。RTN 与 OPT RTN 模式的精度对比详见[《精度对比报告》](./opt_rtn.md)。

对于 GGUF 格式，我们参考 llamacpp 的思路，优化了 RTN 算法。若需使用原始（非优化）RTN 算法，开启 `--disable_opt_rtn` 即可。
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

### GGUF 格式量化
实验性功能。该格式适用 CPU 设备，在社区应用广泛。

除 3-bits 外的各精度均建议使用优化版 RTN 模式（开启 `--iters 0` ）。

```python
from auto_round import AutoRound

model_name_or_path = "Qwen/Qwen3-0.6B"
ar = AutoRound(
    model=model_name_or_path,
)
output_dir = "./tmp_autoround"
ar.quantize_and_save(output_dir, format="gguf:q4_k_m")  # gguf:q*_k_s、gguf:q*_k_0、gguf:q*_k_1
```

### 量化成本
测试基于 Nvidia A100 80G、PyTorch 2.6.0.dev20241029+cu124。注意评测未计入数据加载和打包耗时。**建议在 PyTorch 2.6 及以上版本中开启 torch.compile 以加速**。

若要降低 GPU 显存占用，除开启`low_gpu_mem_usage`外，还可以设置`gradient_accumulate_steps=8`和`batch_size=1`，但这会增加训练耗时。

所有测试均基于 W4G128 量化方案。测试中使用的 3B 、14B 参数的模型为 Qwen 2.5系列，8X7B 的模型为 Mixtral，其余模型为 Llama-3.1 系列。


| Torch version/Config W4G128                                                      | 3B            | 8B             | 14B            | 70B             | 8X7B           |
|----------------------------------------------------------------------------------|---------------|----------------|----------------|-----------------|----------------|
| 2.6 + 开启 torch compile                                                           | 7min<br/>10GB | 12min<br/>18GB | 23min<br/>22GB | 120min<br/>42GB | 28min<br/>46GB |
| 2.6 + 开启 torch compile + 开启低显存模式<br/>low_gpu_mem_usage=True                      | 12min<br/>6GB | 19min<br/>10GB | 33min<br/>11GB | 140min<br/>25GB | 38min<br/>36GB |
| 2.6 + 开启 torch compile + 低显存模式 + 梯度累积8步、批次1<br/>gradient_accumulate_steps=8,bs=1 | 15min<br/>3GB | 25min<br/>6GB  | 45min<br/>7GB  | 187min<br/>19GB | 75min<br/>36GB |
| 2.5 + 关闭 torch compile                                                           | 8min<br/>10GB | 16min<br/>20GB | 30min<br/>25GB | 140min<br/>49GB | 50min<br/>49GB |



### 设备及多 GPU 量化设置
**AutoRound API 的 `device_map` 参数可指定训练设备（注意不是用 Transformers.from_pretrained 中的 `device_map` 参数）**。

AutoRound 采用 block（分块）逐块训练的方式处理模型，尽管单个 block 的规模远小于完整模型，但在训练过程中仍需要占用大量 GPU 显存（通常约为 block 大小的 10 倍）。在处理超大模型时，这可能会导致显存不足（OOM）错误。

也可通过调整相关参数降低显存占用，详情请参阅下文的[超参数调整](#超参数调整)章节，。

若调整参数后依旧无法解决显存问题，最简单的解决方法是在 `device_map` 中增加更多设备，示例如下：

Python API示例：
~~~python
from auto_round import AutoRound

model_name_or_path = "Qwen/Qwen3-0.6B"
ar = AutoRound(
    model=model_name_or_path,
    device_map="0,1,2,3"  # 指定使用0、1、2、3号GPU
)
~~~

命令行示例：
~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3 auto-round --model "Qwen/Qwen3-0.6B" --scheme "W4A16" --device_map "auto"
~~~


通常有两种情况需要启用多 GPU 训练：一是主要针对 lm-head 量化的校准阶段，二是参数量极大（如显存占用超 100GB）的模型。

#### lm_head 量化中开启多 GPU 校准
量化 lm-head 时，AutoRound 需要缓存其输入数据以进行高效的校准，这要求**整个模型驻留在 GPU 显存中** ；若 GPU 显存不足，部分层会退回至 RTN 模式。

<a id="llm-head-multi-gpu"></a>
#### 手动配置设备映射
<details>
<summary>自定义设备映射（device_map）</summary>
若 `device_map=auto` 未能正确分配，还可以通过 AutoRound API 的 `device_map` ，将同一 bolck 内的不同层映射到不同设备。这里我们提供一个参考示例：如何在 5 张 80GB GPU 上量化 DeepSeekV3-BF16（1.4T）模型。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "opensourcerelease/DeepSeek-R1-bf16"

# 加载分词器和模型
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

### 超参数调整
#### 降低 GPU 显存占用
以下方法可单独或组合使用，其中部分方式会增加训练耗时或带来轻微的精度损失：
- 将 `enable_torch_compile` 设为 True（开启 PyTorch 编译加速，不损失精度）
- 开启 `low_gpu_mem_usage`（低显存模式，**增加训练耗时**）
- 设置 `--bs 1 --gradient_accumulate_steps 8`（批次1+梯度累积8步，**增加训练耗时**）
- 将 `bs` 降至 4（**可能会有轻微的损失精度**）
- 将 `seqlen` 降至 512（**可能会有损失精度**）

#### 降低 CPU 内存占用
- 开启 `low_cpu_mem_usage`（实验性功能）：仅支持**导出指定一种格式**。每个 block 量化封装完成后会立即保存，从而降低峰值内存占用。
- 触发立即封装：使用命令行或 `quantize_and_save` API 时，只要指定**单一导出格式**，就会自动触发即时打包，无需额外配置。

#### 提升训练速度
以下方法可单独或组合使用，其中部分方式可能带来精度损失：
- 将 `enable_torch_compile` 设为 True（无精度损失）
- 使用 `auto-round-light` （小模型/ 2-bits 场景可能有明显精度损失）
- 将 `seqlen` 降至 512（**部分场景可能出现大幅精度损失**）
- 将 `bs` 降至 4（**仅有轻微精度损失**）

#### 开启 lm-head 层量化
该配置目前**仅支持 AutoRound 原生格式的推理**，命令行启用方式如下：
```bash
auto-round --model_name Qwen/Qwen3-0.6B  --scheme "W4A16" --quant_lm_head --format "auto_round"
```

#### 使用 AdamW 优化器
添加 `--adam` 参数即可启用；**注意**：在我们的多项测试场景中，AdamW 优化器的效果均不如符号梯度下降（sign gradient descent）。



## 4 推理部署
AutoRound 支持十余种推理后端，并会根据已安装的库自动选择最优后端；如果检测到系统中存在更优后端但缺少相关依赖，也会主动提示用户安装。

​**请勿在推理过程中手动将量化后的模型迁移到其他设备**​（例如执行 `model.to('cpu')`），否则可能导致意外错误。

### CPU
支持 2、4、8 bits 模型，其中 **4-bits 推理推荐搭配 intel-extension-for-pytorch（IPEX）** 使用，示例：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "OPEA/Qwen2.5-1.5B-Instruct-int4-sym-inc"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "有一个喜欢冒险的女孩，"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
```

### 英特尔 GPU
**仅支持 4-bits 模型**，推荐搭配 **IPEX** 使用，示例：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "OPEA/Qwen2.5-1.5B-Instruct-int4-sym-inc"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="xpu", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "有一个喜欢冒险的女孩，"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
```

### CUDA
支持 2、3、4、8 bits 量化模型，**4-bits/8-bits 推理推荐使用 GPTQModel 后端**，使用示例：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "OPEA/Qwen2.5-1.5B-Instruct-int4-sym-inc"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "有一个喜欢冒险的女孩，"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
```

### HPU
推荐使用集成 Habana 软件栈的 Docker 镜像，详情请参阅[Habana官方指南](https://docs.habana.ai/en/latest/)，使用示例：
```python
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Intel/Qwen2-7B-int4-inc"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("hpu").to(torch.bfloat16)

text = "有一个喜欢冒险的女孩，"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
```

### 指定推理后端
AutoRound 会根据兼容性为每个层自动选择推理后端，默认优先级大致为：Marlin > ExLLaMAV2 > Triton。最终的选择会受到 **group size、bit width、packing format、hardware device** 等多种因素的影响。

默认选择的后端并非在所有设备上都是最优的，你可根据需求或硬件兼容性手动指定后端：
- CPU/英特尔 GPU：推荐`ipex`
- CUDA：可指定`marlin`/`exllamav2`/`triton`

**注意**：手动指定后端的话，可能要安装相关依赖。

指定后端的示例：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoRoundConfig

model_name = "OPEA/Qwen2.5-1.5B-Instruct-int4-sym-inc"
quantization_config = AutoRoundConfig(backend="ipex")
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="cpu", quantization_config=quantization_config, torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "有一个喜欢冒险的女孩，"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
```

#### 各推理后端的详细支持说明
| 后端名称                | 支持设备       | 支持位宽       | 支持数据类型 | 优先级 | 打包格式        | 依赖库要求                     |
|-------------------------|----------------|----------------|--------------|--------|-----------------|--------------------------------|
| ipex                    | cpu/xpu        | 4              | BF16/FP16    | 5      | gptq_zp+-1/awq  | intel-extension-for-pytorch    |
| marlin                  | cuda           | 4、8           | BF16/FP16    | 6      | gptq/gptq_zp+-1 | gptqmodel                      |
| exllamav2/<br/>gptqmodel:exllamav2 | cuda    | 4              | BF16/FP16    | 5      | gptq            | gptqmodel                      |
| exllamav2/<br/>gptq:exllamav2      | cuda    | 4              | FP16         | 5      | gptq_zp+-1      | auto-gptq                      |
| gptq:cuda               | cuda           | 2、3、4、8     | FP16         | 1      | gptq_zp+-1      | auto-gptq                      |
| triton                  | xpu/cuda       | 2、4、8        | BF16/FP16    | 2      | gptq/gptq_zp+-1 | auto-round                     |
| awq                     | cuda           | 4              | FP16         | 5      | awq             | auto-awq                       |
| hpu                     | hpu            | 4              | BF16         | 0      | gptq/gptq_zp+-1 | auto-round                     |
| torch                   | xpu/cpu/cuda   | 2、3、4、8     | BF16/FP16    | 0      | gptq/gptq_zp+-1 | auto-round                     |

### 将 GPTQ 或 AWQ 模型转换为 AutoRound 格式
为了提升兼容性（尤其是英特尔设备），大部分 GPTQ/AWQ 量化模型均可转换为 AutoRound 格式。**注意**：若模型经过序列化处理，其量化配置可能会发生变更。

转换并推理的示例：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoRoundConfig

model_name = "ybelkada/opt-125m-gptq-4bit"
quantization_config = AutoRoundConfig()
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="cpu", quantization_config=quantization_config, torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "有一个喜欢冒险的女孩，"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
```



## 5 效果评估

AutoRound 借助 [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) 来评估模型。如果没有指定评估任务（`--task`），会使用默认的任务集（十余个常用评测任务）。

### 单 GPU 评估

**HF 后端（默认）:**
```bash
auto-round --model Qwen/Qwen3-0.6B --bits 4 --format "auto_round,auto_gptq" --tasks mmlu
```

**vLLM 后端:**
```bash
auto-round --model Qwen/Qwen3-0.6B --bits 4 --format "auto_round,auto_gptq" --tasks mmlu --eval_backend vllm
```

### 多 GPU 评估

**HF 后端:**
```bash
auto-round --model="your_model_path" --eval --device_map 0,1 --tasks lambada_openai --eval_bs 16
```

**vLLM 后端（用法一：用 `--device_map` 参数）**
```bash
auto-round "your_model_path" --eval --device_map 0,1 --tasks lambada_openai --eval_backend vllm
```

**vLLM 后端（用法二：手动配置）:**
```bash
CUDA_VISIBLE_DEVICES=0,1 auto-round "your_model_path" --eval --tasks lambada_openai --eval_backend vllm --vllm_args="tensor_parallel_size=2,gpu_memory_utilization=0.8"
```

### 注意事项

- 对于原始模型和量化后的模型，都支持用 `--eval` 参数直接评估。
- 为应对部分任务运行失败的情况，可使用 `--eval_task_by_task` 参数，按顺序执行评测任务（该参数目前只适用于 HF 后端）。
- 若导出了多种格式，会自动选用列表中的**最后一种格式**的模型评估。
- 对于 vLLM 后端，可通过 `--device 0,1,2` 指定 GPU 设备。该参数会自动设置 `CUDA_VISIBLE_DEVICES`，并根据设备数量配置 `tensor_parallel_size` 。此外，也支持通过环境变量和 `--vllm_args` 参数进行手动设置。


## 6 已知问题
量化过程存在的随机性可能会影响到部分模型的训练效果。若要保证实验结果可复现，可开启确定性算法（ `enable_deterministic_algorithms=True` ）。

部分视觉语言模型（VLM）需要手动适配。

暂不支持 Mamba 架构的模型。
