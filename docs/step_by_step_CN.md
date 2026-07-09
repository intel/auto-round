操作指南
============

[English](./step_by_step.md) | 简体中文

本文档介绍了如何用 auto-round 量化大语言模型（LLM）。如需量化视觉大语言模型（VLM），请参阅[视觉大语言模型用户指南](../auto_round/compressors/mllm/README.md)；如需量化扩散模型，请参阅[扩散模型用户指南](../auto_round/compressors/diffusion/README.md)。

* [1 安装必要库](#1-安装必要库)
* [2 准备标定数据集](#2-准备标定数据集)
  + [默认数据集](#默认数据集)
  + [自定义数据集](#自定义数据集)
  + [数据集操作](#数据集操作)
* [3 模型量化](#3-模型量化)
  + [可选量化配置](#可选量化配置)
  + [支持的导出格式](#支持的导出格式)
  + [硬件兼容性](#硬件兼容性)
  + [环境参数配置](#环境参数配置)
  + [命令行使用方法](#命令行用法)
  + [API 使用方法](#api-使用方法)
    - [AutoRound API 基础用法](#autoround-api-基础用法)
    - [混合精度量化方案](#混合精度量化)
    - [AutoRoundBest 配置方案](#autoroundbest-高精度配置用法)
    - [AutoRoundLight 配置方案](#autoroundlight-高速度配置用法)
    - [超参方案推荐](#超参方案推荐)
  + [AutoScheme 自动混合精度量化方案](#autoscheme-自动混合精度量化方案)
    - [命令行用法](#命令行用法-1)
    - [API 用法](#api-用法)
    - [AutoScheme 中的超参数](#autoscheme-超参数说明)
  + [OPT RTN 模式](#opt-rtn-模式)
  + [AWQ 算法-实验性功能](#awq-算法)
  + [免模型架构量化模式](#免模型架构量化模式)
  + [GGUF 格式](#gguf-格式量化)
  + [量化成本](#量化成本)
  + [设备及多卡量化设置](#设备及多卡量化设置)
    - [lm_head 量化中开启多 GPU 标定](#lm_head-量化中开启多-gpu-标定)
    - [手动配置设备映射](#手动配置设备映射)
  + [超参数调整](#超参数调整)
  + [旋转（Rotation）（实验性）](#旋转rotation实验性)
* [4 推理部署](#4-推理部署)
  + [CPU](#cpu)
  + [英特尔 GPU](#英特尔-gpu)
  + [CUDA](#cuda)
  + [HPU](#hpu)
  + [指定推理后端](#指定推理后端)
  + [将 GPTQ/AWQ 模型转换为 AutoRound 格式](#将-gptq-或-awq-模型转换为-autoround-格式)
* [5 效果评估](#5-效果评估)
  + [单卡评估](#单-gpu-评估)
  + [多卡评估](#多-gpu-评估)
  + [注意事项](#注意事项)
* [6 已知问题](#6-已知问题)

## 1 安装必要库

请执行下面的指令安装 auto-round 库（或从源码编译安装）

```bash
pip install auto-round
```

## 2 准备标定数据集

### 默认数据集
**对于中国大陆用户推荐使用 ModelScope 中的 swift/pile-val-backup 以解决 Huggingface 不能访问的问题**

默认标定数据集为 Hugging Face 上的 [NeelNanda/pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k) ，该数据集会自动从 Huggingface Hub 下载。同时也支持使用以下数据集：
- ModelScope 中的 `swift/pile-val-backup`：用于解决 HF 访问问题
- `BAAI/CCI3-HQ`：用于中文场景
- `codeparrot/github-code-clean`：用于代码场景
- `HuggingFaceH4/ultrachat_200k`：用于对话数据
- `madao33/new-title-chinese`：用于中文场景
- `mbpp`：用于代码场景
- `openbmb/Ultra-FineWeb`

### 自定义数据集
**建议用户还是尽量不使用 padding 的数据**。虽然对于 padding 过的数据我们有做特殊处理，但是目前验证的比较少。
可通过以下方式指定：
- 用法一：向 `dataset` 参数传入本地 JSON 文件路径
- 用法二：参照[示例代码](../auto_round/calib_dataset.py)注册数据集，然后使用新的数据集初始化 AutoRound 对象。示例： `autoround=Autoround(dataset="NeelNanda/pile-10k:train", ...)`
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


## 3 模型量化

### 可选量化配置

AutoRound 支持多种量化配置：
- **W4A16**（bits:4, group_size:128, sym:True, act_bits:16）  # 4位权重，分组大小为128，对称量化，16位激活，
- **W8A16**（bits:8, group_size:128, sym:True, act_bits:16）  
- **W6A16**（bits:6, group_size:128, sym:True, act_bits:16） — 仅 `mlx` 格式支持
- **W5A16**（bits:5, group_size:128, sym:True, act_bits:16） — 仅 `mlx` 格式支持
- **W3A16**（bits:3, group_size:128, sym:True, act_bits:16）  
- **W2A16**（bits:2, group_size:128, sym:True, act_bits:16）  
- **GGUF:Q4_K_M**（支持 llamacpp 提供的所有 Q*_K、Q*_0、Q*_1 量化类型）
- **混合bit**: （实验性功能）请使用 AutoScheme 接口或者使用 API 中的 `layer_config` 参数自己自定义
- **NVFP4**（实验性功能）推荐导出为`llm_compressor`格式，参数：data_type=nvfp4, act_data_type=nvfp4, static_global_scale, group_size=16
- **MXFP4**（研究性功能，暂无实际内核）：标准 MXFP4 量化，参数：data_type=mxfp, act_data_type=mxfp, bits=4, act_bits=4, group_size=32
- **MXINT4**（研究性功能，暂无实际内核）：标准 MXINT4 量化，参数：data_type=mxint, act_data_type=mxint, bits=4, act_bits=4, group_size=32
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

**LLM-Compressor 格式**：**支持 NVFP4、MXFP4（kernel 开发中）、MXFP8** 等。需设置 `--format llm_compressor`。

**MLX 格式(实验性功能)**：面向 Apple Silicon (M1/M2/M3/...)，可直接被 [`mlx-lm`](https://github.com/ml-explore/mlx-lm)（纯文本 LLM）或 [`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm)（多模态 VLM）加载推理。
- 支持 **2、3、4、5、6、8 bits**（其中 5/6 bits 是 MLX 独有，GPTQ/AWQ 没有标准打包格式）。
- 原生支持 **混合 bit / 混合 group_size**：通过 `layer_config` 或 AutoScheme（如 `--target_bits 3.5 --options "..."`），按层覆盖会写入 `config.json["quantization"]`，
- `--format mlx` 导出原生 MLX checkpoint；`--format auto_round:mlx` 则让 HuggingFace `transformers` + AutoRound 加载它（在 Darwin 上 post-init 会把每层重新打包成 MLX 的 `QuantLinear`）。
- 已经问题: 没有支持嵌入层的量化

#### 格式与方案支持对照表

> 灰色背景的 schemes 表示它没有专门优化的内核，或只有效率极低的参考内核。

| 格式                              | 支持的量化方案                                                                                                                                                                                                 |
|:--------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **auto_round**                  | W4A16、W2A16、W3A16、W8A16、W2A16G64、W2A16G32、`MXFP4`、`MXFP8`、`MXFP4_RCEIL`、`MXFP8_RCEIL`、`NVFP4`、`FPW8A16`、`FP8_STATIC`、`FP8_BLOCK`、`BF16`, `MXINT4`                                                               |
| **auto_awq**                    | W4A16、BF16                                                                                                                                                                                                   |
| **auto_gptq**                   | W4A16、W2A16、W3A16、W8A16、W2A16G64、W2A16G32、BF16                                                                                                                                                           |
| **llm_compressor**              | NVFP4、`MXFP4`、`MXFP8`、`FPW8A16`、`FP8_STATIC`、FP8_BLOCK                                                                                                                                                              |
| **mlx** / **auto_round:mlx** (实验性功能) | W2A16、W3A16、W4A16、W5A16、W6A16、W8A16、BF16、混合 bit / 混合 group_size（仅 Apple Silicon）                                                                                                                  |
| **gguf**                        | GGUF:Q4_K_M、GGUF:Q2_K_S、GGUF:Q3_K_S、GGUF:Q3_K_M、GGUF:Q3_K_L、GGUF:Q4_K_S、GGUF:Q5_K_S、GGUF:Q5_K_M、GGUF:Q6_K、GGUF:Q4_0、GGUF:Q4_1、GGUF:Q5_0、GGUF:Q5_1、GGUF:Q8_0                                           |
| **fp8**                         | FP8_BLOCK  |
| **fake**                        | `所有方案（仅用于研究场景）`                                                                                                                                                                                   |

### 硬件兼容性

量化和推理均支持 CPU、英特尔 GPU、HPU 和 CUDA。**MLX 格式**的推理仅支持 **Apple Silicon (macOS / Darwin)**，但量化（导出）阶段在任意平台均可进行。

### 环境参数配置

为优化运行性能，量化前建议配置 AutoRound 的环境变量。关于日志级别、ModelScope 集成、工作区设置等可用的环境变量等更多细节，可参考[环境变量指南](./environments.md)。

### 命令行用法


- **AutoRound 默认超参**：
  
  该方案很好地兼顾了精度和训练耗时，**推荐在绝大多数场景下使用**。

  ```bash
  auto-round --model Qwen/Qwen3-0.6B  --scheme "W4A16"  --format "auto_gptq,auto_awq,auto_round"
  ```

- **AutoRoundBest 高精度超参**：
  
  绝大多数场景下，该方案能实现最好的模型精度，缺点是训练耗时是基础方案的 4~5 倍；**特别适合 2-bit 量化**，若算力充足，可作为首选。跟默认参数的区别是调整了样本数量，从128提高了512.另外迭代次数从200次提高了1000.
  
  ```bash
  auto-round-best --model Qwen/Qwen3-0.6B  --scheme "W4A16"  --format "auto_gptq,auto_awq,auto_round"
  ```

- **AutoRoundLight 轻量级超参**：
  
  该方案训练速度最快（比基础方案快 2~3 倍），但小模型和 2-bit 量化下可能导致模型精度显著下降。所以**推荐在 4-bit 量化或参数量大于 3B 的模型的场景下使用**。
  
  ```bash
  auto-round-light --model Qwen/Qwen3-0.6B  --scheme "W4A16"  --format "auto_gptq,auto_awq,auto_round"
  ```

- **AutoRoundOptRTN 优化 RTN 超参（无需梯度下降）**：

  该方案启用优化版 RTN（Round-To-Nearest，就近舍入）模式（`iters=0` 且 `disable_opt_rtn=False`）。**无需标定数据**，相比基础方案快数倍，同时仍会执行 AutoRound 在 RTN 侧的优化（如更优的 scale/zero-point 搜索、GGUF 中借鉴 llamacpp 的优化等）。**推荐在标定数据不足或调优时间有限的场景下作为快速基线使用**。详情参见 [OPT-RTN 模式](#opt-rtn-模式)章节。

  ```bash
  auto-round-opt-rtn --model Qwen/Qwen3-0.6B  --scheme "W4A16"  --format "auto_round"
  ```

- **AutoRoundRTN 原始 RTN 超参（不做任何优化，不需要任何数据）**：

  该方案使用原始 RTN（`iters=0` 且 `disable_opt_rtn=True`），不启用任何 AutoRound 优化。**速度最快、显存占用最低**，但精度通常低于 `auto-round-opt-rtn`。当搭配受支持的 INT WOQ scheme 时，会自动路由至[免模型架构量化模式](#免模型架构量化模式)，进一步降低内存占用。适合用作快速验证或等价于传统 RTN 的无标定基线。

  ```bash
  auto-round-rtn --model Qwen/Qwen3-0.6B  --scheme "W4A16"  --format "auto_round"
  ```

### API 使用方法
#### AutoRound API 基础用法
该方案兼顾精度和训练耗时，**绝大多数场景下使用，2bit 等量化损失很大的场景尽量不要使用**。

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
自 0.8 版本起，AutoRound 提供了 AutoScheme 功能，可自动生成混合精度方案，详情请参阅 [Auto Scheme自动方案](#autoscheme-自动混合精度量化方案)章节。

Auto-GPTQ 和 Auto-AWQ 仅支持有限的混合精度。如果您不熟悉具体细节，**建议导出 AutoRound 格式**。

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

#### 超参方案推荐
综上所述，**4-bits（W4A16）推荐使用基础方案（auto-round），2-bits（W2A16）推荐使用高精度方案（auto-round-best）**；你也可根据实际需求和算力，灵活调整相关的配置。

<details>
  <summary>各配置方案详细参数</summary>

| 配置方案 | 批次大小 | 迭代次数 | 序列长度 | 标定样本数 | 学习率 | disable_opt_rtn |
|---------|----------|----------|----------|-------|--------|-----------------|
| 基础版（default）   | 8        | 200      | 2048     | 128   | 自动适配 | False           |
| 高精度版（best）    | 8        | 1000     | 2048     | 512   | 自动适配 | False           |
| 高速版（light）     | 8        | 50       | 2048     | 128   | 5e-3   | False           |
| 优化 RTN（opt_rtn） | 8        | 0        | 2048     | 128   | 自动适配 | False           |
| 原始 RTN（rtn）     | 8        | 0        | 2048     | 0     | 自动适配 | True            |

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

### AWQ 算法

实验性功能：原始实现中未使用 weight clipping（权重裁剪）逻辑，因此相比原版 AWQ 算法，可能会存在一定精度下降

AWQ（Activation-Aware Weight Quantization，激活感知权重量化）是一种可选的量化算法。AWQ 通过分析激活模式来保护关键权重通道，在标准量化前对权重施加通道级缩放，从而降低量化误差。

AWQ 的标准部署路径是 **W4A16**，通过 vLLM 的 AWQ/Marlin CUDA 内核提供服务。**W8A8** 搭配 AWQ 平滑化也可通过 vLLM 的 compressed_tensors 后端（cutlass INT8 GEMM）提供服务。

#### 命令行用法

```bash
auto-round --model Qwen/Qwen3-0.6B --scheme "W4A16" --algorithm awq --format "auto_round"
```

AWQ 专用选项：
- `--duo_scaling`：同时使用激活和权重计算缩放因子。选项：`true`、`false` 或 `both`（搜索两种模式并选择最佳）。（默认：True）。
- `--n_grid`：缩放比率搜索的网格点数（默认：20）。

#### API 用法

W8A8 搭配 AWQ 平滑化：

```python
from auto_round import AutoRound

ar = AutoRound(
    "Qwen/Qwen3-0.6B",
    scheme="INT8",
    algorithm="awq",
)

output_dir = "./tmp_awq"
ar.quantize_and_save(output_dir, format="auto_round:llm_compressor")
```


### AutoScheme 自动混合精度量化方案

AutoScheme 自动生成自适应的混合比特/混合数据类型量化方案。精度测试结果请参考 [AutoScheme 精度报告](./auto_scheme_acc.md)。

**说明：** 混合数据类型支持调优，但目前无法将其导出到实际模型中。

#### 命令行用法

- **`--iters 0`**：基于 RTN 的 量化方案，速度快（秒到分钟级）。
- **`--iters 200`**：调优感知的量化方案，更精确但慢很多。

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

**MoE 的 expert 层会自动按块分组** — 同一个 transformer block 内所有 expert 的投影层（gate/up/down，跨所有 experts）会被视为一个整体进行 DP 优化。它们共享相同的量化方案，loss 和 numel 直接求和。无需手动配置 `shared_layers` 来处理 expert 层。

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

当 AutoScheme 与 `model_free=True` 联合使用时，仅支持 INT（`W2A16`/`W4A16`/`W8A16`）和 MXFP（`MXFP4`/`MXFP8`）两种选项族。`W3A16`、`GGUF:*`、`NVFP4` 等不支持的选项会直接抛出 `ValueError`；同一 `AutoScheme` 中也不允许混用 INT 和 MXFP 选项族。

### AWQ 量化算法

AWQ（`algorithm="awq"`）是一种预处理量化算法，通过分析激活分布并应用通道缩放（channel-wise scaling）来保护重要的权重。它在实际量化（默认为 RTN，或使用 auto_round/SignRound）之前运行。

#### 命令行用法
```bash
# AWQ + 默认 RTN (自动选择 iters=0)
auto-round --model Qwen/Qwen3-0.6B --algorithm awq --scheme W4A16

# AWQ + AutoRound 优化
auto-round --model Qwen/Qwen3-0.6B --algorithm awq,auto_round --scheme W4A16

# AWQ 相关参数
--duo-scaling true|false|both  (默认: true)
--n-grid 20                    (默认: 20)
```

#### API 用法
```python
from auto_round import AutoRound
from auto_round.algorithms.quantization.awq.config import AWQConfig
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig

# AWQ + 默认 RTN (最简用法)
ar = AutoRound(model, tokenizer, algorithm="awq", scheme="W4A16")

# 通过 alg_configs 指定 AWQ + AutoRound (显式流水线)
ar = AutoRound(model, tokenizer, alg_configs=[AWQConfig(), SignRoundConfig(iters=200)], scheme="W4A16")
ar.quantize_and_save(output_dir="./qmodel")
```

**重要提示**：`algorithm="awq"`（量化算法）与 `format="auto_awq"`（导出格式）是相互独立的。你可以使用：
- `algorithm="awq"` + `format="auto_round"`：AWQ 平滑 + AutoRound 打包
- `algorithm="auto_round"` + `format="auto_awq"`：不使用 AWQ 平滑 + AutoAWQ 打包

### OPT-RTN 模式
AutoRound 还提供优化版 RTN（Round-To-Nearest，就近舍入）模式，无需标定数据即可实现快速基线量化。**启用方式为 `iters=0`**。同时为获得更好的效果，推荐搭配 `group_size=32` 。RTN 与 OPT RTN 模式的精度对比详见[《精度对比报告》](./opt_rtn.md)。

对于 GGUF 格式，我们参考 llamacpp 的思路，优化了 RTN 算法。若需使用原始（非优化）RTN 算法，开启 `--disable_opt_rtn` 即可。

#### 命令行使用

我们提供了两个专用的 CLI 入口作为快捷方式：

- `auto-round-opt-rtn` — 等价于 `auto-round --iters 0 --enable_opt_rtn`（优化版 RTN，推荐使用）。
- `auto-round-rtn` — 等价于 `auto-round --iters 0 --disable_opt_rtn`（原始 RTN，不做任何优化；对于受支持的 INT WOQ scheme 会自动路由到[免模型架构量化模式](#免模型架构量化模式)）。

```bash
# 优化版 RTN（推荐的快速基线方案）
auto-round-opt-rtn --model Qwen/Qwen3-0.6B --scheme "W4A16" --format "auto_round"

# 原始 RTN（速度最快、显存占用最低；仅作基线参考）
auto-round-rtn --model Qwen/Qwen3-0.6B --scheme "W4A16" --format "auto_round"
```

#### API 使用

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

### 免模型架构量化模式

免模型架构量化模式（Model-Free Mode）可以**无需将完整模型加载到内存中**即可执行 RTN WOQ 量化。它直接下载 safetensors 文件，逐分片地对每个 Linear 权重张量进行量化并保存打包结果。当您需要快速、无标定数据的量化且资源有限时，该模式非常实用。

> **默认自动启用。** 自 v0.13 起，当您同时传入 `--iters 0 --disable_opt_rtn` 与一个受支持的 INT WOQ 或 MXFP scheme 时，CLI 会自动走免模型路径。该路径与原始 `--iters 0 --disable_opt_rtn` 流程**位级（bit-exact）等价**，但内存占用大幅降低。如需关闭自动路由、强制使用原始流程，可加 `--disable_model_free`。

**主要特性：**
- **无需模型对象** — 仅需 `config.json` 和 safetensors 文件
- **低磁盘内存** (如果无本地模型) — 逐个下载并量化分片，处理完成后立即删除源分片
- **逐层配置** — 支持 `--layer_config` 设置逐层位宽，以及 `--ignore_layers` 保持特定层全精度
- **预定义忽略层** — 根据模型配置自动跳过特定层（如 MoE 门控层、MTP 层等）
- 与标准 `--iters 0 --disable_opt_rtn` 流程对所有受支持的 scheme **位级等价**
- **AutoScheme 集成** — 将 `AutoScheme` 对象传入 `scheme` 参数，即可在免模型模式下完成自动混合精度选择与逐分片打包（两阶段：短暂加载模型评分 → 释放模型 → 逐分片打包）

<details>
  <summary>Model-free 并行量化基准（分钟向上取整）</summary>

时间归一化规则：所有 `mm:ss` 均向上取整到分钟。例如：`4:20 -> 5`、`15:45 -> 16`、`9:07 -> 10`、`7:29 -> 8`、`4:09 -> 5`。

| 模型 | 设备 | 方案 | 并行度 | 峰值显存 (G) | 耗时（分钟，向上取整） |
|---|---|---|---:|---:|---:|
| Qwen/Qwen3-Next-80B-A3B-Instruct | A100 | W4A16 | 1 | 2 | N/A |
| Qwen/Qwen3-Next-80B-A3B-Instruct | A100 | W4A16 | 10 | 8 | 7 |
| Qwen3-235B-A22B-Instruct-2507 | A100 | W4A16 | 1 | 2 | 17 |
| Qwen3-235B-A22B-Instruct-2507 | A100 | W4A16 | 10 | 8 | 5 |
| zai-org/GLM-5.2 | B200 | MXFP4-Mixed | 1 | 2 | 60 |
| zai-org/GLM-5.2 | B200 | MXFP4-Mixed | 10 | 27 | 16 |
| zai-org/GLM-5.2 | B200 | W4A16 | 1 | 3 | 30 |
| zai-org/GLM-5.2 | B200 | W4A16 | 10 | 16 | 10 |
| zai-org/GLM-5.2 | B200 | W4A16 | 20 | 32 | 8 |
| MiniMaxAI/MiniMax-M2.7 (FP8) | B200 | W4A16 | 1 | 2 | 18 |
| MiniMaxAI/MiniMax-M2.7 (FP8) | B200 | W4A16 | 10 | 10 | 5 |
| deepseek-ai/DeepSeek-V4-Pro (MXFP) | B200 | W4A16 | 1 | 6 | 80 |
| deepseek-ai/DeepSeek-V4-Pro (MXFP) | B200 | W4A16 | 10 | 50 | 13 |

| 模型 | 方案 | 对比 | 耗时变化（分钟） | 加速比 | 时间节省 | 峰值显存变化 |
|---|---|---|---|---:|---:|---|
| Qwen3-235B | W4A16 | 并行 1 -> 10 | 17 -> 5 | 3.40x | 70.6% | 2G -> 8G |
| GLM-5.2 | MXFP4-Mixed | 并行 1 -> 10 | 60 -> 16 | 3.75x | 73.3% | 2G -> 27G |
| GLM-5.2 | W4A16 | 并行 1 -> 10 | 30 -> 10 | 3.00x | 66.7% | 3G -> 16G |
| GLM-5.2 | W4A16 | 并行 1 -> 20 | 30 -> 8 | 3.75x | 73.3% | 3G -> 32G |
| MiniMax-M2.7 | W4A16 | 并行 1 -> 10 | 18 -> 5 | 3.60x | 72.2% | 2G -> 10G |
| DeepSeek-V4-Pro | W4A16 | 并行 1 -> 10 | 80 -> 13 | 6.15x | 83.8% | 6G -> 50G |

结论：在 model-free 量化中，提高并行度通常可带来约 `3x-6x` 的耗时加速，但峰值显存会明显上升。

</details>

<details>
  <summary>点击展开支持的 Scheme 与示例</summary>

**支持的 Scheme**

免模型模式支持以下量化预设：

**整数权重量化**（使用 `auto_round:auto_gptq` 打包格式）：

| Preset | Bits | Group size | Sym |
| --- | --- | --- | --- |
| `W2A16` | 2 | 128 | true |
| `W2A16G32` | 2 | 32 | true |
| `W2A16G64` | 2 | 64 | true |
| `W4A16`（默认） | 4 | 128 | true |
| `W4A16_MIXED` | 4 | 128 | true |
| `W8A16` | 8 | 128 | true |

上述 2-bit 和 8-bit 预设（`W2A16`、`W2A16G32`、`W2A16G64`、`W8A16`）同样支持**非对称量化**（`sym=False`），输出使用 `auto_round:auto_gptq` 打包格式，并与标准流程**位级等价**。4-bit 非对称量化时标准流程建议使用 `auto_round:auto_awq` 打包格式，如需该场景请使用标准 AutoRound 流程。

也可以传入自定义的 `QuantizationScheme(bits=N, group_size=G, sym=True/False, data_type="int", act_bits=16)`，其中 `bits ∈ {2, 4, 8}`，group_size / sym 可任意设置。

**MXFP（微缩放浮点）**（使用 `mxfp4-pack-quantized` / `mxfp8-quantized` 格式，兼容 compressed-tensors / vLLM）：

| Preset | Bits | Group size | 格式 |
| --- | --- | --- | --- |
| `MXFP4` | 4 | 32 | mxfp4-pack-quantized |
| `MXFP8` | 8 | 32 | mxfp8-quantized |

需要特殊打包内核的 scheme（`W3A16`、`FPW8A16`、`BF16`、`MXINT4`、`NVFP4`、`FP8_BLOCK`、`FP8_STATIC`、`INT8_W8A8`、`GGUF:*` 等）**不被支持**，传入会抛 `ValueError`。这些请使用标准 AutoRound 流程。

#### 命令行用法

```bash
# 最简单：--iters 0 --disable_opt_rtn 自动路由到免模型
auto_round meta-llama/Llama-3.2-1B-Instruct \
  --scheme W4A16 \
  --iters 0 --disable_opt_rtn \
  --output_dir ./int4-llama

# 等价的显式调用
auto_round meta-llama/Llama-3.2-1B-Instruct \
  --model_free \
  --scheme W4A16 \
  --output_dir ./int4-llama

# 关闭自动路由，强制使用原始流程
auto_round meta-llama/Llama-3.2-1B-Instruct \
  --scheme W4A16 \
  --iters 0 --disable_opt_rtn --disable_model_free \
  --output_dir ./int4-llama

# 搭配逐层配置和忽略层
auto_round meta-llama/Llama-3.2-1B-Instruct \
  --model_free \
  --scheme W4A16 \
  --group_size 32 \
  --asym \
  --layer_config "{k_proj:{bits:8},v_proj:{bits:8}}" \
  --ignore_layers "mlp" \
  --output_dir ./int4-llama

# MXFP4 量化
auto_round meta-llama/Llama-3.2-1B-Instruct \
  --model_free \
  --scheme MXFP4 \
  --output_dir ./mxfp4-llama

# MXFP8 量化
auto_round meta-llama/Llama-3.2-1B-Instruct \
  --model_free \
  --scheme MXFP8 \
  --output_dir ./mxfp8-llama
```

#### API 用法

```python
from auto_round import AutoRound

AutoRound(
    model="meta-llama/Llama-3.2-1B-Instruct",
    scheme="W4A16",  # 也支持 QuantizationScheme 对象自定义 group_size / sym
    layer_config={
        ".*k_proj": {"bits": 8, "group_size": 32},
        ".*v_proj": {"bits": 8, "group_size": 32},
    },
    ignore_layers="mlp",
    model_free=True,
).quantize_and_save("./int4-llama")
```

> **注意：** 免模型量化模式使用 RTN（无标定数据、无迭代调优）。INT scheme 输出为 `auto_round:auto_gptq` 格式；MXFP scheme 输出为 compressed-tensors 格式（`mxfp4-pack-quantized` / `mxfp8-quantized`）。如需更高质量的量化结果或使用受支持列表外的 scheme，请使用标准 AutoRound 流程。

</details>

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
该数据有点过时，目前量化显存会比下面表格报告的要少点。
测试基于 Nvidia A100 80G、PyTorch 2.6.0.dev20241029+cu124。注意评测未计入数据加载和打包耗时。**建议在 PyTorch 2.6 及以上版本中开启 torch.compile 以加速**。

若要降低 GPU 显存占用，除开启`low_gpu_mem_usage`外，还可以设置`gradient_accumulate_steps=8`和`batch_size=1`，但这会增加训练耗时。

所有测试均基于 W4G128 量化方案。测试中使用的 3B 、14B 参数的模型为 Qwen 2.5系列，8X7B 的模型为 Mixtral，其余模型为 Llama-3.1 系列。


| Torch version/Config W4G128                                                      | 3B            | 8B             | 14B            | 70B             | 8X7B           |
|----------------------------------------------------------------------------------|---------------|----------------|----------------|-----------------|----------------|
| 2.6 + 开启 torch compile                                                           | 7min<br/>10GB | 12min<br/>18GB | 23min<br/>22GB | 120min<br/>42GB | 28min<br/>46GB |
| 2.6 + 开启 torch compile + 开启低显存模式<br/>low_gpu_mem_usage=True                      | 12min<br/>6GB | 19min<br/>10GB | 33min<br/>11GB | 140min<br/>25GB | 38min<br/>36GB |
| 2.6 + 开启 torch compile + 低显存模式 + 梯度累积8步、批次1<br/>gradient_accumulate_steps=8,bs=1 | 15min<br/>3GB | 25min<br/>6GB  | 45min<br/>7GB  | 187min<br/>19GB | 75min<br/>36GB |
| 2.5 + 关闭 torch compile                                                           | 8min<br/>10GB | 16min<br/>20GB | 30min<br/>25GB | 140min<br/>49GB | 50min<br/>49GB |

W4G128 量化耗时与显存占用（英特尔 GPU B60 24G）
（测试环境：英特尔 GPU B60 24G，PyTorch 2.11.0+xpu 正式版。注意评测未计入数据加载和打包耗时。所有测试均使用 Qwen2.5 系列模型。）
| Torch version/Config W4G128                                                                                            | 0.5B              | 1.5B              | 3B                  | 7B                  |
|------------------------------------------------------------------------------------------------------------------------|-------------------|-------------------|---------------------|---------------------|
| 2.11.0+xpu 开启 torch compile                                                                                          | 6min<br/>2.9GB    | 13min<br/>5.4GB   | 22min<br/>7.1GB     | 40min<br/>14.9GB    |
| 2.11.0+xpu 开启 torch compile<br/>low_gpu_mem_usage=True                                                               | 10min<br/>1.7GB   | 17min<br/>3.3GB   | 30min<br/>4.3GB     | 50min<br/>8.5GB     |
| 2.11.0+xpu 开启 torch compile<br/>low_gpu_mem_usage=True<br/>gradient_accumulate_steps=8,bs=1                          | 14min<br/>0.4GB   | 22min<br/>1.1GB   | 38min<br/>1.5GB     | 1h 4min<br/>4.1GB   |
| 2.11.0+xpu 关闭 torch compile                                                                                           | 6min<br/>2.9GB    | 14min<br/>5.7GB   | 26min<br/>7.6GB     | 51min<br/>15.5GB    |


### 设备及多卡量化设置
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


通常有两种情况需要启用多 GPU 训练：一是主要针对 lm-head 量化的标定阶段，二是参数量极大（如显存占用超 100GB）的模型。

#### lm_head 量化中开启多 GPU 标定
量化 lm-head 时，AutoRound 需要缓存其输入数据以进行高效的标定，这要求**整个模型驻留在 GPU 显存中** ；若 GPU 显存不足，部分层会退回至 Pure RTN 模式。

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
- 将 `bs` 降至 4（**可能会有轻微的精度损失**）
- 将 `seqlen` 降至 512（**可能会有精度损失**）

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

### 旋转（Rotation）（实验性）

> ⚠️ **实验性功能**：旋转变换仍处于实验阶段。推理依赖 forward hook 机制，目前仅支持 Hugging Face Transformers 后端，因此相比非旋转模型，旋转后的模型推理速度可能较慢。

旋转在量化前对权重和激活中的离群点进行重分布，使分布更加均匀、对量化更友好。它对 MXFP4、NVFP4、W4A4 等激进的低比特方案最为有效。

AutoRound 通过 `rotation_config` 参数应用旋转。推荐在大多数场景中使用 `"quarot"` 预设——确定性 Hadamard 旋转（QuaRot / SpinQuant），无需训练、无需校准数据。

#### API 用法

```python
from auto_round import AutoRound

model_name = "Qwen/Qwen3-0.6B"

# QuaRot 预设：确定性 Hadamard，无需训练
ar = AutoRound(model_name, scheme="MXFP4", rotation_config="quarot")
ar.quantize_and_save(output_dir="./Qwen3-0.6B-mxfp4-quarot", format="auto_round")
```

带旋转的量化模型可透明地保存和加载——旋转矩阵和 hook 会在加载时自动恢复，推理无需额外步骤。

关于旋转位置（R1–R4）、完整配置选项、确定性与随机 Hadamard、可训练的 SpinQuant、逐线性层块旋转变体以及保存/加载的内部细节，请参阅 [Rotation Details](./rotation_details.md)（英文）。


## 4 推理部署
AutoRound 支持十余种推理后端，并会根据已安装的库自动选择最优后端；如果检测到系统中存在更优后端但缺少相关依赖，也会主动提示用户安装。

​**请勿在推理过程中手动将量化后的模型迁移到其他设备**​（例如执行 `model.to('cpu')`），否则可能导致意外错误。

### CPU
支持 2、4、8 bits 模型，推荐搭配 **auto-round-lib（ark）**（需 PyTorch>=2.8.0）进行推理，示例：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "OPEA/Qwen2.5-1.5B-Instruct-int4-sym-inc"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
```

### 英特尔 GPU
支持 4、8 bits 模型，推荐搭配 **auto-round-lib（ark）**（需 PyTorch>=2.8.0）进行推理，示例：
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
支持 2、3、4、8 bits 量化模型，**4-bits/8-bits 推理推荐使用 GPTQModel 后端**，使用示例：
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
推荐使用集成 Habana 软件栈的 Docker 镜像，详情请参阅[Habana官方指南](https://docs.habana.ai/en/latest/)，使用示例：
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

### 指定推理后端
AutoRound 会根据兼容性为每个层自动选择推理后端，默认优先级大致为：Marlin > ExLLaMAV2 > Triton。最终的选择会受到 **group size、bit width、packing format、hardware device** 等多种因素的影响。

默认选择的后端并非在所有设备上都是最优的，你可根据需求或硬件兼容性手动指定后端：
- CPU/英特尔 GPU：推荐`ark`（需要 `auto-round-lib` 和 `torch>=2.8.0`）
- CUDA：可指定`marlin`/`exllamav2`/`triton`

**注意**：手动指定后端的话，可能要安装相关依赖。

指定后端的示例：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoRoundConfig

model_name = "OPEA/Qwen2.5-1.5B-Instruct-int4-sym-inc"
quantization_config = AutoRoundConfig(backend="ark")
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="cpu", quantization_config=quantization_config, torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
```

#### 各推理后端的详细支持说明
| 后端名称                | 支持设备       | 支持位宽       | 支持数据类型 | 优先级 | 打包格式        | 依赖库要求                     |
|-------------------------|----------------|----------------|--------------|--------|-----------------|--------------------------------|
| ark                     | cpu            | 2、4、8        | FP32/FP16/BF16 | 6    | gptq/gptq_zp+-1 | auto-round-lib<br/>torch>=2.8.0 |
| ark                     | cpu            | 4              | FP32/FP16/BF16 | 6    | awq             | auto-round-lib<br/>torch>=2.8.0 |
| ark                     | xpu            | 4、8           | FP32/FP16/BF16 | 6    | gptq/gptq_zp+-1 | auto-round-lib<br/>torch>=2.8.0 |
| ark                     | xpu            | 4              | FP32/FP16/BF16 | 6    | awq             | auto-round-lib<br/>torch>=2.8.0 |
| marlin                  | cuda           | 4、8           | BF16/FP16    | 6      | gptq/gptq_zp+-1 | gptqmodel                      |
| exllamav2/<br/>gptqmodel:exllamav2 | cuda    | 4              | BF16/FP16    | 5      | gptq/gptq_zp+-1 | gptqmodel                      |
| exllamav2/<br/>gptq:exllamav2      | cuda    | 4              | FP16         | 3      | gptq_zp+-1      | auto-gptq<br/>transformers<5.0.0  |
| gptq:cuda               | cuda           | 2、3、4、8     | FP16         | 1      | gptq_zp+-1      | auto-gptq<br/>transformers<5.0.0  |
| triton                  | xpu/cuda       | 2、4、8        | BF16/FP16    | 2      | gptq/gptq_zp+-1 | auto-round                     |
| awq                     | cuda           | 4              | FP16         | 5      | awq             | auto-awq<br/>transformers<4.57.0 |
| gptqmodel:awq/<br/>gptqmodel:awq_exllamav2 | cuda | 4         | BF16/FP16    | 6      | awq             | gptqmodel                      |
| gptqmodel:awq_marlin    | cuda           | 4、8           | FP16         | 5      | awq             | gptqmodel                      |
| gptqmodel:awq_gemm      | cuda           | 4              | FP16         | 3      | awq             | gptqmodel                      |
| gptqmodel:awq_torch     | cuda/cpu       | 4              | FP16         | 2      | awq             | gptqmodel                      |
| hpu                     | hpu            | 4              | BF16         | 0      | gptq/gptq_zp+-1 | auto-round                     |
| torch                   | xpu/cpu/cuda   | 2、3、4、8     | BF16/FP16    | 0      | gptq/gptq_zp+-1 | auto-round                     |

### 将 GPTQ 或 AWQ 模型转换为 AutoRound 格式
为了提升兼容性（尤其是英特尔设备），大部分 GPTQ/AWQ 量化模型均可转换为 AutoRound 格式。**注意**：若模型再次存储，其量化配置可能会发生变更， 由 gptq/awq 量化变化成 auto-round 量化。

转换并推理的示例：
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
