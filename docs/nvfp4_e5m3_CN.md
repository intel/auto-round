# NVFP4 E5M3 量化（实验性功能）

> **注意**：这是一个实验性功能，后续版本中 API、格式及行为可能发生变化。

## 概述

NVFP4 E5M3 是一种量化方案，以 FP4（E2M1）格式存储权重，并使用**无符号 E5M3 块级 scale**（区别于标准 NVFP4 使用全局 scale 的方式）。激活值通过 QDQ（量化-反量化）操作在推理时动态量化。

主要参数：

| 参数 | 值 |
|---|---|
| 预设名称 | `NVFP4_E5M3` |
| 权重数据类型 | FP4 E2M1（`nvfp4_v2`） |
| Scale 数据类型 | UE5M3（块级别） |
| 分组大小 | 16 |
| 激活量化位宽 | 4（动态 QDQ） |

---

## 支持的输出格式

| 格式标志 | 说明 |
|---|---|
| `auto_round`（默认） | 打包的 `.weight_packed` + `.weight_scale` 张量，可使用 AutoRound 推理加载 |
| `llm_compressor` | `nvfp4-e5m3-pack-quantized` compressed-tensors 布局 |
| `fake` | 全精度（BF16）`.weight` 张量，推理时仅做激活 QDQ，无打包开销 |

---

## 支持的推理后端

| 后端 | 说明 |
|---|---|
| `auto_round:cute_nvfp4_e5m3` | CUDA CuTe DSL 内核（权重反量化 + 激活 QDQ），需要 `cutlass` 包及 SM80+，**速度最快** |
| `auto_round:torch_nvfp4_e5m3` | 纯 PyTorch 反量化 + 激活 QDQ，适用于任意 CUDA 设备 |
| `auto_round:fake` | BF16 权重 + 仅激活 QDQ（无权重打包），用于调试 / 性能分析 |

---

## 使用方法

### 标准 AutoRound 流程

```python
from auto_round import AutoRound
from auto_round.schemes import NVFP4_E5M3

model, tokenizer = ...  # 加载你的模型

autoround = AutoRound(model, tokenizer, scheme=NVFP4_E5M3)
autoround.quantize()

# 以 auto_round 打包格式保存（默认）
autoround.save_quantized("output_dir")

# 或以 llm_compressor 格式保存
autoround.save_quantized("output_dir", format="llm_compressor")

# 或以 fake 格式保存（全精度权重 + 推理时 QDQ）
autoround.save_quantized("output_dir", format="fake")
```

### 无模型量化（无需 GPU 训练）

```bash
# 对已有模型进行直接打包（来自 HuggingFace 或本地目录）
auto-round \
  --model <model_id_or_path> \
  --scheme NVFP4_E5M3 \
  --model_free \
  --output_dir ./nvfp4_e5m3_model
```

```python
from auto_round.compressors.model_free import ModelFreeCompressor
from auto_round.schemes import NVFP4_E5M3

compressor = ModelFreeCompressor(
    model="<model_id_or_path>",
    scheme=NVFP4_E5M3,
    output_dir="./nvfp4_e5m3_model",
)
compressor.run()
```

### 加载量化后的模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import auto_round  # 注册 AutoRound 推理钩子

model = AutoModelForCausalLM.from_pretrained(
    "./nvfp4_e5m3_model",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("./nvfp4_e5m3_model")
```

---

## 环境变量

### `AR_NVFP4_E5M3_CACHE_HP_WEIGHT`

控制 NVFP4 E5M3 `QuantLinear` 是否在首次前向传播后缓存反量化的高精度（BF16）权重，而不是每次调用时重新反量化。

| 取值 | 行为 |
|---|---|
| `0` / `false` / `no` / `off`（默认） | 每次前向传播都进行反量化（内存更少，计算量更多） |
| `1` / `true` / `yes` / `on` | 首次前向后缓存反量化权重（计算量更少，内存占用更多） |

启用缓存后，打包权重缓冲区在缓存生成后会被释放，以较高的稳态内存换取较低的运行时开销。

```bash
export AR_NVFP4_E5M3_CACHE_HP_WEIGHT=1
```

---

## 精度结果（Llama-3.1-8B-Instruct，LAMBADA OpenAI）

| 配置 | 耗时 | 准确率 |
|---|---|---|
| BF16 基线 | 1m 5s | 0.7213 |
| `auto_round` + `auto_round:cute_nvfp4_e5m3` | 1m 48s | 0.7196 |
| `auto_round` + `auto_round:torch_nvfp4_e5m3` | 4m 23s | 0.7204 |
| `llm_compressor` + `auto_round:cute_nvfp4_e5m3` | 1m 48s | 0.7196 |
| `fake` + `auto_round:fake` | 3m 26s | 0.7204 |
