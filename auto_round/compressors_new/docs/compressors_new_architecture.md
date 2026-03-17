# Compressor New Architecture

## Overview

本文档介绍了 `compressors_new` 的新架构设计，该设计统一了 LLM、MLLM 和 Diffusion 模型的量化入口。

## 架构设计

### 核心思想

通过 `entry.py` 中的 `Compressor` 类作为统一入口，根据模型类型和算法配置动态选择合适的 Compressor 实现。

### 组件结构

```
compressors_new/
├── entry.py                # 统一入口，自动检测模型类型
├── base.py                 # BaseCompressor 基类
├── calib.py                # CalibCompressor (需要校准的算法)
├── zero_shot.py            # ZeroShotCompressor (不需要校准的算法)
├── mllm_mixin.py           # MLLMCalibCompressor (MLLM + 校准)
└── diffusion_mixin.py      # DiffusionCalibCompressor (Diffusion + 校准)
```

### 类继承关系

```
BaseCompressor
    ├── CalibCompressor (基于校准的压缩)
    │   ├── MLLMCalibCompressor (MLLM 专用)
    │   └── DiffusionCalibCompressor (Diffusion 专用)
    │
    └── ZeroShotCompressor (不需要校准)
```

## 使用方法

### 1. 基本用法

```python
from auto_round.algorithms.quantization.auto_round.config import AutoRoundConfig
from auto_round.compressors_new.entry import Compressor

# 创建配置
config = AutoRoundConfig(
    scheme="W4A16",
    iters=200,
    nsamples=128,
)

# 统一入口 - 自动检测模型类型
compressor = Compressor(
    config=config,
    model="/path/to/model",  # 可以是 LLM/MLLM/Diffusion
    tokenizer=tokenizer,
    platform="hf",
    format=None,
)

# 执行量化
quantized_model, layer_config = compressor.quantize()
```

### 2. MLLM 模型量化

```python
from auto_round.compressors_new.entry import Compressor
from auto_round.algorithms.quantization.auto_round.config import AutoRoundConfig

config = AutoRoundConfig(scheme="W4A16", iters=200)

# 会自动使用 MLLMCalibCompressor
compressor = Compressor(
    config=config,
    model="/models/Qwen2-VL-2B-Instruct",
    tokenizer=tokenizer,
    processor=processor,  # MLLM 特定参数
    image_processor=image_processor,  # MLLM 特定参数
    template="qwen2_vl",  # MLLM 特定参数
    extra_data_dir="/path/to/images",  # MLLM 特定参数
)

quantized_model, layer_config = compressor.quantize()
```

### 3. Diffusion 模型量化

```python
from auto_round.compressors_new.entry import Compressor
from auto_round.algorithms.quantization.auto_round.config import AutoRoundConfig

config = AutoRoundConfig(scheme="W4A16", iters=200)

# 会自动使用 DiffusionCalibCompressor
compressor = Compressor(
    config=config,
    model="/models/stable-diffusion-2-1",
    platform="hf",
    guidance_scale=7.5,  # Diffusion 特定参数
    num_inference_steps=50,  # Diffusion 特定参数
)

quantized_model, layer_config = compressor.quantize()
```

## 模型类型检测

`entry.py` 中的 `detect_model_type()` 函数自动检测模型类型:

```python
def detect_model_type(model):
    """检测模型类型

    Returns:
        "mllm" | "diffusion" | "llm"
    """
    if is_diffusion_model(model):
        return "diffusion"
    if is_mllm_model(model):
        return "mllm"
    return "llm"
```

检测逻辑:
1. 优先检测是否为 Diffusion 模型(检查 `model_index.json`)
2. 然后检测是否为 MLLM 模型(检查 `processor_config.json` 等)
3. 默认为标准 LLM 模型

## 动态 Compressor 选择

`entry.py` 中的 `Compressor.__new__()` 方法根据以下条件动态选择:

### 决策树

```
Compressor.__new__()
│
├── AutoRoundConfig (需要校准)
│   ├── MLLM → MLLMCalibCompressor
│   ├── Diffusion → DiffusionCalibCompressor
│   └── LLM → CalibCompressor
│
└── RTNConfig
    ├── enable_imatrix=True → ImatrixCompressor
    └── enable_imatrix=False → ZeroShotCompressor
```

## 扩展新模型类型

如果需要支持新的模型类型,按照以下步骤:

### 1. 创建专用 Compressor

```python
# compressors_new/new_model_calib.py
from auto_round.compressors_new.calib import CalibCompressor


class NewModelCalibCompressor(CalibCompressor):
    def __init__(self, config, model, **kwargs):
        # 存储模型特定参数
        self.special_param = kwargs.pop("special_param", None)
        super().__init__(config, model, **kwargs)

    @torch.no_grad()
    def calib(self, nsamples, bs):
        # 实现模型特定的校准逻辑
        # 通常需要:
        # 1. 加载模型特定的 dataloader
        # 2. 处理模型特定的数据格式
        # 3. 执行前向传播进行校准
        pass
```

### 2. 更新模型检测逻辑

```python
# 在 entry.py 的 detect_model_type() 中添加
def detect_model_type(model):
    if is_new_model_type(model):  # 添加新的检测函数
        return "new_model_type"
    if is_diffusion_model(model):
        return "diffusion"
    # ...
```

### 3. 更新 Compressor 入口

```python
# 在 entry.py 的 Compressor.__new__() 中添加
if isinstance(config, AutoRoundConfig):
    if model_type == "new_model_type":
        from auto_round.compressors_new.new_model_calib import NewModelCalibCompressor
        return NewModelCalibCompressor(config, **local_args, **kwargs)
    elif model_type == "mllm":
        # ...
```

## 与旧架构的兼容性

### 旧架构 (compressors/)

```python
from auto_round.compressors.mllm.compressor import MLLMCompressor

compressor = MLLMCompressor(
    model=model,
    # ... 参数
)
```

### 新架构 (compressors_new/)

```python
from auto_round.compressors_new.entry import Compressor
from auto_round.algorithms.quantization.auto_round.config import AutoRoundConfig

config = AutoRoundConfig(...)
compressor = Compressor(
    config=config,
    model=model,
    # ... 参数
)
```

**优势:**
1. 统一入口,无需手动选择 Compressor
2. 自动模型类型检测
3. 更好的代码组织和复用
4. 易于扩展新模型类型

## 实现细节

### MLLMCalibCompressor

重写的关键方法:
- `calib()`: 使用 MLLM 专用的 dataloader 和 template
- 处理 processor, image_processor, template 等 MLLM 特定参数

### DiffusionCalibCompressor

重写的关键方法:
- `post_init()`: 预先加载 diffusion pipeline
- `_load_diffusion_model()`: 加载 pipeline 并提取 transformer/unet
- `calib()`: 使用 diffusion 专用的 dataloader

### 数据流

```
1. Compressor.__new__()
   └── 检测模型类型
   └── 创建对应的 Compressor 实例

2. CompressorInstance.__init__()
   └── 存储模型特定参数
   └── 调用 super().__init__()

3. CompressorInstance.quantize()
   └── post_init()
       └── _load_model() (可能被重写)
   └── calib() (可能被重写)
   └── 执行量化算法

4. 返回量化后的模型
```

## 测试

运行测试脚本:

```bash
python test_compressor_new_arch.py
```

这将测试:
- 模型类型检测
- LLM Compressor 创建
- MLLM Compressor 创建
- Diffusion Compressor 创建

## 总结

新架构的主要优势:

1. **统一入口**: 一个 `Compressor` 类处理所有模型类型
2. **自动检测**: 无需手动判断模型类型
3. **易于扩展**: 添加新模型类型只需3步
4. **代码复用**: 通过继承复用基类功能
5. **清晰结构**: 每种模型类型有独立的 Compressor 实现

这种设计符合开闭原则(Open-Closed Principle),对扩展开放,对修改关闭。
