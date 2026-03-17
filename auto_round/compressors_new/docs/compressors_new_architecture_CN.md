# Compressor 新架构说明

## 概述

本文档介绍了 `compressors_new` 的新架构设计,实现了对 LLM、MLLM 和 Diffusion 模型的统一量化入口。

## 架构设计

### 核心思想

通过 `entry.py` 中的 `Compressor` 类作为统一入口点,根据模型类型和算法配置自动选择合适的 Compressor 实现类。

### 目录结构

```
compressors_new/
├── entry.py                # 统一入口,自动检测模型类型
├── base.py                 # BaseCompressor 基类
├── calib.py                # CalibCompessor (基于校准的压缩)
├── zero_shot.py            # ZeroShotCompressor (零样本压缩)
├── mllm_mixin.py           # MLLMCalibCompressor (多模态模型校准压缩)
└── diffusion_mixin.py      # DiffusionCalibCompressor (扩散模型校准压缩)
```

### 类继承关系

```
BaseCompressor (基础压缩器)
    │
    ├── CalibCompessor (基于校准的压缩器)
    │   │
    │   ├── MLLMCalibCompressor (多模态模型专用)
    │   │   └── 支持视觉-语言模型(如 Qwen2-VL, LLaVA 等)
    │   │
    │   └── DiffusionCalibCompressor (扩散模型专用)
    │       └── 支持文生图模型(如 Stable Diffusion, FLUX 等)
    │
    └── ZeroShotCompressor (零样本压缩器)
        └── 用于 RTN 等不需要校准的算法
```

## 使用方法

### 1. 基本用法(自动检测)

```python
from auto_round.algorithms.quantization.auto_round.config import AutoRoundConfig
from auto_round.compressors_new.entry import Compressor

# 创建量化配置
config = AutoRoundConfig(
    scheme="W4A16",  # 量化方案: 权重4比特,激活16比特
    iters=200,  # 迭代次数
    nsamples=128,  # 校准样本数
)

# 统一入口 - 自动检测模型类型并选择合适的 Compressor
compressor = Compressor(
    config=config,
    model="/path/to/model",  # 支持 LLM/MLLM/Diffusion 模型
    tokenizer=tokenizer,
    platform="hf",  # 平台: "hf" 或 "model_scope"
)

# 执行量化
quantized_model, layer_config = compressor.quantize()
```

### 2. MLLM 多模态模型量化

```python
from auto_round.compressors_new.entry import Compressor
from auto_round.algorithms.quantization.auto_round.config import AutoRoundConfig
from transformers import AutoProcessor, AutoTokenizer

# 准备 tokenizer 和 processor
tokenizer = AutoTokenizer.from_pretrained("/models/Qwen2-VL-2B-Instruct")
processor = AutoProcessor.from_pretrained("/models/Qwen2-VL-2B-Instruct")

config = AutoRoundConfig(scheme="W4A16", iters=200, nsamples=128)

# 自动使用 MLLMCalibCompressor
compressor = Compressor(
    config=config,
    model="/models/Qwen2-VL-2B-Instruct",
    tokenizer=tokenizer,
    processor=processor,  # MLLM 特定: 多模态处理器
    image_processor=None,  # MLLM 特定: 图像处理器
    template="qwen2_vl",  # MLLM 特定: 模板名称
    extra_data_dir="/path/to/images",  # MLLM 特定: 额外数据路径
    quant_nontext_module=False,  # 是否量化非文本模块
)

quantized_model, layer_config = compressor.quantize()
```

### 3. Diffusion 扩散模型量化

```python
from auto_round.compressors_new.entry import Compressor
from auto_round.algorithms.quantization.auto_round.config import AutoRoundConfig

config = AutoRoundConfig(scheme="W4A16", iters=200, nsamples=128)

# 自动使用 DiffusionCalibCompressor
compressor = Compressor(
    config=config,
    model="/models/stable-diffusion-2-1",
    platform="hf",
    guidance_scale=7.5,  # Diffusion 特定: 引导强度
    num_inference_steps=50,  # Diffusion 特定: 推理步数
    generator_seed=42,  # Diffusion 特定: 随机种子
    dataset="coco2014",  # 校准数据集
)

quantized_model, layer_config = compressor.quantize()
```

### 4. RTN 量化(零样本)

```python
from auto_round.compressors_new.entry import Compressor
from auto_round.algorithms.quantization.rtn.config import RTNConfig

# RTN 不需要校准数据
config = RTNConfig(scheme="W4A16")

# 自动使用 ZeroShotCompressor 或 ImatrixCompressor
compressor = Compressor(
    config=config,
    model="/path/to/model",
    format="gguf_k",  # 如果是 gguf_k 格式,会使用 ImatrixCompressor
)

quantized_model, layer_config = compressor.quantize()
```

## 模型类型自动检测

`entry.py` 中的 `detect_model_type()` 函数负责自动检测模型类型:

```python
def detect_model_type(model):
    """检测模型类型

    Args:
        model: 模型实例或模型路径字符串

    Returns:
        str: "mllm" | "diffusion" | "llm"
    """
    from auto_round.utils import is_mllm_model, is_diffusion_model

    # 1. 优先检测 Diffusion 模型
    if is_diffusion_model(model):
        return "diffusion"

    # 2. 检测 MLLM 模型
    if is_mllm_model(model):
        return "mllm"

    # 3. 默认为标准 LLM
    return "llm"
```

### 检测逻辑说明

1. **Diffusion 模型检测** (`is_diffusion_model`):
   - 检查目录中是否存在 `model_index.json` 文件
   - 检查是否为 `DiffusionPipeline` 实例

2. **MLLM 模型检测** (`is_mllm_model`):
   - 检查是否存在 `processor_config.json`
   - 检查是否存在 `preprocessor_config.json`
   - 检查 config 中是否包含多模态相关键(vision_config 等)

3. **LLM 模型** (默认):
   - 所有其他情况

## Compressor 动态选择逻辑

`Compressor.__new__()` 方法根据配置类型和模型类型动态创建实例:

### 决策流程图

```
Compressor.__new__()
│
├─ 检测模型类型 (detect_model_type)
│  ├─ "diffusion"
│  ├─ "mllm"  
│  └─ "llm"
│
├─ AutoRoundConfig (需要校准)
│  ├─ model_type == "mllm"
│  │  └─> MLLMCalibCompressor
│  │      └─ 使用 MLLM dataloader
│  │      └─ 支持 processor, template 等
│  │
│  ├─ model_type == "diffusion"
│  │  └─> DiffusionCalibCompressor
│  │      └─ 加载 diffusion pipeline
│  │      └─ 提取 transformer/unet
│  │
│  └─ model_type == "llm"
│     └─> CalibCompessor
│         └─ 标准文本数据集
│
└─ RTNConfig (零样本量化)
   ├─ enable_imatrix == True
   │  └─> ImatrixCompressor
   │      └─ 使用 importance matrix
   │
   └─ enable_imatrix == False
      └─> ZeroShotCompressor
          └─ 纯 RTN 量化
```

### 代码实现

```python
class Compressor(object):
    def __new__(cls, config, model, tokenizer=None, platform="hf", format=None, **kwargs):
        # 检测模型类型
        model_type = detect_model_type(model)

        if isinstance(config, AutoRoundConfig):
            # AutoRound 需要校准
            if model_type == "mllm":
                from auto_round.compressors_new.mllm_mixin import MLLMCalibCompressor

                return MLLMCalibCompressor(config, model, tokenizer, platform, format, **kwargs)
            elif model_type == "diffusion":
                from auto_round.compressors_new.diffusion_mixin import DiffusionCalibCompressor

                return DiffusionCalibCompressor(config, model, tokenizer, platform, format, **kwargs)
            else:
                return CalibCompessor(config, model, tokenizer, platform, format, **kwargs)

        elif isinstance(config, RTNConfig):
            # RTN 可能需要 imatrix
            if enable_imatrix:
                from auto_round.compressors_new.calib import ImatrixCompressor

                return ImatrixCompressor(config, model, tokenizer, platform, format, **kwargs)
            return ZeroShotCompressor(config, model, tokenizer, platform, format, **kwargs)
```

## 扩展新模型类型

如果需要支持新的模型类型,按照以下步骤操作:

### 步骤 1: 创建专用 Compressor 类

在 `compressors_new/` 下创建新文件,例如 `audio_calib.py`:

```python
# compressors_new/audio_calib.py
from typing import Union
import torch
from auto_round.algorithms.alg_config import AlgConfig
from auto_round.compressors_new.calib import CalibCompessor
from auto_round.logger import logger


class AudioCalibCompressor(CalibCompessor):
    """音频模型专用校准压缩器"""

    def __init__(
        self,
        config: Union[AlgConfig, list[AlgConfig]],
        model: Union[torch.nn.Module, str],
        tokenizer=None,
        platform="hf",
        format=None,
        audio_processor=None,  # 音频特定参数
        **kwargs,
    ):
        # 保存音频特定参数
        self.audio_processor = audio_processor

        # 调用父类初始化
        super().__init__(
            config=config,
            model=model,
            tokenizer=tokenizer,
            platform=platform,
            format=format,
            **kwargs,
        )

    @torch.no_grad()
    def calib(self, nsamples, bs):
        """实现音频模型特定的校准逻辑"""
        from your_audio_module import get_audio_dataloader

        logger.info("Preparing audio dataloader...")

        # 获取音频专用的 dataloader
        self.dataloader = get_audio_dataloader(
            model=self.model_context.model,
            audio_processor=self.audio_processor,
            dataset=self.dataset,
            nsamples=nsamples,
            batch_size=bs,
            seed=self.seed,
        )

        # 执行校准前向传播
        total_cnt = 0
        for data in self.dataloader:
            if data is None:
                continue

            # 处理并前向传播
            try:
                if isinstance(data, dict):
                    self.model_context.model(**data)
                else:
                    self.model_context.model(data)
            except Exception as e:
                logger.warning(f"Calibration failed: {e}")

            total_cnt += bs
            if total_cnt >= nsamples:
                break

        if total_cnt == 0:
            logger.error("No calibration data processed")
            exit(-1)
```

### 步骤 2: 更新模型检测逻辑

在 `entry.py` 中添加音频模型检测:

```python
# entry.py
def detect_model_type(model):
    """检测模型类型"""
    from auto_round.utils import is_mllm_model, is_diffusion_model, is_audio_model

    # 按特殊性从高到低检测
    if is_diffusion_model(model):
        return "diffusion"

    if is_audio_model(model):  # 新增音频检测
        return "audio"

    if is_mllm_model(model):
        return "mllm"

    return "llm"
```

### 步骤 3: 更新 Compressor 入口

在 `entry.py` 的 `Compressor.__new__()` 中添加音频分支:

```python
class Compressor(object):
    def __new__(cls, config, model, tokenizer=None, platform="hf", format=None, **kwargs):
        local_args = {k: v for k, v in locals().items() if k not in cls.SKIP_ARGS}

        # 检测模型类型
        model_type = detect_model_type(model)

        if isinstance(config, AutoRoundConfig):
            # 新增音频分支
            if model_type == "audio":
                from auto_round.compressors_new.audio_calib import AudioCalibCompressor

                return AudioCalibCompressor(config, **local_args, **kwargs)
            elif model_type == "mllm":
                from auto_round.compressors_new.mllm_mixin import MLLMCalibCompressor

                return MLLMCalibCompressor(config, **local_args, **kwargs)
            # ... 其他分支
```

### 步骤 4: 实现模型检测函数

在 `auto_round/utils/model.py` 中添加:

```python
def is_audio_model(model_or_path: Union[str, torch.nn.Module]) -> bool:
    """检测是否为音频模型"""
    if isinstance(model_or_path, str):
        # 检查配置文件中的特征
        config_path = os.path.join(model_or_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            # 检查是否包含音频相关配置
            if "audio_config" in config:
                return True
            if config.get("model_type") in ["whisper", "wav2vec2", "hubert"]:
                return True

    if isinstance(model_or_path, torch.nn.Module):
        # 检查模块中是否有音频相关组件
        for name, module in model_or_path.named_modules():
            if "audio" in name.lower():
                return True

    return False
```

## 实现细节

### MLLMCalibCompressor 关键实现

```python
class MLLMCalibCompressor(CalibCompessor):
    def __init__(
        self, config, model, processor=None, image_processor=None, template=None, extra_data_dir=None, **kwargs
    ):
        # 保存 MLLM 特定参数
        self.processor = processor
        self.image_processor = image_processor
        self.template = template
        self.extra_data_dir = extra_data_dir
        super().__init__(config, model, **kwargs)

    @torch.no_grad()
    def calib(self, nsamples, bs):
        # 1. 选择合适的 template
        self.template_obj = get_template(self.template or "default")

        # 2. 获取 MLLM dataloader
        self.dataloader = get_mllm_dataloader(
            model=self.model_context.model,
            tokenizer=self.tokenizer,
            dataset=self.dataset,
            processor=self.processor,
            image_processor=self.image_processor,
            nsamples=nsamples,
            seqlen=self.quantize_config.seqlen,
            seed=self.seed,
            batch_size=bs,
            template=self.template_obj,
            extra_data_dir=self.extra_data_dir,
        )

        # 3. 执行校准
        for data in self.dataloader:
            self.model_context.model(**data)
```

**关键点:**
- 处理 `processor`, `image_processor`, `template` 等 MLLM 特定参数
- 使用 `get_mllm_dataloader` 获取多模态数据
- 支持自定义数据目录 (`extra_data_dir`)

### DiffusionCalibCompressor 关键实现

```python
class DiffusionCalibCompressor(CalibCompessor):
    def __init__(self, config, model, guidance_scale=7.5, num_inference_steps=50, **kwargs):
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.pipe = None
        super().__init__(config, model, **kwargs)

    def post_init(self):
        # 预先加载 diffusion pipeline
        if isinstance(self.model_context.model, str):
            self._load_diffusion_model()
        super().post_init()

    def _load_diffusion_model(self):
        # 加载完整的 pipeline
        pipe, pipe_config = diffusion_load_model(
            pretrained_model_name_or_path=self.model_context.model,
            platform=self.platform,
            device=self.compress_context.device,
        )
        self.pipe = pipe

        # 提取 transformer 或 unet 用于量化
        if hasattr(pipe, "transformer"):
            self.model_context.model = pipe.transformer
        elif hasattr(pipe, "unet"):
            self.model_context.model = pipe.unet

    @torch.no_grad()
    def calib(self, nsamples, bs):
        # 获取 diffusion dataloader
        self.dataloader = get_diffusion_dataloader(
            pipe=self.pipe,
            dataset=self.dataset,
            nsamples=nsamples,
            batch_size=bs,
            seed=self.seed,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
        )

        # 执行校准
        for data in self.dataloader:
            self.model_context.model(**data)
```

**关键点:**
- 需要加载完整的 diffusion pipeline
- 从 pipeline 中提取 transformer/unet 组件
- 使用扩散模型特定的数据生成逻辑

### 完整数据流

```
1. 用户调用 Compressor(config, model, ...)
   │
   ├─> Compressor.__new__() 
   │   ├─> detect_model_type(model)
   │   │   └─> 返回 "llm" | "mllm" | "diffusion"
   │   │
   │   └─> 根据 config 类型和 model_type 创建实例
   │       ├─> MLLMCalibCompressor (MLLM + AutoRound)
   │       ├─> DiffusionCalibCompressor (Diffusion + AutoRound)
   │       ├─> CalibCompessor (LLM + AutoRound)
   │       ├─> ImatrixCompressor (RTN + imatrix)
   │       └─> ZeroShotCompressor (RTN)
   │
2. 实例.__init__()
   │   ├─> 保存模型特定参数
   │   └─> super().__init__() 调用父类
   │
3. 用户调用 compressor.quantize()
   │
   ├─> post_init()
   │   ├─> _load_model() (可能被子类重写)
   │   └─> 初始化 quantizer
   │
   ├─> calib(nsamples, bs) (可能被子类重写)
   │   ├─> 准备 dataloader (模型特定)
   │   └─> 执行校准前向传播
   │
   ├─> cache_inter_data()
   │   └─> 缓存中间激活值
   │
   ├─> 对每个 block 执行量化
   │   └─> 运行量化算法 (AutoRound/RTN 等)
   │
   └─> 返回 (quantized_model, layer_config)
```

## 与旧架构对比

### 旧架构 (`compressors/`)

**使用方式:**
```python
# 需要手动选择 Compressor
from auto_round.compressors.mllm.compressor import MLLMCompressor
from auto_round.compressors.diffusion.compressor import DiffusionCompressor

# MLLM
mllm_compressor = MLLMCompressor(
    model=model,
    scheme="W4A16",
    iters=200,
    # ... 很多参数
)

# Diffusion
diffusion_compressor = DiffusionCompressor(
    model=model,
    scheme="W4A16",
    iters=200,
    # ... 很多参数
)
```

**问题:**
- 用户需要手动判断模型类型
- 需要导入不同的 Compressor 类
- 参数直接传给 Compressor,没有统一的配置对象
- 每个 Compressor 都是独立实现,代码重复

### 新架构 (`compressors_new/`)

**使用方式:**
```python
# 统一入口,自动检测
from auto_round.compressors_new.entry import Compressor
from auto_round.algorithms.quantization.auto_round.config import AutoRoundConfig

config = AutoRoundConfig(scheme="W4A16", iters=200, nsamples=128)

# 同一个入口处理所有模型类型
compressor = Compressor(
    config=config,
    model=model,  # 自动检测是 LLM/MLLM/Diffusion
    tokenizer=tokenizer,
    # 模型特定参数...
)
```

**优势:**
- ✅ 自动模型类型检测
- ✅ 统一的配置对象 (AlgConfig)
- ✅ 单一入口点
- ✅ 通过继承复用代码
- ✅ 易于扩展新模型类型

## 测试

### 运行测试脚本

```bash
# 运行完整测试
python test_compressor_new_arch.py

# 测试特定类型
python -c "from test_compressor_new_arch import test_mllm_compressor; test_mllm_compressor()"
```

### 测试内容

1. **模型类型检测测试**
   ```python
   from auto_round.compressors_new.entry import detect_model_type

   assert detect_model_type("/models/opt-125m/") == "llm"
   assert detect_model_type("/models/Qwen2-VL-2B-Instruct") == "mllm"
   assert detect_model_type("/models/stable-diffusion-2-1") == "diffusion"
   ```

2. **Compressor 创建测试**
   ```python
   from auto_round.compressors_new.entry import Compressor
   from auto_round.algorithms.quantization.auto_round.config import AutoRoundConfig

   config = AutoRoundConfig(scheme="W4A16")

   # 测试 LLM
   comp = Compressor(config=config, model="/models/opt-125m/")
   assert isinstance(comp, CalibCompessor)

   # 测试 MLLM
   comp = Compressor(config=config, model="/models/Qwen2-VL-2B-Instruct")
   assert isinstance(comp, MLLMCalibCompressor)

   # 测试 Diffusion
   comp = Compressor(config=config, model="/models/stable-diffusion-2-1")
   assert isinstance(comp, DiffusionCalibCompressor)
   ```

## 常见问题

### Q1: 如何判断我的模型会使用哪个 Compressor?

**A:** 运行以下代码查看:

```python
from auto_round.compressors_new.entry import detect_model_type, Compressor
from auto_round.algorithms.quantization.auto_round.config import AutoRoundConfig

model_path = "/your/model/path"

# 检测模型类型
model_type = detect_model_type(model_path)
print(f"Model type: {model_type}")

# 创建 compressor 并查看类型
config = AutoRoundConfig(scheme="W4A16")
comp = Compressor(config=config, model=model_path)
print(f"Compressor type: {type(comp).__name__}")
```

### Q2: 如何传递模型特定的参数?

**A:** 直接传递给 `Compressor()`,它会自动转发:

```python
# MLLM 特定参数
compressor = Compressor(
    config=config,
    model=mllm_model_path,
    processor=processor,  # MLLM 特定
    template="qwen2_vl",  # MLLM 特定
    extra_data_dir="/data/imgs",  # MLLM 特定
)

# Diffusion 特定参数
compressor = Compressor(
    config=config,
    model=diffusion_model_path,
    guidance_scale=7.5,  # Diffusion 特定
    num_inference_steps=50,  # Diffusion 特定
)
```

### Q3: 新架构是否向后兼容?

**A:** 是的,旧的 `compressors/` 仍然可用:

```python
# 旧方式仍然工作
from auto_round.compressors.mllm.compressor import MLLMCompressor

comp = MLLMCompressor(model=..., scheme="W4A16", ...)

# 新方式 (推荐)
from auto_round.compressors_new.entry import Compressor
from auto_round.algorithms.quantization.auto_round.config import AutoRoundConfig

config = AutoRoundConfig(scheme="W4A16")
comp = Compressor(config=config, model=...)
```

### Q4: RTN 和 AutoRound 的区别?

**A:**

| 特性 | RTN | AutoRound |
|------|-----|-----------|
| 需要校准数据 | ❌ 否 | ✅ 是 |
| 量化质量 | 较低 | 较高 |
| 量化速度 | 快 | 慢 |
| Compressor | ZeroShotCompressor | CalibCompessor 系列 |

```python
# RTN - 快速但质量较低
from auto_round.algorithms.quantization.rtn.config import RTNConfig

config = RTNConfig(scheme="W4A16")

# AutoRound - 慢但质量较高
from auto_round.algorithms.quantization.auto_round.config import AutoRoundConfig

config = AutoRoundConfig(scheme="W4A16", iters=200)
```

## 总结

新架构的核心优势:

| 特性 | 说明 | 好处 |
|------|------|------|
| 🎯 **统一入口** | 一个 `Compressor` 类处理所有模型 | 简化使用,降低学习成本 |
| 🔍 **自动检测** | 自动识别 LLM/MLLM/Diffusion | 无需手动判断模型类型 |
| 🧩 **配置对象** | 使用 `AlgConfig` 统一配置 | 参数管理更清晰 |
| 🏗️ **继承复用** | 通过继承共享基类功能 | 减少代码重复 |
| 🔌 **易于扩展** | 3步添加新模型类型 | 符合开闭原则 |
| 🔄 **向后兼容** | 旧 API 仍然可用 | 平滑迁移 |

### 迁移建议

**从旧架构迁移到新架构:**

```python
# 旧代码
from auto_round.compressors.mllm.compressor import MLLMCompressor

comp = MLLMCompressor(
    model=model,
    scheme="W4A16",
    iters=200,
    nsamples=128,
    # ... 更多参数
)

# 新代码
from auto_round.compressors_new.entry import Compressor
from auto_round.algorithms.quantization.auto_round.config import AutoRoundConfig

config = AutoRoundConfig(
    scheme="W4A16",
    iters=200,
    nsamples=128,
)
comp = Compressor(
    config=config,
    model=model,
    # 模型特定参数自动识别
)
```

**迁移步骤:**
1. 导入 `Compressor` 和 `AutoRoundConfig`
2. 创建 `config` 对象,将量化相关参数放入 config
3. 将模型特定参数直接传递给 `Compressor()`
4. 移除手动的模型类型判断代码

这种设计使得代码更加模块化、可维护和可扩展,同时保持了简单易用的 API 接口。
