# Compressor 新架构说明

## 概述

本文档介绍 `compressors_new` 的新架构设计，为 LLM、MLLM 和 Diffusion 模型提供统一的量化入口。

## 架构设计

### 核心思想

`entry.py` 中的 `Compressor` 是唯一入口。构造时自动检测模型类型和配置类型，通过多重继承（Mixin 模式）动态创建正确的具体类。

### 目录结构

```
compressors_new/
├── entry.py               # 统一入口 — Compressor + AutoRound 兼容层
├── base.py                # BaseCompressor 基类 + SerializedCompressorConfig
├── calib.py               # CalibCompressor（AutoRound 梯度校准）
│                          # CalibratedRTNCompressor（RTN + imatrix / 激活校准）
├── zero_shot.py           # ZeroShotCompressor（零样本 RTN）
├── mllm_mixin.py          # MLLMMixin（视觉-语言模型扩展逻辑）
├── diffusion_mixin.py     # DiffusionMixin（扩散模型 pipeline 扩展逻辑）
└── docs/                  # 本文档
```

### 类继承关系

```
BaseCompressor
    ├── CalibCompressor            （AutoRound，基于梯度的校准量化）
    ├── CalibratedRTNCompressor    （RTN + importance-matrix 或激活校准）
    └── ZeroShotCompressor         （RTN，不需要校准数据）

Mixin（在 entry.py 中动态组合）：
    MLLMMixin      + {CalibCompressor | CalibratedRTNCompressor | ZeroShotCompressor}
    DiffusionMixin + {CalibCompressor | CalibratedRTNCompressor | ZeroShotCompressor}
```

## 配置层

### QuantizationConfig（dataclass）

`QuantizationConfig` 声明为 `@dataclass(kw_only=True)`，消除了 `__init__` 中的样板代码。
子类仍然用 `super().__init__(scheme=..., **kwargs)` 正常调用：

```python
@dataclass(kw_only=True)
class QuantizationConfig(AlgConfig):
    _alg_cls: ClassVar[str] = None  # 指定使用哪个量化器类

    scheme: Union[str, dict, QuantizationScheme, AutoScheme] = "W4A16"
    bits: int = None
    group_size: int = None  # 也接受 tuple，如 (128,128) 用于块状 FP8
    # ... 其他字段

    def __post_init__(self):
        self._early_resolve_scheme()  # 构造时即刻解析 scheme 属性
```

子类：
- `RTNConfig(QuantizationConfig)` — 新增 `disable_opt_rtn`、`seqlen`、`nsamples`、`batch_size`
- `SignRoundConfig(QuantizationConfig)` — 新增 `iters`、`lr`、`nblocks`、`enable_minmax_tuning` 等

### AlgConfig

`AlgConfig` 是基类，用于 `compressors_new/` 各处的类型标注。
`QuantizationConfig` 及未来的非量化配置都继承自它。

## ModelContext

`ModelContext.__init__` **立即加载模型** —— `BaseCompressor.__init__` 返回时，模型已经在 CPU 内存中。

```python
class ModelContext(BaseContext):
    def __init__(self, model, tokenizer, platform, ..., formats, is_act_quantize, quant_nontext_module):
        # ... 存储属性
        self._load_model()                  # 加载 LLM / MLLM / Diffusion 模型
        check_and_mark_quantized_module(self.model)
        self.model = self.model.eval()
        self.shared_cache_keys = get_shared_keys(self.model)
        self.is_moe_model = is_moe_model(self.model)
        self._set_amp_dtype()

    def apply_patches(self, formats):
        """应用格式相关的模型结构补丁。
        由 BaseCompressor.post_init() 在 formats 解析完毕后调用。
        """
        self._patch_custom_moe_modules()    # 如 Qwen3VL top_k 修复
        self.model = update_module(self.model, formats=formats, ...)
        for n, m in self.model.named_modules():
            m.global_name = n               # 赋予量化器使用的全局名称
        self._is_initialized = True
```

## BaseCompressor.post_init() 执行流程

`post_init()` 在 `quantize()` 开始时调用（不在 `__init__` 中）。
顺序至关重要——模型补丁必须在量化器初始化之前完成：

```
post_init()
│
├─ 1. 解析 formats（str → list[OutputFormat]）
│
├─ 2. 应用模型补丁
│     model_context.apply_patches(formats)
│     ├── _patch_custom_moe_modules()
│     ├── update_module(model, formats)     # 插入 gguf_pack_linear 等
│     └── 为所有模块赋予 m.global_name
│
├─ 3. 在已补丁的模型上初始化量化器
│     quantizer = BaseQuantizers.from_config(config)
│     quantizer.post_init()
│     ├── _parse_scheme() → 解析最终量化属性
│     ├── get_block_names(quant_vision=quant_nontext_module)
│     ├── find_matching_blocks() → quant_block_list
│     ├── 反填 to_quant_block_names（如果原来为 None）
│     └── configure_layer_config()
│
└─ 4. 设置 device_map、torch compile、offloader
```

> **无 `refresh_quantizer_for_initialized_model()`** —— 旧调用已通过先执行 `apply_patches`、
> 再调用 `quantizer.post_init()` 的顺序调整消除。

## BaseQuantizers 接口

所有量化器接受**名称**（str），而非模块对象。
模块在内部通过 `get_module(model, name)` 获取：

```python
class BaseQuantizers:
    def quantize_block(
        self,
        block_name: Union[str, list[str]],  # list[str] 用于 nblocks > 1
        input_ids=None,
        input_others=None,
        **kwargs,
    ): ...

    def quantize_layer(self, layer_name: str, **kwargs): ...
```

- `str` → `get_module(model, block_name)`
- `list[str]` → `WrapperMultiblock([get_module(model, n) for n in block_name])`（多块模式）

## Compressor 选择决策树

```
Compressor.__new__(config, model, format, **kwargs)
│
├─ 检测模型类型
│  ├─ is_diffusion_model() → "diffusion"
│  ├─ is_mllm_model()      → "mllm"
│  └─ 其他               → "llm"
│
├─ isinstance(config, SignRoundConfig)
│  ├─ mllm      → class MLLMCalibCompressor(MLLMMixin, CalibCompressor)
│  ├─ diffusion → class DiffusionCalibCompressor(DiffusionMixin, CalibCompressor)
│  └─ llm       → CalibCompressor
│
└─ isinstance(config, RTNConfig)
   ├─ enable_imatrix 或 needs_act_calib  →  CalibratedRTNCompressor 路径
   │  ├─ gguf_k 格式            → enable_imatrix = True
   │  ├─ 对称 int RTN           → enable_imatrix = True
   │  ├─ 静态激活量化           → needs_act_calib = True
   │  │
   │  ├─ mllm      → class MLLMCalibratedRTNCompressor(MLLMMixin, CalibratedRTNCompressor)
   │  ├─ diffusion → class DiffusionCalibratedRTNCompressor(DiffusionMixin, CalibratedRTNCompressor)
   │  └─ llm       → CalibratedRTNCompressor
   │
   └─ 其他（零样本）  →  ZeroShotCompressor 路径
      ├─ mllm      → class MLLMZeroShotCompressor(MLLMMixin, ZeroShotCompressor)
      ├─ diffusion → class DiffusionZeroShotCompressor(DiffusionMixin, ZeroShotCompressor)
      └─ llm       → ZeroShotCompressor
```

## MLLMMixin

```python
class MLLMMixin:
    def __init__(
        self,
        *args,
        processor=None,
        image_processor=None,
        template=None,
        extra_data_dir=None,
        quant_nontext_module=False,
        **kwargs
    ):
        self.processor = processor
        self.template = template
        self.quant_nontext_module = quant_nontext_module
        # 传给 ModelContext，使 get_block_names 包含视觉编码器的块
        kwargs.setdefault("quant_nontext_module", quant_nontext_module)
        super().__init__(*args, **kwargs)

    def calib(self, nsamples, bs):
        # 使用 get_mllm_dataloader，带 template / processor
        ...
```

`quant_nontext_module` 传递链路：
`MLLMMixin.__init__` → `kwargs.setdefault` → `BaseCompressor.__init__` pop
→ `ModelContext(quant_nontext_module=...)` → `BaseQuantizers.post_init()`
调用 `get_block_names(quant_vision=quant_nontext_module)`

## MRO（方法解析顺序）示例

```
MLLMCalibCompressor（entry.py 中动态创建）
    └─> MLLMMixin
        └─> CalibCompressor
            └─> BaseCompressor
                └─> object

调用 __init__() 的执行顺序：
  1. MLLMCalibCompressor.__init__()  → 未定义，向上查找
  2. MLLMMixin.__init__()
     - 保存 MLLM 专属属性：processor、template、quant_nontext_module 等
     - kwargs.setdefault("quant_nontext_module", ...)
     - super().__init__() → 进入 CalibCompressor
  3. CalibCompressor.__init__() → BaseCompressor.__init__()
     - pop quant_nontext_module from kwargs
     - 创建 ModelContext(..., quant_nontext_module=quant_nontext_module)
     - ModelContext.__init__ 立即加载模型
     - 创建 CompressContext 单例

结果：MLLMCalibCompressor 实例同时具备：
  ✓ MLLMMixin 提供的 MLLM 特性（processor、template、calib() 重写）
  ✓ CalibCompressor 提供的梯度校准量化
  ✓ BaseCompressor 提供的模型/上下文管理
```

## 使用示例

### 基本 LLM 量化

```python
from auto_round.compressors_new.entry import Compressor
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig

config = SignRoundConfig(scheme="W4A16", iters=200, nsamples=128)
compressor = Compressor(config=config, model="/path/to/llm", tokenizer=tokenizer)
quantized_model, layer_config = compressor.quantize()
```

### MLLM（视觉-语言模型）

```python
config = SignRoundConfig(scheme="W4A16", iters=200)
compressor = Compressor(
    config=config,
    model="/models/Qwen2-VL-2B-Instruct",
    processor=processor,
    template="qwen2_vl",
    quant_nontext_module=False,  # True 则同时量化视觉编码器
)
# 创建：MLLMCalibCompressor(MLLMMixin, CalibCompressor)
quantized_model, layer_config = compressor.quantize()
```

### Diffusion 扩散模型

```python
config = SignRoundConfig(scheme="W4A16", iters=200)
compressor = Compressor(
    config=config,
    model="/models/stable-diffusion-2-1",
    guidance_scale=7.5,
)
# 创建：DiffusionCalibCompressor(DiffusionMixin, CalibCompressor)
```

### RTN 零样本

```python
from auto_round.algorithms.quantization.rtn.config import RTNConfig

config = RTNConfig(scheme="W4A16")
compressor = Compressor(config=config, model="/path/to/model")
```

### RTN + imatrix（GGUF k-quants）

```python
config = RTNConfig(scheme="W4A16")
compressor = Compressor(config=config, model="/path/to/model", format="gguf_k")
# 创建：CalibratedRTNCompressor（enable_imatrix=True）
```

## 扩展新模型类型

**第 1 步**：在 `compressors_new/` 中创建新 Mixin：

```python
class AudioMixin:
    def __init__(self, *args, audio_processor=None, **kwargs):
        self.audio_processor = audio_processor
        super().__init__(*args, **kwargs)

    def calib(self, nsamples, bs):
        # 音频专用 dataloader
        ...
```

**第 2 步**：在 `entry.py` 中添加检测逻辑：

```python
def detect_model_type(model):
    if is_audio_model(model):
        return "audio"
    if is_diffusion_model(model):
        return "diffusion"
    ...
```

**第 3 步**：在 `Compressor.__new__()` 中添加路由：

```python
if model_type == "audio":
    from auto_round.compressors_new.audio_mixin import AudioMixin

    class AudioCalibCompressor(AudioMixin, CalibCompressor):
        pass

    return AudioCalibCompressor(config, **local_args, **kwargs)
```

## 常见问题

### Q1：如何确认我的模型会使用哪个 Compressor？

```python
from auto_round.compressors_new.entry import detect_model_type, Compressor
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig

model_path = "/your/model/path"
print(f"模型类型: {detect_model_type(model_path)}")

config = SignRoundConfig(scheme="W4A16")
comp = Compressor(config=config, model=model_path)
print(f"Compressor 类型: {type(comp).__name__}")
```

### Q2：RTN 和 AutoRound 有什么区别？

| 特性 | RTN | AutoRound |
|------|-----|-----------|
| 需要校准数据 | ❌ 否（ZeroShot）/ ✅ 是（Calibrated） | ✅ 是 |
| 量化质量 | 较低 | 较高 |
| 量化速度 | 快 | 慢 |
| Compressor | ZeroShotCompressor / CalibratedRTNCompressor | CalibCompressor |

### Q3：`group_size` 可以是 tuple 吗？

可以。块状 FP8（如 `FP8_BLOCK` scheme）会将 `group_size` 设置为 `(128, 128)`，
`check_config()` 已通过 `_is_valid_group_size()` 静态方法正确处理 tuple/list/scalar 三种形式。

## 总结

| 特性 | 说明 |
|---|---|
| **统一入口** | 单一 `Compressor` 类，自动检测模型类型 |
| **配置** | `QuantizationConfig` dataclass；子类 `RTNConfig`、`SignRoundConfig` |
| **模型加载** | `ModelContext.__init__` 立即加载；`apply_patches()` 在量化器初始化前运行 |
| **9 种组合** | 3 种模型类型 × 3 种 Compressor，通过 Mixin 动态创建 |
| **量化器接口** | 基于名称的 `quantize_block(name)` / `quantize_layer(name)`，非模块对象 |
| **扩展** | 3 步添加新模型类型（Mixin 类、检测函数、路由） |

