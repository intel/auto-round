# Compressor New Architecture

## Overview

This document describes the new architecture of `compressors_new`, which provides a unified
quantization entry point for LLM, MLLM, and Diffusion models.

## Architecture Design

### Core Idea

`Compressor` in `entry.py` is the single entry point. It detects the model type and config
type at construction time and dynamically creates the correct concrete class using multiple
inheritance (Mixin pattern).

### Directory Structure

```
compressors_new/
├── entry.py               # Unified entry point — Compressor + AutoRound wrapper
├── base.py                # BaseCompressor base class + SerializedCompressorConfig
├── calib.py               # CalibCompressor (AutoRound gradient-based)
│                          # CalibratedRTNCompressor (RTN + imatrix / act-calib)
├── zero_shot.py           # ZeroShotCompressor (zero-shot RTN)
├── mllm_mixin.py          # MLLMMixin (vision-language model extra logic)
├── diffusion_mixin.py     # DiffusionMixin (diffusion pipeline extra logic)
└── docs/                  # This document
```

### Class Hierarchy

```
BaseCompressor
    ├── CalibCompressor            (AutoRound, gradient-based calibration)
    ├── CalibratedRTNCompressor    (RTN + importance-matrix or act calibration)
    └── ZeroShotCompressor         (RTN, no calibration data needed)

Mixins (combined dynamically in entry.py):
    MLLMMixin      + {CalibCompressor | CalibratedRTNCompressor | ZeroShotCompressor}
    DiffusionMixin + {CalibCompressor | CalibratedRTNCompressor | ZeroShotCompressor}
```

## Configuration Layer

### QuantizationConfig (dataclass)

`QuantizationConfig` is declared as a `@dataclass(kw_only=True)`, which eliminates
`__init__` boilerplate. Subclasses call `super().__init__(scheme=..., **kwargs)` as normal:

```python
@dataclass(kw_only=True)
class QuantizationConfig(AlgConfig):
    _alg_cls: ClassVar[str] = None  # which quantizer class to use

    scheme: Union[str, dict, QuantizationScheme, AutoScheme] = "W4A16"
    bits: int = None
    group_size: int = None  # also accepts tuple, e.g. (128,128) for block-FP8
    # ... other fields

    def __post_init__(self):
        self._early_resolve_scheme()  # eagerly resolves scheme attrs at construction time
```

Subclasses:
- `RTNConfig(QuantizationConfig)` — adds `disable_opt_rtn`, `seqlen`, `nsamples`, `batch_size`
- `SignRoundConfig(QuantizationConfig)` — adds `iters`, `lr`, `nblocks`, `enable_minmax_tuning`, …

### AlgConfig

`AlgConfig` is the base class used as type annotation throughout `compressors_new/`.
Both `QuantizationConfig` and future non-quantization configs inherit from it.

## ModelContext

`ModelContext.__init__` **eagerly loads the model** — by the time `BaseCompressor.__init__`
returns, the model is already loaded in CPU memory.

```python
class ModelContext(BaseContext):
    def __init__(self, model, tokenizer, platform, ..., formats, is_act_quantize, quant_nontext_module):
        # ... store attrs
        self._load_model()                  # load LLM / MLLM / Diffusion model
        check_and_mark_quantized_module(self.model)
        self.model = self.model.eval()
        self.shared_cache_keys = get_shared_keys(self.model)
        self.is_moe_model = is_moe_model(self.model)
        self._set_amp_dtype()

    def apply_patches(self, formats):
        """Apply format-specific model structure patches.
        Called by BaseCompressor.post_init() after formats are resolved.
        """
        self._patch_custom_moe_modules()    # e.g. Qwen3VL top_k fix
        self.model = update_module(self.model, formats=formats, ...)
        for n, m in self.model.named_modules():
            m.global_name = n               # assign names used by quantizers
        self._is_initialized = True
```

## BaseCompressor.post_init() Flow

`post_init()` is called at the start of `quantize()` (not in `__init__`).
The order matters — model patches must come before quantizer setup:

```
post_init()
│
├─ 1. Resolve formats  (str → list[OutputFormat])
│
├─ 2. Apply model patches
│     model_context.apply_patches(formats)
│     ├── _patch_custom_moe_modules()
│     ├── update_module(model, formats)     # insert gguf_pack_linear, etc.
│     └── assign m.global_name to all modules
│
├─ 3. Setup quantizer on the patched model
│     quantizer = BaseQuantizers.from_config(config)
│     quantizer.post_init()
│     ├── _parse_scheme() → resolve final quant attrs
│     ├── get_block_names(quant_vision=quant_nontext_module)
│     ├── find_matching_blocks() → quant_block_list
│     ├── back-fill to_quant_block_names (if was None)
│     └── configure_layer_config()
│
└─ 4. Setup device map, torch compile, offloader
```

> **No `refresh_quantizer_for_initialized_model()`** — eliminated by running `apply_patches`
> *before* `quantizer.post_init()`.

## BaseQuantizers Interface

All quantizers accept **names** (str), not module objects.
The module is retrieved internally via `get_module(model, name)`:

```python
class BaseQuantizers:
    def quantize_block(
        self,
        block_name: Union[str, list[str]],  # list[str] for nblocks > 1
        input_ids=None,
        input_others=None,
        **kwargs,
    ): ...

    def quantize_layer(self, layer_name: str, **kwargs): ...
```

- `str` → `get_module(model, block_name)`
- `list[str]` → `WrapperMultiblock([get_module(model, n) for n in block_name])` (multi-block)

## Compressor Selection Decision Tree

```
Compressor.__new__(config, model, format, **kwargs)
│
├─ Detect model type
│  ├─ is_diffusion_model() → "diffusion"
│  ├─ is_mllm_model()      → "mllm"
│  └─ else                 → "llm"
│
├─ isinstance(config, SignRoundConfig)
│  ├─ mllm      → class MLLMCalibCompressor(MLLMMixin, CalibCompressor)
│  ├─ diffusion → class DiffusionCalibCompressor(DiffusionMixin, CalibCompressor)
│  └─ llm       → CalibCompressor
│
└─ isinstance(config, RTNConfig)
   ├─ enable_imatrix OR needs_act_calib  →  CalibratedRTNCompressor path
   │  ├─ gguf_k format              → enable_imatrix = True
   │  ├─ symmetric int RTN          → enable_imatrix = True
   │  ├─ static act quantization    → needs_act_calib = True
   │  │
   │  ├─ mllm      → class MLLMCalibratedRTNCompressor(MLLMMixin, CalibratedRTNCompressor)
   │  ├─ diffusion → class DiffusionCalibratedRTNCompressor(DiffusionMixin, CalibratedRTNCompressor)
   │  └─ llm       → CalibratedRTNCompressor
   │
   └─ else  →  ZeroShotCompressor path
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
        # Pass to ModelContext so get_block_names includes vision blocks
        kwargs.setdefault("quant_nontext_module", quant_nontext_module)
        super().__init__(*args, **kwargs)

    def calib(self, nsamples, bs):
        # Uses get_mllm_dataloader with template / processor
        ...
```

`quant_nontext_module` flow:
`MLLMMixin.__init__` → `kwargs.setdefault` → `BaseCompressor.__init__` pops → `ModelContext(quant_nontext_module=...)`
→ `BaseQuantizers.post_init()` calls `get_block_names(quant_vision=quant_nontext_module)`

## Usage Examples

### Basic LLM quantization

```python
from auto_round.compressors_new.entry import Compressor
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig

config = SignRoundConfig(scheme="W4A16", iters=200, nsamples=128)
compressor = Compressor(config=config, model="/path/to/llm", tokenizer=tokenizer)
quantized_model, layer_config = compressor.quantize()
```

### MLLM (vision-language model)

```python
config = SignRoundConfig(scheme="W4A16", iters=200)
compressor = Compressor(
    config=config,
    model="/models/Qwen2-VL-2B-Instruct",
    processor=processor,
    template="qwen2_vl",
    quant_nontext_module=False,  # True to also quantize vision encoder
)
# Creates: MLLMCalibCompressor(MLLMMixin, CalibCompressor)
```

### Diffusion model

```python
config = SignRoundConfig(scheme="W4A16", iters=200)
compressor = Compressor(
    config=config,
    model="/models/stable-diffusion-2-1",
    guidance_scale=7.5,
)
# Creates: DiffusionCalibCompressor(DiffusionMixin, CalibCompressor)
```

### RTN zero-shot

```python
from auto_round.algorithms.quantization.rtn.config import RTNConfig

config = RTNConfig(scheme="W4A16")
compressor = Compressor(config=config, model="/path/to/model")
```

### RTN with imatrix (GGUF k-quants)

```python
config = RTNConfig(scheme="W4A16")
compressor = Compressor(config=config, model="/path/to/model", format="gguf_k")
# Creates: CalibratedRTNCompressor (enable_imatrix=True)
```

## Extending with New Model Types

**Step 1**: Create a new Mixin in `compressors_new/`:

```python
class AudioMixin:
    def __init__(self, *args, audio_processor=None, **kwargs):
        self.audio_processor = audio_processor
        super().__init__(*args, **kwargs)

    def calib(self, nsamples, bs): ...
```

**Step 2**: Add detection in `entry.py`:

```python
def detect_model_type(model):
    if is_audio_model(model):
        return "audio"
    if is_diffusion_model(model):
        return "diffusion"
    ...
```

**Step 3**: Add routing in `Compressor.__new__()`:

```python
if model_type == "audio":
    from auto_round.compressors_new.audio_mixin import AudioMixin

    class AudioCalibCompressor(AudioMixin, CalibCompressor):
        pass

    return AudioCalibCompressor(config, **local_args, **kwargs)
```

## Summary

| Aspect | Description |
|---|---|
| **Entry point** | Single `Compressor` class, auto-detects model type |
| **Config** | `QuantizationConfig` dataclass; subclasses `RTNConfig`, `SignRoundConfig` |
| **Model loading** | `ModelContext.__init__` loads eagerly; `apply_patches()` runs before quantizer setup |
| **9 combinations** | 3 model types × 3 compressors, dynamic classes via Mixin |
| **Quantizer interface** | Name-based `quantize_block(name)` / `quantize_layer(name)`, not module objects |
| **Extension** | Add new model type in 3 steps (Mixin class, detect fn, routing) |
