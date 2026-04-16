---
name: adapt-new-llm
description: "Adapt AutoRound to support a new LLM architecture that doesn't work out-of-the-box. Use when quantization fails for a new model type, block detection doesn't find layers, MoE models need unfusing, custom forward passes are needed, or non-standard linear layer types need handling."
---

# Adapting AutoRound for a New LLM Architecture

## Overview

Most standard Transformers-based LLMs work with AutoRound out-of-the-box. This
skill covers what to do when a new model architecture requires code changes. The
need for adaptation typically arises from:

- Non-standard layer hierarchy (block detection fails)
- Fused Mixture-of-Experts (MoE) weights
- Non-standard linear layer types (not `nn.Linear` or `Conv1D`)
- Complex multi-component architectures (multimodal routing)
- Shared cache keys or position embeddings

## Step 0: Diagnose the Problem

Try quantizing the model first:

```python
from auto_round import AutoRound

ar = AutoRound("your-org/your-model", scheme="W4A16", iters=2, nsamples=2)
ar.quantize_and_save(output_dir="./test_output", format="auto_round")
```

Common failure modes and their fixes:

| Error / Symptom | Root Cause | Fix Section |
|-----------------|-----------|-------------|
| "No quantizable layers found" | Block detection failed | Step 1 |
| "Quantized 0/N layers" | Layers not `nn.Linear`/`Conv1D` | Step 4 |
| Shape mismatch in MoE layers | Fused expert weights | Step 2 |
| Wrong outputs / calibration diverges | Forward pass not exercised correctly | Step 3 |
| Cache key errors (Gemma3-style) | Shared position embeddings | Step 5 |

## Step 1: Fix Block Detection

AutoRound discovers quantizable blocks via `get_block_names()` which searches
recursively for `nn.ModuleList` instances. If your model has a non-standard
layer hierarchy, block detection may fail.

### Check current detection

```python
from auto_round.utils import get_block_names
model = ...  # loaded model
print(get_block_names(model))
```

### Option A: Use `to_quant_block_names` parameter

For simple cases, override block names without code changes:

```python
ar = AutoRound(
    model,
    to_quant_block_names="model.decoder.layers",  # explicit path
)
```

### Option B: Register in `SPECIAL_MULTIMODAL_BLOCK`

For multimodal or multi-component models, add a custom block handler in
`auto_round/special_model_handler.py`:

```python
def _get_your_model_multimodal_block(model, quant_vision=False):
    """Get block names for YourModel.

    YourModel structure:
    - encoder.layers: encoder blocks
    - decoder.layers: decoder blocks
    """
    block_names = []

    if quant_vision and hasattr(model, "encoder"):
        block_names.append([
            f"encoder.layers.{i}" for i in range(len(model.encoder.layers))
        ])

    block_names.append([
        f"decoder.layers.{i}" for i in range(len(model.decoder.layers))
    ])

    return block_names


# Register: key must match model.config.model_type
SPECIAL_MULTIMODAL_BLOCK["your_model_type"] = _get_your_model_multimodal_block
```

Also add to support lists if applicable:

```python
# If text-only calibration works for this multimodal model:
SUPPORT_ONLY_TEXT_MODELS.append("your_model_type")

# If batch_size must be limited:
mllms_with_limited_bs = (..., "your_model_type")
```

## Step 2: Handle MoE (Mixture-of-Experts) Models

MoE models often have fused 3D expert weights (shape
`[num_experts, hidden, intermediate]`) that must be "unfused" into per-expert
`nn.Linear` layers for quantization.

### Check if auto-handled

Transformers >= 5.0 has a `linear_loop` experts interface that auto-unfuses
most MoE models. Test first — it may just work.

### Register custom unfusing

If auto-unfusing fails, create a custom module in
`auto_round/modeling/fused_moe/`:

**1. Create `auto_round/modeling/fused_moe/your_moe.py`:**

```python
"""Unfuse fused MoE weights for YourModel."""
import torch
import torch.nn as nn
from auto_round.modeling.fused_moe.replace_modules import register_replacement

@register_replacement("YourMoELayer")
def replace_your_moe_layer(module, name, model):
    """Replace FusedMoE with per-expert nn.Linear layers."""
    experts = nn.ModuleList()
    for i in range(module.num_experts):
        linear = nn.Linear(
            module.hidden_size, module.intermediate_size, bias=False
        )
        linear.weight.data = module.weight[i].clone()
        experts.append(linear)
    return experts
```

**2. Register in `BUILTIN_MODULES`:**

Edit `auto_round/modeling/fused_moe/replace_modules.py`:

```python
BUILTIN_MODULES["your_model_type"] = LazyImport("auto_round.modeling.fused_moe.your_moe")
```

### Existing MoE implementations

| Model Type | File | Pattern |
|------------|------|---------|
| `llama4` | `fused_moe/llama4.py` | Custom replacement for no `use_experts_implementation` |
| `deepseek_v2` | `fused_moe/deepseek_v2.py` | q_scale calibration for Gaudi |
| `qwen3_5_moe` | `fused_moe/qwen3_5_moe.py` | Transformers >= 5.0 support |
| `step3p5` | `fused_moe/step3_5_moe.py` | Splits fused MoELinear |
| `qwen3_omni_moe` | `fused_moe/qwen3_omni.py` | Thinker + talker MoE |

## Step 3: Add Custom Forward Pass

Some models have non-standard forward passes that don't get calibrated correctly
with the default `model.forward()`. This is common for multi-component
architectures.

Edit `_handle_special_model()` in `auto_round/special_model_handler.py`:

```python
def _your_model_forward(model, **kwargs):
    """Custom forward that routes through all quantizable components."""
    # Example: route through both encoder and decoder
    encoder_output = model.encoder(**kwargs)
    decoder_output = model.decoder(encoder_output, **kwargs)
    return decoder_output


def _handle_special_model(model):
    ...
    if hasattr(model, "config") and model.config.model_type == "your_model_type":
        from functools import partial
        model.forward = partial(_your_model_forward, model)
    return model
```

### When is this needed?

- Model has multiple sub-models (thinker/talker, encoder/decoder)
- Default forward doesn't exercise all quantizable layers
- Model needs special input preprocessing during calibration

### Existing examples

| Model | Custom Forward | Purpose |
|-------|---------------|---------|
| `deepseek_vl_v2` | `_deepseek_vl2_forward` | Route through language component |
| `qwen2_5_omni` | `_qwen2_5_omni_forward` | Route through thinker → talker |
| `qwen3_omni_moe` | `_qwen3_omni_moe_forward` | Handle MoE routing in omni model |

## Step 4: Handle Non-Standard Linear Layers

AutoRound quantizes these layer types by default:

```python
# auto_round/utils/common.py
SUPPORTED_LAYER_TYPES = (torch.nn.Linear, transformers.pytorch_utils.Conv1D)
INNER_SUPPORTED_LAYER_TYPES = ("FP8Linear",)  # matched by class name string
```

If your model uses a custom linear type (e.g., `QuantizedLinear`, `FP8Linear`),
it won't be quantized unless registered.

### Option A: String-based matching

`INNER_SUPPORTED_LAYER_TYPES` matches by class name string — useful for
external classes that can't be imported directly:

```python
INNER_SUPPORTED_LAYER_TYPES = ("FP8Linear", "YourCustomLinear")
```

### Option B: Type-based registration

If you can import the class:

```python
from your_library import YourLinear
SUPPORTED_LAYER_TYPES = SUPPORTED_LAYER_TYPES + (YourLinear,)
```

## Step 5: Handle Shared Cache Keys

Some models share tensors across blocks during inference (e.g., Gemma3's
rotary position embeddings). These must be declared so the calibration cache
doesn't duplicate or corrupt them.

Edit `SPECIAL_SHARED_CACHE_KEYS` in `auto_round/special_model_handler.py`:

```python
SPECIAL_SHARED_CACHE_KEYS["YourModelForCausalLM"] = ("shared_position_embeddings", "shared_rope")
```

The key is the **class name** of the model (not `model_type`).

## Step 6: Test

```python
def test_your_model_quantization():
    ar = AutoRound(
        "your-org/your-model",
        scheme="W4A16",
        iters=2,
        nsamples=2,
        batch_size=2,
    )
    compressed_model, layer_config = ar.quantize()
    # Verify layers were quantized
    assert len(layer_config) > 0, "No layers were quantized"

    ar.save_quantized(output_dir="./tmp_your_model", format="auto_round")

    # Verify inference works
    from auto_round.utils import model_infer
    output = model_infer(compressed_model, tokenizer, "Hello world")
    assert output is not None
```

## Step 7: Update Documentation

1. Add model to supported list in `README.md`
2. Update `README_CN.md` with equivalent Chinese content
3. Add example script if the model has notable differences

## Checklist

- [ ] `get_block_names()` finds all quantizable blocks
- [ ] MoE layers (if any) are unfused correctly
- [ ] `calib()` runs without shape errors
- [ ] All target layers are quantized (check "Quantized X/Y layers" log)
- [ ] Forward pass exercises all quantizable components
- [ ] Quantized model produces valid outputs
- [ ] Export to target format works
- [ ] README.md + README_CN.md updated

## Key Files

| File | Purpose |
|------|---------|
| `auto_round/special_model_handler.py` | Block handlers, custom forwards, shared cache keys |
| `auto_round/modeling/fused_moe/replace_modules.py` | MoE unfusing registry (`BUILTIN_MODULES`) |
| `auto_round/utils/common.py` | `SUPPORTED_LAYER_TYPES`, `INNER_SUPPORTED_LAYER_TYPES` |
| `auto_round/utils/model.py` | `get_block_names()`, `is_mllm_model()`, model loading |
| `auto_round/compressors/base.py` | Core quantization loop, `calib()`, `_quantize_blocks()` |
| `auto_round/autoround.py` | `AutoRound` factory — model type routing logic |
