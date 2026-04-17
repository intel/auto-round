---
name: adapt-new-diffusion-model
description: "Adapt AutoRound to support a new diffusion model architecture (DiT, UNet, hybrid AR+DiT). Use when a new diffusion model fails quantization, needs custom output configs, requires a custom pipeline function, or is a hybrid architecture with both autoregressive and diffusion components."
---

# Adapting AutoRound for a New Diffusion Model Architecture

## Overview

AutoRound's DiffusionCompressor works with standard diffusers pipelines
(e.g., FLUX). This skill covers what code changes are needed when a new
diffusion model doesn't work out-of-the-box. Common reasons for adaptation:

- Transformer block type not registered in `output_configs`
- Non-standard pipeline API (not compatible with `pipe(prompts, ...)`)
- Hybrid architecture with both AR and diffusion components
- Model not detected as a diffusion model

## Step 0: Diagnose the Problem

```python
from auto_round import AutoRound

ar = AutoRound(
    "your-org/your-diffusion-model",
    scheme="W4A16",
    iters=2,
    nsamples=2,
    num_inference_steps=5,
)
ar.quantize_and_save(output_dir="./test_output", format="fake")
```

| Error / Symptom | Root Cause | Fix Section |
|-----------------|-----------|-------------|
| "using LLM mode" instead of Diffusion | Model not detected as diffusion | Step 1 |
| `assert len(output_config) == len(tmp_output)` | Block output config mismatch | Step 2 |
| Pipeline call fails | Non-standard inference API | Step 3 |
| Hybrid model only quantizes DiT | AR component not handled | Step 4 |

## Step 1: Ensure Model Detection

AutoRound detects diffusion models by checking for `model_index.json` in the
model directory:

```python
# auto_round/utils/model.py
def is_diffusion_model(model_or_path):
    # Checks for model_index.json presence
```

If your model doesn't have `model_index.json`, either:
- Create one in the model directory
- Force diffusion mode via `ExtraConfig`:

```python
from auto_round.compressors import ExtraConfig

ar = AutoRound(
    model,
    extra_config=ExtraConfig(diffusion_config=DiffusionConfig(...)),
)
```

### Pipeline Loading

`diffusion_load_model()` uses `AutoPipelineForText2Image.from_pretrained()` and
extracts `pipe.transformer` as the quantizable model. If your model uses a
different attribute (e.g., `pipe.unet`), this needs adjustment in
`auto_round/utils/model.py`.

## Step 2: Register Transformer Block Output Config

This is the **most common** adaptation needed. The `output_configs` dict maps
transformer block class names to their output tensor names. Without this,
calibration crashes because AutoRound doesn't know how to collect activations.

### Find your block class name

```python
import diffusers

pipe = diffusers.AutoPipelineForText2Image.from_pretrained("your-model")
for name, module in pipe.transformer.named_modules():
    if hasattr(module, "forward") and "block" in name.lower():
        print(f"{name}: {type(module).__name__}")
```

### Register in `output_configs`

Edit `auto_round/compressors/diffusion/compressor.py`:

```python
output_configs = {
    "FluxTransformerBlock": ["encoder_hidden_states", "hidden_states"],
    "FluxSingleTransformerBlock": ["encoder_hidden_states", "hidden_states"],
    # Add your block type:
    "YourTransformerBlock": ["hidden_states"],  # output tensor names in order
}
```

The list must match the **exact order** of tensors returned by the block's
`forward()` method.

### How to determine output tensor names

1. Read the block's `forward()` method in diffusers source code
2. Identify what tensors it returns (usually `hidden_states`, sometimes also
   `encoder_hidden_states`)
3. List them in the order they're returned

**Example**: If `forward()` returns `(hidden_states, encoder_hidden_states)`:
```python
output_configs["YourBlock"] = ["hidden_states", "encoder_hidden_states"]
```

**Example**: If `forward()` returns just `hidden_states`:
```python
output_configs["YourBlock"] = ["hidden_states"]
```

## Step 3: Handle Non-Standard Pipeline API

If your model's inference API differs from the standard
`pipe(prompts, guidance_scale=..., num_inference_steps=...)`, provide a custom
pipeline function.

### Option A: Pass `pipeline_fn` parameter (no code changes)

```python
def your_model_pipeline_fn(pipe, prompts, guidance_scale=7.5, num_inference_steps=28, generator=None, **kwargs):
    """Custom pipeline function for YourModel."""
    for prompt in (prompts if isinstance(prompts, list) else [prompts]):
        pipe.generate(
            prompt=prompt,
            cfg_scale=guidance_scale,
            steps=num_inference_steps,
            generator=generator,
        )


ar = AutoRound(
    "your-model",
    pipeline_fn=your_model_pipeline_fn,
    num_inference_steps=28,
    guidance_scale=7.5,
)
```

### Option B: Attach to pipe object

If using `diffusion_load_model()` directly:

```python
pipe._autoround_pipeline_fn = your_model_pipeline_fn
```

### Option C: Subclass DiffusionCompressor

For full control, override `_run_pipeline()`:

```python
from auto_round.compressors.diffusion.compressor import DiffusionCompressor


class YourModelCompressor(DiffusionCompressor):
    def _run_pipeline(self, prompts):
        generator = (
            None
            if self.generator_seed is None
            else torch.Generator(device=self.pipe.device).manual_seed(self.generator_seed)
        )
        self.pipe.your_custom_generate(
            prompts,
            steps=self.num_inference_steps,
            cfg=self.guidance_scale,
            generator=generator,
        )
```

## Step 4: Add Hybrid AR+DiT Support

For models with both autoregressive and diffusion components (e.g., GLM-Image).

### 4a. Register AR component

Edit `auto_round/compressors/diffusion/hybrid.py`:

```python
HYBRID_AR_COMPONENTS = [
    "vision_language_encoder",  # GLM-Image
    "your_ar_component",  # Your model's AR attribute name
]
```

The attribute name must match what exists on the diffusers pipeline object
(i.e., `pipe.your_ar_component`).

### 4b. Register DiT block output config

Also in `hybrid.py`, add the DiT-specific output config:

```python
output_configs["YourDiTBlock"] = ["hidden_states", "encoder_hidden_states"]
```

### 4c. Register AR block handler

In `auto_round/special_model_handler.py`, add a block handler for the AR
component so AutoRound knows which layers to quantize:

```python
def _get_your_hybrid_multimodal_block(model, quant_vision=False):
    block_names = []
    if quant_vision and hasattr(model, "vision_encoder"):
        block_names.append([f"vision_encoder.blocks.{i}" for i in range(len(model.vision_encoder.blocks))])
    block_names.append([f"language_model.layers.{i}" for i in range(len(model.language_model.layers))])
    return block_names


SPECIAL_MULTIMODAL_BLOCK["your_model_type"] = _get_your_hybrid_multimodal_block
```

### Hybrid quantization flow

The `HybridCompressor` runs two phases:
1. **Phase 1 (AR)**: Quantizes the AR component using text calibration data
   (MLLM-style)
2. **Phase 2 (DiT)**: Quantizes the DiT component using diffusion pipeline
   calibration

```python
ar = AutoRound(
    "your-hybrid-model",
    dataset="coco2014",  # DiT calibration
    ar_dataset="NeelNanda/pile-10k",  # AR calibration
    quant_ar=True,
    quant_dit=True,
)
```

## Step 5: Add Custom Calibration Dataset (Optional)

If your model needs a specific dataset format:

Edit `auto_round/compressors/diffusion/dataset.py`:

```python
def get_diffusion_dataloader(dataset_name, nsamples, ...):
    # Add handling for your dataset format
    if dataset_name == "your_custom_dataset":
        return _load_your_dataset(dataset_name, nsamples)
    ...
```

The default `coco2014` dataset works for most text-to-image models. Custom
datasets need a TSV file with `id` and `caption` columns.

## Step 6: Test

```python
def test_your_diffusion_model():
    ar = AutoRound(
        "your-org/your-diffusion-model",
        scheme="W4A16",
        iters=2,
        nsamples=4,
        num_inference_steps=5,
        guidance_scale=7.5,
    )
    compressed_model, layer_config = ar.quantize()
    assert len(layer_config) > 0, "No layers quantized"
    ar.save_quantized(output_dir="./test_output", format="fake")
```

For hybrid models, test both phases:
```python
ar = AutoRound(
    "your-hybrid-model",
    quant_ar=True,
    quant_dit=True,
    iters=2,
    nsamples=4,
)
```

## Checklist

- [ ] `is_diffusion_model()` detects model (or forced via extra_config)
- [ ] Transformer block class name identified
- [ ] `output_configs` entry added with correct output tensor names and order
- [ ] Pipeline runs without errors during calibration
- [ ] Custom `pipeline_fn` provided if non-standard API
- [ ] For hybrid: AR component registered in `HYBRID_AR_COMPONENTS`
- [ ] For hybrid: AR block handler in `SPECIAL_MULTIMODAL_BLOCK`
- [ ] For hybrid: DiT output config in `hybrid.py`
- [ ] Quantization produces valid layers (check "Quantized X/Y layers" log)
- [ ] Export to `fake` format works
- [ ] README.md + README_CN.md updated

## Key Files

| File | Purpose |
|------|---------|
| `auto_round/compressors/diffusion/compressor.py` | `DiffusionCompressor`, `output_configs` dict |
| `auto_round/compressors/diffusion/hybrid.py` | `HybridCompressor`, `HYBRID_AR_COMPONENTS` |
| `auto_round/compressors/diffusion/dataset.py` | Calibration dataset loading |
| `auto_round/utils/model.py` | `is_diffusion_model()`, `diffusion_load_model()` |
| `auto_round/special_model_handler.py` | AR block handlers for hybrid models |
| `auto_round/autoround.py` | Model type routing (diffusion vs hybrid vs LLM) |

## Reference: Existing Adaptations

| Model | Type | What Was Adapted |
|-------|------|-----------------|
| FLUX.1-dev | Pure DiT | `output_configs` for `FluxTransformerBlock`/`FluxSingleTransformerBlock` |
| GLM-Image | Hybrid AR+DiT | `HYBRID_AR_COMPONENTS` + `SPECIAL_MULTIMODAL_BLOCK` + DiT `output_configs` |
| NextStep | Custom pipeline | `pipeline_fn` parameter for non-standard inference API |
