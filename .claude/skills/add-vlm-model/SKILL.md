---
name: add-vlm-model
description: "Add support for a new Vision-Language Model (VLM) to AutoRound, including multimodal block handler, calibration dataset template, and special model handling. Use when integrating a new VLM like LLaVA, Qwen2-VL, GLM-Image, Phi-Vision, or similar multi-modal models for quantization."
---

# Adding a New Vision-Language Model to AutoRound

## Overview

This skill guides you through adding support for a new Vision-Language Model
(VLM) to AutoRound. VLMs require special handling because they typically have
separate vision encoder and language model components, and calibration may need
multi-modal data.

The integration involves three parts:
1. **Multimodal Block Handler** — Tell AutoRound how to find quantizable blocks
2. **Calibration Template** — Define how to build calibration prompts
3. **Special Model Handler** — Handle model-specific forward pass quirks

## Prerequisites

Before starting, determine:

1. **Model architecture**: What sub-modules exist? (vision encoder, projector,
   language model, audio tower, etc.)
2. **Model type**: The `model_type` string from `config.json`
3. **Block structure**: Where are the transformer layers? (e.g.,
   `model.layers`, `thinker.model.layers`, `language_model.layers`)
4. **Text-only support**: Can the model be calibrated with text-only data?
5. **Batch size limitations**: Does the VLM have restrictions on batch size?

## Step 1: Add Multimodal Block Handler

Edit `auto_round/special_model_handler.py`:

### 1a. Create a block discovery function

```python
def _get_your_vlm_multimodal_block(model, quant_vision=False):
    """Get block names for YourVLM model.

    YourVLM structure:
    - model.vision_encoder.blocks: vision encoder
    - model.projector.layers: vision-language projector
    - model.language_model.layers: text decoder

    By default, only the text decoder is quantized. Set quant_vision=True
    to include vision encoder and projector blocks.
    """
    block_names = []

    if quant_vision:
        if hasattr(model, "model") and hasattr(model.model, "vision_encoder"):
            if hasattr(model.model.vision_encoder, "blocks"):
                block_names.append(
                    [f"model.vision_encoder.blocks.{i}" for i in range(len(model.model.vision_encoder.blocks))]
                )
        # Add projector if it has quantizable layers
        if hasattr(model, "model") and hasattr(model.model, "projector"):
            if hasattr(model.model.projector, "layers"):
                block_names.append([f"model.projector.layers.{i}" for i in range(len(model.model.projector.layers))])

    # Language model layers (always quantized)
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        if hasattr(model.model.language_model, "layers"):
            block_names.append(
                [f"model.language_model.layers.{i}" for i in range(len(model.model.language_model.layers))]
            )

    return block_names
```

### 1b. Register in the `SPECIAL_MULTIMODAL_BLOCK` dict

Find the `SPECIAL_MULTIMODAL_BLOCK` dictionary (in `special_model_handler.py`)
and add your model:

```python
SPECIAL_MULTIMODAL_BLOCK["your_vlm"] = _get_your_vlm_multimodal_block
```

The key must match the `model_type` from the model's `config.json`.

### 1c. Add to support lists

```python
# If your VLM supports text-only calibration (most do):
SUPPORT_ONLY_TEXT_MODELS.append("your_vlm")

# If your VLM has batch size limitations:
mllms_with_limited_bs = (
    ...,
    "your_vlm",
)
```

## Step 2: Add Calibration Template

### 2a. Create template JSON

Create `auto_round/compressors/mllm/templates/your_vlm.json`:

```json
{
    "model_type": "your_vlm",
    "format_user": "<|user|>\n{content}\n",
    "format_assistant": "<|assistant|>\n{content}\n",
    "format_system": "<|system|>\n{content}\n",
    "format_observation": "",
    "system": "",
    "separator": "",
    "stop_words": ["<|end|>"]
}
```

Adjust the template fields to match your model's chat format. Check the model's
`tokenizer_config.json` or documentation for the correct chat template.

### 2b. Register the template

Edit `auto_round/compressors/mllm/template.py`:

```python
_register_template(
    "your_vlm",
    default_dataset="liuhaotian/llava_conv_58k",  # or appropriate dataset
    processor=PROCESSORS["default"],  # or a custom processor
)
```

### 2c. Add a custom processor (if needed)

If your model requires special image/prompt processing for calibration, create a
processor in `auto_round/compressors/mllm/template.py`:

```python
def _your_vlm_processor(raw_data, model_path, seqlen, processor=None, **kwargs):
    """Process calibration data for YourVLM.

    Args:
        raw_data: Dataset samples
        model_path: Path to the model
        seqlen: Sequence length for calibration
        processor: The model's processor

    Returns:
        list: Processed samples ready for calibration
    """
    # Build prompts with images and text
    ...
```

Register it:
```python
PROCESSORS["your_vlm"] = _your_vlm_processor
```

## Step 3: Handle Special Forward Pass (If Needed)

If your VLM's `forward()` method is non-standard (e.g., requires special
kwargs, has multiple model components that need separate handling), add a
custom forward wrapper in `special_model_handler.py`:

```python
def _your_vlm_forward(model, **kwargs):
    """Custom forward pass for YourVLM during calibration."""
    # Handle special input processing
    # Route inputs to correct sub-models
    return model.language_model(**kwargs)
```

Register it in `_handle_special_model()`:

```python
def _handle_special_model(model):
    ...
    if hasattr(model, "config") and model.config.model_type == "your_vlm":
        from functools import partial

        model.forward = partial(_your_vlm_forward, model)
    return model
```

## Step 4: Add Custom Calibration Dataset (Optional)

If your model needs a specialized calibration dataset loader, create one in
`auto_round/calib_dataset.py` using the `@register_dataset` decorator:

```python
@register_dataset("your_vlm_dataset")
class YourVLMDataset:
    def __init__(self, dataset_name, model_path, seqlen, **kwargs): ...

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for sample in self.data:
            yield sample
```

## Step 5: Test

```python
def test_your_vlm_quantization():
    model_name = "your-org/your-vlm-small"
    ar = AutoRound(
        model_name,
        bits=4,
        group_size=128,
        iters=2,
        nsamples=2,
        quant_nontext_module=False,  # text-only quantization
    )
    compressed_model, _ = ar.quantize()
    ar.save_quantized(output_dir="./tmp_your_vlm", format="auto_round")
```

Test with vision quantization:
```python
ar = AutoRound(
    model_name,
    bits=4,
    group_size=128,
    quant_nontext_module=True,  # also quantize vision encoder
)
```

## Step 6: Update Documentation

1. Add your model to the supported VLM list in `README.md`
2. Update `README_CN.md` with the same changes (Chinese translation required)
3. Add example quantization script if the model has special usage patterns

## Reference: Existing VLM Implementations

| Model Type | Block Handler | Template | Special Forward |
|------------|--------------|----------|-----------------|
| `llava` | `_get_llava_multimodal_block` | llava template | No |
| `qwen2_vl` | `_get_qwen2_vl_multimodal_block` | qwen2_vl template | No |
| `qwen2_5_omni` | `_get_qwen2_5_omni_multimodal_block` | qwen2_5_omni template | Yes (`_qwen2_5_omni_forward`) |
| `qwen3_omni_moe` | `_get_qwen3_omni_moe_multimodal_block` | qwen3_omni_moe template | Yes (`_qwen3_omni_moe_forward`) |
| `deepseek_vl_v2` | `_get_deepseek_vl2_multimodal_block` | deepseek_vl_v2 template | Yes (`_deepseek_vl2_forward`) |
| `glm_image` | `_get_glm_image_multimodal_block` | glm_image template | No |
| `phi3_v` | via generic handler | phi3_v template | No |

## Key Registration Points

| What | Where | Mechanism |
|------|-------|-----------|
| Block handler | `special_model_handler.py` | `SPECIAL_MULTIMODAL_BLOCK[model_type]` |
| Text-only support | `special_model_handler.py` | `SUPPORT_ONLY_TEXT_MODELS` list |
| Batch limit | `special_model_handler.py` | `mllms_with_limited_bs` tuple |
| Template | `compressors/mllm/templates/*.json` | `_register_template()` |
| Processor | `compressors/mllm/template.py` | `PROCESSORS` dict |
| Custom forward | `special_model_handler.py` | `_handle_special_model()` |
| Dataset loader | `calib_dataset.py` | `@register_dataset()` |
