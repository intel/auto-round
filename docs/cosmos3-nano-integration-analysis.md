# Cosmos3-Nano Integration Analysis for AutoRound

## Model Overview

**nvidia/Cosmos3-Nano** (16B params) is NVIDIA's omni-modal world model built on a Mixture-of-Transformers (MoT) architecture. It has two complementary transformers:

- **Reasoner tower** (autoregressive) — text + vision understanding, generates text tokens
- **Generator tower** (diffusion) — generates images, video, audio, and action trajectories

### Transformers Integration

`Cosmos3OmniForConditionalGeneration` is a thin subclass of `Qwen3VLForConditionalGeneration` (transformers >= 5.11.0). Loading via this class gets only the **Reasoner tower**.

### Module Tree (from Qwen3VL structure)

```
Cosmos3OmniForConditionalGeneration
├── model (Cosmos3OmniModel / Qwen3VLModel)
│   ├── visual (Qwen3VLVisionModel)
│   │   ├── patch_embed (Conv2D)
│   │   ├── pos_embed (Embedding)
│   │   ├── rotary_pos_emb (RotaryEmbedding)
│   │   ├── blocks: ModuleList (27 layers)  ← vision encoder blocks
│   │   └── merger / deepstack_merger_list
│   └── language_model (Qwen3VLTextModel)
│       ├── embed_tokens (Embedding)
│       ├── layers: ModuleList (36 layers)  ← text decoder layers
│       │   └── [0].self_attn (q/k/v/o_proj, q_norm/k_norm)
│       │   └── [0].mlp (gate/up/down_proj)
│       ├── norm (RMSNorm)
│       └── rotary_emb (MRotaryEmbedding)
└── lm_head (nn.Linear)
```

### Key Config

| Property | Value |
|----------|-------|
| model_type | `cosmos3_omni` |
| architectures | `["Cosmos3ForConditionalGeneration"]` |
| Text hidden_size | 4096 |
| Text layers | 36 |
| Text heads | 32, 8 KV heads (GQA) |
| Text intermediate | 12288 |
| Vision depth | 27 |
| Vision hidden_size | 1152 |
| Vocab | 151936 |
| Requires transformers >= 5.11.0 |
| Requires `trust_remote_code` for loading |

## Current AutoRound Support Status

**Not supported.** No references to `cosmos3_omni` or `Cosmos3` exist in the auto-round codebase.

## Critical Detection Issue: `is_diffusion_model()` Returns True

The model repo contains both `config.json` (transformers) and `model_index.json` (diffusers pipeline marker). The function `is_diffusion_model()` in `auto_round/utils/model.py` checks for `model_index.json` as a fallback, causing `nvidia/Cosmos3-Nano` to be classified as a "diffusion" model (wrong — we want "mllm").

When already loaded as a `torch.nn.Module`, `is_diffusion_model()` correctly returns False (it's not a `DiffusionPipeline`), and `is_mllm_model()` returns True (due to vision modules). So **pre-loaded models work but string paths don't**.

## Integration Steps

### 1. Fix Model Type Detection

**File**: `auto_round/utils/model.py` — `is_diffusion_model()`

Add `cosmos3_omni` to skip the `model_index.json` check when AutoConfig returns this model_type. The model has both config.json and model_index.json; its primary interface is as an MLLM, not a diffusion pipeline.

### 2. Add Model Loading Support

**File**: `auto_round/utils/model.py` — `mllm_load_model()`

Add a special case for `model_type == "cosmos3_omni"`:
- Check `transformers.__version__ >= 5.11.0`
- Load via `transformers.Cosmos3OmniForConditionalGeneration.from_pretrained()` (or the architecture's `ForConditionalGeneration` class)
- Load tokenizer via `AutoTokenizer.from_pretrained()`
- Load processor via `AutoProcessor.from_pretrained()` (Qwen3VL-style with `apply_chat_template`)
- Load image_processor via `AutoImageProcessor.from_pretrained()`

### 3. Add Multimodal Block Handler

**File**: `auto_round/special_model_handler.py`

Register in `SPECIAL_MULTIMODAL_BLOCK`:

```python
SPECIAL_MULTIMODAL_BLOCK["cosmos3_omni"] = _get_cosmos3_multimodal_block
```

The block handler function returns these block names:
- `model.language_model.layers.N` (always quantized)
- `model.visual.blocks.N` (when `quant_vision=True`)

### 4. Add to Support Lists

**File**: `auto_round/special_model_handler.py`

```python
SUPPORT_ONLY_TEXT_MODELS.append("cosmos3_omni")  
# Text-only calibration works; model_type not in NOT_SUPPORT_ONLY_TEXT_MODELS
```

### 5. Register Calibration Template

**File**: `auto_round/compressors/mllm/template.py`

Register a template for `cosmos3_omni` — uses the default HF processor (same as Qwen3-VL):

```python
_register_template("cosmos3_omni", default_dataset="NeelNanda/pile-10k", processor=PROCESSORS["hf"])
```

Without this registration, `get_template()` falls back to `TEMPLATES["default"]`, which works for text-only calibration but should be explicit.

### 6. Special Model Handler (Maybe Not Needed)

Since Cosmos3 inherits from Qwen3VL, its forward method accepts the standard `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`, etc. For text-only calibration, `pixel_values` can be None. **No custom forward pass should be needed.**

## What Likely Works Already (Pre-loaded Model Path)

If the user pre-loads the model:

```python
model = Cosmos3OmniForConditionalGeneration.from_pretrained("nvidia/Cosmos3-Nano", device_map="auto")
ar = AutoRound(model, scheme="W4A16", iters=2, nsamples=2)
```

- ✅ `detect_model_type()` → "mllm" (not a DiffusionPipeline, has vision modules)
- ✅ `get_block_names()` → finds `model.language_model.layers` via ModuleList search
- ✅ `SUPPORT_ONLY_TEXT_MODELS` → would need to be added first
- ✅ Calibration → templates fallback to "default" which works for text-only
- ❌ Template not registered (falls back, works but logs warning)

## What Needs Code Changes

| Step | File | Change | Priority |
|------|------|--------|----------|
| 1 | `model.py` | Fix `is_diffusion_model()` for `cosmos3_omni` | **High** |
| 2 | `model.py` | Add loading case in `mllm_load_model()` | **High** |
| 3 | `special_model_handler.py` | Add block handler + support lists | **Medium** |
| 4 | `template.py` | Register template | **Medium** |
| 5 | Upgrade transformers >= 5.11.0 | Environment requirement | **Required** |

## Test Plan

1. **Basic quantization** (text-only):
   ```python
   from auto_round import AutoRound
   model = Cosmos3OmniForConditionalGeneration.from_pretrained("nvidia/Cosmos3-Nano", device_map="auto")
   ar = AutoRound(model, scheme="W4A16", iters=2, nsamples=2)
   compressed_model, layer_config = ar.quantize()
   assert len(layer_config) > 0
   ```

2. **Vision encoder quantization**:
   ```python
   ar = AutoRound(model, scheme="W4A16", iters=2, nsamples=2, quant_nontext_module=True)
   ```

3. **Save and reload**:
   ```python
   ar.save_quantized(output_dir="./cosmos3_w4a16", format="auto_round")
   ```

## Key Files to Modify

| File | Modification |
|------|-------------|
| `auto_round/utils/model.py` | `is_diffusion_model()` + `mllm_load_model()` |
| `auto_round/special_model_handler.py` | `SPECIAL_MULTIMODAL_BLOCK`, `SUPPORT_ONLY_TEXT_MODELS` |
| `auto_round/compressors/mllm/template.py` | Register template |

## Dependencies

- `transformers >= 5.11.0` (support `Cosmos3OmniForConditionalGeneration`)
- `diffusers >= 0.37.1` (for full pipeline, not needed for transformers-only path)
- `torch >= 2.0.0` (standard)
