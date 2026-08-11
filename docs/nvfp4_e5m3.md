# NVFP4 E5M3 Quantization (Experimental)

> **Note**: This is an experimental feature. APIs, formats, and behaviours may
> change in future releases.

## Overview

NVFP4 E5M3 is a quantization scheme that stores weights in FP4 (E2M1) format
with **unsigned E5M3 block scales** (instead of the global-scale approach used
by standard NVFP4).  Activations are quantized on-the-fly via a QDQ
(Quantize-Dequantize) pass.

Key properties:

| Property | Value |
|---|---|
| Preset name | `NVFP4_E5M3` |
| Weight dtype | FP4 E2M1 (`nvfp4_v2`) |
| Scale dtype | UE5M3 (per-block) |
| Group size | 16 |
| Activation bits | 4 (dynamic QDQ) |

---

## Supported Output Formats

| Format flag | Description |
|---|---|
| `auto_round` (default) | Packed `.weight_packed` + `.weight_scale` tensors; loadable with AutoRound inference |
| `llm_compressor` | `nvfp4-e5m3-pack-quantized` compressed-tensors layout |
| `fake` | Full-precision (BF16) `.weight` tensors with activation QDQ at runtime; no packing overhead |

---

## Supported Inference Backends

| Backend | Description |
|---|---|
| `auto_round:cute_nvfp4_e5m3` | CUDA CuTe DSL kernel (dequant weight + QDQ act). Requires `cutlass` package and SM80+. **Fastest**. |
| `auto_round:torch_nvfp4_e5m3` | Pure PyTorch dequant + QDQ act. Works on any CUDA device. |
| `auto_round:fake` | BF16 weight + activation QDQ only (no weight packing). For debugging / profiling. |

---

## Usage

### Standard AutoRound Flow

```python
from auto_round import AutoRound
from auto_round.schemes import NVFP4_E5M3

model, tokenizer = ...  # load your model

autoround = AutoRound(model, tokenizer, scheme=NVFP4_E5M3)
autoround.quantize()

# Save as packed auto_round format (default)
autoround.save_quantized("output_dir")

# Or save as llm_compressor format
autoround.save_quantized("output_dir", format="llm_compressor")

# Or save as fake format (full-precision weights + QDQ at runtime)
autoround.save_quantized("output_dir", format="fake")
```

### Model-Free Quantization (no GPU training required)

```bash
# Pack a pre-quantized model (from HuggingFace or local dir)
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

### Loading a Quantized Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import auto_round  # registers the AutoRound inference hooks

model = AutoModelForCausalLM.from_pretrained(
    "./nvfp4_e5m3_model",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("./nvfp4_e5m3_model")
```

---

## Environment Variables

### `AR_NVFP4_E5M3_CACHE_HP_WEIGHT`

Controls whether NVFP4 E5M3 `QuantLinear` caches a dequantized high-precision
(BF16) weight after the first forward pass instead of dequantizing on every
call.

| Value | Behaviour |
|---|---|
| `0` / `false` / `no` / `off` (default) | Dequantize on every forward pass (lower memory, higher compute) |
| `1` / `true` / `yes` / `on` | Cache the dequantized weight after first forward (lower compute, higher memory) |

When caching is enabled, the packed weight buffers are released after the
cache is materialized, trading lower runtime overhead for higher steady-state
memory usage.

```bash
export AR_NVFP4_E5M3_CACHE_HP_WEIGHT=1
```

---

## Accuracy Results (Llama-3.1-8B-Instruct, LAMBADA OpenAI)

| Configuration | Time | Accuracy |
|---|---|---|
| BF16 baseline | 1m 5s | 0.7213 |
| `auto_round` + `auto_round:cute_nvfp4_e5m3` | 1m 48s | 0.7196 |
| `auto_round` + `auto_round:torch_nvfp4_e5m3` | 4m 23s | 0.7204 |
| `llm_compressor` + `auto_round:cute_nvfp4_e5m3` | 1m 48s | 0.7196 |
| `fake` + `auto_round:fake` | 3m 26s | 0.7204 |
