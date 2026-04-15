---
name: add-export-format
description: "Add a new model export format to AutoRound (e.g., auto_round, auto_gptq, auto_awq, gguf, llm_compressor). Use when implementing a new quantized model serialization format, adding a new packing method, or extending export compatibility for deployment frameworks like vLLM, SGLang, or llama.cpp."
---

# Adding a New Export Format to AutoRound

## Overview

This skill guides you through adding a new export format for saving quantized
models. An export format defines how quantized weights, scales, and zero-points
are packed and serialized for deployment. Each format is registered via the
`@OutputFormat.register()` decorator in `auto_round/formats.py`.

## Prerequisites

Before starting, determine:

1. **Target deployment framework**: vLLM, llama.cpp, Transformers, SGLang, etc.
2. **Packing scheme**: How quantized weights are packed (e.g., INT32 packing,
   safetensors, GGUF binary)
3. **Supported quantization schemes**: Which bit-widths, data types, and configs
   are compatible
4. **Config format**: How quantization metadata is stored (e.g., `quantize_config.json`,
   GGUF metadata)

## Step 1: Create Export Module Directory

Create a new directory:

```
auto_round/export/export_to_yourformat/
├── __init__.py
└── export.py
```

## Step 2: Implement the Export Logic

In `export.py`, implement two core functions:

### `pack_layer()`

Packs a single quantized layer's weights, scales, and zero-points:

```python
def pack_layer(layer_name, model, backend, output_dtype=torch.float16):
    """Pack a quantized layer for serialization.

    Args:
        layer_name: Full module path (e.g., "model.layers.0.self_attn.q_proj")
        model: The quantized model
        backend: Backend configuration string
        output_dtype: Output tensor dtype

    Returns:
        dict: Packed tensors ready for serialization
    """
    import auto_round_extension.cuda.qlinear_triton as qlinear_triton

    layer = get_module(model, layer_name)
    device = layer.weight.device

    # Get quantization parameters from layer
    bits = layer.bits
    group_size = layer.group_size
    scale = layer.scale
    zp = layer.zp
    weight = layer.weight

    # Pack weights according to your format
    packed_weight = _pack_weights(weight, bits, group_size)

    return {
        f"{layer_name}.qweight": packed_weight,
        f"{layer_name}.scales": scale,
        f"{layer_name}.qzeros": zp,
    }
```

### `save_quantized_as_yourformat()`

Saves the complete quantized model:

```python
def save_quantized_as_yourformat(output_dir, model, tokenizer, layer_config,
                                  serialization_dict=None, **kwargs):
    """Save quantized model in your format.

    Args:
        output_dir: Directory to save to
        model: The quantized model
        tokenizer: Model tokenizer
        layer_config: Per-layer quantization configuration
        serialization_dict: Pre-packed layer tensors (optional)
        **kwargs: Additional format-specific arguments
    """
    import os
    from safetensors.torch import save_file

    os.makedirs(output_dir, exist_ok=True)

    # 1. Pack all quantized layers (if not pre-packed)
    if serialization_dict is None:
        serialization_dict = {}
        for layer_name, config in layer_config.items():
            serialization_dict.update(pack_layer(layer_name, model, ...))

    # 2. Save weights
    save_file(serialization_dict, os.path.join(output_dir, "model.safetensors"))

    # 3. Save quantization config
    quant_config = {
        "quant_method": "yourformat",
        "bits": ...,
        "group_size": ...,
        # format-specific metadata
    }
    # Write config to output_dir

    # 4. Save tokenizer
    tokenizer.save_pretrained(output_dir)
```

## Step 3: Register the Format

Create the `OutputFormat` subclass in `auto_round/formats.py`:

```python
@OutputFormat.register("yourformat")
class YourFormat(OutputFormat):
    format_name = "yourformat"
    support_schemes = ["W4A16", "W8A16"]  # List supported scheme names

    def __init__(self, format: str, ar):
        super().__init__(format, ar)

    @classmethod
    def check_scheme_args(cls, scheme: QuantizationScheme) -> bool:
        """Check if a QuantizationScheme is compatible with this format."""
        return (
            scheme.bits in [4, 8]
            and scheme.data_type == "int"
            and scheme.act_bits >= 16
        )

    def pack_layer(self, layer_name, model, output_dtype=torch.float16):
        from auto_round.export.export_to_yourformat.export import pack_layer
        return pack_layer(layer_name, model, self.get_backend_name(), output_dtype)

    def save_quantized(self, output_dir, model, tokenizer, layer_config,
                       serialization_dict=None, **kwargs):
        from auto_round.export.export_to_yourformat.export import save_quantized_as_yourformat
        return save_quantized_as_yourformat(
            output_dir, model, tokenizer, layer_config,
            serialization_dict=serialization_dict, **kwargs
        )
```

## Step 4: Update SUPPORTED_FORMATS

Add your format name to the `SUPPORTED_FORMATS` list in `auto_round/utils/common.py`
(or wherever the constant is defined) so it appears in CLI help and validation.

## Step 5: Wire Up Backend Info (If Needed)

If your format requires specific inference backends, register them in
`auto_round/inference/backend.py`:

```python
BackendInfos["auto_round:yourformat"] = BackendInfo(
    device=["cuda"],
    sym=[True, False],
    packing_format=["yourformat"],
    bits=[4, 8],
    group_size=[32, 64, 128],
    priority=2,
)
```

## Step 6: Test

```python
def test_yourformat_export(tiny_opt_model_path, dataloader):
    ar = AutoRound(
        tiny_opt_model_path,
        bits=4,
        group_size=128,
        dataset=dataloader,
        iters=2,
        nsamples=2,
    )
    compressed_model, _ = ar.quantize()
    ar.save_quantized(output_dir="./tmp_yourformat", format="yourformat")

    # Verify saved files exist
    assert os.path.exists("./tmp_yourformat/model.safetensors")

    # Verify model can be loaded back
    from transformers import AutoModelForCausalLM
    loaded = AutoModelForCausalLM.from_pretrained("./tmp_yourformat")
```

## Reference: Existing Export Format Implementations

| Directory | Format Name | Key Patterns |
|-----------|-------------|--------------|
| `export_to_autoround/` | auto_round | Native format, QuantLinear packing, safetensors |
| `export_to_autogptq/` | auto_gptq | GPTQ-compatible INT packing |
| `export_to_awq/` | auto_awq | AWQ-compatible format |
| `export_to_gguf/` | gguf | Binary GGUF format with super-block quantization, uses `@register_qtype()` |
| `export_to_llmcompressor/` | llm_compressor | CompressedTensors format for vLLM |

## Key Registration Points

| What | Where | Mechanism |
|------|-------|-----------|
| Format class | `auto_round/formats.py` | `@OutputFormat.register("name")` |
| Support matrix | `OutputFormat.support_schemes` | Class attribute list |
| Backend info | `auto_round/inference/backend.py` | `BackendInfos["name"]` dict |
| CLI format list | `auto_round/utils/common.py` | `SUPPORTED_FORMATS` list |
