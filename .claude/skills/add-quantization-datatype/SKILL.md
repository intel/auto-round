---
name: add-quantization-datatype
description: "Add a new quantization data type to AutoRound (e.g., INT, FP8, MXFP, NVFP, GGUF variants). Use when implementing a new weight/activation quantization scheme, registering a new quant function, or extending the data_type registry."
---

# Adding a New Quantization Data Type to AutoRound

## Overview

This skill guides you through adding a new quantization data type to AutoRound.
A data type defines how tensors are quantized and dequantized (e.g., INT
symmetric, FP8 per-block, MXFP4). Each data type is registered via a decorator
and plugged into the quantization loop automatically.

## Prerequisites

Before starting, determine:

1. **Data type category**: Integer (INT), floating-point (FP8, BF16), mixed-format
   (MXFP, NVFP), or GGUF variant
2. **Quantization parameters**: bits, group_size, symmetric/asymmetric, scale format
3. **Special requirements**: Block-wise scaling, imatrix support, custom rounding

## Step 1: Create the Data Type Module

Create a new file at `auto_round/data_type/your_dtype.py`.

### Function Signature

The quantization function must follow this contract:

```python
from auto_round.data_type.register import register_dtype
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad


@register_dtype("your_dtype_name")
def quant_tensor_your_dtype(
    tensor,
    bits=4,
    group_size=128,
    v=0,
    min_scale=0,
    max_scale=0,
    scale_dtype=torch.float16,
    q_scale_thresh=0,
    weight_fp8_max_scale=0,
    imatrix=None,
    **kwargs
):
    """Quantize a tensor using your data type.

    Args:
        tensor: The weight tensor to quantize (2D: [out_features, in_features])
        bits: Number of quantization bits
        group_size: Number of elements per quantization group
        v: Learnable perturbation tensor (for SignSGD optimization, same shape as tensor)
        min_scale: Minimum scale clipping value
        max_scale: Maximum scale clipping value
        scale_dtype: Data type for quantization scales
        q_scale_thresh: Threshold for scale quantization
        weight_fp8_max_scale: Max scale for FP8 weight quantization
        imatrix: Importance matrix for weighted quantization (optional)
        **kwargs: Additional parameters

    Returns:
        tuple: (qdq_tensor, scale, zp)
            - qdq_tensor: Quantized-then-dequantized tensor (same shape as input)
            - scale: Quantization scale tensor
            - zp: Zero-point tensor (or maxq for symmetric)
    """
    # 1. Apply perturbation
    tensor = tensor + v

    # 2. Reshape by group_size
    orig_shape = tensor.shape
    tensor, orig_out_features = reshape_pad_tensor_by_group_size(tensor, group_size)

    # 3. Compute scale and zero-point
    # ... your quantization logic here ...

    # 4. Quantize and dequantize (Straight-Through Estimator for gradients)
    from auto_round.data_type.utils import round_ste

    tensor_q = round_ste(tensor / scale) + zp  # or your rounding logic
    qdq_tensor = (tensor_q - zp) * scale

    # 5. Revert padding
    qdq_tensor = revert_tensor_by_pad(qdq_tensor, orig_out_features, orig_shape)

    return qdq_tensor, scale, zp
```

### Key Utilities from `auto_round/data_type/utils.py`

- `reshape_pad_tensor_by_group_size(tensor, group_size)` — Reshape tensor into
  groups, padding if needed
- `revert_tensor_by_pad(tensor, orig_out_features, orig_shape)` — Undo padding
  and restore original shape
- `round_ste(x)` — Round with Straight-Through Estimator (gradient passthrough)
- `get_quant_func(data_type, bits)` — Look up registered quant function

## Step 2: Register Multiple Variants (Optional)

If your data type has variants, register them all:

```python
@register_dtype(["your_dtype", "your_dtype_v2"])
def quant_tensor_your_dtype(tensor, bits=4, group_size=128, v=0, **kwargs):
    variant = kwargs.get("data_type", "your_dtype")
    # Branch logic based on variant
    ...
```

## Step 3: Register in `__init__.py`

Add your import to `auto_round/data_type/__init__.py`:

```python
import auto_round.data_type.your_dtype
```

This triggers the `@register_dtype` decorator, populating `QUANT_FUNC_WITH_DTYPE`.

## Step 4: Add Scheme Preset (If Needed)

If your data type corresponds to a named scheme (e.g., "W4A16", "MXFP4"), add
it to `auto_round/schemes.py`:

```python
YOUR_SCHEME = QuantizationScheme(
    bits=4,
    group_size=32,
    sym=True,
    data_type="your_dtype",
)
PRESET_SCHEMES["YOUR_SCHEME"] = YOUR_SCHEME
```

## Step 5: Update Export Format Support

If your data type needs specific export handling, update the relevant export
format's `support_schemes` list in the corresponding `OutputFormat` subclass
under `auto_round/export/`.

## Step 6: Test

Create tests in the appropriate test directory (e.g., `test/test_cuda/` or
`test/test_cpu/`):

```python
def test_your_dtype_quantization(tiny_opt_model_path, dataloader):
    ar = AutoRound(
        tiny_opt_model_path,
        bits=4,
        group_size=128,
        data_type="your_dtype",
        dataset=dataloader,
        iters=2,
        nsamples=2,
    )
    compressed_model, _ = ar.quantize()
    # Verify model produces valid outputs
```

## Reference: Existing Data Type Implementations

| File | Data Types | Key Patterns |
|------|-----------|--------------|
| `auto_round/data_type/int.py` | int (sym/asym) | Basic INT quantization with min/max scaling |
| `auto_round/data_type/fp8.py` | fp8_e4m3fn, fp8_e5m2, fp8_dynamic, fp8_block | Per-tensor/block FP8 with amax-based scaling |
| `auto_round/data_type/mxfp.py` | mx_fp, mx_fp_rceil | Microscaling with shared exponent |
| `auto_round/data_type/nvfp.py` | nv_fp, nv_fp4 | NVIDIA FP4 with static group scale |
| `auto_round/data_type/w4fp8.py` | w4fp8 | Hybrid INT4 weight + FP8 activation |
| `auto_round/data_type/gguf.py` | GGUF Q2_K through Q8_0 | Super-block quantization with multiple sub-types |
