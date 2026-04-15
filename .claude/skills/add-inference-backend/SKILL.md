---
name: add-inference-backend
description: "Add a new hardware inference backend to AutoRound for deploying quantized models (e.g., CUDA/Marlin, Triton, IPEX/CPU, HPU, ARK). Use when implementing QuantLinear kernels, registering backend capabilities, or enabling quantized model inference on a new hardware platform."
---

# Adding a New Inference Backend to AutoRound

## Overview

This skill guides you through adding a new inference backend for running
quantized models on a specific hardware platform. A backend defines how
quantized weights are unpacked and computed at inference time. AutoRound
automatically selects the best available backend based on hardware, quantization
config, and priority.

## Prerequisites

Before starting, determine:

1. **Target hardware**: CPU (Intel/AMD), CUDA GPU, Intel XPU, Habana HPU, etc.
2. **Supported quantization configs**: Which bit-widths, group sizes, and data
   types your backend handles
3. **Kernel implementation**: Triton, CUDA C++, PyTorch native, or external
   library (e.g., GPTQModel Marlin)
4. **Packing format**: How quantized weights are stored in memory

## Step 1: Register Backend Info

Edit `auto_round/inference/backend.py` to register your backend's capabilities:

```python
BackendInfos["auto_round:your_backend"] = BackendInfo(
    device=["cuda"],                    # Supported devices
    sym=[True, False],                  # Symmetric and/or asymmetric
    packing_format=["auto_round"],      # Compatible packing formats
    bits=[2, 4, 8],                     # Supported bit-widths
    group_size=[32, 64, 128, -1],       # Supported group sizes (-1 = per-channel)
    compute_dtype=["float16", "bfloat16"],  # Compute precision
    data_type=["int"],                  # Quantization data types
    act_bits=[16, 32],                  # Activation bit-widths (16 = WxA16)
    priority=2,                         # Higher = preferred (0-5 typical range)
    checkers=[your_feature_checker],    # Validation functions (optional)
    alias=["your_backend_short"],       # Alternative names (optional)
    requirements=["some_package>=1.0"], # Required packages (optional)
    systems=["linux"],                  # OS restriction (optional)
)
```

### BackendInfo Fields Reference

| Field | Type | Description |
|-------|------|-------------|
| `device` | `list[str]` | Hardware targets: `"cpu"`, `"cuda"`, `"xpu"`, `"hpu"` |
| `sym` | `list[bool]` | `True` for symmetric, `False` for asymmetric |
| `packing_format` | `list[str]` | How weights are packed: `"auto_round"`, `"auto_gptq"`, etc. |
| `bits` | `list[int]` | Supported weight bit-widths |
| `group_size` | `list[int]` | Group sizes; `-1` means per-channel |
| `compute_dtype` | `list[str]` | Compute precision during inference |
| `data_type` | `list[str]` | Quantization data types: `"int"`, `"nv_fp"`, `"mx_fp"` |
| `act_bits` | `list[int]` | Activation bits: `[16, 32]` for weight-only, `[8]` for W8A8 |
| `priority` | `int` | Selection priority (higher wins when multiple backends match) |
| `checkers` | `list[Callable]` | Functions to validate layer compatibility |
| `alias` | `list[str]` | Alternative names for CLI/API usage |
| `requirements` | `list[str]` | pip-installable dependency specifications |
| `systems` | `list[str]` | OS names: `"linux"`, `"windows"`, `"darwin"` |

### Checker Functions

Use these pre-built checkers or create your own:

```python
# Require in_features and out_features divisible by 32
from auto_round.inference.backend import feature_multiply_checker_32

# Require in_features divisible by group_size
from auto_round.inference.backend import in_feature_checker_group_size

# Custom checker
def your_feature_checker(in_feature, out_feature, config):
    """Check if layer dimensions are compatible with your backend."""
    return (
        in_feature % 64 == 0
        and out_feature % 64 == 0
        and config["group_size"] in [64, 128]
    )
```

## Step 2: Implement QuantLinear Module

Create `auto_round_extension/your_device/qlinear_your_backend.py`:

```python
import torch
import torch.nn as nn

QUANT_TYPE = "your_backend"

class QuantLinear(nn.Module):
    """Quantized linear layer for your backend.

    Stores packed quantized weights and performs dequantize-then-matmul
    (or fused quantized matmul) at inference time.
    """

    QUANT_TYPE = QUANT_TYPE

    def __init__(self, bits, group_size, in_features, out_features,
                 bias=True, sym=True, **kwargs):
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.in_features = in_features
        self.out_features = out_features
        self.sym = sym

        # Register packed weight buffers
        # Example: INT4 packed into INT32
        pack_factor = 32 // bits
        self.register_buffer(
            "qweight",
            torch.zeros(in_features // pack_factor, out_features, dtype=torch.int32),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (in_features // group_size, out_features),
                dtype=torch.float16,
            ),
        )
        if not sym:
            self.register_buffer(
                "qzeros",
                torch.zeros(
                    (in_features // group_size, out_features // pack_factor),
                    dtype=torch.int32,
                ),
            )
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def forward(self, x):
        """Dequantize weights and compute linear transformation."""
        weight = self._dequantize()
        out = torch.matmul(x, weight.T)
        if self.bias is not None:
            out += self.bias
        return out

    def _dequantize(self):
        """Unpack and dequantize weights."""
        # Implement your dequantization kernel here
        # Can use Triton, CUDA, or PyTorch operations
        ...

    @classmethod
    def pack(cls, linear, scales, zeros, bias=None):
        """Pack a standard nn.Linear into this quantized format.

        Called during export to convert calibrated weights into packed format.
        """
        ...
```

## Step 3: Register QuantLinear for Auto-Discovery

Ensure your QuantLinear is discoverable by the model conversion system in
`auto_round/inference/convert_model.py`. The system typically looks for modules
in the `auto_round_extension/` directory matching the backend name.

## Step 4: Add Extension `__init__.py`

Create `auto_round_extension/your_device/__init__.py` if the directory is new:

```python
# Auto-Round extension for YourDevice backend
```

## Step 5: Test

### Unit test for the QuantLinear

```python
def test_your_backend_qlinear():
    from auto_round_extension.your_device.qlinear_your_backend import QuantLinear

    ql = QuantLinear(bits=4, group_size=128, in_features=256, out_features=512)
    x = torch.randn(1, 256, dtype=torch.float16, device="cuda")
    out = ql(x)
    assert out.shape == (1, 512)
```

### End-to-end test

```python
def test_your_backend_e2e(tiny_opt_model_path, dataloader):
    ar = AutoRound(
        tiny_opt_model_path,
        bits=4,
        group_size=128,
        dataset=dataloader,
        iters=2,
        nsamples=2,
    )
    compressed_model, _ = ar.quantize()
    ar.save_quantized(output_dir="./tmp_backend_test", format="auto_round")

    # Load and verify inference with your backend
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("./tmp_backend_test")
    tokenizer = AutoTokenizer.from_pretrained("./tmp_backend_test")
    inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=10)
    assert outputs.shape[1] > inputs["input_ids"].shape[1]
```

## Reference: Existing Backend Implementations

| Backend Key | Device | Extension Dir | Key Patterns |
|-------------|--------|---------------|--------------|
| `auto_gptq:exllamav2` | CUDA | `cuda/` | Marlin kernels via GPTQModel, priority=3 |
| `auto_round:triton_*` | CUDA | `triton/` | Triton JIT-compiled kernels |
| `auto_round:torch_*` | CPU/CUDA | `torch/` | Pure PyTorch fallback |
| `auto_round:ark` | ARK | `ark/` | ARK accelerator kernels |
| HPU backends | HPU | `hpu/` | Habana Gaudi optimized |
| IPEX backends | CPU | `ipex/` | Intel Extension for PyTorch |

## Key Registration Points

| What | Where | Mechanism |
|------|-------|-----------|
| Backend capabilities | `auto_round/inference/backend.py` | `BackendInfos["name"]` dict |
| QuantLinear module | `auto_round_extension/<device>/qlinear_*.py` | `QUANT_TYPE` class attr |
| Model conversion | `auto_round/inference/convert_model.py` | Auto-discovery |
| Feature checkers | `auto_round/inference/backend.py` | `functools.partial` wrappers |
