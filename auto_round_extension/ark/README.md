## What is AutoRound Kernel?

AutoRound Kernel (ARK) is a low-bit acceleration library for Intel platform, providing high-performance kernels for LLM inference including weight-only quantization, flash attention (with [SageAttention](https://github.com/thu-ml/SageAttention) v1 support), and Mixture-of-Experts (MOE) computation.

The kernels are optimized for the following CPUs:
* Intel Xeon Scalable processor (formerly Sapphire Rapids, and Emerald Rapids)
* Intel Xeon 6 processors (formerly Sierra Forest and Granite Rapids)

The kernels are optimized for the following GPUs:
* Intel Arc B-Series Graphics and Intel Arc Pro B-Series Graphics
  (formerly Battlemage)

## Key Features

AutoRound Kernel provides the following computational capabilities for LLM inference:

### Weight-Only Quantization (WOQ) Linear
| Weight dtype     |          Compute dtype           |    Scale dtype    | Algorithm<sup>[1]</sup> |
|------------------|:--------------------------------:|:-----------------:|:-----------------------:|
| INT8             | INT8<sup>[2]</sup> / BF16 / FP32 |    BF16 / FP32    |       sym / asym        |
| INT4             |        INT8 / BF16 / FP32        |    BF16 / FP32    |       sym / asym        |
| INT3             |        INT8 / BF16 / FP32        |    BF16 / FP32    |       sym / asym        |
| INT2             |        INT8 / BF16 / FP32        |    BF16 / FP32    |       sym / asym        |
| INT5             |        INT8 / BF16 / FP32        |    BF16 / FP32    |       sym / asym        |
| INT6             |        INT8 / BF16 / FP32        |    BF16 / FP32    |       sym / asym        |
| INT7             |        INT8 / BF16 / FP32        |    BF16 / FP32    |       sym / asym        |
| INT1             |        INT8 / BF16 / FP32        |    BF16 / FP32    |       sym / asym        |
| FP8 (E4M3, E5M2) |           BF16 / FP32            | FP32 / FP8 (E8M0) |           NA            |
| FP4 (E2M1)       |           BF16 / FP32            |    BF16 / FP32    |           NA            |

### XPU Weight-Only Quantization
| Weight dtype     |  Compute dtype |     Scale dtype   |  Algorithm |
|------------------|:--------------:|:-----------------:|:----------:|
| INT8             |  INT8 / FP16   |       FP16        |    sym     |
| INT4             |  INT8 / FP16   |       FP16        |    sym     |
| INT2             |  INT8 / FP16   |       FP16        |    sym     |
| FP8 (E4M3, E5M2) |      FP16      | FP16 / FP8 (E8M0) |     NA     |

<sup>[1]</sup>: Quantization algorithms for integer types: symmetric or asymmetric.  
<sup>[2]</sup>: Includes dynamic activation quantization; results are dequantized to floating-point formats.  

### Flash Attention (XPU only)

ARK provides multiple attention backends for prefill and decode, with [SageAttention](https://github.com/thu-ml/SageAttention) v1 support — offering INT8-quantized attention variants for higher throughput on Intel Arc GPUs. (SageAttention v2/v3 are under development.)

| Backend | Description | Q/K/V dtype | Head dim | Features |
|---------|-------------|-------------|----------|----------|
| `sdpa` | Standard flash attention (FP16/BF16) | FP16 / BF16 | 64, 96, 128, 192 | Causal mask, additive mask, GQA |
| `sagev1` | **SageAttention v1** with INT8 Q/K quantization | FP16 / BF16 | 64, 128 | Block-wise INT8 QK, PV in half |
| `sagev1_pvi8` | **SageAttention v1** with INT8 Q/K/V quantization | FP16 / BF16 | 64, 128 | Block-wise INT8 QK + INT8 PV |
| `sage` | Low-level SageAttention with pre-quantized INT8 Q/K | INT8 (Q/K), FP16 (V) | 64, 128 | External Q/K scales |
| `sage_pvi8` | Low-level SageAttention with pre-quantized INT8 Q/K/V | INT8 (Q/K/V) | 64, 128 | External Q/K/V scales |
| `sage_dynquant` | SageAttention with fused dynamic INT8 quantization | FP16 | 64, 128 | Auto-quantizes Q/K internally |

All attention backends support both `HND` (`[B, H, N, D]`) and `NHD` (`[B, N, H, D]`) tensor layouts, as well as non-contiguous (sliced) input tensors.

### MOE GEMM (XPU only)

Grouped GEMM for Mixture-of-Experts layers, supporting FP16/BF16 with variable token counts per expert.


## Installation

### 1. Install via pip
```bash
pip install auto-round-lib
```

### 2. Install from Source

Requires a sourced oneAPI environment (2025.3+ recommended for SYCL TLA support).

```bash
# Source oneAPI environment
source /opt/intel/oneapi/setvars.sh

# Build and install
pip install . --no-build-isolation
# or
python setup.py bdist_wheel; pip install dist/*
```

The build system automatically detects the oneAPI version. SYCL TLA (Tensor Linear Algebra) kernels are enabled when oneAPI >= 2025.3.

### Validated Hardware Environment

#### CPU based on [Intel 64 architecture or compatible processors](https://en.wikipedia.org/wiki/X86-64):
* Intel Xeon Scalable processor (Granite Rapids)

#### GPU built on Intel's Xe architecture:
* Intel Arc B-Series Graphics (Battlemage)

## API Reference

### QuantLinear (Weight-Only Quantization)

ARK exposes a unified weight-only linear interface through `QuantLinear`, `QuantLinearGPTQ`, `QuantLinearAWQ`, and `QuantLinearFP8`. Please refer to the [QLinear](auto_round_kernel/qlinear.py) for more integration details.

The expected lifecycle is: create the module, load quantized tensors from the checkpoint, call `post_init()` once to repack weights into the ARK-friendly layout, and then call `forward()` during inference.

Minimal usage:
```python
from auto_round_kernel.qlinear import QuantLinear

qlinear = QuantLinear(
    bits=4,
    group_size=128,
    sym=True,
    in_features=in_features,
    out_features=out_features,
    bias=bias is not None,
    weight_dtype=weight_dtype,
)
# Load qweight, qzeros, scales, and bias from checkpoint.
qlinear.post_init()

# Run inference
y = qlinear(x)
```

### Attention APIs

#### `ark.sdpa` — Standard Flash Attention

```python
import auto_round_kernel as ark

output = ark.sdpa(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    tensor_layout="HND",  # "HND" or "NHD"
)
```

- Q/K/V dtype: FP16 or BF16
- Supported head dims: 64, 96, 128, 192
- Supports GQA (grouped query attention) via different Hq/Hkv

#### `ark.sagev1` — SAGE v1 Attention

```python
output = ark.sagev1(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
    quant_block_size=64,  # block size for INT8 QK quantization
    tensor_layout="HND",
)
```

- Q/K/V dtype: FP16 or BF16
- Supported head dims: 64, 128
- Internally quantizes Q/K to INT8 per block; PV computed in half precision
- Falls back to `ark.sdpa` when `quant_block_size <= 0`

#### `ark.sagev1_pvi8` — SAGE v1 with INT8 PV

Same interface as `sagev1` but also quantizes V to INT8 internally for higher throughput.

#### `ark.sage_dynquant` — SAGE with Fused Dynamic Quantization

```python
output = ark.sage_dynquant(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
    quant_block_size=64,
)
```

- Takes FP16 Q/K/V, performs fused block-wise INT8 quantization of Q/K via SYCL kernel
- Supports quant_block_size: 1 (per-token), 32, 64, 128, 256
- Auto-pads sequence lengths for block alignment

#### `ark.sageattn` — SageAttention-Compatible Dispatcher

ARK provides a drop-in replacement for the `sageattention.sageattn` API, enabling seamless integration with existing SageAttention workflows:

```python
import auto_round_kernel as ark

# Drop-in replacement for sageattention.sageattn
output = ark.sageattn(
    q,
    k,
    v,
    tensor_layout="HND",
    is_causal=False,
    sm_scale=None,
    return_lse=False,
    kernel="v1_pvhalf",  # or "v1_pvi8"
    **kwargs,
)
```

This mirrors the [SageAttention](https://github.com/thu-ml/SageAttention) interface for drop-in compatibility, allowing models using SageAttention to run on Intel Arc GPUs without code changes.

### MOE GEMM

```python
output = ark.moe_gemm(
    activations,  # [total_tokens, K] FP16/BF16
    weights,  # [num_experts, K, N] FP16/BF16
    num_tokens_per_expert,  # [num_experts] int32
    scales=None,  # optional [num_experts, N] FP16/BF16
)
```

### Patching torch SDPA

ARK can globally replace `torch.nn.functional.scaled_dot_product_attention` for evaluation, including SageAttention backends:

```python
import auto_round_kernel as ark

# Patch with standard SDPA backend
ark.patch_torch_sdpa(backend="sdpa")

# Patch with SageAttention v1 backend (INT8 QK, PV half)
ark.patch_torch_sdpa(backend="sagev1", quant_block_size=64)

# Patch with SageAttention v1 + INT8 PV backend
ark.patch_torch_sdpa(backend="sagev1_pvi8", quant_block_size=64)

# Restore original
ark.unpatch_torch_sdpa()
```

Or use the helper launcher for lm-eval:

```bash
cd /path/to/auto_round_extension/ark
PYTHONPATH=$PWD python tools/lm_eval_with_ark_sdpa.py \
  --model hf \
  --model_args pretrained=/path/to/model,trust_remote_code=True,dtype=bfloat16 \
  --tasks hellaswag,piqa,winogrande \
  --device xpu:0 \
  --batch_size 1
```

### Low-Level Matrix Operations

```python
# FP16/BF16 matrix multiply with bias
C = ark.matmul(A, B, bias)

# INT8 matrix multiply (s8s8s32)
C = ark.igemm_s8s8s32(A, B)

# Weight-only quantized GEMM with INT8 weights
C = ark.woqgemm_s8(A, B, scaleB, bias)

# General weight-only quantized GEMM
C = ark.woqgemm(A, B, bias, n, k, groupsize, compute_type, weight_type, scale_type, asym)
```

## Testing

Unit tests are available in the [test](test/) directory:

| Test file | Description |
|-----------|-------------|
| `test_weightonly.py` | Weight-only quantized GEMM (CPU + XPU) |
| `test_flash_attn.py` | Flash attention (sdpa) correctness |
| `test_sdpa.py` | SDPA benchmark suite |
| `test_sdpa_parity.py` | SDPA parity with non-contiguous inputs and layouts |
| `test_sage_dynquant.py` | SageDynQuant block-wise benchmark |
| `test_bench_bmg.py` | BMG comparison benchmark |
| `test_matmul.py` | General matrix multiply |
| `test_packq.py` | Weight packing/unpacking |
| `test_moe.py` | MOE GEMM correctness |

## Notes

- The SDPA patch only routes calls to ARK on XPU when inputs match kernel constraints; otherwise it falls back to the original torch SDPA.
- Supported Q/K/V dtypes for attention are FP16 and BF16 (except SAGE variants which may use INT8 internally).
- `dropout_p` must be 0.0 for all ARK attention paths.
- Additive masks are supported when they can be normalized to `[B, 1, Sq, Skv]`; boolean masks fall back to torch.
- Non-contiguous (sliced) input tensors are supported for all attention backends.
