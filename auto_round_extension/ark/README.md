## What is AutoRound Kernel (ARK)?

AutoRound Kernel (ARK) is a low-bit acceleration library for Intel platform, providing three categories of optimized operators for LLM inference.

| Operator Category | CPU | XPU (Battlemage) |
|:------------------|:---:|:----------------:|
| **Weight-Only Quantized Linear** (INT4/INT8/FP8/FP4) | ✅ | ✅ |
| **MoE Grouped GEMM** | ❌ | ✅ |
| **SageAttention** (SDPA / SAGE) | ❌ | ✅ |

**Validated CPU:** Intel Xeon Scalable (Sapphire Rapids / Emerald Rapids), Intel Xeon 6 (Sierra Forest / Granite Rapids)<br>
**Validated GPU:** Intel Arc B-Series / Arc Pro B-Series (Battlemage)

### Highlights — Ecosystem Integration

ARK kernels are integrated into the following projects:

| Project | Integration | Description |
|---------|-------------|-------------|
| [vllm](https://github.com/vllm-project/vllm) | [`inc_wna16_linear.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_linear.py#L379-L470) | `INCXPUARKLinearMethod` — weight-only quantized linear on XPU via `auto_round_kernel.qlinear.QuantLinear`. The goal is to support all AutoRound-quantized models. The plan is tracked in [Intel Quantization Support Roadmap](https://github.com/vllm-project/vllm/issues/37979) |
| [vllm-omni](https://github.com/vllm-project/vllm-omni) | [`sage_attn.py`](https://github.com/vllm-project/vllm-omni/blob/f8340d078e4e9c3b793bd92d55d891b29703f0a8/vllm_omni/diffusion/attention/backends/sage_attn.py#L27) | `SageAttentionBackend` — diffusion model attention on XPU via `ARK.sagev1` |
| [Transformers](https://github.com/huggingface/transformers) (via [auto-round](https://github.com/intel/auto-round)) | [`backend.py`](https://github.com/intel/auto-round/blob/main/auto_round/inference/backend.py#L531) | All models quantized by AutoRound automatically use ARK on CPU/XPU by default; no additional configuration required. 6 backends registered: `auto_round_kernel[_xpu]` (GPTQ no-zp), `auto_round_kernel_zp[_xpu]` (GPTQ +zp), `auto_round_kernel_awq[_xpu]` (AWQ).<br>**CPU:** INT2/INT4/INT8; **XPU:** INT4/INT8 |

---

## 1. Linear (Weight-Only Quantized GEMM)

Low-bit weight-only linear for LLM inference. Both CPU and XPU are supported.

### API

| API | Description | Platform |
|-----|-------------|----------|
| `QuantLinear` ([example ↓](#example)) | Unified PyTorch module (GPTQ/AWQ/raw quantized checkpoint) | CPU / XPU |
| `QuantLinearGPTQ` | GPTQ-format checkpoint loader | CPU / XPU |
| `QuantLinearAWQ` | AWQ-format checkpoint loader | CPU / XPU |
| `QuantLinearFP8` | FP8 weight-only linear | CPU / XPU |
| `woqgemm` | Low-level weight-only GEMM (packed format) | CPU / XPU |
| `woqgemm_s8` | Low-level INT8-weight GEMM with scale | CPU / XPU |
| `_repack_quantized_weight` | Repack raw qweight/qzero/scale → ARK format | CPU / XPU |
| `_unpack_weight` | Unpack ARK-format weight back to full precision | CPU / XPU |

### Key Features

> **W4A8 / W2A8 Rescale (QQQ-style)** — On XPU, ARK supports \~[QQQ](https://github.com/HandH1998/QQQ)-style compute: low-bit weights (INT2/INT4) are re-scaled to INT8 and computed via INT8 GEMM, avoiding FP16 dequantization for better throughput. Enabled automatically via environment variable `ARK_AUTO_S8`; see [xpu_wrapper.hpp](auto_round_kernel/wrapper/include/xpu_wrapper.hpp).

**Dense W4A16 on XPU (SYCL TLA)**

These opt-ins apply only to device ID `0x674f` with symmetric INT4 weights, INT8 compute configuration, FP16 activation and scale at the kernel boundary, group size 128, group-major packed scales, and no bias. All are disabled by default:

- `ARK_WOQ_S4_DPAS_M1=1` allows unpadded single-row DPAS for shapes listed in `ARK_WOQ_S4_DPAS_M1_SHAPES` as comma-separated `KxN` pairs. The shape table is read once, on the first enabled single-row call. An unset, empty, or invalid table keeps GEMV.
- `ARK_WOQ_S4_DPAS_SMALL_M_TILE_K=64` selects K64 for `M=2, K=14336, N=4096`.
- `ARK_WOQ_S4_DPAS_SPLIT_K4_SHAPES=KxN,...` selects four-way K64 split-K for listed shapes at both `M=1` and `M=2`, taking precedence over the other opt-ins. It requires `K` divisible by 512, `N` divisible by 128 and at most 16384, and an in-order queue outside graph recording. It reuses a 512 KiB FP32 scratch buffer per participating queue. The shape table is parsed once on the first call that checks it; an unset, empty, malformed, or unmatched table keeps the existing dispatch.

Populate the single-row table only with shapes measured on the target device against GEMV at the same batch size. Require a lower median latency, no tail-latency regression, and acceptable numerical results for each input dtype the process uses. The native selector sees FP16 for both FP16 input and converted BF16 input, so the table cannot distinguish them.

Tune the split-K table separately using whole-call latency and correctness for both row counts and all input dtypes in use. On the tested CRI device, `14336x4096` improved median latency for FP16 and BF16 inputs at both row counts. The same improvement did not hold across dtypes for `4096x4096` or `4096x14336`. There are no default entries. Split-K changes FP32 accumulation order, so its outputs need not match unsplit DPAS bitwise.

All other calls, including calls on B70, retain the existing dispatch. BF16 conversion and packed weights are unchanged.

### Supported Data Types

#### CPU

| Weight dtype | Compute dtype | Scale dtype | Algorithm |
|--------------|:-------------:|:-----------:|:---------:|
| INT1–INT8 | INT8<sup>[1]</sup> / BF16 / FP32 | BF16 / FP32 | sym / asym |
| FP8 (E4M3, E5M2) | BF16 / FP32 | FP32 / FP8 (E8M0) | NA |
| FP4 (E2M1) | BF16 / FP32 | BF16 / FP32 | NA |

#### XPU

| Weight dtype | Compute dtype | Scale dtype | Algorithm |
|--------------|:-------------:|:-----------:|:---------:|
| INT4, INT8 | INT8 / FP16 | FP16 | sym |
| FP8 (E4M3, E5M2) | FP16 | FP16 / FP8 (E8M0) | NA |

<sup>[1]</sup> INT8 compute includes dynamic activation quantization; results are dequantized to floating-point.

### Example

```python
import auto_round_kernel as ark

# Prepare quantized weight: qweight [K, N] int4/int2, scale [K/G, N] fp16/fp32, zp [K/G, N] int4/int2
packw = ark.repack_quantized_weight(
    qweight,
    scale,
    zp,
    blocksize=128,
    compute_type="fp16",
    weight_type="int4",
    scale_type="fp16",
    asym=False,
)

# Run weight-only quantized GEMM: activation [M, K] → output [M, N]
output = ark.woqgemm(
    activation,  # [M, K] fp16/bf16
    packw,  # packed weight blob (INT8)
    bias,  # [1, N] optional bias
    n,  # output features
    k,  # input features
    groupsize=128,
    compute_type="fp16",
    weight_type="int4",
    scale_type="fp16",
    asym=False,
)

# Decompose back to full precision for verification
decompressed = ark.unpack_weight(
    packw,
    dtype=torch.float16,
    n=n,
    k=k,
    groupsize=128,
    compute_type="fp16",
    weight_type="int4",
    scale_type="fp16",
    asym=False,
)
```

See [test_weightonly.py](test/test_weightonly.py) for an end-to-end example of weight repack, verification, and woqgemm execution on CPU and XPU.

---

## 2. MoE (Mixture-of-Experts Grouped GEMM)

Grouped GEMM for MoE layers where different experts process varying numbers of tokens.

### API

| Function | Description | Platform | Activation Dtype | Weight Dtype |
|----------|-------------|----------|:----------------:|:------------:|
| `ark.moe_gemm(...)` ([example ↓](#example-1)) | Grouped GEMM across experts | XPU | FP16 / BF16 | FP16 / BF16 |
| `ark.moe_gemm(...)` (WIP) | Grouped GEMM with INT4 weight | XPU | FP16 / BF16 | INT4 🚧 |
| `ark.moe_gemm(...)` (WIP) | Grouped GEMM with INT2 weight | XPU | FP16 / BF16 | INT2 🚧 |
| `ark.moe_gemm(...)` (WIP) | Grouped GEMM with INT8 weight | XPU | FP16 / BF16 | INT8 🚧 |

> 🚧 INT2 / INT4 / INT8 weight support is under active development. See [#PR](https://github.com/intel/auto-round/pull).

### Details

| Parameter | Shape | Dtype |
|-----------|-------|-------|
| activations | `[total_tokens, K]` | FP16 / BF16 |
| weights | `[num_experts, K, N]` (row-major) | FP16 / BF16 |
| num_tokens_per_expert | `[num_experts]` | INT32 |
| scales (optional) | `[num_experts, N]` | FP16 / BF16 |
| **output** | `[total_tokens, N]` | same as activations |

### Example

```python
# FP16/BF16 MoE
output = ark.moe_gemm(activations, weights, num_tokens_per_expert)

# INT4 MoE (coming soon)
# output = ark.moe_gemm(activations, q4_weights, num_tokens_per_expert, scales=scales)
```

Build requirement: `ARK_SYCL_TLA=ON`. See [test_moe.py](test/test_moe.py).

---

## 3. SageAttention (XPU SDPA Acceleration)

ARK provides a full family of scaled dot-product attention kernels on XPU, ranging from vanilla FP16 SDPA to INT8-quantized SageAttention variants.

### API Overview

| Function | Description | Q/K/V Input | PV Precision | Head Dim |
|----------|-------------|-------------|:------------:|:--------:|
| `ark.sdpa` ([example ↓](#drop-in-sdpa-replacement)) | FP16/BF16 SDPA (flash attention) | FP16 / BF16 | FP16 | 64, 96, 128, 192 |
| `ark.sage` | Low-level INT8 SAGE (pre-quantized Q/K) | INT8 (Q/K), FP16 (V) | FP16 | 64, 128 |
| `ark.sage_pvi8` | Low-level INT8 SAGE (pre-quantized Q/K/V) | INT8 | INT8 | 64, 128 |
| `ark.sagev1` | High-level FP16 → internal Q/K quant → SAGE | FP16 / BF16 | FP16 | 64, 128 |
| `ark.sagev1_pvi8` | High-level FP16 → internal Q/K/V quant → SAGE PV INT8 | FP16 / BF16 | INT8 | 64, 128 |
| `ark.sageattn` | Dispatcher (sageattention-compatible API) | FP16 / BF16 | FP16 / INT8 | 64, 128 |
| `ark.sage_dynquant` | Dynamic INT8 block-wise Q/K quant → SAGE (drop-in SDPA replacement) | FP16 / BF16 | FP16 | 64, 128 |

### Comparison

| Feature | `sdpa` | `sagev1` | `sagev1_pvi8` | `sage_dynquant` |
|---------|:------:|:--------:|:-------------:|:---------------:|
| Q/K quantization | None | Internal INT8 | Internal INT8 | Internal INT8 |
| PV quantization | None | None | Internal INT8 | None |
| quant_block_size | N/A | 1 / ≥32 | 1 / ≥32 | 1 / ≥32 |
| Additive mask | ✅ [B,1,Sq,Skv] FP32 | ✅ | ✅ | ✅ |
| Causal mask | ✅ | ✅ | ✅ | ✅ |
| GQA | ✅ | ✅ | ✅ | ✅ |
| Tensor layout | HND / NHD | HND / NHD | HND / NHD | HND |

### Drop-in SDPA Replacement

Replace `torch.nn.functional.scaled_dot_product_attention` globally for lm-eval:
#### Replace torch SDPA and run lm-eval
  ARK exposes a standard SDPA interface through `ARK.sdpa(...)`. The implementation borrows from Neural Speed route logic internally, but the public contract is the standard scaled-dot-product-attention surface. If you want to replace `torch.nn.functional.scaled_dot_product_attention` globally for evaluation without editing model code, use the helper launcher in [tools/lm_eval_with_ark_sdpa.py](tools/lm_eval_with_ark_sdpa.py).

```bash
cd /path/to/auto_round_extension/ark
PYTHONPATH=$PWD python tools/lm_eval_with_ark_sdpa.py \
  --model hf \
  --model_args pretrained=/path/to/model,trust_remote_code=True,dtype=bfloat16 \
  --tasks hellaswag,piqa,winogrande \
  --device xpu:0 --batch_size 1
```

The patching logic (in [`auto_round_kernel/torch_sdpa_patch.py`](auto_round_kernel/torch_sdpa_patch.py)) routes to ARK on XPU when constraints are met; otherwise falls back to PyTorch SDPA.

### Constraints

| Constraint | `sdpa` | `sagev1` / `sagev1_pvi8` / `sage_dynquant` |
|------------|:------:|:------------------------------------------:|
| Q/K/V dtype | FP16, BF16 | FP16, BF16 |
| Head dim | 64, 96, 128, 192 | 64, 128 |
| `dropout_p` | must be 0.0 | must be 0.0 |
| Boolean mask | falls back to torch | falls back to torch |
| Additive mask shape | `[B, 1, Sq, Skv]` FP32 | `[B, 1, Sq, Skv]` FP32 |
| quant_block_size | N/A | 1 (per-token) or ≥32 |

---

## Installation

### Install via pip
```bash
pip install auto-round-lib
```

### Install from Source
```bash
pip install . --no-build-isolation
# or
python setup.py bdist_wheel; pip install dist/*
```

Build with MoE / SageAttention support requires `ARK_SYCL_TLA=ON`.

For oneAPI/SYCL-TLA builds, template compilation can require several gigabytes
per compiler process. If the build host runs out of memory, limit concurrent
XPU compilations with `ARK_SYCL_TLA_JOBS`:

```bash
ARK_SYCL_TLA_JOBS=1 pip install . --no-build-isolation
```

On a oneAPI-equipped build host, configure an XPU build with compile commands
enabled and measure each translation unit before choosing a higher job count:

```bash
python tools/measure_sycl_tla_compile_memory.py \
  --build-dir /path/to/ark/xbuild \
  --output /tmp/ark-sycl-tla-rss.json
```

The measurement tool runs compiler commands serially and reports peak RSS,
direct dispatch count, and reachable template declarations. The current
environment must provide `icx` and a generated `compile_commands.json` for
these measurements to represent the SYCL-TLA build.

---

## Tests

| Test | Description |
|------|-------------|
| [test_weightonly.py](test/test_weightonly.py) | WOQ GEMM pack/unpack/run on CPU & XPU |
| [test_moe.py](test/test_moe.py) | MoE grouped GEMM |
| [test_flash_attn.py](test/test_flash_attn.py) | SDPA (flash attention) prefill |
| [test_sdpa.py](test/test_sdpa.py) | SDPA benchmark suite |
| [test_sdpa_parity.py](test/test_sdpa_parity.py) | SDPA vs PyTorch parity check |
| [test_sage_dynquant.py](test/test_sage_dynquant.py) | SageAttention dynamic INT8 quant benchmarks |
| [test_bench_bmg.py](test/test_bench_bmg.py) | BMG SDPA / SageAttention benchmarking |
| [test_matmul.py](test/test_matmul.py) | Low-level matmul |
| [test_packq.py](test/test_packq.py) | Weight packing utilities |
