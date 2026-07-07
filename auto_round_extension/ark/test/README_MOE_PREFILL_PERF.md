# MoE Prefill Performance Test

## Overview

The `test_moe_prefill_perf.py` file provides comprehensive performance benchmarks for MoE (Mixture of Experts) prefill operations with TFLOPS (Tera Floating Point Operations Per Second) calculations.

## What is MoE Prefill?

**Prefill** is the phase during LLM inference where many tokens (e.g., the entire prompt or a batch of sequences) are processed simultaneously. In MoE models, tokens are routed to different experts, and each expert may receive multiple tokens. This is different from **decode** (token generation), where typically only one token per expert is processed at a time.

## Features

### 1. **Comprehensive Data Type Support**
- FP16 (float16)
- BF16 (bfloat16)
- INT8 (symmetric and asymmetric quantization)
- INT4 (symmetric and asymmetric quantization)
- INT2 (symmetric and asymmetric quantization)
- FP8 (float8_e4m3fn and float8_e5m2)

### 2. **TFLOPS Calculation**
The test calculates TFLOPS for each configuration using the formula:
```
FLOPs = total_tokens × K × N × 2
TFLOPS = FLOPs / (time_in_seconds) / 1e12
```

Where:
- `total_tokens`: Total number of tokens across all experts
- `K`: Input feature dimension
- `N`: Output feature dimension
- `×2`: Each multiply-add operation counts as 2 FLOPs

### 3. **Various MoE Configurations**
The test covers multiple realistic MoE scenarios:
- **Small models** (8 experts, Mixtral-style): 4096×4096, 4096×14336, 14336×4096
- **Medium models** (8 experts): Various token distributions
- **Large models** (16, 32, 64 experts, DeepSeek-style): 2048×2048
- **Uneven distributions**: Simulates real-world routing patterns

### 4. **Baseline Comparison**
Each test compares the ARK MoE kernel against a baseline PyTorch implementation:
- **Baseline**: Single `torch.bmm` over a `[E, M_max, K]` padded activations buffer (each expert's token slice padded to the global maximum tokens-per-expert). Replaces the previous 192-iteration per-expert loop so the kernel-launch overhead doesn't dominate small-token cases. For quantized tests, weights are pre-dequantized so the `baseline(ms)` column measures matmul cost only.
- **ARK Kernel**: Optimized `ark.moe_gemm` with fused operations.
- **Speedup**: Reports `baseline / ark` -- the fused kernel's speedup over the matmul-only baseline.

## How to Run

### Run all tests:
```bash
cd /path/to/auto_round_extension/ark/test
pytest -v -s test_moe_prefill_perf.py
```

### Run specific data type:
```bash
# FP16 tests only
pytest -v -s test_moe_prefill_perf.py::TestMoEGemmPrefillPerf::test_perf_fp

# INT4 tests only
pytest -v -s test_moe_prefill_perf.py::TestMoEGemmPrefillPerf::test_perf_int4

# INT8 symmetric quantization with bfloat16 activations
pytest -v -s test_moe_prefill_perf.py::TestMoEGemmPrefillPerf::test_perf_int8 -k "bfloat16 and not asym"
```

**Note**: The `-s` flag is required to see the printed timing tables and TFLOPS output.

## Output Format

The test prints formatted tables with the following columns:

```
shape          E      N      K  tokens    baseline(ms)        ark(ms)     speedup    TFLOPS
small  E=8     8   4096   4096     252         12.3456         4.5678       2.70x       45.2
medium E=8     8   4096  14336     528         23.4567         8.9012       2.63x       78.9
...
```

Where:
- **shape**: Configuration label
- **E**: Number of experts
- **N**: Output feature dimension
- **K**: Input feature dimension
- **tokens**: Total tokens across all experts
- **baseline(ms)**: PyTorch matmul-only latency (weights pre-dequantized for quantized tests).
- **ark(ms)**: ARK kernel latency (milliseconds)
- **speedup**: `baseline / ark` -- fused kernel's speedup over the matmul-only baseline
- **TFLOPS**: Throughput in tera floating point operations per second

## Requirements

- Intel XPU (Arc GPU) with PyTorch XPU support
- `auto_round_kernel` built with `ARK_SYCL_TLA=ON`
- Test dependencies from `test_moe.py` (pack/dequant helpers)

## Architecture

```
test_moe_prefill_perf.py
├── Timing utilities (_xpu_time_ms)
│   └── Uses XPU events for accurate GPU timing
├── FLOPS calculation (_compute_moe_flops)
│   └── Computes theoretical FLOPs for TFLOPS metric
├── Baseline implementation (_default_moe_prefill, _build_bmm_pad_layout)
│   └── Single `torch.bmm` over [E, M_max, K] padded activations
├── Test shapes (PREFILL_SHAPES)
│   └── Various realistic MoE configurations
└── Test cases (TestMoEGemmPrefillPerf)
    ├── test_perf_fp (FP16/BF16)
    ├── test_perf_int4 (INT4 sym/asym)
    ├── test_perf_int8 (INT8 sym/asym)
    ├── test_perf_int2 (INT2 sym/asym)
    └── test_perf_fp8 (FP8 e4m3fn/e5m2)
```

## Example Output

```
==================================================================
FP weights (float16)  -- ark.moe_gemm (prefill) vs single torch.bmm
==================================================================
shape              E      N      K  tokens    baseline(ms)        ark(ms)     speedup    TFLOPS
------------------------------------------------------------------
small  E=8         8   4096   4096     252         12.3456         4.5678       2.70x       45.2
medium E=8         8   4096  14336     528         23.4567         8.9012       2.64x       78.9
medium E=8         8  14336   4096     528         25.6789         9.1234       2.82x       76.5
large  E=16       16   2048   2048     256          5.6789         2.3456       2.42x       91.2
large  E=32       32   2048   2048     256          5.7890         2.4567       2.36x       87.3
large  E=64       64   2048   2048     256          5.8901         2.5678       2.29x       83.5
uneven E=8         8   4096   4096     610         28.9012        10.1234       2.86x       52.1
```

## Key Metrics

1. **TFLOPS**: Higher is better - indicates compute throughput
2. **Speedup**: Higher is better - shows performance gain over baseline
3. **Latency (ms)**: Lower is better - actual kernel execution time

## Integration with CI/CD

This test can be integrated into performance regression testing:
- Set minimum TFLOPS thresholds for each configuration
- Track speedup ratios over time
- Alert on performance degradation

## Related Files

- `test_moe.py`: Correctness tests for MoE GEMM
- `test_moe_decode_perf.py`: Performance tests for MoE decode (single token per expert)
- `test_bench_bmg.py`: SDPA performance benchmarks with TFLOPS

## FP8 Prefill Paths (opt-in env flags)

The FP8 prefill benchmark (`test_perf_fp8`) measures the default ARK path in
the `ark(ms)` column and, on the same shapes, a fused **native FP8** path in
the `native(ms)` / `native TFLOPS` columns and a mixed-input **DPAS FP8**
path in the `dpas(ms)` / `dpas TFLOPS` columns. The four underlying kernels
are selected by three independent env flags — read once on first use inside
the extension and cached — with the following precedence:

| Precedence | Env flag(s)                                                       | Kernel                                                                                                       |
| ---------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| 1 (highest)| `ARK_MOE_PREFILL_DPAS_FP8` unset or truthy (**default ON**)       | **Mixed-input DPAS FP8 grouped GEMM (Variant B).** Ported from `vllm-project/vllm-xpu-kernels` `xe_gemm_4bits` — FP8 bytes are upcast to `act_dtype` in registers via CuTe `reorder`, then the per-K-group scale is applied inline (`apply_scale` IGA asm). XMX-bound; expected ~2-2.5× the scalar native path. Same `[E, N, K/group_size]` scale layout as auto-round's calibration output — no re-quantisation needed. Implemented in `sycl_tla_moe_prefill_fp8_dpas.hpp`. **Status: NEEDS-HARDWARE-VALIDATION** (untested port). |
| 2          | `ARK_MOE_PREFILL_NATIVE_FP8=1`                                    | Scalar native FP8 fused GEMM. No `[E, K, N]` bf16/fp16 workspace. FP8 bytes are upcast to `act_dtype` in registers inside the GEMM kernel and the per-K-group scale is folded into the accumulator. Only the final output row is written back. Implemented in `sycl_tla_moe_prefill_fp8_native.hpp`. |
| 3          | `ARK_MOE_PREFILL_FUSED_FP8=1`                                     | SLM-transposed dequant kernel (`sycl_tla_moe_prefill_fused.hpp`) followed by the stock bf16/fp16 grouped GEMM. Still writes an `[E, K, N]` workspace to DRAM. FP8-E4M3 only.                                            |
| 4 (default)| all above unset                                                   | v1 dequant kernel (`sycl_tla_moe_mixed.hpp::launch_dequant_fp8`) followed by the stock bf16/fp16 grouped GEMM. FP8-E4M3 and FP8-E5M2.                                                                                    |

**DPAS path shape preconditions** — the `moe_gemm_prefill` dispatcher
silently falls back to precedence 2/3/4 whenever any of these fail:

- `N % 64 == 0` (BN)
- `K % 32 == 0` (BK)
- `K % group_size == 0`
- `group_size ∈ {32, 64, 128, 256}`
- `asym == False` (FP8 quant is symmetric only)

**Native path shape preconditions** — same fallback semantics:

- `N % 16 == 0` (BN = SG_SIZE = 16)
- `K % 32 == 0` (BK)
- `K % group_size == 0`
- `group_size % 32 == 0` (so per-tile scale is constant along K)
- `asym == False` (FP8 quant is symmetric only)

Both native and DPAS support **E4M3** and **E5M2**, and both **F16** and
**BF16** activations, covering the same `PREFILL_SHAPES` matrix as the
default column.

### Variant A — per-tensor FP8 DPAS (separate entry point)

The port also exposes a **Variant A** per-tensor FP8 DPAS grouped GEMM as a
separate Python entry point:

```python
outputs = ark.moe_gemm_prefill(
    activations,  # [total_tokens, K], f16/bf16
    weights,  # [E, K, N] row-major FP8 (vllm layout)
    num_tokens_per_expert,  # [E] int32
    scales=scales,  # [E] fp32, one per-tensor scale per expert
    scale_scheme="per_tensor",
)
```

This mirrors vllm-xpu-kernels' `cutlass_grouped_gemm_xe2_impl` FP8 branch
byte-for-byte. It requires a **re-quantised checkpoint** (one FP32 scalar
per expert, weights transposed to `[E, K, N]`), so it is best treated as a
future option for latency-critical decode paths rather than a drop-in for
existing auto-round FP8 checkpoints — Variant B is preferred there.

**Status: NEEDS-HARDWARE-VALIDATION** (untested port).

Enable via env at test-run time:

```bash
# Default (DPAS Variant B) — auto-round-native calibration scheme.
pytest -v -s test_moe_prefill_perf.py::TestMoEGemmPrefillPerf::test_perf_fp8

# Force the scalar native path only (disables DPAS).
ARK_MOE_PREFILL_DPAS_FP8=0 ARK_MOE_PREFILL_NATIVE_FP8=1 pytest -v -s test_moe_prefill_perf.py::TestMoEGemmPrefillPerf::test_perf_fp8

# Force the fused-dequant path.
ARK_MOE_PREFILL_DPAS_FP8=0 ARK_MOE_PREFILL_FUSED_FP8=1 pytest -v -s test_moe_prefill_perf.py::TestMoEGemmPrefillPerf::test_perf_fp8

# The perf test toggles the env internally per row so the `ark(ms)`,
# `native(ms)`, and `dpas(ms)` columns each measure a specific path
# regardless of the outer env setting.
```

For accuracy parity, `test_moe_prefill_accuracy.py::test_accuracy_fp8`
covers the dequant/native paths and
`test_accuracy_fp8_dpas_per_group` / `test_accuracy_fp8_per_tensor_dpas`
cover the DPAS Variants B / A at the same production shapes; all paths
share the tolerance `rtol=atol=1e-1` (E4M3) / `1e-1` (E5M2).

## INT8 Prefill Paths (opt-in env flag)

The INT8 sym prefill benchmark (`test_perf_int8` with `asym=False`) also
carries a mixed-input **DPAS INT8** column (`dpas(ms)` / `dpas TFLOPS`),
mirroring the FP8 per-K-group Variant B path. The `ark(ms)` column
forces `ARK_MOE_PREFILL_DPAS_INT8=0` and measures the legacy dequant +
GEMM path; the `dpas(ms)` column re-enables the flag and measures the
new mixed-input path on the same shapes.

| Precedence | Env flag                                                | Kernel                                                                                                                                                                                                                                                                                                                                                                                                    |
| ---------- | ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 (highest)| `ARK_MOE_PREFILL_DPAS_INT8` unset or truthy (**default ON**) | **Mixed-input DPAS INT8 grouped GEMM (Variant B).** INT8 bytes are upcast to `act_dtype` in registers via CuTe `reorder`, then the per-K-group scale is applied through the deferred group-boundary fold (identical to the FP8 per-group path). Same `[E, N, K/group_size]` scale layout as auto-round's INT8 calibration output — no re-quantisation needed. Both sym and asym are supported: asym additionally uses a per-M-row per-K-group activation-sum precompute so the fold becomes `Σ_g s · (Σ w·a − z · Σ a)`. Implemented in `sycl_tla_moe_prefill_int_dpas.hpp`. **Status: NEEDS-HARDWARE-VALIDATION** (untested port). |
| 2 (default)| `ARK_MOE_PREFILL_DPAS_INT8=0`                           | v1 dequant kernel (`sycl_tla_moe_mixed.hpp::launch_dequant_int8`) followed by the stock bf16/fp16 grouped GEMM. Handles both sym and asym.                                                                                                                                                                                                                                                                       |

**DPAS path shape preconditions** — the `moe_gemm_prefill` dispatcher
silently falls back to precedence 2 whenever any of these fail
(identical to the FP8 per-group predicate):

- `N % 64 == 0` (BN)
- `K % 32 == 0` (BK)
- `K % group_size == 0`
- `group_size ∈ {32, 64, 128, 256}`
- `asym`: sym and asym are both supported (asym uses an extra `Σ a` precompute pass)

Accuracy parity is covered by
`test_moe_prefill_accuracy.py::test_accuracy_int8_dpas_per_group` at the
same production shapes as `test_accuracy_int8`, with the standard INT8
tolerance (`rtol=atol=1e-1`).

## INT4-sym Prefill Paths (opt-in env flags)

The INT4 sym prefill benchmark (`test_perf_int4` with `asym=False`)
carries a mixed-input **DPAS S4** column (`dpas(ms)` / `dpas TFLOPS`).
`test_perf_int4` forces `ARK_MOE_PREFILL_DPAS_S4=0` and
`ARK_MOE_PREFILL_DPAS_INT8=0` for the `ark(ms)` column (legacy dequant
+ GEMM path) and re-enables `ARK_MOE_PREFILL_DPAS_S4=1` for the
`dpas(ms)` column (single-pass packed-nibble mainloop).

Two independent DPAS paths are available for S4-sym; asym S4 always
falls through to the dequant path.

| Precedence  | Env flag                                                              | Kernel                                                                                                                                                                                                                                                                                                                                                                       |
| ----------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 (highest) | `ARK_MOE_PREFILL_DPAS_S4` unset or truthy (**default ON**)            | **S4-sym single-pass DPAS mixed-input mainloop.** Reads packed `[E, N, K/2]` `uint8_t` nibbles directly and folds the S4→`act_dtype` upcast into the DPAS mainloop via CuTe's `reorder(tBrB, tCrB)` (which relies on `NumericArrayConverter<ElementA, cutlass::int4b_t, N>`). B-side global traffic is exactly half of the S8 path. Per-K-group scale is applied through the same deferred group-boundary fold as INT8. Implemented in `sycl_tla_moe_prefill_s4_dpas.hpp`. **Status: NEEDS-HARDWARE-VALIDATION** (untested port). |
| 2 (fallback)| `ARK_MOE_PREFILL_DPAS_S4=0` and `ARK_MOE_PREFILL_DPAS_INT8` truthy (**default ON**) | **S4→S8 upcast + shared INT8 DPAS mainloop.** Two-pass: `launch_upcast_int4_sym_to_int8` writes an `[E, N, K]` `int8_t` view of the dequant workspace, then the standard INT8 per-group DPAS mainloop consumes it. Robust but pays the ~E·N·K byte round-trip vs. path 1. Implemented in `sycl_tla_moe_mixed.hpp` + `sycl_tla_moe_prefill_int_dpas.hpp`. |
| 3 (default) | `ARK_MOE_PREFILL_DPAS_S4=0` and `ARK_MOE_PREFILL_DPAS_INT8=0`         | v1 dequant kernel (`sycl_tla_moe_mixed.hpp::launch_dequant_int4`) followed by the stock bf16/fp16 grouped GEMM. Handles both sym and asym.                                                                                                                                                                                                                                                                                                                     |

**S4 DPAS path shape preconditions** — the `moe_gemm_prefill`
dispatcher silently falls back to precedence 2 (then 3) whenever any of
these fail:

- `N % 64 == 0` (BN)
- `K % 32 == 0` (BK)
- `K % group_size == 0`
- `group_size % 2 == 0` (nibble pair never straddles a group boundary)
- `group_size ∈ {32, 64, 128, 256}`
- `asym == false` (asym S4 is out of scope for both DPAS paths)

Accuracy parity is covered by
`test_moe_prefill_accuracy.py::test_accuracy_int4_dpas_per_group`,
which forces `ARK_MOE_PREFILL_DPAS_S4=1` +
`ARK_MOE_PREFILL_DPAS_INT8=0` so the single-pass mainloop is
exclusively exercised, at the same production shapes as
`test_accuracy_int4`, with tolerance `rtol=atol=1e-1`.

## FP8 per-expert (per-tensor) perf tests

`test_perf_fp8_per_tensor` benchmarks the Variant A DPAS path against
the single-`torch.bmm` baseline for the **one-FP32-scalar-per-expert**
quantisation scheme (`scales.shape == [E]`, weights `[E, K, N]` row-major
FP8 — vllm layout). Parametrised across all dtype combinations
(fp16/bf16 × E4M3/E5M2) over the same `PREFILL_SHAPES` matrix as
`test_perf_fp8`.

```bash
# Prefill: dispatches to moe_gemm_prefill_fp8_dpas (Variant A) via
# scale_scheme="per_tensor". Silently skipped on builds without that
# pybind symbol.
pytest -v -s test_moe_prefill_perf.py::TestMoEGemmPrefillPerf::test_perf_fp8_per_tensor
```

`test_moe_decode_perf.py::test_perf_fp8_per_tensor` covers the same
quantisation scheme for the decode phase. The C++ decode kernel does
NOT expose a native `[E]` per-tensor entry point (only per-K-group
`[E, N, K/group_size]` scales are accepted), so the test **broadcasts**
the per-expert scalar over the K-group dimension before feeding the
existing kernel. Semantically this matches a per-tensor quantised
checkpoint and runs on the same code path as `test_perf_fp8`; the
timings validate that the scheme incurs the same decode-kernel cost as
the richer per-group scheme.

```bash
pytest -v -s test_moe_decode_perf.py::TestMoEGemmDecodePerf::test_perf_fp8_per_tensor
```

## INT8 per-expert (per-tensor) perf tests

`test_perf_int8_per_tensor` benchmarks the **INT8** sibling of the FP8
Variant A DPAS path. Weights are stored as one signed byte per element
in `[E, K, N]` row-major `torch.int8`; scales are one FP32 scalar per
expert (`scales.shape == [E]`). The kernel keeps the DPAS atom running
on `bf16`/`fp16` (identical to the FP8 Variant A path) and upcasts
`int8` → activation dtype in register before the multiply, so the
speed-of-light matches the FP8 case at a smaller weight footprint.

```python
outputs = ark.moe_gemm_prefill(
    activations,  # [total_tokens, K], f16/bf16
    weights,  # [E, K, N] row-major torch.int8 (vllm layout)
    num_tokens_per_expert,  # [E] int32
    scales=scales,  # [E] fp32, one per-tensor scale per expert
    scale_scheme="per_tensor",
)
```

Dispatches to `moe_gemm_prefill_int_dpas` (Variant A INT8) — the
`per_tensor` scheme now routes by `weights.dtype` (FP8 → existing FP8
DPAS entry point; `torch.int8` → the new INT8 DPAS entry point).
Silently skipped on builds without that pybind symbol.

```bash
pytest -v -s test_moe_prefill_perf.py::TestMoEGemmPrefillPerf::test_perf_int8_per_tensor
```

Accuracy parity is covered by
`test_moe_prefill_accuracy.py::test_accuracy_int8_per_tensor_dpas` at
the same production shapes, with the standard INT8 tolerance
(`rtol=atol=1e-1`).

**Status: NEEDS-HARDWARE-VALIDATION** (untested port; sym-only for
Phase 1 — per-group and asym INT4 / INT2 DPAS are follow-up phases
that will reuse the same mainloop skeleton with an added unpack step).
