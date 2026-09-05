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

# INT4-sym on the Qwen3-MoE shapes (hidden=2048, inter=768, E=128, top_k=8)
pytest -v -s test_moe_prefill_perf.py -k test_perf_int4_sym_qwen3_moe --run-moe-prefill-perf
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
├── Qwen3-MoE shapes (_QWEN3_NK / _QWEN3_BATCHES)
│   └── hidden=2048, inter=768, E=128, top_k=8, group_size=32 (issue repro)
└── Test cases (TestMoEGemmPrefillPerf)
    ├── test_perf_fp (FP16/BF16)
    ├── test_perf_int4 (INT4 sym/asym)
    ├── test_perf_int4_sym_qwen3_moe (INT4 sym, Qwen3-MoE shapes)
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

**Qwen3-MoE shape rows (issue repro).** `test_perf_int4_sym_qwen3_moe`
benchmarks a second, independent shape group, captured from a fused-MoE
layer where ARK lost to the native backend (`native_over_ark=0.604x`) at
small batch: `hidden_size=2048`, `intermediate_size=768`,
`num_local_experts=128`, `num_experts_per_tok=8`, `group_size=32` — i.e.
`w13 [128, 1536, 1024]` (gemm1 `N=1536`, `K=2048`) and
`w2 [128, 2048, 384]` (gemm2 `N=2048`, `K=768`), where the trailing dim
is the packed nibble count `K/2` and the scale tensors
`[128, 1536, 64]` / `[128, 2048, 24]` pin `group_size` to 32. Unlike
`test_perf_int4` (MiniMax-M2 shapes at `group_size=128`) it fills **all
three** ARK columns from the same packed weights, so the dispatcher's
default choice can be checked against the measurement:

| Column       | Path                                                            | Env                                    |
| ------------ | --------------------------------------------------------------- | -------------------------------------- |
| `ark(ms)`    | legacy dequant into `[E, K, N]` + stock grouped GEMM            | `DPAS_S4=0`, `DPAS_INT8=0`             |
| `native(ms)` | two-pass S4→S8 upcast + shared INT8 DPAS mainloop               | `DPAS_S4=0`, `DPAS_INT8=1`             |
| `dpas(ms)`   | single-pass S4 DPAS (packed-nibble read) — the shipped default  | `DPAS_S4=1`                            |

Routed rows are `batch * top_k` spread round-robin over the 128 experts,
so the reported batch of 2 model tokens reproduces the issue's
`rows_per_expert_sum=16`. Only that batch runs by default; `--all-shapes`
extends the sweep to 32/128/512/2048 model tokens, bracketing the
8-rows-per-expert DPAS tile occupancy point (`batch >= 128`).

```bash
pytest -v -s test_moe_prefill_perf.py -k test_perf_int4_sym_qwen3_moe \
    --run-moe-prefill-perf
```

The matching decode benchmarks live in `test_moe_decode_perf.py`:
`test_perf_int4_sym_qwen3_moe` (default dispatch vs. the dequant +
per-expert `A @ W.T` baseline) and
`test_perf_int4_sym_qwen3_moe_dpas_vs_scalar` (scalar GEMV vs. S4 DPAS
with `ARK_MOE_DECODE_DPAS_S4_MIN_TPE=0`, isolating the occupancy gate's
routing decision — at batch 2 this shape sits at 0.125 tokens per
expert, far below the 8-tokens-per-expert gate).

```bash
pytest -v -s test_moe_decode_perf.py -k qwen3_moe
```

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

**S4 DPAS tile policies** — the single-pass mainloop (precedence 1)
now selects a dedicated 4-bit tile policy by the average tokens-per-
expert (`A_avg_M = total_tokens / E`), mirroring the reference
`w4a16` dispatch in `vllm-project/vllm-xpu-kernels`
(`grouped_gemm_xe2_interface.hpp`). Because the packed-nibble B stream
is half the byte volume of the INT8 path, the large-M tile is widened
to `128×256×32` (vs. the INT8 `128×128×16`) so the DPAS accumulators
and the halved B-side bandwidth are better utilised:

| `A_avg_M` bucket | WG tile (M×N×K) | Policy (`sycl_tla_moe_prefill_fp8_dpas.hpp`) |
| ---------------- | --------------- | -------------------------------------------- |
| `≤ 4`            | `8×64×32`       | `dpas_w4a16_policy_m_8`                       |
| `≤ 8`            | `16×64×32`      | `dpas_w4a16_policy_m_16` (= `w8a16_m_16`)     |
| `≤ 128`          | `32×64×32`      | `dpas_w4a16_policy_m_32` (= `w8a16_m_32`)     |
| `> 128`          | `128×256×32`    | `dpas_w4a16_policy`                           |

The mid-size `32×64` tile now covers `A_avg_M` up to 128 (previously it
jumped to the wide tile at 33), which avoids padding waste on the
common chunked-prefill batch sizes.

**S4 DPAS decode path** — the *decode* phase (int4-sym / `S4_CLIP`,
`!asym`, `ARK_MOE_DECODE_DPAS_S4` default ON) has its own dedicated
dispatch in the generated `sycl_tla_moe_decode_int4.cpp` translation unit,
mirroring vLLM-xpu-kernels' `w4a16` decode dispatch. It selects the DPAS
tile from the same `A_avg_M` ladder as prefill (`_m_8` → `_m_16` → `_m_32`
→ wide): the 8-row tile is used only for the tiny-batch tail (`A_avg_M ≤
4`), and the M tile grows once more than four tokens route to an expert
on average. An earlier revision hard-pinned the 8-row tile on the
assumption that decode only ever sees a handful of tokens per expert, but
that under-fills the M dimension and re-streams the (bandwidth-bound)
packed weights 2–4× on larger decode batches (many sequences, high top-k,
or few experts), roughly halving throughput versus the reference. It
reuses the shared per-group mainloop's 2D VNNI block load
(`get_block_2d_copy_A/B` + `make_block_2d_prefetch`) and register-resident
per-N scale (`sg_scale[]`, folded once per K-group), reading the same
`[E, N, K/2]` packed weights + `[E, N, K/group]` scales with no repack.
`ARK_MOE_DECODE_S4_DPAS_M8=1` forces the legacy hard-pinned 8-row tile for
A/B comparison (numerically identical; only the tile shape differs).
**Status: NEEDS-HARDWARE-VALIDATION** (untested port).

**Occupancy gate — decode-sized batches use the int4-asym kernel.** Even
the smallest DPAS tile processes 8 token rows per expert, so a batch with
fewer than 8 tokens per expert on average pays full weight-streaming cost
for mostly-padding rows. That is precisely the decode regime: on
MiniMax-M2 (192 experts) bs1 is 8 tokens and bs32 is 256 tokens, i.e.
0.04–1.3 tokens per expert, and measurements showed int4-sym (DPAS) at
0.31–0.34 ms/1.55 ms against int4-asym (scalar GEMV) at 0.12 ms/1.45 ms
for the same shapes. int4-sym decode is therefore routed to the *same*
scalar GEMV kernel that int4-asym uses (`launch_int4` / its coalesced
variant, with `Asym=false`) unless the batch supplies at least 8 tokens per
expert.
`ARK_MOE_DECODE_DPAS_S4_MIN_TPE` overrides the tokens-per-expert
threshold; `0` disables the gate (always DPAS when the shape gate allows),
which is what the accuracy and DPAS-vs-scalar perf tests set.

**Word-native nibble decode; sym keeps its signed nibbles.** Once both
modes shared the scalar GEMV, int4-sym was still slower than int4-asym in
the *same* kernel despite doing strictly fewer floating point operations.
The asymmetry was the nibble decode, and the first attempt at fixing it
(the `^ 0x88` sign-flip identity `signed == (unsigned ^ 8) - 8`) did not
close the gap: it kept sym on 8-bit-typed operations — a `sycl::vec<uint8_t,N>`
XOR plus per-byte mask/shift — which Xe expands into narrow-type ALU work
rather than executing on the native 32-bit datapath, and it forced sym to
carry a constant zero-point of 8 (see *activation sums* below).

Both modes now decode through the shared `decode_int4_octet` primitive,
which takes the 8 nibbles of a packed *32-bit word* and extracts each one
with a single DWORD shift/mask pair (asym) or a DWORD shift-left +
arithmetic shift-right pair (sym). No 8-bit-typed vector, no XOR, no
narrowing casts, and one 32-bit load per 8 K elements instead of a byte
vector. The per-nibble results are bit-identical to `decode_int4_pair` for
every one of the 2^32 input words in both modes (verified exhaustively), so
this is a pure instruction-selection change. It applies to `launch_int4`,
`launch_int4_coalesced`, and — since the primitive is shared — the prefill
mixed-dtype path.

Because sym once again recovers *true signed* nibbles, its per-group fold
collapses to `acc += scale * Σ a·q` with no zero-point term at all, whereas
asym keeps `acc += scale * (Σ a·q − zero · Σ a)`.

**4-byte-blocked coalesced repack.** The coalesced fallback
(`launch_int4_coalesced`, `ARK_MOE_DECODE_COALESCE_INT4` default ON)
repacks the `[E, N, K/2]` weights on-device so sub-group loads are
contiguous. The original repack layout `[E, N/16, K/2, 16]` put one byte
per lane per step, so although the 16 lanes together covered one cache
line, each lane still issued a *byte* load. The layout is now
`[E, N/16, ceil(K/8), 16, 4]`: a chunk holds four consecutive packed bytes
for each of the 16 columns of a tile, lane-major, so lane `l` reads its
four bytes at chunk offset `l*4` and the sub-group still spans 64
contiguous bytes. A lane's four bytes are contiguous, hence exactly one
little-endian 32-bit word: the lane issues a single DWORD load (4× fewer
weight-load instructions) and feeds it straight to `decode_int4_octet`, so
all eight nibbles come out with native 32-bit ops in both modes. Group
sizes that are a multiple of 8 (16/32/64/128/256 — every shipped quant
config) start each K-group on a chunk boundary so the vector stage covers
the whole group; other even group sizes fall back to a scalar
prologue/epilogue over the same layout. The external `[E, N, K/2]` weight
contract is unchanged.

**Hoisted activation sums (asym only).** The asym int4 GEMVs fold the
per-group scale/zero as `scale * (Σ a·q − zero · Σ a)`. `Σ a` depends only
on the activation row and the K-group, not on the output column, yet it
used to be recomputed inside the inner loop — once per sub-group lane (16×
redundant) and again for every N-tile work-group — costing one extra float
add per K element. It is now precomputed once into a
`[total_tokens, K/group_size]` fp32 table (`launch_act_group_sums`), so
the GEMV inner loop only accumulates `Σ a·q` and reads one float per
group. The summation order changes by a few fp32 ULPs, far inside the
kernel's quantization tolerance.

**Sym skips the pre-pass entirely.** `launch_act_group_sums` is a separate
`parallel_for`, and on an in-order queue it fully serializes ahead of the
GEMV. That is a poor trade at decode sizes: it saves one float add per K
element in a loop that is already memory-bound, but adds a whole kernel
dispatch to a call whose GEMV is only tens of microseconds at bs1 — which
is why routing sym through the biased-unsigned decode made sym *slower*,
not faster. Now that sym decodes true signed nibbles it has no zero-point
term, so the table is computed (and the kernel launched) only when
`Asym` is true.

**Pooled scratch instead of per-call `malloc_device`.** The repack buffer
used to be a transient USM allocation that had to be freed behind a
blocking `queue::wait()` on *every* decode call — and decode issues one
call per generated token, so that allocation plus sync was on the order of
the GEMV itself. Both the repack buffer and the activation-sum table now
come from a persistent, grow-on-demand slab, so steady-state decode
performs no allocation and introduces no host-side synchronization;
ordering between the producer kernels and the GEMV is already guaranteed
by the in-order queue. `ark.moe_decode_release_scratch()` (pybind
`moe_decode_release_scratch`) hands the memory back.

The slabs are served from the extension-wide `DeviceMemoryPool` (slots 9
and 10), the same manager the dnnl/xpu wrappers, the SDPA kernels and the
DPAS work-group counter use. It keys on the *device UUID*, so slab
identity follows the device rather than a raw `sycl::queue*` — which
means a slab can no longer outlive the queue it was keyed on, a recycled
queue address can no longer be handed someone else's slab, and two
wrappers around one device no longer allocate two slabs. The bookkeeping
lives in `sycl_tla_moe_decode_scratch.cpp` so the module holds exactly one
instance of it (`sycl_tla_moe_decode_scratch.hpp` declares the API only),
and `release_decode_scratch()` detaches the slabs from the pool under its
lock and then runs `queue::wait()`/`sycl::free()` *after* releasing it, so
no unbounded device sync happens inside the lock. The flip side of keying
by device is that two queues on the same device now share a slab, so these
entry points must not be driven concurrently from two queues on one
device.

The repack *kernel* still runs on every call by default. Setting
`ARK_MOE_DECODE_INT4_REPACK_CACHE=1` reuses the previous repack when the
weight buffer address and shape are unchanged, which is valid for a real
inference loop where the weights are fixed. It is **off by default**
because the tag is a pointer identity: a freed-then-reallocated weight
tensor can land on the same address (torch's caching allocator makes this
common in test loops), and a stale repack would silently produce wrong
results. Callers that enable it must call
`ark.moe_decode_release_scratch()` before dropping the weight tensor.
Growing the slab also drops the cached repack, since the reallocated
bytes are undefined.

| Env var | Default | Effect |
| ------- | ------- | ------ |
| `ARK_MOE_DECODE_COALESCE_INT4` | ON | Use the coalesced, 4-byte-blocked repack GEMV for the int4 scalar fallback; `0` forces the legacy per-lane-strided `launch_int4`. |
| `ARK_MOE_DECODE_COALESCE_MIN_TOKENS` | `num_experts * TOKEN_BLOCK` | Minimum total tokens before the coalesced kernel is worth its repack pass; `0` disables the gate (what the parity/A-B tests set). |
| `ARK_MOE_DECODE_INT4_REPACK_CACHE` | OFF | Reuse the repack across calls on the same weight buffer. Only safe when the caller owns the weight lifetime. |

Perf A/B for the coalesced path is
`test_moe_decode_perf.py::test_perf_int4_coalesced_vs_strided` (toggles
`ARK_MOE_DECODE_COALESCE_INT4` 0/1 on the same shapes). Correctness is
covered by `test_moe.py::test_decode_int4_coalesced_matches_scalar`,
`::test_decode_int4_coalesced_token_blocking`,
`::test_decode_int4_coalesced_unaligned_group_size` (group sizes that are
not a multiple of 8, exercising the scalar prologue/epilogue) and
`::test_decode_int4_repack_cache`.

**Occupancy-gate threshold sweep.** The default
`ARK_MOE_DECODE_DPAS_S4_MIN_TPE` of 8 was derived from the row count of
`dpas_w4a16_policy_m_8` rather than measured. The sweep that locates the
real crossing point is
`test_moe_decode_perf.py::test_perf_int4_sym_dpas_vs_scalar_threshold`;
its default token counts (16–128) all sit far below the gate (8 × 192
experts == 1536 tokens), so pass `--all-shapes` to extend the sweep to
256/512/1024/1536/3072 tokens and bracket the gate from both sides. The
default stays at 8 until hardware numbers say otherwise.

Accuracy parity is covered by
`test_moe_prefill_accuracy.py::test_accuracy_int4_dpas_per_group`,
which forces `ARK_MOE_PREFILL_DPAS_S4=1` +
`ARK_MOE_PREFILL_DPAS_INT8=0` so the single-pass mainloop is
exclusively exercised, at the same production shapes as
`test_accuracy_int4`, with tolerance `rtol=atol=1e-1`.

## FP8 Decode Paths (`sycl_tla_moe_decode.hpp`)

int4-sym decode is now at target, and the same two levers that got it
there apply to FP8: get the dequant off the byte-typed datapath, and stop
paying setup cost per decode call. On top of that, the FP8 MoE dispatch
from vllm-xpu-kernels is mirrored into a decode-specialised entry point.
Both levers have since landed and **FP8 decode is at target too** — the
word-native dequant, the K-split lane mapping with its N-blocking, and the
removal of the per-call routing sync. That is what moved the unified
`ark.moe(phase="auto")` cutoff from 32 to 128 total tokens (see
*Auto-dispatch cutoff* below).

**Word-native FP8 decode (`ARK_FP8_DECODE_MODE`, default `word`).** The
decode GEMV does roughly one multiply-add per weight byte, so the dequant
*is* the kernel. Both legacy decoders paid real work per byte: `lut`
issues a memory load per weight element into the 128-entry magnitude table
plus a sign select, and `bits` runs a branchy `ldexp` chain. Both also
indexed an 8-bit-typed `sycl::vec<uint8_t, 16>`, which Xe's 32-bit ALU
lanes cannot address directly, so IGC expands it into narrow-type
regioning — exactly the problem `decode_int4_octet` fixed for nibbles.

None of that work is necessary, because an FP8 byte is already an
IEEE-style float and fp16 is a *superset* of both FP8 formats: the whole
conversion is a bit-field move.

| Format | fp16 bit pattern | Exactness |
| ------ | ---------------- | --------- |
| E5M2   | `byte << 8` | Bit-exact for all 256 encodings — same sign position, same 5-bit exponent, same bias 15. Subnormals stay subnormal, `exp==31` stays Inf/NaN. |
| E4M3   | `(byte + (byte & 0x80)) << 7` | Bit-exact for all 254 finite encodings (normals, subnormals, both zeros), yielding the true value × `2^-8`. |

E4M3's 4-bit exponent has bias 7 against fp16's bias 15, so the field move
leaves a constant `2^-8` factor; `fp8_word_scale_bias<IsE4M3>()` (`256.0f`)
is folded into the per-K-group scale, an exact power of two applied once
per group, so it costs nothing per element. Adding the sign bit to itself
carries it exactly one position further, which is why the sign move and
the magnitude move collapse into one add plus one shift.

The kernel reads the weights as `sycl::vec<uint32_t, 4>` — the same
16-byte transaction and the same 16-byte alignment requirement as the byte
vector it replaces — and `decode_fp8_quad_half_bits` turns each 32-bit word
into four fp16 bit patterns in a handful of native DWORD ops (SWAR, no
cross-lane carry). Two partial accumulators break the fp32 dependency
chain, as in `int4_decode_chunk`. Both primitives live in
`sycl_tla_moe_dequant.hpp`, and both were verified exhaustively over all
256 byte values in both formats.

**E4M3 NaN caveat.** The two E4M3 NaN encodings (`0x7F` / `0xFF`;
`torch.float8_e4m3fn` has no Inf) decode to ±480 instead of NaN, since fp16
has no NaN pattern reachable by a pure field move. auto-round FP8
checkpoints are produced by scaling to `finfo(float8_e4m3fn).max == 448`
and clamping, so those two encodings cannot occur. Callers that need NaN
propagation can select `ARK_FP8_DECODE_MODE=lut` or `=bits`.

**K-split lane mapping (`ARK_MOE_DECODE_FP8_KSPLIT`, default ON).** Once
the dequant is a couple of DWORD ops, the scalar GEMV is purely a
bandwidth problem: at ~1 multiply-add per weight byte it can only run as
fast as the expert tile streams in. The original mapping gave one output
element to one *work-item*, so a lane walked a whole `[n, K]` weight row on
its own. Two costs follow:

* **Uncoalesced weight loads.** Lanes `l` and `l+1` of a sub-group read
  bytes `K` apart, so each 16-byte load instruction is split into 16
  cache-line requests. No DRAM byte is wasted (each lane consumes its lines
  as it walks the row) but the memory controller sees 16 independent
  streams per sub-group, the pattern DRAM row buffers handle worst.
* **Too few threads.** The grid is `total_tokens × N / 16` sub-groups —
  768 SIMD16 threads for a MiniMax-M2 batch-1 step (8 tokens, N=1536),
  below the thread slots of a BMG-class GPU, so there are never enough
  loads in flight to hide DRAM latency.

`launch_fp8_ksplit` transposes the mapping: one *sub-group* per output
element, with the 16 lanes splitting K. Lane `l` owns the 16 consecutive K
elements at `l*16` inside each 256-element step, so one instruction covers
256 **contiguous** weight bytes (four full cache lines) and 512 contiguous
activation bytes, each thread walks a single sequential stream, and the
thread count grows 16× (12288 sub-groups for that batch-1 step). The price
is one `reduce_over_group` per output element — a handful of shuffles
against `K` multiply-adds — and 16× more activation traffic out of L1,
which has ample headroom at this arithmetic intensity.

This is the same problem the int4 fallback solves by repacking weights
into an N-tiled layout (`ARK_MOE_DECODE_COALESCE_INT4`), which costs a
full extra pass over the weight tensor and a scratch buffer. FP8 weights
are one byte per element and already K-contiguous, so K-splitting the lane
mapping gets the same coalescing with no repack, no scratch and no extra
kernel launch.

The kernel indexes the scale array with a shift, so the shape gate
requires a power-of-two `group_size ≥ 16` (every shipped FP8 config — 32 /
64 / 128 / 256 — passes) plus `N%16==0`, `K%group_size==0` and `K ≥ 256` (so
every lane of the sub-group owns at least one chunk); anything else keeps the
legacy GEMV, which handles arbitrary group sizes. All three
`ARK_FP8_DECODE_MODE` decoders run under both mappings, so the mode A/B
stays apples-to-apples. **Status: hardware-validated** — this mapping is
what put FP8 decode at target.

**N-blocking inside the K-split kernel (`ARK_MOE_DECODE_FP8_KSPLIT_NCOLS`,
default 2).** With one output column per sub-group the hot loop issues, per
16-byte weight chunk, one weight message *and* one 32-byte activation
message — half of what a thread requests is the activation row, which every
column of that token re-reads — and only two weight loads are ever in
flight. Giving a sub-group `NCOLS` consecutive columns loads the activation
chunk once and reuses it for all of them:

| | `NCOLS=1` | `NCOLS=n` |
| --- | --- | --- |
| activation messages per weight chunk | 1 | 1/n |
| independent weight loads in flight | 2 | 2n |

The first effect cuts request-queue pressure; the second raises
memory-level parallelism, which is what a streaming GEMV sitting well below
peak DRAM bandwidth is actually limited by. The cost is `n` times the live
weight vectors and accumulators, so past some point the kernel spills — hence
the small ladder (1, 2, 4) and the conservative default.

A work-group still holds 16 sub-groups, so it now covers `16 * NCOLS`
columns; an `N` that cannot be tiled at the requested factor falls back to
the largest valid smaller power of two on the host side (`N=1536` and
`N=3072` tile at every factor). The lane → K-chunk mapping, the per-chunk
scale fold and the final `reduce_over_group` are untouched, so the
arithmetic per output element is unchanged and `NCOLS=1` reproduces the
previous kernel exactly. `test_perf_fp8_ksplit_ncols_sweep` prints all three
factors per shape so the default can be set from measured data.
**Status: hardware-validated at the shipped default (`NCOLS=2`).**

*Per-mode translation units.* The mode and `NCOLS` ladders multiply out:
2 dtypes × 2 formats × 3 decode modes × (one plain kernel + three `NCOLS`
factors) = 48 SYCL kernels. Instantiating all of them inside
`sycl_tla_moe_decode_fp8.cpp` — which is what calling `launch_fp8_by_mode`
directly did — measured **~14.4 GiB peak compiler RSS** for that single file,
against 4–8 kernels for every other decode TU. Each (dtype, format, mode)
triple therefore lives in its own generated
`sycl_tla_moe_decode_fp8_{f16,bf16}_{e4m3,e5m2}_{word,lut,bits}.cpp`, declared
in the light `sycl_tla_moe_decode_fp8_helpers.hpp`, leaving exactly four
kernels per TU. `sycl_tla_moe_decode_fp8.cpp` itself now instantiates **no
kernels at all**: it reads `fp8_decode_mode()` on the host and calls the
matching `moe_decode_fp8_detail::dispatch_*`, exactly as `launch_fp8_by_mode`
used to. The K-split decision still happens inside the selected TU from the
same inputs, so every variant stays numerically identical.

**Routing-table validation (`ARK_MOE_VALIDATE_ROUTING`, default OFF).** The
Python entry point used to check `sum(num_tokens_per_expert) == total_tokens`
on every call. For a routing table that already lives on the device that
sum means a reduction kernel plus a *blocking* device-to-host copy, i.e. a
full pipeline flush — on a decode step whose kernel takes ~150 µs, and once
per generated token. It also lands inside the timed region of every decode
benchmark, because the queue is idle when the timing event is recorded.
The sum is now a caller contract (the C++ side never needed the host value:
it consumes the device pointer and derives `expert_id_per_token` on-device,
clamped to `num_experts - 1`); set `ARK_MOE_VALIDATE_ROUTING=1` to restore
the eager check when debugging a router. Host-side (CPU) routing tables are
still checked unconditionally, since summing those is free.

**FP8 DPAS decode dispatch.** The FP8 decode path (`ARK_MOE_DECODE_DPAS_FP8`
default ON) is the FP8 twin of the S4 decode dispatch: same mainloop, same
`[E, N, K]` FP8 bytes + `[E, N, K/group]` scales, no repack. It differs from
the prefill dispatch in two decode-specific ways.

*Finer small-M ladder.* The reference `w8a16` dispatch in vllm-xpu-kernels
bottoms out at the 16-row tile, while its `w4a16` dispatch has an extra
8-row bucket. Decode `A_avg_M` sits far below 16, so the missing rung means
half of every M tile is padding and the bandwidth-bound FP8 weights get
streamed for rows that contribute nothing. `dpas_w4a16_policy_m_8` carries
no 4-bit-specific types — it is purely an `8×64×32` `WGTile` / `SGLayout`
shape — so the FP8 mainloop reuses it verbatim, closing that gap:

| `A_avg_M` bucket | WG tile (M×N×K) | Policy |
| ---------------- | --------------- | ------ |
| `≤ 4`            | `8×64×32`       | `dpas_w4a16_policy_m_8` |
| `≤ 8`            | `16×64×32`      | `dpas_w8a16_policy_m_16` |
| `≤ 128`          | `32×64×32`      | `dpas_w8a16_policy_m_32` |
| `> 128`          | `128×128×16`    | `dpas_w8a16_policy` |

The upper rungs match the S4 *decode* ladder rather than the FP8 prefill
one, whose `≤ 512 → m_32` rung is tuned for prefill-sized batches.

Like S4, the rungs are *not* instantiated inside `sycl_tla_moe_decode_fp8.cpp`.
Each one lives in its own generated translation unit
(`sycl_tla_moe_prefill_fp8_{f16,bf16}_{e4m3,e5m2}[_tiny|_mid|_large].cpp`,
declared in `sycl_tla_moe_prefill_fp8_helpers.hpp`), and the decode TU only
picks a rung host-side and calls the corresponding `moe_prefill_fp8_detail::
dispatch_*` function. Expanding the whole ladder in one TU means
2 dtypes × 2 formats × 4 policies × 4 group sizes = 64 cutlass grouped-GEMM
instantiations, which measured **~14.4 GiB peak compiler RSS** for that single
file; splitting keeps each TU to one policy. `sycl_tla_moe_decode_fp8.cpp` no
longer includes `sycl_tla_moe_prefill_fp8_dpas.hpp` at all — its shape gate
goes through the light `moe_prefill_fp8_detail::shape_ok` declaration.

*Per-policy translation units for the bf16/fp16 grouped GEMM.* The unquantized
`moe_gemm` path picks one of three work-group tile policies from the output
width `N` (`≤ 64 → 256×64×32` 8×1, `≤ 512 → 256×128×32` 8×2, `> 512 →
256×256×32` 8×4; `ARK_MOE_GEMM_FIXED_TILE=1` pins the historical 8×2 tile).
Expanding all three inside `sycl_tla_moe_{f16,bf16}.cpp` put three full cutlass
grouped-GEMM instantiations in a single TU and measured **~2219 MB peak
compiler RSS** for each of those two files, while every other cutlass TU in the
build carries exactly one policy. Each policy therefore lives in its own
generated `sycl_tla_moe_{f16,bf16}_{n64,n128,n256}.cpp`, declared in the light
`sycl_tla_moe_gemm_helpers.hpp`. `sycl_tla_moe_{f16,bf16}.cpp` no longer
includes `sycl_tla_moe.hpp` and instantiates **no kernels at all**: it calls
`moe_gemm_detail::select_tile_policy(N)` on the host — the same heuristic,
moved verbatim into the light header — and forwards to the matching
`moe_gemm_detail::dispatch_*`. All three policies keep the same tile shapes and
therefore the same SYCL kernel names as before, so the split is numerically and
behaviourally invisible.

*Pooled atomic counter.* Allocating the work-group counter with
`sycl::malloc_device` and releasing it with `sycl::free` on every call
forces two queue synchronizations per dispatch. At prefill sizes that is
noise, at decode sizes — where the GEMM itself is only tens of
microseconds and one call is issued per generated token — it is a large
fraction of the total. Every DPAS dispatch, prefill and decode alike,
therefore serves the counter from the extension-wide device scratch pool
instead (`get_atomic_scratch_buffer`, a thin wrapper over
`DeviceMemoryPool::get_scratch_mem` on a dedicated slot, shared by the
FP8, INT8 and S4 headers so all three use one allocation per device).
Taking the fast path also skips the `fill_expert_id_per_token` pre-pass,
since the DPAS dispatch consumes `num_tokens_per_expert` directly — one
fewer kernel launch on the decode timeline. **Status:
NEEDS-HARDWARE-VALIDATION** (this header is an untested port).

**Occupancy gate — real decode batches stay on the scalar GEMV.** Same
reasoning as int4-sym: the smallest tile the decode ladder can pick
processes 8 token rows per expert, so below 8 tokens per expert on average
the tile is mostly padding. That is exactly the decode regime (MiniMax-M2,
192 experts: 0.04–1.3 tokens per expert), so FP8 decode is routed to the
scalar GEMV unless the batch supplies at least 8 tokens per expert.
`ARK_MOE_DECODE_DPAS_FP8_MIN_TPE` overrides the threshold; `0` disables the
gate, which is what the parity and A/B perf tests set. Shapes that fail the
per-group shape gate (`N%64==0`, `K%32==0`, `K%group_size==0`,
`group_size ∈ {32,64,128,256}`) always fall back to the scalar GEMV.

**Auto-dispatch cutoff (`ARK_MOE_AUTO_DECODE_MAX_TOKENS`, default 128).**
`ark.moe(phase="auto")` routes to `moe_gemm_decode` when
`activations.shape[0] <= cutoff` and to `moe_gemm_prefill` otherwise. The
cutoff was 32 while the decode GEMV was still the bottleneck: only the tiny
single-/few-stream case was worth keeping off the prefill grouped GEMM. Now
that the FP8 decode GEMV is at target (as int4-sym already was) the GEMV
stays ahead across the whole small-batch range rather than just at the bs1
extreme, so the cutoff is 128 total tokens; above that each expert receives
enough rows to fill the DPAS M tile, which is where the grouped GEMM wins.
The `decode_threshold=` keyword overrides it per call and takes precedence
over the env var, and `phase="decode"` / `phase="prefill"` bypass the
heuristic entirely. Dispatch parity is covered by
`test_moe_unified.py::TestMoeUnifiedDispatch`, which pins both the cutoff
boundary (128 tokens still decode) and the overrides.

| Env var | Default | Effect |
| ------- | ------- | ------ |
| `ARK_FP8_DECODE_MODE` | `word` | FP8 decode implementation for the scalar GEMV: `word` (bit-field move + folded scale bias), `lut` (128-entry magnitude table), `bits` (inline bit manipulation). |
| `ARK_FP8_DECODE_USE_LUT` | unset | Legacy selector, still honoured when set explicitly and when `ARK_FP8_DECODE_MODE` is unset/unrecognised: truthy → `lut`, falsy → `bits`. Also still drives the mixed-input prefill path. |
| `ARK_MOE_AUTO_DECODE_MAX_TOKENS` | `128` | Total-token cutoff used by `ark.moe(phase="auto")`: at or below it the call goes to `moe_gemm_decode`, above it to `moe_gemm_prefill`. Non-positive/unparsable values fall back to the default; the `decode_threshold=` keyword wins over both. |
| `ARK_MOE_DECODE_DPAS_FP8` | ON | Route FP8 decode to the per-group DPAS grouped GEMM when the shape and occupancy gates pass; `0` forces the scalar GEMV. |
| `ARK_MOE_DECODE_DPAS_FP8_MIN_TPE` | `8` | Minimum tokens per expert before the DPAS path is taken; `0` disables the gate (what the parity/A-B tests set). |
| `ARK_MOE_DECODE_FP8_KSPLIT` | ON | Scalar-GEMV lane mapping: one sub-group per output element with the lanes splitting K (coalesced weight loads, 16× the threads); `0` forces the legacy one-work-item-per-output-element GEMV. Shapes outside the gate (power-of-two `group_size ≥ 16`, `N%16==0`, `K%group_size==0`, `K ≥ 256`) always use the legacy mapping. |
| `ARK_MOE_DECODE_FP8_KSPLIT_NCOLS` | `2` | Output columns one sub-group owns in the K-split GEMV (1, 2 or 4). Higher values reuse one activation load across more columns and keep more weight loads in flight, at the cost of more live registers. An `N` that `16 * NCOLS` cannot tile falls back to the largest valid smaller power of two. |
| `ARK_MOE_VALIDATE_ROUTING` | OFF | Eagerly check `sum(num_tokens_per_expert) == activations.shape[0]` for device-resident routing tables. The check costs a blocking device-to-host sync per call, so it is opt-in; CPU-resident tables are always checked. |

Perf A/B rows are `test_moe_decode_perf.py::test_perf_fp8_word_vs_lut`
(`speedup` is `lut / word`), `::test_perf_fp8_ksplit_vs_strided`
(`speedup` is `strided / ksplit`), `::test_perf_fp8_ksplit_ncols_sweep`
(`speedup` is `NCOLS=1 / best NCOLS`, with all factors printed) and
`::test_perf_fp8_dpas_vs_scalar` (`speedup` is `scalar / dpas`).
Correctness is covered by
`test_moe.py::test_decode_fp8_modes_match` (all three decoders agree, and
each tracks the dequant reference),
`::test_decode_fp8_ksplit_matches_strided` (both lane mappings agree, plus
a non-power-of-two `group_size` fallback case),
`::test_decode_fp8_ksplit_ncols_match` (every blocking factor agrees, plus
an untileable-`N` fallback case) and
`::test_decode_fp8_dpas_matches_scalar`.

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
