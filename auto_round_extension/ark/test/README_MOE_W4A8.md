# W4A8 MoE Kernel (int4 weight / int8 compute) — Performance & Accuracy

## Overview

`test_moe_w4a8_perf.py` benchmarks the **W4A8** ARK XPU MoE kernel for both the
**prefill** and **decode** phases, and checks its numerical accuracy against an
fp32 reference and against the existing W4A16 ARK path.

**W4A8** means:

| Component | Format |
|---|---|
| Weights on disk / in the checkpoint | int4 symmetric, `group_size = 32` (auto-round packing) |
| Weights in the GEMM mainloop | **int8** (`group = -1`, one scale per output channel) |
| Activations | dynamically quantized to **int8** per token (absmax) inside the kernel |
| Accumulator | int32 (`s8 × s8 → s32` DPAS) |
| Output | fp16 / bf16 |

## Why int8 compute is faster than int4 weight-only

The Xe DPAS pipeline has a native `s8 × s8 → s32` atom. A weight-only int4 path
must first widen the nibbles into the activation dtype, then run an fp16/bf16
matmul, and — because the int4 scales are per K-group of 32 — must fold the
accumulator every 32 K elements. That fold destroys the long-K accumulation the
DPAS pipeline needs to run at peak.

ARK already solves this for dense GEMM with its **`AUTO_S8`** option, which
re-scales int4 `group=32` weights into int8 `group=-1` weights:

```
sxt[e][n][j] = max_{g in block j} |s[e][n][g]| * 8 / 127     # 8 = 2^(4-1), the int4 full range
w8[e][n][k]  = round( w4[e][n][k] * s[e][n][k / group_size] / sxt[e][n][j] )
```

Because `|w4| <= 8`, the re-scaled value satisfies `|w8| <= 127` — the
conversion never clips. With the default block (the whole K axis) the mainloop
becomes a **single full-K int32 accumulation** followed by one scalar multiply
in the epilogue, which is the highest-throughput configuration.

This kernel applies the same idea to the MoE grouped GEMM. The conversion runs
**once** (at model load), not per forward pass.

## Performance targets and the roofline

The goals for this kernel are **prefill > 100 TFLOPS** and **decode > 300 GB/s**
of weight bandwidth. Whether the prefill goal is reachable at all is decided by
the *routing*, not by the kernel. A W4A8 grouped GEMM reads every active
expert's int8 weights exactly once and does `2 × rows_per_expert` FLOPs per
weight byte, so

```
arithmetic intensity = 2 × rows_per_expert            [FLOP / byte]
TFLOPS              <= 2 × rows_per_expert × weight_bandwidth
rows_per_expert      = batch × top_k / active_experts
```

The `N` and `K` factors cancel — only the routing matters:

| Model tokens | Routed rows | rows/expert | Bandwidth needed for 100 TFLOPS |
|---|---|---|---|
| 128 (default prefill batch) | 1024 | 8 | 6250 GB/s |
| 512 | 4096 | 32 | 1563 GB/s |
| 2048 | 16384 | 128 | 391 GB/s |
| 4096 (`test_perf_prefill_compute_bound`) | 32768 | 256 | 195 GB/s |
| 8192 (`--all-shapes`) | 65536 | 512 | 98 GB/s |

So ~4.5 TFLOPS at the default batch is **not** a kernel deficiency: at 8 rows
per expert and the ~285 GB/s of weight bandwidth the kernel actually achieves,
the ceiling is `2 × 8 × 285e9 = 4.56 TFLOPS` — the measured value, i.e. the
kernel is already running at the DRAM roofline. Reaching 100 TFLOPS there would
require 6.25 TB/s, more than 10× any current GPU. On a device streaming
~285 GB/s the target first becomes reachable at ~176 rows per expert (~2816
model tokens), which is why `test_perf_prefill_compute_bound` measures at 4096
model tokens.

The perf table therefore prints `rows/E` and `BW@100T` next to the measured
numbers, and each sweep ends with a verdict block:

```
targets [prefill]: prefill compute > 100 TFLOPS
  device copy bandwidth probe: 400 GB/s
  qwen3 up     tokens=1024   rows/E=8.0        4.56 TFLOPS vs 100 -> N/A (bandwidth bound: ...)
  qwen3 down   tokens=32768  rows/E=256.0    102.40 TFLOPS vs 100 -> PASS
```

A row is reported `N/A` rather than `FAIL` when the device bandwidth probe (one
large device-to-device copy, measured once per run) shows the target is
unreachable at that routing. The verdict is informational by default; pass
`--enforce-targets` to turn it into a hard assertion.

### Why `vs w4a16` is below 1.0 at small batches

The same intensity argument explains the `vs w4a16` column. W4A8 streams **2×
the weight bytes** of the int4 path (one byte vs. half a byte per element) in
exchange for ~2× the DPAS peak, so it only wins once the GEMM is compute bound:

```
crossover rows/expert ~= int8_peak_TOPS / (4 × weight_bandwidth)
```

With ~233 TOPS of int8 DPAS and ~285 GB/s that is ~200 rows per expert
(~3200 model tokens). Decode (1 row per expert) and small-batch prefill are far
below it, so readings of 0.55–0.71× are expected there: W4A8 is a large-batch
prefill optimization, and at decode it can only help by improving the *memory*
path.

## Decode: coalesced K-split mapping

The decode GEMV originally assigned **one work-item per output element**: lane
`l` of a sub-group computed column `n0 + l` and walked the whole K axis alone.
Consecutive lanes then read addresses `K` bytes apart, so a single load touched
16 different cache lines and used 16 of the 64 bytes each one delivered. At
batch 1 the kernel also launched only `total_tokens × N/16` sub-groups (768
SIMD16 work-items for the up-proj) — far too few to cover memory latency.

The fix is the **K-split** mapping that already put the FP8 decode path at its
bandwidth target (`launch_fp8_ksplit` in `sycl_tla_moe_decode.hpp`): one
sub-group cooperates on `NCOLS` output columns, and lane `l` owns the 16
consecutive K elements at `l × 16` within each 256-element step. Every load now
covers **256 contiguous weight bytes**, the grid grows ~16×, and one
`sycl::reduce_over_group` per output element folds the lane partials.

The loop is *block-outer* (for each AUTO_S8 re-scale block, then over K inside
it), so the block scale is hoisted to a scalar and the hot loop contains no
division — and, unlike the FP8 variant, no power-of-two constraint on the block
size. The arithmetic is unchanged: int32 partials per lane per block, scaled by
the block scale, summed across the sub-group, then multiplied by the per-token
activation scale. `test_decode_ksplit_matches_legacy` asserts that both mappings
produce bit-identical output.

The mapping requires `N % 16 == 0`, a re-scale block that is a multiple of 16
and at least 256, and `K % block == 0`. Anything else (for example an explicit
`--rescale-group-size 64`) falls back to the original kernel automatically.

Decode also issues **one kernel launch fewer per step**: each token's expert id
is derived inside the activation-quantization kernel — which already runs one
sub-group per token — instead of by a separate `fill_expert_id_per_token`
launch. At batch 1 the entire GEMV takes ~45 µs, so a saved launch is not noise.

## What the script measures

### Accuracy table

| Column | Meaning |
|---|---|
| `block` | Resolved `AUTO_S8` re-scale block size (K = one scale per output channel) |
| `SNR ref(dB)` | W4A8 vs. an fp32 reference built from the **dequantized** int4 weights. Isolates the error added by int8 activations + the AUTO_S8 re-scale, excluding the int4 weight-quantization error itself. |
| `cos ref` | Cosine similarity against the same reference |
| `maxrel ref` | Max relative error, normalized by `max(|ref|, 0.01 · max|ref|)` so near-zero outputs don't dominate |
| `SNR w4a16(dB)` / `cos w4a16` | W4A8 vs. the existing W4A16 ARK kernel — the quality delta a caller sees when switching paths |
| `w4a16 SNR ref` | W4A16 vs. the same fp32 reference, so the two paths can be compared on equal footing |

The pytest cases assert `SNR ref >= 20 dB` and `cosine >= 0.99`. Per-token
absmax int8 activations lose roughly 7 bits of mantissa, so healthy runs land
comfortably above that; anything below indicates a *structural* bug (wrong scale
block, transposed layout, wrong expert offset) rather than mere lossiness.

### Performance table

| Column | Meaning |
|---|---|
| `torch(ms)` | Per-expert `A @ W.T` on **pre-dequantized** weights (the dequant is outside the timed region) — the matmul-only PyTorch ceiling. `--` when the baseline is skipped (compute-bound rows, where the dequantized `[E, N, K]` copy would not fit alongside everything else) |
| `w4a16(ms)` | The existing ARK int4 kernel for the same phase (`moe_gemm_decode` / `moe_gemm_prefill`) |
| `w4a8(ms)` | The new int8-compute path (`ark.moe_gemm_w4a8`) |
| `rows/E` | Routed tokens per **active** expert. Arithmetic intensity is `2 × rows/E` FLOPs per weight byte, so this single number decides whether a shape can be compute bound at all |
| `TFLOPS` | `total_tokens × N × K × 2 / time` |
| `W GB/s` | Expert-weight bandwidth actually touched by the routed tokens (`active_experts × N × K × 1 byte / time`) — the limiter for memory-bound decode |
| `BW@100T` | Weight bandwidth this shape would need to reach 100 TFLOPS. When it exceeds what the device can stream, `TFLOPS` is capped by memory and no kernel change can hit the target at that shape |
| `vs torch` / `vs w4a16` | Speedups (`other / w4a8`) |
| `prepack(ms)` | One-shot int4 → int8 AUTO_S8 conversion cost. Paid once at model load, **not** per forward. |

Each sweep is followed by a `targets [...]` block with the PASS / FAIL / N/A
verdict described in [Performance targets and the roofline](#performance-targets-and-the-roofline).

## Shapes

Qwen3-MoE, matching the shape group the int4 MoE work targets:

```
hidden_size = 2048,  intermediate_size = 768
num_local_experts = 128,  num_experts_per_tok = 8
int4-sym weights, group_size = 32

qwen3 up    (gate/up-proj):  N = 2 × 768 = 1536,  K = 2048
qwen3 down  (down-proj)   :  N = 2048,            K =  768
```

Routed expert-token rows are `batch × top_k`, spread round-robin over the 128
experts. Default batches: `128` for prefill and `1` for decode; `--all-shapes`
widens them to `{128, 512, 2048, 8192}` and `{1, 2, 8, 16}` respectively.
`test_perf_prefill_compute_bound` adds a single batch of `4096` model tokens
(256 rows per expert) — the smallest sweep point where the 100 TFLOPS goal is
not capped by weight bandwidth.

## How to run

### As a pytest suite

```bash
cd /path/to/auto_round_extension/ark/test

# Everything (accuracy + perf, both phases), smallest batch only
pytest -v -s test_moe_w4a8_perf.py

# Full batch sweep
pytest -v -s test_moe_w4a8_perf.py --all-shapes

# Accuracy only / perf only
pytest -v -s test_moe_w4a8_perf.py -k accuracy
pytest -v -s test_moe_w4a8_perf.py -k perf

# One phase
pytest -v -s test_moe_w4a8_perf.py -k decode

# The compute-bound prefill case (4096 model tokens), where the TFLOPS goal is reachable
pytest -v -s test_moe_w4a8_perf.py -k compute_bound

# Make the performance goals hard assertions instead of a printed verdict
pytest -v -s test_moe_w4a8_perf.py -k perf --enforce-targets
```

The `-s` flag is required to see the printed tables.

### As a standalone script (no pytest)

```bash
python test_moe_w4a8_perf.py                       # both phases, smallest batch
python test_moe_w4a8_perf.py --all-shapes          # full sweep
python test_moe_w4a8_perf.py --phase decode        # decode only
python test_moe_w4a8_perf.py --skip-accuracy       # perf only
python test_moe_w4a8_perf.py --compute-bound       # add the 4096-token prefill case
python test_moe_w4a8_perf.py --dtype fp16          # fp16 activations
python test_moe_w4a8_perf.py --rescale-group-size 256
python test_moe_w4a8_perf.py --warmup 10 --iters 100
```

The script exits non-zero if any accuracy gate fails.

## Python API

```python
import auto_round_kernel as ark

# 1) One-shot conversion at model load.
#    weights : [E, N, K // 2] uint8  (packed int4-sym)
#    scales  : [E, N, K // group_size] fp16/bf16
weights_s8, wscales, block = ark.moe_w4a8_prepack(weights, scales, group_size=32)

# 2) Per forward pass (prefill or decode).
out = ark.moe_gemm_w4a8(
    activations,  # [total_tokens, K] fp16/bf16, rows sorted by expert
    weights_s8,  # [E, N, K] int8
    wscales,  # [E, N, K // block] fp32
    num_tokens_per_expert,  # [E] int32
    rescale_block_size=block,
    phase="auto",  # "auto" | "decode" | "prefill"
)
```

A convenience wrapper does both, caching the conversion on the weight/scale
tensor identity:

```python
out = ark.moe_w4a8(
    activations,
    weights,
    num_tokens_per_expert,
    scales=scales,
    group_size=32,
    phase="auto",
)

ark.clear_moe_w4a8_prepack_cache()  # drop the cached int8 weights
ark.moe_w4a8_release_scratch()  # hand back the device scratch slabs
```

Helper: `ark.moe_w4a8_rescale_block_size(K, group_size, rescale_group_size)`
resolves the effective block size (and therefore the `wscales` shape) without
allocating anything.

## Memory cost

The prepacked weights are `E × N × K` **bytes** (int8), i.e. **2× the packed
int4 weights**:

| Shape | int4 packed | int8 prepacked |
|---|---|---|
| qwen3 up (E=128, N=1536, K=2048) | 201 MB | 402 MB |
| qwen3 down (E=128, N=2048, K=768) | 100 MB | 201 MB |

Because they are kept for the process lifetime, W4A8 trades memory for compute
throughput. The cache entry also pins the source int4 `weights` / `scales`
tensors (its key is pointer identity, so a freed-and-reallocated buffer could
otherwise collide with another layer's weights). Use `cache_prepack=False` on
`ark.moe_w4a8` (or `clear_moe_w4a8_prepack_cache()`) if that trade isn't worth
it for a given deployment.

## Environment variables

| Variable | Effect |
|---|---|
| `ARK_MOE_W4A8_AUTO_S8` | Override the AUTO_S8 re-scale block size. Unset / `-1` = one scale per output channel (fastest). Values that aren't a multiple of both `group_size` and 64, or that don't divide K, silently fall back to K. |
| `ARK_MOE_W4A8_DECODE_MAX_TOKENS` | Token count at or below which `phase="auto"` picks the GEMV (default `128`). |
| `ARK_MOE_W4A8_DECODE_KSPLIT` | Coalesced K-split decode mapping; **on by default**. Set to `0` to fall back to the original one-work-item-per-output GEMV (useful for A/B measurements). Ignored when the shape doesn't qualify. |
| `ARK_MOE_W4A8_DECODE_KSPLIT_NCOLS` | Output columns per sub-group in the K-split mapping: `1`, `2` (default) or `4`. Higher values amortize the activation loads over more columns but need `N % (16 × NCOLS) == 0`. |

## Shape constraints

The kernel requires:

* `N % 16 == 0` (the GEMV N tile and the DPAS N tile)
* `K % 64 == 0` (the DPAS K tile)
* `group_size % 8 == 0` and `K % group_size == 0`
* the resolved re-scale block must be a multiple of 64 and divide K

Both Qwen3-MoE GEMMs satisfy these (`K = 2048` and `K = 768`).

The decode K-split mapping additionally needs a re-scale block of at least 256
that is a multiple of 16; shapes that miss it use the original GEMV instead of
failing.

## Status

The W4A8 kernel is a new SYCL/CuTe port and is marked
`STATUS: NEEDS-HARDWARE-VALIDATION` in
`auto_round_kernel/wrapper/include/sycl_tla_moe_w4a8.hpp`. This script is the
intended on-hardware validation vehicle: run the accuracy sweep first (it will
catch layout/scale bugs immediately), then the perf sweep to tune the tile
ladder and the decode threshold.

The decode K-split mapping is likewise unvalidated on hardware. Its index math
was checked against the legacy mapping with a host-side mock, and
`test_decode_ksplit_matches_legacy` re-checks it on device; if it ever
regresses, `ARK_MOE_W4A8_DECODE_KSPLIT=0` restores the previous behaviour
without a rebuild.
