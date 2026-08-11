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
| `torch(ms)` | Per-expert `A @ W.T` on **pre-dequantized** weights (the dequant is outside the timed region) — the matmul-only PyTorch ceiling |
| `w4a16(ms)` | The existing ARK int4 kernel for the same phase (`moe_gemm_decode` / `moe_gemm_prefill`) |
| `w4a8(ms)` | The new int8-compute path (`ark.moe_gemm_w4a8`) |
| `TFLOPS` | `total_tokens × N × K × 2 / time` |
| `W GB/s` | Expert-weight bandwidth actually touched by the routed tokens (`active_experts × N × K × 1 byte / time`) — the limiter for memory-bound decode |
| `vs torch` / `vs w4a16` | Speedups (`other / w4a8`) |
| `prepack(ms)` | One-shot int4 → int8 AUTO_S8 conversion cost. Paid once at model load, **not** per forward. |

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
```

The `-s` flag is required to see the printed tables.

### As a standalone script (no pytest)

```bash
python test_moe_w4a8_perf.py                       # both phases, smallest batch
python test_moe_w4a8_perf.py --all-shapes          # full sweep
python test_moe_w4a8_perf.py --phase decode        # decode only
python test_moe_w4a8_perf.py --skip-accuracy       # perf only
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

## Shape constraints

The kernel requires:

* `N % 16 == 0` (the GEMV N tile and the DPAS N tile)
* `K % 64 == 0` (the DPAS K tile)
* `group_size % 8 == 0` and `K % group_size == 0`
* the resolved re-scale block must be a multiple of 64 and divide K

Both Qwen3-MoE GEMMs satisfy these (`K = 2048` and `K = 768`).

## Status

The W4A8 kernel is a new SYCL/CuTe port and is marked
`STATUS: NEEDS-HARDWARE-VALIDATION` in
`auto_round_kernel/wrapper/include/sycl_tla_moe_w4a8.hpp`. This script is the
intended on-hardware validation vehicle: run the accuracy sweep first (it will
catch layout/scale bugs immediately), then the perf sweep to tune the tile
ladder and the decode threshold.
