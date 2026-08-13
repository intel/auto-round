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

### The device every number in this document was measured on

An **Intel Arc Pro B60** (Battlemage, `BMG-G21` — the default AOT target
`intel_gpu_bmg_g21`): 20 Xe2 cores / 160 XVEs at ~2.4 GHz, 24 GB GDDR6 on a
192-bit bus. The ceilings it sets:

| Ceiling | Value |
|---|---|
| int8 XMX (DPAS) | 160 XVEs × 512 int8 ops/clk × 2.4 GHz ≈ **197 TOPS** |
| bf16 / fp16 XMX | ≈ **98 TFLOPS**, half the int8 rate — the reason W4A8 exists |
| DRAM pin bandwidth | **456 GB/s**; the harness' device-copy probe reads ~400 GB/s (88% of pin) |
| Occupancy ceiling | 160 XVEs × 8 thread slots = **1280 concurrent SIMD16 sub-groups** |

So the two targets below are 51% of int8 peak and 66% of pin bandwidth. The
`Arc Pro B60 Dual` card exposes two such devices; the kernel sees one.

The goals for this kernel are **prefill > 100 TFLOPS** and **decode > 300 GB/s**
of weight bandwidth. Whether the prefill goal is reachable at all is decided by
the *routing*, not by the kernel.

### The weights are not the only stream

Earlier rounds of this document modelled the prefill roofline on the weights
alone: a W4A8 grouped GEMM reads every active expert's int8 weights exactly once
and does `2 × rows_per_expert` FLOPs per weight byte, so

```
arithmetic intensity = 2 × rows_per_expert            [FLOP / byte]
TFLOPS              <= 2 × rows_per_expert × weight_bandwidth
rows_per_expert      = batch × top_k / active_experts
```

and the `N`/`K` factors cancel. That is correct about the weight stream and
wrong about the total. One `moe_gemm_w4a8` call moves five streams, not one
(`T = batch × top_k` routed rows):

| Stream | Bytes | Scales with |
|---|---|---|
| activations, read by the quantizer | `T × K × sizeof(act)` | `T` |
| the int8 copy, written | `T × K` | `T` |
| the int8 copy, read back by the GEMM | `T × K` | `T` |
| every active expert's weights | `E_active × N × K` | `E_active` |
| the output | `T × N × sizeof(out)` | `T` |

Only the fourth line is what `W GB/s` reports and what the old model counted.
Because it is the only one that does *not* grow with the token count, it
dominates at small batches — where the old formula is very nearly exact — and
becomes a *minority* of the traffic exactly in the compute-bound regime the
target is measured in. At 256 rows per expert (the batch this harness used to
measure at):

| Shape | Weights | Total traffic | Weights' share | BW for 100 TFLOPS (old model) | Ceiling at 400 GB/s |
|---|---|---|---|---|---|
| qwen3 up (N=1536, K=2048) | 403 MB | 772 MB | 52% | **374** GB/s (195) | 107 TFLOPS |
| qwen3 down (N=2048, K=768) | 201 MB | 436 MB | 46% | **423** GB/s (195) | **94 TFLOPS** |
| minimax up (N=1536, K=3072) | 906 MB | 1661 MB | 55% | **358** GB/s (195) | 112 TFLOPS |
| minimax down (N=3072, K=1536) | 906 MB | 1510 MB | 60% | **326** GB/s (195) | 123 TFLOPS |

The old model printed 195 GB/s for all four. The real requirement is 1.7×–2.2×
that — and for the qwen3 down-projection it is **past what the device
delivers**: 423 GB/s against a 456 GB/s pin rate that probes at ~400 GB/s. Its
ceiling at that routing is 94 TFLOPS, so **that shape could not reach 100 TFLOPS
at the batch it was being measured at, whatever the kernel did**. It is also the
shape that has read furthest from target in every sweep (50–56 TFLOPS), which
is not a coincidence: smallest K means the largest non-weight share.

So `_PREFILL_TARGET_ROWS_PER_EXPERT` moved from 256 to **384**, the smallest
round routing whose ceiling clears 100 TFLOPS on all four shapes (112 / 130 /
137 / 154 TFLOPS at a 400 GB/s probe; qwen3 down alone needs ≥ 290 rows per
expert). In model tokens that is 6144 for Qwen3-MoE and 9216 for MiniMax.

The small-batch conclusions are unchanged, because the weight term dominates
there:

| Model tokens | Routed rows | rows/expert | Bandwidth needed for 100 TFLOPS |
|---|---|---|---|
| 128 (default prefill batch) | 1024 | 8 | ~6300 GB/s |
| 512 | 4096 | 32 | ~1600 GB/s |
| 2048 | 16384 | 128 | ~440 GB/s |
| 6144 (`test_perf_prefill_compute_bound`) | 49152 | 384 | ~310 GB/s |

~4.5 TFLOPS at the default batch is **not** a kernel deficiency: at 8 rows per
expert and the ~285 GB/s the kernel achieves, the ceiling is
`2 × 8 × 285e9 = 4.56 TFLOPS` — the measured value, i.e. the kernel is already
running at the DRAM roofline. Reaching 100 TFLOPS there would require more than
6 TB/s, over 13× the B60's 456 GB/s.

The perf table therefore prints `rows/E`, `DRAM GB/s` (all five streams) and
`BW@100T` next to the measured numbers, and each sweep ends with a verdict
block:

```
targets [prefill]: prefill compute > 100 TFLOPS
  device copy bandwidth probe: 400 GB/s
  qwen3 up     tokens=1024   rows/E=8.0        4.56 TFLOPS vs 100 -> N/A (bandwidth bound: ...)
  qwen3 down   tokens=49152  rows/E=384.0    102.40 TFLOPS vs 100 -> PASS (92% of the 112 TFLOPS bandwidth ceiling)
```

A row is reported `N/A` rather than `FAIL` when the device bandwidth probe (one
large device-to-device copy, measured once per run) shows the target is
unreachable at that routing; a reachable row also prints how much of its ceiling
it reaches, which is the part a kernel change can move. The verdict is
informational by default; pass `--enforce-targets` to turn it into a hard
assertion.

### The 8K-prompt point: a prompt is not a rows/expert target

`_PREFILL_TARGET_ROWS_PER_EXPERT` is *derived* per model, so both shape groups
land on the same 384 rows per expert (6144 model tokens for Qwen3-MoE, 9216 for
MiniMax). A real prefill does the opposite: the prompt length is fixed and the
expert count divides it. `test_perf_prefill_long_seq` therefore runs a single
**8K-token prompt** — 8192 model tokens, 65536 routed rows, the same 8K group
`test_moe_prefill_perf.py` sweeps — and the two models land in *different*
regimes:

| Shape | rows/expert | Weights | Total traffic | BW for 100 TFLOPS | Ceiling at 400 GB/s | Ceiling at 384 rows/E |
|---|---|---|---|---|---|---|
| qwen3 up (N=1536, K=2048) | 512 | 403 MB | 1141 MB | 277 GB/s | **145 TFLOPS** | 129 |
| qwen3 down (N=2048, K=768) | 512 | 201 MB | 671 MB | 326 GB/s | **123 TFLOPS** | 112 |
| minimax up (N=1536, K=3072) | 341 | 906 MB | 1913 MB | 309 GB/s | **129 TFLOPS** | 137 |
| minimax down (N=3072, K=1536) | 341 | 906 MB | 1711 MB | 277 GB/s | **145 TFLOPS** | 154 |

For Qwen3-MoE's 128 experts an 8K prompt is 512 rows each, a third more than the
compute-bound batch, so its ceiling rises by 10–12% and the 100 TFLOPS target
gains margin — this is the point where the suite's highest prefill TFLOPS should
be. For MiniMax's 192 experts the same prompt is only 341 rows each, *below* the
compute-bound batch, so its ceiling falls by 6%. The same kernel therefore reads
faster on one model and slower on the other at the same prompt length, which is
the reason to measure both: throughput follows the routing, not the sequence
length.

512 rows per expert also moves the **tile ladder**. The 256-row tile is gated on
padding no worse than the 128-row one (`⌈M/256⌉·256 == ⌈M/128⌉·128`) — false at
384, true at 512 — so the Qwen3-MoE shapes take the `256x256` rung here and
nowhere else in the suite. `test_perf_prefill_tile_sweep_long_seq` re-runs the
tile sweep at this routing for exactly that reason: the rung the ladder picks at
an 8K prompt has never been measured against its alternatives at a routing where
it does not pad (see [Prefill tile](#prefill-tile)).

### Why `vs w4a16` is below 1.0 at small batches

The same intensity argument explains the `vs w4a16` column. W4A8 streams **2×
the weight bytes** of the int4 path (one byte vs. half a byte per element) in
exchange for ~2× the DPAS peak, so it only wins once the GEMM is compute bound:

```
crossover rows/expert ~= int8_peak_TOPS / (4 × weight_bandwidth)
```

With the B60's ~197 TOPS of int8 DPAS and the ~285 GB/s the kernel streams, that
is ~173 rows per expert (~2760 model tokens) — essentially the same routing at
which the bandwidth roofline first admits 100 TFLOPS (~176 rows above), so on
this part the two crossings coincide. Decode (1 row per expert) and small-batch
prefill are far below it, so readings of 0.55–0.71× are expected there: W4A8 is
a large-batch prefill optimization, and at decode it can only help by improving
the *memory* path.

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
activation scale. Only the float *summation order* differs (per lane then
across lanes, instead of one lane folding every block), so the two mappings are
not bit-identical; `test_decode_ksplit_matches_legacy` asserts they agree to
better than 40 dB SNR / 0.9999 cosine, far tighter than any real mapping bug
could pass.

The mapping requires `N % 16 == 0`, a re-scale block that is a multiple of 16
and at least 256, and `K % block == 0`. Anything else (for example an explicit
`--rescale-group-size 64`) falls back to the original kernel automatically.

Decode also issues **one kernel launch fewer per step**: each token's expert id
is derived inside the activation-quantization kernel — which already runs one
sub-group per token — instead of by a separate `fill_expert_id_per_token`
launch. At batch 1 the entire GEMV takes ~45 µs, so a saved launch is not noise.

## Prefill: message width and register pressure

Two costs sit next to the grouped GEMM at prefill sizes, and both were paid in
full before this change.

**Activation quantization.** Converting the routed activations to int8 is a pure
streaming pass — read `[T, K]` twice (absmax, then quantize), write `[T, K]`
int8. At 32768 routed rows and `K = 2048` that is ~200 MB, next to the ~400 MB
the qwen3 up-proj GEMM streams for weights, so it is a real share of the call
rather than a preamble. The original mapping (`k = lane; k += SG_SIZE`) moved it
in the *narrowest* messages a sub-group can issue: 16 lanes × one 16-bit element
is a 32-byte load, and 16 lanes × one int8 is a **16-byte store** — a quarter of
a cache line per store message. That is exactly the defect the decode GEMV had
before the K-split rewrite, where fixing it was worth 1.09–1.93×.

Each lane now owns `VEC` *consecutive* elements, so one message covers
`SG_SIZE × VEC` contiguous elements: 256 bytes of activations and 128 bytes of
int8 at `VEC = 8`. `VEC` is picked from K — 8 when `K % 128 == 0` (every shipped
MoE shape: 768 / 1536 / 2048 / 3072), otherwise 4, which the `K % 64 == 0` shape
gate always satisfies — and a misaligned base pointer falls back to the scalar
kernel. Nothing that rounds is reordered: the per-lane partial reduction is
`fmax`, which is exact and order-independent, so both mappings feed the
sub-group reduce the same absmax and quantize every element identically.
`test_act_quant_vec_matches_scalar` asserts the two are **bit-identical**, and
`ARK_MOE_W4A8_ACT_QUANT_VEC=0` restores the scalar mapping for A/B measurement.

**Loads in flight.** Widening the messages fixed how many bytes each *request*
moves; it did not change how many requests a work-item has outstanding. The
pass walks K with a runtime trip count (`steps = K / (SG_SIZE × VEC)`) and folds
every vector into the same `local_max`, so the loop reads as: issue one load,
stall until it returns, `fmax` it, repeat. Xe cores execute in order and `fmax`
is not reassociated without fast-math, so a thread keeps roughly *one* 256-byte
load in flight. That is a Little's-law problem, not a bandwidth one: 1280
concurrent sub-groups (the B60's occupancy ceiling — 160 XVEs × 8 thread slots)
× 256 bytes is ~320 KB of in-flight reads, under the ~456 KB a 456 GB/s device
needs to stay busy across a ~1 µs memory latency, and a real launch rarely fills
every slot. It is the same argument that made the decode GEMV load two chunks
per iteration.

Each iteration now loads `UNROLL` *independent* vectors before consuming any of
them, and reduces them into `UNROLL` separate partial maxima so the loads do not
serialize behind the accumulator chain either; the quantize pass batches its
loads the same way. At the default `UNROLL = 4` a thread holds 1 KB, which
clears the 456 KB well before every slot is occupied. `steps % UNROLL` vectors
are left to a tail loop — at `K = 768` a lane walks 6 vectors, so with the
default `UNROLL = 4` the tail is
real code rather than a formality. Nothing that rounds changes (`fmax` is exact
and order-independent, so the partial maxima merge to the same bits), and
`UNROLL = 1` is the previous kernel instruction for instruction, so
`ARK_MOE_W4A8_ACT_QUANT_UNROLL=1` is an exact A/B baseline.
`test_act_quant_unroll_matches` asserts bit-identity at both a K that divides
the unroll depth and one that leaves a tail.

**Reading the row once.** Batching the loads did not change how many there are.
The absmax has to see the whole row before the first element can be quantized,
so the pass reads `[T, K]`, reduces, then reads `[T, K]` again. The re-read is
L2-resident while the row is still there, but the rows a work-group quantizes
second evict the ones it quantized first well before the pass ends — at 8 MB of
L2 and 4 KB per bf16 row of `K = 2048`, only ~2000 rows fit *if nothing else is
resident*, and the GEMM's weights compete for the same cache immediately after.

A row is small enough to keep in registers instead: a lane owns `K / 16`
elements, so `K = 2048` is 256 bytes — 64 of the 128 dwords per lane the
quantizer gets (it launches without `grf_size<256>`, unlike the GEMM). The
single-pass kernel loads the row once, reduces it, and quantizes out of the
registers; the second read disappears, and every load is issued before any is
consumed, which subsumes what `UNROLL` was doing rather than competing with it.

`MAX_STEPS` is the compile-time cap that makes the fragment a register array
rather than scratch: the loop is `#pragma unroll` over `MAX_STEPS` with an
`if (s < steps)` guard, so every index is constant and SROA can promote it. Two
rungs are instantiated — 8 vectors (`K ≤ 1024` at `VEC = 8`, 32 dwords) and 16
(`K ≤ 2048`, 64 dwords) — and anything longer keeps the two-pass kernel, which
is why minimax's `K = 3072` up-projection still takes the old path. The partial
maxima stay at four accumulators, so the reduction is unchanged in both cost and
value.

This is a register-pressure gamble: if 64 dwords of row plus addressing spills,
the pass gets *slower*. `ARK_MOE_W4A8_ACT_QUANT_SINGLE_PASS=0` restores the
two-pass kernel exactly, `test_perf_prefill_act_quant_single_pass_sweep` times
the pair, and `test_act_quant_single_pass_matches` asserts they agree bit for
bit at both a K that fills a rung exactly (2048) and one that does not (768).

**The GEMM epilogue.** The mainloop kept two C fragments live: the int32 DPAS
accumulator, cleared once per AUTO_S8 re-scale block, and a float shadow that
had to survive across blocks because each block's weight scale is applied before
the next block overwrites the accumulator. A lane holds both in GRF for the
*entire* mainloop:

| tile | sub-group C fragment | int32 regs/lane | + float regs/lane |
|---|---|---|---|
| `128x128` | 32 × 32 | 64 | 64 |
| `128x256` | 32 × 64 | 128 | 128 |

With `grf_size<256>` a lane has 256 registers, so at `128x128` the float shadow
alone reserved a quarter of the register file for the whole mainloop, and at
`128x256` the two fragments together *are* the register file — leaving nothing
for the staged A/B tiles. That was the 35–50% penalty the 256-wide tiles used to
pay in [Tuned defaults](#tuned-defaults-measured); with the shadow gone — and
with the epilogue's scalar store gone after it — the 256-wide tiles are now the
ladder's default choice. At the default re-scale block the shadow was pure
overhead: `blks == 1` (the AUTO_S8 `group=-1` default) has nothing to
carry across blocks, so the scale can be folded on the way out instead. That
path now runs without the float fragment, and applies
`scale_b[col] × scale_a[row]` in a single pass — the same shape as the
`AccumBlock == false` branch of the reference dense int8 GEMM.

The same epilogue also stopped branching around out-of-range elements. A grouped
GEMM's per-expert M is arbitrary, so tiles at the M edge are partial and the
*store* must stay predicated — but the scale *loads* need not be: their indices
are clamped into range instead, which makes both reads unconditional loads at a
compile-time offset from a uniform base. That is what lets the compiler collapse
the per-element reads into the handful of distinct addresses a sub-group's
fragment actually covers (all lanes of a row group share `scale_a[row]`, and a
lane repeats the same `scale_b[col]` for every row it owns); under the previous
`continue` guard each read sat in its own basic block and none of it could be
hoisted.

**Interior tiles.** Both of those guards — the store predicate and the two
clamps — are needed only where a tile hangs off the edge of the expert's rows or
of N, and whether it does is uniform across the work-group: `m`, `n` and the
tile coordinates are all kernel-uniform. Testing it once per tile instead of
once per fragment element removes ~4 instructions from every output element of
an interior tile, and at `K = 768` (12 k-tiles per tile) the epilogue is a real
share of the tile's time. The guarded path stays for edge tiles and behind
`ARK_MOE_W4A8_PREFILL_FULL_TILE=0`; the arithmetic and its order are untouched,
so the two are bit-identical and
`test_full_tile_epilogue_matches_predicated` asserts exactly that at a batch
that gives every expert one interior tile and one ragged one.

**The store itself.** Removing instructions from around the store left the
store. The Xe DPAS C fragment gives a lane one *column* of each 8×16 atom, so
the 16 lanes of a sub-group hold 16 *consecutive columns of one row*: a scalar
`c[row * n + col] = ...` is a 32-byte message for 16-bit `ElementD` — half a
cache line — and a 32×32 sub-group fragment issues **64** of them. The same
bytes go out in a handful of messages through the hardware 2D block store, which
is what every sibling prefill kernel already uses for D
(`sycl_tla_moe_prefill_{fp8,int,s4}_dpas.hpp`) and what the dense GEMM in
`sycl_tla_dense_gemm.hpp` uses on this exact accumulator shape.

D is why this is worth doing at prefill sizes rather than as a tidy-up: at 384
rows per expert the qwen3 down-projection writes 1.5 MB of fp16 per expert —
exactly as many bytes as the int8 weights it reads, because N (2048) is larger
than K (768) there, and over a third of that expert's traffic — and it is the
same shape whose mainloop is shortest, so it pays the epilogue twice.

The port follows `dense_gemm_detail::gemm_device_impl` rather than the sibling
MoE kernels. Those `reorder(tCrC, tCrC_out)` from the MMA fragment into an
explicitly chosen `XE_STORE_2D` atom's fragment, and `reorder` moves *registers*:
free with a `float` accumulator, but this kernel accumulates in `int32`
(`FrgTypeC` of `XE_DPAS_TT<8, int32_t, int8_t, int8_t>`) and has to scale and
numerically convert first, which `reorder` does not do. `make_block_2d_copy_D`
derives its layout from the MMA's own C partition, so the scaled `ElementD`
fragment — `make_tensor_like<ElementD>(tCrC)`, filled through the same `tCgC(i)`
coordinates the scalar path indexes with — goes straight to
`copy(copy_d, tCrD, tCgC)` with no `reorder` in between.

It also *removes* the store predicate rather than skipping it: the 2D block
message clips to the surface described by the D tensor, so a partial tile at the
M edge drops its out-of-range rows in hardware, exactly as the sibling grouped
GEMMs rely on for their ragged experts. Only the scale loads still clamp, and
only on edge tiles.

The descriptor wants a 64-byte aligned base and a row pitch that is a multiple
of 16 bytes. D's per-expert base is `outputs + pre_rows × N` for a
routing-dependent `pre_rows`, so the dispatcher gates on
`N × sizeof(ElementD) % 64 == 0` — which makes *every* expert's base aligned
given an aligned tensor, and covers the pitch too — plus the tensor base itself.
Every shipped N (1536 / 2048 / 3072 with 16-bit D) clears it; anything else keeps
the scalar store. `ARK_MOE_W4A8_PREFILL_STORE_2D=0` also keeps it, for A/B
measurement, and `test_prefill_2d_store_matches_scalar` asserts the two write
identical bits at a batch that gives every expert one interior and one ragged
tile — the case where a store that did *not* clip would corrupt the next
expert's rows.

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
| `DRAM GB/s` | *All* the traffic the call moves: the fp16 activations read, the int8 copy written and read back, the expert weights, and the output — see [The weights are not the only stream](#the-weights-are-not-the-only-stream). This is the number to compare against the device's 456 GB/s |
| `BW@100T` | DRAM bandwidth (all five streams) this shape would need to reach 100 TFLOPS. When it exceeds what the device can stream, `TFLOPS` is capped by memory and no kernel change can hit the target at that shape |
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
`test_perf_prefill_compute_bound` adds a single batch sized so every expert
gets 384 rows (6144 model tokens for Qwen3-MoE) — the smallest round sweep point
where the 100 TFLOPS goal is under the device's bandwidth ceiling on *every*
shipped shape, counting [all five streams](#the-weights-are-not-the-only-stream)
and not just the weights. `test_perf_prefill_long_seq` adds the other kind of
prefill point: a fixed **8K-token prompt** (8192 model tokens, 65536 routed
rows), which is 512 rows per expert for Qwen3-MoE and 341 for MiniMax — see
[The 8K-prompt point](#the-8k-prompt-point-a-prompt-is-not-a-rowsexpert-target).

A second shape group covers MiniMax-M2, matching `test_moe_prefill_perf.py`:

```
hidden_size = 3072,  intermediate_size = 1536
num_local_experts = 192,  num_experts_per_tok = 8

minimax up    :  N = 1536,  K = 3072
minimax down  :  N = 3072,  K = 1536
```

It matters because both targets are shape dependent: 192 experts spread a given
batch over 1.5× more experts (fewer rows per expert, so a *lower* compute
ceiling at the same batch), while the longer K gives the decode GEMV a longer
sequential stream and the prefill tile more K per tile-load. The compute-bound
batch is derived per model, so MiniMax runs 9216 model tokens for the same 384
rows per expert. Shape groups are selected with `--models`
(`qwen3` — the default —, `minimax`, a comma-separated list, or `all`); the
heavy-tailed empirical routing for MiniMax lives in `test_moe_prefill_perf.py`.

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

# The compute-bound prefill case (6144 model tokens), where the TFLOPS goal is reachable
pytest -v -s test_moe_w4a8_perf.py -k compute_bound

# The 8K-prompt prefill case (8192 model tokens) and its tile sweep
pytest -v -s test_moe_w4a8_perf.py -k long_seq

# Make the performance goals hard assertions instead of a printed verdict
pytest -v -s test_moe_w4a8_perf.py -k perf --enforce-targets

# Add the MiniMax shapes (or --models all for both groups)
pytest -v -s test_moe_w4a8_perf.py -k perf --models minimax

# Sweep the kernel dispatch configurations and print the fastest equivalent one
pytest -v -s test_moe_w4a8_perf.py -k sweep
```

`test_perf_decode_config_sweep`, `test_perf_prefill_tile_sweep`,
`test_perf_prefill_act_quant_sweep`, `test_perf_prefill_act_quant_unroll_sweep`
and `test_perf_prefill_epilogue_sweep` build one workload, prepack it once, then
time every dispatch configuration against it — the decode lane mapping (legacy
GEMV plus every `CH` × `NCOLS` combination), the prefill work-group tile, the
activation-quantization message width and loads-in-flight depth, and the
epilogue's edge guard. Each configuration is checked for numerical equivalence
with the first one, and the table is followed by a `best configuration` block
naming the winning environment variables per shape, so the tuning knobs can be
settled in a single on-hardware run.

`test_perf_prefill_tile_sweep_long_seq` is the same tile sweep at the 8K-prompt
routing, where the ladder's rung differs (512 rows per expert for Qwen3-MoE
instead of 384); a shape swept at more than one batch gets one `best
configuration` line per batch, because the winner is a property of the routing
as much as of the shape.

The `-s` flag is required to see the printed tables.

### As a standalone script (no pytest)

```bash
python test_moe_w4a8_perf.py                       # both phases, smallest batch
python test_moe_w4a8_perf.py --all-shapes          # full sweep
python test_moe_w4a8_perf.py --phase decode        # decode only
python test_moe_w4a8_perf.py --skip-accuracy       # perf only
python test_moe_w4a8_perf.py --compute-bound       # add the 6144-token prefill case
python test_moe_w4a8_perf.py --long-seq            # add the 8K-prompt prefill case
python test_moe_w4a8_perf.py --dtype fp16          # fp16 activations
python test_moe_w4a8_perf.py --rescale-group-size 256
python test_moe_w4a8_perf.py --warmup 10 --iters 100
```

`--long-seq` also repeats the prefill tile sweep at the 8K prompt when combined
with `--sweep-configs`.

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

On the 24 GB B60 that trade has a hard limit: the two GEMMs above are ~0.6 GB of
int8 per MoE layer on top of the ~0.3 GB of int4 they pin, so a 48-layer
Qwen3-MoE stack would ask for ~29 GB of prepacked weights and does not fit.
Caching the whole model is a multi-card or larger-VRAM configuration; on one
B60, cache the layers that are prefill-bound and leave the rest on
`cache_prepack=False`.

## Tuned defaults (measured)

Every default below comes from `-k sweep` runs on the Arc Pro B60 above (bf16
activations, 8 routed rows for decode, **384 rows/expert** for prefill — the
compute-bound batch the suite uses). Each configuration is checked for numerical
equivalence with the first one before it is timed.

Nothing in the prefill path is unmeasured any more: the single-pass quantizer
and the 2D block store, which used to be on by reasoning alone, both have their
own tables below.

**Reading the tables — the noise floor is ~4%.** The unroll sweep times three
shapes whose K puts them on the single-pass quantizer, where `UNROLL` is dead
code, so those rows are three sets of *identical* kernels: they spread 0.4%,
2.1% and 3.9%. Treat anything below ~4% as run-to-run variation.

### Prefill tile

| shape | `128x128` | `128x256` | `256x128` | `256x256` |
|---|---|---|---|---|
| qwen3 up | **3.457 ms** | 3.507 ms | 4.507 ms | 3.952 ms |
| qwen3 down | 2.512 ms | **2.488 ms** | 2.777 ms | 2.696 ms |
| minimax up | **6.940 ms** | 6.978 ms | 8.895 ms | 8.111 ms |
| minimax down | 7.255 ms | **6.710 ms** | 9.166 ms | 7.779 ms |

**M is about padding, not registers.** An expert launches `ceil(M / TileM)`
*full* tiles, so at 384 rows/expert the 256-row tile schedules 512 rows for 384
rows of data — a third of the MACs spent on nothing — against exactly three
128-row tiles. That is the whole 1.11–1.30× deficit in the `256x*` columns: on
the two long-K shapes, where the mainloop dominates, the measured ratio (1.30×
qwen3 up, 1.28× minimax up) *is* the padding ratio 512/384 = 1.33 to within
noise. It is not an argument against `TileM = 256` — the earlier run at 256
rows/expert, where the tile divides the rows exactly, had it 1.3–3.9% ahead on
all four shapes.

So the rung is gated on padding rather than on a row count: take the 256-row
tile only where `ceil(M/256)·256 == ceil(M/128)·128`, i.e. where it launches no
more rows than the 128-row tile would. True at 256, 400, 512, 2048; false at
384, which is exactly the batch that used to fall off the cliff.

**N is now free.** The 256-wide tile is ahead or level everywhere the table can
compare it: at `TileM = 128` it takes minimax down by 1.08× and ties the other
three (within 1.4% either way), and at `TileM = 256` it takes all four, by
2.9–15.1%. The 35–50% cliff the first sweep saw on 256-wide N tiles was the
float C shadow (see
[Prefill: message width and register pressure](#prefill-message-width-and-register-pressure));
what was left of it in the second sweep — 0–8% behind on three shapes — was
measured with the *scalar* epilogue store, and it disappears now that a 32×64
fragment goes out in a handful of block messages instead of 128 scalar ones. So
the ladder is 256 wide wherever `N % 256 == 0` (every shipped N: 1536 / 2048 /
3072), and 128 wide otherwise, where the wider tile would only pad.

The ladder therefore is: `< 16` rows/expert → `8x128`, `< 128` → `64x128`, then
`256x256` / `256x128` when the 256-row tile costs no extra padding and
`128x256` / `128x128` when it does — the N choice being `N % 256 == 0` in both
cases. All six policies stay compiled and selectable with
`ARK_MOE_W4A8_PREFILL_TILE`.

At the ladder's choice the four swept shapes land on 3.507 / 2.488 / 6.978 /
6.710 ms — 88.2 / 62.1 / 99.7 / 103.7 TFLOPS, against 69.9 / 52.7 / 76.9 / 75.5
for the ladder as it was (1.26× / 1.18× / 1.30× / 1.37×), and within 1.4% of the
fastest tile on every shape. Three of the four now clear the 100 TFLOPS target
or sit within 1%; qwen3 down is still the outlier at 62 TFLOPS, because at
`K = 768` a tile runs only 12 k-tiles and the epilogue and prologue are a large
share of it.

One caveat that has not changed: the ladder compares `total_tokens / E`, the
*average* rows/expert, so a skewed routing that averages 384 can still leave
individual experts with very different tile counts.

The table above is measured at 384 rows/expert, where the padding gate keeps the
256-row tile *out*, so the `256x*` columns there are the cost of padding rather
than a verdict on the tile. The routing that actually selects it is the 8K
prompt (512 rows/expert on Qwen3-MoE, an exact multiple of 256), and
`test_perf_prefill_tile_sweep_long_seq` is the sweep that measures the rung the
ladder picks there against its alternatives — the one open question left in the
tile ladder, since the only prior evidence for `TileM = 256` is the older run at
256 rows/expert (1.3–3.9% ahead on all four shapes).

### Prefill activation quantization

| shape | scalar | vectorized (default) | speedup |
|---|---|---|---|
| qwen3 up | 4.806 ms | **4.574 ms** | 1.05× |
| qwen3 down | 3.120 ms | **2.769 ms** | 1.13× |
| minimax up | 9.959 ms | **8.981 ms** | 1.11× |
| minimax down | 9.581 ms | **9.221 ms** | 1.04× |

Quantizing the routed activations is a streaming pass over `[T, K]` next to a
GEMM that already moves ~400 MB, and it is worth 4–13% of the whole call purely
by issuing 256-byte loads and 128-byte stores instead of 32-byte and 16-byte
ones. `ARK_MOE_W4A8_ACT_QUANT_VEC=0` restores the scalar mapping.

How many of those wide loads a work-item keeps *outstanding* is the separate
knob `ARK_MOE_W4A8_ACT_QUANT_UNROLL` (1, 2, or 4 = default). Only minimax up is
a real A/B for it — the other three shapes take the single-pass kernel below,
where `UNROLL` is dead code — and there the default wins: **8.959 ms** at 4,
8.967 ms at 2, 9.139 ms at 1.

### Prefill single-pass activation quantization

| shape | K | two-pass | single-pass (default) | speedup |
|---|---|---|---|---|
| qwen3 up | 2048 | 4.455 ms | **4.416 ms** | 1.01× |
| qwen3 down | 768 | 2.836 ms | **2.694 ms** | 1.05× |
| minimax down | 1536 | **9.232 ms** | 9.244 ms | 1.00× |
| minimax up | 3072 | 8.991 ms | 8.962 ms | — (not eligible) |

This was the one change with real downside risk: the row is held in registers
between the absmax and the quantize pass, and a spill would have made the pass
slower rather than faster. It does not spill. minimax up's `K = 3072` is past
the 16-vector rung, so both of its rows run the same two-pass kernel and their
0.3% gap is noise.

### Prefill store

| shape | scalar store | 2D block store (default) | speedup |
|---|---|---|---|
| qwen3 up | 5.111 ms | **4.388 ms** | 1.16× |
| qwen3 down | 3.603 ms | **2.669 ms** | 1.35× |
| minimax up | 9.604 ms | **8.570 ms** | 1.12× |
| minimax down | 10.837 ms | **9.048 ms** | 1.20× |

The largest single prefill win of the set, and it is the epilogue rather than
the mainloop: a 32×32 sub-group fragment goes out in a handful of block messages
instead of 64 half-cache-line scalar ones. The ordering follows the argument —
qwen3 down, whose 12-k-tile mainloop amortizes the epilogue least and whose D is
as large as its weights, gains the most.

### Prefill epilogue guard

| shape | guarded | interior-tile (default) | speedup |
|---|---|---|---|
| qwen3 up | **4.428 ms** | 4.466 ms | 0.99× |
| qwen3 down | 2.840 ms | **2.625 ms** | 1.08× |
| minimax up | 8.971 ms | **8.794 ms** | 1.02× |
| minimax down | **8.990 ms** | 9.020 ms | 1.00× |

The mainloop is identical in both columns; only the store differs, so this is
the cost of ~4 instructions per output element. It is largest exactly where the
mainloop is shortest — qwen3 down runs 12 k-tiles per tile at `K = 768` — which
is the shape the ordering argument predicted, and the two rows that come out
behind do so by 0.9% and 0.3%, inside the noise floor.
`ARK_MOE_W4A8_PREFILL_FULL_TILE=0` restores the guarded epilogue; the two are
bit-identical.

### Decode chunk width and column blocking

| shape | fastest equivalent config | default (`CH=16`, `NCOLS=2`) | `CH=32`, same `NCOLS` |
|---|---|---|---|
| qwen3 up | ch16 ncols2 — **284.0 GB/s** | 284.0 GB/s | 278.9 GB/s |
| qwen3 down | ch16 ncols4 — **285.7 GB/s** | 280.1 GB/s | 244.4 GB/s |
| minimax up | ch16 ncols1 — **271.0 GB/s** | 268.1 GB/s | 259.9 GB/s |
| minimax down | ch16 ncols2 — **315.5 GB/s** | 315.5 GB/s | 308.7 GB/s |

`CH = 32` never wins and costs up to 13%, so `16` stays the default. `NCOLS = 2`
is the fastest configuration on two of the four shapes and within 2% of the best
on the other two, while `1` loses 47% on qwen3 up and `4` loses 14% on minimax
up, so it stays the default as well. At those defaults the K-split mapping is
worth 1.09–1.93× over the legacy GEMV.

Those readings are 59–69% of the B60's 456 GB/s of pin bandwidth (68–79% of what
the device-copy probe actually reaches), so only minimax down clears the
300 GB/s target. A decode step reads one weight byte per multiply-add and
nothing else, so the remaining gap is message efficiency, not arithmetic.

## Environment variables

| Variable | Effect |
|---|---|
| `ARK_MOE_W4A8_AUTO_S8` | Override the AUTO_S8 re-scale block size. Unset / `-1` = one scale per output channel (fastest). Values that aren't a multiple of both `group_size` and 64, or that don't divide K, silently fall back to K. |
| `ARK_MOE_W4A8_DECODE_MAX_TOKENS` | Token count at or below which `phase="auto"` picks the GEMV (default `128`). |
| `ARK_MOE_W4A8_DECODE_KSPLIT` | Coalesced K-split decode mapping; **on by default**. Set to `0` to fall back to the original one-work-item-per-output GEMV (useful for A/B measurements). Ignored when the shape doesn't qualify. |
| `ARK_MOE_W4A8_DECODE_KSPLIT_NCOLS` | Output columns per sub-group in the K-split mapping: `1`, `2` (default) or `4`. Higher values amortize the activation loads over more columns but need `N % (16 × NCOLS) == 0`. `2` is the measured default, see [Tuned defaults](#tuned-defaults-measured). |
| `ARK_MOE_W4A8_DECODE_KSPLIT_CH` | K elements (= bytes) a lane loads per chunk: `16` (default) or `32`. `32` halves the number of memory messages and doubles the bytes a thread keeps in flight, at the cost of GRF; it needs a re-scale block of at least 512 and silently falls back to `16` otherwise. Measured slower than `16` on every swept shape, so it is a sweep point rather than a recommendation. |
| `ARK_MOE_W4A8_PREFILL_TILE` | Force a prefill work-group tile: `8x128`, `64x128`, `128x128`, `128x256`, `256x128`, `256x256`. Unset (default) uses the ladder: `< 16` rows/expert → `8x128`, `< 128` → `64x128`, then the 256-row tile when it pads no further than the 128-row one would (`⌈M/256⌉·256 == ⌈M/128⌉·128`) and the 128-row tile otherwise, each 256 wide in N when `N % 256 == 0` (see [Tuned defaults](#tuned-defaults-measured)). Forcing a tile the ladder would not pick costs up to 1.30× at 384 rows/expert. |
| `ARK_MOE_W4A8_ACT_QUANT_VEC` | Vectorized per-token activation quantization (each lane owns 4 or 8 consecutive K elements instead of striding by the sub-group width); **on by default**, worth 1.04–1.13× on the swept shapes. Set to `0` to force the scalar mapping for A/B measurement. Ignored when K or the buffer alignment doesn't qualify, in which case the scalar kernel runs anyway. |
| `ARK_MOE_W4A8_ACT_QUANT_UNROLL` | Vectors the activation quantizer loads before it consumes any of them: `1`, `2` or `4` (default, measured fastest). Higher values raise the bytes a work-item keeps in flight — the pass is latency-bound, not bandwidth-bound, at one outstanding load per thread — at the cost of GRF. `1` is the kernel as it was before the batching, so it is the A/B baseline; every value is bit-identical. Values outside `{1, 2, 4}` fall back to the default. Only applies to the vectorized *two-pass* mapping: the single-pass kernel below issues the whole row at once and ignores this. |
| `ARK_MOE_W4A8_ACT_QUANT_SINGLE_PASS` | Keep the activation row in registers between the absmax and the quantize pass instead of reading `[T, K]` twice; **on by default** where the row fits (`K ≤ 2048` at `VEC = 8`, 64 of the 128 dwords a lane gets), worth 1.00–1.05× on the shapes that qualify. Set to `0` to force the two-pass kernel, which is also what runs for longer rows. Bit-identical to it. |
| `ARK_MOE_W4A8_PREFILL_FULL_TILE` | Skip the epilogue's store predicate and scale-index clamps on tiles that touch neither the M nor the N edge; **on by default**, worth up to 1.08× on the swept shapes (and never more than 0.9% behind). The choice is uniform across the work-group, so it costs one comparison per tile instead of several per output element. Set to `0` to force the guarded epilogue everywhere (the two must be bit-identical). |
| `ARK_MOE_W4A8_PREFILL_STORE_2D` | Write D through the hardware 2D block store instead of one scalar 32-byte message per fragment element; **on by default** where the output is aligned (`N × sizeof(ElementD) % 64 == 0`, true for every shipped shape), and the largest single prefill win of the set at 1.12–1.35×. Set to `0` to force the scalar store, which is also what runs for shapes that miss the alignment gate. Bit-identical to it. |

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

The W4A8 kernel is a new SYCL/CuTe port, marked
`STATUS: PARTIALLY HARDWARE-VALIDATED` in
`auto_round_kernel/wrapper/include/sycl_tla_moe_w4a8.hpp`. Every perf sweep has
been run on an Intel Arc Pro B60, and **every** dispatch default now comes from
those runs — the tile ladder, the activation quantizer's message width, unroll
depth and single-pass rung, the interior-tile epilogue, the 2D block store, and
the decode `CH` / `NCOLS` (see [Tuned defaults](#tuned-defaults-measured)).
Every swept configuration also passed the cross-configuration equivalence check,
and all six bit-identity tests — `test_act_quant_vec_matches_scalar`,
`test_act_quant_unroll_matches`, `test_act_quant_single_pass_matches`,
`test_full_tile_epilogue_matches_predicated`,
`test_prefill_2d_store_matches_scalar` and `test_decode_ksplit_matches_legacy` —
pass on device, so each optimization is checked against its predecessor as well
as timed.

Still to run on device: the accuracy sweep against the fp32 reference, which
will catch layout/scale bugs immediately, and the two 8K-prompt prefill cases
(`test_perf_prefill_long_seq`, `test_perf_prefill_tile_sweep_long_seq`) — they
add no new kernel code, only a routing the suite did not measure at, but the
`256x256` rung the ladder takes there has not been timed against its
alternatives at a non-padding routing.

Three prefill changes used to be listed here as reasoned-through but unmeasured,
because the authoring environment has no XPU and no SYCL compiler. All three
have now been timed, and all three kept their default:

| Change | Revert with | Measured |
|---|---|---|
| Activation quantizer's batched loads — `UNROLL` vectors in flight instead of one | `ARK_MOE_W4A8_ACT_QUANT_UNROLL=1` | 1.02× at `UNROLL = 4` on the only shape that exercises it |
| Single-pass activation quantizer — the row stays in registers, so `[T, K]` is read once instead of twice | `ARK_MOE_W4A8_ACT_QUANT_SINGLE_PASS=0` | 1.00–1.05×; the register-resident row does not spill |
| 2D block store for D — a handful of block messages instead of 64 scalar 32-byte ones per sub-group fragment | `ARK_MOE_W4A8_PREFILL_STORE_2D=0` | 1.12–1.35×, the largest single prefill win |

The 2D store was previously listed as needing a device rather than a flag, on
the grounds that the sibling MoE kernels reach it through
`partition_sg_fragment_S` + `reorder` and no sibling 2D-stores a *scaled int32*
accumulator. That turned out to be the wrong reference: `reorder` moves
registers and does not convert, so it could never have carried an int32→fp16
epilogue. `sycl_tla_dense_gemm.hpp` — in the same translation unit — already
compiles the sequence that does (`make_block_2d_copy_D(mma, D)` +
`make_tensor_like<ElementD>(tCrC)` + `copy(copy_d, tCrD, tCgC)`, with a 32-bit
accumulator and a 16-bit output), so the port was a pure-C++ change after all.

The single-pass quantizer was the one with a real downside risk — a spilled
register-resident row makes the pass slower rather than faster — and the sweep
settled it in its favour on every shape that takes it.

What the sweeps should be read against has also changed. The prefill roofline in
this document used to count weight bytes only, which understated the bandwidth
these shapes need by 1.7–2.2× and made a 94-TFLOPS-ceiling shape look like a
kernel deficiency (see [the roofline](#the-weights-are-not-the-only-stream)).
With every stream counted, the four compute-bound shapes were running at 60–74%
of their true ceilings, and the compute-bound batch moved from 256 to 384 rows
per expert so that 100 TFLOPS is reachable on all of them. The remaining gap is
traffic, not arithmetic: the largest single win still on the table is fusing the
activation quantization into the GEMM's A-tile load, which would delete the int8
copy's write *and* read — 2 of the 5 streams, 14–22% of the traffic depending on
K — but that is a mainloop change and wants a device to develop against.

### Where the remaining prefill headroom is

At the ladder's choice the four compute-bound shapes read 88.2 / 62.1 / 99.7 /
103.7 TFLOPS, i.e. 56–73% of their bandwidth ceilings, so the headroom splits
into traffic the call still moves and ceiling the routing sets:

| Lead | What it would change | Where it shows |
|---|---|---|
| Fusing the activation quantization into the GEMM's A-tile load | Deletes 2 of the 5 streams (the int8 copy written, then read back) — 14% of the traffic at `K = 768`, 21% at `K = 2048`, 22% at `K = 3072` | Every shape; it is the largest single item left |
| Routing more rows per expert | Nothing in the kernel — it *raises* the ceiling, because the weight stream is the only one that does not grow with the token count | The 8K prompt is exactly this experiment for Qwen3-MoE: 512 rows/expert lifts the ceilings from 129 / 112 to 145 / 123 TFLOPS |
| The `256x256` rung the 8K prompt selects | Halves how often B is re-read per M tile, on top of the N-tile saving already taken | `test_perf_prefill_tile_sweep_long_seq`; unmeasured at a non-padding routing since the 256 rows/expert run |
| A single-pass activation quantizer for `K = 3072` | The second read of `[T, K]`, ~450 MB at the compute-bound batch | minimax up only; its row is 96 dwords per lane, past the 16-vector rung |

`qwen3 down` (`N = 2048, K = 768`) stays the outlier at ~62–69 TFLOPS: 12
k-tiles per tile is the shortest mainloop of the four, its output is as large as
its weights, and its ceiling is the lowest of the set at every routing. It is
also the shape the traffic-side leads above would help most.

