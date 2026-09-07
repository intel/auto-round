// SYCL MoE W4A8 -- INT4 weights / INT8 compute (prefill + decode)
//
// STATUS: PARTIALLY HARDWARE-VALIDATED -- every perf sweep in
// `test_moe_w4a8_perf.py` (`test_perf_prefill_tile_sweep`,
// `test_perf_prefill_tile_sweep_long_seq`, `test_perf_prefill_act_quant_-
// sweep`, `..._unroll_sweep`, `..._single_pass_sweep`,
// `test_perf_prefill_store_sweep`, `test_perf_prefill_epilogue_sweep`,
// `test_perf_decode_config_sweep`) and
// every cross-configuration equivalence test (`test_act_quant_vec_matches_-
// scalar`, `test_act_quant_unroll_matches`, `test_act_quant_single_pass_-
// matches`, `test_full_tile_epilogue_matches_predicated`,
// `test_prefill_2d_store_matches_scalar`, `test_decode_ksplit_matches_legacy`)
// has been run on an Intel Arc Pro B60
// (Battlemage, BMG-G21 -- 20 Xe2 cores / 160 XVEs at ~2.4 GHz, ~197 int8 TOPS,
// 24 GB GDDR6 at 456 GB/s), so both phases compile and run and **every**
// dispatch default -- tile ladder, activation-quant message width / unroll /
// single-pass, interior-tile epilogue, 2D block store, decode CH / NCOLS --
// comes from those measurements at the compute-bound batch (384 rows/expert),
// with the tile ladder measured at the 8K-prompt routing (512 / 341
// rows/expert) as well, and each configuration checked numerically against the
// others before it was
// timed. The accuracy gates against the fp32 reference still need a device
// run. The authoring environment has no XPU and no SYCL compiler, so anything
// added *since* that run follows the porting conventions of its siblings
// `sycl_tla_moe_prefill_int_dpas.hpp` /
// `sycl_tla_moe_prefill_fp8_dpas.hpp`; nothing is currently in that state.
// ---------------------------------------------------------------------------
//
// File layout
// -----------
// The W4A8 path is spread over four headers so that no translation unit pays
// for more than it uses:
//
//   sycl_tla_moe_w4a8_scratch.hpp  device scratch slabs; declarations only, so
//                                  that `utils.hpp` (and bestla's JIT headers
//                                  behind it) stay out of the light TUs
//   sycl_tla_moe_w4a8_helpers.hpp  declarations, host helpers, tile ladder
//                                  (cutlass-free; what the dispatcher sees)
//   sycl_tla_moe_w4a8_kernels.hpp  activation quant, AUTO_S8 prepack, decode
//                                  GEMV (cutlass-free, plain SYCL)
//   sycl_tla_moe_w4a8.hpp          this file: DPAS tile policies, the grouped
//                                  prefill GEMM and its launcher (needs CuTe)
//
// `sycl_tla_generation.cmake` then emits one translation unit per variant, so
// the twelve (dtype x tile) prefill instantiations compile separately rather
// than all landing in one 4.2 GB TU. The design notes below cover the path as
// a whole.
//
// What this file implements
// -------------------------
// A W4A8 MoE path: **weights are stored as int4** (auto-round's packed
// `[E, N, K/2]` nibble layout with `[E, N, K/group_size]` per-group scales),
// **the DPAS compute dtype is int8**, and **activations are dynamically
// quantized to int8** (per-token absmax) on the fly.
//
// It covers both MoE phases:
//   * prefill -- persistent grouped GEMM over experts, `XE_DPAS_TT<8, int32_t,
//     int8_t, int8_t>` (`s8 x s8 -> s32`), modelled on the W4A8 weight-only
//     GEMM in `sycl_tla_s8_gemm.hpp` (`sycl_tla_igemm_s8s8_dequant`) and the
//     grouped scheduler in `sycl_tla_moe_prefill_int_dpas.hpp`.
//   * decode -- int8 GEMV. The default mapping splits K across the sub-group
//     lanes (coalesced 256-byte weight reads, `NCOLS` output columns per
//     sub-group), mirroring `moe_decode_detail::launch_fp8_ksplit`;
//     `ARK_MOE_W4A8_DECODE_KSPLIT=0` restores the original one-lane-per-output
//     mapping modelled on `moe_decode_detail::launch_int8`.
//
// The AUTO_S8 re-scale trick
// --------------------------
// ARK's weight-only GEMM has an `AUTO_S8` option (`ARK_AUTO_S8` /
// `env_params::auto_s8`, see `xpu_wrapper.hpp`): rather than feeding the int8
// mainloop a per-K-group scale (which forces a partial-accumulator fold at
// every group boundary), it *re-scales* the int4 weights into int8 with a
// coarser block size -- typically `group=-1`, i.e. one scale per output
// channel spanning the whole K axis. The int8 GEMM then runs a single
// full-K int32 accumulation with one scalar multiply in the epilogue, which
// is the most efficient shape for DPAS.
//
// The conversion is exactly the one `packscale` + `unpackq(S8, ...)` perform
// in `xpu_wrapper.hpp`:
//
//     sxt[e][n][j] = max_{g in block j} |s[e][n][g]| * fullrange / 127
//     w8[e][n][k]  = round(w4[e][n][k] * s[e][n][k/group_size] / sxt[e][n][j])
//
// with `fullrange = 2^(bits-1) = 8` for int4. Because `|w4| <= 8` and
// `s <= sxt * 127 / 8` inside the block, `|w8| <= 127`: the re-scaled weight
// always fits in int8 without clipping, and the dequantized value
// `w8 * sxt` reproduces `w4 * s` up to the int8 rounding step.
//
// The block size is `rescale_group_size` (`-1` / `K` == per output channel ==
// the `group=-1` maximum-efficiency case). It can be overridden per-process
// with `ARK_MOE_W4A8_AUTO_S8` (`-1` or a multiple of both `group_size` and 64
// that divides K). Any invalid value falls back to per-channel.
//
// Because the conversion only depends on the checkpoint it is exposed as a
// separate one-shot entry point (`moe_w4a8_prepack`) so callers can run it at
// load time and keep the int8 weights + FP32 block scales resident, instead of
// paying for it on every forward.
//
// Numerics
// --------
//   out[t][n] = (Σ_j sxt[e][n][j] * Σ_{k in block j} qa[t][k] * w8[e][n][k])
//               * sa[t]
// with `qa = round(a / sa)`, `sa = max_k |a[t][k]| / 127`. The activation
// scale is per token (row), the weight scale is per (output channel, block),
// mirroring `sycl_tla_igemm_s8s8_dequant`'s `scale_a[row] * scale_b[col]`
// epilogue.
//
// Layout convention (identical to `moe_gemm_decode` / `moe_gemm_prefill`)
// ----------------------------------------------------------------------
//   activations : [total_tokens, K]  act dtype (tokens pre-sorted by expert)
//   weights_s4  : [E, N, K/2]        uint8, two nibbles per byte (sym)
//   scales      : [E, N, K/group_size] act dtype
//   weights_s8  : [E, N, K]          int8   (prepack output)
//   wscales     : [E, N, K/rescale_block] float (prepack output)
//   outputs     : [total_tokens, N]  act dtype
//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef ARK_XPU
#include <sycl/sycl.hpp>
#endif

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

// Pulls in the cutlass-sycl / CuTe include set, the `dpas_policy_base` policy
// root, `make_moe_tensor`, and `get_atomic_scratch_buffer` (via the FP8
// header it includes). This is the expensive include, so only the generated
// per-tile prefill TUs see this header at all -- the decode / quant / prepack
// TUs take `sycl_tla_moe_w4a8_kernels.hpp` and the dispatcher TU takes
// `sycl_tla_moe_w4a8_helpers.hpp`, neither of which reaches cutlass.
#include "sycl_tla_moe_prefill_int_dpas.hpp"
// The cutlass-free half: activation quant, AUTO_S8 prepack and decode GEMV,
// plus the scratch pools and parameter structs they share with the prefill.
#include "sycl_tla_moe_w4a8_kernels.hpp"

namespace ark {
namespace moe_w4a8 {

using namespace cute;

// Kernel name tag for the grouped prefill GEMM (the other tags live in
// `sycl_tla_moe_w4a8_kernels.hpp`).
template <class Policy, typename ElementD>
class MoEW4A8GemmName;

// ---------------------------------------------------------------------------
// Tile policies.
//
// `WGTile`'s K extent is 64 -- the int8 DPAS atom's K granularity, matching
// `sycl_tla_s8_gemm.hpp`'s `Shape<Int<TileM>, Int<TileN>, _64>`. The M/N
// shapes and sub-group layouts are the reference GEMM's tile ladder
// (`SmallTileSG` / `SmallMidTileSG` / `MediumTileSG` / `LargeTileSG`), which
// keeps `size(mma)` at 64 / 128 / 256 / 512 threads -- all divisors of the 512
// threads-per-SM budget the persistent scheduler assumes.
//
// Tile shape *is* the prefill bandwidth knob. A `TileM x TileN` tile reads its
// own A and B slabs, so the bytes a whole expert pulls through L2/DRAM are
//
//     M*K * ceil(N/TileN)  +  N*K * ceil(M/TileM)  ~=  M*N*K * (1/TileN + 1/TileM)
//
// i.e. A is re-read once per N tile and B once per M tile. Both halvings are
// real, and both are cancelled by *padding*: an expert launches
// `ceil(M/TileM) * ceil(N/TileN)` full tiles, so a `TileM` that does not
// divide the rows/expert pays for rows that do not exist.
//
// `test_perf_prefill_tile_sweep` on BMG at the compute-bound batch the suite
// runs (384 rows/expert, bf16 act), with the 2D block store and the
// single-pass activation quantizer in:
//
//   shape         auto     128x128   128x256   256x128   256x256
//   qwen3 up      3.540 ms  3.518 ms  3.585 ms  4.404 ms  3.970 ms
//   qwen3 down    2.472 ms  2.473 ms  2.432 ms  2.696 ms  2.547 ms
//   minimax up    6.976 ms  6.823 ms  6.878 ms  8.899 ms  8.019 ms
//   minimax down  6.749 ms  7.227 ms  6.874 ms  9.096 ms  7.823 ms
//
// and `test_perf_prefill_tile_sweep_long_seq`, the same sweep at one 8K prompt
// -- 512 rows/expert on qwen3 (128 experts), 341 on minimax (192):
//
//   shape         auto     128x128   128x256   256x128   256x256
//   qwen3 up      4.371 ms  4.382 ms  4.393 ms  4.468 ms  4.394 ms
//   qwen3 down    3.075 ms  3.030 ms  2.903 ms  3.025 ms  3.059 ms
//   minimax up    6.564 ms  6.673 ms  6.449 ms  9.057 ms  7.744 ms
//   minimax down  6.466 ms  6.725 ms  6.450 ms  9.373 ms  7.307 ms
//
// M: `TileM = 256` never pays. At 384 and 341 rows/expert it is 1.05-1.45x
// *behind*, and that part is arithmetic rather than a register effect: 384
// rows take `ceil(384/256) = 2` 256-row tiles -- 512 rows scheduled for 384
// rows of data, a third of the MACs spent on padding -- against exactly 3 full
// 128-row tiles, and the like-for-like ratio on the long-K shapes, where the
// mainloop dominates, is that padding ratio (512/384 = 1.33) to within noise
// (1.25x qwen3 up, 1.30x minimax up, both at `TileN = 128`).
// The 8K prompt is the case where that argument does *not* apply: 512 rows per
// expert is an exact multiple of 256, so both tiles schedule the same rows.
// The 256-row tile is still not ahead there. Like for like on `TileN` it reads
// -2.0% / 0.0% (qwen3 up at `TileN` 128 / 256) and +0.2% / -5.4% (qwen3 down),
// i.e. never better than a tie and 5.4% behind on the shape with the shortest
// mainloop -- so halving how often B is pulled per M tile buys nothing that the
// larger work-group (512 threads, one per Xe core) does not give back in
// scheduling granularity. The only reading ever in its favour is an older run
// at 256 rows/expert, 1.3-3.9% ahead, inside the noise floor. The ladder
// therefore stops taking it (see `moe_w4a8_prefill_launch`): it has no
// measured upside, and a routing skewed around the average the ladder sees
// puts individual experts back on the padding cliff.
//
// N: the 256-wide tile is ahead or level everywhere the tables can compare it
// -- at 384 rows/expert it takes minimax down by 1.05x and qwen3 down by 1.02x
// and is 0.8-1.9% behind on the other two, and at the 8K prompt it takes three
// of four by 3.5-4.4% and ties qwen3 up (0.3%). The 35-50% cliff the first
// sweep saw on every 256-wide N tile is gone -- it was the float C shadow the
// mainloop used to keep live (see `xe_gemm_w4a8`), which doubled the per-lane
// C footprint and made `TileN = 256` ask for the entire 256-register large-GRF
// file -- and what remained of it in the second sweep (0-8% behind on three
// shapes, measured with the *scalar* epilogue store) is gone too, now that a
// 32x64 fragment goes out in a handful of block messages instead of 128 scalar
// ones. So the ladder is 256 wide in N wherever N divides into it.
//
// Noise floor for reading all of this: the `auto` column is not an independent
// measurement -- it launches whichever explicit tile the ladder picks, so each
// row above contains one *duplicate* pair (`128x256` at 384 rows/expert and at
// the minimax 8K point, `256x256` at the qwen3 8K point). Across the eight
// pairs the two readings of the same kernel differ by 0.2-1.9%, which is the
// run-to-run floor for these tables; the padding effect reaches 45%.
//
// Every policy stays reachable through `ARK_MOE_W4A8_PREFILL_TILE` for a
// re-sweep on a device with a different register budget.
// ---------------------------------------------------------------------------
class w4a8_policy_m_8 : public moe_dpas_fp8::dpas_policy_base {
 public:
  using WGTile = Shape<_8, _128, _64>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_0, _1, _0>>;
};

class w4a8_policy_m_64 : public moe_dpas_fp8::dpas_policy_base {
 public:
  using WGTile = Shape<_64, _128, _64>;
  using SGLayout = Layout<Shape<_2, _4, _1>, Stride<_4, _1, _0>>;
};

class w4a8_policy_m_128 : public moe_dpas_fp8::dpas_policy_base {
 public:
  using WGTile = Shape<_128, _128, _64>;
  using SGLayout = Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>;
};

class w4a8_policy_m_128_n256 : public moe_dpas_fp8::dpas_policy_base {
 public:
  using WGTile = Shape<_128, _256, _64>;
  using SGLayout = Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>;
};

class w4a8_policy_m_256_n128 : public moe_dpas_fp8::dpas_policy_base {
 public:
  using WGTile = Shape<_256, _128, _64>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
};

class w4a8_policy_large : public moe_dpas_fp8::dpas_policy_base {
 public:
  using WGTile = Shape<_256, _256, _64>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
};

// ---------------------------------------------------------------------------
// Optional fused top-k reduction (prefill only).
//
// The grouped GEMM's natural output is `[T, N]`, one row per *routed* row, and
// every caller immediately reduces it: a token's `top_k` rows are scaled by
// their routing weights and summed into one `[batch, N]` row. That reduction
// reads `T*N` and writes `batch*N`, and the GEMM wrote `T*N` for it to read --
// so the unfused contract moves `2*T*N + batch*N` elements where the fused one
// moves `2*batch*N` (a read-modify-write of the accumulator).
//
// It is the largest lever on the down-projection shapes, where D is a third of
// the call's traffic: at qwen3's routing (`top_k = 8`) it takes D from
// `T*N*sizeof(ElementD)` to `batch*N*4*2`, i.e. 192 MB -> 48 MB at 384
// rows/expert, and deletes the caller's reduction kernel outright.
//
// The accumulator is fp32 and the caller must zero it: rows of the same token
// land on different experts, hence on different work-groups, so the only
// portable combiner is a device-scope atomic add. That makes the result
// **order-dependent** and therefore not bit-identical to the unfused path --
// the equivalence test for this contract is an SNR/cosine gate, not
// `torch.equal`. Scaling is applied before the atomic (one multiply per
// element), so the atomic itself stays a plain `fetch_add`.
//
// `out == nullptr` selects the unfused path and compiles to the same code as
// before; the branch is uniform across the work-group (it is a kernel
// argument).
// ---------------------------------------------------------------------------
struct MoEFusedReduce {
  const int* row_to_token = nullptr;  // routed row -> model token (expert-local base)
  const float* row_weight = nullptr;  // routed row -> routing weight (expert-local base)
  float* out = nullptr;               // [batch, N] fp32 accumulator, zeroed by the caller
  int batch = 0;                      // rows of `out`; bounds the scatter

  CUTE_HOST_DEVICE bool enabled() const { return out != nullptr; }
};

CUTE_DEVICE inline void atomic_add_f32(float* addr, float value) {
  sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::global_space>
      ref(*addr);
  ref.fetch_add(value);
}

// ---------------------------------------------------------------------------
// Single-tile int8 x int8 -> int32 mainloop with a per-block weight scale and
// a per-row activation scale.
//
// Structure is `igemm_kblock_device_impl` from `sycl_tla_s8_gemm.hpp` (the
// W4A8 weight-only GEMM), with two changes for the grouped/MoE case:
//   * the tile coordinate is passed in by the persistent scheduler instead of
//     being derived from the work-item's group id, and
//   * A/B/D base pointers are the per-expert slices.
//
// `blks == 1` (the AUTO_S8 `group=-1` default) collapses the outer loop to a
// single full-K int32 accumulation -- the maximum-efficiency shape -- and gets
// its own instantiation, because on this architecture the *register file*, not
// the tile traffic, is what the prefill GEMM runs out of.
//
// Why `blks == 1` is specialized (the register-file argument)
// -----------------------------------------------------------
// The blocked path needs two C fragments: the int32 DPAS accumulator `tCrC`,
// cleared once per re-scale block, and a float `tFrC` that survives across
// blocks because the per-block weight scale has to be applied before the next
// block overwrites `tCrC`. Both are the size of the work-group tile divided by
// the sub-group count, and a lane holds them in GRF for the *entire* mainloop:
//
//   tile      SG C fragment   int32 regs/lane   + float regs/lane
//   128x128       32 x 32          64                  64
//   128x256       32 x 64         128                 128
//
// With `grf_size<256>` a lane has 256 registers in total, so at 128x128 the
// float shadow alone reserves a quarter of the register file for the whole
// mainloop, and at 128x256 the two fragments together *are* the register file
// -- leaving nothing for the staged A/B tiles. That is the measured cliff
// documented in the tile-policy comment above (256-wide N tiles 35-50% slower
// than 128-wide ones, split by `TileN` and not by `TileM`), and it is pure
// overhead when `blks == 1`: with one block there is nothing to carry across
// blocks, so the scale can be folded on the way out and no float fragment
// needs to exist while the mainloop runs.
//
// The single-block epilogue therefore keeps only `tCrC` live and applies
// `scale_b[col] * scale_a[row]` in one pass, exactly like the `AccumBlock ==
// false` branch of the reference `igemm_device_impl`.
//
// The epilogue writes through the raw `[m, n]` row-major output pointer using
// the coordinates of `thr_mma.partition_C(...)`, exactly like the reference,
// because the int32 accumulator has to be converted and scaled per element
// anyway. A grouped GEMM's per-expert M is arbitrary, so tiles at the M edge
// are partial and their *store* has to be predicated -- but the scale *loads*
// are not: their indices are clamped into range instead. Both scale reads are
// then unconditional loads at a compile-time offset from a uniform base, which
// is what lets the compiler collapse the `size(tCrC)` per-element reads into
// the handful of distinct addresses a sub-group's fragment actually covers
// (all lanes of a row group share `scale_a[row]`, and a lane repeats the same
// `scale_b[col]` for every row it owns). Under the previous `continue` guard
// each read sat in its own basic block and none of that could be hoisted.
//
// Interior tiles skip the guard entirely (the cost that shows up at small K)
// -------------------------------------------------------------------------
// `m`, `n`, `m_coord` and `n_coord` are all uniform across the work-group, so
// "does this tile touch the M or N edge" is one uniform compare, not a
// per-element one. Off the edge the clamps and the store predicate are dead
// weight: per fragment element they add two compares plus two selects for the
// scale indices and another compare pair for the store, roughly doubling the
// instruction count of an epilogue whose real work is one int32->float
// convert, two multiplies and one store.
//
// That matters because the epilogue is *not* amortized over a long mainloop at
// these shapes. A 128x128 tile runs `K / 64` k-tiles -- 12 of them for the
// qwen3 down-projection (K = 768) -- while it always writes `TileM * TileN`
// elements, and qwen3 down is exactly the shape the sweep reports furthest
// from the compute target (63 TFLOPS against 87-103 for the other three). The
// fast path emits the same expression in the same order for every element it
// stores, so it is bit-identical to the guarded one
// (`test_full_tile_epilogue_matches_predicated`), and
// `ARK_MOE_W4A8_PREFILL_FULL_TILE=0` forces the guarded path for A/B
// measurement. `test_perf_prefill_epilogue_sweep` at 384 rows/expert has it at
// 1.04x (qwen3 down), 1.03x (qwen3 up) and 1.00-1.01x on the two minimax
// shapes -- the shape ordering the instruction-count argument predicted, with
// the gain concentrated where the mainloop is shortest.
//
// The store itself: one 2D block message instead of `size(tCrC)` scalar ones
// -----------------------------------------------------------------------
// Removing instructions from around the store left the store. The Xe DPAS C
// fragment gives a lane one *column* of each 8x16 atom, so the 16 lanes of a
// sub-group hold 16 *consecutive columns of one row*: a scalar
// `c[row * n + col] = ...` is a 32-byte message for 16-bit `ElementD`, half a
// cache line, and a 32x32 sub-group fragment issues **64** of them. The same
// bytes go out in 4 messages through the hardware 2D block store, which is
// what every sibling prefill kernel already uses for D
// (`sycl_tla_moe_prefill_{fp8,int,s4}_dpas.hpp`) and what the dense GEMM in
// `sycl_tla_dense_gemm.hpp` uses on this exact accumulator shape.
//
// D is the reason this is worth doing at prefill sizes rather than a tidy-up:
// at 384 rows per expert the qwen3 down-projection writes `M*N` fp16 (1.5 MB
// per expert) -- exactly as many bytes as the `N*K` int8 weights it reads,
// because N (2048) is larger than K (768) there, and over a third of the
// expert's traffic. It is the same shape whose mainloop is shortest, so it
// pays the epilogue twice.
//
// The port follows `dense_gemm_detail::gemm_device_impl` rather than the
// sibling MoE kernels, because those `reorder(tCrC, tCrC_out)` from the MMA
// fragment into an explicitly chosen `XE_STORE_2D` atom's fragment, and
// `reorder` moves *registers*: with a `float` accumulator that is free, but
// this kernel's accumulator is `int32` (`FrgTypeC` of
// `XE_DPAS_TT<8, int32_t, int8_t, int8_t>`) and has to be scaled and
// numerically converted first, which `reorder` does not do. `dense_gemm`'s
// shape is the one that fits: `make_block_2d_copy_D(mma, D)` derives its
// layout from the MMA's own C partition, so the scaled `ElementD` fragment
// (`make_tensor_like<ElementD>(tCrC)`, filled through the same `tCgC(i)`
// coordinates the scalar path uses) can be handed straight to
// `copy(copy_d, tCrD, tCgC)` with no `reorder` in between.
//
// It also *removes* the store predicate rather than skipping it: the 2D block
// message clips to the surface (`m` rows x `n` columns) described by the D
// tensor, so a partial tile at the M edge drops its out-of-range rows in
// hardware -- exactly how the sibling grouped GEMMs handle their ragged
// experts. Only the scale *loads* still need their index clamps, and only on
// edge tiles. The value written is computed by the same expression in the same
// order as the scalar path, so the two are bit-identical
// (`test_prefill_2d_store_matches_scalar`); `ARK_MOE_W4A8_PREFILL_STORE_2D=0`
// restores the scalar store for A/B measurement. `test_perf_prefill_store_-
// sweep` at 384 rows/expert makes it the largest single prefill win of the
// set: 1.14x (qwen3 up), 1.21x (qwen3 down -- the shape that pays the epilogue
// twice), 1.09x (minimax up) and 1.16x (minimax down); the run before read
// 1.16 / 1.35 / 1.12 / 1.20, same ordering.
//
// The block 2D descriptor wants a 64-byte aligned base and a row pitch that is
// a multiple of 16 bytes. The base here is the expert's slice
// `Outputs + pre_rows * N`, with `pre_rows` a runtime routing value, so the
// dispatcher gates on `N * sizeof(ElementD) % 64 == 0` (which makes *every*
// expert's base 64-byte aligned given an aligned tensor) and on the base
// pointer itself; anything else keeps the scalar store.
// ---------------------------------------------------------------------------
template <class GmemTiledCopyA, class GmemTiledCopyB, class TiledMMA, typename ElementD>
CUTE_DEVICE void xe_gemm_w4a8(const int8_t* a, const int8_t* b, ElementD* c, const float* scale_a,
                              const float* scale_b, int m, int n, int k, int blocksize, int blks, int m_coord,
                              int n_coord, bool allow_full_tile, bool allow_block_2d_store, int prefetch_dist,
                              MoEFusedReduce const& reduce, TiledMMA const& mma) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int local_id = static_cast<int>(item.get_local_linear_id());

  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(m_coord, n_coord, 0);

  // The fused path never writes through `c` (it scatters into `reduce.out`
  // instead) and its caller has no `[T, N]` buffer to hand over, so `c` is
  // null there. D and its 2D copy atom are still built -- they are ordinary
  // objects, not lazily constructed -- so give them a valid base to describe.
  ElementD* d_base = c != nullptr ? c : reinterpret_cast<ElementD*>(reduce.out);

  auto A = make_tensor(make_gmem_ptr(const_cast<int8_t*>(a)), make_shape(m, k), make_stride(k, _1{}));
  auto B = make_tensor(make_gmem_ptr(const_cast<int8_t*>(b)), make_shape(n, k), make_stride(k, _1{}));
  auto D = make_tensor(make_gmem_ptr(d_base), make_shape(m, n), make_stride(n, _1{}));

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B.shape());
  Tensor cC = make_identity_tensor(D.shape());

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(m_coord, _));
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(n_coord, _));
  Tensor gC = local_tile(cC, wg_tile, wg_coord, Step<_1, _1, X>{});

  auto copy_a = get_block_2d_copy_A<GmemTiledCopyA>(mma, A);
  auto copy_b = get_block_2d_copy_B<GmemTiledCopyB>(mma, B);
  auto copy_d = make_block_2d_copy_D(mma, D);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  Tensor tCrC = partition_fragment_C(mma, select<0, 1>(wg_tile));
  Tensor tCgC = thr_mma.partition_C(gC);

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto pAgA = prefetch_a.get_slice(local_id).partition_S(gA);
  auto pBgB = prefetch_b.get_slice(local_id).partition_S(gB);

  constexpr auto barrier_scope = ScopeWorkgroup;

  const int k_tile_size = static_cast<int>(get<2>(wg_tile));
  const int k_tiles_per_block = blocksize / k_tile_size;
  const int k_tile_count = blks * k_tiles_per_block;
  int k_tile_prefetch = 0;

  // One k-tile of the DPAS pipeline. Shared by both paths so the two
  // instantiations differ only in what they keep live around it.
  auto run_k_tile = [&](int k_tile) {
    barrier_arrive(barrier_scope);

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }
    ++k_tile_prefetch;

    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);
    cute::gemm(mma, tCrA, tCrB, tCrC);

    barrier_wait(barrier_scope);
  };

  // Runtime bound (`ARK_MOE_W4A8_PREFILL_PREFETCH`), so no unroll pragma: the
  // prologue runs once per tile, ahead of a mainloop of `k_tile_count`
  // iterations, and its trip count is uniform across the work-group.
  for (; k_tile_prefetch < prefetch_dist && k_tile_prefetch < k_tile_count; ++k_tile_prefetch) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }

  // Does this tile touch the M or N edge? Uniform across the work-group (`m`,
  // `n` and both coordinates are), so the epilogues below branch once instead
  // of testing every fragment element.
  const bool full_tile = allow_full_tile && (m_coord + 1) * static_cast<int>(get<0>(wg_tile)) <= m &&
                         (n_coord + 1) * static_cast<int>(get<1>(wg_tile)) <= n;

  // `blks` is a kernel argument, so it is uniform across the work-group and
  // this branch never splits the split-barrier pairing below.
  if (blks == 1) {
    clear(tCrC);

    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
      run_k_tile(k_tile);
    }

    // Single expression, instantiated once guarded and once not. `FullTile`
    // only removes work: the value stored is computed by the same operations
    // in the same order, so the two paths are bit-identical.
    auto store_scaled = [&](auto full) {
      CUTE_UNROLL
      for (int i = 0; i < size(tCrC); ++i) {
        auto coord = tCgC(i);
        const int row = static_cast<int>(get<0>(coord));
        const int col = static_cast<int>(get<1>(coord));
        if constexpr (decltype(full)::value) {
          c[static_cast<size_t>(row) * n + col] = static_cast<ElementD>(
              static_cast<float>(tCrC(i)) * scale_b[static_cast<size_t>(col)] * scale_a[row]);
        } else {
          // Clamp rather than branch: an out-of-range element's value is
          // dropped by the guarded store, and unconditional loads let the
          // redundant reads across the fragment collapse. `m` and `n` are both
          // >= 1 here (an expert with no rows contributes no tiles).
          const int row_in = row < m ? row : m - 1;
          const int col_in = col < n ? col : n - 1;
          const float value = static_cast<float>(tCrC(i)) * scale_b[static_cast<size_t>(col_in)] * scale_a[row_in];
          if (row < m && col < n) {
            c[static_cast<size_t>(row) * n + col] = static_cast<ElementD>(value);
          }
        }
      }
    };

    // Same values in the same order, through the hardware 2D block store. The
    // element predicate is gone because the message clips to the `m x n`
    // surface; only the scale loads still clamp their indices.
    auto store_scaled_2d = [&](auto full) {
      Tensor tCrD = make_tensor_like<ElementD>(tCrC);
      CUTE_UNROLL
      for (int i = 0; i < size(tCrC); ++i) {
        auto coord = tCgC(i);
        const int row = static_cast<int>(get<0>(coord));
        const int col = static_cast<int>(get<1>(coord));
        if constexpr (decltype(full)::value) {
          tCrD(i) = static_cast<ElementD>(static_cast<float>(tCrC(i)) * scale_b[static_cast<size_t>(col)] *
                                          scale_a[row]);
        } else {
          const int row_in = row < m ? row : m - 1;
          const int col_in = col < n ? col : n - 1;
          tCrD(i) = static_cast<ElementD>(static_cast<float>(tCrC(i)) * scale_b[static_cast<size_t>(col_in)] *
                                          scale_a[row_in]);
        }
      }
      copy(copy_d, tCrD, tCgC);
    };

    // Fused top-k reduction: scale the row by its routing weight and
    // accumulate it into the token's row of the `[batch, n]` fp32 output.
    // Out-of-range rows are dropped rather than clamped -- a clamped scatter
    // would corrupt a *valid* token's accumulator, which the guarded store
    // above cannot do -- but the loads stay unconditional so they still
    // collapse across the fragment. `row_to_token` is caller data, so its
    // value is range-checked as well: a bad index drops the contribution
    // instead of writing outside the accumulator.
    auto store_fused = [&](auto full) {
      CUTE_UNROLL
      for (int i = 0; i < size(tCrC); ++i) {
        auto coord = tCgC(i);
        const int row = static_cast<int>(get<0>(coord));
        const int col = static_cast<int>(get<1>(coord));
        const int row_in = decltype(full)::value ? row : (row < m ? row : m - 1);
        const int col_in = decltype(full)::value ? col : (col < n ? col : n - 1);
        const float value = static_cast<float>(tCrC(i)) * scale_b[static_cast<size_t>(col_in)] *
                            scale_a[row_in] * reduce.row_weight[row_in];
        const int token = reduce.row_to_token[row_in];
        const bool in_tile = decltype(full)::value || (row < m && col < n);
        if (in_tile && token >= 0 && token < reduce.batch) {
          atomic_add_f32(&reduce.out[static_cast<size_t>(token) * n + col], value);
        }
      }
    };

    if (reduce.enabled()) {
      if (full_tile) {
        store_fused(std::true_type{});
      } else {
        store_fused(std::false_type{});
      }
    } else if (allow_block_2d_store) {
      if (full_tile) {
        store_scaled_2d(std::true_type{});
      } else {
        store_scaled_2d(std::false_type{});
      }
    } else if (full_tile) {
      store_scaled(std::true_type{});
    } else {
      store_scaled(std::false_type{});
    }
    return;
  }

  Tensor tFrC = make_tensor_like<float>(tCrC);

  CUTE_UNROLL
  for (int i = 0; i < size(tFrC); ++i) {
    tFrC(i) = 0.0f;
  }

  for (int ib = 0; ib < blks; ++ib) {
    clear(tCrC);

    for (int bk = 0; bk < k_tiles_per_block; ++bk) {
      run_k_tile(ib * k_tiles_per_block + bk);
    }

    if (full_tile) {
      CUTE_UNROLL
      for (int i = 0; i < size(tCrC); ++i) {
        const int col = static_cast<int>(get<1>(tCgC(i)));
        tFrC(i) += static_cast<float>(tCrC(i)) * scale_b[static_cast<size_t>(col) * blks + ib];
      }
    } else {
      CUTE_UNROLL
      for (int i = 0; i < size(tCrC); ++i) {
        const int col = static_cast<int>(get<1>(tCgC(i)));
        const int col_in = col < n ? col : n - 1;
        tFrC(i) += static_cast<float>(tCrC(i)) * scale_b[static_cast<size_t>(col_in) * blks + ib];
      }
    }
  }

  if (reduce.enabled()) {
    CUTE_UNROLL
    for (int i = 0; i < size(tFrC); ++i) {
      auto coord = tCgC(i);
      const int row = static_cast<int>(get<0>(coord));
      const int col = static_cast<int>(get<1>(coord));
      const int row_in = full_tile ? row : (row < m ? row : m - 1);
      const float value = tFrC(i) * scale_a[row_in] * reduce.row_weight[row_in];
      const int token = reduce.row_to_token[row_in];
      const bool in_tile = full_tile || (row < m && col < n);
      if (in_tile && token >= 0 && token < reduce.batch) {
        atomic_add_f32(&reduce.out[static_cast<size_t>(token) * n + col], value);
      }
    }
    return;
  }

  if (allow_block_2d_store) {
    Tensor tCrD = make_tensor_like<ElementD>(tFrC);
    if (full_tile) {
      CUTE_UNROLL
      for (int i = 0; i < size(tFrC); ++i) {
        const int row = static_cast<int>(get<0>(tCgC(i)));
        tCrD(i) = static_cast<ElementD>(tFrC(i) * scale_a[row]);
      }
    } else {
      CUTE_UNROLL
      for (int i = 0; i < size(tFrC); ++i) {
        const int row = static_cast<int>(get<0>(tCgC(i)));
        const int row_in = row < m ? row : m - 1;
        tCrD(i) = static_cast<ElementD>(tFrC(i) * scale_a[row_in]);
      }
    }
    copy(copy_d, tCrD, tCgC);
    return;
  }

  if (full_tile) {
    CUTE_UNROLL
    for (int i = 0; i < size(tFrC); ++i) {
      auto coord = tCgC(i);
      const int row = static_cast<int>(get<0>(coord));
      const int col = static_cast<int>(get<1>(coord));
      c[static_cast<size_t>(row) * n + col] = static_cast<ElementD>(tFrC(i) * scale_a[row]);
    }
    return;
  }

  CUTE_UNROLL
  for (int i = 0; i < size(tFrC); ++i) {
    auto coord = tCgC(i);
    const int row = static_cast<int>(get<0>(coord));
    const int col = static_cast<int>(get<1>(coord));
    const int row_in = row < m ? row : m - 1;
    const float value = tFrC(i) * scale_a[row_in];
    if (row < m && col < n) {
      c[static_cast<size_t>(row) * n + col] = static_cast<ElementD>(value);
    }
  }
}

// ---------------------------------------------------------------------------
// Persistent atomic scheduler over `rows_per_expert`.
//
// Structurally identical to `moe_dpas_int::MoEGEMM_int` (which is itself the
// vllm-xpu-kernels grouped-GEMM scheduler); only the per-expert pointer
// arithmetic and the mainloop call differ:
//   * A / D advance by the expert's token offset (`pre_rows`), and so does the
//     per-token activation scale.
//   * B advances by `expert * N * K` int8 elements, the block scales by
//     `expert * N * blks` floats.
// ---------------------------------------------------------------------------
template <class GmemTiledCopyA, class GmemTiledCopyB, class TiledMMA, typename ElementD>
CUTE_DEVICE void MoEGEMM_w4a8(const int8_t* Activations, const int8_t* Weights, const float* ScaleA,
                              const float* ScaleB, ElementD* Outputs, TiledMMA const& mma,
                              const int* rows_per_expert, const int32_t num_experts, const int32_t gemm_n,
                              const int32_t gemm_k, const int32_t blocksize, const int32_t blks,
                              const bool allow_full_tile, const bool allow_block_2d_store,
                              const int32_t prefetch_dist, MoEFusedReduce reduce, int32_t* atomic_buffer,
                              const sycl::local_accessor<int32_t, 1>& slm_mem_const) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto wg_tile = mma.tile_mnk();
  auto wg_tile_m = get<0>(wg_tile);
  auto wg_tile_n = get<1>(wg_tile);

  int group_id = item.get_group_linear_id();
  int gemm_n_pad = (gemm_n + wg_tile_n - 1) / wg_tile_n * wg_tile_n;
  int group_m_id = (group_id * wg_tile_n) / gemm_n_pad;
  int group_range = item.get_group_range(1);
  int local_id = item.get_local_linear_id();

  if (group_id == 0 && local_id == 0) {
    auto atm = sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                sycl::access::address_space::global_space>(atomic_buffer[0]);
    atm.store(0);
  }

  int pre_rows = 0;
  int pre_tiles = 0;

  int32_t* slm_mem =
      static_cast<int32_t*>(slm_mem_const.template get_multi_ptr<sycl::access::decorated::no>().get());

  for (int i = 0; i < num_experts; ++i) {
    int gemm_m = rows_per_expert[i];
    int cumsum_rows_for_experts = pre_rows + gemm_m;
    int cumsum_tiles_for_experts = (gemm_m + wg_tile_m - 1) / wg_tile_m + pre_tiles;

    if (group_m_id >= cumsum_tiles_for_experts) {
      pre_rows = cumsum_rows_for_experts;
      pre_tiles = cumsum_tiles_for_experts;
      continue;
    }

    const int expert_id = i;
    const int64_t B_offset =
        static_cast<int64_t>(expert_id) * static_cast<int64_t>(gemm_n) * static_cast<int64_t>(gemm_k);
    const int64_t SB_offset =
        static_cast<int64_t>(expert_id) * static_cast<int64_t>(gemm_n) * static_cast<int64_t>(blks);

    const int8_t* ptr_A_curr_batch = Activations + static_cast<int64_t>(pre_rows) * gemm_k;
    const int8_t* ptr_B_curr_batch = Weights + B_offset;
    const float* ptr_SA_curr_batch = ScaleA + pre_rows;
    const float* ptr_SB_curr_batch = ScaleB + SB_offset;
    ElementD* ptr_D_curr_batch = Outputs == nullptr ? nullptr : Outputs + static_cast<int64_t>(pre_rows) * gemm_n;

    // The scatter targets a `[batch, N]` accumulator shared by every expert,
    // so only the per-row side tables advance with the expert; `reduce.out`
    // stays put.
    MoEFusedReduce expert_reduce = reduce;
    if (reduce.enabled()) {
      expert_reduce.row_to_token = reduce.row_to_token + pre_rows;
      expert_reduce.row_weight = reduce.row_weight + pre_rows;
    }

    while (group_m_id < cumsum_tiles_for_experts) {
      const int n_coord = (group_id * wg_tile_n) % gemm_n_pad / wg_tile_n;
      const int m_coord = (group_m_id - pre_tiles);

      xe_gemm_w4a8<GmemTiledCopyA, GmemTiledCopyB>(ptr_A_curr_batch, ptr_B_curr_batch, ptr_D_curr_batch,
                                                   ptr_SA_curr_batch, ptr_SB_curr_batch, gemm_m, gemm_n, gemm_k,
                                                   blocksize, blks, m_coord, n_coord, allow_full_tile,
                                                   allow_block_2d_store, prefetch_dist, expert_reduce, mma);

      if (local_id == 0) {
        slm_mem[0] = cutlass::atomicAdd(atomic_buffer, 1);
      }
      item.barrier(sycl::access::fence_space::local_space);
      group_id = group_range + slm_mem[0];
      group_m_id = (group_id * wg_tile_n) / gemm_n_pad;
    }
    pre_rows = cumsum_rows_for_experts;
    pre_tiles = cumsum_tiles_for_experts;
  }
}

// ---------------------------------------------------------------------------
// Grouped-GEMM launcher (fork of `moe_dpas_int::MoEGEMMLauncher_int`, with the
// int8 DPAS atom of `sycl_tla_s8_gemm.hpp`).
// ---------------------------------------------------------------------------
template <class Policy, typename ElementD>
void MoEGEMMLauncher_w4a8(sycl::queue& stream, const int8_t* activations, const int8_t* weights,
                          const float* scale_a, const float* scale_b, ElementD* outputs, const int gemm_n,
                          const int gemm_k, const int* rows_per_expert, const int num_experts, const int blocksize,
                          const int blks, const bool allow_full_tile, const bool allow_block_2d_store,
                          const int prefetch_dist, MoEFusedReduce reduce, int32_t* atomic_buffer) {
  using Op = XE_DPAS_TT<8, int32_t, int8_t, int8_t>;
  using WGTile = typename Policy::WGTile;
  using SGLayout = typename Policy::SGLayout;
  using MMA = typename TiledMMAHelper<MMA_Atom<Op>, Layout<WGTile>, SGLayout>::TiledMMA;
  auto mma = MMA{};

  int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  auto MaxThreadsPerWorkgroup = size(mma);

  static constexpr int MaxThreadsPerSM = 512;
  if (MaxThreadsPerSM % MaxThreadsPerWorkgroup != 0) {
    throw std::runtime_error("moe_gemm_w4a8: MaxThreadsPerSM must be divisible by MaxThreadsPerWorkgroup");
  }

  sycl::range<3> local(1, 1, MaxThreadsPerWorkgroup);
  sycl::range<3> global(1, sm_count * MaxThreadsPerSM / MaxThreadsPerWorkgroup, 1);

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props{syclex::sub_group_size<16>, intelex::grf_size<256>};

  using GmemTiledCopyA = typename Policy::GmemTiledCopyA;
  using GmemTiledCopyB = typename Policy::GmemTiledCopyB;

  auto event = stream.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> local_mem(sycl::range<1>(1), cgh);
    cgh.parallel_for<MoEW4A8GemmName<Policy, ElementD>>(
        sycl::nd_range<3>{global * local, local}, kernel_props, [=](auto) {
          MoEGEMM_w4a8<GmemTiledCopyA, GmemTiledCopyB>(activations, weights, scale_a, scale_b, outputs, mma,
                                                       rows_per_expert, num_experts, gemm_n, gemm_k, blocksize,
                                                       blks, allow_full_tile, allow_block_2d_store, prefetch_dist,
                                                       reduce, atomic_buffer, local_mem);
        });
  });

  EventManager::getInstance().addEvent(event);
  event.wait();
}

// ---------------------------------------------------------------------------
// Prefill launch: one instantiation per tile policy (see `moe_w4a8_prefill_select_tile`).
//
// The rungs match the tile ladder of `launch_igemm_kblock` in
// `sycl_tla_s8_gemm.hpp`: a grouped GEMM's M is *per expert*, so the ladder
// walks the average rows/expert rather than the total token count.
//
// The M rung is a row threshold and the N rung a divisibility test, and both
// are about not scheduling work the shape does not have:
//
//   * `TileM` stops at 128. The 256-row tile halves how often each expert's B
//     panel is pulled through L2/DRAM (B is read once per M tile), but an
//     expert launches `ceil(M / TileM)` *full* tiles, so it only breaks even
//     where `ceil(M/256)*256 == ceil(M/128)*128` -- false at the 384 and 341
//     rows/expert the perf suite measures, where it computes 512 rows for 384
//     rows of data and reads 1.05-1.45x slower. The 8K prompt puts Qwen3-MoE
//     at exactly 512 rows/expert, where the padding argument does not apply,
//     and `test_perf_prefill_tile_sweep_long_seq` measures it there: still not
//     ahead (a tie on qwen3 up, 5.4% behind on qwen3 down). So the rung is
//     gone rather than gated -- there is no routing at which it has been
//     measured to win, and the ladder only sees the *average* rows/expert, so
//     a skewed routing would put individual experts back on the padding cliff
//     even when the average divides. Both 256-row policies stay compiled and
//     reachable through `ARK_MOE_W4A8_PREFILL_TILE` for a re-sweep.
//
//   * `TileN = 256` halves how often A is re-read (once per N tile) and is
//     ahead or level everywhere the sweeps can compare it, so it is taken
//     whenever N divides into it exactly. `N % 256 != 0` would pad the last
//     tile the same way a ragged M does, and no shipped shape needs it: every
//     N here (1536 / 2048 / 3072) is a multiple of 256.
//
// The rung used to be `A_avg_M >= 256 -> 256x128` with no padding test and a
// 128-wide N at every rung, which is what made the 384 rows/expert batch --
// the compute-bound batch the perf suite now runs -- land on the slowest
// column of its own sweep; it then became a padding-gated 256-row tile, which
// the 8K-prompt sweep has now retired.
//
// The tile choice itself lives in `moe_w4a8_prefill_select_tile`
// (`sycl_tla_moe_w4a8_helpers.hpp`) and happens before this function is
// reached, so each translation unit instantiates exactly one policy. See
// `ARK_MOE_W4A8_PREFILL_TILE` there for the override.
//
// `ARK_MOE_W4A8_PREFILL_FULL_TILE=0` makes every tile take the guarded
// epilogue (see `xe_gemm_w4a8`), which is the A/B baseline for the interior-
// tile fast path; it is read here, once per call, rather than on the device.
//
// `ARK_MOE_W4A8_PREFILL_STORE_2D=0` puts the epilogue back on the scalar
// predicated store instead of the hardware 2D block store, the A/B baseline
// for that change. The block message needs a 64-byte aligned surface base and
// a 16-byte multiple row pitch; D's per-expert base is `outputs + pre_rows * N`
// for a routing-dependent `pre_rows`, so the gate is on the row stride itself
// (`N * sizeof(ElementD) % 64 == 0`, which covers the pitch as well) plus the
// tensor base. Every shipped N (1536 / 2048 / 3072 with 16-bit D) clears it;
// anything that does not keeps the scalar store rather than risking a
// misaligned descriptor.
// ---------------------------------------------------------------------------
template <class Policy, typename ElementD>
void moe_w4a8_prefill_launch(const moe_w4a8_detail::W4A8PrefillParams& p) {
  if (p.num_experts == 0 || p.N == 0 || p.K == 0 || p.total_tokens == 0) return;

  compat::set_default_queue(*p.q);

  auto* outputs = static_cast<ElementD*>(p.outputs);

  MoEFusedReduce reduce{};
  if (p.fused_out != nullptr) {
    reduce.row_to_token = p.row_to_token;
    reduce.row_weight = p.row_weight;
    reduce.out = p.fused_out;
    reduce.batch = p.fused_batch;
  }

  const bool allow_full_tile = moe_decode_detail::env_flag_enabled("ARK_MOE_W4A8_PREFILL_FULL_TILE", true);
  const bool store_2d_aligned = !reduce.enabled() && (static_cast<size_t>(p.N) * sizeof(ElementD)) % 64 == 0 &&
                                reinterpret_cast<uintptr_t>(outputs) % 64 == 0;
  const bool allow_block_2d_store =
      store_2d_aligned && moe_decode_detail::env_flag_enabled("ARK_MOE_W4A8_PREFILL_STORE_2D", true);
  const int prefetch_dist = moe_w4a8_prefill_prefetch_dist();
  int32_t* atomic_buffer = moe_dpas_fp8::get_atomic_scratch_buffer(p.q);

  MoEGEMMLauncher_w4a8<Policy, ElementD>(*p.q, p.qact, p.weights, p.ascale, p.wscale, outputs, p.N, p.K,
                                         p.num_tokens_per_expert, p.num_experts, p.blocksize, p.blks,
                                         allow_full_tile, allow_block_2d_store, prefetch_dist, reduce,
                                         atomic_buffer);
}

}  // namespace moe_w4a8

}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
