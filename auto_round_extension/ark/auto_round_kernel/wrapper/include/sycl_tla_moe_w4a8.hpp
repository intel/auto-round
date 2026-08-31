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
#include <stdexcept>
#include <type_traits>

#ifdef ARK_XPU
#include <sycl/sycl.hpp>
#endif

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

// Pulls in the cutlass-sycl / CuTe include set, the `dpas_policy_base` policy
// root, `make_moe_tensor`, and `get_persistent_atomic_buffer` (via the FP8
// header it includes).
#include "sycl_tla_moe_prefill_int_dpas.hpp"
// `DeviceScratchPool`, `env_flag_enabled`, `fill_expert_id_per_token`,
// `SG_SIZE` / `N_TILE`, and the shared nibble decoders.
#include "sycl_tla_moe_decode.hpp"

namespace ark {
namespace moe_w4a8 {

using namespace cute;

using moe_decode_detail::DeviceScratchPool;
using moe_decode_detail::N_TILE;
using moe_decode_detail::SG_SIZE;
using moe_dequant::decode_int4_octet;

// Symmetric int4 full range: 2^(bits-1). Matches `fullrange` in
// `xpu_wrapper.hpp`'s `packscale` rescale kernel.
constexpr float kInt4FullRange = 8.0f;
constexpr float kInt8Max = 127.0f;

// K elements decoded per work-item in the prepack kernel (one 32-bit word of
// packed nibbles). Requires `K % 8 == 0`, which the shape gate enforces.
constexpr int kPrepackOctet = 8;

// ---------------------------------------------------------------------------
// Kernel name tags (one per specialization, required for SYCL kernel naming)
// ---------------------------------------------------------------------------
template <typename ScalarT>
class MoEW4A8ActQuant;

template <typename ScalarT, int VEC, int UNROLL>
class MoEW4A8ActQuantVec;

template <typename ScalarT, int VEC, int MAX_STEPS>
class MoEW4A8ActQuantSingle;

template <typename ScalarT>
class MoEW4A8ScaleReduce;

template <typename ScalarT>
class MoEW4A8Repack;

template <typename ElementD>
class MoEW4A8DecodeGemv;

template <typename ElementD, int NCOLS, int CH>
class MoEW4A8DecodeKSplit;

template <class Policy, typename ElementD>
class MoEW4A8GemmName;

// ---------------------------------------------------------------------------
// Scratch pools.
//
// The activation-quantization buffers (`[total_tokens, K]` int8 +
// `[total_tokens]` fp32) and the decode expert map (`[total_tokens]` int32)
// are recomputed on every call, so they come from persistent per-queue slabs
// instead of a hot-path `malloc_device`. Same lifetime contract as
// `moe_decode_detail::int4_repack_pool` -- released explicitly through
// `moe_w4a8_release_scratch`, never from a static destructor.
// ---------------------------------------------------------------------------
inline DeviceScratchPool& qact_pool() {
  static DeviceScratchPool pool;
  return pool;
}

inline DeviceScratchPool& expert_map_pool() {
  static DeviceScratchPool pool;
  return pool;
}

// ---------------------------------------------------------------------------
// Per-token dynamic activation quantization: act dtype -> int8 + fp32 scale.
//
// One sub-group per token: lanes stride the K axis (coalesced), reduce the
// absmax with `reduce_over_group`, then write back the quantized row. A row
// that is entirely zero gets `scale = 0` and quantizes to all zeros (the
// reciprocal is forced to 0 instead of inf).
//
// The decode path also needs `expert_id_per_token`, which
// `moe_decode_detail::fill_expert_id_per_token` produces in a kernel of its
// own. That kernel does one tiny scan per token, so at decode sizes it is pure
// launch overhead on a timeline where the GEMV itself is only tens of
// microseconds and one call is issued per generated token. This kernel already
// runs one sub-group per token, so when `expert_id_per_token != nullptr` lane 0
// folds the same scan in and the separate launch disappears -- the same "one
// fewer kernel launch on the decode timeline" the FP8 DPAS decode dispatch
// gets by consuming `num_tokens_per_expert` directly. The scan is the verbatim
// body of `fill_expert_id_per_token`, including its clamp to
// `num_experts - 1` for a routing table that sums to less than `total_tokens`.
//
// Message width (the prefill cost that matters)
// ---------------------------------------------
// This kernel is a pure streaming pass -- it reads `[T, K]` activations twice
// (absmax, then quantize) and writes `[T, K]` int8 -- so at prefill sizes it is
// a sizeable fraction of the whole `moe_gemm_w4a8` call, not a preamble. At
// 32768 routed rows and K = 2048 it touches ~200 MB, next to the ~400 MB the
// grouped GEMM streams for the qwen3 up-proj weights.
//
// The scalar mapping below (`k = lane; k < K; k += SG_SIZE`) moves that traffic
// in the *narrowest* messages the sub-group can issue: 16 lanes x one 16-bit
// element is a 32-byte load and 16 lanes x one int8 is a **16-byte** store, i.e.
// a quarter of a cache line per store message. That is the same defect the
// decode GEMV had before the K-split rewrite, and it was worth 1.09-1.93x there.
//
// `launch_act_dynamic_quant_vec` fixes it the same way: each lane owns `VEC`
// *consecutive* elements, so one message covers `SG_SIZE * VEC` contiguous
// elements -- 256 bytes of activations and 128 bytes of int8 at `VEC = 8`.
// Both passes read the same `sycl::vec`, and the second pass re-reads a row the
// first pass just touched, so it is served by the cache rather than DRAM.
//
// `VEC` is chosen from K: 8 when `K % 128 == 0` (every shipped MoE shape --
// 768 / 1536 / 2048 / 3072), otherwise 4, which the `K % 64 == 0` shape gate
// always satisfies. A misaligned base pointer (never the case for torch
// allocations or the scratch pool) falls back to the scalar kernel, and
// `ARK_MOE_W4A8_ACT_QUANT_VEC=0` forces it for A/B measurement.
//
// `test_perf_prefill_act_quant_sweep` on BMG (384 rows/expert, bf16 act) puts
// the widened messages at 1.12x (qwen3 up), 1.10x (qwen3 down), 1.15x (minimax
// up) and 1.07x (minimax down) of the scalar mapping on the *whole*
// `moe_gemm_w4a8` call -- the quantization pass alone is a larger share of
// prefill than that, since the GEMM around it is unchanged. (Earlier runs of
// the same sweep read 1.05 / 1.13 / 1.11 / 1.04, 1.13 / 1.14 / 1.12 / 1.04 and
// 1.14 / 1.07 / 1.10 / 1.07: the ranking is stable, the individual ratios move
// by a few percent between runs.)
//
// `sycl::vec<uint16_t, VEC>` is used rather than `sycl::vec<ScalarT, VEC>`
// because `sycl::vec` of `bfloat16` is not universally available; the elements
// are `bit_cast` back one at a time, exactly like the decode kernels'
// `ActVec` loads in `sycl_tla_moe_decode.hpp`.
//
// Requests in flight (the cost widening the messages did not address)
// -------------------------------------------------------------------
// Wide messages fix how many bytes each *request* moves; they do not change
// how many requests a work-item has outstanding. This kernel walks K with a
// runtime trip count (`steps = K / (SG_SIZE * VEC)`) and folds every vector
// into the same `local_max` accumulator, so the loop reads as: issue one load,
// stall until it returns, `fmax` it, repeat. Xe cores execute in order and
// `fmax` is not reassociated without fast-math, so each thread keeps roughly
// *one* 256-byte load in flight.
//
// That is a Little's-law problem, not a bandwidth one: 1280 concurrent
// sub-groups (the B60's occupancy ceiling -- 160 XVEs x 8 thread slots) x 256
// bytes is ~320 KB of in-flight reads, under the ~456 KB a 456 GB/s device
// needs to stay busy across a ~1 us memory latency, and a real launch rarely
// fills every slot. The same argument is why the decode GEMV loads two chunks
// per iteration (`launch_w4a8_decode_ksplit`), and why the pair is spelled out
// there rather than left to the compiler.
//
// `UNROLL` gives the pass the same treatment: each iteration loads `UNROLL`
// *independent* vectors before consuming any of them, and reduces them into
// `UNROLL` separate partial maxima so the loads do not serialize behind the
// accumulator chain either. At the default `UNROLL = 4` a thread holds 1 KB,
// which clears the 456 KB well before every slot is occupied. The quantize
// pass batches its loads the same way, and its stores are already independent.
// `steps % UNROLL` vectors are left to a tail loop -- `K = 768` (qwen3 down)
// gives `steps = 6`, so the tail is real code, not a formality.
//
// Nothing that rounds changes: the per-lane partial reduction is still `fmax`
// over the same values (exact and order-independent, so partial maxima merge
// to the same bits), and every element goes through the same `rint`/`clamp`
// expression. `UNROLL = 1` is the previous kernel instruction for instruction,
// so `ARK_MOE_W4A8_ACT_QUANT_UNROLL=1` is an exact A/B baseline;
// `test_act_quant_unroll_matches` asserts every depth is bit-identical and
// `test_perf_prefill_act_quant_unroll_sweep` times them.
//
// That sweep keeps the default, though not by much: only minimax up is a real
// A/B in it (the other three shapes take the single-pass kernel below, where
// `UNROLL` is dead code, so their rows are three sets of identical kernels --
// a useful noise probe, spreading 3.3-7.3% in the latest run and 0.4-3.9% in
// the one before). On that shape the three depths read 6.982 ms at 1,
// 6.795 ms at 2 and 6.837 ms at 4: the batched loads are worth 1.02-1.03x over
// `UNROLL = 1`, and the 0.6% between 2 and 4 is an order of magnitude inside
// the noise the identical-kernel rows show, so the default stays at 4 (the
// earlier run had it 8.959 / 8.967 / 9.139 ms, i.e. 4 fastest).
//
// Reading the row once (the traffic the two passes duplicate)
// -----------------------------------------------------------
// Batching the loads did not change how many there are. The absmax has to see
// the whole row before the first element can be quantized, so the kernel reads
// `[T, K]`, reduces, then reads `[T, K]` again -- and at 384 rows per expert
// the activation matrix is 1.5 MB for K = 2048, against 3.1 MB of weights for
// the whole GEMM. The re-read is L2-resident when the row is still there, but
// the rows a work-group quantizes second are evicted by the ones it quantized
// first well before the pass ends: at 8 MB of L2 and 4 KB per bf16 row of
// K = 2048, only ~2000 of 2048 tokens' rows fit *if nothing else is resident*,
// and the GEMM's weights are competing for the same cache immediately after.
//
// A row is small enough to keep in registers instead: a lane owns `K / 16`
// elements, so `K = 2048` is 256 bytes -- 64 of the 128 dwords per lane the
// quantizer gets (it launches without `grf_size<256>`, unlike the GEMM). Load
// the row once, reduce it, then quantize out of the registers. The second read
// disappears, and every load is issued before any of them is consumed, which
// subsumes what `UNROLL` was doing (`UNROLL = steps`, effectively) rather than
// competing with it.
//
// `MAX_STEPS` is the compile-time cap that makes the fragment a register array
// rather than scratch: the loop is `#pragma unroll` over `MAX_STEPS` with an
// `if (s < steps)` guard, so every index is a constant and SROA can promote it.
// Two rungs are instantiated -- 8 vectors (K <= 1024 at VEC = 8, 32 dwords) and
// 16 (K <= 2048, 64 dwords) -- and anything longer keeps the two-pass kernel,
// which is why minimax's K = 3072 up-projection still takes the old path. The
// partial maxima stay at four accumulators, as in the two-pass kernel, so the
// reduction chain is unchanged in both cost and value.
//
// This was a register-pressure gamble -- if 64 dwords of row plus addressing
// spilled, the pass would get slower, not faster -- and the sweep settled it in
// its favour: at 384 rows/expert the single-pass kernel is 1.06x (qwen3 down,
// K = 768), 1.04x (qwen3 up, K = 2048, the rung filled exactly) and 1.02x
// (minimax down, K = 1536) against the two-pass one. Nothing spills. minimax
// up (K = 3072) is past the last rung, so both of its rows run the *same*
// two-pass kernel and their 1.01x is this sweep's noise probe.
// `ARK_MOE_W4A8_ACT_QUANT_SINGLE_PASS=0` still restores the two-pass
// kernel exactly, `test_perf_prefill_act_quant_single_pass_sweep` times the
// pair, and `test_act_quant_single_pass_matches` asserts they agree bit for bit
// (same `fmax` set, same `inv`, same `rint`/`clamp`).
// ---------------------------------------------------------------------------

// Fold the per-token expert scan (decode only) into the quantization kernel.
// Verbatim body of `moe_decode_detail::fill_expert_id_per_token`.
inline void act_quant_write_scale(float* ascale, int token, float scale, int* expert_id_per_token,
                                  const int* num_tokens_per_expert, int num_experts) {
  ascale[token] = scale;
  if (expert_id_per_token == nullptr) return;
  int offset = 0;
  int expert = num_experts - 1;
  for (int e = 0; e < num_experts; ++e) {
    const int n = num_tokens_per_expert[e];
    if (token < offset + n) {
      expert = e;
      break;
    }
    offset += n;
  }
  expert_id_per_token[token] = expert;
}

template <typename ScalarT, int VEC, int UNROLL>
void launch_act_dynamic_quant_vec(sycl::queue* q, const ScalarT* activations, int8_t* qact, float* ascale,
                                  int total_tokens, int K, int* expert_id_per_token,
                                  const int* num_tokens_per_expert, int num_experts) {
  static_assert(sizeof(ScalarT) == sizeof(uint16_t), "ScalarT must be a 16-bit floating type");
  static_assert(VEC == 4 || VEC == 8, "VEC must be 4 or 8");
  static_assert(UNROLL >= 1 && (UNROLL & (UNROLL - 1)) == 0, "UNROLL must be a power of two");
  using ActVec = sycl::vec<uint16_t, VEC>;
  using QVec = sycl::vec<int8_t, VEC>;

  // Vectors a lane walks over. `K % (SG_SIZE * VEC) == 0` is checked by the
  // caller, so the loop needs no tail -- but `steps` need not be a multiple of
  // `UNROLL` (K = 768 gives 6 vectors at VEC = 8), hence the second loop.
  const int steps = K / (SG_SIZE * VEC);
  const int main_steps = steps - (steps % UNROLL);

  sycl::range<2> global{static_cast<size_t>(total_tokens), static_cast<size_t>(SG_SIZE)};
  sycl::range<2> local{1, static_cast<size_t>(SG_SIZE)};

  q->parallel_for<MoEW4A8ActQuantVec<ScalarT, VEC, UNROLL>>(
      sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(SG_SIZE)]] {
        const int token = static_cast<int>(it.get_global_id(0));
        const int lane = static_cast<int>(it.get_local_id(1));
        const ActVec* row = reinterpret_cast<const ActVec*>(activations + static_cast<size_t>(token) * K);
        QVec* out = reinterpret_cast<QVec*>(qact + static_cast<size_t>(token) * K);

        // One partial maximum per unrolled slot: `fmax` is exact, so merging
        // them below gives the same absmax as a single chain, but the loads no
        // longer wait on it.
        float part_max[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) part_max[u] = 0.0f;

        for (int s = 0; s < main_steps; s += UNROLL) {
          ActVec v[UNROLL];
#pragma unroll
          for (int u = 0; u < UNROLL; ++u) {
            v[u] = row[static_cast<size_t>(s + u) * SG_SIZE + lane];
          }
#pragma unroll
          for (int u = 0; u < UNROLL; ++u) {
#pragma unroll
            for (int e = 0; e < VEC; ++e) {
              const ScalarT a = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(v[u][e]));
              part_max[u] = sycl::fmax(part_max[u], sycl::fabs(static_cast<float>(a)));
            }
          }
        }
        for (int s = main_steps; s < steps; ++s) {
          const ActVec v = row[static_cast<size_t>(s) * SG_SIZE + lane];
#pragma unroll
          for (int e = 0; e < VEC; ++e) {
            const ScalarT a = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(v[e]));
            part_max[0] = sycl::fmax(part_max[0], sycl::fabs(static_cast<float>(a)));
          }
        }

        float local_max = part_max[0];
#pragma unroll
        for (int u = 1; u < UNROLL; ++u) local_max = sycl::fmax(local_max, part_max[u]);

        auto sg = it.get_sub_group();
        const float absmax = sycl::reduce_over_group(sg, local_max, sycl::maximum<float>{});

        const float scale = absmax / kInt8Max;
        const float inv = absmax > 0.0f ? kInt8Max / absmax : 0.0f;
        if (lane == 0) {
          act_quant_write_scale(ascale, token, scale, expert_id_per_token, num_tokens_per_expert, num_experts);
        }

        // Same batching on the way back: the re-read of a row the first pass
        // just touched is served by the cache, but only if enough of it is
        // requested at once.
        for (int s = 0; s < main_steps; s += UNROLL) {
          ActVec v[UNROLL];
#pragma unroll
          for (int u = 0; u < UNROLL; ++u) {
            v[u] = row[static_cast<size_t>(s + u) * SG_SIZE + lane];
          }
#pragma unroll
          for (int u = 0; u < UNROLL; ++u) {
            QVec qv;
#pragma unroll
            for (int e = 0; e < VEC; ++e) {
              const ScalarT a = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(v[u][e]));
              const float x = sycl::rint(static_cast<float>(a) * inv);
              qv[e] = static_cast<int8_t>(sycl::clamp(x, -kInt8Max, kInt8Max));
            }
            out[static_cast<size_t>(s + u) * SG_SIZE + lane] = qv;
          }
        }
        for (int s = main_steps; s < steps; ++s) {
          const ActVec v = row[static_cast<size_t>(s) * SG_SIZE + lane];
          QVec qv;
#pragma unroll
          for (int e = 0; e < VEC; ++e) {
            const ScalarT a = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(v[e]));
            const float x = sycl::rint(static_cast<float>(a) * inv);
            qv[e] = static_cast<int8_t>(sycl::clamp(x, -kInt8Max, kInt8Max));
          }
          out[static_cast<size_t>(s) * SG_SIZE + lane] = qv;
        }
      });
}

// Vectors a lane loads before it consumes any of them. `4` covers every
// shipped shape's `steps` (6 / 12 / 16 / 24 at VEC = 8) with at most a
// two-vector tail. `ARK_MOE_W4A8_ACT_QUANT_UNROLL` selects 1 (the previous
// kernel), 2 or 4; anything else falls back to the default.
constexpr int kActQuantUnrollDefault = 4;

// Partial maxima the single-pass kernel reduces into, matching the two-pass
// kernel's default `UNROLL` so the two produce the same value bit for bit.
constexpr int kActQuantSinglePartials = 4;

// Longest row a lane keeps in registers, in vectors. 16 vectors is 64 dwords
// per lane at VEC = 8 (K = 2048), half the default 128-dword budget; longer
// rows take the two-pass kernel rather than risk a spill.
constexpr int kActQuantSingleMaxSteps = 16;

// Single-pass variant: the row is loaded once into registers, reduced, then
// quantized out of them. `MAX_STEPS` bounds the register array at compile time
// (see the design note above); `steps <= MAX_STEPS` is the caller's contract.
template <typename ScalarT, int VEC, int MAX_STEPS>
void launch_act_dynamic_quant_vec_single(sycl::queue* q, const ScalarT* activations, int8_t* qact, float* ascale,
                                         int total_tokens, int K, int* expert_id_per_token,
                                         const int* num_tokens_per_expert, int num_experts) {
  static_assert(sizeof(ScalarT) == sizeof(uint16_t), "ScalarT must be a 16-bit floating type");
  static_assert(VEC == 4 || VEC == 8, "VEC must be 4 or 8");
  static_assert(MAX_STEPS >= kActQuantSinglePartials, "MAX_STEPS must cover the partial accumulators");
  using ActVec = sycl::vec<uint16_t, VEC>;
  using QVec = sycl::vec<int8_t, VEC>;

  const int steps = K / (SG_SIZE * VEC);

  sycl::range<2> global{static_cast<size_t>(total_tokens), static_cast<size_t>(SG_SIZE)};
  sycl::range<2> local{1, static_cast<size_t>(SG_SIZE)};

  q->parallel_for<MoEW4A8ActQuantSingle<ScalarT, VEC, MAX_STEPS>>(
      sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(SG_SIZE)]] {
        const int token = static_cast<int>(it.get_global_id(0));
        const int lane = static_cast<int>(it.get_local_id(1));
        const ActVec* row = reinterpret_cast<const ActVec*>(activations + static_cast<size_t>(token) * K);
        QVec* out = reinterpret_cast<QVec*>(qact + static_cast<size_t>(token) * K);

        // The whole row, one load per slot, every one issued before the first
        // is consumed. Constant indices under the unroll keep it in registers.
        ActVec v[MAX_STEPS];
#pragma unroll
        for (int s = 0; s < MAX_STEPS; ++s) {
          if (s < steps) {
            v[s] = row[static_cast<size_t>(s) * SG_SIZE + lane];
          }
        }

        float part_max[kActQuantSinglePartials];
#pragma unroll
        for (int u = 0; u < kActQuantSinglePartials; ++u) part_max[u] = 0.0f;

#pragma unroll
        for (int s = 0; s < MAX_STEPS; ++s) {
          if (s < steps) {
#pragma unroll
            for (int e = 0; e < VEC; ++e) {
              const ScalarT a = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(v[s][e]));
              part_max[s % kActQuantSinglePartials] =
                  sycl::fmax(part_max[s % kActQuantSinglePartials], sycl::fabs(static_cast<float>(a)));
            }
          }
        }

        float local_max = part_max[0];
#pragma unroll
        for (int u = 1; u < kActQuantSinglePartials; ++u) local_max = sycl::fmax(local_max, part_max[u]);

        auto sg = it.get_sub_group();
        const float absmax = sycl::reduce_over_group(sg, local_max, sycl::maximum<float>{});

        const float scale = absmax / kInt8Max;
        const float inv = absmax > 0.0f ? kInt8Max / absmax : 0.0f;
        if (lane == 0) {
          act_quant_write_scale(ascale, token, scale, expert_id_per_token, num_tokens_per_expert, num_experts);
        }

        // No second read of the row: it is already here.
#pragma unroll
        for (int s = 0; s < MAX_STEPS; ++s) {
          if (s < steps) {
            QVec qv;
#pragma unroll
            for (int e = 0; e < VEC; ++e) {
              const ScalarT a = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(v[s][e]));
              const float x = sycl::rint(static_cast<float>(a) * inv);
              qv[e] = static_cast<int8_t>(sycl::clamp(x, -kInt8Max, kInt8Max));
            }
            out[static_cast<size_t>(s) * SG_SIZE + lane] = qv;
          }
        }
      });
}

inline int moe_w4a8_act_quant_unroll() {
  const char* env = std::getenv("ARK_MOE_W4A8_ACT_QUANT_UNROLL");
  if (env != nullptr) {
    char* end = nullptr;
    const long long v = std::strtoll(env, &end, 10);
    if (end != env && (v == 1 || v == 2 || v == 4)) return static_cast<int>(v);
  }
  return kActQuantUnrollDefault;
}

// How many k-tiles of A and B the prefill mainloop keeps prefetched ahead of
// the tile it is computing. The prologue issues `prefetch_dist` pairs before
// the first DPAS and the loop then issues one pair per tile, so this is the
// depth of the memory pipeline the mainloop runs against -- too shallow and
// the DPAS waits on L2, too deep and the prefetched lines are evicted before
// use (and the prologue itself becomes a serial stall on short K).
//
// 3 is the value the mainloop was written with and the sibling prefill kernels
// use. The shapes here are short in K (12 k-tiles at K = 768), which is exactly
// where the depth is worth re-measuring, so it is a runtime knob rather than a
// constant; `test_perf_prefill_prefetch_sweep` walks it.
inline constexpr int kPrefillPrefetchDefault = 3;

inline int moe_w4a8_prefill_prefetch_dist() {
  const char* env = std::getenv("ARK_MOE_W4A8_PREFILL_PREFETCH");
  if (env != nullptr) {
    char* end = nullptr;
    const long long v = std::strtoll(env, &end, 10);
    if (end != env && v >= 1 && v <= 8) return static_cast<int>(v);
  }
  return kPrefillPrefetchDefault;
}

// Runtime unroll depth -> compile-time bridge.
template <typename ScalarT, int VEC>
void launch_act_dynamic_quant_vec_unroll(int unroll, sycl::queue* q, const ScalarT* activations, int8_t* qact,
                                         float* ascale, int total_tokens, int K, int* expert_id_per_token,
                                         const int* num_tokens_per_expert, int num_experts) {
  // Register-resident single pass when the row fits, the two-pass kernel
  // otherwise. The smallest rung that covers `steps` is chosen so a short row
  // does not reserve registers for slots it never loads.
  const int steps = K / (SG_SIZE * VEC);
  if (steps <= kActQuantSingleMaxSteps &&
      moe_decode_detail::env_flag_enabled("ARK_MOE_W4A8_ACT_QUANT_SINGLE_PASS", true)) {
    if (steps <= kActQuantSingleMaxSteps / 2) {
      launch_act_dynamic_quant_vec_single<ScalarT, VEC, kActQuantSingleMaxSteps / 2>(
          q, activations, qact, ascale, total_tokens, K, expert_id_per_token, num_tokens_per_expert, num_experts);
    } else {
      launch_act_dynamic_quant_vec_single<ScalarT, VEC, kActQuantSingleMaxSteps>(
          q, activations, qact, ascale, total_tokens, K, expert_id_per_token, num_tokens_per_expert, num_experts);
    }
    return;
  }

  if (unroll == 1) {
    launch_act_dynamic_quant_vec<ScalarT, VEC, 1>(q, activations, qact, ascale, total_tokens, K,
                                                  expert_id_per_token, num_tokens_per_expert, num_experts);
  } else if (unroll == 2) {
    launch_act_dynamic_quant_vec<ScalarT, VEC, 2>(q, activations, qact, ascale, total_tokens, K,
                                                  expert_id_per_token, num_tokens_per_expert, num_experts);
  } else {
    launch_act_dynamic_quant_vec<ScalarT, VEC, 4>(q, activations, qact, ascale, total_tokens, K,
                                                  expert_id_per_token, num_tokens_per_expert, num_experts);
  }
}

template <typename ScalarT>
void launch_act_dynamic_quant(sycl::queue* q, const ScalarT* activations, int8_t* qact, float* ascale,
                              int total_tokens, int K, int* expert_id_per_token = nullptr,
                              const int* num_tokens_per_expert = nullptr, int num_experts = 0) {
  static_assert(sizeof(ScalarT) == sizeof(uint16_t), "ScalarT must be a 16-bit floating type");
  if (total_tokens == 0) return;

  // Widest lane chunk this shape and these buffers support. The alignment
  // checks never fire for torch allocations or the scratch pool (both are at
  // least 256-byte aligned), but a caller-supplied activation view could be
  // offset, and an unaligned `sycl::vec` access would be undefined.
  if (moe_decode_detail::env_flag_enabled("ARK_MOE_W4A8_ACT_QUANT_VEC", true)) {
    const auto act_addr = reinterpret_cast<std::uintptr_t>(activations);
    const auto q_addr = reinterpret_cast<std::uintptr_t>(qact);
    const int unroll = moe_w4a8_act_quant_unroll();
    if (K % (SG_SIZE * 8) == 0 && act_addr % (8 * sizeof(ScalarT)) == 0 && q_addr % 8 == 0) {
      launch_act_dynamic_quant_vec_unroll<ScalarT, 8>(unroll, q, activations, qact, ascale, total_tokens, K,
                                                      expert_id_per_token, num_tokens_per_expert, num_experts);
      return;
    }
    if (K % (SG_SIZE * 4) == 0 && act_addr % (4 * sizeof(ScalarT)) == 0 && q_addr % 4 == 0) {
      launch_act_dynamic_quant_vec_unroll<ScalarT, 4>(unroll, q, activations, qact, ascale, total_tokens, K,
                                                      expert_id_per_token, num_tokens_per_expert, num_experts);
      return;
    }
  }

  sycl::range<2> global{static_cast<size_t>(total_tokens), static_cast<size_t>(SG_SIZE)};
  sycl::range<2> local{1, static_cast<size_t>(SG_SIZE)};

  q->parallel_for<MoEW4A8ActQuant<ScalarT>>(
      sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(SG_SIZE)]] {
        const int token = static_cast<int>(it.get_global_id(0));
        const int lane = static_cast<int>(it.get_local_id(1));
        const ScalarT* row = activations + static_cast<size_t>(token) * K;
        int8_t* out = qact + static_cast<size_t>(token) * K;

        float local_max = 0.0f;
        for (int k = lane; k < K; k += SG_SIZE) {
          local_max = sycl::fmax(local_max, sycl::fabs(static_cast<float>(row[k])));
        }
        auto sg = it.get_sub_group();
        const float absmax = sycl::reduce_over_group(sg, local_max, sycl::maximum<float>{});

        const float scale = absmax / kInt8Max;
        const float inv = absmax > 0.0f ? kInt8Max / absmax : 0.0f;
        if (lane == 0) {
          act_quant_write_scale(ascale, token, scale, expert_id_per_token, num_tokens_per_expert, num_experts);
        }

        for (int k = lane; k < K; k += SG_SIZE) {
          const float v = sycl::rint(static_cast<float>(row[k]) * inv);
          out[k] = static_cast<int8_t>(sycl::clamp(v, -kInt8Max, kInt8Max));
        }
      });
}

// ---------------------------------------------------------------------------
// AUTO_S8 stage 1: per-(expert, output channel, block) re-scale factor.
//
//   sxt[e][n][j] = max_{g in block j} |s[e][n][g]| * (8 / 127)
//
// Verbatim port of the rescale branch of `packscale` in `xpu_wrapper.hpp`
// (with `fullrange = 8` for int4). An all-zero block yields `sxt = 0`; stage 2
// turns that into all-zero int8 weights, so the (equally zero) product is
// still exact.
// ---------------------------------------------------------------------------
template <typename ScalarT>
void launch_weight_scale_reduce(sycl::queue* q, const ScalarT* scales, float* wscale_out, int E, int N, int K,
                                int group_size, int rescale_block, int nblk) {
  const int groups_k = K / group_size;
  const int groups_per_block = rescale_block / group_size;

  q->parallel_for<MoEW4A8ScaleReduce<ScalarT>>(
      sycl::range<1>(static_cast<size_t>(E) * static_cast<size_t>(N) * static_cast<size_t>(nblk)),
      [=](sycl::id<1> id) {
        const size_t idx = id[0];
        const int blk = static_cast<int>(idx % static_cast<size_t>(nblk));
        const size_t row = idx / static_cast<size_t>(nblk);  // e * N + n
        const ScalarT* s_row =
            scales + row * static_cast<size_t>(groups_k) + static_cast<size_t>(blk) * groups_per_block;

        float absmax = 0.0f;
        for (int g = 0; g < groups_per_block; ++g) {
          absmax = sycl::fmax(absmax, sycl::fabs(static_cast<float>(s_row[g])));
        }
        wscale_out[idx] = absmax * (kInt4FullRange / kInt8Max);
      });
}

// ---------------------------------------------------------------------------
// AUTO_S8 stage 2: int4 -> int8 re-scale.
//
//   w8[k] = round(w4[k] * s[k / group_size] / sxt[k / rescale_block])
//
// Verbatim port of the `CfgDequantS8Rescale` branch of `unpackq` in
// `xpu_wrapper.hpp`. One work-item decodes one 32-bit word (8 nibbles); the
// shape gate guarantees `group_size % 8 == 0` and `rescale_block % 8 == 0`, so
// all 8 K indices of a word share the same group scale and the same block
// scale and both loads hoist out of the inner loop.
// ---------------------------------------------------------------------------
template <typename ScalarT>
void launch_weight_rescale_s4_to_s8(sycl::queue* q, const uint8_t* weights, const ScalarT* scales,
                                    const float* wscale, int8_t* w8_out, int E, int N, int K, int group_size,
                                    int rescale_block, int nblk) {
  const int groups_k = K / group_size;
  const int octets = K / kPrepackOctet;

  q->parallel_for<MoEW4A8Repack<ScalarT>>(
      sycl::range<2>(static_cast<size_t>(E) * static_cast<size_t>(N), static_cast<size_t>(octets)),
      [=](sycl::id<2> id) {
        const size_t row = id[0];  // e * N + n
        const int oct = static_cast<int>(id[1]);
        const int k_base = oct * kPrepackOctet;

        const uint8_t* w_ptr = weights + row * static_cast<size_t>(K / 2) + static_cast<size_t>(oct) * 4;
        const uint32_t word = *reinterpret_cast<const uint32_t*>(w_ptr);
        int q4[kPrepackOctet];
        decode_int4_octet<false>(word, q4);

        const float s = static_cast<float>(scales[row * static_cast<size_t>(groups_k) + k_base / group_size]);
        const float sx = wscale[row * static_cast<size_t>(nblk) + k_base / rescale_block];
        const float f = sx > 0.0f ? s / sx : 0.0f;

        int8_t* out = w8_out + row * static_cast<size_t>(K) + k_base;
#pragma unroll
        for (int j = 0; j < kPrepackOctet; ++j) {
          const float v = sycl::rint(static_cast<float>(q4[j]) * f);
          out[j] = static_cast<int8_t>(sycl::clamp(v, -kInt8Max, kInt8Max));
        }
      });
}

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
// therefore stops taking it (see `moe_w4a8_prefill_dispatch`): it has no
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
// Prefill driver: policy selection on the average per-expert M.
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
// `ARK_MOE_W4A8_PREFILL_TILE` overrides the choice with an explicit `MxN` tile
// (`8x128`, `64x128`, `128x128`, `128x256`, `256x128`, `256x256`); anything
// else -- including the default `auto` -- keeps the ladder. It exists so the
// tile can be swept on hardware without a rebuild.
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
template <typename ElementD>
void moe_w4a8_prefill_dispatch(sycl::queue* q, const int8_t* qact, const float* ascale, const int8_t* weights,
                               const float* wscale, ElementD* outputs, const int* num_tokens_per_expert, int E,
                               int N, int K, int blocksize, int blks, int total_tokens,
                               MoEFusedReduce reduce = MoEFusedReduce{}) {
  if (E == 0 || N == 0 || K == 0 || total_tokens == 0) return;

  compat::set_default_queue(*q);

  const int A_avg_M = total_tokens / E;
  const bool tile_n_256 = (N % 256) == 0;
  const bool allow_full_tile = moe_decode_detail::env_flag_enabled("ARK_MOE_W4A8_PREFILL_FULL_TILE", true);
  const bool store_2d_aligned = !reduce.enabled() && (static_cast<size_t>(N) * sizeof(ElementD)) % 64 == 0 &&
                                reinterpret_cast<uintptr_t>(outputs) % 64 == 0;
  const bool allow_block_2d_store =
      store_2d_aligned && moe_decode_detail::env_flag_enabled("ARK_MOE_W4A8_PREFILL_STORE_2D", true);
  const int prefetch_dist = moe_w4a8_prefill_prefetch_dist();
  int32_t* atomic_buffer = moe_dpas_fp8::get_persistent_atomic_buffer(q);

#define ARK_MOE_W4A8_LAUNCH(policy)                                                                        \
  MoEGEMMLauncher_w4a8<policy, ElementD>(*q, qact, weights, ascale, wscale, outputs, N, K,                  \
                                         num_tokens_per_expert, E, blocksize, blks, allow_full_tile,       \
                                         allow_block_2d_store, prefetch_dist, reduce, atomic_buffer);

  const char* tile_env = std::getenv("ARK_MOE_W4A8_PREFILL_TILE");
  if (tile_env != nullptr) {
    if (std::strcmp(tile_env, "8x128") == 0) {
      ARK_MOE_W4A8_LAUNCH(w4a8_policy_m_8)
      return;
    } else if (std::strcmp(tile_env, "64x128") == 0) {
      ARK_MOE_W4A8_LAUNCH(w4a8_policy_m_64)
      return;
    } else if (std::strcmp(tile_env, "128x128") == 0) {
      ARK_MOE_W4A8_LAUNCH(w4a8_policy_m_128)
      return;
    } else if (std::strcmp(tile_env, "128x256") == 0) {
      ARK_MOE_W4A8_LAUNCH(w4a8_policy_m_128_n256)
      return;
    } else if (std::strcmp(tile_env, "256x128") == 0) {
      ARK_MOE_W4A8_LAUNCH(w4a8_policy_m_256_n128)
      return;
    } else if (std::strcmp(tile_env, "256x256") == 0) {
      ARK_MOE_W4A8_LAUNCH(w4a8_policy_large)
      return;
    }
  }

  if (A_avg_M < 16) {
    ARK_MOE_W4A8_LAUNCH(w4a8_policy_m_8)
  } else if (A_avg_M < 128) {
    ARK_MOE_W4A8_LAUNCH(w4a8_policy_m_64)
  } else if (tile_n_256) {
    ARK_MOE_W4A8_LAUNCH(w4a8_policy_m_128_n256)
  } else {
    ARK_MOE_W4A8_LAUNCH(w4a8_policy_m_128)
  }
#undef ARK_MOE_W4A8_LAUNCH
}

// ---------------------------------------------------------------------------
// Decode GEMV: int8 x int8 -> int32, one output column per sub-group lane.
//
// Same work decomposition as `moe_decode_detail::launch_int8` (work-group =
// one sub-group covering 16 consecutive N columns of one token), with the
// per-K-group float dequantization replaced by a per-block int32 dot product.
// Two accumulators hide the multiply-add latency; int32 cannot overflow here
// (|a|,|w| <= 127 gives < 2^14 per product, so K would have to exceed 130k).
// ---------------------------------------------------------------------------
template <typename ElementD>
void launch_w4a8_decode(sycl::queue* q, const int8_t* qact, const float* ascale, const int8_t* weights,
                        const float* wscale, ElementD* outputs, const int* expert_id_per_token, int total_tokens,
                        int N, int K, int blocksize, int blks) {
  if (N % N_TILE != 0) {
    throw std::invalid_argument("moe_gemm_w4a8(decode): N must be a multiple of 16");
  }
  if (total_tokens == 0) return;

  const int n_tiles = N / N_TILE;
  sycl::range<2> global{static_cast<size_t>(total_tokens), static_cast<size_t>(n_tiles * SG_SIZE)};
  sycl::range<2> local{1, static_cast<size_t>(SG_SIZE)};

  q->parallel_for<MoEW4A8DecodeGemv<ElementD>>(
      sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(SG_SIZE)]] {
        const int token = static_cast<int>(it.get_global_id(0));
        const int n_tile = static_cast<int>(it.get_group(1));
        const int lane = static_cast<int>(it.get_local_id(1));
        const int n_global = n_tile * N_TILE + lane;

        const int expert = expert_id_per_token[token];
        const int8_t* act_row = qact + static_cast<size_t>(token) * K;
        const int8_t* w_row = weights + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * K;
        const float* s_row =
            wscale + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * blks;

        constexpr int SUB = 16;
        using QVec = sycl::vec<int8_t, SUB>;

        float accf = 0.0f;
        for (int ib = 0; ib < blks; ++ib) {
          const int k_base = ib * blocksize;
          int acc0 = 0;
          int acc1 = 0;
          int kk = 0;
          const int end = (blocksize / SUB) * SUB;
          for (; kk < end; kk += SUB) {
            const QVec av = *reinterpret_cast<const QVec*>(act_row + k_base + kk);
            const QVec wv = *reinterpret_cast<const QVec*>(w_row + k_base + kk);
#pragma unroll
            for (int u = 0; u < SUB; u += 2) {
              acc0 += static_cast<int>(av[u]) * static_cast<int>(wv[u]);
              acc1 += static_cast<int>(av[u + 1]) * static_cast<int>(wv[u + 1]);
            }
          }
          for (; kk < blocksize; ++kk) {
            acc0 += static_cast<int>(act_row[k_base + kk]) * static_cast<int>(w_row[k_base + kk]);
          }
          accf += static_cast<float>(acc0 + acc1) * s_row[ib];
        }

        outputs[static_cast<size_t>(token) * N + n_global] = static_cast<ElementD>(accf * ascale[token]);
      });
}

// ---------------------------------------------------------------------------
// Decode GEMV, K-split lane mapping (default) -- one sub-group per output
// element, lanes splitting K, plus N-blocking over `NCOLS` columns.
//
// `launch_w4a8_decode` above maps one *work-item* to one output element, so a
// lane walks a whole `[n_global, K]` int8 weight row on its own. That is the
// same mapping the FP8 decode GEMV started from, and it costs the same two
// things on a kernel that does exactly one multiply-add per weight byte:
//
//   1. Weight loads are not coalesced. Lanes `l` and `l+1` read bytes `K`
//      apart, so each 16-byte load turns into 16 scattered cache-line
//      requests. No DRAM byte is wasted (each lane consumes its lines as it
//      walks the row), but the memory controller sees 16 independent streams
//      per sub-group -- the pattern DRAM row buffers handle worst.
//   2. The grid is small: `total_tokens * N / 16` sub-groups, i.e. 768 SIMD16
//      threads for a Qwen3-MoE batch-1 step (8 routed rows, N = 1536). That is
//      below the thread slots of a BMG-class GPU (1280 on a B60), so there are
//      never enough loads in flight to cover DRAM latency.
//
// This kernel transposes the mapping exactly like `launch_fp8_ksplit`: a whole
// sub-group cooperates on one output element and lane `l` owns the `CH`
// consecutive K elements at `l * CH` inside each `SG_SIZE * CH`-wide K tile.
// One instruction then covers `SG_SIZE * CH` *contiguous* weight bytes (256 B =
// four full cache lines at the default `CH = 16`, 512 B at `CH = 32`) and the
// same span of int8 activations, every thread walks a single sequential stream,
// and the sub-group count grows by `SG_SIZE` (12288 for that batch-1 step). The
// price is one `reduce_over_group` per output element -- a handful of shuffles
// against `K` multiply-adds.
//
// On top of that the sub-group blocks N: it owns `NCOLS` consecutive columns
// and reuses one activation load for all of them, which cuts activation
// messages per weight chunk to `1 / NCOLS` and multiplies the number of
// independent weight loads in flight by `NCOLS` (see
// `moe_w4a8_decode_ksplit_ncols`).
//
// Numerics are equivalent, not bit-identical: the int32 partial sums are still
// folded to float once per AUTO_S8 block with that block's scale, but they are
// split across the 16 lanes and summed at the end. Integer addition is exact
// and associative, so the *integer* partition is lossless; only the float
// accumulation is reordered (per lane, then across lanes, instead of one lane
// folding every block in sequence), which can differ from the legacy result by
// a rounding step. A lane's chunk is `CH` consecutive K elements starting at a
// multiple of `CH`, and the shape gate requires the block to be a multiple of
// `CH`, so a chunk never straddles two blocks.
// ---------------------------------------------------------------------------

// K elements a lane owns per step: `KSPLIT_CH_DEFAULT` is one 16-byte int8
// weight load and one 16-byte int8 activation load, the same transactions the
// legacy GEMV issues. `ARK_MOE_W4A8_DECODE_KSPLIT_CH=32` doubles them to
// 32-byte loads, which halves the number of memory messages per byte and
// doubles the bytes a thread keeps in flight; it costs GRF (2 x NCOLS chunks
// live at once) and needs `blocksize >= SG_SIZE * CH`.
//
// Measured (`test_perf_decode_config_sweep`, BMG, 8 routed rows, bf16 act), at
// the default NCOLS: 284.0 -> 278.9 GB/s (qwen3 up), 280.1 -> 244.4 (qwen3
// down), 268.1 -> 259.9 (minimax up), 315.5 -> 308.7 (minimax down). The wider
// chunk never wins at any NCOLS and costs up to 13%, so 16 stays the default
// and 32 stays an opt-in sweep point.
constexpr int KSPLIT_CH_DEFAULT = 16;
constexpr int KSPLIT_CH_MAX = 32;
// Sub-groups per work-group. Each owns `NCOLS` output columns, so a work-group
// covers `KSPLIT_WG_SGS * NCOLS` consecutive columns.
constexpr int KSPLIT_WG_SGS = N_TILE;
// `NCOLS = 2` is the measured default: it is the fastest configuration on two
// of the four swept shapes and within 2% of the best (`4` on qwen3 down, `1` on
// minimax up) on the other two, while `1` costs 47% on qwen3 up and `4` costs
// 14% on minimax up.
constexpr int KSPLIT_NCOLS_DEFAULT = 2;
constexpr int KSPLIT_NCOLS_MAX = 4;

// A lane's `CH`-byte chunk, as a register type.
//
// `sycl::vec` only exists for 1, 2, 3, 4, 8 and 16 elements, so a `CH = 32`
// chunk cannot be spelled `sycl::vec<int8_t, 32>`: that instantiation is a hard
// static_assert in the SYCL headers ("Invalid number of elements for
// sycl::vec"). A chunk is therefore an aggregate of `CH / 16` 16-byte
// sub-vectors covering *consecutive* bytes. The lane still reads one contiguous
// `CH`-byte span at a `CH`-aligned address (`moe_w4a8_decode_ksplit_shape_ok`
// makes every chunk offset a multiple of `CH` off a row base that is a multiple
// of `K`, itself a multiple of `CH`), the sub-vectors are adjacent both in
// memory and in the GRF, and the declared alignment lets IGC fold the pair back
// into a single wider message. At the default `CH = 16` the aggregate holds a
// single `sycl::vec<int8_t, 16>`, i.e. exactly the load this kernel issued
// before.
//
// `operator[]` is only ever called from the fully unrolled inner loops, so the
// sub-vector selection folds away at compile time and no dynamic indexing
// (which would push the chunk out to scratch) is generated.
template <int CH>
struct alignas(CH) QChunk {
  static constexpr int kSub = KSPLIT_CH_DEFAULT;
  static_assert(CH % kSub == 0, "chunk width must be a whole number of 16-byte sub-vectors");
  sycl::vec<int8_t, kSub> v[CH / kSub];

  int8_t operator[](int i) const { return v[i / kSub][i % kSub]; }
};

// `ARK_MOE_W4A8_DECODE_KSPLIT` (default ON). Setting it to "0" / "false" /
// "off" / "no" forces the legacy per-lane-strided GEMV, for A/B comparison and
// as a regression escape. Re-read on every call so benchmarks can toggle the
// path in-process.
inline bool moe_w4a8_decode_ksplit_enabled() {
  return moe_decode_detail::env_flag_enabled("ARK_MOE_W4A8_DECODE_KSPLIT", true);
}

// Per-lane chunk width in K elements (= bytes). 16 or 32; anything else falls
// back to the default.
inline int moe_w4a8_decode_ksplit_chunk() {
  const char* env = std::getenv("ARK_MOE_W4A8_DECODE_KSPLIT_CH");
  if (env != nullptr) {
    char* end = nullptr;
    const long long v = std::strtoll(env, &end, 10);
    if (end != env && (v == 16 || v == 32)) return static_cast<int>(v);
  }
  return KSPLIT_CH_DEFAULT;
}

// Shape gate. `blocksize >= SG_SIZE * ch` keeps every lane of the sub-group
// busy: below that some lanes own no chunk in a block and only pay the
// reduction, which is the one regime where splitting K cannot pay for itself.
// `blocksize % ch == 0` combined with `K % blocksize == 0` also makes every
// chunk offset a multiple of `ch` off a row base that is a multiple of `K`, so
// the vector loads stay naturally aligned. The resolved AUTO_S8 block is always
// a multiple of 64 that divides K, so the conditions hold for every shipped
// configuration and only very fine re-scale blocks fall back to the legacy
// GEMV.
inline bool moe_w4a8_decode_ksplit_shape_ok(int N, int K, int blocksize, int ch = KSPLIT_CH_DEFAULT) {
  if (N % N_TILE != 0) return false;
  if (blocksize < SG_SIZE * ch) return false;
  if (blocksize % ch != 0) return false;
  if (K % blocksize != 0) return false;
  return true;
}

// N-blocking factor. A work-group covers `KSPLIT_WG_SGS * ncols` columns, so
// `ncols` shrinks until it tiles N. `ARK_MOE_W4A8_DECODE_KSPLIT_NCOLS`
// overrides the default (1, 2 or 4); `NCOLS == 1` reproduces the plain K-split
// mapping instruction for instruction.
inline int moe_w4a8_decode_ksplit_ncols(int N) {
  int ncols = KSPLIT_NCOLS_DEFAULT;
  const char* env = std::getenv("ARK_MOE_W4A8_DECODE_KSPLIT_NCOLS");
  if (env != nullptr) {
    char* end = nullptr;
    const long long v = std::strtoll(env, &end, 10);
    if (end != env && v >= 1 && v <= KSPLIT_NCOLS_MAX && (v & (v - 1)) == 0) {
      ncols = static_cast<int>(v);
    }
  }
  while (ncols > 1 && (N % (KSPLIT_WG_SGS * ncols)) != 0) ncols /= 2;
  return ncols;
}

template <typename ElementD, int NCOLS, int CH = KSPLIT_CH_DEFAULT>
void launch_w4a8_decode_ksplit(sycl::queue* q, const int8_t* qact, const float* ascale, const int8_t* weights,
                               const float* wscale, ElementD* outputs, const int* expert_id_per_token,
                               int total_tokens, int N, int K, int blocksize, int blks) {
  static_assert(NCOLS >= 1 && (NCOLS & (NCOLS - 1)) == 0, "NCOLS must be a power of two");
  static_assert(CH == 16 || CH == KSPLIT_CH_MAX, "CH must be 16 or 32");
  // K elements a sub-group covers per step -- the contiguous span its 16 lanes
  // read in one instruction.
  constexpr int STEP = SG_SIZE * CH;
  if (!moe_w4a8_decode_ksplit_shape_ok(N, K, blocksize, CH) || (N % (KSPLIT_WG_SGS * NCOLS)) != 0) {
    throw std::invalid_argument("moe_gemm_w4a8(decode): K-split GEMV called on an unsupported shape");
  }
  if (total_tokens == 0) return;

  // One sub-group per (token, NCOLS columns); `KSPLIT_WG_SGS` of them per
  // work-group.
  sycl::range<2> global{static_cast<size_t>(total_tokens), static_cast<size_t>(N / NCOLS) * SG_SIZE};
  sycl::range<2> local{1, static_cast<size_t>(KSPLIT_WG_SGS * SG_SIZE)};

  q->parallel_for<MoEW4A8DecodeKSplit<ElementD, NCOLS, CH>>(
      sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(SG_SIZE)]] {
        const auto sg = it.get_sub_group();
        const int token = static_cast<int>(it.get_global_id(0));
        const int local_id = static_cast<int>(it.get_local_id(1));
        // The work-group is one row of `KSPLIT_WG_SGS * SG_SIZE` work-items, so
        // sub-group index and lane index are the halves of the local id.
        const int lane = local_id % SG_SIZE;
        const int n_base = (static_cast<int>(it.get_group(1)) * KSPLIT_WG_SGS + local_id / SG_SIZE) * NCOLS;

        const int expert = expert_id_per_token[token];
        const int8_t* act_row = qact + static_cast<size_t>(token) * K;
        const size_t row0 = static_cast<size_t>(expert) * N + static_cast<size_t>(n_base);
        const int8_t* w_rows[NCOLS];
        const float* s_rows[NCOLS];
#pragma unroll
        for (int c = 0; c < NCOLS; ++c) {
          w_rows[c] = weights + (row0 + static_cast<size_t>(c)) * K;
          s_rows[c] = wscale + (row0 + static_cast<size_t>(c)) * blks;
        }

        using QVec = QChunk<CH>;

        float acc[NCOLS];
#pragma unroll
        for (int c = 0; c < NCOLS; ++c) acc[c] = 0.0f;

        for (int ib = 0; ib < blks; ++ib) {
          const int block_begin = ib * blocksize;
          const int block_end = block_begin + blocksize;
          int32_t iacc[NCOLS];
#pragma unroll
          for (int c = 0; c < NCOLS; ++c) iacc[c] = 0;

          int k0 = block_begin + lane * CH;
          // Two chunks per iteration: their loads are independent, so the pair
          // doubles the weight requests a thread keeps in flight. All
          // `2 * NCOLS` weight loads are issued before the first is consumed.
          for (; k0 + STEP + CH <= block_end; k0 += 2 * STEP) {
            const QVec av0 = *reinterpret_cast<const QVec*>(act_row + k0);
            const QVec av1 = *reinterpret_cast<const QVec*>(act_row + k0 + STEP);
            QVec wv0[NCOLS], wv1[NCOLS];
#pragma unroll
            for (int c = 0; c < NCOLS; ++c) {
              wv0[c] = *reinterpret_cast<const QVec*>(w_rows[c] + k0);
              wv1[c] = *reinterpret_cast<const QVec*>(w_rows[c] + k0 + STEP);
            }
#pragma unroll
            for (int c = 0; c < NCOLS; ++c) {
              int p0 = 0;
              int p1 = 0;
#pragma unroll
              for (int u = 0; u < CH; u += 2) {
                p0 += static_cast<int>(av0[u]) * static_cast<int>(wv0[c][u]);
                p1 += static_cast<int>(av0[u + 1]) * static_cast<int>(wv0[c][u + 1]);
                p0 += static_cast<int>(av1[u]) * static_cast<int>(wv1[c][u]);
                p1 += static_cast<int>(av1[u + 1]) * static_cast<int>(wv1[c][u + 1]);
              }
              iacc[c] += p0 + p1;
            }
          }
          // Tail: the chunk of a lane whose partner a full step away falls
          // outside the block. At most one chunk per lane.
          for (; k0 < block_end; k0 += STEP) {
            const QVec av = *reinterpret_cast<const QVec*>(act_row + k0);
#pragma unroll
            for (int c = 0; c < NCOLS; ++c) {
              const QVec wv = *reinterpret_cast<const QVec*>(w_rows[c] + k0);
              int p0 = 0;
              int p1 = 0;
#pragma unroll
              for (int u = 0; u < CH; u += 2) {
                p0 += static_cast<int>(av[u]) * static_cast<int>(wv[u]);
                p1 += static_cast<int>(av[u + 1]) * static_cast<int>(wv[u + 1]);
              }
              iacc[c] += p0 + p1;
            }
          }

#pragma unroll
          for (int c = 0; c < NCOLS; ++c) acc[c] += static_cast<float>(iacc[c]) * s_rows[c][ib];
        }

        const float sa = ascale[token];
#pragma unroll
        for (int c = 0; c < NCOLS; ++c) {
          const float total = sycl::reduce_over_group(sg, acc[c], sycl::plus<float>{});
          if (lane == 0) {
            outputs[static_cast<size_t>(token) * N + n_base + c] = static_cast<ElementD>(total * sa);
          }
        }
      });
}

// Runtime (NCOLS, CH) -> compile-time bridge, plus the K-split / legacy choice.
// `CH = 32` needs a block of at least 512 elements, so it silently falls back to
// 16 on shapes it cannot serve rather than dropping to the legacy GEMV.
template <typename ElementD>
void launch_w4a8_decode_dispatch(sycl::queue* q, const int8_t* qact, const float* ascale, const int8_t* weights,
                                 const float* wscale, ElementD* outputs, const int* expert_id_per_token,
                                 int total_tokens, int N, int K, int blocksize, int blks) {
  if (moe_w4a8_decode_ksplit_enabled() && moe_w4a8_decode_ksplit_shape_ok(N, K, blocksize)) {
    const int ncols = moe_w4a8_decode_ksplit_ncols(N);
    const int ch = moe_w4a8_decode_ksplit_chunk() == KSPLIT_CH_MAX &&
                           moe_w4a8_decode_ksplit_shape_ok(N, K, blocksize, KSPLIT_CH_MAX)
                       ? KSPLIT_CH_MAX
                       : KSPLIT_CH_DEFAULT;

#define ARK_MOE_W4A8_KSPLIT(ncols_v, ch_v)                                                                        \
  launch_w4a8_decode_ksplit<ElementD, ncols_v, ch_v>(q, qact, ascale, weights, wscale, outputs,                    \
                                                     expert_id_per_token, total_tokens, N, K, blocksize, blks);   \
  return;

    if (ch == KSPLIT_CH_MAX) {
      switch (ncols) {
        case 4:
          ARK_MOE_W4A8_KSPLIT(4, KSPLIT_CH_MAX)
        case 2:
          ARK_MOE_W4A8_KSPLIT(2, KSPLIT_CH_MAX)
        default:
          ARK_MOE_W4A8_KSPLIT(1, KSPLIT_CH_MAX)
      }
    }
    switch (ncols) {
      case 4:
        ARK_MOE_W4A8_KSPLIT(4, KSPLIT_CH_DEFAULT)
      case 2:
        ARK_MOE_W4A8_KSPLIT(2, KSPLIT_CH_DEFAULT)
      default:
        ARK_MOE_W4A8_KSPLIT(1, KSPLIT_CH_DEFAULT)
    }
#undef ARK_MOE_W4A8_KSPLIT
  }
  launch_w4a8_decode<ElementD>(q, qact, ascale, weights, wscale, outputs, expert_id_per_token, total_tokens, N, K,
                               blocksize, blks);
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------

// Resolve the effective AUTO_S8 re-scale block size.
//
// `requested <= 0` (the `group=-1` spelling) or any value that is not a valid
// block size falls back to `K`, i.e. one scale per output channel -- the
// maximum-efficiency shape. `ARK_MOE_W4A8_AUTO_S8` overrides the argument so
// benchmarks can sweep the block size without touching the caller.
inline int moe_w4a8_rescale_block_size(int K, int group_size, int requested) {
  int v = requested;
  const char* env = std::getenv("ARK_MOE_W4A8_AUTO_S8");
  if (env != nullptr) {
    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end != env) v = static_cast<int>(parsed);
  }
  if (K <= 0) return K;
  if (v <= 0 || v >= K) return K;
  if (group_size > 0 && (v < group_size || v % group_size != 0)) return K;
  if (K % v != 0) return K;
  // The mainloop slices each block into 64-wide DPAS K tiles.
  if (v % 64 != 0) return K;
  return v;
}

// Shape preconditions shared by the prepack, prefill and decode paths.
inline bool moe_w4a8_shape_ok(int N, int K, int group_size) {
  if (N <= 0 || K <= 0 || group_size <= 0) return false;
  if (N % N_TILE != 0) return false;
  if (K % 64 != 0) return false;
  if (group_size % kPrepackOctet != 0) return false;
  if (K % group_size != 0) return false;
  return true;
}

// Token count at or below which the auto phase selection picks the decode
// GEMV. Mirrors `ARK_MOE_AUTO_DECODE_MAX_TOKENS` used by the Python `moe()`
// dispatcher; overridable with `ARK_MOE_W4A8_DECODE_MAX_TOKENS`.
inline int moe_w4a8_decode_max_tokens() {
  const char* env = std::getenv("ARK_MOE_W4A8_DECODE_MAX_TOKENS");
  if (env == nullptr) return 128;
  char* end = nullptr;
  const long parsed = std::strtol(env, &end, 10);
  if (end == env || parsed < 0) return 128;
  return static_cast<int>(parsed);
}

inline void moe_w4a8_release_scratch() {
  qact_pool().release_all();
  expert_map_pool().release_all();
}

}  // namespace moe_w4a8

// The public `ark::` entry points below are thin wrappers emitted by the
// generated `sycl_tla_moe_w4a8.cpp` translation unit (MOE_SOURCE_MODE
// 19); they live in their own namespace here so the header stays free of
// external definitions and only that one TU pays the kernel compile cost.
namespace moe_w4a8_detail {

// ---------------------------------------------------------------------------
// Public entry point 1 -- one-shot AUTO_S8 prepack.
//
// Converts auto-round's packed int4-sym weights + per-group scales into the
// int8 weights + FP32 block scales the W4A8 kernels consume. Callers are
// expected to run this once per checkpoint and keep the results resident.
// ---------------------------------------------------------------------------
inline void moe_w4a8_prepack(sycl::queue* q, void* weights_s4, void* scales, void* weights_s8, void* wscales,
                             BTLA_DTYPE act_dtype, int num_experts, int N, int K, int group_size,
                             int rescale_group_size) {
  if (num_experts <= 0) return;
  if (!moe_w4a8::moe_w4a8_shape_ok(N, K, group_size)) {
    throw std::invalid_argument(
        "moe_w4a8_prepack: unsupported shape (need N % 16 == 0, K % 64 == 0, "
        "group_size % 8 == 0 and K % group_size == 0)");
  }
  if (weights_s4 == nullptr || scales == nullptr || weights_s8 == nullptr || wscales == nullptr) {
    throw std::invalid_argument("moe_w4a8_prepack: null buffer");
  }

  const int blocksize = moe_w4a8::moe_w4a8_rescale_block_size(K, group_size, rescale_group_size);
  const int blks = K / blocksize;

  if (act_dtype == BTLA_DTYPE::F16) {
    using ScalarT = sycl::half;
    moe_w4a8::launch_weight_scale_reduce<ScalarT>(q, static_cast<const ScalarT*>(scales),
                                                  static_cast<float*>(wscales), num_experts, N, K, group_size,
                                                  blocksize, blks);
    moe_w4a8::launch_weight_rescale_s4_to_s8<ScalarT>(
        q, static_cast<const uint8_t*>(weights_s4), static_cast<const ScalarT*>(scales),
        static_cast<const float*>(wscales), static_cast<int8_t*>(weights_s8), num_experts, N, K, group_size,
        blocksize, blks);
  } else if (act_dtype == BTLA_DTYPE::BF16) {
    using ScalarT = sycl::ext::oneapi::bfloat16;
    moe_w4a8::launch_weight_scale_reduce<ScalarT>(q, static_cast<const ScalarT*>(scales),
                                                  static_cast<float*>(wscales), num_experts, N, K, group_size,
                                                  blocksize, blks);
    moe_w4a8::launch_weight_rescale_s4_to_s8<ScalarT>(
        q, static_cast<const uint8_t*>(weights_s4), static_cast<const ScalarT*>(scales),
        static_cast<const float*>(wscales), static_cast<int8_t*>(weights_s8), num_experts, N, K, group_size,
        blocksize, blks);
  } else {
    throw std::invalid_argument("moe_w4a8_prepack: act_dtype must be F16 or BF16");
  }
}

// ---------------------------------------------------------------------------
// Public entry point 2 -- W4A8 MoE GEMM (prefill + decode).
//
// `phase`: 0 = auto (decode when `total_tokens <=
// ARK_MOE_W4A8_DECODE_MAX_TOKENS`), 1 = force decode GEMV, 2 = force prefill
// grouped GEMM.
//
// Two optional call contracts trade interface work for DRAM traffic. Both are
// opt-in and the defaults are unchanged.
//
// Pre-quantized activations (`qact_in` + `ascale_in`)
// ---------------------------------------------------
// By default the call quantizes `[T, K]` itself: it reads the 16-bit
// activations, writes an int8 copy and the GEMM reads that copy back, i.e.
// `4 * T * K` bytes on top of the GEMM's own operands. On the down-projection
// that is 27% of everything the call moves -- and it is redundant, because the
// producer of those activations (the SiLU/gate elementwise kernel) already
// writes `[T, K]` once and could write int8 plus a per-row scale instead: the
// absmax it needs is a reduction over the row it is already holding. When both
// pointers are supplied all three streams disappear, along with a kernel
// launch. `ascale_in` is `[T]` fp32, `scale = absmax / 127`, matching what
// `launch_act_dynamic_quant` writes.
//
// Fused top-k reduction (`row_to_token` + `routing_weights` + `fused_out`)
// -----------------------------------------------------------------------
// See `MoEFusedReduce`. Prefill only, and the accumulator must be zeroed by
// the caller; `outputs` is then unused and may be null.
// ---------------------------------------------------------------------------
inline void moe_gemm_w4a8(sycl::queue* q, void* activations, void* weights_s8, void* wscales, void* outputs,
                          BTLA_DTYPE act_dtype, int N, int K, int rescale_block_size,
                          int* num_tokens_per_expert, int num_experts, int total_tokens, int phase,
                          const void* qact_in = nullptr, const float* ascale_in = nullptr,
                          const int* row_to_token = nullptr, const float* routing_weights = nullptr,
                          float* fused_out = nullptr, int fused_batch = 0) {
  if (total_tokens == 0 || num_experts <= 0) return;
  if (N % moe_w4a8::N_TILE != 0) {
    throw std::invalid_argument("moe_gemm_w4a8: N must be a multiple of 16");
  }
  if (K % 64 != 0) {
    throw std::invalid_argument("moe_gemm_w4a8: K must be a multiple of 64");
  }
  if (rescale_block_size <= 0 || rescale_block_size > K || K % rescale_block_size != 0 ||
      rescale_block_size % 64 != 0) {
    throw std::invalid_argument(
        "moe_gemm_w4a8: rescale_block_size must be a multiple of 64 that divides K "
        "(use moe_w4a8_rescale_block_size to resolve it)");
  }
  if (act_dtype != BTLA_DTYPE::F16 && act_dtype != BTLA_DTYPE::BF16) {
    throw std::invalid_argument("moe_gemm_w4a8: act_dtype must be F16 or BF16");
  }

  const bool prequantized = qact_in != nullptr || ascale_in != nullptr;
  if (prequantized && (qact_in == nullptr || ascale_in == nullptr)) {
    throw std::invalid_argument("moe_gemm_w4a8: pre-quantized activations need both qact and ascale");
  }
  if (!prequantized && activations == nullptr) {
    throw std::invalid_argument("moe_gemm_w4a8: null activations");
  }

  const bool fused_reduce = row_to_token != nullptr || routing_weights != nullptr || fused_out != nullptr;
  if (fused_reduce && (row_to_token == nullptr || routing_weights == nullptr || fused_out == nullptr ||
                       fused_batch <= 0)) {
    throw std::invalid_argument(
        "moe_gemm_w4a8: the fused top-k reduction needs row_to_token, routing_weights, a zeroed [batch, N] "
        "fp32 output and batch > 0");
  }
  if (!fused_reduce && outputs == nullptr) {
    throw std::invalid_argument("moe_gemm_w4a8: null outputs");
  }

  const int blocksize = rescale_block_size;
  const int blks = K / blocksize;

  const bool use_decode =
      phase == 1 || (phase != 2 && total_tokens <= moe_w4a8::moe_w4a8_decode_max_tokens());

  if (fused_reduce && use_decode) {
    throw std::invalid_argument("moe_gemm_w4a8: the fused top-k reduction is prefill-only");
  }

  const int8_t* qact = static_cast<const int8_t*>(qact_in);
  const float* ascale = ascale_in;
  int8_t* qact_scratch = nullptr;
  float* ascale_scratch = nullptr;

  if (!prequantized) {
    // Quantized activations + per-token scales share one slab: `[T, K]` int8
    // followed by `[T]` fp32 (the int8 region is already 4-byte aligned because
    // K is a multiple of 64).
    const size_t qact_bytes = static_cast<size_t>(total_tokens) * static_cast<size_t>(K);
    const size_t scale_offset = (qact_bytes + sizeof(float) - 1) / sizeof(float) * sizeof(float);
    const size_t slab_bytes = scale_offset + static_cast<size_t>(total_tokens) * sizeof(float);
    uint8_t* slab = moe_w4a8::qact_pool().acquire(q, slab_bytes);
    qact_scratch = reinterpret_cast<int8_t*>(slab);
    ascale_scratch = reinterpret_cast<float*>(slab + scale_offset);
    qact = qact_scratch;
    ascale = ascale_scratch;
  }

  // Decode consumes `expert_id_per_token`; the activation-quant kernel already
  // runs one sub-group per token, so it derives the map as well instead of
  // paying for a second launch (`fill_expert_id_per_token`) on a timeline where
  // one call is issued per generated token. Prefill passes nullptr and the scan
  // is not compiled into the work. With pre-quantized activations that kernel
  // does not run at all, so decode falls back to the standalone scan.
  int* expert_map = nullptr;
  if (use_decode) {
    expert_map = reinterpret_cast<int*>(
        moe_w4a8::expert_map_pool().acquire(q, static_cast<size_t>(total_tokens) * sizeof(int)));
  }

  if (prequantized) {
    if (use_decode) {
      moe_decode_detail::fill_expert_id_per_token(q, expert_map, num_tokens_per_expert, num_experts,
                                                  total_tokens);
    }
  } else if (act_dtype == BTLA_DTYPE::F16) {
    moe_w4a8::launch_act_dynamic_quant<sycl::half>(q, static_cast<const sycl::half*>(activations), qact_scratch,
                                                   ascale_scratch, total_tokens, K, expert_map,
                                                   num_tokens_per_expert, num_experts);
  } else {
    using BF = sycl::ext::oneapi::bfloat16;
    moe_w4a8::launch_act_dynamic_quant<BF>(q, static_cast<const BF*>(activations), qact_scratch, ascale_scratch,
                                           total_tokens, K, expert_map, num_tokens_per_expert, num_experts);
  }

  const auto* weights = static_cast<const int8_t*>(weights_s8);
  const auto* wscale = static_cast<const float*>(wscales);

  if (use_decode) {
    if (act_dtype == BTLA_DTYPE::F16) {
      moe_w4a8::launch_w4a8_decode_dispatch<sycl::half>(q, qact, ascale, weights, wscale,
                                                        static_cast<sycl::half*>(outputs), expert_map,
                                                        total_tokens, N, K, blocksize, blks);
    } else {
      using BF = sycl::ext::oneapi::bfloat16;
      moe_w4a8::launch_w4a8_decode_dispatch<BF>(q, qact, ascale, weights, wscale, static_cast<BF*>(outputs),
                                                expert_map, total_tokens, N, K, blocksize, blks);
    }
    return;
  }

  moe_w4a8::MoEFusedReduce reduce{};
  if (fused_reduce) {
    reduce.row_to_token = row_to_token;
    reduce.row_weight = routing_weights;
    reduce.out = fused_out;
    reduce.batch = fused_batch;
  }

  if (act_dtype == BTLA_DTYPE::F16) {
    moe_w4a8::moe_w4a8_prefill_dispatch<sycl::half>(q, qact, ascale, weights, wscale,
                                                    static_cast<sycl::half*>(outputs), num_tokens_per_expert,
                                                    num_experts, N, K, blocksize, blks, total_tokens, reduce);
  } else {
    using BF = sycl::ext::oneapi::bfloat16;
    moe_w4a8::moe_w4a8_prefill_dispatch<BF>(q, qact, ascale, weights, wscale, static_cast<BF*>(outputs),
                                            num_tokens_per_expert, num_experts, N, K, blocksize, blks,
                                            total_tokens, reduce);
  }
}

// Resolve the effective AUTO_S8 block size (host helper, also exported to
// Python so callers can size the `wscales` tensor consistently).
inline int moe_w4a8_rescale_block_size(int K, int group_size, int rescale_group_size) {
  return moe_w4a8::moe_w4a8_rescale_block_size(K, group_size, rescale_group_size);
}

// Free the W4A8 activation-quantization / expert-map scratch slabs.
inline void moe_w4a8_release_scratch() { moe_w4a8::moe_w4a8_release_scratch(); }

}  // namespace moe_w4a8_detail

}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
