// SYCL MoE W4A8 -- cutlass-free kernels (activation quant, prepack, decode)
//
// Split out of `sycl_tla_moe_w4a8.hpp` so the translation units that only need
// these kernels never parse the cutlass-sycl / CuTe include set. Nothing here
// touches CuTe: the activation-quantization, AUTO_S8 prepack and decode-GEMV
// kernels are plain SYCL, and only the grouped prefill GEMM needs DPAS.
//
// The narrative for each kernel -- the message-width / unroll / single-pass
// choices for the quant pass, the AUTO_S8 re-scale, and the decode K-split
// lane mapping -- stays inline with the code below. See
// `sycl_tla_moe_w4a8.hpp` for the overall W4A8 design.
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

// `env_flag_enabled`, `fill_expert_id_per_token`, `SG_SIZE` / `N_TILE`, and
// the shared nibble decoders. Cutlass-free.
#include "sycl_tla_moe_decode.hpp"
// Scratch pools, host-side helpers and the parameter structs the launch
// wrappers below are expressed in terms of.
#include "sycl_tla_moe_w4a8_helpers.hpp"

namespace ark {
namespace moe_w4a8 {

using moe_dequant::decode_int4_octet;

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

// Both prepack kernels for one dtype, behind the shared parameter struct so a
// generated TU can define `prepack_f16` / `prepack_bf16` in three lines.
template <typename ScalarT>
void moe_w4a8_prepack_launch(const moe_w4a8_detail::W4A8PrepackParams& p) {
  launch_weight_scale_reduce<ScalarT>(p.q, static_cast<const ScalarT*>(p.scales), p.wscales, p.num_experts, p.N,
                                      p.K, p.group_size, p.blocksize, p.blks);
  launch_weight_rescale_s4_to_s8<ScalarT>(p.q, static_cast<const uint8_t*>(p.weights_s4),
                                          static_cast<const ScalarT*>(p.scales), p.wscales, p.weights_s8,
                                          p.num_experts, p.N, p.K, p.group_size, p.blocksize, p.blks);
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

// Activation quantization for one dtype, behind the shared parameter struct.
template <typename ScalarT>
void moe_w4a8_quant_launch(const moe_w4a8_detail::W4A8QuantParams& p) {
  launch_act_dynamic_quant<ScalarT>(p.q, static_cast<const ScalarT*>(p.activations), p.qact, p.ascale,
                                    p.total_tokens, p.K, p.expert_map, p.num_tokens_per_expert, p.num_experts);
}

// Decode GEMV for one dtype, behind the shared parameter struct. The K-split
// vs. legacy choice and the (NCOLS, CH) specialization stay inside the TU.
template <typename ElementD>
void moe_w4a8_decode_launch(const moe_w4a8_detail::W4A8DecodeParams& p) {
  launch_w4a8_decode_dispatch<ElementD>(p.q, p.qact, p.ascale, p.weights, p.wscale,
                                        static_cast<ElementD*>(p.outputs), p.expert_id_per_token, p.total_tokens,
                                        p.N, p.K, p.blocksize, p.blks);
}


}  // namespace moe_w4a8
}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
