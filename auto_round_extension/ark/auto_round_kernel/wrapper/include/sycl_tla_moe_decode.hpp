// SYCL MoE Decode Kernel
//
// GEMV-style MoE kernel optimized for the decode phase, where each expert
// typically processes only 1-2 tokens (top-k routing with batch size 1).
//
// Layout convention (caller already sorted activations per expert,
// identical to the prefill `moe_gemm` interface):
//   - activations:           [total_tokens, K]            row-major
//   - weights (fp/bf16):     [num_experts, N, K]          row-major
//   - weights (int8):        [num_experts, N, K]          row-major, one
//                            int8 per byte (sym: signed -128..127;
//                            asym: unsigned 0..255 with zero-point)
//   - weights (int4 packed): [num_experts, N, K/2]        row-major, two
//                            4-bit values per byte (low nibble at lower K).
//                            The scalar-GEMV fallback repacks this on-device
//                            into an N-tiled [E, N/16, K/2, 16] layout so that
//                            sub-group weight loads are coalesced; the external
//                            [E, N, K/2] contract is unchanged.
//   - weights (int2 packed): [num_experts, N, K/4]        row-major, four
//                            2-bit values per byte (field j at K index
//                            4*i+j is bits [2j+1:2j])
//   - weights (fp8):         [num_experts, N, K]          row-major, one
//                            FP8 byte per weight (E4M3 / E5M2); scales
//                            applied per-group, no zero-points
//   - scales:                [num_experts, N, K/group_size]
//   - zeros (asym only):     [num_experts, N, K/group_size]
//   - num_tokens_per_expert: [num_experts]                int32
//   - outputs:               [total_tokens, N]
//
// Target: Intel BMG (Xe2), sub_group_size = 16. One sub-group per (token, N-tile)
// with N_TILE == SG_SIZE: each lane independently computes one output element,
// so no cross-lane reduction is needed and activation reads are coalesced across
// the sub-group through the L1 cache.
//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "bestla/bestla.h"
#include "sycl_tla_moe_dequant.hpp"
// S4-sym per-group DPAS grouped-GEMM (shared with the prefill path). The
// header self-guards on `ARK_XPU && ARK_SYCL_TLA`, so including it here is a
// no-op when the DPAS backend is disabled. Decode routes small-M int4-sym
// GEMV through this kernel; see `moe_gemm_decode` below.
#include "sycl_tla_moe_prefill_s4_dpas.hpp"

#ifdef ARK_XPU
#include <sycl/sycl.hpp>
#endif

// ----------------------------------------------------------------------------
// FP8 decode implementation switch (runtime)
//
// FP8 weight bytes can be dequantized either via inline bit manipulation or
// via the 128-entry magnitude LUT in `bestla/sycl/fp8_lut.h` (sign applied
// separately). Both paths are mathematically equivalent for finite values;
// pick whichever is faster on the target hardware.
//
// Selection is done at runtime through the environment variable
// `ARK_FP8_DECODE_USE_LUT`:
//   - unset / "1" / "true" / "on" / "yes" (case-insensitive) -> LUT path (default)
//   - "0" / "false" / "off" / "no" (case-insensitive)        -> inline bit-manip
//
// The env var is read once on the host (cached) and passed as a template
// parameter into the SYCL kernel, so there is no per-element runtime branch.
// The actual primitives live in `sycl_tla_moe_dequant.hpp` (shared with the
// mixed-input prefill path); this file just re-exports them via `using`.
// ----------------------------------------------------------------------------

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

namespace ark {
namespace moe_decode_detail {

constexpr int SG_SIZE = 16;
constexpr int N_TILE = SG_SIZE;  // one output element per sub-group lane

// Token-blocking factor for the coalesced int4 decode GEMV. A work-item that
// owns one (n_tile, lane) output column processes up to TOKEN_BLOCK consecutive
// tokens, loading each packed weight byte from the (expert, n_tile) tile once
// and applying it to every token in the block that routes to the same expert.
// When decode routing is bursty (runs of tokens hitting the same expert), this
// amortizes the dominant weight traffic across the block instead of re-reading
// the tile once per token, moving the problem from pure GEMV toward GEMM. A
// value of 1 reproduces the one-token-per-work-item behaviour exactly.
constexpr int TOKEN_BLOCK = 4;

// ----------------------------------------------------------------------------
// Kernel name tags (one per specialization, required for SYCL kernel naming)
// ----------------------------------------------------------------------------
template <typename ScalarT>
class MoEDecodeKernelFP;

template <typename ScalarT, bool Asym>
class MoEDecodeKernelInt4;

template <typename ScalarT, bool Asym>
class MoEDecodeKernelInt4Coalesced;

template <typename ScalarT, bool Asym>
class MoEDecodeRepackInt4;

template <typename ScalarT, bool Asym>
class MoEDecodeKernelInt8;

template <typename ScalarT, bool Asym>
class MoEDecodeKernelInt2;

template <typename ScalarT, bool IsE4M3, bool UseLut>
class MoEDecodeKernelFP8;

// ----------------------------------------------------------------------------
// FP8 weight dequantization primitives + host-side env-var reader live in
// `sycl_tla_moe_dequant.hpp` so the prefill (mixed-input Grouped GEMM) and
// decode (GEMV) paths share one definition. The `using` declarations below
// keep the in-kernel call sites (`decode_fp8<...>(byte)`) and the host-side
// `fp8_decode_use_lut()` lookup inside `moe_decode_detail` working unchanged.
// ----------------------------------------------------------------------------
using moe_dequant::decode_fp8;
using moe_dequant::decode_fp8_e4m3_bits;
using moe_dequant::decode_fp8_e4m3_lut;
using moe_dequant::decode_fp8_e5m2_bits;
using moe_dequant::decode_fp8_e5m2_lut;
using moe_dequant::decode_int2_quad;
using moe_dequant::decode_int4_pair;
using moe_dequant::decode_int8;
using moe_dequant::fp8_decode_use_lut;

// ----------------------------------------------------------------------------
// Build a [total_tokens] -> expert_id mapping from num_tokens_per_expert.
// Runs on host (num_experts is small, total_tokens is small in decode).
// Caller-managed buffer (USM device allocation) keeps host noise out of the
// hot path; here we just fill it via a tiny SYCL kernel for simplicity.
// ----------------------------------------------------------------------------
inline void fill_expert_id_per_token(sycl::queue* q, int* expert_id_per_token,
                                     const int* num_tokens_per_expert, int num_experts,
                                     int total_tokens) {
  // Parallel fill: each work-item independently scans the small
  // num_tokens_per_expert array (typ. <= 256) to find its expert id. This
  // removes the single-task serialization point and avoids an explicit
  // host-device sync; the in-order queue chains this with the GEMV launch.
  if (total_tokens == 0) return;
  q->parallel_for(sycl::range<1>(static_cast<size_t>(total_tokens)), [=](sycl::id<1> idx) {
    const int i = static_cast<int>(idx[0]);
    int offset = 0;
    int expert = num_experts - 1;
    for (int e = 0; e < num_experts; ++e) {
      const int n = num_tokens_per_expert[e];
      if (i < offset + n) {
        expert = e;
        break;
      }
      offset += n;
    }
    expert_id_per_token[i] = expert;
  });
}

// ----------------------------------------------------------------------------
// FP16 / BF16 baseline GEMV (no quantization).
// ----------------------------------------------------------------------------
template <typename ScalarT>
void launch_fp(sycl::queue* q, const ScalarT* activations, const ScalarT* weights, ScalarT* outputs,
               const int* expert_id_per_token, int total_tokens, int N, int K) {
  if (N % N_TILE != 0) {
    throw std::invalid_argument("moe_gemm_decode: N must be a multiple of 16");
  }
  if (total_tokens == 0) return;

  const int n_tiles = N / N_TILE;
  sycl::range<2> global{static_cast<size_t>(total_tokens), static_cast<size_t>(n_tiles * SG_SIZE)};
  sycl::range<2> local{1, static_cast<size_t>(SG_SIZE)};

  q->parallel_for<MoEDecodeKernelFP<ScalarT>>(
       sycl::nd_range<2>(global, local),
       [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(SG_SIZE)]] {
         const int token = static_cast<int>(it.get_global_id(0));
         const int n_tile = static_cast<int>(it.get_group(1));
         const int lane = static_cast<int>(it.get_local_id(1));
         const int n_global = n_tile * N_TILE + lane;

         const int expert = expert_id_per_token[token];
         const ScalarT* act_row = activations + static_cast<size_t>(token) * K;
         const ScalarT* w_row =
             weights + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * K;

         float acc = 0.0f;
         // Unroll by 8 with a 16-byte vector load for both activations and
         // weights. Activations are sub-group-uniform so they coalesce via
         // L1; each lane's weight load is an independent 16-byte transaction.
         // We load through a uint16_t vector to stay portable across SYCL
         // implementations that may not provide sycl::vec<bfloat16, N>.
         int k = 0;
         constexpr int VEC = 8;
         using LoadVec = sycl::vec<uint16_t, VEC>;
         static_assert(sizeof(ScalarT) == sizeof(uint16_t),
                       "ScalarT must be a 16-bit floating type");
         const int k_vec_end = (K / VEC) * VEC;
         for (; k < k_vec_end; k += VEC) {
           const LoadVec av = *reinterpret_cast<const LoadVec*>(act_row + k);
           const LoadVec wv = *reinterpret_cast<const LoadVec*>(w_row + k);
#pragma unroll
           for (int u = 0; u < VEC; ++u) {
             const ScalarT a = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[u]));
             const ScalarT w = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(wv[u]));
             acc += static_cast<float>(a) * static_cast<float>(w);
           }
         }
         for (; k < K; ++k) {
           acc += static_cast<float>(act_row[k]) * static_cast<float>(w_row[k]);
         }

         outputs[static_cast<size_t>(token) * N + n_global] = static_cast<ScalarT>(acc);
       });
}

// ----------------------------------------------------------------------------
// INT4 (S4_CLIP) GEMV with group-wise dequantization.
//
// Asym=false: signed nibble in [-8, 7], dequant = q * scale
// Asym=true : unsigned nibble in [0, 15], dequant = (q - zero) * scale
//
// Packing: two 4-bit values per byte; the value at k = 2*i is the LOW nibble
// of byte i, the value at k = 2*i+1 is the HIGH nibble. This matches the
// existing CPU/XPU `packq` layout for S4_CLIP weights.
// ----------------------------------------------------------------------------

// Vectorized inner accumulation over CHUNK consecutive K elements (CHUNK/2
// packed weight bytes + a vec<ScalarT,CHUNK> activation block). Templated on
// CHUNK so the caller can run a wide (32) stage first and a narrower (16)
// stage for the remainder, which keeps the fast path active for group sizes
// that are a multiple of 32 (32/64/128/256 -- the shipped quant configs)
// without regressing group_size == 16 (which drops straight to the 16-wide
// stage).
//
// Both sym and asym decode the nibbles through the *same* unsigned path.
// Measurements on MiniMax-M2 decode shapes showed int4-sym ~1.9x slower than
// int4-asym in this very kernel even though sym does strictly fewer floating
// point operations. The only difference was the per-nibble sign extension
// (`(int8_t)(byte << 4) >> 4`) -- a serial shift/narrow/shift chain per nibble
// that defeats the byte-wise vectorization the asym mask+shift form gets. The
// sym decode is therefore expressed with the standard sign-flip identity
//
//     signed_nibble == (unsigned_nibble ^ 8) - 8
//
// so XOR-ing the packed byte with `0x88` (flipping the sign bit of *both*
// nibbles at once, on the whole loaded vector register) turns sym into exactly
// the asym computation with a constant zero-point of 8. The decoded integers
// are bit-identical to the sign-extending decode for all 256 byte values, so
// the only change is that sym now accumulates the biased sum and subtracts
// `8 * sum a` at the end -- exactly the fp32 accumulation pattern asym has
// always used, and well inside the kernel's existing quantization tolerance.
//
// The per-group scale and zero-point are NOT applied here: this accumulates the
// raw integer-weighted dot product into `acc_q0`/`acc_q1` and the plain
// activation sum into `acc_a` (now needed by both modes, since sym carries the
// constant zero-point of 8). The caller folds the group's scale/zero in once
// (sum a*((q-z)*s) == s*(sum a*q - z*sum a)). Hoisting the scale removes one
// float multiply per K element on the decode hot path, and because the fold is
// exact-once per group the result stays well within the kernel's existing
// quantization tolerance.
//
// Two independent partial accumulators (``acc_q0``/``acc_q1``) break the
// single fp32 dependency chain so the FMA pipeline is not latency-bound; the
// caller reduces the pair. ``acc_a`` reuses the same split.
template <typename ScalarT, bool Asym, int CHUNK>
static inline void int4_decode_chunk(const ScalarT* act_ptr, const uint8_t* w_ptr, float& acc_q0, float& acc_q1,
                                     float& acc_a) {
  static_assert(sizeof(ScalarT) == sizeof(uint16_t), "ScalarT must be a 16-bit floating type");
  static_assert(CHUNK % 16 == 0, "CHUNK must be a multiple of 16");
  // sycl::vec only supports widths of 1, 2, 3, 4, 8 or 16, so a single
  // vec<uint16_t, 32> load is illegal. Process the chunk in 16-wide sub-blocks
  // (16 activations + 8 packed weight bytes each), which keeps CHUNK == 32
  // valid while reusing the same code path for CHUNK == 16.
  constexpr int SUB = 16;
  using ActVec = sycl::vec<uint16_t, SUB>;
  using PackVec = sycl::vec<uint8_t, SUB / 2>;
#pragma unroll
  for (int s = 0; s < CHUNK / SUB; ++s) {
    const ActVec av = *reinterpret_cast<const ActVec*>(act_ptr + s * SUB);
    PackVec pv = *reinterpret_cast<const PackVec*>(w_ptr + s * (SUB / 2));
    if constexpr (!Asym) {
      // Sign-flip the whole packed vector in one vector XOR so the sym nibbles
      // can be decoded by the (vectorizable) unsigned path below; the constant
      // zero-point of 8 is folded by the caller.
      pv = pv ^ PackVec(static_cast<uint8_t>(0x88));
    }
#pragma unroll
    for (int b = 0; b < SUB / 2; ++b) {
      int q0, q1;
      // Always the unsigned decode: asym nibbles are unsigned by definition and
      // sym nibbles were biased by the XOR above.
      decode_int4_pair<true>(pv[b], q0, q1);
      const ScalarT a0 = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[2 * b]));
      const ScalarT a1 = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[2 * b + 1]));
      const float fa0 = static_cast<float>(a0);
      const float fa1 = static_cast<float>(a1);
      acc_q0 += fa0 * static_cast<float>(q0);
      acc_q1 += fa1 * static_cast<float>(q1);
      acc_a += fa0 + fa1;
    }
  }
}

template <typename ScalarT, bool Asym>
void launch_int4(sycl::queue* q, const ScalarT* activations, const uint8_t* weights, const ScalarT* scales,
                 const ScalarT* zeros, ScalarT* outputs, const int* expert_id_per_token, int total_tokens, int N,
                 int K, int group_size) {
  if (N % N_TILE != 0) {
    throw std::invalid_argument("moe_gemm_decode(int4): N must be a multiple of 16");
  }
  if (K % group_size != 0 || (group_size & 1) != 0) {
    throw std::invalid_argument("moe_gemm_decode(int4): K must be a multiple of group_size and group_size must be even");
  }
  if (Asym && zeros == nullptr) {
    throw std::invalid_argument("moe_gemm_decode(int4): zeros pointer required when asym=true");
  }
  if (total_tokens == 0) return;

  const int n_tiles = N / N_TILE;
  const int num_groups_k = K / group_size;
  const int k_packed = K / 2;  // bytes of packed weight per (expert, n)

  sycl::range<2> global{static_cast<size_t>(total_tokens), static_cast<size_t>(n_tiles * SG_SIZE)};
  sycl::range<2> local{1, static_cast<size_t>(SG_SIZE)};

  q->parallel_for<MoEDecodeKernelInt4<ScalarT, Asym>>(
       sycl::nd_range<2>(global, local),
       [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(SG_SIZE)]] {
         const int token = static_cast<int>(it.get_global_id(0));
         const int n_tile = static_cast<int>(it.get_group(1));
         const int lane = static_cast<int>(it.get_local_id(1));
         const int n_global = n_tile * N_TILE + lane;

         const int expert = expert_id_per_token[token];
         const ScalarT* act_row = activations + static_cast<size_t>(token) * K;

         const uint8_t* w_row =
             weights + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * k_packed;
         const ScalarT* s_row =
             scales + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * num_groups_k;
         const ScalarT* z_row = Asym
             ? zeros + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * num_groups_k
             : nullptr;

         float acc = 0.0f;
         for (int g = 0; g < num_groups_k; ++g) {
           const float scale = static_cast<float>(s_row[g]);
           // Sym uses the constant zero-point of 8 that the `^0x88` sign-flip
           // in the decode introduces, so both modes run the identical fold.
           // `if constexpr` keeps the null `z_row` out of the sym instantiation.
           float zero = 8.0f;
           if constexpr (Asym) {
             zero = static_cast<float>(z_row[g]);
           }
           const int k_base = g * group_size;
           // Vectorized ladder: process 32 K-elements at a time (16 packed
           // weight bytes + vec<ScalarT,32> activation block), then a 16-wide
           // stage for the remainder, then a scalar tail. Widening the first
           // stage to 32 amortizes the per-group scale load and loop overhead
           // across twice as many multiply-adds for the shipped group sizes
           // (32/64/128/256), while the 16-wide stage keeps group_size == 16
           // on the fast path.
           //
           // The scale and zero are constant across the group, so the wide
           // stages accumulate the raw integer-weighted dot product
           // ``Σ a·q`` (split across two partial accumulators to break the
           // fp32 dependency chain) plus ``Σ a``, and the fold below applies
           // the scale/zero exactly once per group, identically for both modes:
           //   acc += scale * ((acc_q0 + acc_q1) - zero * acc_a)
           // with `zero` == the per-group zero-point (asym) or the constant 8
           // that the `^0x88` sign-flip decode introduces (sym).
           float acc_q0 = 0.0f;
           float acc_q1 = 0.0f;
           float acc_a = 0.0f;
           int kk = 0;
           constexpr int CHUNK32 = 32;
           const int end32 = (group_size / CHUNK32) * CHUNK32;
           for (; kk < end32; kk += CHUNK32) {
             int4_decode_chunk<ScalarT, Asym, CHUNK32>(act_row + k_base + kk, w_row + (k_base + kk) / 2, acc_q0,
                                                       acc_q1, acc_a);
           }
           constexpr int CHUNK16 = 16;
           const int end16 = kk + ((group_size - kk) / CHUNK16) * CHUNK16;
           for (; kk < end16; kk += CHUNK16) {
             int4_decode_chunk<ScalarT, Asym, CHUNK16>(act_row + k_base + kk, w_row + (k_base + kk) / 2, acc_q0,
                                                       acc_q1, acc_a);
           }
           // Scalar tail for group_size not divisible by 16. Uses the same
           // raw-accumulation convention as the wide stages so the single
           // scale/zero fold below stays valid.
           for (; kk < group_size; kk += 2) {
             uint8_t packed = w_row[(k_base + kk) / 2];
             if constexpr (!Asym) packed ^= static_cast<uint8_t>(0x88);
             int q0, q1;
             decode_int4_pair<true>(packed, q0, q1);
             const float fa0 = static_cast<float>(act_row[k_base + kk]);
             const float fa1 = static_cast<float>(act_row[k_base + kk + 1]);
             acc_q0 += fa0 * static_cast<float>(q0);
             acc_q1 += fa1 * static_cast<float>(q1);
             acc_a += fa0 + fa1;
           }
           acc += scale * ((acc_q0 + acc_q1) - zero * acc_a);
         }

         outputs[static_cast<size_t>(token) * N + n_global] = static_cast<ScalarT>(acc);
       });
}

// ----------------------------------------------------------------------------
// INT4 (S4_CLIP) coalesced-load GEMV.
//
// The scalar `launch_int4` above is memory-bandwidth-bound: for a single decode
// token it just streams the whole packed weight matrix once with ~1 MAC per
// byte, so the arithmetic tweaks (split accumulators, hoisted scale) cannot
// help. Its real cost is that weight loads are *not coalesced across the
// sub-group*: with the `[E, N, K/2]` (K-contiguous) layout, lane `l` and lane
// `l+1` of a sub-group read packed bytes `K/2` apart at a fixed `k`, so each
// step issues 16 scattered transactions instead of one contiguous cache line.
//
// This path fixes that by first repacking the weights on-device into an
// N-tiled layout `[E, N/16, K/2, 16]`: the trailing dim of 16 holds one packed
// byte for each of the 16 columns owned by a sub-group tile, so at a fixed
// packed-byte index the 16 lanes read 16 contiguous bytes -> a single coalesced
// load. The dequant math is byte-for-byte identical to `launch_int4` (same
// `decode_int4_pair`, same per-group scale/zero fold), only the weight memory
// access pattern changes. The repack buffer is a transient USM device
// allocation freed after the queue drains; the caller's `[E, N, K/2]` weight
// contract is unchanged.
//
// The trailing lane stride means each lane's own K-bytes are 16 apart, so the
// vectorized `int4_decode_chunk` (contiguous per-lane load) does not apply
// here; the inner loop reads one packed byte per lane per step, which the
// hardware coalesces across the sub-group into one wide transaction.
//
// On top of coalescing, this path blocks tokens: each work-item owns one
// output column but processes up to `TOKEN_BLOCK` consecutive tokens. For each
// distinct expert appearing in the block it makes a single weight-streaming
// pass and reuses every loaded (coalesced) byte across all tokens in the block
// routed to that expert. When decode routing is bursty -- runs of tokens
// hitting the same expert -- this amortizes the dominant weight traffic across
// the block (GEMV -> small GEMM). Fully-scattered routing degrades gracefully
// to one pass per token with the same per-pass weight reads as before, so the
// result is bit-identical to the one-token-per-work-item kernel regardless of
// routing.
// ----------------------------------------------------------------------------
template <typename ScalarT, bool Asym>
void launch_int4_coalesced(sycl::queue* q, const ScalarT* activations, const uint8_t* weights,
                           const ScalarT* scales, const ScalarT* zeros, ScalarT* outputs,
                           const int* expert_id_per_token, int total_tokens, int N, int K, int group_size,
                           int num_experts) {
  if (N % N_TILE != 0) {
    throw std::invalid_argument("moe_gemm_decode(int4): N must be a multiple of 16");
  }
  if (K % group_size != 0 || (group_size & 1) != 0) {
    throw std::invalid_argument("moe_gemm_decode(int4): K must be a multiple of group_size and group_size must be even");
  }
  if (Asym && zeros == nullptr) {
    throw std::invalid_argument("moe_gemm_decode(int4): zeros pointer required when asym=true");
  }
  if (total_tokens == 0) return;

  const int n_tiles = N / N_TILE;
  const int num_groups_k = K / group_size;
  const int k_packed = K / 2;  // bytes of packed weight per (expert, n)

  const size_t repacked_bytes =
      static_cast<size_t>(num_experts) * static_cast<size_t>(n_tiles) *
      static_cast<size_t>(k_packed) * static_cast<size_t>(N_TILE);
  uint8_t* repacked = sycl::malloc_device<uint8_t>(repacked_bytes, *q);
  if (repacked == nullptr) {
    throw std::runtime_error("moe_gemm_decode(int4): failed to allocate repack buffer");
  }

  // Repack kernel: one work-item per (expert, column, packed byte). The write
  // index places the 16 columns of a tile contiguously in the trailing dim.
  {
    sycl::range<3> rp_global{static_cast<size_t>(num_experts), static_cast<size_t>(N),
                             static_cast<size_t>(k_packed)};
    q->parallel_for<MoEDecodeRepackInt4<ScalarT, Asym>>(rp_global, [=](sycl::id<3> id) {
      const int e = static_cast<int>(id[0]);
      const int n = static_cast<int>(id[1]);
      const int kb = static_cast<int>(id[2]);
      const int t = n / N_TILE;
      const int l = n % N_TILE;
      const size_t src = (static_cast<size_t>(e) * N + static_cast<size_t>(n)) * k_packed + kb;
      const size_t dst =
          ((static_cast<size_t>(e) * n_tiles + t) * k_packed + kb) * N_TILE + l;
      repacked[dst] = weights[src];
    });
  }

  sycl::range<2> global{static_cast<size_t>((total_tokens + TOKEN_BLOCK - 1) / TOKEN_BLOCK),
                        static_cast<size_t>(n_tiles * SG_SIZE)};
  sycl::range<2> local{1, static_cast<size_t>(SG_SIZE)};

  q->parallel_for<MoEDecodeKernelInt4Coalesced<ScalarT, Asym>>(
       sycl::nd_range<2>(global, local),
       [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(SG_SIZE)]] {
         const int token_base = static_cast<int>(it.get_global_id(0)) * TOKEN_BLOCK;
         const int n_tile = static_cast<int>(it.get_group(1));
         const int lane = static_cast<int>(it.get_local_id(1));
         const int n_global = n_tile * N_TILE + lane;

         // Number of tokens this work-item owns (last block may be short).
         int block = TOKEN_BLOCK;
         if (token_base + block > total_tokens) {
           block = total_tokens - token_base;
         }

         // Experts routed by each token in the block. The tile weight byte is
         // loaded once per k-step and reused only for tokens whose expert
         // matches the byte's owning expert, so blocking tokens that share an
         // expert amortizes the dominant weight traffic; tokens with a
         // different expert contribute nothing from this pass and are handled
         // by the pass whose leader expert matches theirs.
         int experts[TOKEN_BLOCK];
         for (int b = 0; b < block; ++b) {
           experts[b] = expert_id_per_token[token_base + b];
         }

         // Which distinct experts appear in this block. For each we make one
         // weight-streaming pass, reusing every loaded byte across all tokens
         // in the block routed to that expert. Bursty routing collapses to a
         // single pass; fully-scattered routing degrades to one pass per token
         // (i.e. the previous behaviour) with no extra weight reads per pass.
         for (int lead = 0; lead < block; ++lead) {
           const int expert = experts[lead];
           // Skip experts already streamed by an earlier token in this block.
           bool seen = false;
           for (int p = 0; p < lead; ++p) {
             if (experts[p] == expert) {
               seen = true;
               break;
             }
           }
           if (seen) continue;

           // Base of this (expert, n_tile) weight tile in the repacked buffer.
           // Layout [E, N/16, K/2, 16]; this lane reads byte kb at
           // w_tile[kb*16 + lane], so adjacent lanes read adjacent bytes.
           const uint8_t* w_tile =
               repacked + ((static_cast<size_t>(expert) * n_tiles + n_tile) * k_packed) * N_TILE;
           const ScalarT* s_row =
               scales + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * num_groups_k;
           const ScalarT* z_row = Asym
               ? zeros + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * num_groups_k
               : nullptr;

           // Compact the tokens routed to `expert` into a dense member list
           // once per pass. Hoisting the routing filter out of the hot k-loop
           // removes a per-(kb, token) branch and lets the compiler keep the
           // per-member activation base pointers in registers; the numerics
           // are identical to the previous per-kb `experts[b] != expert`
           // filter.
           int members[TOKEN_BLOCK];
           const ScalarT* act_rows[TOKEN_BLOCK];
           int nmembers = 0;
           for (int b = 0; b < block; ++b) {
             if (experts[b] != expert) continue;
             members[nmembers] = b;
             act_rows[nmembers] = activations + static_cast<size_t>(token_base + b) * K;
             ++nmembers;
           }

           float acc[TOKEN_BLOCK];
           for (int m = 0; m < nmembers; ++m) acc[m] = 0.0f;

           for (int g = 0; g < num_groups_k; ++g) {
             const float scale = static_cast<float>(s_row[g]);
             // Constant zero-point of 8 for sym (see `int4_decode_chunk`): the
             // `^0x88` sign-flip lets sym reuse the asym unsigned decode and
             // fold, so both modes emit the identical instruction stream.
             float zero = 8.0f;
             if constexpr (Asym) {
               zero = static_cast<float>(z_row[g]);
             }
             const int k_base = g * group_size;
             // Per-token split accumulators; the per-group scale/zero is folded
             // once after the K-loop, exactly as in the scalar path. Each
             // iteration processes two K-elements (one packed byte); the byte
             // load is coalesced across the sub-group and reused across every
             // token in the block routed to `expert`.
             float acc_q0[TOKEN_BLOCK];
             float acc_q1[TOKEN_BLOCK];
             float acc_a[TOKEN_BLOCK];
             for (int m = 0; m < nmembers; ++m) {
               acc_q0[m] = 0.0f;
               acc_q1[m] = 0.0f;
               acc_a[m] = 0.0f;
             }
             const int kb_base = k_base / 2;
             const int kb_count = group_size / 2;
             for (int kb = 0; kb < kb_count; ++kb) {
               uint8_t packed = w_tile[(kb_base + kb) * N_TILE + lane];
               if constexpr (!Asym) packed ^= static_cast<uint8_t>(0x88);
               int q0, q1;
               decode_int4_pair<true>(packed, q0, q1);
               const float fq0 = static_cast<float>(q0);
               const float fq1 = static_cast<float>(q1);
               const int k0 = k_base + 2 * kb;
               for (int m = 0; m < nmembers; ++m) {
                 const ScalarT* act_row = act_rows[m];
                 const float fa0 = static_cast<float>(act_row[k0]);
                 const float fa1 = static_cast<float>(act_row[k0 + 1]);
                 acc_q0[m] += fa0 * fq0;
                 acc_q1[m] += fa1 * fq1;
                 acc_a[m] += fa0 + fa1;
               }
             }
             for (int m = 0; m < nmembers; ++m) {
               acc[m] += scale * ((acc_q0[m] + acc_q1[m]) - zero * acc_a[m]);
             }
           }

           for (int m = 0; m < nmembers; ++m) {
             const int b = members[m];
             outputs[static_cast<size_t>(token_base + b) * N + n_global] = static_cast<ScalarT>(acc[m]);
           }
         }
       });

  q->wait();
  sycl::free(repacked, *q);
}

// ----------------------------------------------------------------------------
// INT8 (S8) GEMV with group-wise dequantization.
//
// Asym=false: signed byte in [-128, 127], dequant = q * scale
// Asym=true : unsigned byte in [0, 255], dequant = (q - zero) * scale
//
// Weights are stored as raw uint8 bytes (1 byte per weight). The same buffer
// type is used for sym and asym; the only difference is the sign interpretation
// performed at decode time.
// ----------------------------------------------------------------------------

// Vectorized inner accumulation over CHUNK consecutive K elements (CHUNK weight
// bytes + a vec<ScalarT,CHUNK> activation block). Templated on CHUNK so the
// caller can run a wide (32) stage first and a narrower (16) stage for the
// remainder, mirroring the int4 path: widening the first stage amortizes the
// per-group scale load and loop overhead across twice as many multiply-adds for
// the shipped group sizes (32/64/128/256) without regressing group_size == 16.
// sycl::vec only supports widths of 1, 2, 3, 4, 8 or 16, so CHUNK is processed
// in 16-wide sub-blocks. The math is identical to the scalar path.
template <typename ScalarT, bool Asym, int CHUNK>
static inline void int8_decode_chunk(const ScalarT* act_ptr, const uint8_t* w_ptr, float scale, float zero,
                                     float& acc) {
  static_assert(sizeof(ScalarT) == sizeof(uint16_t), "ScalarT must be a 16-bit floating type");
  static_assert(CHUNK % 16 == 0, "CHUNK must be a multiple of 16");
  constexpr int SUB = 16;
  using ActVec = sycl::vec<uint16_t, SUB>;
  using ByteVec = sycl::vec<uint8_t, SUB>;
#pragma unroll
  for (int s = 0; s < CHUNK / SUB; ++s) {
    const ActVec av = *reinterpret_cast<const ActVec*>(act_ptr + s * SUB);
    const ByteVec wv = *reinterpret_cast<const ByteVec*>(w_ptr + s * SUB);
#pragma unroll
    for (int u = 0; u < SUB; ++u) {
      const int qv = decode_int8<Asym>(wv[u]);
      float w;
      if constexpr (Asym) {
        w = (static_cast<float>(qv) - zero) * scale;
      } else {
        w = static_cast<float>(qv) * scale;
      }
      const ScalarT a = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[u]));
      acc += static_cast<float>(a) * w;
    }
  }
}

template <typename ScalarT, bool Asym>
void launch_int8(sycl::queue* q, const ScalarT* activations, const uint8_t* weights, const ScalarT* scales,
                 const ScalarT* zeros, ScalarT* outputs, const int* expert_id_per_token, int total_tokens, int N,
                 int K, int group_size) {
  if (N % N_TILE != 0) {
    throw std::invalid_argument("moe_gemm_decode(int8): N must be a multiple of 16");
  }
  if (K % group_size != 0) {
    throw std::invalid_argument("moe_gemm_decode(int8): K must be a multiple of group_size");
  }
  if (Asym && zeros == nullptr) {
    throw std::invalid_argument("moe_gemm_decode(int8): zeros pointer required when asym=true");
  }
  if (total_tokens == 0) return;

  const int n_tiles = N / N_TILE;
  const int num_groups_k = K / group_size;

  sycl::range<2> global{static_cast<size_t>(total_tokens), static_cast<size_t>(n_tiles * SG_SIZE)};
  sycl::range<2> local{1, static_cast<size_t>(SG_SIZE)};

  q->parallel_for<MoEDecodeKernelInt8<ScalarT, Asym>>(
       sycl::nd_range<2>(global, local),
       [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(SG_SIZE)]] {
         const int token = static_cast<int>(it.get_global_id(0));
         const int n_tile = static_cast<int>(it.get_group(1));
         const int lane = static_cast<int>(it.get_local_id(1));
         const int n_global = n_tile * N_TILE + lane;

         const int expert = expert_id_per_token[token];
         const ScalarT* act_row = activations + static_cast<size_t>(token) * K;

         const uint8_t* w_row =
             weights + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * K;
         const ScalarT* s_row =
             scales + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * num_groups_k;
         const ScalarT* z_row = Asym
             ? zeros + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * num_groups_k
             : nullptr;

         float acc = 0.0f;
         for (int g = 0; g < num_groups_k; ++g) {
           const float scale = static_cast<float>(s_row[g]);
           float zero = 0.0f;
           if constexpr (Asym) {
             zero = static_cast<float>(z_row[g]);
           }
           const int k_base = g * group_size;
           // Vectorized ladder mirroring the int4 path: process 32 K-elements
           // (32 weight bytes + vec<ScalarT,32> activations) at a time, then a
           // 16-wide stage for the remainder, then a scalar tail. Widening the
           // first stage amortizes the per-group scale load and loop overhead
           // for the shipped group sizes (32/64/128/256), while the 16-wide
           // stage keeps group_size == 16 on the fast path.
           int kk = 0;
           constexpr int CHUNK32 = 32;
           const int end32 = (group_size / CHUNK32) * CHUNK32;
           for (; kk < end32; kk += CHUNK32) {
             int8_decode_chunk<ScalarT, Asym, CHUNK32>(act_row + k_base + kk, w_row + k_base + kk, scale, zero,
                                                       acc);
           }
           constexpr int CHUNK16 = 16;
           const int end16 = kk + ((group_size - kk) / CHUNK16) * CHUNK16;
           for (; kk < end16; kk += CHUNK16) {
             int8_decode_chunk<ScalarT, Asym, CHUNK16>(act_row + k_base + kk, w_row + k_base + kk, scale, zero,
                                                       acc);
           }
           for (; kk < group_size; ++kk) {
             const int qv = decode_int8<Asym>(w_row[k_base + kk]);
             float w;
             if constexpr (Asym) {
               w = (static_cast<float>(qv) - zero) * scale;
             } else {
               w = static_cast<float>(qv) * scale;
             }
             acc += static_cast<float>(act_row[k_base + kk]) * w;
           }
         }

         outputs[static_cast<size_t>(token) * N + n_global] = static_cast<ScalarT>(acc);
       });
}

// ----------------------------------------------------------------------------
// INT2 (S2_CLIP) GEMV with group-wise dequantization.
//
// Packing: 4 values per byte. The value at K index 4*i + j is stored in
// bits [2j+1 : 2j] of byte i (i.e. byte = q0 | (q1<<2) | (q2<<4) | (q3<<6)).
//
// Asym=false: signed 2-bit value in [-2, 1]; dequant = q * scale
// Asym=true : unsigned 2-bit value in [0, 3]; dequant = (q - zero) * scale
// ----------------------------------------------------------------------------

// Vectorized inner accumulation over CHUNK consecutive K elements (CHUNK/4
// packed weight bytes + a vec<ScalarT,CHUNK> activation block). Templated on
// CHUNK so the caller can run a wide (32) stage first and a narrower (16) stage
// for the remainder, mirroring the int4/int8 paths. sycl::vec only supports
// widths of 1, 2, 3, 4, 8 or 16, so CHUNK is processed in 16-wide sub-blocks
// (16 activations + 4 packed bytes each). The math is identical to the scalar
// path.
template <typename ScalarT, bool Asym, int CHUNK>
static inline void int2_decode_chunk(const ScalarT* act_ptr, const uint8_t* w_ptr, float scale, float zero,
                                     float& acc) {
  static_assert(sizeof(ScalarT) == sizeof(uint16_t), "ScalarT must be a 16-bit floating type");
  static_assert(CHUNK % 16 == 0, "CHUNK must be a multiple of 16");
  constexpr int SUB = 16;
  using ActVec = sycl::vec<uint16_t, SUB>;
  using PackVec = sycl::vec<uint8_t, SUB / 4>;
#pragma unroll
  for (int s = 0; s < CHUNK / SUB; ++s) {
    const ActVec av = *reinterpret_cast<const ActVec*>(act_ptr + s * SUB);
    const PackVec pv = *reinterpret_cast<const PackVec*>(w_ptr + s * (SUB / 4));
#pragma unroll
    for (int b = 0; b < SUB / 4; ++b) {
      int qq[4];
      decode_int2_quad<Asym>(pv[b], qq);
      float w0, w1, w2, w3;
      if constexpr (Asym) {
        w0 = (static_cast<float>(qq[0]) - zero) * scale;
        w1 = (static_cast<float>(qq[1]) - zero) * scale;
        w2 = (static_cast<float>(qq[2]) - zero) * scale;
        w3 = (static_cast<float>(qq[3]) - zero) * scale;
      } else {
        w0 = static_cast<float>(qq[0]) * scale;
        w1 = static_cast<float>(qq[1]) * scale;
        w2 = static_cast<float>(qq[2]) * scale;
        w3 = static_cast<float>(qq[3]) * scale;
      }
      const ScalarT a0 = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[4 * b + 0]));
      const ScalarT a1 = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[4 * b + 1]));
      const ScalarT a2 = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[4 * b + 2]));
      const ScalarT a3 = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[4 * b + 3]));
      acc += static_cast<float>(a0) * w0;
      acc += static_cast<float>(a1) * w1;
      acc += static_cast<float>(a2) * w2;
      acc += static_cast<float>(a3) * w3;
    }
  }
}

template <typename ScalarT, bool Asym>
void launch_int2(sycl::queue* q, const ScalarT* activations, const uint8_t* weights, const ScalarT* scales,
                 const ScalarT* zeros, ScalarT* outputs, const int* expert_id_per_token, int total_tokens, int N,
                 int K, int group_size) {
  if (N % N_TILE != 0) {
    throw std::invalid_argument("moe_gemm_decode(int2): N must be a multiple of 16");
  }
  if ((K & 0x3) != 0) {
    throw std::invalid_argument("moe_gemm_decode(int2): K must be a multiple of 4");
  }
  if (K % group_size != 0 || (group_size & 0x3) != 0) {
    throw std::invalid_argument(
        "moe_gemm_decode(int2): K must be a multiple of group_size and group_size must be a multiple of 4");
  }
  if (Asym && zeros == nullptr) {
    throw std::invalid_argument("moe_gemm_decode(int2): zeros pointer required when asym=true");
  }
  if (total_tokens == 0) return;

  const int n_tiles = N / N_TILE;
  const int num_groups_k = K / group_size;
  const int k_packed = K / 4;  // bytes of packed weight per (expert, n)

  sycl::range<2> global{static_cast<size_t>(total_tokens), static_cast<size_t>(n_tiles * SG_SIZE)};
  sycl::range<2> local{1, static_cast<size_t>(SG_SIZE)};

  q->parallel_for<MoEDecodeKernelInt2<ScalarT, Asym>>(
       sycl::nd_range<2>(global, local),
       [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(SG_SIZE)]] {
         const int token = static_cast<int>(it.get_global_id(0));
         const int n_tile = static_cast<int>(it.get_group(1));
         const int lane = static_cast<int>(it.get_local_id(1));
         const int n_global = n_tile * N_TILE + lane;

         const int expert = expert_id_per_token[token];
         const ScalarT* act_row = activations + static_cast<size_t>(token) * K;

         const uint8_t* w_row =
             weights + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * k_packed;
         const ScalarT* s_row =
             scales + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * num_groups_k;
         const ScalarT* z_row = Asym
             ? zeros + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * num_groups_k
             : nullptr;

         float acc = 0.0f;
         for (int g = 0; g < num_groups_k; ++g) {
           const float scale = static_cast<float>(s_row[g]);
           float zero = 0.0f;
           if constexpr (Asym) {
             zero = static_cast<float>(z_row[g]);
           }
           const int k_base = g * group_size;
           // Vectorized ladder mirroring the int4/int8 paths: process 32
           // K-elements (8 packed bytes + vec<ScalarT,32> activations) at a
           // time, then a 16-wide stage for the remainder, then a scalar tail.
           // group_size is a multiple of 4; the wide stage amortizes the
           // per-group scale load for the shipped group sizes (32/64/128/256).
           int kk = 0;
           constexpr int CHUNK32 = 32;
           const int end32 = (group_size / CHUNK32) * CHUNK32;
           for (; kk < end32; kk += CHUNK32) {
             int2_decode_chunk<ScalarT, Asym, CHUNK32>(act_row + k_base + kk, w_row + (k_base + kk) / 4, scale,
                                                       zero, acc);
           }
           constexpr int CHUNK16 = 16;
           const int end16 = kk + ((group_size - kk) / CHUNK16) * CHUNK16;
           for (; kk < end16; kk += CHUNK16) {
             int2_decode_chunk<ScalarT, Asym, CHUNK16>(act_row + k_base + kk, w_row + (k_base + kk) / 4, scale,
                                                       zero, acc);
           }
           // Scalar tail (4 values per byte).
           for (; kk < group_size; kk += 4) {
             const uint8_t packed = w_row[(k_base + kk) / 4];
             int q[4];
             decode_int2_quad<Asym>(packed, q);
             float w[4];
             if constexpr (Asym) {
               w[0] = (static_cast<float>(q[0]) - zero) * scale;
               w[1] = (static_cast<float>(q[1]) - zero) * scale;
               w[2] = (static_cast<float>(q[2]) - zero) * scale;
               w[3] = (static_cast<float>(q[3]) - zero) * scale;
             } else {
               w[0] = static_cast<float>(q[0]) * scale;
               w[1] = static_cast<float>(q[1]) * scale;
               w[2] = static_cast<float>(q[2]) * scale;
               w[3] = static_cast<float>(q[3]) * scale;
             }
             acc += static_cast<float>(act_row[k_base + kk + 0]) * w[0];
             acc += static_cast<float>(act_row[k_base + kk + 1]) * w[1];
             acc += static_cast<float>(act_row[k_base + kk + 2]) * w[2];
             acc += static_cast<float>(act_row[k_base + kk + 3]) * w[3];
           }
         }

         outputs[static_cast<size_t>(token) * N + n_global] = static_cast<ScalarT>(acc);
       });
}

// ----------------------------------------------------------------------------
// FP8 (E4M3 / E5M2) GEMV with group-wise scale (no zero-point).
//
// Weights are 1 FP8 byte per element [E, N, K]. The byte is decoded via the
// `decode_fp8<IsE4M3, UseLut>` helper, which selects between the LUT and the
// inline bit-manipulation path at compile time. The choice is driven at
// launch time by the env var `ARK_FP8_DECODE_USE_LUT` (default: ON).
// ----------------------------------------------------------------------------

// Vectorized inner accumulation over CHUNK consecutive K elements (CHUNK weight
// bytes + a vec<ScalarT,CHUNK> activation block). Templated on CHUNK so the
// caller can run a wide (32) stage first and a narrower (16) stage for the
// remainder, mirroring the int4/int8 paths. sycl::vec only supports widths of
// 1, 2, 3, 4, 8 or 16, so CHUNK is processed in 16-wide sub-blocks.
//
// The per-group scale is constant across the whole group, so it is NOT applied
// here: this accumulates the raw dot product (sum of act * decoded_fp8) and the
// caller multiplies the group total by the scale once (Σ a·(w·s) == s·Σ a·w).
// For the per-expert / per-tensor scale case (group_size == K, one scale per
// output row) this collapses the whole K reduction to a single scale multiply,
// removing one multiply per K element on the decode hot path.
template <typename ScalarT, bool IsE4M3, bool UseLut, int CHUNK>
static inline void fp8_decode_chunk(const ScalarT* act_ptr, const uint8_t* w_ptr, float& acc) {
  static_assert(sizeof(ScalarT) == sizeof(uint16_t), "ScalarT must be a 16-bit floating type");
  static_assert(CHUNK % 16 == 0, "CHUNK must be a multiple of 16");
  constexpr int SUB = 16;
  using ActVec = sycl::vec<uint16_t, SUB>;
  using ByteVec = sycl::vec<uint8_t, SUB>;
#pragma unroll
  for (int s = 0; s < CHUNK / SUB; ++s) {
    const ActVec av = *reinterpret_cast<const ActVec*>(act_ptr + s * SUB);
    const ByteVec wv = *reinterpret_cast<const ByteVec*>(w_ptr + s * SUB);
#pragma unroll
    for (int u = 0; u < SUB; ++u) {
      const float w = decode_fp8<IsE4M3, UseLut>(wv[u]);
      const ScalarT a = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[u]));
      acc += static_cast<float>(a) * w;
    }
  }
}

template <typename ScalarT, bool IsE4M3, bool UseLut>
void launch_fp8(sycl::queue* q, const ScalarT* activations, const uint8_t* weights, const ScalarT* scales,
                ScalarT* outputs, const int* expert_id_per_token, int total_tokens, int N, int K, int group_size) {
  if (N % N_TILE != 0) {
    throw std::invalid_argument("moe_gemm_decode(fp8): N must be a multiple of 16");
  }
  if (K % group_size != 0) {
    throw std::invalid_argument("moe_gemm_decode(fp8): K must be a multiple of group_size");
  }
  if (total_tokens == 0) return;

  const int n_tiles = N / N_TILE;
  const int num_groups_k = K / group_size;

  sycl::range<2> global{static_cast<size_t>(total_tokens), static_cast<size_t>(n_tiles * SG_SIZE)};
  sycl::range<2> local{1, static_cast<size_t>(SG_SIZE)};

  q->parallel_for<MoEDecodeKernelFP8<ScalarT, IsE4M3, UseLut>>(
       sycl::nd_range<2>(global, local),
       [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(SG_SIZE)]] {
         const int token = static_cast<int>(it.get_global_id(0));
         const int n_tile = static_cast<int>(it.get_group(1));
         const int lane = static_cast<int>(it.get_local_id(1));
         const int n_global = n_tile * N_TILE + lane;

         const int expert = expert_id_per_token[token];
         const ScalarT* act_row = activations + static_cast<size_t>(token) * K;

         const uint8_t* w_row =
             weights + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * K;
         const ScalarT* s_row =
             scales + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * num_groups_k;

         float acc = 0.0f;
         for (int g = 0; g < num_groups_k; ++g) {
           const float scale = static_cast<float>(s_row[g]);
           const int k_base = g * group_size;
           // Vectorized ladder mirroring the int4/int8 paths: process 32
           // K-elements (32 weight bytes + vec<ScalarT,32> activations) at a
           // time, then a 16-wide stage for the remainder, then a scalar tail.
           // The per-group scale is constant across the group, so accumulate the
           // raw dot product here and apply the scale once below (Σ a·(w·s) ==
           // s·Σ a·w). Widening the first stage amortizes the per-group scale
           // load for the shipped group sizes (32/64/128/256); hoisting the
           // scale removes one multiply per K element, which is the dominant
           // cost for the per-expert / per-tensor scale case (group_size == K).
           float group_acc = 0.0f;
           int kk = 0;
           constexpr int CHUNK32 = 32;
           const int end32 = (group_size / CHUNK32) * CHUNK32;
           for (; kk < end32; kk += CHUNK32) {
             fp8_decode_chunk<ScalarT, IsE4M3, UseLut, CHUNK32>(act_row + k_base + kk, w_row + k_base + kk,
                                                                group_acc);
           }
           constexpr int CHUNK16 = 16;
           const int end16 = kk + ((group_size - kk) / CHUNK16) * CHUNK16;
           for (; kk < end16; kk += CHUNK16) {
             fp8_decode_chunk<ScalarT, IsE4M3, UseLut, CHUNK16>(act_row + k_base + kk, w_row + k_base + kk,
                                                                group_acc);
           }
           for (; kk < group_size; ++kk) {
             const uint8_t raw = w_row[k_base + kk];
             const float w = decode_fp8<IsE4M3, UseLut>(raw);
             group_acc += static_cast<float>(act_row[k_base + kk]) * w;
           }
           acc += group_acc * scale;
         }

         outputs[static_cast<size_t>(token) * N + n_global] = static_cast<ScalarT>(acc);
       });
}

}  // namespace moe_decode_detail

// ----------------------------------------------------------------------------
// Env-flag helper -- `ARK_MOE_DECODE_DPAS_S4` (default ON). When ON, int4-sym
// (S4_CLIP, !asym) decode is routed to the dedicated decode-phase S4 DPAS
// grouped GEMM (`moe_dpas_s4::moe_decode_s4_dpas_per_group_dispatch`) instead
// of the scalar FMA GEMV (`launch_int4`). Mirroring vLLM-xpu-kernels'
// `w4a16` decode dispatch, this path selects the DPAS tile from the average
// tokens-per-expert (`A_avg_M`) ladder (`_m_8` -> `_m_16` -> `_m_32` -> wide),
// reusing the shared per-group mainloop's 2D VNNI block load + register-resident
// per-N scale. It reads the same `[E, N, K/2]` packed weights + `[E, N, K/group]`
// scales, so no repack is needed. (`ARK_MOE_DECODE_S4_DPAS_M8=1` forces the
// legacy hard-pinned 8-row tile for A/B comparison; the two are numerically
// identical.)
//
// Setting the var to "0" / "false" / "off" / "no" (case-insensitive) forces
// the legacy scalar GEMV, for A/B comparison and regression escape. Asym
// weights, shapes that fail the DPAS shape gate, and batches that fail the
// tokens-per-expert occupancy gate (`moe_decode_dpas_s4_occupancy_ok`, see
// below -- this is what keeps real decode batches on the fast scalar GEMV)
// always fall back to the scalar path regardless of this flag. Re-read on every call so tests /
// benchmarks can toggle the path in-process.
// ----------------------------------------------------------------------------
inline bool moe_decode_dpas_s4_enabled() {
  const char* env = std::getenv("ARK_MOE_DECODE_DPAS_S4");
  if (env == nullptr) return true;  // default ON
  std::string s(env);
  for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "0" || s == "false" || s == "off" || s == "no") return false;
  return true;
}

// ----------------------------------------------------------------------------
// Occupancy gate for the int4-sym S4 DPAS decode path.
//
// The DPAS grouped GEMM pays off only when its M tile is actually filled: the
// smallest tile (`dpas_w4a16_policy_m_8`) processes 8 token rows per expert, so
// with fewer than 8 tokens routed to an expert on average the tile is mostly
// padding and the (bandwidth-bound) packed weights are streamed for rows that
// contribute nothing. Real decode batches are exactly that regime -- e.g.
// MiniMax-M2 decode is 8 tokens (bs1) or 256 tokens (bs32) spread over 192
// experts, i.e. 0.04-1.3 tokens per expert -- and there the shared scalar GEMV
// (`launch_int4`, the very kernel the *asym* path uses, where sym is just
// `Asym=false`) is up to ~3x faster because it reads each weight byte exactly
// once per active token with no tile padding.
//
// So route int4-sym decode through the same scalar GEMV as int4-asym unless the
// batch has at least one full 8-row tile of tokens per expert on average.
// `ARK_MOE_DECODE_DPAS_S4_MIN_TPE` overrides the tokens-per-expert threshold;
// "0" disables the gate (always take DPAS when the shape gate allows), which is
// what the accuracy tests use to exercise the DPAS kernel on tiny shapes.
// ----------------------------------------------------------------------------
inline bool moe_decode_dpas_s4_occupancy_ok(int total_tokens, int num_experts) {
  if (num_experts <= 0) return true;
  long long min_tokens_per_expert = 8;  // rows in `dpas_w4a16_policy_m_8`
  const char* env = std::getenv("ARK_MOE_DECODE_DPAS_S4_MIN_TPE");
  if (env != nullptr) {
    char* end = nullptr;
    long long v = std::strtoll(env, &end, 10);
    if (end != env && v >= 0) min_tokens_per_expert = v;
  }
  if (min_tokens_per_expert == 0) return true;
  return static_cast<long long>(total_tokens) >= min_tokens_per_expert * static_cast<long long>(num_experts);
}

// ----------------------------------------------------------------------------
// Env-flag helper -- `ARK_MOE_DECODE_COALESCE_INT4` (default ON). When ON, the
// int4 scalar-GEMV fallback (asym, or sym with the DPAS path disabled / shape
// or occupancy gate miss) uses `launch_int4_coalesced`, which repacks the weights on-device
// into an N-tiled layout so sub-group weight loads are coalesced. Setting the
// var to "0" / "false" / "off" / "no" (case-insensitive) forces the legacy
// per-lane-strided `launch_int4`, for A/B comparison and regression escape.
// Re-read on every call so tests / benchmarks can toggle it in-process.
// ----------------------------------------------------------------------------
inline bool moe_decode_coalesce_int4_enabled() {
  const char* env = std::getenv("ARK_MOE_DECODE_COALESCE_INT4");
  if (env == nullptr) return true;  // default ON
  std::string s(env);
  for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "0" || s == "false" || s == "off" || s == "no") return false;
  return true;
}

// ----------------------------------------------------------------------------
// Amortization gate for the coalesced int4 decode kernel. The coalesced path
// repacks the *entire* weight tensor for all `num_experts` on every call
// (cost proportional to num_experts * N * K/2) before running the GEMV. That
// one-time repack only pays off when a work-group reuses each repacked weight
// tile across many tokens -- i.e. when there are enough active tokens relative
// to the number of experts. For tiny decode batches (e.g. 8 tokens spread
// across 192 experts) the repack dominates and the coalesced kernel is far
// slower than the per-lane-strided `launch_int4`, which reads the weights in
// place with no repack. Require at least one full TOKEN_BLOCK worth of tokens
// per expert on average before coalescing; otherwise fall back to `launch_int4`.
// `ARK_MOE_DECODE_COALESCE_MIN_TOKENS` overrides the threshold (tokens per
// expert scaled by TOKEN_BLOCK); "0" disables the gate (always coalesce).
// ----------------------------------------------------------------------------
inline bool moe_decode_coalesce_int4_amortized(int total_tokens, int num_experts) {
  if (num_experts <= 0) return true;
  long long min_tokens = static_cast<long long>(num_experts) * moe_decode_detail::TOKEN_BLOCK;
  const char* env = std::getenv("ARK_MOE_DECODE_COALESCE_MIN_TOKENS");
  if (env != nullptr) {
    char* end = nullptr;
    long long v = std::strtoll(env, &end, 10);
    if (end != env && v >= 0) min_tokens = v;
  }
  return static_cast<long long>(total_tokens) >= min_tokens;
}

// ----------------------------------------------------------------------------
//
// weight_dtype:
//   BTLA_DTYPE::F16  / BF16       : weights stored as [E, N, K] in matching
//                                   floating dtype, no scales/zeros needed
//   BTLA_DTYPE::S8                : int8 weights [E, N, K] (uint8 buffer,
//                                   interpreted as signed when asym==false,
//                                   unsigned with zero-points when asym==true)
//   BTLA_DTYPE::S4_CLIP           : packed int4 weights [E, N, K/2] (uint8),
//                                   scales [E, N, K/group_size] in act dtype,
//                                   zeros optional (asym==true requires it).
//                                   Sym weights are routed to the shared
//                                   per-group S4 DPAS grouped GEMM only when
//                                   the batch fills its M tile (>= 8 tokens per
//                                   expert on average, `ARK_MOE_DECODE_DPAS_S4`
//                                   default ON); asym, a disabled flag, a
//                                   shape-gate miss, or a decode-sized batch
//                                   uses the shared scalar GEMV.
//   BTLA_DTYPE::S2_CLIP           : packed int2 weights [E, N, K/4] (uint8),
//                                   4 values per byte, sym/asym like int4
//   BTLA_DTYPE::F8_E4M3 / F8_E5M2 : FP8 weights [E, N, K] (uint8 buffer),
//                                   group-wise scales, no zero-points
// act_dtype: F16 or BF16 (must match scales/outputs dtype)
// ----------------------------------------------------------------------------
inline void moe_gemm_decode(sycl::queue* q, void* activations, void* weights, void* scales, void* zeros,
                            void* outputs, int* expert_id_per_token_buf, BTLA_DTYPE act_dtype,
                            BTLA_DTYPE weight_dtype, int N, int K, int group_size, int* num_tokens_per_expert,
                            int num_experts, int total_tokens, bool asym) {
  // The S4-sym DPAS fast path consumes `num_tokens_per_expert` directly and
  // never reads `expert_id_per_token_buf`. Skipping the fill on that path
  // removes an extra device-timeline kernel launch from the decode hot path;
  // every other path (fp, int8, int2, fp8, and the scalar int4 fallback) still
  // needs the per-token expert mapping.
  const bool s4_dpas_fastpath = weight_dtype == BTLA_DTYPE::S4_CLIP && !asym && moe_decode_dpas_s4_enabled() &&
                                moe_decode_dpas_s4_occupancy_ok(total_tokens, num_experts) &&
                                moe_dpas_s4::moe_prefill_dpas_s4_pergroup_shape_ok(N, K, group_size);
  if (!s4_dpas_fastpath) {
    moe_decode_detail::fill_expert_id_per_token(q, expert_id_per_token_buf, num_tokens_per_expert, num_experts,
                                                total_tokens);
  }

  if (weight_dtype == BTLA_DTYPE::F16 || weight_dtype == BTLA_DTYPE::BF16) {
    if (weight_dtype != act_dtype) {
      throw std::invalid_argument("moe_gemm_decode: unquantized weight_dtype must match act_dtype");
    }
    if (act_dtype == BTLA_DTYPE::F16) {
      moe_decode_detail::launch_fp<sycl::half>(q, static_cast<const sycl::half*>(activations),
                                               static_cast<const sycl::half*>(weights),
                                               static_cast<sycl::half*>(outputs), expert_id_per_token_buf,
                                               total_tokens, N, K);
    } else {
      moe_decode_detail::launch_fp<sycl::ext::oneapi::bfloat16>(
          q, static_cast<const sycl::ext::oneapi::bfloat16*>(activations),
          static_cast<const sycl::ext::oneapi::bfloat16*>(weights),
          static_cast<sycl::ext::oneapi::bfloat16*>(outputs), expert_id_per_token_buf, total_tokens, N, K);
    }
    return;
  }

  if (weight_dtype == BTLA_DTYPE::S4_CLIP) {
    if (act_dtype != BTLA_DTYPE::F16 && act_dtype != BTLA_DTYPE::BF16) {
      throw std::invalid_argument("moe_gemm_decode(int4): act_dtype must be FP16 or BF16");
    }
    // Fast path: sym int4 through the shared per-group S4 DPAS grouped GEMM.
    // Falls back to the scalar GEMV for asym weights (DPAS S4 is sym-only),
    // when the env flag is off, when the batch is too small to fill the DPAS M
    // tile (the usual decode case -- sym then runs the exact same
    // `launch_int4*` kernel as asym, with `Asym=false`), or when the shape gate
    // rejects the tile geometry (e.g. N%64!=0, K%32!=0, unsupported
    // group_size). Reuses the
    // `s4_dpas_fastpath` predicate computed above (which also gated the
    // `fill_expert_id_per_token` skip) so the two decisions cannot diverge.
    if (s4_dpas_fastpath) {
      if (act_dtype == BTLA_DTYPE::F16) {
        moe_dpas_s4::moe_decode_s4_dpas_per_group_dispatch<sycl::half>(
            q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const sycl::half*>(scales), static_cast<sycl::half*>(outputs), num_tokens_per_expert,
            num_experts, N, K, group_size, total_tokens);
      } else {
        using BF = sycl::ext::oneapi::bfloat16;
        moe_dpas_s4::moe_decode_s4_dpas_per_group_dispatch<BF>(
            q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const BF*>(scales), static_cast<BF*>(outputs), num_tokens_per_expert, num_experts, N, K,
            group_size, total_tokens);
      }
      return;
    }
    // Scalar FMA GEMV fallback (asym, flag off, or shape gate miss). By
    // default this uses the coalesced-load variant, which repacks the weights
    // on-device so sub-group loads are contiguous; `ARK_MOE_DECODE_COALESCE_INT4=0`
    // forces the legacy per-lane-strided kernel.
    const bool coalesce = moe_decode_coalesce_int4_enabled() &&
                          moe_decode_coalesce_int4_amortized(total_tokens, num_experts);
    if (act_dtype == BTLA_DTYPE::F16) {
      if (asym) {
        if (coalesce) {
          moe_decode_detail::launch_int4_coalesced<sycl::half, true>(
              q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
              static_cast<const sycl::half*>(scales), static_cast<const sycl::half*>(zeros),
              static_cast<sycl::half*>(outputs), expert_id_per_token_buf, total_tokens, N, K, group_size,
              num_experts);
        } else {
          moe_decode_detail::launch_int4<sycl::half, true>(
              q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
              static_cast<const sycl::half*>(scales), static_cast<const sycl::half*>(zeros),
              static_cast<sycl::half*>(outputs), expert_id_per_token_buf, total_tokens, N, K, group_size);
        }
      } else {
        if (coalesce) {
          moe_decode_detail::launch_int4_coalesced<sycl::half, false>(
              q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
              static_cast<const sycl::half*>(scales), static_cast<const sycl::half*>(zeros),
              static_cast<sycl::half*>(outputs), expert_id_per_token_buf, total_tokens, N, K, group_size,
              num_experts);
        } else {
          moe_decode_detail::launch_int4<sycl::half, false>(
              q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
              static_cast<const sycl::half*>(scales), static_cast<const sycl::half*>(zeros),
              static_cast<sycl::half*>(outputs), expert_id_per_token_buf, total_tokens, N, K, group_size);
        }
      }
    } else {
      using BF = sycl::ext::oneapi::bfloat16;
      if (asym) {
        if (coalesce) {
          moe_decode_detail::launch_int4_coalesced<BF, true>(
              q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
              static_cast<const BF*>(scales), static_cast<const BF*>(zeros), static_cast<BF*>(outputs),
              expert_id_per_token_buf, total_tokens, N, K, group_size, num_experts);
        } else {
          moe_decode_detail::launch_int4<BF, true>(
              q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
              static_cast<const BF*>(scales), static_cast<const BF*>(zeros), static_cast<BF*>(outputs),
              expert_id_per_token_buf, total_tokens, N, K, group_size);
        }
      } else {
        if (coalesce) {
          moe_decode_detail::launch_int4_coalesced<BF, false>(
              q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
              static_cast<const BF*>(scales), static_cast<const BF*>(zeros), static_cast<BF*>(outputs),
              expert_id_per_token_buf, total_tokens, N, K, group_size, num_experts);
        } else {
          moe_decode_detail::launch_int4<BF, false>(
              q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
              static_cast<const BF*>(scales), static_cast<const BF*>(zeros), static_cast<BF*>(outputs),
              expert_id_per_token_buf, total_tokens, N, K, group_size);
        }
      }
    }
    return;
  }

  if (weight_dtype == BTLA_DTYPE::S8) {
    if (act_dtype == BTLA_DTYPE::F16) {
      if (asym) {
        moe_decode_detail::launch_int8<sycl::half, true>(
            q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const sycl::half*>(scales), static_cast<const sycl::half*>(zeros),
            static_cast<sycl::half*>(outputs), expert_id_per_token_buf, total_tokens, N, K, group_size);
      } else {
        moe_decode_detail::launch_int8<sycl::half, false>(
            q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const sycl::half*>(scales), static_cast<const sycl::half*>(zeros),
            static_cast<sycl::half*>(outputs), expert_id_per_token_buf, total_tokens, N, K, group_size);
      }
    } else if (act_dtype == BTLA_DTYPE::BF16) {
      using BF = sycl::ext::oneapi::bfloat16;
      if (asym) {
        moe_decode_detail::launch_int8<BF, true>(
            q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const BF*>(scales), static_cast<const BF*>(zeros), static_cast<BF*>(outputs),
            expert_id_per_token_buf, total_tokens, N, K, group_size);
      } else {
        moe_decode_detail::launch_int8<BF, false>(
            q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const BF*>(scales), static_cast<const BF*>(zeros), static_cast<BF*>(outputs),
            expert_id_per_token_buf, total_tokens, N, K, group_size);
      }
    } else {
      throw std::invalid_argument("moe_gemm_decode(int8): act_dtype must be FP16 or BF16");
    }
    return;
  }

  if (weight_dtype == BTLA_DTYPE::S2_CLIP) {
    if (act_dtype == BTLA_DTYPE::F16) {
      if (asym) {
        moe_decode_detail::launch_int2<sycl::half, true>(
            q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const sycl::half*>(scales), static_cast<const sycl::half*>(zeros),
            static_cast<sycl::half*>(outputs), expert_id_per_token_buf, total_tokens, N, K, group_size);
      } else {
        moe_decode_detail::launch_int2<sycl::half, false>(
            q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const sycl::half*>(scales), static_cast<const sycl::half*>(zeros),
            static_cast<sycl::half*>(outputs), expert_id_per_token_buf, total_tokens, N, K, group_size);
      }
    } else if (act_dtype == BTLA_DTYPE::BF16) {
      using BF = sycl::ext::oneapi::bfloat16;
      if (asym) {
        moe_decode_detail::launch_int2<BF, true>(
            q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const BF*>(scales), static_cast<const BF*>(zeros), static_cast<BF*>(outputs),
            expert_id_per_token_buf, total_tokens, N, K, group_size);
      } else {
        moe_decode_detail::launch_int2<BF, false>(
            q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const BF*>(scales), static_cast<const BF*>(zeros), static_cast<BF*>(outputs),
            expert_id_per_token_buf, total_tokens, N, K, group_size);
      }
    } else {
      throw std::invalid_argument("moe_gemm_decode(int2): act_dtype must be FP16 or BF16");
    }
    return;
  }

  if (weight_dtype == BTLA_DTYPE::F8_E4M3 || weight_dtype == BTLA_DTYPE::F8_E5M2) {
    if (asym) {
      throw std::invalid_argument("moe_gemm_decode(fp8): asym mode is not supported");
    }
    const bool is_e4m3 = (weight_dtype == BTLA_DTYPE::F8_E4M3);
    const bool use_lut = moe_decode_detail::fp8_decode_use_lut();
    if (act_dtype == BTLA_DTYPE::F16) {
      if (is_e4m3) {
        if (use_lut) {
          moe_decode_detail::launch_fp8<sycl::half, true, true>(
              q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
              static_cast<const sycl::half*>(scales), static_cast<sycl::half*>(outputs), expert_id_per_token_buf,
              total_tokens, N, K, group_size);
        } else {
          moe_decode_detail::launch_fp8<sycl::half, true, false>(
              q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
              static_cast<const sycl::half*>(scales), static_cast<sycl::half*>(outputs), expert_id_per_token_buf,
              total_tokens, N, K, group_size);
        }
      } else {
        if (use_lut) {
          moe_decode_detail::launch_fp8<sycl::half, false, true>(
              q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
              static_cast<const sycl::half*>(scales), static_cast<sycl::half*>(outputs), expert_id_per_token_buf,
              total_tokens, N, K, group_size);
        } else {
          moe_decode_detail::launch_fp8<sycl::half, false, false>(
              q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
              static_cast<const sycl::half*>(scales), static_cast<sycl::half*>(outputs), expert_id_per_token_buf,
              total_tokens, N, K, group_size);
        }
      }
    } else if (act_dtype == BTLA_DTYPE::BF16) {
      using BF = sycl::ext::oneapi::bfloat16;
      if (is_e4m3) {
        if (use_lut) {
          moe_decode_detail::launch_fp8<BF, true, true>(
              q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
              static_cast<const BF*>(scales), static_cast<BF*>(outputs), expert_id_per_token_buf, total_tokens, N, K,
              group_size);
        } else {
          moe_decode_detail::launch_fp8<BF, true, false>(
              q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
              static_cast<const BF*>(scales), static_cast<BF*>(outputs), expert_id_per_token_buf, total_tokens, N, K,
              group_size);
        }
      } else {
        if (use_lut) {
          moe_decode_detail::launch_fp8<BF, false, true>(
              q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
              static_cast<const BF*>(scales), static_cast<BF*>(outputs), expert_id_per_token_buf, total_tokens, N, K,
              group_size);
        } else {
          moe_decode_detail::launch_fp8<BF, false, false>(
              q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
              static_cast<const BF*>(scales), static_cast<BF*>(outputs), expert_id_per_token_buf, total_tokens, N, K,
              group_size);
        }
      }
    } else {
      throw std::invalid_argument("moe_gemm_decode(fp8): act_dtype must be FP16 or BF16");
    }
    return;
  }

  throw std::invalid_argument(
      "moe_gemm_decode: unsupported weight_dtype (supported: F16, BF16, S8, S4_CLIP, S2_CLIP, F8_E4M3, F8_E5M2)");
}

}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
