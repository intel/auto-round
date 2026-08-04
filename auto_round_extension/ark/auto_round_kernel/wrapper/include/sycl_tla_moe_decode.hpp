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
//                            4-bit values per byte (low nibble at lower K)
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

// ----------------------------------------------------------------------------
// Kernel name tags (one per specialization, required for SYCL kernel naming)
// ----------------------------------------------------------------------------
template <typename ScalarT>
class MoEDecodeKernelFP;

template <typename ScalarT, bool Asym>
class MoEDecodeKernelInt4;

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
// stage). The math is identical to the scalar path.
template <typename ScalarT, bool Asym, int CHUNK>
static inline void int4_decode_chunk(const ScalarT* act_ptr, const uint8_t* w_ptr, float scale, float zero,
                                     float& acc) {
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
    const PackVec pv = *reinterpret_cast<const PackVec*>(w_ptr + s * (SUB / 2));
#pragma unroll
    for (int b = 0; b < SUB / 2; ++b) {
      int q0, q1;
      decode_int4_pair<Asym>(pv[b], q0, q1);
      float w0, w1;
      if constexpr (Asym) {
        w0 = (static_cast<float>(q0) - zero) * scale;
        w1 = (static_cast<float>(q1) - zero) * scale;
      } else {
        w0 = static_cast<float>(q0) * scale;
        w1 = static_cast<float>(q1) * scale;
      }
      const ScalarT a0 = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[2 * b]));
      const ScalarT a1 = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[2 * b + 1]));
      acc += static_cast<float>(a0) * w0;
      acc += static_cast<float>(a1) * w1;
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
           float zero = 0.0f;
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
           int kk = 0;
           constexpr int CHUNK32 = 32;
           const int end32 = (group_size / CHUNK32) * CHUNK32;
           for (; kk < end32; kk += CHUNK32) {
             int4_decode_chunk<ScalarT, Asym, CHUNK32>(act_row + k_base + kk, w_row + (k_base + kk) / 2, scale,
                                                       zero, acc);
           }
           constexpr int CHUNK16 = 16;
           const int end16 = kk + ((group_size - kk) / CHUNK16) * CHUNK16;
           for (; kk < end16; kk += CHUNK16) {
             int4_decode_chunk<ScalarT, Asym, CHUNK16>(act_row + k_base + kk, w_row + (k_base + kk) / 2, scale,
                                                       zero, acc);
           }
           // Scalar tail for group_size not divisible by 16.
           for (; kk < group_size; kk += 2) {
             const uint8_t packed = w_row[(k_base + kk) / 2];
             int q0, q1;
             decode_int4_pair<Asym>(packed, q0, q1);
             float w0, w1;
             if constexpr (Asym) {
               w0 = (static_cast<float>(q0) - zero) * scale;
               w1 = (static_cast<float>(q1) - zero) * scale;
             } else {
               w0 = static_cast<float>(q0) * scale;
               w1 = static_cast<float>(q1) * scale;
             }
             acc += static_cast<float>(act_row[k_base + kk]) * w0;
             acc += static_cast<float>(act_row[k_base + kk + 1]) * w1;
           }
         }

         outputs[static_cast<size_t>(token) * N + n_global] = static_cast<ScalarT>(acc);
       });
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
// (S4_CLIP, !asym) decode is routed to the shared per-group S4 DPAS grouped
// GEMM (`moe_dpas_s4::moe_prefill_s4_dpas_per_group_dispatch`) instead of the
// scalar FMA GEMV (`launch_int4`). The DPAS path already handles the tiny
// total-token counts typical of decode (its `A_avg_M <= 4` bucket selects the
// 8-row `dpas_w4a16_policy_m_8` tile) and reads the same `[E, N, K/2]` packed
// weights + `[E, N, K/group]` scales, so no repack is needed.
//
// Setting the var to "0" / "false" / "off" / "no" (case-insensitive) forces
// the legacy scalar GEMV, for A/B comparison and regression escape. Asym
// weights and shapes that fail the DPAS shape gate always fall back to the
// scalar path regardless of this flag. Re-read on every call so tests /
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
//                                   per-group S4 DPAS grouped GEMM by default
//                                   (`ARK_MOE_DECODE_DPAS_S4`, default ON);
//                                   asym, a disabled flag, or a shape-gate
//                                   miss falls back to the scalar GEMV.
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
  moe_decode_detail::fill_expert_id_per_token(q, expert_id_per_token_buf, num_tokens_per_expert, num_experts,
                                              total_tokens);

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
    // when the env flag is off, or when the shape gate rejects the tile
    // geometry (e.g. N%64!=0, K%32!=0, unsupported group_size).
    if (!asym && moe_decode_dpas_s4_enabled() &&
        moe_dpas_s4::moe_prefill_dpas_s4_pergroup_shape_ok(N, K, group_size)) {
      if (act_dtype == BTLA_DTYPE::F16) {
        moe_dpas_s4::moe_prefill_s4_dpas_per_group_dispatch<sycl::half>(
            q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const sycl::half*>(scales), static_cast<sycl::half*>(outputs), num_tokens_per_expert,
            num_experts, N, K, group_size, total_tokens);
      } else {
        using BF = sycl::ext::oneapi::bfloat16;
        moe_dpas_s4::moe_prefill_s4_dpas_per_group_dispatch<BF>(
            q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const BF*>(scales), static_cast<BF*>(outputs), num_tokens_per_expert, num_experts, N, K,
            group_size, total_tokens);
      }
      return;
    }
    // Scalar FMA GEMV fallback (asym, flag off, or shape gate miss).
    if (act_dtype == BTLA_DTYPE::F16) {
      if (asym) {
        moe_decode_detail::launch_int4<sycl::half, true>(
            q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const sycl::half*>(scales), static_cast<const sycl::half*>(zeros),
            static_cast<sycl::half*>(outputs), expert_id_per_token_buf, total_tokens, N, K, group_size);
      } else {
        moe_decode_detail::launch_int4<sycl::half, false>(
            q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const sycl::half*>(scales), static_cast<const sycl::half*>(zeros),
            static_cast<sycl::half*>(outputs), expert_id_per_token_buf, total_tokens, N, K, group_size);
      }
    } else {
      using BF = sycl::ext::oneapi::bfloat16;
      if (asym) {
        moe_decode_detail::launch_int4<BF, true>(
            q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const BF*>(scales), static_cast<const BF*>(zeros), static_cast<BF*>(outputs),
            expert_id_per_token_buf, total_tokens, N, K, group_size);
      } else {
        moe_decode_detail::launch_int4<BF, false>(
            q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const BF*>(scales), static_cast<const BF*>(zeros), static_cast<BF*>(outputs),
            expert_id_per_token_buf, total_tokens, N, K, group_size);
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
