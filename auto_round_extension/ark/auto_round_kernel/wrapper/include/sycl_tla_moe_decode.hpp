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
#include <limits>
#include <stdexcept>

#include "bestla/bestla.h"
#include "bestla/sycl/fp8_lut.h"

#ifdef ARK_XPU
#include <sycl/sycl.hpp>
#endif

// ----------------------------------------------------------------------------
// FP8 decode implementation switch
//
// Define `ARK_FP8_DECODE_USE_LUT` (e.g. -DARK_FP8_DECODE_USE_LUT) to dequantize
// each FP8 byte via the 128-entry magnitude LUT in `bestla/sycl/fp8_lut.h`
// (sign applied separately). Leave it undefined to keep the inline bit-manip
// decode below, which is the default and matches the previous behavior.
//
// Both paths are mathematically equivalent for finite values; pick whichever
// is faster on the target hardware. The LUT trades a handful of bit/branch
// ops for a single constant-memory load per weight byte.
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

template <typename ScalarT, bool IsE4M3>
class MoEDecodeKernelFP8;

// ----------------------------------------------------------------------------
// FP8 byte -> float decode.
// Matches IEEE-style layout used by torch.float8_e4m3fn / torch.float8_e5m2:
//   E4M3 (finite-only): 1 sign, 4 exp (bias 7), 3 mantissa; 0x7F/0xFF = NaN.
//   E5M2 (IEEE-like):   1 sign, 5 exp (bias 15), 2 mantissa; exp==31 -> Inf/NaN.
//
// Two implementations are provided and selected at compile time via
// `ARK_FP8_DECODE_USE_LUT` (see the comment near the top of this file):
//   - default (macro undefined): inline bit-manipulation, fully self-contained,
//     no LUT / SLM required. Per-lane bit ops are cheap relative to the global
//     memory loads in this kernel, so this is a reasonable default.
//   - macro defined: read the magnitude from the 128-entry constexpr LUT in
//     `bestla/sycl/fp8_lut.h` and apply the sign bit separately.
// ----------------------------------------------------------------------------
#if defined(ARK_FP8_DECODE_USE_LUT)

inline float decode_fp8_e4m3(uint8_t byte) {
  const uint32_t mag = byte & 0x7Fu;
  const float v = bestla::sycl_prologue_b::fp8_lut::lut_e4m3_128[mag];
  return (byte & 0x80u) ? -v : v;
}

inline float decode_fp8_e5m2(uint8_t byte) {
  const uint32_t mag = byte & 0x7Fu;
  const float v = bestla::sycl_prologue_b::fp8_lut::lut_e5m2_128[mag];
  return (byte & 0x80u) ? -v : v;
}

#else  // !ARK_FP8_DECODE_USE_LUT

inline float decode_fp8_e4m3(uint8_t byte) {
  const uint32_t mag = byte & 0x7Fu;
  const uint32_t sign = byte >> 7;
  float v;
  if (mag == 0u) {
    v = 0.0f;
  } else if (mag == 0x7Fu) {
    v = sycl::nan(0u);
  } else {
    const int exp = static_cast<int>((mag >> 3) & 0xFu);
    const int man = static_cast<int>(mag & 0x7u);
    if (exp == 0) {
      // subnormal: value = man * 2^(1 - bias - mbits) = man / 512
      v = static_cast<float>(man) * (1.0f / 512.0f);
    } else {
      // normal: (1 + man/8) * 2^(exp - bias), bias = 7
      v = (1.0f + static_cast<float>(man) * 0.125f) * sycl::ldexp(1.0f, exp - 7);
    }
  }
  return sign ? -v : v;
}

inline float decode_fp8_e5m2(uint8_t byte) {
  const uint32_t mag = byte & 0x7Fu;
  const uint32_t sign = byte >> 7;
  const int exp = static_cast<int>((mag >> 2) & 0x1Fu);
  const int man = static_cast<int>(mag & 0x3u);
  float v;
  if (exp == 0) {
    // subnormal (incl. zero): value = man * 2^(1 - bias - mbits) = man / 65536
    v = static_cast<float>(man) * (1.0f / 65536.0f);
  } else if (exp == 31) {
    v = (man == 0) ? std::numeric_limits<float>::infinity() : sycl::nan(0u);
  } else {
    // normal: (1 + man/4) * 2^(exp - bias), bias = 15
    v = (1.0f + static_cast<float>(man) * 0.25f) * sycl::ldexp(1.0f, exp - 15);
  }
  return sign ? -v : v;
}

#endif  // ARK_FP8_DECODE_USE_LUT

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
           // Vectorized path: process 16 K-elements at a time, which is
           // 8 packed weight bytes and a vec<ScalarT,16> activation block.
           // group_size is a multiple of 16 in every supported config
           // (group_size >= 32, even); a scalar tail loop covers leftovers.
           constexpr int CHUNK = 16;
           using ActVec = sycl::vec<uint16_t, CHUNK>;
           using PackVec = sycl::vec<uint8_t, CHUNK / 2>;
           static_assert(sizeof(ScalarT) == sizeof(uint16_t),
                         "ScalarT must be a 16-bit floating type");
           const int chunk_end = (group_size / CHUNK) * CHUNK;
           int kk = 0;
           for (; kk < chunk_end; kk += CHUNK) {
             const ActVec av = *reinterpret_cast<const ActVec*>(act_row + k_base + kk);
             const PackVec pv = *reinterpret_cast<const PackVec*>(w_row + (k_base + kk) / 2);
#pragma unroll
             for (int b = 0; b < CHUNK / 2; ++b) {
               const uint8_t packed = pv[b];
               float w0, w1;
               if constexpr (Asym) {
                 const int q0 = static_cast<int>(packed & 0x0F);
                 const int q1 = static_cast<int>((packed >> 4) & 0x0F);
                 w0 = (static_cast<float>(q0) - zero) * scale;
                 w1 = (static_cast<float>(q1) - zero) * scale;
               } else {
                 const int q0 = static_cast<int>(static_cast<int8_t>(packed << 4) >> 4);
                 const int q1 = static_cast<int>(static_cast<int8_t>(packed & 0xF0) >> 4);
                 w0 = static_cast<float>(q0) * scale;
                 w1 = static_cast<float>(q1) * scale;
               }
               const ScalarT a0 = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[2 * b]));
               const ScalarT a1 = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[2 * b + 1]));
               acc += static_cast<float>(a0) * w0;
               acc += static_cast<float>(a1) * w1;
             }
           }
           // Scalar tail for group_size not divisible by CHUNK.
           for (; kk < group_size; kk += 2) {
             const uint8_t packed = w_row[(k_base + kk) / 2];
             float w0, w1;
             if constexpr (Asym) {
               const int q0 = static_cast<int>(packed & 0x0F);
               const int q1 = static_cast<int>((packed >> 4) & 0x0F);
               w0 = (static_cast<float>(q0) - zero) * scale;
               w1 = (static_cast<float>(q1) - zero) * scale;
             } else {
               const int q0 = static_cast<int>(static_cast<int8_t>(packed << 4) >> 4);
               const int q1 = static_cast<int>(static_cast<int8_t>(packed & 0xF0) >> 4);
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
           // Vectorized path: 16 weights (16 bytes) + 16 activations per load.
           // group_size is typically 128 (mult of 16); scalar tail handles
           // anything that doesn't divide evenly.
           constexpr int CHUNK = 16;
           using ActVec = sycl::vec<uint16_t, CHUNK>;
           using ByteVec = sycl::vec<uint8_t, CHUNK>;
           static_assert(sizeof(ScalarT) == sizeof(uint16_t),
                         "ScalarT must be a 16-bit floating type");
           const int chunk_end = (group_size / CHUNK) * CHUNK;
           int kk = 0;
           for (; kk < chunk_end; kk += CHUNK) {
             const ActVec av = *reinterpret_cast<const ActVec*>(act_row + k_base + kk);
             const ByteVec wv = *reinterpret_cast<const ByteVec*>(w_row + k_base + kk);
#pragma unroll
             for (int u = 0; u < CHUNK; ++u) {
               const uint8_t raw = wv[u];
               float w;
               if constexpr (Asym) {
                 w = (static_cast<float>(raw) - zero) * scale;
               } else {
                 w = static_cast<float>(static_cast<int8_t>(raw)) * scale;
               }
               const ScalarT a = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[u]));
               acc += static_cast<float>(a) * w;
             }
           }
           for (; kk < group_size; ++kk) {
             const uint8_t raw = w_row[k_base + kk];
             float w;
             if constexpr (Asym) {
               w = (static_cast<float>(raw) - zero) * scale;
             } else {
               w = static_cast<float>(static_cast<int8_t>(raw)) * scale;
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
           // Vectorized: 16 K-elements per chunk = 4 packed bytes (4 values
           // each) plus a vec<uint16_t,16> activation block. group_size is a
           // multiple of 4 and typically 128 (mult of 16); scalar tail covers
           // any leftover. We load activations via uint16_t to stay portable
           // across SYCL implementations that may not provide
           // sycl::vec<bfloat16, N>.
           constexpr int CHUNK = 16;
           using ActVec = sycl::vec<uint16_t, CHUNK>;
           using PackVec = sycl::vec<uint8_t, CHUNK / 4>;
           static_assert(sizeof(ScalarT) == sizeof(uint16_t),
                         "ScalarT must be a 16-bit floating type");
           const int chunk_end = (group_size / CHUNK) * CHUNK;
           int kk = 0;
           for (; kk < chunk_end; kk += CHUNK) {
             const ActVec av = *reinterpret_cast<const ActVec*>(act_row + k_base + kk);
             const PackVec pv = *reinterpret_cast<const PackVec*>(w_row + (k_base + kk) / 4);
#pragma unroll
             for (int b = 0; b < CHUNK / 4; ++b) {
               const uint8_t packed = pv[b];
               float w0, w1, w2, w3;
               if constexpr (Asym) {
                 const int q0 = static_cast<int>(packed & 0x3);
                 const int q1 = static_cast<int>((packed >> 2) & 0x3);
                 const int q2 = static_cast<int>((packed >> 4) & 0x3);
                 const int q3 = static_cast<int>((packed >> 6) & 0x3);
                 w0 = (static_cast<float>(q0) - zero) * scale;
                 w1 = (static_cast<float>(q1) - zero) * scale;
                 w2 = (static_cast<float>(q2) - zero) * scale;
                 w3 = (static_cast<float>(q3) - zero) * scale;
               } else {
                 const int q0 = static_cast<int>(static_cast<int8_t>(packed << 6) >> 6);
                 const int q1 = static_cast<int>(static_cast<int8_t>((packed << 4) & 0xC0) >> 6);
                 const int q2 = static_cast<int>(static_cast<int8_t>((packed << 2) & 0xC0) >> 6);
                 const int q3 = static_cast<int>(static_cast<int8_t>(packed & 0xC0) >> 6);
                 w0 = static_cast<float>(q0) * scale;
                 w1 = static_cast<float>(q1) * scale;
                 w2 = static_cast<float>(q2) * scale;
                 w3 = static_cast<float>(q3) * scale;
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
           // Scalar tail (4 values per byte).
           for (; kk < group_size; kk += 4) {
             const uint8_t packed = w_row[(k_base + kk) / 4];
             float w[4];
             if constexpr (Asym) {
               const int q0 = static_cast<int>(packed & 0x3);
               const int q1 = static_cast<int>((packed >> 2) & 0x3);
               const int q2 = static_cast<int>((packed >> 4) & 0x3);
               const int q3 = static_cast<int>((packed >> 6) & 0x3);
               w[0] = (static_cast<float>(q0) - zero) * scale;
               w[1] = (static_cast<float>(q1) - zero) * scale;
               w[2] = (static_cast<float>(q2) - zero) * scale;
               w[3] = (static_cast<float>(q3) - zero) * scale;
             } else {
               const int q0 = static_cast<int>(static_cast<int8_t>(packed << 6) >> 6);
               const int q1 = static_cast<int>(static_cast<int8_t>((packed << 4) & 0xC0) >> 6);
               const int q2 = static_cast<int>(static_cast<int8_t>((packed << 2) & 0xC0) >> 6);
               const int q3 = static_cast<int>(static_cast<int8_t>(packed & 0xC0) >> 6);
               w[0] = static_cast<float>(q0) * scale;
               w[1] = static_cast<float>(q1) * scale;
               w[2] = static_cast<float>(q2) * scale;
               w[3] = static_cast<float>(q3) * scale;
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
// `decode_fp8_e4m3` / `decode_fp8_e5m2` helpers above, which can be compiled
// either as inline bit manipulation (default) or as a LUT lookup by defining
// `ARK_FP8_DECODE_USE_LUT`.
// ----------------------------------------------------------------------------
template <typename ScalarT, bool IsE4M3>
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

  q->parallel_for<MoEDecodeKernelFP8<ScalarT, IsE4M3>>(
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
           // Vectorized: 16 weights (16 bytes) + 16 activations per load.
           // Decode each FP8 byte to float inline, then apply the per-group
           // scale. group_size is typically 128 (mult of 16); scalar tail
           // covers anything that doesn't divide evenly.
           constexpr int CHUNK = 16;
           using ActVec = sycl::vec<uint16_t, CHUNK>;
           using ByteVec = sycl::vec<uint8_t, CHUNK>;
           static_assert(sizeof(ScalarT) == sizeof(uint16_t),
                         "ScalarT must be a 16-bit floating type");
           const int chunk_end = (group_size / CHUNK) * CHUNK;
           int kk = 0;
           for (; kk < chunk_end; kk += CHUNK) {
             const ActVec av = *reinterpret_cast<const ActVec*>(act_row + k_base + kk);
             const ByteVec wv = *reinterpret_cast<const ByteVec*>(w_row + k_base + kk);
#pragma unroll
             for (int u = 0; u < CHUNK; ++u) {
               const uint8_t raw = wv[u];
               float w;
               if constexpr (IsE4M3) {
                 w = decode_fp8_e4m3(raw) * scale;
               } else {
                 w = decode_fp8_e5m2(raw) * scale;
               }
               const ScalarT a = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[u]));
               acc += static_cast<float>(a) * w;
             }
           }
           for (; kk < group_size; ++kk) {
             const uint8_t raw = w_row[k_base + kk];
             float w;
             if constexpr (IsE4M3) {
               w = decode_fp8_e4m3(raw) * scale;
             } else {
               w = decode_fp8_e5m2(raw) * scale;
             }
             acc += static_cast<float>(act_row[k_base + kk]) * w;
           }
         }

         outputs[static_cast<size_t>(token) * N + n_global] = static_cast<ScalarT>(acc);
       });
}

}  // namespace moe_decode_detail

// ----------------------------------------------------------------------------
// Public API
//
// weight_dtype:
//   BTLA_DTYPE::F16  / BF16       : weights stored as [E, N, K] in matching
//                                   floating dtype, no scales/zeros needed
//   BTLA_DTYPE::S8                : int8 weights [E, N, K] (uint8 buffer,
//                                   interpreted as signed when asym==false,
//                                   unsigned with zero-points when asym==true)
//   BTLA_DTYPE::S4_CLIP           : packed int4 weights [E, N, K/2] (uint8),
//                                   scales [E, N, K/group_size] in act dtype,
//                                   zeros optional (asym==true requires it)
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
    } else if (act_dtype == BTLA_DTYPE::BF16) {
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
    } else {
      throw std::invalid_argument("moe_gemm_decode(int4): act_dtype must be FP16 or BF16");
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
    if (act_dtype == BTLA_DTYPE::F16) {
      if (is_e4m3) {
        moe_decode_detail::launch_fp8<sycl::half, true>(
            q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const sycl::half*>(scales), static_cast<sycl::half*>(outputs), expert_id_per_token_buf,
            total_tokens, N, K, group_size);
      } else {
        moe_decode_detail::launch_fp8<sycl::half, false>(
            q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const sycl::half*>(scales), static_cast<sycl::half*>(outputs), expert_id_per_token_buf,
            total_tokens, N, K, group_size);
      }
    } else if (act_dtype == BTLA_DTYPE::BF16) {
      using BF = sycl::ext::oneapi::bfloat16;
      if (is_e4m3) {
        moe_decode_detail::launch_fp8<BF, true>(
            q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const BF*>(scales), static_cast<BF*>(outputs), expert_id_per_token_buf, total_tokens, N, K,
            group_size);
      } else {
        moe_decode_detail::launch_fp8<BF, false>(
            q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const BF*>(scales), static_cast<BF*>(outputs), expert_id_per_token_buf, total_tokens, N, K,
            group_size);
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
