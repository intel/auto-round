// SYCL MoE Decode Kernel
//
// GEMV-style MoE kernel optimized for the decode phase, where each expert
// typically processes only 1-2 tokens (top-k routing with batch size 1).
//
// Layout convention (caller already sorted activations per expert,
// identical to the prefill `moe_gemm` interface):
//   - activations:           [total_tokens, K]            row-major
//   - weights (fp/bf16):     [num_experts, N, K]          row-major
//   - weights (int4 packed): [num_experts, N, K/2]        row-major, two
//                            4-bit values per byte (low nibble at lower K)
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
#include <stdexcept>

#include "bestla/bestla/bestla.h"

#ifdef ARK_XPU
#include <sycl/sycl.hpp>
#endif

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

// ----------------------------------------------------------------------------
// Build a [total_tokens] -> expert_id mapping from num_tokens_per_expert.
// Runs on host (num_experts is small, total_tokens is small in decode).
// Caller-managed buffer (USM device allocation) keeps host noise out of the
// hot path; here we just fill it via a tiny SYCL kernel for simplicity.
// ----------------------------------------------------------------------------
inline void fill_expert_id_per_token(sycl::queue* q, int* expert_id_per_token,
                                     const int* num_tokens_per_expert, int num_experts,
                                     int total_tokens) {
  // Sequential prefix-scan on a single thread; cheap because num_experts is
  // small (typ. <= 256) and we avoid host-device sync entirely.
  q->single_task([=]() {
     int offset = 0;
     for (int e = 0; e < num_experts; ++e) {
       int n = num_tokens_per_expert[e];
       for (int i = 0; i < n; ++i) {
         if (offset + i < total_tokens) {
           expert_id_per_token[offset + i] = e;
         }
       }
       offset += n;
     }
   }).wait();
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
         // Unroll by 8 to hide latency; arbitrary K (any multiple of 8).
         int k = 0;
         constexpr int UNROLL = 8;
         for (; k + UNROLL <= K; k += UNROLL) {
#pragma unroll
           for (int u = 0; u < UNROLL; ++u) {
             acc += static_cast<float>(act_row[k + u]) * static_cast<float>(w_row[k + u]);
           }
         }
         for (; k < K; ++k) {
           acc += static_cast<float>(act_row[k]) * static_cast<float>(w_row[k]);
         }

         outputs[static_cast<size_t>(token) * N + n_global] = static_cast<ScalarT>(acc);
       })
      .wait();
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
           // Two nibbles per byte; iterate in pairs.
           for (int kk = 0; kk < group_size; kk += 2) {
             const uint8_t packed = w_row[(k_base + kk) / 2];
             float w0, w1;
             if constexpr (Asym) {
               const int q0 = static_cast<int>(packed & 0x0F);
               const int q1 = static_cast<int>((packed >> 4) & 0x0F);
               w0 = (static_cast<float>(q0) - zero) * scale;
               w1 = (static_cast<float>(q1) - zero) * scale;
             } else {
               // Sign-extend each nibble from 4-bit signed to 8-bit signed.
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
       })
      .wait();
}

}  // namespace moe_decode_detail

// ----------------------------------------------------------------------------
// Public API
//
// weight_dtype:
//   BTLA_DTYPE::F16  / BF16       : weights stored as [E, N, K] in matching
//                                   floating dtype, no scales/zeros needed
//   BTLA_DTYPE::S4_CLIP           : packed int4 weights [E, N, K/2] (uint8),
//                                   scales [E, N, K/group_size] in act dtype,
//                                   zeros optional (asym==true requires it)
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

  throw std::invalid_argument("moe_gemm_decode: unsupported weight_dtype (supported: F16, BF16, S4_CLIP)");
}

}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
