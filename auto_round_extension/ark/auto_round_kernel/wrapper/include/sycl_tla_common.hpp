// SYCL-TLA Flash Attention Common Definitions
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "bestla/bestla.h"
#ifdef ARK_XPU
#include <sycl/sycl.hpp>
#endif

namespace ark {

#if defined(ARK_SYCL_TLA)

/**
 * @brief MOE GEMM Grouped GEMM
 * Implementation in moe_gemm.cpp
 *
 * @param q SYCL queue
 * @param activations Input activations [total_tokens, K]
 * @param weights Expert weights [num_experts, N, K]
 * @param scales Optional scales (nullptr if not used)
 * @param outputs Output [total_tokens, N]
 * @param dtype Data type (FP16/BF16)
 * @param N Output feature dimension
 * @param K Input feature dimension
 * @param num_tokens_per_expert Array of token counts per expert [num_experts]
 * @param num_experts Number of experts
 */
void moe_gemm(sycl::queue* q, void* activations, void* weights, void* scales, void* outputs, BTLA_DTYPE dtype, int N,
              int K, int* num_tokens_per_expert, int num_experts);

/**
 * @brief MoE GEMV optimized for the decode phase (M per expert is typically
 * 1-2 tokens). Supports unquantized FP16/BF16 weights and int4 (S4_CLIP)
 * weights with group-wise scales and optional zero-points.
 *
 * Implementation is header-only in `sycl_tla_moe_decode.hpp`.
 *
 * @param q                       SYCL queue
 * @param activations             [total_tokens, K] in `act_dtype`
 * @param weights                 Unquantized: [num_experts, N, K] in act_dtype
 *                                Int4: packed [num_experts, N, K/2] uint8
 * @param scales                  [num_experts, N, K/group_size] (act_dtype),
 *                                ignored when weight_dtype is FP16/BF16
 * @param zeros                   [num_experts, N, K/group_size] (act_dtype) or
 *                                nullptr; required when asym==true
 * @param outputs                 [total_tokens, N] in act_dtype
 * @param expert_id_per_token_buf [total_tokens] int32 scratch buffer (device)
 * @param act_dtype               BTLA_DTYPE::F16 or BTLA_DTYPE::BF16
 * @param weight_dtype            BTLA_DTYPE::F16/BF16/S4_CLIP
 * @param N                       Output feature dim (must be multiple of 16)
 * @param K                       Input feature dim
 * @param group_size              Quantization group along K (int4 only); must
 *                                divide K and be even. Default 128.
 * @param num_tokens_per_expert   [num_experts] int32
 * @param num_experts             Number of experts
 * @param total_tokens            Sum of num_tokens_per_expert (== rows of
 *                                activations / outputs)
 * @param asym                    Whether int4 weights are asymmetric
 *                                (zeros required when true).
 */
void moe_gemm_decode(sycl::queue* q, void* activations, void* weights, void* scales, void* zeros, void* outputs,
                     int* expert_id_per_token_buf, BTLA_DTYPE act_dtype, BTLA_DTYPE weight_dtype, int N, int K,
                     int group_size, int* num_tokens_per_expert, int num_experts, int total_tokens, bool asym);

// ========================================================================
// Public API
// ========================================================================
/**
 * @brief Flash Attention Prefill (FP16)
 *
 * @param q SYCL queue
 * @param Q_ptr  Pointer to Q tensor [B, Hq, Sq, D]
 * @param K_ptr  Pointer to K tensor [B, Hkv, Skv, D]
 * @param V_ptr  Pointer to V tensor [B, Hkv, Skv, D]
 * @param O_ptr  Pointer to output tensor [B, Hq, Sq, D], fp32
 * @param mask  Pointer to attention mask tensor [B, 1, Sq, Skv], uint8 (0 for valid, 1 for masked)
 * @param q_dtype  Q/K/V data type (FP16)
 * @param batch  Batch size
 * @param num_heads_q  Number of query heads
 * @param num_heads_kv Number of KV heads
 * @param seq_len_q  Query sequence length
 * @param seq_len_kv  KV sequence length
 * @param head_dim  Head dimension (64 or 128)
 * @param softmax_scale  Softmax scale factor
 * @param is_causal  Whether to apply causal mask
 */
void sdpa_impl(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask, BTLA_DTYPE q_dtype,
               int batch, int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim,
               float softmax_scale, bool is_causal);

void sdpa_impl_qks8_pvhalf(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask, int scale_block_size, void* qscale,
               void* kscale, int batch, int num_heads_q, int num_heads_kv, int seq_len_q,
               int seq_len_kv, int head_dim, float softmax_scale, bool is_causal);

void sdpa_impl_qks8_pvi8(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask,
                         int scale_block_size, void* qscale, void* kscale, void* vscale, int batch, int num_heads_q,
                         int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale,
                         bool is_causal);
#endif  // ARK_SYCL_TLA

}  // namespace ark
