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
               int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b, int k_stride_s, int k_stride_d,
               int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s, int v_stride_h, int v_stride_b,
               int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int batch, int num_heads_q,
               int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim,
               float softmax_scale, bool is_causal,
               float* lse = nullptr);

void sdpa_impl_qks8_pvhalf(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask,
               int scale_block_size, void* qscale, void* kscale, int q_stride_s, int q_stride_d, int q_stride_h,
               int q_stride_b, int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d,
               int v_stride_s, int v_stride_h, int v_stride_b, int o_stride_s, int o_stride_d, int o_stride_h,
               int o_stride_b, int batch, int num_heads_q, int num_heads_kv, int seq_len_q,
               int seq_len_kv, int head_dim, float softmax_scale, bool is_causal,
               BTLA_DTYPE pv_dtype = BTLA_DTYPE::F16,
               float* lse = nullptr);

void sdpa_impl_qks8_pvi8(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask,
                         int scale_block_size, void* qscale, void* kscale, void* vscale, int q_stride_s,
                         int q_stride_d, int q_stride_h, int q_stride_b, int k_stride_s, int k_stride_d,
                         int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s, int v_stride_h,
                         int v_stride_b, int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b,
                         int batch, int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim,
                         float softmax_scale, bool is_causal, BTLA_DTYPE o_dtype = BTLA_DTYPE::F16,
                         float* lse = nullptr);

/**
 * @brief Flash Attention Prefill with variable-length sequences (no padding).
 *
 * Q/K/V are flat 3-D tensors: [total_tokens, num_heads, head_dim].
 * cu_seqlens_q/k are device-side int32 arrays of cumulative lengths.
 *
 * Internally dispatches to the same kernel infrastructure as sdpa_impl but
 * with varlen=true and cumulative_length pointers set on the problem shape,
 * so the kernel reads per-sequence offsets directly from device memory.
 */
void sdpa_varlen_impl(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask,
                      BTLA_DTYPE q_dtype, int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b,
                      int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d,
                      int v_stride_s, int v_stride_h, int v_stride_b, int o_stride_s, int o_stride_d,
                      int o_stride_h, int o_stride_b, int batch, int num_heads_q, int num_heads_kv,
                      int total_seqlen_q, int total_seqlen_kv, int max_seqlen_q, int max_seqlen_kv,
                      int head_dim, float softmax_scale, bool is_causal,
                      const int* cu_seqlens_q, const int* cu_seqlens_k,
                      float* lse = nullptr);

/**
 * @brief SAGE (INT8 Q/K) attention prefill with variable-length sequences.
 *
 * Q/K/V are flat 3-D tensors: [total_tokens, num_heads, head_dim].
 * cu_seqlens_q/k are device-side int32 arrays of cumulative lengths.
 *
 * Internally dispatches to the SAGE kernel with isVarLen=true and
 * cumulative_length pointers set on the problem shape.
 */
void sage_prefill_varlen(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask,
                         int scale_block_size, void* qscale, void* kscale, void* vscale, bool use_int8_pv,
                         BTLA_DTYPE q_dtype, BTLA_DTYPE pv_dtype,
                         int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b,
                         int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b,
                         int v_stride_d, int v_stride_s, int v_stride_h, int v_stride_b,
                         int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b,
                         int batch, int num_heads_q, int num_heads_kv,
                         int total_seqlen_q, int total_seqlen_kv, int max_seqlen_q, int max_seqlen_kv,
                         int head_dim, float softmax_scale, bool is_causal,
                         const int* cu_seqlens_q, const int* cu_seqlens_k,
                         float* lse = nullptr);
                         float softmax_scale, bool is_causal, BTLA_DTYPE o_dtype = BTLA_DTYPE::F16,
                         float* lse = nullptr);

void sdpa_impl_qks8_sparse_pvhalf(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask,
                                  int scale_block_size, void* qscale, void* kscale, void* lut,
                                  void* valid_block_num, int num_q_blocks, int num_k_blocks, int q_tile_override,
                                  int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b, int k_stride_s,
                                  int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s,
                                  int v_stride_h, int v_stride_b, int o_stride_s, int o_stride_d, int o_stride_h,
                                  int o_stride_b, int batch, int num_heads_q, int num_heads_kv, int seq_len_q,
                                  int seq_len_kv, int head_dim, float softmax_scale, bool is_causal,
                                  BTLA_DTYPE pv_dtype = BTLA_DTYPE::F16);

void sdpa_impl_qks8_sparse_row_linear_pvhalf(
    sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask, int scale_block_size,
    void* qscale, void* kscale, void* lut, void* valid_block_num, int num_q_blocks, int num_k_blocks,
    int q_tile_override, int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b, int k_stride_s,
    int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s, int v_stride_h, int v_stride_b,
    int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int batch, int num_heads_q, int num_heads_kv,
    int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale, bool is_causal,
    BTLA_DTYPE pv_dtype = BTLA_DTYPE::F16);

void sdpa_impl_qks8_sparse_qtile256_row64k_pvhalf(
    sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask, int scale_block_size,
    void* qscale, void* kscale, void* lut, void* valid_block_num, int num_q_blocks, int num_k_blocks,
    int q_tile_override, int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b, int k_stride_s,
    int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s, int v_stride_h, int v_stride_b,
    int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int batch, int num_heads_q, int num_heads_kv,
    int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale, bool is_causal,
    BTLA_DTYPE pv_dtype = BTLA_DTYPE::F16);

void sdpa_impl_qks8_sparse_decode_pvhalf(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* K_cache_ptr,
                                         void* V_cache_ptr, void* O_ptr, void* mask, int scale_block_size,
                                         void* qscale, void* kscale, void* lut, void* valid_block_num,
                                         int num_q_blocks, int num_k_blocks, int q_stride_s, int q_stride_d,
                                         int q_stride_h, int q_stride_b, int k_stride_s, int k_stride_d,
                                         int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s,
                                         int v_stride_h, int v_stride_b, int o_stride_s, int o_stride_d,
                                         int o_stride_h, int o_stride_b, int batch, int num_heads_q, int num_heads_kv,
                                         int seq_len_q, int seq_len_kv, int seq_len_kv_cache, int head_dim,
                                         float softmax_scale, bool is_causal, BTLA_DTYPE pv_dtype = BTLA_DTYPE::F16);
#endif  // ARK_SYCL_TLA

}  // namespace ark
