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

/**
 * @brief MoE Grouped GEMM optimized for the prefill phase, supporting the
 * same set of weight encodings as `moe_gemm_decode` (FP16/BF16, INT8 sym/asym,
 * INT4 sym/asym, INT2 sym/asym, FP8 E4M3/E5M2).
 *
 * Stage-1 implementation: dequantizes weights into a `[num_experts, K, N]`
 * temporary buffer (must be supplied by the caller via `dequant_workspace`,
 * sized `num_experts * K * N * sizeof(act_dtype)`) and then dispatches to the
 * existing `moe_gemm` baseline. This guarantees numerical parity with the
 * decode path. Mainloop fusion is the follow-up perf-tuning step.
 *
 * Implementation is header-only in `sycl_tla_moe_mixed.hpp`.
 *
 * Layout convention (matches `moe_gemm_decode`):
 *   - activations:           [total_tokens, K]      in act_dtype
 *   - weights (quantized):   [num_experts, N, K_p]  uint8 (decode-style packed)
 *   - weights (FP16/BF16):   [num_experts, K, N]    in act_dtype (matches `moe_gemm`)
 *   - scales:                [num_experts, N, K/group_size] in act_dtype
 *   - zeros (asym only):     [num_experts, N, K/group_size] in act_dtype
 *   - dequant_workspace:     [num_experts, K, N]    in act_dtype, may be null
 *                            for the unquantized fast path
 *   - outputs:               [total_tokens, N]      in act_dtype
 */
void moe_gemm_prefill(sycl::queue* q, void* activations, void* weights, void* scales, void* zeros, void* outputs,
                      void* dequant_workspace, BTLA_DTYPE act_dtype, BTLA_DTYPE weight_dtype, int N, int K,
                      int group_size, int* num_tokens_per_expert, int num_experts, int total_tokens, bool asym);

/**
 * @brief MoE prefill Grouped GEMM -- FP8 per-tensor mixed-input DPAS
 * (Variant A of the vllm-xpu-kernels FP8 port).
 *
 * Weights `[num_experts, K, N]` row-major (vllm convention, one FP8 byte
 * per element). Scales are `[num_experts]` FP32 (one per-tensor scale per
 * expert). Activations / outputs match `moe_gemm_prefill`. `act_dtype` is
 * `F16` or `BF16`; `weight_dtype` is `F8_E4M3` or `F8_E5M2`.
 *
 * STATUS: NEEDS-HARDWARE-VALIDATION. See
 * `sycl_tla_moe_prefill_fp8_dpas.hpp` for the port's provenance & the
 * on-hardware TODOs.
 *
 * Implementation is header-only in `sycl_tla_moe_prefill_fp8_dpas.hpp`.
 */
void moe_gemm_prefill_fp8_dpas(sycl::queue* q, void* activations, void* weights, void* scales, void* outputs,
                               BTLA_DTYPE act_dtype, BTLA_DTYPE weight_dtype, int N, int K,
                               int* num_tokens_per_expert, int num_experts, int total_tokens);

/**
 * @brief MoE prefill Grouped GEMM -- INT8 per-tensor mixed-input DPAS
 * (Variant A, sibling of `moe_gemm_prefill_fp8_dpas`).
 *
 * Weights `[num_experts, K, N]` row-major (vllm convention, one signed byte
 * per element). Scales are `[num_experts]` FP32 (one per-tensor scale per
 * expert). Activations / outputs are FP16 or BF16; `weight_dtype` must be
 * `BTLA_DTYPE::S8`.
 *
 * Storage-only INT8: the DPAS atom still runs `act_dtype x act_dtype ->
 * fp32`; the mainloop upcasts each INT8 weight byte to `act_dtype` in
 * register before feeding DPAS. The per-tensor scale is folded once per
 * output element in the epilogue.
 *
 * STATUS: NEEDS-HARDWARE-VALIDATION. See
 * `sycl_tla_moe_prefill_int_dpas.hpp` for the port's provenance & the
 * on-hardware TODOs.
 *
 * Implementation is header-only in `sycl_tla_moe_prefill_int_dpas.hpp`.
 */
void moe_gemm_prefill_int_dpas(sycl::queue* q, void* activations, void* weights, void* scales, void* outputs,
                               BTLA_DTYPE act_dtype, BTLA_DTYPE weight_dtype, int N, int K,
                               int* num_tokens_per_expert, int num_experts, int total_tokens);

/**
 * @brief W4A8 MoE -- one-shot AUTO_S8 weight prepack.
 *
 * Converts auto-round's packed int4-sym MoE weights `[E, N, K/2]` (uint8, two
 * nibbles per byte) plus per-group scales `[E, N, K/group_size]` (act dtype)
 * into int8 weights `[E, N, K]` and FP32 block scales
 * `[E, N, K/rescale_block]`, using ARK's `AUTO_S8` re-scale formula
 * (`xpu_wrapper.hpp`): `sxt = max|s| * 8 / 127`, `w8 = round(w4 * s / sxt)`.
 *
 * `rescale_group_size <= 0` (the `group=-1` spelling) selects one scale per
 * output channel, which lets the GEMM run a single full-K int32 accumulation.
 * Use `moe_w4a8_rescale_block_size` to resolve the effective block size (and
 * therefore the `wscales` shape) before allocating.
 *
 * STATUS: NEEDS-HARDWARE-VALIDATION. Implementation is header-only in
 * `sycl_tla_moe_w4a8.hpp`.
 */
void moe_w4a8_prepack(sycl::queue* q, void* weights_s4, void* scales, void* weights_s8, void* wscales,
                      BTLA_DTYPE act_dtype, int num_experts, int N, int K, int group_size,
                      int rescale_group_size);

/**
 * @brief W4A8 MoE GEMM -- int4 weights, int8 compute, dynamically quantized
 * int8 activations. Covers both the prefill (grouped GEMM, `s8 x s8 -> s32`
 * DPAS) and decode (GEMV) phases.
 *
 * Activations `[total_tokens, K]` in `act_dtype` are quantized per token to
 * int8 on entry; `weights_s8` / `wscales` come from `moe_w4a8_prepack`.
 *
 *   - activations : [total_tokens, K]           act dtype (sorted by expert)
 *   - weights_s8  : [num_experts, N, K]         int8
 *   - wscales     : [num_experts, N, K/rescale_block] float
 *   - outputs     : [total_tokens, N]           act dtype
 *
 * @param rescale_block_size Effective AUTO_S8 block size (see
 *        `moe_w4a8_rescale_block_size`); must be a multiple of 64 dividing K.
 * @param phase 0 = auto (decode when the batch is small), 1 = force decode,
 *        2 = force prefill.
 * @param qact_in Optional `[total_tokens, K]` int8 activations already
 *        quantized by the caller, with `ascale_in` their `[total_tokens]` fp32
 *        per-row scales (`absmax / 127`). Supplying both skips the internal
 *        quantization pass and `activations` is then unused; neither may be
 *        given without the other.
 * @param row_to_token / routing_weights / fused_out / fused_batch Optional
 *        fused top-k reduction: the epilogue scales each row and scatter-adds
 *        it into the zeroed `[fused_batch, N]` fp32 `fused_out` instead of
 *        writing the unreduced `[total_tokens, N]` to `outputs`. Prefill only,
 *        and all four must be given together.
 *
 * STATUS: NEEDS-HARDWARE-VALIDATION. Implementation is header-only in
 * `sycl_tla_moe_w4a8.hpp`, which is also where the trailing optional
 * parameters get their defaults.
 */
void moe_gemm_w4a8(sycl::queue* q, void* activations, void* weights_s8, void* wscales, void* outputs,
                   BTLA_DTYPE act_dtype, int N, int K, int rescale_block_size, int* num_tokens_per_expert,
                   int num_experts, int total_tokens, int phase, const void* qact_in, const float* ascale_in,
                   const int* row_to_token, const float* routing_weights, float* fused_out, int fused_batch);

/**
 * @brief Resolve the effective W4A8 AUTO_S8 re-scale block size for a given
 * K / group_size, honouring `ARK_MOE_W4A8_AUTO_S8`. Returns K (one scale per
 * output channel) for `rescale_group_size <= 0` or any unusable value.
 */
int moe_w4a8_rescale_block_size(int K, int group_size, int rescale_group_size);

/**
 * @brief Release the W4A8 activation-quantization / expert-map scratch slabs.
 */
void moe_w4a8_release_scratch();

// ========================================================================
// Public API
// ========================================================================
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

void sdpa_impl_qks8_sparse_d64_pvhalf(
    sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask, int scale_block_size,
    void* qscale, void* kscale, void* lut, void* valid_block_num, int num_q_blocks, int num_k_blocks,
    int q_tile_override, int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b, int k_stride_s,
    int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s, int v_stride_h, int v_stride_b,
    int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int batch, int num_heads_q, int num_heads_kv,
    int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale, bool is_causal,
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
#endif  // ARK_SYCL_TLA

}  // namespace ark
