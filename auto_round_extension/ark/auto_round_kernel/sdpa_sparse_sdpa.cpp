//  Copyright (c) 2026 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

// Independent native-precision sparse SDPA path (BF16 + FP16). This dispatches to
// the sparse SDPA mainloop (SPARSESDPAFwdMainloop via SparseSDPAConfig), separate
// from the INT8-centric sparse SAGE path in sdpa_sparse.cpp / sdpa_sparse_bf16.cpp.

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

#include <cute/numeric/numeric_types.hpp>
#include <sycl/aliases.hpp>
#include "bestla/bestla.h"

#include <sycl/sycl.hpp>
#include "sycl_tla_sdpa_sparse.hpp"

namespace ark {

namespace detail = sparse_detail;

namespace {

using KernelLauncher = int (*)(detail::Options const& options);

int launch_prefill_kernel_bf16_128_sparse_sdpa(detail::Options const& options) {
  return detail::launch_sparse_sdpa_prefill_kernel_128<cute::bfloat16_t, cute::bfloat16_t, cute::bfloat16_t>(options);
}

int launch_prefill_kernel_bf16_128_sparse_sdpa_qtile64(detail::Options const& options) {
  return detail::launch_sparse_sdpa_prefill_kernel_128_qtile64<
      cute::bfloat16_t, cute::bfloat16_t, cute::bfloat16_t>(options);
}

int launch_prefill_kernel_bf16_64_sparse_sdpa(detail::Options const& options) {
  return detail::launch_sparse_sdpa_prefill_kernel_64<cute::bfloat16_t, cute::bfloat16_t, cute::bfloat16_t>(options);
}

int launch_prefill_kernel_f16_128_sparse_sdpa(detail::Options const& options) {
  return detail::launch_sparse_sdpa_prefill_kernel_128<cute::half_t, cute::half_t, cute::half_t>(options);
}

int launch_prefill_kernel_f16_128_sparse_sdpa_qtile64(detail::Options const& options) {
  return detail::launch_sparse_sdpa_prefill_kernel_128_qtile64<cute::half_t, cute::half_t, cute::half_t>(options);
}

int launch_prefill_kernel_f16_64_sparse_sdpa(detail::Options const& options) {
  return detail::launch_sparse_sdpa_prefill_kernel_64<cute::half_t, cute::half_t, cute::half_t>(options);
}

KernelLauncher select_sparse_sdpa_prefill_launcher(BTLA_DTYPE q_dtype, int head_dim, int q_tile_override) {
  switch (head_dim) {
    case 128:
      if (q_tile_override == 64) {
        return q_dtype == BTLA_DTYPE::BF16 ? launch_prefill_kernel_bf16_128_sparse_sdpa_qtile64
                                           : launch_prefill_kernel_f16_128_sparse_sdpa_qtile64;
      }
      if (q_tile_override != 0 && q_tile_override != 256) return nullptr;
      return q_dtype == BTLA_DTYPE::BF16 ? launch_prefill_kernel_bf16_128_sparse_sdpa
                                         : launch_prefill_kernel_f16_128_sparse_sdpa;
    case 64:
      if (q_tile_override != 0 && q_tile_override != 64 && q_tile_override != 128) return nullptr;
      return q_dtype == BTLA_DTYPE::BF16 ? launch_prefill_kernel_bf16_64_sparse_sdpa
                                         : launch_prefill_kernel_f16_64_sparse_sdpa;
    default:
      return nullptr;
  }
}

detail::Options make_common_options(void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask, int q_stride_s,
                                    int q_stride_d, int q_stride_h, int q_stride_b, int k_stride_s, int k_stride_d,
                                    int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s, int v_stride_h,
                                    int v_stride_b, int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b,
                                    int batch, int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv,
                                    int head_dim, float softmax_scale, bool is_causal) {
  if (q_stride_d != 1 || k_stride_d != 1 || v_stride_d != 1 || o_stride_d != 1) {
    throw std::invalid_argument("make_common_options: head-dim stride must be 1 for Q/K/V/O");
  }
  detail::Options options;
  options.q = Q_ptr;
  options.k = K_ptr;
  options.v = V_ptr;
  options.mask = mask;
  options.o = O_ptr;
  options.use_tensor_strides = true;
  options.q_stride_s = q_stride_s;
  options.q_stride_d = q_stride_d;
  options.q_stride_h = q_stride_h;
  options.q_stride_b = q_stride_b;
  options.k_stride_s = k_stride_s;
  options.k_stride_d = k_stride_d;
  options.k_stride_h = k_stride_h;
  options.k_stride_b = k_stride_b;
  options.v_stride_d = v_stride_d;
  options.v_stride_s = v_stride_s;
  options.v_stride_h = v_stride_h;
  options.v_stride_b = v_stride_b;
  options.o_stride_s = o_stride_s;
  options.o_stride_d = o_stride_d;
  options.o_stride_h = o_stride_h;
  options.o_stride_b = o_stride_b;
  options.batch = batch;
  options.num_heads_q = num_heads_q;
  options.num_heads_kv = num_heads_kv;
  options.seq_len_qo = seq_len_q;
  options.seq_len_kv = seq_len_kv;
  options.head_size_qk = head_dim;
  options.head_size_vo = head_dim;
  options.softmax_scale = softmax_scale;
  options.is_causal = is_causal;
  return options;
}

void sparse_sdpa_prefill(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask, void* lut,
                         void* valid_block_num, int num_q_blocks, int num_k_blocks, int q_tile_override,
                         int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b, int k_stride_s,
                         int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s,
                         int v_stride_h, int v_stride_b, int o_stride_s, int o_stride_d, int o_stride_h,
                         int o_stride_b, int batch, int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv,
                         int head_dim, float softmax_scale, bool is_causal, BTLA_DTYPE q_dtype,
                         int sparse_q_block_size = 0) {
  const int effective_q_tile_override =
      (head_dim == 128 && q_tile_override == 0) ? 64 : ((head_dim == 64 && q_tile_override == 0) ? 64 : q_tile_override);
  detail::Options options =
      make_common_options(Q_ptr, K_ptr, V_ptr, O_ptr, mask, q_stride_s, q_stride_d, q_stride_h, q_stride_b,
                          k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
                          v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q,
                          num_heads_kv, seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
  // Native-precision path: no INT8 dequant scales, softmax scale applied directly.
  options.scale_block_size = 0;
  options.sparse_q_block_size = sparse_q_block_size;
  options.q_tile_override = effective_q_tile_override;
  options.qscale = nullptr;
  options.kscale = nullptr;
  options.vscale = nullptr;
  options.lut = static_cast<int const*>(lut);
  options.valid_block_num = static_cast<int const*>(valid_block_num);
  options.num_q_blocks = num_q_blocks;
  options.num_k_blocks = num_k_blocks;
  compat::set_default_queue(*q);

  KernelLauncher launcher = select_sparse_sdpa_prefill_launcher(q_dtype, head_dim, effective_q_tile_override);
  if (launcher == nullptr) {
    throw std::runtime_error("Unsupported sparse_sdpa_prefill config");
  }

  launcher(options);
}

}  // namespace

void sdpa_impl_bf16_sparse_sdpa_d64(
    sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask, void* lut, void* valid_block_num,
    int num_q_blocks, int num_k_blocks, int q_tile_override, int q_stride_s, int q_stride_d, int q_stride_h,
    int q_stride_b, int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s,
    int v_stride_h, int v_stride_b, int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int batch,
    int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale,
    bool is_causal) {
  if (mask && is_causal) {
    throw std::invalid_argument("sdpa_impl_bf16_sparse_sdpa_d64: mask and is_causal cannot both be set");
  }
  if (seq_len_q <= 0 || seq_len_kv <= 0) {
    throw std::invalid_argument("sdpa_impl_bf16_sparse_sdpa_d64: seq_len_q and seq_len_kv must be greater than 0");
  }
  if (lut == nullptr || valid_block_num == nullptr) {
    throw std::invalid_argument("sdpa_impl_bf16_sparse_sdpa_d64: lut and valid_block_num must be provided");
  }
  if (num_q_blocks <= 0 || num_k_blocks <= 0) {
    throw std::invalid_argument("sdpa_impl_bf16_sparse_sdpa_d64: num_q_blocks and num_k_blocks must be greater than 0");
  }
  if (head_dim != 64) {
    throw std::invalid_argument("sdpa_impl_bf16_sparse_sdpa_d64: head_dim must be 64");
  }

  sparse_sdpa_prefill(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, lut, valid_block_num, num_q_blocks, num_k_blocks,
                      q_tile_override, q_stride_s, q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d,
                      k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h, v_stride_b, o_stride_s, o_stride_d,
                      o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim,
                      softmax_scale, is_causal, BTLA_DTYPE::BF16, /*sparse_q_block_size=*/64);
}

void sdpa_impl_fp16_sparse_sdpa_d64(
    sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask, void* lut, void* valid_block_num,
    int num_q_blocks, int num_k_blocks, int q_tile_override, int q_stride_s, int q_stride_d, int q_stride_h,
    int q_stride_b, int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s,
    int v_stride_h, int v_stride_b, int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int batch,
    int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale,
    bool is_causal) {
  if (mask && is_causal) {
    throw std::invalid_argument("sdpa_impl_fp16_sparse_sdpa_d64: mask and is_causal cannot both be set");
  }
  if (seq_len_q <= 0 || seq_len_kv <= 0) {
    throw std::invalid_argument("sdpa_impl_fp16_sparse_sdpa_d64: seq_len_q and seq_len_kv must be greater than 0");
  }
  if (lut == nullptr || valid_block_num == nullptr) {
    throw std::invalid_argument("sdpa_impl_fp16_sparse_sdpa_d64: lut and valid_block_num must be provided");
  }
  if (num_q_blocks <= 0 || num_k_blocks <= 0) {
    throw std::invalid_argument("sdpa_impl_fp16_sparse_sdpa_d64: num_q_blocks and num_k_blocks must be greater than 0");
  }
  if (head_dim != 64) {
    throw std::invalid_argument("sdpa_impl_fp16_sparse_sdpa_d64: head_dim must be 64");
  }

  sparse_sdpa_prefill(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, lut, valid_block_num, num_q_blocks, num_k_blocks,
                      q_tile_override, q_stride_s, q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d,
                      k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h, v_stride_b, o_stride_s, o_stride_d,
                      o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim,
                      softmax_scale, is_causal, BTLA_DTYPE::F16, /*sparse_q_block_size=*/64);
}

void sdpa_impl_bf16_sparse_sdpa_row_linear(
    sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask, void* lut, void* valid_block_num,
    int num_q_blocks, int num_k_blocks, int q_tile_override, int q_stride_s, int q_stride_d, int q_stride_h,
    int q_stride_b, int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s,
    int v_stride_h, int v_stride_b, int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int batch,
    int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale,
    bool is_causal) {
  if (q_tile_override != 0 && q_tile_override != 64) {
    throw std::invalid_argument("sdpa_impl_bf16_sparse_sdpa_row_linear: q_tile_override must be 0 or 64");
  }
  if (mask && is_causal) {
    throw std::invalid_argument("sdpa_impl_bf16_sparse_sdpa_row_linear: mask and is_causal cannot both be set");
  }
  if (seq_len_q <= 0 || seq_len_kv <= 0) {
    throw std::invalid_argument(
        "sdpa_impl_bf16_sparse_sdpa_row_linear: seq_len_q and seq_len_kv must be greater than 0");
  }
  if (lut == nullptr || valid_block_num == nullptr) {
    throw std::invalid_argument("sdpa_impl_bf16_sparse_sdpa_row_linear: lut and valid_block_num must be provided");
  }
  if (num_q_blocks <= 0 || num_k_blocks <= 0) {
    throw std::invalid_argument(
        "sdpa_impl_bf16_sparse_sdpa_row_linear: num_q_blocks and num_k_blocks must be greater than 0");
  }

  sparse_sdpa_prefill(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, lut, valid_block_num, num_q_blocks, num_k_blocks, 64,
                      q_stride_s, q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d, k_stride_h,
                      k_stride_b, v_stride_d, v_stride_s, v_stride_h, v_stride_b, o_stride_s, o_stride_d, o_stride_h,
                      o_stride_b, batch, num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim, softmax_scale,
                      is_causal, BTLA_DTYPE::BF16, /*sparse_q_block_size=*/64);
}

void sdpa_impl_fp16_sparse_sdpa_row_linear(
    sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask, void* lut, void* valid_block_num,
    int num_q_blocks, int num_k_blocks, int q_tile_override, int q_stride_s, int q_stride_d, int q_stride_h,
    int q_stride_b, int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s,
    int v_stride_h, int v_stride_b, int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int batch,
    int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale,
    bool is_causal) {
  if (q_tile_override != 0 && q_tile_override != 64) {
    throw std::invalid_argument("sdpa_impl_fp16_sparse_sdpa_row_linear: q_tile_override must be 0 or 64");
  }
  if (mask && is_causal) {
    throw std::invalid_argument("sdpa_impl_fp16_sparse_sdpa_row_linear: mask and is_causal cannot both be set");
  }
  if (seq_len_q <= 0 || seq_len_kv <= 0) {
    throw std::invalid_argument(
        "sdpa_impl_fp16_sparse_sdpa_row_linear: seq_len_q and seq_len_kv must be greater than 0");
  }
  if (lut == nullptr || valid_block_num == nullptr) {
    throw std::invalid_argument("sdpa_impl_fp16_sparse_sdpa_row_linear: lut and valid_block_num must be provided");
  }
  if (num_q_blocks <= 0 || num_k_blocks <= 0) {
    throw std::invalid_argument(
        "sdpa_impl_fp16_sparse_sdpa_row_linear: num_q_blocks and num_k_blocks must be greater than 0");
  }

  sparse_sdpa_prefill(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, lut, valid_block_num, num_q_blocks, num_k_blocks, 64,
                      q_stride_s, q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d, k_stride_h,
                      k_stride_b, v_stride_d, v_stride_s, v_stride_h, v_stride_b, o_stride_s, o_stride_d, o_stride_h,
                      o_stride_b, batch, num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim, softmax_scale,
                      is_causal, BTLA_DTYPE::F16, /*sparse_q_block_size=*/64);
}

void sdpa_impl_bf16_sparse_sdpa_qtile256_row64k(
    sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask, void* lut, void* valid_block_num,
    int num_q_blocks, int num_k_blocks, int q_tile_override, int q_stride_s, int q_stride_d, int q_stride_h,
    int q_stride_b, int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s,
    int v_stride_h, int v_stride_b, int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int batch,
    int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale,
    bool is_causal) {
  if (q_tile_override != 256) {
    throw std::invalid_argument("sdpa_impl_bf16_sparse_sdpa_qtile256_row64k: q_tile_override must be 256");
  }
  if (head_dim != 128) {
    throw std::invalid_argument("sdpa_impl_bf16_sparse_sdpa_qtile256_row64k: head_dim must be 128");
  }

  sparse_sdpa_prefill(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, lut, valid_block_num, num_q_blocks, num_k_blocks,
                      q_tile_override, q_stride_s, q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d,
                      k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h, v_stride_b, o_stride_s, o_stride_d,
                      o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim,
                      softmax_scale, is_causal, BTLA_DTYPE::BF16, /*sparse_q_block_size=*/256);
}

void sdpa_impl_fp16_sparse_sdpa_qtile256_row64k(
    sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask, void* lut, void* valid_block_num,
    int num_q_blocks, int num_k_blocks, int q_tile_override, int q_stride_s, int q_stride_d, int q_stride_h,
    int q_stride_b, int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s,
    int v_stride_h, int v_stride_b, int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int batch,
    int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale,
    bool is_causal) {
  if (q_tile_override != 256) {
    throw std::invalid_argument("sdpa_impl_fp16_sparse_sdpa_qtile256_row64k: q_tile_override must be 256");
  }
  if (head_dim != 128) {
    throw std::invalid_argument("sdpa_impl_fp16_sparse_sdpa_qtile256_row64k: head_dim must be 128");
  }

  sparse_sdpa_prefill(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, lut, valid_block_num, num_q_blocks, num_k_blocks,
                      q_tile_override, q_stride_s, q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d,
                      k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h, v_stride_b, o_stride_s, o_stride_d,
                      o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim,
                      softmax_scale, is_causal, BTLA_DTYPE::F16, /*sparse_q_block_size=*/256);
}

}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
