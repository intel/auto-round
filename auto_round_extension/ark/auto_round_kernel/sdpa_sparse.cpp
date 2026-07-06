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

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

#include <cute/numeric/int.hpp>
#include <cute/numeric/numeric_types.hpp>
#include <sycl/aliases.hpp>
#include "bestla/bestla.h"

#include <sycl/sycl.hpp>
#include "sycl_tla_sdpa_sparse.hpp"

namespace ark {

namespace detail = sparse_detail;

namespace {

using KernelLauncher = int (*)(detail::Options const& options);
constexpr int kSparseProfileModeFull = 0;

int launch_prefill_kernel_f16_128_sparse_sage(detail::Options const& options) {
  return detail::launch_sparse_sage_prefill_kernel_128<cute::int8_t, cute::int8_t, cute::half_t>(options);
}

int launch_prefill_kernel_f16_128_sparse_sage_qtile128(detail::Options const& options) {
  return detail::launch_sparse_sage_prefill_kernel_128_qtile128<cute::int8_t, cute::int8_t, cute::half_t>(options);
}

int launch_prefill_kernel_f16_128_sparse_sage_qtile64(detail::Options const& options) {
  return detail::launch_sparse_sage_prefill_kernel_128_qtile64<cute::int8_t, cute::int8_t, cute::half_t>(options);
}

int launch_prefill_kernel_f16_64_sparse_sage(detail::Options const& options) {
  return detail::launch_sparse_sage_prefill_kernel_64<cute::int8_t, cute::int8_t, cute::half_t>(options);
}

int launch_prefill_kernel_bf16_128_sparse_sage(detail::Options const& options) {
  return detail::launch_sparse_sage_prefill_kernel_128<cute::int8_t, cute::int8_t, cute::bfloat16_t>(options);
}

int launch_prefill_kernel_bf16_128_sparse_sage_qtile128(detail::Options const& options) {
  return detail::launch_sparse_sage_prefill_kernel_128_qtile128<cute::int8_t, cute::int8_t, cute::bfloat16_t>(options);
}

int launch_prefill_kernel_bf16_128_sparse_sage_qtile64(detail::Options const& options) {
  return detail::launch_sparse_sage_prefill_kernel_128_qtile64<cute::int8_t, cute::int8_t, cute::bfloat16_t>(options);
}

int launch_prefill_kernel_bf16_64_sparse_sage(detail::Options const& options) {
  return detail::launch_sparse_sage_prefill_kernel_64<cute::int8_t, cute::int8_t, cute::bfloat16_t>(options);
}

KernelLauncher select_sparse_sage_prefill_launcher(BTLA_DTYPE pv_dtype, int head_dim, int q_tile_override,
                                                   int sparse_profile_mode = kSparseProfileModeFull) {
  if (sparse_profile_mode != kSparseProfileModeFull) {
    return nullptr;
  }

  switch (head_dim) {
    case 128:
      if (q_tile_override == 64) {
        return pv_dtype == BTLA_DTYPE::BF16 ? launch_prefill_kernel_bf16_128_sparse_sage_qtile64
                                            : launch_prefill_kernel_f16_128_sparse_sage_qtile64;
      }
      if (q_tile_override == 128) {
        return pv_dtype == BTLA_DTYPE::BF16 ? launch_prefill_kernel_bf16_128_sparse_sage_qtile128
                                            : launch_prefill_kernel_f16_128_sparse_sage_qtile128;
      }
      if (q_tile_override != 0 && q_tile_override != 256) return nullptr;
      return pv_dtype == BTLA_DTYPE::BF16 ? launch_prefill_kernel_bf16_128_sparse_sage
                                          : launch_prefill_kernel_f16_128_sparse_sage;
    case 64:
      if (q_tile_override != 0 && q_tile_override != 128) return nullptr;
      return pv_dtype == BTLA_DTYPE::BF16 ? launch_prefill_kernel_bf16_64_sparse_sage
                                          : launch_prefill_kernel_f16_64_sparse_sage;
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

}  // namespace

void sparse_sage_prefill(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask,
                         int scale_block_size, void* qscale, void* kscale, void* lut, void* valid_block_num,
                         int num_q_blocks, int num_k_blocks, int q_tile_override, BTLA_DTYPE pv_dtype, int q_stride_s,
                         int q_stride_d, int q_stride_h, int q_stride_b, int k_stride_s, int k_stride_d,
                         int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s, int v_stride_h,
                         int v_stride_b, int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int batch,
                         int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim,
                         float softmax_scale, bool is_causal, int sparse_q_block_size = 0) {
  const int effective_q_tile_override = (head_dim == 128 && q_tile_override == 0) ? 64 : q_tile_override;
  detail::Options options =
      make_common_options(Q_ptr, K_ptr, V_ptr, O_ptr, mask, q_stride_s, q_stride_d, q_stride_h, q_stride_b,
                          k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
                          v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q,
                          num_heads_kv, seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
  options.scale_block_size = scale_block_size;
  options.sparse_q_block_size = sparse_q_block_size;
  options.q_tile_override = effective_q_tile_override;
  options.qscale = qscale;
  options.kscale = kscale;
  options.lut = static_cast<int const*>(lut);
  options.valid_block_num = static_cast<int const*>(valid_block_num);
  options.num_q_blocks = num_q_blocks;
  options.num_k_blocks = num_k_blocks;
  compat::set_default_queue(*q);

  KernelLauncher launcher = select_sparse_sage_prefill_launcher(pv_dtype, head_dim, effective_q_tile_override);
  if (launcher == nullptr) {
    throw std::runtime_error("Unsupported sparse_sage_prefill config");
  }

  launcher(options);
}

void sparse_sage_decode(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* K_cache_ptr, void* V_cache_ptr,
                        void* O_ptr, void* mask, int scale_block_size, void* qscale, void* kscale, void* lut,
                        void* valid_block_num, int num_q_blocks, int num_k_blocks, BTLA_DTYPE pv_dtype, int q_stride_s,
                        int q_stride_d, int q_stride_h, int q_stride_b, int k_stride_s, int k_stride_d,
                        int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s, int v_stride_h,
                        int v_stride_b, int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int batch,
                        int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv, int seq_len_kv_cache,
                        int head_dim, float softmax_scale, bool is_causal) {
  const int effective_q_tile_override = head_dim == 128 ? 64 : 0;
  detail::Options options =
      make_common_options(Q_ptr, K_ptr, V_ptr, O_ptr, mask, q_stride_s, q_stride_d, q_stride_h, q_stride_b,
                          k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
                          v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q,
                          num_heads_kv, seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
  options.scale_block_size = scale_block_size;
  options.q_tile_override = effective_q_tile_override;
  options.qscale = qscale;
  options.kscale = kscale;
  options.lut = static_cast<int const*>(lut);
  options.valid_block_num = static_cast<int const*>(valid_block_num);
  options.num_q_blocks = num_q_blocks;
  options.num_k_blocks = num_k_blocks;
  options.block_K = K_cache_ptr;
  options.block_V = V_cache_ptr;
  options.seq_len_kv_cache = seq_len_kv_cache;
  compat::set_default_queue(*q);

  KernelLauncher launcher = select_sparse_sage_prefill_launcher(pv_dtype, head_dim, effective_q_tile_override);
  if (launcher == nullptr) {
    throw std::runtime_error(
        "Unsupported dtype or head dimension for sparse_sage_decode (only F16/BF16 PV and 64/128 are supported)");
  }

  launcher(options);
}

void sdpa_impl_qks8_sparse_d64_pvhalf(
    sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask, int scale_block_size,
    void* qscale, void* kscale, void* lut, void* valid_block_num, int num_q_blocks, int num_k_blocks,
    int q_tile_override, int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b, int k_stride_s,
    int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s, int v_stride_h, int v_stride_b,
    int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int batch, int num_heads_q, int num_heads_kv,
    int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale, bool is_causal, BTLA_DTYPE pv_dtype) {
  if (mask && is_causal) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_d64_pvhalf: mask and is_causal cannot both be set");
  }
  if (seq_len_q <= 0 || seq_len_kv <= 0) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_d64_pvhalf: seq_len_q and seq_len_kv must be greater than 0");
  }
  if (pv_dtype != BTLA_DTYPE::F16 && pv_dtype != BTLA_DTYPE::BF16) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_d64_pvhalf: only F16 and BF16 are supported for V/O dtype");
  }
  if (qscale == nullptr || kscale == nullptr) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_d64_pvhalf: qscale and kscale must be provided");
  }
  if (lut == nullptr || valid_block_num == nullptr) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_d64_pvhalf: lut and valid_block_num must be provided");
  }
  if (num_q_blocks <= 0 || num_k_blocks <= 0) {
    throw std::invalid_argument(
        "sdpa_impl_qks8_sparse_d64_pvhalf: num_q_blocks and num_k_blocks must be greater than 0");
  }
  if (scale_block_size <= 0) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_d64_pvhalf: scale_block_size must be greater than 0");
  }
  if (head_dim != 64) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_d64_pvhalf: head_dim must be 64");
  }

  sparse_sage_prefill(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, scale_block_size, qscale, kscale, lut, valid_block_num,
                      num_q_blocks, num_k_blocks, q_tile_override, pv_dtype, q_stride_s, q_stride_d, q_stride_h,
                      q_stride_b, k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
                      v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv,
                      seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
}

void sdpa_impl_qks8_sparse_row_linear_pvhalf(
    sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask, int scale_block_size,
    void* qscale, void* kscale, void* lut, void* valid_block_num, int num_q_blocks, int num_k_blocks,
    int q_tile_override, int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b, int k_stride_s,
    int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s, int v_stride_h, int v_stride_b,
    int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int batch, int num_heads_q, int num_heads_kv,
    int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale, bool is_causal, BTLA_DTYPE pv_dtype) {
  if (q_tile_override != 0 && q_tile_override != 64) {
    throw std::invalid_argument(
        "sdpa_impl_qks8_sparse_row_linear_pvhalf: q_tile_override must be 0 or 64 for the row-linear backend");
  }
  if (scale_block_size != 64) {
    throw std::invalid_argument(
        "sdpa_impl_qks8_sparse_row_linear_pvhalf: scale_block_size must be 64 so one sparse row maps to one workgroup");
  }
  if (mask && is_causal) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_row_linear_pvhalf: mask and is_causal cannot both be set");
  }
  if (seq_len_q <= 0 || seq_len_kv <= 0) {
    throw std::invalid_argument(
        "sdpa_impl_qks8_sparse_row_linear_pvhalf: seq_len_q and seq_len_kv must be greater than 0");
  }
  if (pv_dtype != BTLA_DTYPE::F16 && pv_dtype != BTLA_DTYPE::BF16) {
    throw std::invalid_argument(
        "sdpa_impl_qks8_sparse_row_linear_pvhalf: only F16 and BF16 are supported for V/O dtype");
  }
  if (qscale == nullptr || kscale == nullptr) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_row_linear_pvhalf: qscale and kscale must be provided");
  }
  if (lut == nullptr || valid_block_num == nullptr) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_row_linear_pvhalf: lut and valid_block_num must be provided");
  }
  if (num_q_blocks <= 0 || num_k_blocks <= 0) {
    throw std::invalid_argument(
        "sdpa_impl_qks8_sparse_row_linear_pvhalf: num_q_blocks and num_k_blocks must be greater than 0");
  }

  sparse_sage_prefill(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, scale_block_size, qscale, kscale, lut, valid_block_num,
                      num_q_blocks, num_k_blocks, 64, pv_dtype, q_stride_s, q_stride_d, q_stride_h, q_stride_b,
                      k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h, v_stride_b,
                      o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv, seq_len_q,
                      seq_len_kv, head_dim, softmax_scale, is_causal);
}

void sdpa_impl_qks8_sparse_qtile256_row64k_pvhalf(
    sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask, int scale_block_size,
    void* qscale, void* kscale, void* lut, void* valid_block_num, int num_q_blocks, int num_k_blocks,
    int q_tile_override, int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b, int k_stride_s,
    int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s, int v_stride_h, int v_stride_b,
    int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int batch, int num_heads_q, int num_heads_kv,
    int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale, bool is_causal, BTLA_DTYPE pv_dtype) {
  if (q_tile_override != 256) {
    throw std::invalid_argument(
        "sdpa_impl_qks8_sparse_qtile256_row64k_pvhalf: q_tile_override must be 256 for the decoupled backend");
  }
  if (scale_block_size != 64) {
    throw std::invalid_argument(
        "sdpa_impl_qks8_sparse_qtile256_row64k_pvhalf: scale_block_size must be 64 for the decoupled backend");
  }
  if (head_dim != 128) {
    throw std::invalid_argument(
        "sdpa_impl_qks8_sparse_qtile256_row64k_pvhalf: head_dim must be 128 for the decoupled backend");
  }
  sparse_sage_prefill(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, scale_block_size, qscale, kscale, lut, valid_block_num,
                      num_q_blocks, num_k_blocks, q_tile_override, pv_dtype, q_stride_s, q_stride_d, q_stride_h,
                      q_stride_b, k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
                      v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv,
                      seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal, /*sparse_q_block_size=*/256);
}

void sdpa_impl_qks8_sparse_decode_pvhalf(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* K_cache_ptr,
                                         void* V_cache_ptr, void* O_ptr, void* mask, int scale_block_size,
                                         void* qscale, void* kscale, void* lut, void* valid_block_num,
                                         int num_q_blocks, int num_k_blocks, int q_stride_s, int q_stride_d,
                                         int q_stride_h, int q_stride_b, int k_stride_s, int k_stride_d,
                                         int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s,
                                         int v_stride_h, int v_stride_b, int o_stride_s, int o_stride_d,
                                         int o_stride_h, int o_stride_b, int batch, int num_heads_q, int num_heads_kv,
                                         int seq_len_q, int seq_len_kv, int seq_len_kv_cache, int head_dim,
                                         float softmax_scale, bool is_causal, BTLA_DTYPE pv_dtype) {
  if (mask && is_causal) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_decode_pvhalf: mask and is_causal cannot both be set");
  }
  if (seq_len_q != 1) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_decode_pvhalf: only seq_len_q == 1 is supported in v1");
  }
  if (seq_len_kv <= 0 || seq_len_kv_cache <= 0) {
    throw std::invalid_argument(
        "sdpa_impl_qks8_sparse_decode_pvhalf: seq_len_kv and seq_len_kv_cache must be greater than 0");
  }
  if (pv_dtype != BTLA_DTYPE::F16 && pv_dtype != BTLA_DTYPE::BF16) {
    throw std::invalid_argument(
        "sdpa_impl_qks8_sparse_decode_pvhalf: only F16 and BF16 are supported for V/O dtype");
  }
  if (qscale == nullptr || kscale == nullptr) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_decode_pvhalf: qscale and kscale must be provided");
  }
  if (lut == nullptr || valid_block_num == nullptr) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_decode_pvhalf: lut and valid_block_num must be provided");
  }
  if (K_cache_ptr == nullptr || V_cache_ptr == nullptr) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_decode_pvhalf: K_cache and V_cache must be provided");
  }
  if (num_q_blocks != 1 || num_k_blocks <= 0) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_decode_pvhalf: only one Q block row is supported in v1");
  }
  if (scale_block_size <= 0) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_decode_pvhalf: scale_block_size must be greater than 0");
  }

  sparse_sage_decode(q, Q_ptr, K_ptr, V_ptr, K_cache_ptr, V_cache_ptr, O_ptr, mask, scale_block_size, qscale, kscale,
                     lut, valid_block_num, num_q_blocks, num_k_blocks, pv_dtype, q_stride_s, q_stride_d, q_stride_h,
                     q_stride_b, k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
                     v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv,
                     seq_len_q, seq_len_kv, seq_len_kv_cache, head_dim, softmax_scale, is_causal);
}

}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
