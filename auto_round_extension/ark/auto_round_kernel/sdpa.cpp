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

// Flash Attention Prefill implementation
// Separated from decode to avoid convert_type ODR violation in sycl-tla

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

#include <cute/numeric/int.hpp>
#include <cute/numeric/numeric_types.hpp>
#include <sycl/aliases.hpp>
#include "bestla/bestla.h"

#include <sycl/sycl.hpp>
#include "sycl_tla_sdpa.hpp"
#include "sdpa_kernel_declarations.hpp"  // generated from CMake, contains declarations for all kernel launchers

namespace ark {

namespace {

using KernelLauncher = int (*)(detail::Options const& options);

int launch_prefill_kernel_f16_128_sage(detail::Options const& options) {
  return launch_sage_prefill_kernel_128<cute::int8_t, cute::int8_t, cute::half_t>(options);
}

int launch_prefill_kernel_f16_128_sage_i8pv(detail::Options const& options) {
  return launch_sage_prefill_kernel_128<cute::int8_t, cute::int8_t, cute::int8_t, cute::half_t, true, true, true>(
      options);
}

int launch_prefill_kernel_f16_64_sage(detail::Options const& options) {
  return launch_sage_prefill_kernel_64<cute::int8_t, cute::int8_t, cute::half_t>(options);
}

int launch_prefill_kernel_f16_64_sage_i8pv(detail::Options const& options) {
  return launch_sage_prefill_kernel_64<cute::int8_t, cute::int8_t, cute::int8_t, cute::half_t, true, true, true>(
      options);
}

int launch_prefill_kernel_bf16_128_sage(detail::Options const& options) {
  return launch_sage_prefill_kernel_128<cute::int8_t, cute::int8_t, cute::bfloat16_t>(options);
}

int launch_prefill_kernel_bf16_128_sage_i8pv(detail::Options const& options) {
  return launch_sage_prefill_kernel_128<cute::int8_t, cute::int8_t, cute::int8_t, cute::bfloat16_t, true, true,
                                        true>(options);
}

int launch_prefill_kernel_bf16_64_sage(detail::Options const& options) {
  return launch_sage_prefill_kernel_64<cute::int8_t, cute::int8_t, cute::bfloat16_t>(options);
}

int launch_prefill_kernel_bf16_64_sage_i8pv(detail::Options const& options) {
  return launch_sage_prefill_kernel_64<cute::int8_t, cute::int8_t, cute::int8_t, cute::bfloat16_t, true, true,
                                        true>(options);
}

int launch_prefill_kernel_f16_128_sparse_sage(detail::Options const& options) {
  return launch_sparse_sage_prefill_kernel_128<cute::int8_t, cute::int8_t, cute::half_t>(options);
}

int launch_prefill_kernel_f16_64_sparse_sage(detail::Options const& options) {
  return launch_sparse_sage_prefill_kernel_64<cute::int8_t, cute::int8_t, cute::half_t>(options);
}

int launch_prefill_kernel_bf16_128_sparse_sage(detail::Options const& options) {
  return launch_sparse_sage_prefill_kernel_128<cute::int8_t, cute::int8_t, cute::bfloat16_t>(options);
}

int launch_prefill_kernel_bf16_64_sparse_sage(detail::Options const& options) {
  return launch_sparse_sage_prefill_kernel_64<cute::int8_t, cute::int8_t, cute::bfloat16_t>(options);
}

KernelLauncher select_sage_prefill_launcher(BTLA_DTYPE dtype, BTLA_DTYPE pv_dtype, int head_dim, bool use_int8_pv) {
  switch (dtype) {
    case BTLA_DTYPE::S8:
      switch (head_dim) {
        case 128:
          if (use_int8_pv) {
            if (pv_dtype == BTLA_DTYPE::BF16) return launch_prefill_kernel_bf16_128_sage_i8pv;
            return launch_prefill_kernel_f16_128_sage_i8pv;
          }
          if (pv_dtype == BTLA_DTYPE::BF16) return launch_prefill_kernel_bf16_128_sage;
          return launch_prefill_kernel_f16_128_sage;
        case 64:
          if (use_int8_pv) {
            if (pv_dtype == BTLA_DTYPE::BF16) return launch_prefill_kernel_bf16_64_sage_i8pv;
            return launch_prefill_kernel_f16_64_sage_i8pv;
          }
          if (pv_dtype == BTLA_DTYPE::BF16) return launch_prefill_kernel_bf16_64_sage;
          return launch_prefill_kernel_f16_64_sage;
        default:
          return nullptr;
      }

    default:
      return nullptr;
  }
}

KernelLauncher select_sage_prefill_launcher(BTLA_DTYPE dtype, int head_dim) {
  switch (dtype) {
    case BTLA_DTYPE::S8:
      switch (head_dim) {
        case 128:
          return launch_prefill_kernel_f16_128_sage;
        case 64:
          return launch_prefill_kernel_f16_64_sage;
        default:
          return nullptr;
      }
   
    default:
      return nullptr;
  }
}

KernelLauncher select_sparse_sage_prefill_launcher(BTLA_DTYPE pv_dtype, int head_dim) {
  // Sparse Sage reuses the dense dtype/head-dim family split, but always routes through the dedicated
  // sparse mainloop that consumes LUT metadata instead of walking dense KV tiles.
  switch (head_dim) {
    case 128:
      return pv_dtype == BTLA_DTYPE::BF16 ? launch_prefill_kernel_bf16_128_sparse_sage
                                          : launch_prefill_kernel_f16_128_sparse_sage;
    case 64:
      return pv_dtype == BTLA_DTYPE::BF16 ? launch_prefill_kernel_bf16_64_sparse_sage
                                          : launch_prefill_kernel_f16_64_sparse_sage;
    default:
      return nullptr;
  }
}


KernelLauncher select_prefill_launcher(BTLA_DTYPE dtype, int head_dim) {
  switch (dtype) {
    case BTLA_DTYPE::F16:
      switch (head_dim) {
        case 128:
          return detail::launch_prefill_kernel_f16_128;
        case 64:
          return detail::launch_prefill_kernel_f16_64;
        case 96:
          return detail::launch_prefill_kernel_f16_96;
        case 192:
          return detail::launch_prefill_kernel_f16_192;
        default:
          return nullptr;
      }
    case BTLA_DTYPE::BF16:
      switch (head_dim) {
        case 64:
          return detail::launch_prefill_kernel_bf16_64;
        case 96:
          return detail::launch_prefill_kernel_bf16_96;
        case 128:
          return detail::launch_prefill_kernel_bf16_128;
        case 192:
          return detail::launch_prefill_kernel_bf16_192;
        default:
          return nullptr;
      }
    default:
      return nullptr;
  }
}

KernelLauncher select_decode_launcher(BTLA_DTYPE dtype, int head_dim) {
  switch (dtype) {
    case BTLA_DTYPE::F16:
      switch (head_dim) {
        case 128:
          return detail::launch_decode_kernel_f16_128;
        case 64:
          return detail::launch_decode_kernel_f16_64;
        case 96:
          return detail::launch_decode_kernel_f16_96;
        case 192:
          return detail::launch_decode_kernel_f16_192;
        default:
          return nullptr;
      }
    case BTLA_DTYPE::BF16:
      switch (head_dim) {
        case 64:
          return detail::launch_decode_kernel_bf16_64;
        case 96:
          return detail::launch_decode_kernel_bf16_96;
        case 128:
          return detail::launch_decode_kernel_bf16_128;
        case 192:
          return detail::launch_decode_kernel_bf16_192;
        default:
          return nullptr;
      }
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

void sage_prefill(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask,
          int scale_block_size, void* qscale, void* kscale, void* vscale, bool use_int8_pv,
          BTLA_DTYPE q_dtype, BTLA_DTYPE pv_dtype, int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b,
          int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s,
          int v_stride_h, int v_stride_b, int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b,
          int batch, int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim,
          float softmax_scale, bool is_causal) {
  detail::Options options =
    make_common_options(Q_ptr, K_ptr, V_ptr, O_ptr, mask, q_stride_s, q_stride_d, q_stride_h, q_stride_b,
              k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
              v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q,
              num_heads_kv, seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
  options.scale_block_size = scale_block_size;
  options.qscale = qscale;
  options.kscale = kscale;
  options.vscale = vscale;
  compat::set_default_queue(*q);

  KernelLauncher launcher = select_sage_prefill_launcher(q_dtype, pv_dtype, head_dim, use_int8_pv);
  if (launcher == nullptr) {
    throw std::runtime_error(
        "Unsupported dtype or head dimension for sage_prefill / SAGE (only F16/BF16 PV and 64/128 are supported)");
  }

  launcher(options);
}

void sparse_sage_prefill(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask,
                         int scale_block_size, void* qscale, void* kscale, void* lut, void* valid_block_num,
                         int num_q_blocks, int num_k_blocks, BTLA_DTYPE pv_dtype, int q_stride_s, int q_stride_d,
                         int q_stride_h, int q_stride_b, int k_stride_s, int k_stride_d, int k_stride_h,
                         int k_stride_b, int v_stride_d, int v_stride_s, int v_stride_h, int v_stride_b,
                         int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int batch, int num_heads_q,
                         int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale,
                         bool is_causal) {
  detail::Options options =
      make_common_options(Q_ptr, K_ptr, V_ptr, O_ptr, mask, q_stride_s, q_stride_d, q_stride_h, q_stride_b,
                          k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
                          v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv,
                          seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
  options.scale_block_size = scale_block_size;
  options.qscale = qscale;
  options.kscale = kscale;
  options.lut = static_cast<int const*>(lut);
  options.valid_block_num = static_cast<int const*>(valid_block_num);
  options.num_q_blocks = num_q_blocks;
  options.num_k_blocks = num_k_blocks;
  compat::set_default_queue(*q);

  KernelLauncher launcher = select_sparse_sage_prefill_launcher(pv_dtype, head_dim);
  if (launcher == nullptr) {
    throw std::runtime_error(
        "Unsupported dtype or head dimension for sparse_sage_prefill (only F16/BF16 PV and 64/128 are supported)");
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
  detail::Options options =
      make_common_options(Q_ptr, K_ptr, V_ptr, O_ptr, mask, q_stride_s, q_stride_d, q_stride_h, q_stride_b,
                          k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
                          v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv,
                          seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
  options.scale_block_size = scale_block_size;
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

  KernelLauncher launcher = select_sparse_sage_prefill_launcher(pv_dtype, head_dim);
  if (launcher == nullptr) {
    throw std::runtime_error(
        "Unsupported dtype or head dimension for sparse_sage_decode (only F16/BF16 PV and 64/128 are supported)");
  }

  launcher(options);
}

void flash_attn_prefill(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask,
            BTLA_DTYPE q_dtype, int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b,
            int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d,
            int v_stride_s, int v_stride_h, int v_stride_b, int o_stride_s, int o_stride_d,
            int o_stride_h, int o_stride_b, int batch, int num_heads_q, int num_heads_kv,
            int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale, bool is_causal) {
  detail::Options options =
    make_common_options(Q_ptr, K_ptr, V_ptr, O_ptr, mask, q_stride_s, q_stride_d, q_stride_h, q_stride_b,
              k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
              v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q,
              num_heads_kv, seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
  compat::set_default_queue(*q);

  KernelLauncher launcher = select_prefill_launcher(q_dtype, head_dim);
  if (launcher == nullptr) {
    throw std::runtime_error(
        "Unsupported dtype or head dimension for flash_attn_prefill (only F16/BF16 and 64/96/128/192 are supported)");
  }

  launcher(options);
}

void flash_attn_decode(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask,
             BTLA_DTYPE q_dtype, int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b,
             int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d,
             int v_stride_s, int v_stride_h, int v_stride_b, int o_stride_s, int o_stride_d,
             int o_stride_h, int o_stride_b, int batch, int num_heads_q, int num_heads_kv,
             int seq_len_kv, int head_dim, float softmax_scale, bool is_causal) {
  detail::Options options =
    make_common_options(Q_ptr, K_ptr, V_ptr, O_ptr, mask, q_stride_s, q_stride_d, q_stride_h, q_stride_b,
              k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
              v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q,
              num_heads_kv, 1, seq_len_kv, head_dim, softmax_scale, is_causal);
  compat::set_default_queue(*q);

  KernelLauncher launcher = select_decode_launcher(q_dtype, head_dim);
  if (launcher == nullptr) {
    throw std::runtime_error(
        "Unsupported dtype or head dimension for flash_attn_decode (only F16/BF16 and 64/96/128/192 are supported)");
  }

  launcher(options);
}

void sdpa_impl(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask, BTLA_DTYPE q_dtype,
               int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b, int k_stride_s, int k_stride_d,
               int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s, int v_stride_h, int v_stride_b,
               int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int batch, int num_heads_q,
               int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale,
               bool is_causal) {
  //  if (q_dtype != BTLA_DTYPE::F16 && q_dtype != BTLA_DTYPE::BF16) {
  //   throw std::invalid_argument("sdpa_impl: only FP16 and BF16 are supported");
  // }
  if (mask && is_causal) {
    throw std::invalid_argument("sdpa_impl: mask and is_causal cannot both be set");
  }

  if (seq_len_q <= 0 || seq_len_kv <= 0) {
    throw std::invalid_argument("sdpa_impl: seq_len_q and seq_len_kv must be greater than 0");
  }
  if (seq_len_q == 1) {
    flash_attn_decode(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, q_dtype, q_stride_s, q_stride_d, q_stride_h, q_stride_b,
                      k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
                      v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv,
                      seq_len_kv, head_dim, softmax_scale, is_causal);
    return;
  }

  flash_attn_prefill(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, q_dtype, q_stride_s, q_stride_d, q_stride_h, q_stride_b,
                     k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
                     v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv,
                     seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
}

void sdpa_impl_qks8_pvhalf(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask,
               int scale_block_size, void* qscale, void* kscale, int q_stride_s, int q_stride_d, int q_stride_h,
               int q_stride_b, int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d,
               int v_stride_s, int v_stride_h, int v_stride_b, int o_stride_s, int o_stride_d, int o_stride_h,
               int o_stride_b, int batch, int num_heads_q, int num_heads_kv, int seq_len_q,
               int seq_len_kv, int head_dim, float softmax_scale, bool is_causal, BTLA_DTYPE pv_dtype) {
  if (mask && is_causal) {
    throw std::invalid_argument("sdpa_impl: mask and is_causal cannot both be set");
  }

  if (seq_len_q <= 0 || seq_len_kv <= 0) {
    throw std::invalid_argument("sdpa_impl: seq_len_q and seq_len_kv must be greater than 0");
  }
  if (pv_dtype != BTLA_DTYPE::F16 && pv_dtype != BTLA_DTYPE::BF16) {
    throw std::invalid_argument("sdpa_impl_qks8_pvhalf: only F16 and BF16 are supported for V/O dtype");
  }
  if (seq_len_q == 1) {
    flash_attn_decode(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, pv_dtype, q_stride_s, q_stride_d, q_stride_h,
                      q_stride_b, k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s,
                      v_stride_h, v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q,
                      num_heads_kv, seq_len_kv, head_dim, softmax_scale, is_causal);
    return;
  }

  sage_prefill(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, scale_block_size, qscale, kscale, nullptr, false,
                     BTLA_DTYPE::S8, pv_dtype, q_stride_s, q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d,
                     k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h, v_stride_b, o_stride_s,
                     o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv, seq_len_q,
                     seq_len_kv, head_dim, softmax_scale, is_causal);
}

void sdpa_impl_qks8_pvi8(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask,
                         int scale_block_size, void* qscale, void* kscale, void* vscale, int q_stride_s,
                         int q_stride_d, int q_stride_h, int q_stride_b, int k_stride_s, int k_stride_d,
                         int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s, int v_stride_h,
                         int v_stride_b, int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b,
                         int batch, int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv,
                         int head_dim, float softmax_scale, bool is_causal, BTLA_DTYPE o_dtype) {
  if (mask && is_causal) {
    throw std::invalid_argument("sdpa_impl: mask and is_causal cannot both be set");
  }

  if (seq_len_q <= 0 || seq_len_kv <= 0) {
    throw std::invalid_argument("sdpa_impl: seq_len_q and seq_len_kv must be greater than 0");
  }
  if (vscale == nullptr) {
    throw std::invalid_argument("sdpa_impl_qks8_pvi8: vscale must be provided for int8 PV");
  }
  if (o_dtype != BTLA_DTYPE::F16 && o_dtype != BTLA_DTYPE::BF16) {
    throw std::invalid_argument("sdpa_impl_qks8_pvi8: only F16 and BF16 are supported for output dtype");
  }
  if (seq_len_q == 1) {
    flash_attn_decode(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, o_dtype, q_stride_s, q_stride_d, q_stride_h,
                      q_stride_b, k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s,
                      v_stride_h, v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q,
                      num_heads_kv, seq_len_kv, head_dim, softmax_scale, is_causal);
    return;
  }

  sage_prefill(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, scale_block_size, qscale, kscale, vscale, true,
               BTLA_DTYPE::S8, o_dtype, q_stride_s, q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d,
               k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h, v_stride_b, o_stride_s, o_stride_d,
               o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim,
               softmax_scale, is_causal);
}

void sdpa_impl_qks8_sparse_pvhalf(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask,
                                  int scale_block_size, void* qscale, void* kscale, void* lut, void* valid_block_num,
                                  int num_q_blocks, int num_k_blocks, int q_stride_s, int q_stride_d, int q_stride_h,
                                  int q_stride_b, int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b,
                                  int v_stride_d, int v_stride_s, int v_stride_h, int v_stride_b, int o_stride_s,
                                  int o_stride_d, int o_stride_h, int o_stride_b, int batch, int num_heads_q,
                                  int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale,
                                  bool is_causal, BTLA_DTYPE pv_dtype) {
  if (mask && is_causal) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_pvhalf: mask and is_causal cannot both be set");
  }
  if (seq_len_q <= 0 || seq_len_kv <= 0) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_pvhalf: seq_len_q and seq_len_kv must be greater than 0");
  }
  if (pv_dtype != BTLA_DTYPE::F16 && pv_dtype != BTLA_DTYPE::BF16) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_pvhalf: only F16 and BF16 are supported for V/O dtype");
  }
  if (qscale == nullptr || kscale == nullptr) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_pvhalf: qscale and kscale must be provided");
  }
  if (lut == nullptr || valid_block_num == nullptr) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_pvhalf: lut and valid_block_num must be provided");
  }
  if (num_q_blocks <= 0 || num_k_blocks <= 0) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_pvhalf: num_q_blocks and num_k_blocks must be greater than 0");
  }
  if (scale_block_size <= 0) {
    throw std::invalid_argument("sdpa_impl_qks8_sparse_pvhalf: scale_block_size must be greater than 0");
  }

  sparse_sage_prefill(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, scale_block_size, qscale, kscale, lut, valid_block_num,
                      num_q_blocks, num_k_blocks, pv_dtype, q_stride_s, q_stride_d, q_stride_h, q_stride_b,
                      k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
                      v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv,
                      seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
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
