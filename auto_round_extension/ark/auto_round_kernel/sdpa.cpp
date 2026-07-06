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
#include "utils.hpp"

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
          float softmax_scale, bool is_causal, float* lse = nullptr) {
  detail::Options options =
    make_common_options(Q_ptr, K_ptr, V_ptr, O_ptr, mask, q_stride_s, q_stride_d, q_stride_h, q_stride_b,
              k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
              v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q,
              num_heads_kv, seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
  options.scale_block_size = scale_block_size;
  options.qscale = qscale;
  options.kscale = kscale;
  options.vscale = vscale;
  options.lse = lse;
  compat::set_default_queue(*q);

  KernelLauncher launcher = select_sage_prefill_launcher(q_dtype, pv_dtype, head_dim, use_int8_pv);
  if (launcher == nullptr) {
    throw std::runtime_error(
        "Unsupported dtype or head dimension for sage_prefill / SAGE (only F16/BF16 PV and 64/128 are supported)");
  }

  launcher(options);
}

void flash_attn_prefill(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask,
            BTLA_DTYPE q_dtype, int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b,
            int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d,
            int v_stride_s, int v_stride_h, int v_stride_b, int o_stride_s, int o_stride_d,
            int o_stride_h, int o_stride_b, int batch, int num_heads_q, int num_heads_kv,
            int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale, bool is_causal,
            float* lse = nullptr) {
  detail::Options options =
    make_common_options(Q_ptr, K_ptr, V_ptr, O_ptr, mask, q_stride_s, q_stride_d, q_stride_h, q_stride_b,
              k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
              v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q,
              num_heads_kv, seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
  options.lse = lse;
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
             int seq_len_kv, int head_dim, float softmax_scale, bool is_causal,
             float* lse = nullptr) {
  detail::Options options =
    make_common_options(Q_ptr, K_ptr, V_ptr, O_ptr, mask, q_stride_s, q_stride_d, q_stride_h, q_stride_b,
              k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
              v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q,
              num_heads_kv, 1, seq_len_kv, head_dim, softmax_scale, is_causal);
  options.lse = lse;
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
               bool is_causal, float* lse = nullptr) {
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
                      seq_len_kv, head_dim, softmax_scale, is_causal, lse);
    return;
  }

  flash_attn_prefill(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, q_dtype, q_stride_s, q_stride_d, q_stride_h, q_stride_b,
                     k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
                     v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv,
                     seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal, lse);
}

void sdpa_impl_qks8_pvhalf(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask,
               int scale_block_size, void* qscale, void* kscale, int q_stride_s, int q_stride_d, int q_stride_h,
               int q_stride_b, int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d,
               int v_stride_s, int v_stride_h, int v_stride_b, int o_stride_s, int o_stride_d, int o_stride_h,
               int o_stride_b, int batch, int num_heads_q, int num_heads_kv, int seq_len_q,
               int seq_len_kv, int head_dim, float softmax_scale, bool is_causal, BTLA_DTYPE pv_dtype,
               float* lse = nullptr) {
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
                      num_heads_kv, seq_len_kv, head_dim, softmax_scale, is_causal, lse);
    return;
  }

  sage_prefill(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, scale_block_size, qscale, kscale, nullptr, false,
                     BTLA_DTYPE::S8, pv_dtype, q_stride_s, q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d,
                     k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h, v_stride_b, o_stride_s,
                     o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv, seq_len_q,
                     seq_len_kv, head_dim, softmax_scale, is_causal, lse);
}

void sdpa_impl_qks8_pvi8(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask,
                         int scale_block_size, void* qscale, void* kscale, void* vscale, int q_stride_s,
                         int q_stride_d, int q_stride_h, int q_stride_b, int k_stride_s, int k_stride_d,
                         int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s, int v_stride_h,
                         int v_stride_b, int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b,
                         int batch, int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv,
                         int head_dim, float softmax_scale, bool is_causal, BTLA_DTYPE o_dtype,
                         float* lse = nullptr) {
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
                      num_heads_kv, seq_len_kv, head_dim, softmax_scale, is_causal, lse);
    return;
  }

  sage_prefill(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, scale_block_size, qscale, kscale, vscale, true,
               BTLA_DTYPE::S8, o_dtype, q_stride_s, q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d,
               k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h, v_stride_b, o_stride_s, o_stride_d,
               o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim,
               softmax_scale, is_causal, lse);
}

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
                         float* lse = nullptr) {
  if (mask && is_causal) {
    throw std::invalid_argument("sage_prefill_varlen: mask and is_causal cannot both be set");
  }
  if (batch <= 0 || total_seqlen_q <= 0 || total_seqlen_kv <= 0) {
    throw std::invalid_argument("sage_prefill_varlen: batch, total_seqlen_q, total_seqlen_kv must be > 0");
  }
  if (cu_seqlens_q == nullptr || cu_seqlens_k == nullptr) {
    throw std::invalid_argument("sage_prefill_varlen: cu_seqlens_q and cu_seqlens_k must not be null");
  }

  detail::Options options = make_common_options(
      Q_ptr, K_ptr, V_ptr, O_ptr, mask,
      q_stride_s, q_stride_d, q_stride_h, q_stride_b,
      k_stride_s, k_stride_d, k_stride_h, k_stride_b,
      v_stride_d, v_stride_s, v_stride_h, v_stride_b,
      o_stride_s, o_stride_d, o_stride_h, o_stride_b,
      batch, num_heads_q, num_heads_kv,
      max_seqlen_q, max_seqlen_kv,
      head_dim, softmax_scale, is_causal);
  options.varlen = true;
  options.total_seqlen_q = total_seqlen_q;
  options.total_seqlen_kv = total_seqlen_kv;
  options.max_seqlen_q = max_seqlen_q;
  options.max_seqlen_kv = max_seqlen_kv;
  options.cu_seqlens_q = cu_seqlens_q;
  options.cu_seqlens_k = cu_seqlens_k;
  options.seq_len_kv_cache = 0;
  options.total_seqlen_kv_cache = 0;
  options.max_seqlen_kv_cache = 0;
  options.scale_block_size = scale_block_size;
  options.qscale = qscale;
  options.kscale = kscale;
  options.vscale = vscale;
  options.lse = lse;

  compat::set_default_queue(*q);

  // Zero-filled workspace for cu_seqlens_kv_cache via DnnlContext scratch pool.
  int* zero_cu_buf = static_cast<int*>(
      DnnlContext::Instance()->get_scratch_mem((batch + 1) * sizeof(int), 3, q));
  q->memset(zero_cu_buf, 0, (batch + 1) * sizeof(int));
  options.cu_seqlens_kv_cache = zero_cu_buf;
  options.use_tensor_strides = true;

  KernelLauncher launcher = select_sage_prefill_launcher(q_dtype, pv_dtype, head_dim, use_int8_pv);
  if (launcher == nullptr) {
    throw std::runtime_error(
        "Unsupported dtype or head dimension for sage_prefill_varlen (only S8/64/128)");
  }

  launcher(options);
}

void sdpa_varlen_impl(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask,
                      BTLA_DTYPE q_dtype, int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b,
                      int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d,
                      int v_stride_s, int v_stride_h, int v_stride_b, int o_stride_s, int o_stride_d,
                      int o_stride_h, int o_stride_b, int batch, int num_heads_q, int num_heads_kv,
                      int total_seqlen_q, int total_seqlen_kv, int max_seqlen_q, int max_seqlen_kv,
                      int head_dim, float softmax_scale, bool is_causal,
                      const int* cu_seqlens_q, const int* cu_seqlens_k,
                      float* lse = nullptr) {
  if (mask && is_causal) {
    throw std::invalid_argument("sdpa_varlen_impl: mask and is_causal cannot both be set");
  }
  if (q_dtype != BTLA_DTYPE::F16 && q_dtype != BTLA_DTYPE::BF16) {
    throw std::invalid_argument("sdpa_varlen_impl: only F16 and BF16 are supported");
  }
  if (batch <= 0 || total_seqlen_q <= 0 || total_seqlen_kv <= 0) {
    throw std::invalid_argument("sdpa_varlen_impl: batch, total_seqlen_q, total_seqlen_kv must be > 0");
  }
  if (cu_seqlens_q == nullptr || cu_seqlens_k == nullptr) {
    throw std::invalid_argument("sdpa_varlen_impl: cu_seqlens_q and cu_seqlens_k must not be null");
  }

  detail::Options options = make_common_options(
      Q_ptr, K_ptr, V_ptr, O_ptr, mask,
      q_stride_s, q_stride_d, q_stride_h, q_stride_b,
      k_stride_s, k_stride_d, k_stride_h, k_stride_b,
      v_stride_d, v_stride_s, v_stride_h, v_stride_b,
      o_stride_s, o_stride_d, o_stride_h, o_stride_b,
      batch, num_heads_q, num_heads_kv,
      max_seqlen_q, max_seqlen_kv,
      head_dim, softmax_scale, is_causal);
  options.varlen = true;
  options.total_seqlen_q = total_seqlen_q;
  options.total_seqlen_kv = total_seqlen_kv;
  options.max_seqlen_q = max_seqlen_q;
  options.max_seqlen_kv = max_seqlen_kv;
  options.cu_seqlens_q = cu_seqlens_q;
  options.cu_seqlens_k = cu_seqlens_k;
  options.seq_len_kv_cache = 0;
  options.total_seqlen_kv_cache = 0;
  options.max_seqlen_kv_cache = 0;
  options.lse = lse;

  compat::set_default_queue(*q);

  // When isVarLen=true, the kernel's apply_variable_length accesses
  // cumulative_length for ALL three fields.  Even with max_seqlen_kv_cache=0,
  // the pointer must be non-null and device-accessible.  Use the DnnlContext
  // scratch pool (reuses allocation across calls, only grows when needed).
  int* zero_cu_buf = static_cast<int*>(
      DnnlContext::Instance()->get_scratch_mem((batch + 1) * sizeof(int), 4, q));
  q->memset(zero_cu_buf, 0, (batch + 1) * sizeof(int));
  options.cu_seqlens_kv_cache = zero_cu_buf;
  options.use_tensor_strides = true;

  KernelLauncher launcher = select_prefill_launcher(q_dtype, head_dim);
  if (launcher == nullptr) {
    throw std::runtime_error(
        "Unsupported dtype or head dimension for sdpa_varlen_impl (only F16/BF16 and 64/96/128/192 are supported)");
  }

  launcher(options);
}

}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
