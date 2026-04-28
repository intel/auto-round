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

int launch_prefill_kernel_f16_64_sage(detail::Options const& options) {
  return launch_sage_prefill_kernel_64<cute::int8_t, cute::int8_t, cute::half_t>(options);
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

detail::Options make_common_options(void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask, int batch,
                                    int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim,
                                    float softmax_scale, bool is_causal) {
  detail::Options options;
  options.q = Q_ptr;
  options.k = K_ptr;
  options.v = V_ptr;
  options.mask = mask;
  options.o = O_ptr;
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

void sage_prefill(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask, int scale_block_size, void* qscale,
                  void* kscale, BTLA_DTYPE q_dtype, int batch, int num_heads_q, int num_heads_kv, int seq_len_q,
                  int seq_len_kv, int head_dim, float softmax_scale, bool is_causal) {
  detail::Options options = make_common_options(Q_ptr, K_ptr, V_ptr, O_ptr, mask, batch, num_heads_q, num_heads_kv,
                                                seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
  options.scale_block_size = scale_block_size;
  options.qscale = qscale;
  options.kscale = kscale;
  compat::set_default_queue(*q);

  KernelLauncher launcher = select_sage_prefill_launcher(q_dtype, head_dim);
  if (launcher == nullptr) {
    throw std::runtime_error(
        "Unsupported dtype or head dimension for sage_prefill / SAGE (only F16/BF16 and 64/96/128/192 are supported)");
  }

  launcher(options);
}

void flash_attn_prefill(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask,
                        BTLA_DTYPE q_dtype, int batch, int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv,
                        int head_dim, float softmax_scale, bool is_causal) {
  detail::Options options = make_common_options(Q_ptr, K_ptr, V_ptr, O_ptr, mask, batch, num_heads_q, num_heads_kv,
                                                seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
  compat::set_default_queue(*q);

  KernelLauncher launcher = select_prefill_launcher(q_dtype, head_dim);
  if (launcher == nullptr) {
    throw std::runtime_error(
        "Unsupported dtype or head dimension for flash_attn_prefill (only F16/BF16 and 64/96/128/192 are supported)");
  }

  launcher(options);
}

void flash_attn_decode(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask,
                       BTLA_DTYPE q_dtype, int batch, int num_heads_q, int num_heads_kv, int seq_len_kv, int head_dim,
                       float softmax_scale, bool is_causal) {
  detail::Options options = make_common_options(Q_ptr, K_ptr, V_ptr, O_ptr, mask, batch, num_heads_q,
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
               int batch, int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim,
               float softmax_scale, bool is_causal) {
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
    flash_attn_decode(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, q_dtype, batch, num_heads_q, num_heads_kv, seq_len_kv,
                      head_dim, softmax_scale, is_causal);
    return;
  }

  flash_attn_prefill(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, q_dtype, batch, num_heads_q, num_heads_kv, seq_len_q,
                     seq_len_kv, head_dim, softmax_scale, is_causal);
}

void sdpa_impl_qks8_pvhalf(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, void* mask, int scale_block_size, void* qscale,
               void* kscale, int batch, int num_heads_q, int num_heads_kv, int seq_len_q,
               int seq_len_kv, int head_dim, float softmax_scale, bool is_causal) {
  if (mask && is_causal) {
    throw std::invalid_argument("sdpa_impl: mask and is_causal cannot both be set");
  }

  if (seq_len_q <= 0 || seq_len_kv <= 0) {
    throw std::invalid_argument("sdpa_impl: seq_len_q and seq_len_kv must be greater than 0");
  }
  if (seq_len_q == 1) {
    flash_attn_decode(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, BTLA_DTYPE::F16, batch, num_heads_q, num_heads_kv, seq_len_kv,
                      head_dim, softmax_scale, is_causal);
    return;
  }


  sage_prefill(q, Q_ptr, K_ptr, V_ptr, O_ptr, mask, scale_block_size, qscale, kscale, BTLA_DTYPE::S8, batch, num_heads_q, num_heads_kv, seq_len_q,
                     seq_len_kv, head_dim, softmax_scale, is_causal);
}

}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
