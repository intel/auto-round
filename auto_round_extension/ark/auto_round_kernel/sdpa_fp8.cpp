// Copyright (c) 2026 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

#include <stdexcept>

#include "sage_fp8_api.hpp"
#include "stla/sage_fp8_preprocess.hpp"
#include "sycl_tla_sdpa.hpp"
#include "sdpa_kernel_declarations.hpp"

namespace ark {
namespace {

using KernelLauncher = int (*)(detail::Options const& options);

KernelLauncher select_sage_fp8_launcher(bool decode, int head_dim) {
  if (decode) {
    switch (head_dim) {
      case 64: return detail::launch_sage_fp8_decode_kernel_64;
      case 96: return detail::launch_sage_fp8_decode_kernel_96;
      case 128: return detail::launch_sage_fp8_decode_kernel_128;
      case 192: return detail::launch_sage_fp8_decode_kernel_192;
      default: return nullptr;
    }
  }
  switch (head_dim) {
    case 64: return detail::launch_sage_fp8_prefill_kernel_64;
    case 96: return detail::launch_sage_fp8_prefill_kernel_96;
    case 128: return detail::launch_sage_fp8_prefill_kernel_128;
    case 192: return detail::launch_sage_fp8_prefill_kernel_192;
    default: return nullptr;
  }
}

}  // namespace

void sdpa_impl_fp8(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr,
                   float q_scale, float k_scale, float v_scale, const float* vmean,
                   int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b,
                   int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b,
                   int v_stride_d, int v_stride_s, int v_stride_h, int v_stride_b,
                   int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b,
                   int batch, int num_heads_q, int num_heads_kv, int seq_len_q,
                   int seq_len_kv, int head_dim, float softmax_scale, bool is_causal) {
  if (seq_len_q <= 0 || seq_len_kv <= 0) {
    throw std::invalid_argument("sdpa_impl_fp8: sequence lengths must be greater than zero");
  }
  if (num_heads_kv <= 0 || num_heads_q % num_heads_kv != 0) {
    throw std::invalid_argument("sdpa_impl_fp8: num_heads_q must be divisible by num_heads_kv");
  }
  if (q_stride_d != 1 || k_stride_d != 1 || v_stride_d != 1 || o_stride_d != 1) {
    throw std::invalid_argument("sdpa_impl_fp8: head-dimension strides must be one");
  }
  if (!(q_scale > 0.0f) || !(k_scale > 0.0f) || !(v_scale > 0.0f)) {
    throw std::invalid_argument("sdpa_impl_fp8: dequantization scales must be positive");
  }

  detail::Options options;
  options.q = Q_ptr;
  options.k = K_ptr;
  options.v = V_ptr;
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
  options.q_scale = q_scale;
  options.k_scale = k_scale;
  options.v_scale = v_scale;
  options.vmean = vmean;
  compat::set_default_queue(*q);

  KernelLauncher launcher = select_sage_fp8_launcher(seq_len_q == 1, head_dim);
  if (launcher == nullptr) {
    throw std::runtime_error("sdpa_impl_fp8: supported head dimensions are 64, 96, 128, and 192");
  }
  launcher(options);
}

void sage_fp8_fused_impl(sycl::queue* q, const void* Q, const void* K, const void* V,
                         void* Q_fp8, void* K_fp8, void* V_fp8, void* O,
                         float* K_mean, float* V_mean, float* workspace, int input_dtype,
                         int batch, int num_heads_q, int num_heads_kv, int seq_len_q,
                         int seq_len_kv, int head_dim, float softmax_scale, bool is_causal,
                         int tensor_layout) {
  if (tensor_layout != 0 && tensor_layout != 1) {
    throw std::invalid_argument("sage_fp8_fused_impl: tensor_layout must be HND or NHD");
  }
  auto* q_fp8 = static_cast<cutlass::float_e4m3_t*>(Q_fp8);
  auto* k_fp8 = static_cast<cutlass::float_e4m3_t*>(K_fp8);
  auto* v_fp8 = static_cast<cutlass::float_e4m3_t*>(V_fp8);
  switch (input_dtype) {
    case 16:
      detail::launch_sage_fp8_preprocess(q, static_cast<const sycl::half*>(Q),
          static_cast<const sycl::half*>(K), static_cast<const sycl::half*>(V),
          q_fp8, k_fp8, v_fp8, K_mean, V_mean, workspace, batch, num_heads_q,
          num_heads_kv, seq_len_q, seq_len_kv, head_dim, tensor_layout);
      break;
    case 32:
      detail::launch_sage_fp8_preprocess(q, static_cast<const float*>(Q),
          static_cast<const float*>(K), static_cast<const float*>(V),
          q_fp8, k_fp8, v_fp8, K_mean, V_mean, workspace, batch, num_heads_q,
          num_heads_kv, seq_len_q, seq_len_kv, head_dim, tensor_layout);
      break;
    case 65552:
      detail::launch_sage_fp8_preprocess(q, static_cast<const cutlass::bfloat16_t*>(Q),
          static_cast<const cutlass::bfloat16_t*>(K), static_cast<const cutlass::bfloat16_t*>(V),
          q_fp8, k_fp8, v_fp8, K_mean, V_mean, workspace, batch, num_heads_q,
          num_heads_kv, seq_len_q, seq_len_kv, head_dim, tensor_layout);
      break;
    default:
      throw std::invalid_argument("sage_fp8_fused_impl: input must be FP16, BF16, or FP32");
  }

  int q_stride_s, q_stride_h, q_stride_b;
  int kv_stride_s, kv_stride_h, kv_stride_b;
  if (tensor_layout == 0) {
    q_stride_s = head_dim;
    q_stride_h = seq_len_q * head_dim;
    q_stride_b = num_heads_q * q_stride_h;
    kv_stride_s = head_dim;
    kv_stride_h = seq_len_kv * head_dim;
    kv_stride_b = num_heads_kv * kv_stride_h;
  } else {
    q_stride_s = num_heads_q * head_dim;
    q_stride_h = head_dim;
    q_stride_b = seq_len_q * q_stride_s;
    kv_stride_s = num_heads_kv * head_dim;
    kv_stride_h = head_dim;
    kv_stride_b = seq_len_kv * kv_stride_s;
  }

  detail::Options options;
  options.q = Q_fp8;
  options.k = K_fp8;
  options.v = V_fp8;
  options.o = O;
  options.use_tensor_strides = true;
  options.q_stride_s = q_stride_s; options.q_stride_d = 1;
  options.q_stride_h = q_stride_h; options.q_stride_b = q_stride_b;
  options.k_stride_s = kv_stride_s; options.k_stride_d = 1;
  options.k_stride_h = kv_stride_h; options.k_stride_b = kv_stride_b;
  options.v_stride_d = 1; options.v_stride_s = kv_stride_s;
  options.v_stride_h = kv_stride_h; options.v_stride_b = kv_stride_b;
  options.o_stride_s = q_stride_s; options.o_stride_d = 1;
  options.o_stride_h = q_stride_h; options.o_stride_b = q_stride_b;
  options.batch = batch;
  options.num_heads_q = num_heads_q;
  options.num_heads_kv = num_heads_kv;
  options.seq_len_qo = seq_len_q;
  options.seq_len_kv = seq_len_kv;
  options.head_size_qk = head_dim;
  options.head_size_vo = head_dim;
  options.softmax_scale = softmax_scale;
  options.is_causal = is_causal;
  options.vmean = V_mean;
  options.q_scale_device = workspace + 3;
  options.k_scale_device = workspace + 4;
  options.v_scale_device = workspace + 5;
  compat::set_default_queue(*q);

  KernelLauncher launcher = select_sage_fp8_launcher(seq_len_q == 1, head_dim);
  if (launcher == nullptr) {
    throw std::runtime_error("sage_fp8_fused_impl: supported head dimensions are 64, 96, 128, and 192");
  }
  launcher(options);
}

}  // namespace ark

#endif
