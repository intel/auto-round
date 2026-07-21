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

#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>

#include "sage_fp8_api.hpp"

namespace ark {

using torch_ptr = uintptr_t;
constexpr int TENSOR_LAYOUT_HND = 0;
constexpr int TENSOR_LAYOUT_NHD = 1;

void sage_fp8(torch_ptr stream, torch_ptr Q, torch_ptr K, torch_ptr V, torch_ptr O,
              float q_scale, float k_scale, float v_scale, torch_ptr vmean,
              int batch, int num_heads_q, int num_heads_kv, int seq_len_q,
              int seq_len_kv, int head_dim, float softmax_scale, bool is_causal,
              int tensor_layout) {
  if (tensor_layout != TENSOR_LAYOUT_HND && tensor_layout != TENSOR_LAYOUT_NHD) {
    throw std::invalid_argument("ark::sage_fp8: tensor_layout must be HND or NHD");
  }

  int q_stride_s, q_stride_h, q_stride_b;
  int k_stride_s, k_stride_h, k_stride_b;
  int v_stride_s, v_stride_h, v_stride_b;
  int o_stride_s, o_stride_h, o_stride_b;
  if (tensor_layout == TENSOR_LAYOUT_HND) {
    int q_sh = seq_len_q * head_dim;
    int kv_sh = seq_len_kv * head_dim;
    q_stride_s = head_dim; q_stride_h = q_sh; q_stride_b = num_heads_q * q_sh;
    k_stride_s = head_dim; k_stride_h = kv_sh; k_stride_b = num_heads_kv * kv_sh;
    v_stride_s = head_dim; v_stride_h = kv_sh; v_stride_b = num_heads_kv * kv_sh;
    o_stride_s = head_dim; o_stride_h = q_sh; o_stride_b = num_heads_q * q_sh;
  } else {
    int q_hd = num_heads_q * head_dim;
    int kv_hd = num_heads_kv * head_dim;
    q_stride_s = q_hd; q_stride_h = head_dim; q_stride_b = seq_len_q * q_hd;
    k_stride_s = kv_hd; k_stride_h = head_dim; k_stride_b = seq_len_kv * kv_hd;
    v_stride_s = kv_hd; v_stride_h = head_dim; v_stride_b = seq_len_kv * kv_hd;
    o_stride_s = q_hd; o_stride_h = head_dim; o_stride_b = seq_len_q * q_hd;
  }

  sdpa_impl_fp8(reinterpret_cast<sycl::queue*>(stream), reinterpret_cast<void*>(Q),
                 reinterpret_cast<void*>(K), reinterpret_cast<void*>(V), reinterpret_cast<void*>(O),
                 q_scale, k_scale, v_scale, vmean ? reinterpret_cast<const float*>(vmean) : nullptr,
                 q_stride_s, 1, q_stride_h, q_stride_b,
                 k_stride_s, 1, k_stride_h, k_stride_b,
                 1, v_stride_s, v_stride_h, v_stride_b,
                 o_stride_s, 1, o_stride_h, o_stride_b,
                 batch, num_heads_q, num_heads_kv, seq_len_q, seq_len_kv,
                 head_dim, softmax_scale, is_causal);
}

        void sage_fp8_fused(torch_ptr stream, torch_ptr Q, torch_ptr K, torch_ptr V,
                  torch_ptr Q_fp8, torch_ptr K_fp8, torch_ptr V_fp8, torch_ptr O,
                  torch_ptr K_mean, torch_ptr V_mean, torch_ptr workspace, int input_dtype,
                  int batch, int num_heads_q, int num_heads_kv, int seq_len_q,
                  int seq_len_kv, int head_dim, float softmax_scale, bool is_causal,
                  int tensor_layout) {
          sage_fp8_fused_impl(
            reinterpret_cast<sycl::queue*>(stream), reinterpret_cast<const void*>(Q),
            reinterpret_cast<const void*>(K), reinterpret_cast<const void*>(V),
            reinterpret_cast<void*>(Q_fp8), reinterpret_cast<void*>(K_fp8),
            reinterpret_cast<void*>(V_fp8), reinterpret_cast<void*>(O),
            reinterpret_cast<float*>(K_mean), reinterpret_cast<float*>(V_mean),
            reinterpret_cast<float*>(workspace), input_dtype, batch, num_heads_q, num_heads_kv,
            seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal, tensor_layout);
        }

}  // namespace ark

PYBIND11_MODULE(PY_NAME, module) {
  module.def("sage_fp8", &ark::sage_fp8,
             pybind11::arg("stream"), pybind11::arg("Q"), pybind11::arg("K"), pybind11::arg("V"),
             pybind11::arg("O"), pybind11::arg("q_scale"), pybind11::arg("k_scale"),
             pybind11::arg("v_scale"), pybind11::arg("vmean"),
             pybind11::arg("batch"), pybind11::arg("num_heads_q"), pybind11::arg("num_heads_kv"),
             pybind11::arg("seq_len_q"), pybind11::arg("seq_len_kv"), pybind11::arg("head_dim"),
             pybind11::arg("softmax_scale"), pybind11::arg("is_causal"), pybind11::arg("tensor_layout"));
  module.def("sage_fp8_fused", &ark::sage_fp8_fused,
             pybind11::arg("stream"), pybind11::arg("Q"), pybind11::arg("K"), pybind11::arg("V"),
             pybind11::arg("Q_fp8"), pybind11::arg("K_fp8"), pybind11::arg("V_fp8"), pybind11::arg("O"),
             pybind11::arg("K_mean"), pybind11::arg("V_mean"), pybind11::arg("workspace"),
             pybind11::arg("input_dtype"), pybind11::arg("batch"), pybind11::arg("num_heads_q"),
             pybind11::arg("num_heads_kv"), pybind11::arg("seq_len_q"), pybind11::arg("seq_len_kv"),
             pybind11::arg("head_dim"), pybind11::arg("softmax_scale"), pybind11::arg("is_causal"),
             pybind11::arg("tensor_layout"));
}
