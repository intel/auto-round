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

#pragma once

#include <sycl/sycl.hpp>

namespace ark {

void sdpa_impl_fp8(sycl::queue* q, void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr,
                   float q_scale, float k_scale, float v_scale, const float* vmean,
                   int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b,
                   int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b,
                   int v_stride_d, int v_stride_s, int v_stride_h, int v_stride_b,
                   int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b,
                   int batch, int num_heads_q, int num_heads_kv, int seq_len_q,
                   int seq_len_kv, int head_dim, float softmax_scale, bool is_causal);

void sage_fp8_fused_impl(sycl::queue* q, const void* Q, const void* K, const void* V,
                         void* Q_fp8, void* K_fp8, void* V_fp8, void* O,
                         float* K_mean, float* V_mean, float* workspace, int input_dtype,
                         int batch, int num_heads_q, int num_heads_kv, int seq_len_q,
                         int seq_len_kv, int head_dim, float softmax_scale, bool is_causal,
                         int tensor_layout);

}  // namespace ark
