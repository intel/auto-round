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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <sycl/sycl.hpp>

#include "cutlass/float8.h"

namespace ark::detail {

namespace sage_fp8_preprocess_detail {

constexpr int kWorkgroupSize = 256;
constexpr int kQElementsPerGroup = 4096;
constexpr float kFP8Max = 448.0f;

inline int round_up(int value, int multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

}  // namespace sage_fp8_preprocess_detail

// workspace contains six FP32 values:
//   [q_absmax, k_absmax, v_absmax, q_scale, k_scale, v_scale].
// K/V means use dense [batch, head, dim] FP32 layout.
template <typename ElementInput>
void launch_sage_fp8_preprocess(
    sycl::queue* queue, const ElementInput* query, const ElementInput* key, const ElementInput* value,
    cutlass::float_e4m3_t* query_fp8, cutlass::float_e4m3_t* key_fp8,
    cutlass::float_e4m3_t* value_fp8, float* key_mean, float* value_mean, float* workspace,
    int batch, int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim,
    int tensor_layout) {
  using namespace sage_fp8_preprocess_detail;

  if (!queue->has_property<sycl::property::queue::in_order>()) {
    throw std::runtime_error("sage_fp8 preprocess requires an in-order SYCL queue");
  }

  size_t const query_elements =
      size_t(batch) * size_t(num_heads_q) * size_t(seq_len_q) * size_t(head_dim);
  size_t const kv_elements =
      size_t(batch) * size_t(num_heads_kv) * size_t(seq_len_kv) * size_t(head_dim);
  int const q_groups = int((query_elements + kQElementsPerGroup - 1) / kQElementsPerGroup);
  int const mean_groups = batch * num_heads_kv * head_dim;
  int const stats_groups = q_groups + mean_groups;

  sycl::event clear_event = queue->fill(workspace, 0.0f, 6);
  sycl::event stats_event = queue->submit([&](sycl::handler& handler) {
    handler.depends_on(clear_event);
    handler.parallel_for(
        sycl::nd_range<1>(size_t(stats_groups) * kWorkgroupSize, kWorkgroupSize),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
          int const group_id = int(item.get_group(0));
          int const local_id = int(item.get_local_id(0));
          auto group = item.get_group();

          auto atomic_max_nonnegative = [](float* destination, float candidate) {
            auto* bits = reinterpret_cast<uint32_t*>(destination);
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_bits(*bits);
            atomic_bits.fetch_max(sycl::bit_cast<uint32_t>(candidate));
          };

          if (group_id < q_groups) {
            size_t const begin = size_t(group_id) * kQElementsPerGroup;
            size_t const end = sycl::min(begin + size_t(kQElementsPerGroup), query_elements);
            float local_max = 0.0f;
            for (size_t index = begin + size_t(local_id); index < end; index += kWorkgroupSize) {
              local_max = sycl::fmax(local_max, sycl::fabs(static_cast<float>(query[index])));
            }
            float const group_max = sycl::reduce_over_group(group, local_max, sycl::maximum<float>{});
            if (local_id == 0) atomic_max_nonnegative(workspace, group_max);
            return;
          }

          int const mean_index = group_id - q_groups;
          int const dim = mean_index % head_dim;
          int const batch_head = mean_index / head_dim;
          int const head = batch_head % num_heads_kv;
          int const batch_id = batch_head / num_heads_kv;

          float key_sum = 0.0f;
          float value_sum = 0.0f;
          for (int token = local_id; token < seq_len_kv; token += kWorkgroupSize) {
            size_t source_index;
            if (tensor_layout == 0) {
              source_index = ((size_t(batch_id) * num_heads_kv + head) * seq_len_kv + token) * head_dim + dim;
            } else {
              source_index = ((size_t(batch_id) * seq_len_kv + token) * num_heads_kv + head) * head_dim + dim;
            }
            key_sum += static_cast<float>(key[source_index]);
            value_sum += static_cast<float>(value[source_index]);
          }
          float const key_average =
              sycl::reduce_over_group(group, key_sum, sycl::plus<float>{}) / float(seq_len_kv);
          float const value_average =
              sycl::reduce_over_group(group, value_sum, sycl::plus<float>{}) / float(seq_len_kv);
          if (local_id == 0) {
            key_mean[mean_index] = key_average;
            value_mean[mean_index] = value_average;
          }

          float key_max = 0.0f;
          float value_max = 0.0f;
          for (int token = local_id; token < seq_len_kv; token += kWorkgroupSize) {
            size_t source_index;
            if (tensor_layout == 0) {
              source_index = ((size_t(batch_id) * num_heads_kv + head) * seq_len_kv + token) * head_dim + dim;
            } else {
              source_index = ((size_t(batch_id) * seq_len_kv + token) * num_heads_kv + head) * head_dim + dim;
            }
            key_max = sycl::fmax(key_max, sycl::fabs(static_cast<float>(key[source_index]) - key_average));
            value_max =
                sycl::fmax(value_max, sycl::fabs(static_cast<float>(value[source_index]) - value_average));
          }
          float const key_group_max = sycl::reduce_over_group(group, key_max, sycl::maximum<float>{});
          float const value_group_max = sycl::reduce_over_group(group, value_max, sycl::maximum<float>{});
          if (local_id == 0) {
            atomic_max_nonnegative(workspace + 1, key_group_max);
            atomic_max_nonnegative(workspace + 2, value_group_max);
          }
        });
  });

  size_t const total_elements = query_elements + 2 * kv_elements;
  size_t const quant_global = size_t(round_up(int(total_elements), kWorkgroupSize));
  queue->submit([&](sycl::handler& handler) {
    handler.depends_on(stats_event);
    handler.parallel_for(sycl::nd_range<1>(quant_global, kWorkgroupSize), [=](sycl::nd_item<1> item) {
      size_t const index = item.get_global_linear_id();
      if (index >= total_elements) return;

      float const q_absmax = workspace[0];
      float const k_absmax = workspace[1];
      float const v_absmax = workspace[2];
      if (index == 0) {
        workspace[3] = sycl::fmax(q_absmax / kFP8Max, std::numeric_limits<float>::min());
        workspace[4] = sycl::fmax(k_absmax / kFP8Max, std::numeric_limits<float>::min());
        workspace[5] = sycl::fmax(v_absmax / kFP8Max, std::numeric_limits<float>::min());
      }

      if (index < query_elements) {
        float const inverse_scale = q_absmax > 0.0f ? kFP8Max / q_absmax : 0.0f;
        float quantized = sycl::clamp(static_cast<float>(query[index]) * inverse_scale, -kFP8Max, kFP8Max);
        query_fp8[index] = cutlass::float_e4m3_t(quantized);
        return;
      }

      bool const is_value = index >= query_elements + kv_elements;
      size_t const kv_index = is_value ? index - query_elements - kv_elements : index - query_elements;
      int mean_index;
      if (tensor_layout == 0) {
        mean_index = int(kv_index / size_t(seq_len_kv * head_dim)) * head_dim + int(kv_index % head_dim);
      } else {
        size_t const per_batch = size_t(seq_len_kv) * num_heads_kv * head_dim;
        int const batch_id = int(kv_index / per_batch);
        size_t const within_batch = kv_index % per_batch;
        int const head = int((within_batch / head_dim) % num_heads_kv);
        mean_index = (batch_id * num_heads_kv + head) * head_dim + int(kv_index % head_dim);
      }

      if (is_value) {
        float const inverse_scale = v_absmax > 0.0f ? kFP8Max / v_absmax : 0.0f;
        float quantized = sycl::clamp(
            (static_cast<float>(value[kv_index]) - value_mean[mean_index]) * inverse_scale,
            -kFP8Max, kFP8Max);
        value_fp8[kv_index] = cutlass::float_e4m3_t(quantized);
      } else {
        float const inverse_scale = k_absmax > 0.0f ? kFP8Max / k_absmax : 0.0f;
        float quantized = sycl::clamp(
            (static_cast<float>(key[kv_index]) - key_mean[mean_index]) * inverse_scale,
            -kFP8Max, kFP8Max);
        key_fp8[kv_index] = cutlass::float_e4m3_t(quantized);
      }
    });
  });
}

}  // namespace ark::detail
