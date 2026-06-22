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

#include "ark/cpu/sdpa.h"
#include "ark/cpu/mha_dense_wrapper.h"

#include <stdexcept>
#include <vector>

namespace ark::cpu {
namespace {

size_t cache_offset(int b, int h, int s, int d, int num_heads_kv, int capacity, int head_dim) {
  return (((static_cast<size_t>(b) * num_heads_kv + h) * capacity + s) * head_dim + d);
}

size_t qko_offset(const AttentionStrides& strides, int b, int h, int s, int d) {
  return static_cast<size_t>(b) * strides.batch + static_cast<size_t>(h) * strides.head +
         static_cast<size_t>(s) * strides.seq + static_cast<size_t>(d) * strides.dim;
}

size_t value_offset(const ValueStrides& strides, int b, int h, int s, int d) {
  return static_cast<size_t>(b) * strides.batch + static_cast<size_t>(h) * strides.head +
         static_cast<size_t>(s) * strides.seq + static_cast<size_t>(d) * strides.dim;
}

}  // namespace

void sdpa_forward(const MhaDenseArgs& args) {
  // Pre-allocate the flash-attention scratch once so the inner kernel avoids
  // per-row heap allocations.
  MhaDenseArgs local = args;
  std::vector<float> workspace;
  if (local.workspace == nullptr) {
    workspace.resize(mha_dense_workspace_size(local));
    local.workspace = workspace.empty() ? nullptr : workspace.data();
  }
  mha_dense_forward(local);
}

void kv_cache_update(void* cache_k, void* cache_v, const void* key, const void* value, const AttentionStrides& k_strides,
                     const ValueStrides& v_strides, BTLA_DTYPE dtype, int batch, int num_heads_kv, int append_len,
                     int head_dim, int capacity, int start_pos) {
  if (!cache_k || !cache_v || !key || !value) {
    throw std::invalid_argument("ark::cpu::kv_cache_update: cache and source pointers must be non-null");
  }
  if (batch <= 0 || num_heads_kv <= 0 || append_len <= 0 || head_dim <= 0 || capacity <= 0 || start_pos < 0 ||
      start_pos + append_len > capacity) {
    throw std::invalid_argument("ark::cpu::kv_cache_update: invalid dimensions or append range");
  }
  if (k_strides.dim != 1 || v_strides.dim != 1) {
    throw std::invalid_argument("ark::cpu::kv_cache_update: head-dim stride must be 1 for K/V");
  }
  (void)element_size(dtype);

#pragma omp parallel for collapse(4) schedule(static)
  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < num_heads_kv; ++h) {
      for (int s = 0; s < append_len; ++s) {
        for (int d = 0; d < head_dim; ++d) {
          const size_t dst = cache_offset(b, h, start_pos + s, d, num_heads_kv, capacity, head_dim);
          store_scalar(cache_k, dst, dtype, load_scalar(key, qko_offset(k_strides, b, h, s, d), dtype));
          store_scalar(cache_v, dst, dtype, load_scalar(value, value_offset(v_strides, b, h, s, d), dtype));
        }
      }
    }
  }
}

}  // namespace ark::cpu
