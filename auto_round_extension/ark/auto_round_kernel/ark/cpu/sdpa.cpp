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

// Process-wide BestLA thread pool used to drive the migrated attention wrapper.
// Neural Speed reaches a global pool through `ne_threading::get()`; ARK has no
// such global, so the wrapper takes an explicit `parallel::IThreading&`. This
// singleton mirrors that pool and is configured once on first use.
bestla::parallel::IThreading& bestla_sdpa_threading() {
#if BTLA_OPENMP
  static bestla::parallel::OMPThreading pool;
#else
  static bestla::parallel::StdThreading pool;
#endif
  static const bool initialized = [] {
    pool.set_threads(0, false);  // 0 -> all usable cores
    return true;
  }();
  (void)initialized;
  return pool;
}

// Copy the layout/stride/scale metadata from the type-erased `attn_fwd_args_t`
// into the dtype-typed wrapper struct, reinterpreting the Q/K/V/dst pointers as
// the requested operand types. Field names match one-to-one between the two
// structs, so this is a straight per-field port.
template <typename KV_T>
bestla_mha::attn_fwd_args_t<float, KV_T, KV_T, float> make_typed_attn_args(const attn_fwd_args_t& a) {
  bestla_mha::attn_fwd_args_t<float, KV_T, KV_T, float> t{};
  t.Q = static_cast<float*>(a.Q);
  t.K = static_cast<KV_T*>(a.K);
  t.V = static_cast<KV_T*>(a.V);
  t.dst = static_cast<float*>(a.dst);
  t.Q_sc = a.Q_sc;
  t.K_sc = a.K_sc;
  t.V_sc = a.V_sc;
  t.dst_sc = a.dst_sc;
  t.tmp = a.tmp;
  t.QK_scale = a.QK_scale;
  t.attn_flags = a.attn_flags;
  t.batch_size = a.batch_size;
  t.head_num = a.head_num;
  t.heads_kv = a.heads_kv;
  t.head_size = a.head_size;
  t.sl_q = a.sl_q;
  t.sl_kv = a.sl_kv;
  t.Q_layout = a.Q_layout;
  t.K_layout = a.K_layout;
  t.V_layout = a.V_layout;
  t.dst_layout = a.dst_layout;
  t.step_q_bs = a.step_q_bs;
  t.step_q_head_num = a.step_q_head_num;
  t.step_q_sl = a.step_q_sl;
  t.step_k_bs = a.step_k_bs;
  t.step_k_head_num = a.step_k_head_num;
  t.step_k_sl = a.step_k_sl;
  t.step_k_head_size = a.step_k_head_size;
  t.step_v_bs = a.step_v_bs;
  t.step_v_head_num = a.step_v_head_num;
  t.step_v_sl = a.step_v_sl;
  t.step_v_head_size = a.step_v_head_size;
  t.step_dst_bs = a.step_dst_bs;
  t.step_dst_head_num = a.step_dst_head_num;
  t.step_dst_sl = a.step_dst_sl;
  t.n_padding = a.n_padding;
  return t;
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

void bestla_sdpa_forward(const attn_fwd_args_t& args, BTLA_DTYPE kv_dtype) {
  if (!args.Q || !args.K || !args.V || !args.dst) {
    throw std::invalid_argument("ark::cpu::bestla_sdpa_forward: Q/K/V/dst pointers must be non-null");
  }

  bestla::parallel::IThreading& th = bestla_sdpa_threading();
  switch (kv_dtype) {
    case BTLA_DTYPE::F16: {
      const auto typed = make_typed_attn_args<bestla::utils::fp16>(args);
      bestla_mha::bestla_fusion_attn_forward<float, bestla::utils::fp16, bestla::utils::fp16, float>(typed, th);
      break;
    }
    case BTLA_DTYPE::BF16: {
      const auto typed = make_typed_attn_args<bestla::utils::bf16>(args);
      bestla_mha::bestla_fusion_attn_forward<float, bestla::utils::bf16, bestla::utils::bf16, float>(typed, th);
      break;
    }
    default:
      throw std::invalid_argument(
          "ark::cpu::bestla_sdpa_forward: only F16 and BF16 K/V operands are supported");
  }
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
