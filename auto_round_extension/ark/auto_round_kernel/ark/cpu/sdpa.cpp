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

#include <algorithm>
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

// Scratch (attn_fwd_args_t::tmp) bytes required by the migrated BestLA attention
// wrapper. mha_stable_interface_t::compute uses, per thread,
//   M_TILE * padto(padto(sl_kv, GemmQK::NTILE), GemmPV::KTILE) * sizeof(float)
// bytes for the score/exp tile. The exact tile constants depend on the GemmCore
// chosen at runtime from CPU features, so we use a conservative upper bound over
// every wired core (M_TILE<=16, NTILE<=48, KTILE<=32; AVX2 fp16=4/24/1,
// AVX512F bf16=8/48/1, AMX-BF16=16/48/32). The kernel only ever touches its own
// `tmp + tid * tmp_bytes_actual .. + tmp_bytes_actual` region, and the actual
// per-thread stride never exceeds this bound, so over-allocating keeps every
// thread's slice in range regardless of the dispatched branch.
//
// This intentionally differs from the scalar `attn_workspace_size()` /
// `mha_dense_workspace_size()` helpers, which size the legacy per-row scalar
// kernel rather than the BestLA tiled wrapper. (Neural Speed queries the exact
// size for the selected core; ARK over-allocates to keep one core-independent
// helper.)
size_t bestla_attn_workspace_size(const attn_shape_t& shape, int num_threads) {
  constexpr int kMaxMTile = 16;
  constexpr int kMaxNTile = 48;
  constexpr int kMaxKTile = 32;
  const int sl_kv = std::max(1, shape.sl_kv);
  const int padded_n = ((sl_kv + kMaxNTile - 1) / kMaxNTile) * kMaxNTile;
  const int padded_k = ((padded_n + kMaxKTile - 1) / kMaxKTile) * kMaxKTile;
  const size_t per_thread = static_cast<size_t>(kMaxMTile) * static_cast<size_t>(padded_k) * sizeof(float);
  return per_thread * static_cast<size_t>(std::max(1, num_threads));
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
  // Phase 3 Step 2 wires only the plain-strided mixed-precision dispatch shell.
  // The `step_*` stride interface itself is HND/NHD-friendly, but the BestLA
  // specializations reached below currently require packed/reordered
  // (NTILE24/NTILE48) K/V and will throw for raw PLAIN K/V (see
  // mha_dense_wrapper.h). Reject every other feature whose BestLA path is not
  // migrated yet so callers fail loudly rather than silently producing wrong
  // results. Packed K/V acceptance is a Phase 4 concern and intentionally not
  // implemented here; for now PLAIN is forwarded and the kernel decides.
  if (args.Q_layout != ATTN_FWD_LAYOUT_PLAIN || args.K_layout != ATTN_FWD_LAYOUT_PLAIN ||
      args.V_layout != ATTN_FWD_LAYOUT_PLAIN || args.dst_layout != ATTN_FWD_LAYOUT_PLAIN) {
    throw std::invalid_argument("ark::cpu::bestla_sdpa_forward: only ATTN_FWD_LAYOUT_PLAIN is supported");
  }
  constexpr attn_flags_t kUnsupportedFlags =
      ATTN_FLAG_IS_ALIBI8 | ATTN_FLAG_IS_TANH30 | ATTN_FLAG_PADDING_RIGHT;
  if ((args.attn_flags & kUnsupportedFlags) != 0) {
    throw std::invalid_argument(
        "ark::cpu::bestla_sdpa_forward: alibi, tanh and padding-right are not wired yet");
  }

  // Threading is supplied by the caller (ARK reuses CpuWrapper::get_threading()),
  // type-erased through attn_fwd_args_t::threading. This avoids maintaining a
  // second independent BestLA thread pool in the CPU attention path.
  if (args.threading == nullptr) {
    throw std::invalid_argument("ark::cpu::bestla_sdpa_forward: threading pool must be provided");
  }
  auto* th = static_cast<bestla::parallel::IThreading*>(args.threading);

  // Allocate the BestLA wrapper scratch when the caller did not provide one and
  // keep it alive for the duration of the forward call (Phase 1 attn_fwd_args_t
  // is passed by const ref, so the buffer must outlive the dispatch below).
  // The kernel reinterprets `tmp` as `float*` for its per-thread score/exp tile,
  // so back it with a `float` vector to guarantee correct (>= alignof(float))
  // alignment; a `char` buffer would only be 1-byte aligned and could fault or
  // silently mis-read on the SIMD score tile.
  attn_fwd_args_t local = args;
  std::vector<float> workspace;
  if (local.tmp == nullptr) {
    attn_shape_t shape{local.batch_size, local.head_num, local.heads_kv, local.head_size, local.sl_q, local.sl_kv};
    const size_t bytes = bestla_attn_workspace_size(shape, th->num_threads());
    workspace.resize((bytes + sizeof(float) - 1) / sizeof(float));
    local.tmp = workspace.empty() ? nullptr : workspace.data();
  }

  switch (kv_dtype) {
    case BTLA_DTYPE::F16: {
      const auto typed = make_typed_attn_args<bestla::utils::fp16>(local);
      bestla_mha::bestla_fusion_attn_forward<float, bestla::utils::fp16, bestla::utils::fp16, float>(typed, *th);
      break;
    }
    case BTLA_DTYPE::BF16: {
      const auto typed = make_typed_attn_args<bestla::utils::bf16>(local);
      bestla_mha::bestla_fusion_attn_forward<float, bestla::utils::bf16, bestla::utils::bf16, float>(typed, *th);
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
