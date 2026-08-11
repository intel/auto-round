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
#include "bestla/kernel_avx2.h"

#include <algorithm>
#include <array>
#include <cstring>
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

size_t bestla_attn_workspace_size(const attn_shape_t& shape, int num_threads) {
  // bestla_tmp_layout() reserves a private rewind prefix for EACH thread before
  // that thread's tile scratch. This avoids cross-thread aliasing when a route
  // writes `tmp - i_m * ld_tmp_*` to emulate a full [sl_q, sl_kv] matrix.
  const auto layout = bestla_mha::bestla_tmp_layout(shape.sl_q, shape.sl_kv);
  return layout.thread_stride_bytes * static_cast<size_t>(std::max(1, num_threads));
}

size_t bestla_route4_workspace_size(const attn_shape_t& shape, int num_threads) {
  // Route 4 (homogeneous bf16 over mha_interface_t) follows Neural Speed's
  // compact scratch contract: one bf16 tile buffer per thread, sized only for
  // the route's QK exp-sum tile. It does not need the shared prefix-based
  // layout used by the stable routes.
  constexpr int kRoute4MTile = 16;
  constexpr int kRoute4NTile = 64;
  const size_t thread_bytes = static_cast<size_t>(kRoute4MTile) *
                              static_cast<size_t>(bestla::utils::padto(std::max(1, shape.sl_kv), kRoute4NTile)) *
                              sizeof(bestla::utils::bf16);
  return thread_bytes * static_cast<size_t>(std::max(1, num_threads));
}

char* aligned_bestla_tmp(bestla::utils::aligned_vector<float>& workspace, const attn_shape_t& shape, size_t bytes) {
  const size_t total_floats = (bytes + sizeof(float) - 1) / sizeof(float);
  workspace.resize(total_floats);
  return workspace.size() == 0 ? nullptr : reinterpret_cast<char*>(workspace.data());
}

// ---------------------------------------------------------------------------
// Workspace cache (defect-1 mitigation)
//
// The BestLA attention kernels need an aligned per-thread scratch buffer whose
// size depends only on (sl_q, sl_kv, num_threads).  Without a cache every
// forward call allocates + zero-fills this buffer via aligned_vector::resize().
//
// A small static pool (4 entries, round-robin replacement) reuses the buffer
// across calls whose shape/thread-count matches.  On the first hit for a new
// shape the resize still zero-fills once; subsequent calls with the same shape
// skip both the allocation and the zero-fill entirely.
// ---------------------------------------------------------------------------
static constexpr int kWorkspaceCacheSlots = 4;
static struct {
  bestla::utils::aligned_vector<float> buf;
  int sl_q = 0;
  int sl_kv = 0;
  int num_threads = 0;
  size_t capacity_bytes = 0;
} g_workspace_cache[kWorkspaceCacheSlots];

static char* find_or_alloc_workspace(size_t bytes, int sl_q, int sl_kv, int num_threads) {
  // Linear scan for an existing entry with matching shape and enough capacity.
  for (int i = 0; i < kWorkspaceCacheSlots; ++i) {
    auto& e = g_workspace_cache[i];
    if (e.sl_q == sl_q && e.sl_kv == sl_kv && e.num_threads == num_threads) {
      if (e.capacity_bytes >= bytes) return reinterpret_cast<char*>(e.buf.data());
      // Existing entry too small — resize to fit (one-time cost).
      size_t count = (bytes + sizeof(float) - 1) / sizeof(float);
      e.buf.resize(count);
      e.capacity_bytes = e.buf.size() * sizeof(float);
      return reinterpret_cast<char*>(e.buf.data());
    }
  }
  // No matching entry — allocate in the next round-robin slot.
  static int next_slot = 0;
  int slot = next_slot;
  next_slot = (next_slot + 1) % kWorkspaceCacheSlots;
  auto& e = g_workspace_cache[slot];
  size_t count = (bytes + sizeof(float) - 1) / sizeof(float);
  e.buf.resize(count);
  e.sl_q = sl_q;
  e.sl_kv = sl_kv;
  e.num_threads = num_threads;
  e.capacity_bytes = e.buf.size() * sizeof(float);
  return reinterpret_cast<char*>(e.buf.data());
}

#if CompileAVX2()
bool can_use_16bit_reorder_avx2(const ReorderKVShape& shape, int head_dim_stride) {
  return head_dim_stride == 1 && (shape.dtype == BTLA_DTYPE::F16 || shape.dtype == BTLA_DTYPE::BF16) &&
         bestla::device::CpuDevice::getInstance()->AVX2();
}

void reorder_k_16bit_avx2(uint16_t* dst, const uint16_t* src, const ReorderKVShape& shape,
                          const AttentionStrides& strides, int batch_idx, int head_idx) {
  const size_t head_base = (static_cast<size_t>(batch_idx) * shape.heads_kv + head_idx) * shape.k_head_elems;
  const int seq_full = shape.logical_capacity / 8 * 8;
  const int head_dim_full = shape.head_dim / 8 * 8;

  for (int s = 0; s < seq_full; s += 8) {
    const int tile = s / shape.ntile;
    const int sl_in = s % shape.ntile;
    for (int d = 0; d < head_dim_full; d += 8) {
      std::array<__m128i, 8> rows;
      for (int row = 0; row < 8; ++row) {
        rows[row] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + qko_offset(strides, batch_idx, head_idx,
                                                                                         s + row, d)));
      }
      const auto cols = bestla::kernel::avx2::tr_x8_word(rows);
      if (shape.dtype == BTLA_DTYPE::F16) {
        for (int col = 0; col < 8; ++col) {
          const size_t dst_offset = head_base + static_cast<size_t>(tile) * shape.k_head_size_pad * shape.ntile +
                                    static_cast<size_t>(d + col) * shape.ntile + sl_in;
          _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + dst_offset), cols[col]);
        }
      } else {
        for (int pair = 0; pair < 4; ++pair) {
          const size_t dst_offset =
              head_base + static_cast<size_t>(tile) * shape.k_head_size_pad * shape.ntile +
              static_cast<size_t>((d / 2) + pair) * shape.ntile * shape.rowpack + sl_in * shape.rowpack;
          _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + dst_offset),
                           _mm_unpacklo_epi16(cols[pair * 2], cols[pair * 2 + 1]));
          _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + dst_offset + 8),
                           _mm_unpackhi_epi16(cols[pair * 2], cols[pair * 2 + 1]));
        }
      }
    }
  }
}

void reorder_v_16bit_avx2(uint16_t* dst, const uint16_t* src, const ReorderKVShape& shape,
                          const ValueStrides& strides, int batch_idx, int head_idx) {
  const size_t head_base = (static_cast<size_t>(batch_idx) * shape.heads_kv + head_idx) * shape.v_head_elems;
  const int head_dim_full = shape.head_dim / 8 * 8;

  if (shape.dtype == BTLA_DTYPE::F16) {
    for (int s = 0; s < shape.logical_capacity; ++s) {
      for (int d = 0; d < head_dim_full; d += 8) {
        const auto values =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + value_offset(strides, batch_idx, head_idx, s, d)));
        const size_t dst_offset = head_base + static_cast<size_t>(d / shape.ntile) * shape.v_seq_pad * shape.ntile +
                                  static_cast<size_t>(s) * shape.ntile + d % shape.ntile;
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + dst_offset), values);
      }
    }
    return;
  }

  const int seq_full = shape.logical_capacity / 2 * 2;
  for (int s = 0; s < seq_full; s += 2) {
    for (int d = 0; d < head_dim_full; d += 8) {
      const auto first =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + value_offset(strides, batch_idx, head_idx, s, d)));
      const auto second =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + value_offset(strides, batch_idx, head_idx, s + 1, d)));
      const size_t dst_offset = head_base + static_cast<size_t>(d / shape.ntile) * shape.v_seq_pad * shape.ntile +
                                static_cast<size_t>(s / shape.rowpack) * shape.ntile * shape.rowpack +
                                static_cast<size_t>(d % shape.ntile) * shape.rowpack;
      _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + dst_offset), _mm_unpacklo_epi16(first, second));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + dst_offset + 8), _mm_unpackhi_epi16(first, second));
    }
  }
}
#endif

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
  return t;
}

// Homogeneous variant of make_typed_attn_args: every operand (Q/K/V/dst) shares
// one element type `T`, so all four pointers are reinterpreted as `T*`. Used by
// the Phase 4.5 Step 5 homogeneous dispatch, which reaches the
// `bestla_fusion_attn_forward<T, T, T, T>` overloads (fp16 stable / bf16
// non-stable). Field names match one-to-one, so this is a straight per-field
// port -- identical to make_typed_attn_args except Q and dst are typed `T`
// rather than `float`.
template <typename T>
bestla_mha::attn_fwd_args_t<T, T, T, T> make_typed_attn_args_homogeneous(const attn_fwd_args_t& a) {
  bestla_mha::attn_fwd_args_t<T, T, T, T> t{};
  t.Q = static_cast<T*>(a.Q);
  t.K = static_cast<T*>(a.K);
  t.V = static_cast<T*>(a.V);
  t.dst = static_cast<T*>(a.dst);
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
  return t;
}

void validate_causal_shape(const attn_fwd_args_t& args, const char* func_name) {
  if ((args.attn_flags & ATTN_FLAG_IS_CAUSAL) != 0 && args.sl_q > args.sl_kv) {
    throw std::invalid_argument(std::string(func_name) + ": causal mask requires sl_q <= sl_kv");
  }
}

// ---------------------------------------------------------------------------
// Phase 5 Step 1: feature-support matrix for the migrated CPU attention routes.
//
// First-layer dispatch is by the full Q/K/V/dst dtype tuple (a typed entry plus a
// `bestla_fusion_attn_forward<...>` specialization); the second layer selects the
// concrete kernel by ISA/layout/stride. The four migrated routes are kept DISTINCT
// -- NOT collapsed into generic mixed/homogeneous or stable/non-stable buckets:
//
//   # | entry(dtype)                          | Q/K/V/dst          | launcher (score) | core / ISA
//   --+---------------------------------------+--------------------+------------------+-----------------------
//   1 | bestla_sdpa_forward(F16)              | f32,f16,f16,f32    | stable  (fp32)   | SCoreRowNAvx2 / AVX2
//   2 | bestla_sdpa_forward(BF16)             | f32,bf16,bf16,f32  | stable  (fp32)   | SCoreRowNAvx512f/AVX512F
//     |                                       |                    |                  | or HCoreRowNAmxbf16/AMX-BF16
//   3 | bestla_sdpa_forward_homogeneous(F16)  | f16,f16,f16,f16    | stable  (fp16)   | HCoreRowNAvx512fp16/AVX512-FP16
//   4 | bestla_sdpa_forward_homogeneous(BF16) | bf16,bf16,bf16,bf16| non-stable(exp)  | HCoreRowNAmxbf16 / AMX-BF16
//
// Standard causal and GQA behavior is validated per route.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Phase 6: three-tier exposure policy for the non-int8 CPU BestLA attention routes.
//
// TIER 0 — Production (default Python sdpa())
//   Backend: scalar mha_dense_forward (see sdpa() fallback below).
//   Dtype:   f32 Q/K/V, f16 K/V, or bf16 K/V (homogeneous scalar path).
//   ISA:     any (no SIMD dependency beyond baseline).
//   Features: causal and GQA are implemented.
//   ABI:     stable, no env gate.
//   Status:  ready for public exposure; well-tested via test_ark_cpu_sdpa.py.
//
// TIER 1 — Mixed routes (routes 1/2), enabled by default
//   Backend: bestla_sdpa_forward (F16 = route 1, BF16 = route 2).
//   Dtype:   f32 Q, fp16/bf16 K/V, f32 dst.
//   ISA:     AVX2 (F16), AVX512F or AMX-BF16 (BF16).
//   Features: causal and GQA are implemented.
//   Status:  enabled by default.  Raw->packed reorder bridge is the per-forward path;
//             persistent packed KV cache (bestla_sdpa_forward_packed) is available
//             for long-lived decode workloads.
//   Mixed-route-only features are not part of the public sdpa() contract;
//     they remain reachable through the internal packed/mixed helpers and are
//     numerically covered by test_ark_cpu_mixed_bestla_sdpa.py and
//     test_ark_cpu_internal_sdpa.py.
//
// TIER 2 — Standard-SDPA internal optimizations (routes 3/4 homogeneous)
//   Backend: bestla_sdpa_forward_homogeneous (F16 = route 3, BF16 = route 4).
//   Dtype:   f16/f16/f16/f16 or bf16/bf16/bf16/bf16 (all operands homogeneous).
//   ISA:     AVX512-FP16 (F16), AMX-BF16 (BF16).
//   Features: route 3 supports causal+GQA; route 4 supports causal only.
//   ABI:     not wired in ark.cpp; not reachable from Python.
//   Status:  internal/debug only. Promotion criteria:
//     Route 3: Python-accessible packed K/V layout bridge (the homogeneous fp16
//       stable kernel expects weight_base_t K/V layout that raw PLAIN inputs don't
//       satisfy; bridging is future work).
//     Route 4: expose only if an AMX-BF16 bf16-compute preference use case is
//       identified (currently not justified given route 2 already covers bf16 K/V
//       with the full feature set and fp32-score stability).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Phase 4.5 Step 6: per-route pre-dispatch validation for the homogeneous SDPA
// launcher families.
//
// The homogeneous dtype tuple selects one of TWO DISTINCT launcher families
// (Neural-Speed structure, NOT a single "homogeneous" branch):
//
//   dtype | launcher family                       | core                  | ISA
//   ------+---------------------------------------+-----------------------+-------------
//   F16   | stable   mha_stable_interface_t       | HCoreRowNAvx512fp16   | AVX512-FP16
//   BF16  | non-stable mha_interface_t (exp-sum)  | HCoreRowNAmxbf16      | AMX-BF16
//
// Each launcher has its own layout/stride/head-count contract that the wrapper
// only guards with `assert` (a no-op in release builds). These helpers promote
// those contracts into user-facing std::invalid_argument guards that fire before
// any kernel work, so the two routes stay distinct and their assumptions are
// explicit and testable. Compact contract matrix (see the wrapper `compute`
// asserts the values are mirrored from):
//
//   contract           | fp16 stable route        | bf16 non-stable route
//   -------------------+--------------------------+-------------------------------
//   Q_layout           | PLAIN                     | PLAIN
//   dst_layout         | PLAIN                     | PLAIN
//   K_layout           | PLAIN or NTILE24_ROWPACK1 | PLAIN
//   V_layout           | PLAIN or NTILE24_ROWPACK1 | PLAIN
//   GQA (head_num)     | multiple of heads_kv      | == heads_kv (no GQA)
//   K PLAIN stride     | step_v_head_size == 1     | step_v_head_size == 1
//   V PLAIN stride     | step_k_sl == 1            | step_k_head_size==1 || step_k_sl==1
//   causal shape       | sl_q <= sl_kv             | sl_q <= sl_kv
// ---------------------------------------------------------------------------

// fp16 homogeneous route == the *stable* mha_stable_interface_t over
// gemm::HCoreRowNAvx512fp16. Mirrors the PLAIN/NTILE24 layout, GQA-multiple
// head-count, and PLAIN K/V stride assumptions asserted in
// mha_stable_interface_t::compute.
void validate_homogeneous_fp16_stable_route(const attn_fwd_args_t& a) {
  if (a.Q_layout != ATTN_FWD_LAYOUT_PLAIN || a.dst_layout != ATTN_FWD_LAYOUT_PLAIN) {
    throw std::invalid_argument(
        "ark::cpu::bestla_sdpa_forward_homogeneous: fp16 stable route requires PLAIN Q and dst layouts");
  }
  if (a.K_layout != ATTN_FWD_LAYOUT_PLAIN && a.K_layout != ATTN_FWD_LAYOUT_NTILE24_ROWPACK1) {
    throw std::invalid_argument(
        "ark::cpu::bestla_sdpa_forward_homogeneous: fp16 stable route K layout must be PLAIN or NTILE24_ROWPACK1");
  }
  if (a.V_layout != ATTN_FWD_LAYOUT_PLAIN && a.V_layout != ATTN_FWD_LAYOUT_NTILE24_ROWPACK1) {
    throw std::invalid_argument(
        "ark::cpu::bestla_sdpa_forward_homogeneous: fp16 stable route V layout must be PLAIN or NTILE24_ROWPACK1");
  }
  if (a.heads_kv <= 0 || a.head_num <= 0 || (a.head_num % a.heads_kv) != 0) {
    throw std::invalid_argument(
        "ark::cpu::bestla_sdpa_forward_homogeneous: fp16 stable route requires head_num to be a positive multiple "
        "of heads_kv (GQA groups)");
  }
  // Raw PLAIN K/V stride restrictions the stable interface relies on for its
  // contiguous inner reads (mha_stable_interface_t::compute asserts these).
  if (a.K_layout == ATTN_FWD_LAYOUT_PLAIN && a.step_v_head_size != 1) {
    throw std::invalid_argument(
        "ark::cpu::bestla_sdpa_forward_homogeneous: fp16 stable route requires contiguous V head-size stride "
        "(step_v_head_size == 1) when K is PLAIN");
  }
  if (a.V_layout == ATTN_FWD_LAYOUT_PLAIN && a.step_k_sl != 1) {
    throw std::invalid_argument(
        "ark::cpu::bestla_sdpa_forward_homogeneous: fp16 stable route requires contiguous K seq stride "
        "(step_k_sl == 1) when V is PLAIN");
  }
}

// bf16 homogeneous route == the *non-stable* mha_interface_t exp-sum path over
// gemm::HCoreRowNAmxbf16. Mirrors the all-PLAIN layout, no-GQA head-count, and
// contiguous K/V stride assumptions asserted in mha_interface_t::compute.
void validate_homogeneous_bf16_nonstable_route(const attn_fwd_args_t& a) {
  if (a.Q_layout != ATTN_FWD_LAYOUT_PLAIN || a.K_layout != ATTN_FWD_LAYOUT_PLAIN ||
      a.V_layout != ATTN_FWD_LAYOUT_PLAIN || a.dst_layout != ATTN_FWD_LAYOUT_PLAIN) {
    throw std::invalid_argument(
        "ark::cpu::bestla_sdpa_forward_homogeneous: bf16 non-stable route requires PLAIN Q/K/V/dst layouts");
  }
  if (a.head_num != a.heads_kv) {
    throw std::invalid_argument(
        "ark::cpu::bestla_sdpa_forward_homogeneous: bf16 non-stable route does not support GQA (requires "
        "head_num == heads_kv)");
  }
  if (a.step_v_head_size != 1) {
    throw std::invalid_argument(
        "ark::cpu::bestla_sdpa_forward_homogeneous: bf16 non-stable route requires contiguous V head-size stride "
        "(step_v_head_size == 1)");
  }
  if (a.step_k_head_size != 1 && a.step_k_sl != 1) {
    throw std::invalid_argument(
        "ark::cpu::bestla_sdpa_forward_homogeneous: bf16 non-stable route requires a contiguous K stride "
        "(step_k_head_size == 1 or step_k_sl == 1)");
  }
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
  // results. Phase 4 Step 1 builds the bridge below: raw PLAIN K/V are reordered
  // into the NTILE packed cache the kernels require before dispatch.
  if (args.Q_layout != ATTN_FWD_LAYOUT_PLAIN || args.K_layout != ATTN_FWD_LAYOUT_PLAIN ||
      args.V_layout != ATTN_FWD_LAYOUT_PLAIN || args.dst_layout != ATTN_FWD_LAYOUT_PLAIN) {
    throw std::invalid_argument("ark::cpu::bestla_sdpa_forward: only ATTN_FWD_LAYOUT_PLAIN is supported");
  }
  attn_fwd_args_t local = args;
  validate_causal_shape(local, "ark::cpu::bestla_sdpa_forward");
  // GQA (matrix row GQA == S): the stable interface maps grouped-query heads via
  // ihkv = ihn / (head_num / heads_kv) and requires head_num to be a positive
  // multiple of heads_kv; the raw->packed reorder below also groups K/V by
  // heads_kv, so enforce the same contract up front.
  if (args.heads_kv <= 0 || args.head_num <= 0 || (args.head_num % args.heads_kv) != 0) {
    throw std::invalid_argument(
        "ark::cpu::bestla_sdpa_forward: head_num must be a positive multiple of heads_kv (GQA groups)");
  }

  auto* cpu = bestla::device::CpuDevice::getInstance();
  const bool fp16_plain_avx512 =
      kv_dtype == BTLA_DTYPE::F16 && bestla_mha::MHA_PREFER_AVX512FP16 &&
      cpu->AVX512_FP16() && local.step_k_sl == 1;
  const bool fp16_plain_amx_features_ok = local.head_num == local.heads_kv;
  const bool fp16_plain_amx = kv_dtype == BTLA_DTYPE::F16 && cpu->AMX_BF16() &&
                              local.K_layout == ATTN_FWD_LAYOUT_PLAIN && local.V_layout == ATTN_FWD_LAYOUT_PLAIN &&
                              (local.step_k_head_size == 1 || local.step_k_sl == 1) && fp16_plain_amx_features_ok;

  if (kv_dtype == BTLA_DTYPE::F16 && !(cpu->AVX2() || fp16_plain_avx512 || fp16_plain_amx)) {
    throw std::runtime_error(
        "ark::cpu::bestla_sdpa_forward: fp16 mixed SDPA requires AVX2 packed K/V, "
        "or AVX512-FP16 plain K/V with step_k_sl == 1, or the AMX-BF16 plain route");
  }
  if (kv_dtype == BTLA_DTYPE::BF16 && !cpu->AVX512F() && !cpu->AMX_BF16()) {
    throw std::runtime_error(
        "ark::cpu::bestla_sdpa_forward: bf16 K/V mixed SDPA requires AVX512F or AMX-BF16; "
        "this CPU/build does not provide it");
  }

  // Threading is supplied by the caller (ARK reuses CpuWrapper::get_threading()),
  // type-erased through attn_fwd_args_t::threading. This avoids maintaining a
  // second independent BestLA thread pool in the CPU attention path.
  if (args.threading == nullptr) {
    throw std::invalid_argument("ark::cpu::bestla_sdpa_forward: threading pool must be provided");
  }
  auto* th = static_cast<bestla::parallel::IThreading*>(args.threading);

  // Allocate (or reuse) the BestLA wrapper scratch.  The softmax epilogues
  // issue aligned AVX stores (_mm256_store_ps / _mm512_store_ps), so the base
  // must be 64B-aligned.  The workspace cache (find_or_alloc_workspace) avoids
  // a per-call allocation + zero-fill when the shape/thread-count is stable
  // across calls — common in decode loops.
  if (local.tmp == nullptr) {
    attn_shape_t shape{local.batch_size, local.head_num, local.heads_kv, local.head_size, local.sl_q, local.sl_kv};
    const size_t bytes = bestla_attn_workspace_size(shape, th->num_threads());
    local.tmp = find_or_alloc_workspace(bytes, local.sl_q, local.sl_kv, th->num_threads());
  }

  if (kv_dtype == BTLA_DTYPE::F16 && (fp16_plain_avx512 || fp16_plain_amx)) {
    const auto typed = make_typed_attn_args<bestla::utils::fp16>(local);
    bestla_mha::bestla_fusion_attn_forward<float, bestla::utils::fp16, bestla::utils::fp16, float>(typed, *th);
    return;
  }

  // Phase 4 Step 1: bridge raw PLAIN HND/NHD K/V into the Neural-Speed-style
  // NTILE packed/reordered cache the wired mixed kernels require. The kernel's
  // QK weight is K (NTILE over seq, ROWPACK over head_size) and its PV weight is
  // V (NTILE over head_size, ROWPACK over seq). We allocate per-head packed
  // caches, fill them from the strided inputs, then retarget `local` at the
  // packed layouts/strides. Q and dst stay PLAIN. This path is the default
  // for mixed SDPA. Persistent packed KV cache is available via
  // bestla_sdpa_forward_packed.
  //
  // Buffer alignment: both wired dtypes (fp16/bf16) are 16-bit, so a
  // std::vector<uint16_t> backing matches element_size() and gives the natural
  // 2-byte element alignment. The NTILE24 (fp16->fp32 F16C) and NTILE48
  // (bf16->fp32) weight prologues read these caches with unaligned SIMD loads
  // (load_T_fp32 / vcvtph2ps over 8-lane groups), so 2-byte alignment is
  // sufficient and no over-aligned allocation is required here. int8/ROWPACK4
  // is not wired, so no wider element ever lands in these buffers.
  std::vector<uint16_t> packed_k;
  std::vector<uint16_t> packed_v;
  const ReorderKVShape rshape =
      reorder_kv_shape(local.batch_size, local.heads_kv, local.sl_kv, local.head_size, kv_dtype);
  packed_k.resize(reorder_kv_cache_elems(rshape, /*is_value=*/false));
  packed_v.resize(reorder_kv_cache_elems(rshape, /*is_value=*/true));
  AttentionStrides k_in{local.step_k_sl, local.step_k_head_size, local.step_k_head_num, local.step_k_bs};
  ValueStrides v_in{local.step_v_head_size, local.step_v_sl, local.step_v_head_num, local.step_v_bs};
  reorder_k_to_packed(packed_k.data(), local.K, rshape, k_in);
  reorder_v_to_packed(packed_v.data(), local.V, rshape, v_in);
  local.K = packed_k.data();
  local.V = packed_v.data();
  local.K_layout = rshape.k_layout;
  local.V_layout = rshape.v_layout;
  local.step_k_head_num = rshape.step_k_head_num;
  local.step_k_bs = rshape.step_k_bs;
  local.step_k_sl = rshape.step_k_sl;
  local.step_k_head_size = rshape.step_k_head_size;
  local.step_v_head_num = rshape.step_v_head_num;
  local.step_v_bs = rshape.step_v_bs;
  local.step_v_sl = rshape.step_v_sl;
  local.step_v_head_size = rshape.step_v_head_size;

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

void bestla_sdpa_forward_homogeneous(const attn_fwd_args_t& args, BTLA_DTYPE dtype) {
  if (!args.Q || !args.K || !args.V || !args.dst) {
    throw std::invalid_argument("ark::cpu::bestla_sdpa_forward_homogeneous: Q/K/V/dst pointers must be non-null");
  }
  // First-layer dispatch is by the full Q/K/V/dst dtype tuple (Neural-Speed
  // style). Only the homogeneous fp16/bf16 tuples migrated in Phase 4.5 steps
  // 3-4 are wired; reject any other operand type before touching the kernel.
  if (dtype != BTLA_DTYPE::F16 && dtype != BTLA_DTYPE::BF16) {
    throw std::invalid_argument(
        "ark::cpu::bestla_sdpa_forward_homogeneous: only homogeneous F16 or BF16 (Q==K==V==dst) is supported");
  }
  attn_fwd_args_t local = args;
  validate_causal_shape(local, "ark::cpu::bestla_sdpa_forward_homogeneous");

  // Second-layer route contract: each homogeneous dtype reaches a DISTINCT
  // launcher family (fp16 -> stable mha_stable_interface_t, bf16 -> non-stable
  // mha_interface_t), and each has its own layout/stride/GQA contract. Validate
  // the incoming operands against the exact route that will run so a violation
  // fails loudly here with std::invalid_argument instead of tripping a
  // release-mode-stripped assert (or silently mis-reading) inside the kernel.
  // The two routes are validated separately on purpose -- this is NOT collapsed
  // into one "homogeneous" check.
  if (dtype == BTLA_DTYPE::F16) {
    validate_homogeneous_fp16_stable_route(local);
  } else {  // BTLA_DTYPE::BF16 (guaranteed by the first-layer dtype gate above)
    validate_homogeneous_bf16_nonstable_route(local);
  }

  // Second-layer condition (ISA): the homogeneous overloads compose ISA-specific
  // cores whose prologues silently return BTLA_CODE::NotSupport (behind asserts)
  // on hardware that lacks the extension. Gate up front so the failure is a clear
  // error instead of a release-mode no-op / wrong result:
  //   * F16  -> stable mha_stable_interface_t over HCoreRowNAvx512fp16, needs AVX512-FP16.
  //   * BF16 -> non-stable mha_interface_t exp-sum over HCoreRowNAmxbf16, needs AMX-BF16.
  {
    auto* cpu = bestla::device::CpuDevice::getInstance();
    if (dtype == BTLA_DTYPE::F16 && !cpu->AVX512_FP16()) {
      throw std::runtime_error(
          "ark::cpu::bestla_sdpa_forward_homogeneous: homogeneous fp16 attention requires an AVX512-FP16 CPU with "
          "the stable mha_stable_interface_t over gemm::HCoreRowNAvx512fp16; this CPU/build does not provide it");
    }
    if (dtype == BTLA_DTYPE::BF16 && !cpu->AMX_BF16()) {
      throw std::runtime_error(
          "ark::cpu::bestla_sdpa_forward_homogeneous: homogeneous bf16 attention requires an AMX-BF16 CPU with the "
          "non-stable mha_interface_t over gemm::HCoreRowNAmxbf16; this CPU/build does not provide it");
    }
  }

  if (local.threading == nullptr) {
    throw std::invalid_argument("ark::cpu::bestla_sdpa_forward_homogeneous: threading pool must be provided");
  }
  auto* th = static_cast<bestla::parallel::IThreading*>(local.threading);
  attn_shape_t shape{local.batch_size, local.head_num, local.heads_kv, local.head_size, local.sl_q, local.sl_kv};
  const size_t workspace_bytes =
      dtype == BTLA_DTYPE::BF16 ? bestla_route4_workspace_size(shape, th->num_threads())
                                : bestla_attn_workspace_size(shape, th->num_threads());

  // Allocate (or reuse) the wrapper scratch.
  if (local.tmp == nullptr) {
    local.tmp = find_or_alloc_workspace(workspace_bytes, local.sl_q, local.sl_kv, th->num_threads());
  }

  // No raw->packed reorder bridge here (unlike the mixed route): the homogeneous
  // prologues pack/convert K/V themselves -- bf16 through the batch packers, fp16
  // through the plain `weight_base_t` forward prologue -- so the operands are
  // forwarded with their incoming strides/layout untouched.
  switch (dtype) {
    case BTLA_DTYPE::F16: {
      const auto typed = make_typed_attn_args_homogeneous<bestla::utils::fp16>(local);
      bestla_mha::bestla_fusion_attn_forward<bestla::utils::fp16, bestla::utils::fp16, bestla::utils::fp16,
                                             bestla::utils::fp16>(typed, *th);
      break;
    }
    case BTLA_DTYPE::BF16: {
      const auto typed = make_typed_attn_args_homogeneous<bestla::utils::bf16>(local);
      bestla_mha::bestla_fusion_attn_forward<bestla::utils::bf16, bestla::utils::bf16, bestla::utils::bf16,
                                             bestla::utils::bf16>(typed, *th);
      break;
    }
    default:
      throw std::invalid_argument(
          "ark::cpu::bestla_sdpa_forward_homogeneous: only homogeneous F16 or BF16 is supported");
  }
}

namespace {

// Pad helper.
int pad_up(int v, int p) { return ((v + p - 1) / p) * p; }

size_t packed_k_index(const ReorderKVShape& shape, int seq_pos, int d) {
  const int tile = seq_pos / shape.ntile;
  const int sl_in = seq_pos % shape.ntile;
  const int kp = d / shape.rowpack;
  const int rp_i = d % shape.rowpack;
  return static_cast<size_t>(tile) * shape.k_head_size_pad * shape.ntile +
         static_cast<size_t>(kp) * shape.ntile * shape.rowpack + static_cast<size_t>(sl_in) * shape.rowpack + rp_i;
}

size_t packed_v_index(const ReorderKVShape& shape, int seq_pos, int d) {
  const int tile = d / shape.ntile;
  const int hs_in = d % shape.ntile;
  const int kp = seq_pos / shape.rowpack;
  const int rp_i = seq_pos % shape.rowpack;
  return static_cast<size_t>(tile) * shape.v_seq_pad * shape.ntile +
         static_cast<size_t>(kp) * shape.ntile * shape.rowpack + static_cast<size_t>(hs_in) * shape.rowpack + rp_i;
}

int v_zero_pad_multiple(const ReorderKVShape& shape) { return shape.dtype == BTLA_DTYPE::BF16 ? 32 : 1; }

}  // namespace

void bestla_sdpa_forward_packed(const attn_fwd_args_t& args, const ReorderKVShape& shape) {
  if (!args.Q || !args.K || !args.V || !args.dst) {
    throw std::invalid_argument("ark::cpu::bestla_sdpa_forward_packed: Q/K/V/dst pointers must be non-null");
  }
  // Q and dst stay PLAIN; K/V must already be the NTILE-packed cache for kv_dtype.
  if (args.Q_layout != ATTN_FWD_LAYOUT_PLAIN || args.dst_layout != ATTN_FWD_LAYOUT_PLAIN) {
    throw std::invalid_argument("ark::cpu::bestla_sdpa_forward_packed: Q/dst must be ATTN_FWD_LAYOUT_PLAIN");
  }
  if (args.K_layout != shape.k_layout || args.V_layout != shape.v_layout) {
    throw std::invalid_argument("ark::cpu::bestla_sdpa_forward_packed: K/V layout must match packed cache shape");
  }
  if ((shape.dtype == BTLA_DTYPE::F16 && shape.k_layout != ATTN_FWD_LAYOUT_NTILE24_ROWPACK1) ||
      (shape.dtype == BTLA_DTYPE::BF16 && shape.k_layout != ATTN_FWD_LAYOUT_NTILE48_ROWPACK2)) {
    throw std::invalid_argument("ark::cpu::bestla_sdpa_forward_packed: dtype/layout mismatch for packed cache");
  }
  // sl_kv is the current valid length, never the padded capacity.
  if (args.sl_kv <= 0 || args.sl_kv > shape.logical_capacity) {
    throw std::invalid_argument("ark::cpu::bestla_sdpa_forward_packed: sl_kv must be in (0, logical_capacity]");
  }
  if (args.head_size != shape.head_dim) {
    throw std::invalid_argument("ark::cpu::bestla_sdpa_forward_packed: head_size must match packed cache head_dim");
  }
  attn_fwd_args_t local = args;
  validate_causal_shape(local, "ark::cpu::bestla_sdpa_forward_packed");
  if (args.heads_kv <= 0 || args.head_num <= 0 || (args.head_num % args.heads_kv) != 0) {
    throw std::invalid_argument(
        "ark::cpu::bestla_sdpa_forward_packed: head_num must be a positive multiple of heads_kv (GQA groups)");
  }
  {
    auto* cpu = bestla::device::CpuDevice::getInstance();
    if (shape.dtype == BTLA_DTYPE::F16 && !cpu->AVX2()) {
      throw std::runtime_error("ark::cpu::bestla_sdpa_forward_packed: fp16 K/V mixed SDPA requires AVX2");
    }
    if (shape.dtype == BTLA_DTYPE::BF16 && !cpu->AVX512F()) {
      throw std::runtime_error("ark::cpu::bestla_sdpa_forward_packed: bf16 K/V mixed SDPA requires AVX512F");
    }
  }
  if (args.threading == nullptr) {
    throw std::invalid_argument("ark::cpu::bestla_sdpa_forward_packed: threading pool must be provided");
  }
  auto* th = static_cast<bestla::parallel::IThreading*>(args.threading);

  // Retarget the packed K/V strides from the cache shape (no reorder happens
  // here: K/V are already NTILE-packed). Q/dst pointers/strides are untouched.
  local.step_k_head_num = shape.step_k_head_num;
  local.step_k_bs = shape.step_k_bs;
  local.step_k_sl = shape.step_k_sl;
  local.step_k_head_size = shape.step_k_head_size;
  local.step_v_head_num = shape.step_v_head_num;
  local.step_v_bs = shape.step_v_bs;
  local.step_v_sl = shape.step_v_sl;
  local.step_v_head_size = shape.step_v_head_size;

  // Allocate (or reuse) the wrapper scratch.
  if (local.tmp == nullptr) {
    attn_shape_t ashape{local.batch_size, local.head_num, local.heads_kv, local.head_size, local.sl_q, local.sl_kv};
    const size_t bytes = bestla_attn_workspace_size(ashape, th->num_threads());
    local.tmp = find_or_alloc_workspace(bytes, local.sl_q, local.sl_kv, th->num_threads());
  }

  switch (shape.dtype) {
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
      throw std::invalid_argument("ark::cpu::bestla_sdpa_forward_packed: only F16 and BF16 K/V operands are supported");
  }
}

ReorderKVShape reorder_kv_shape(int batch, int num_heads_kv, int seq_len_kv, int head_dim, BTLA_DTYPE kv_dtype) {
  ReorderKVShape s;
  s.dtype = kv_dtype;
  switch (kv_dtype) {
    case BTLA_DTYPE::F16:
      s.layout = ATTN_FWD_LAYOUT_NTILE24_ROWPACK1;
      s.k_layout = ATTN_FWD_LAYOUT_NTILE24_ROWPACK1;
      s.v_layout = ATTN_FWD_LAYOUT_NTILE24_ROWPACK1;
      s.ntile = 24;
      s.rowpack = 1;
      break;
    case BTLA_DTYPE::BF16:
      s.layout = ATTN_FWD_LAYOUT_NTILE48_ROWPACK2;
      s.k_layout = ATTN_FWD_LAYOUT_NTILE48_ROWPACK2;
      s.v_layout = ATTN_FWD_LAYOUT_NTILE48_ROWPACK2;
      s.ntile = 48;
      s.rowpack = 2;
      break;
    default:
      throw std::invalid_argument("ark::cpu::reorder_kv_shape: only F16 and BF16 K/V are supported");
  }
  if (batch <= 0 || num_heads_kv <= 0 || seq_len_kv <= 0 || head_dim <= 0) {
    throw std::invalid_argument("ark::cpu::reorder_kv_shape: invalid dimensions");
  }
  s.batch_size = batch;
  s.heads_kv = num_heads_kv;
  s.head_dim = head_dim;
  s.logical_capacity = seq_len_kv;
  s.num_heads = batch * num_heads_kv;
  s.elem_bytes = element_size(kv_dtype);
  // K is the QK weight: NTILE blocks over seq, head_size is ROWPACK-packed.
  // NS aligns BF16 K's reduction dimension to 32 for AMX cache ABI parity.
  s.k_seq_pad = pad_up(seq_len_kv, s.ntile);
  s.k_head_size_pad = pad_up(head_dim, kv_dtype == BTLA_DTYPE::BF16 ? 32 : s.rowpack);
  s.k_head_elems = static_cast<size_t>(s.k_seq_pad) * static_cast<size_t>(s.k_head_size_pad);
  s.k_total_elems = s.k_head_elems * static_cast<size_t>(s.num_heads);
  s.k_bytes = s.k_total_elems * s.elem_bytes;
  s.step_k_head_num = static_cast<int>(s.k_head_elems);
  s.step_k_bs = s.step_k_head_num * s.heads_kv;
  s.step_k_sl = s.k_head_size_pad;
  s.step_k_head_size = 1;
  // V is the PV weight: NTILE blocks over head_size, seq is ROWPACK-packed.
  // NS aligns BF16 V's reduction dimension to 32 for AMX cache ABI parity.
  s.v_seq_pad = pad_up(seq_len_kv, kv_dtype == BTLA_DTYPE::BF16 ? 32 : s.rowpack);
  s.v_head_size_pad = pad_up(head_dim, s.ntile);
  s.v_head_elems = static_cast<size_t>(s.v_seq_pad) * static_cast<size_t>(s.v_head_size_pad);
  s.v_total_elems = s.v_head_elems * static_cast<size_t>(s.num_heads);
  s.v_bytes = s.v_total_elems * s.elem_bytes;
  s.step_v_head_num = static_cast<int>(s.v_head_elems);
  s.step_v_bs = s.step_v_head_num * s.heads_kv;
  s.step_v_sl = 1;
  s.step_v_head_size = s.v_seq_pad;
  return s;
}

size_t reorder_kv_cache_elems(const ReorderKVShape& shape, bool is_value) {
  return is_value ? shape.v_total_elems : shape.k_total_elems;
}

void reorder_k_to_packed(void* dst, const void* src, const ReorderKVShape& shape, const AttentionStrides& k_strides) {
  if (!dst || !src) {
    throw std::invalid_argument("ark::cpu::reorder_k_to_packed: dst/src must be non-null");
  }
  const int ntile = shape.ntile;
  const int rp = shape.rowpack;
  const int batch = shape.batch_size;
  const int num_heads_kv = shape.heads_kv;
  const int seq_len_kv = shape.logical_capacity;
  const int head_dim = shape.head_dim;
  const int hs_pad = shape.k_head_size_pad;
  bool use_avx2 = false;
#if CompileAVX2()
  use_avx2 = can_use_16bit_reorder_avx2(shape, k_strides.dim);
#endif
  // K element (sl, hs) -> tile of NTILE over sl, ROWPACK over head_size.
  //   tile = sl/NTILE, sl_in = sl%NTILE, kp = hs/rp, rp_i = hs%rp
  //   idx = tile*(hs_pad*NTILE) + kp*(NTILE*rp) + sl_in*rp + rp_i
  std::memset(dst, 0, shape.k_bytes);
#pragma omp parallel for collapse(2) schedule(static)
  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < num_heads_kv; ++h) {
      const size_t head_base = (static_cast<size_t>(b) * num_heads_kv + h) * shape.k_head_elems;
#if CompileAVX2()
      if (use_avx2) {
        reorder_k_16bit_avx2(static_cast<uint16_t*>(dst), static_cast<const uint16_t*>(src), shape, k_strides, b, h);
      }
#endif
      for (int s = 0; s < seq_len_kv; ++s) {
        const int tile = s / ntile, sl_in = s % ntile;
        for (int d = 0; d < head_dim; ++d) {
          if (use_avx2 && s < seq_len_kv / 8 * 8 && d < head_dim / 8 * 8) {
            continue;
          }
          const float val = load_scalar(src, qko_offset(k_strides, b, h, s, d), shape.dtype);
          const int kp = d / rp, rp_i = d % rp;
          const size_t idx = static_cast<size_t>(tile) * hs_pad * ntile + static_cast<size_t>(kp) * ntile * rp +
                             static_cast<size_t>(sl_in) * rp + rp_i;
          store_scalar(dst, head_base + idx, shape.dtype, val);
        }
      }
    }
  }
}

void reorder_v_to_packed(void* dst, const void* src, const ReorderKVShape& shape, const ValueStrides& v_strides) {
  if (!dst || !src) {
    throw std::invalid_argument("ark::cpu::reorder_v_to_packed: dst/src must be non-null");
  }
  const int ntile = shape.ntile;
  const int rp = shape.rowpack;
  const int batch = shape.batch_size;
  const int num_heads_kv = shape.heads_kv;
  const int seq_len_kv = shape.logical_capacity;
  const int head_dim = shape.head_dim;
  const int sl_pad = shape.v_seq_pad;  // V: ROWPACK over seq
  bool use_avx2 = false;
#if CompileAVX2()
  use_avx2 = can_use_16bit_reorder_avx2(shape, v_strides.dim);
#endif
  // V element (sl, hs) -> tile of NTILE over head_size, ROWPACK over seq.
  //   tile = hs/NTILE, hs_in = hs%NTILE, kp = sl/rp, rp_i = sl%rp
  //   idx = tile*(sl_pad*NTILE) + kp*(NTILE*rp) + hs_in*rp + rp_i
  std::memset(dst, 0, shape.v_bytes);
#pragma omp parallel for collapse(2) schedule(static)
  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < num_heads_kv; ++h) {
      const size_t head_base = (static_cast<size_t>(b) * num_heads_kv + h) * shape.v_head_elems;
#if CompileAVX2()
      if (use_avx2) {
        reorder_v_16bit_avx2(static_cast<uint16_t*>(dst), static_cast<const uint16_t*>(src), shape, v_strides, b, h);
      }
#endif
      for (int s = 0; s < seq_len_kv; ++s) {
        const int kp = s / rp, rp_i = s % rp;
        for (int d = 0; d < head_dim; ++d) {
          const bool simd_covered =
              use_avx2 && d < head_dim / 8 * 8 && (shape.dtype == BTLA_DTYPE::F16 || s < seq_len_kv / 2 * 2);
          if (simd_covered) {
            continue;
          }
          const float val = load_scalar(src, value_offset(v_strides, b, h, s, d), shape.dtype);
          const int tile = d / ntile, hs_in = d % ntile;
          const size_t idx = static_cast<size_t>(tile) * sl_pad * ntile + static_cast<size_t>(kp) * ntile * rp +
                             static_cast<size_t>(hs_in) * rp + rp_i;
          store_scalar(dst, head_base + idx, shape.dtype, val);
        }
      }
    }
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

ReorderKVShape packed_kv_cache_shape(int batch, int num_heads_kv, int capacity, int head_dim, BTLA_DTYPE kv_dtype) {
  // Identical layout/strides to reorder_kv_shape, but the seq dim is padded to
  // the persistent `capacity` instead of the current sequence length. The
  // logical_capacity field preserves the real capacity so the update helpers can
  // reject writes past it even though buffers are padded to NTILE/ROWPACK.
  return reorder_kv_shape(batch, num_heads_kv, capacity, head_dim, kv_dtype);
}

ReorderKVShape packed_kv_cache_info(int batch, int num_heads_kv, int capacity, int head_dim, BTLA_DTYPE kv_dtype) {
  return packed_kv_cache_shape(batch, num_heads_kv, capacity, head_dim, kv_dtype);
}

void clear_packed_k_cache(void* cache_k, const ReorderKVShape& shape) {
  if (!cache_k) {
    throw std::invalid_argument("ark::cpu::clear_packed_k_cache: cache must be non-null");
  }
  std::memset(cache_k, 0, shape.k_bytes);
}

void clear_packed_v_cache(void* cache_v, const ReorderKVShape& shape) {
  if (!cache_v) {
    throw std::invalid_argument("ark::cpu::clear_packed_v_cache: cache must be non-null");
  }
  std::memset(cache_v, 0, shape.v_bytes);
}

void update_packed_k_cache(void* cache_k, const void* key, const ReorderKVShape& shape,
                           const AttentionStrides& k_strides, int append_len, int start_pos, bool no_zeroing) {
  if (!cache_k || !key) {
    throw std::invalid_argument("ark::cpu::update_packed_k_cache: cache/src must be non-null");
  }
  if (shape.dtype != BTLA_DTYPE::F16 && shape.dtype != BTLA_DTYPE::BF16) {
    throw std::invalid_argument("ark::cpu::update_packed_k_cache: only F16 and BF16 K are supported");
  }
  const int batch = shape.batch_size;
  const int num_heads_kv = shape.heads_kv;
  const int head_dim = shape.head_dim;
  const int ntile = shape.ntile, rp = shape.rowpack;
  const int hs_pad = shape.k_head_size_pad;
  // Reject writes beyond the *logical* capacity, not the NTILE-padded capacity.
  const int cap = shape.logical_capacity;
  if (batch <= 0 || num_heads_kv <= 0 || append_len <= 0 || head_dim <= 0 || start_pos < 0 ||
      start_pos + append_len > cap) {
    throw std::invalid_argument("ark::cpu::update_packed_k_cache: invalid dimensions or append range");
  }
  // K (QK weight): NTILE over seq, ROWPACK over head_size. Source read via
  // strides only (HND/NHD agnostic). Padded head_size columns are zero-filled.
#pragma omp parallel for collapse(2) schedule(static)
  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < num_heads_kv; ++h) {
      const size_t head_base = (static_cast<size_t>(b) * num_heads_kv + h) * shape.k_head_elems;
      for (int s = 0; s < append_len; ++s) {
        const int pos = start_pos + s;
        const int tile = pos / ntile, sl_in = pos % ntile;
        const int d_limit = no_zeroing ? head_dim : hs_pad;
        for (int d = 0; d < d_limit; ++d) {
          const float val = d < head_dim ? load_scalar(key, qko_offset(k_strides, b, h, s, d), shape.dtype) : 0.0f;
          const int kp = d / rp, rp_i = d % rp;
          const size_t idx = static_cast<size_t>(tile) * hs_pad * ntile + static_cast<size_t>(kp) * ntile * rp +
                             static_cast<size_t>(sl_in) * rp + rp_i;
          store_scalar(cache_k, head_base + idx, shape.dtype, val);
        }
      }
    }
  }
}

void update_packed_v_cache(void* cache_v, const void* value, const ReorderKVShape& shape,
                           const ValueStrides& v_strides, int append_len, int start_pos, bool no_zeroing) {
  if (!cache_v || !value) {
    throw std::invalid_argument("ark::cpu::update_packed_v_cache: cache/src must be non-null");
  }
  if (shape.dtype != BTLA_DTYPE::F16 && shape.dtype != BTLA_DTYPE::BF16) {
    throw std::invalid_argument("ark::cpu::update_packed_v_cache: only F16 and BF16 V are supported");
  }
  const int batch = shape.batch_size;
  const int num_heads_kv = shape.heads_kv;
  const int head_dim = shape.head_dim;
  const int ntile = shape.ntile, rp = shape.rowpack;
  const int hs_pad = shape.v_head_size_pad;
  const int sl_pad = shape.v_seq_pad;
  // Reject writes beyond the *logical* capacity, not the ROWPACK-padded capacity.
  const int cap = shape.logical_capacity;
  if (batch <= 0 || num_heads_kv <= 0 || append_len <= 0 || head_dim <= 0 || start_pos < 0 ||
      start_pos + append_len > cap) {
    throw std::invalid_argument("ark::cpu::update_packed_v_cache: invalid dimensions or append range");
  }
  // V (PV weight): NTILE over head_size, ROWPACK over seq. Padded head_size rows
  // zero-filled. Source read via strides only (HND/NHD agnostic).
#pragma omp parallel for collapse(2) schedule(static)
  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < num_heads_kv; ++h) {
      const size_t head_base = (static_cast<size_t>(b) * num_heads_kv + h) * shape.v_head_elems;
      const int end = start_pos + append_len;
      const int zero_end = no_zeroing ? end : std::min(shape.v_seq_pad, pad_up(end, v_zero_pad_multiple(shape)));
      for (int pos = start_pos; pos < zero_end; ++pos) {
        const bool is_logical = pos < end;
        const int s = pos - start_pos;
        const int kp = pos / rp, rp_i = pos % rp;
        const int d_limit = no_zeroing ? head_dim : hs_pad;
        for (int d = 0; d < d_limit; ++d) {
          const float val =
              (is_logical && d < head_dim) ? load_scalar(value, value_offset(v_strides, b, h, s, d), shape.dtype) : 0.0f;
          const int tile = d / ntile, hs_in = d % ntile;
          const size_t idx = static_cast<size_t>(tile) * sl_pad * ntile + static_cast<size_t>(kp) * ntile * rp +
                             static_cast<size_t>(hs_in) * rp + rp_i;
          store_scalar(cache_v, head_base + idx, shape.dtype, val);
        }
      }
    }
  }
}

void copy_packed_k_cache(void* dst_cache_k, const void* src_cache_k, const ReorderKVShape& shape, int seq_off,
                         int seq_size, bool no_zeroing) {
  if (!dst_cache_k || !src_cache_k) {
    throw std::invalid_argument("ark::cpu::copy_packed_k_cache: src/dst cache must be non-null");
  }
  const int cap = shape.logical_capacity;
  if (seq_off < 0 || seq_size <= 0 || seq_off + seq_size > cap) {
    throw std::invalid_argument("ark::cpu::copy_packed_k_cache: invalid copy range");
  }
  const int end = seq_off + seq_size;
  const int copy_end = no_zeroing ? end : std::min(shape.k_seq_pad, pad_up(end, shape.ntile));
#pragma omp parallel for collapse(2) schedule(static)
  for (int b = 0; b < shape.batch_size; ++b) {
    for (int h = 0; h < shape.heads_kv; ++h) {
      const size_t head_base = (static_cast<size_t>(b) * shape.heads_kv + h) * shape.k_head_elems;
      const int d_limit = no_zeroing ? shape.head_dim : shape.k_head_size_pad;
      for (int pos = seq_off; pos < copy_end; ++pos) {
        for (int d = 0; d < d_limit; ++d) {
          const size_t idx = packed_k_index(shape, pos, d);
          const float val = load_scalar(src_cache_k, head_base + idx, shape.dtype);
          store_scalar(dst_cache_k, head_base + idx, shape.dtype, val);
        }
      }
    }
  }
}

void copy_packed_v_cache(void* dst_cache_v, const void* src_cache_v, const ReorderKVShape& shape, int seq_off,
                         int seq_size, bool no_zeroing) {
  if (!dst_cache_v || !src_cache_v) {
    throw std::invalid_argument("ark::cpu::copy_packed_v_cache: src/dst cache must be non-null");
  }
  const int cap = shape.logical_capacity;
  if (seq_off < 0 || seq_size <= 0 || seq_off + seq_size > cap) {
    throw std::invalid_argument("ark::cpu::copy_packed_v_cache: invalid copy range");
  }
  const int end = seq_off + seq_size;
  const int copy_end = no_zeroing ? end : std::min(shape.v_seq_pad, pad_up(end, v_zero_pad_multiple(shape)));
#pragma omp parallel for collapse(2) schedule(static)
  for (int b = 0; b < shape.batch_size; ++b) {
    for (int h = 0; h < shape.heads_kv; ++h) {
      const size_t head_base = (static_cast<size_t>(b) * shape.heads_kv + h) * shape.v_head_elems;
      const int d_limit = no_zeroing ? shape.head_dim : shape.v_head_size_pad;
      for (int pos = seq_off; pos < copy_end; ++pos) {
        for (int d = 0; d < d_limit; ++d) {
          const size_t idx = packed_v_index(shape, pos, d);
          const float val = load_scalar(src_cache_v, head_base + idx, shape.dtype);
          store_scalar(dst_cache_v, head_base + idx, shape.dtype, val);
        }
      }
    }
  }
}

void shift_packed_k_cache_rope(void* cache_k, const void* cossin, const ReorderKVShape& shape, int seq_keep) {
  if (!cache_k || !cossin) {
    throw std::invalid_argument("ark::cpu::shift_packed_k_cache_rope: cache and cossin must be non-null");
  }
  if (shape.dtype != BTLA_DTYPE::BF16 || shape.k_layout != ATTN_FWD_LAYOUT_NTILE48_ROWPACK2) {
    throw std::invalid_argument(
        "ark::cpu::shift_packed_k_cache_rope: only BF16 / NTILE48_ROWPACK2 packed K cache is supported");
  }
  if (seq_keep < 0 || seq_keep > shape.logical_capacity) {
    throw std::invalid_argument("ark::cpu::shift_packed_k_cache_rope: seq_keep must be in [0, logical_capacity]");
  }
#pragma omp parallel for collapse(2) schedule(static)
  for (int b = 0; b < shape.batch_size; ++b) {
    for (int h = 0; h < shape.heads_kv; ++h) {
      auto* src = reinterpret_cast<bestla::utils::bf16*>(cache_k) +
                  (static_cast<size_t>(b) * shape.step_k_bs + static_cast<size_t>(h) * shape.step_k_head_num);
      bestla::kernel::jit::CScaleInterleavedBF16FP16::forward<48>(
          src, reinterpret_cast<const bestla::utils::fp16*>(cossin), shape.head_dim, shape.k_seq_pad,
          shape.k_head_size_pad, seq_keep);
    }
  }
}

}  // namespace ark::cpu
