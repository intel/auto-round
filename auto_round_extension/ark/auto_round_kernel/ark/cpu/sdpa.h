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

#pragma once

#include "mha_dense.h"

namespace ark::cpu {

enum class SdpaLayout : int { HND = 0, NHD = 1 };

void sdpa_forward(const MhaDenseArgs& args);

// Neural-Speed-style BestLA attention entry (Phase 3 migration).
//
// Builds the dtype-typed `bestla_mha::attn_fwd_args_t<Q_T, K_T, V_T, DST_T>`
// from the type-erased `attn_fwd_args_t` (Phase 1 ABI struct) and dispatches it
// through `bestla_mha::bestla_fusion_attn_forward`, the migrated wrapper. The
// K/V operand element type selects the specialization that is wired today:
//   * BTLA_DTYPE::F16  -> attn_fwd_args_t<float, fp16, fp16, float>
//   * BTLA_DTYPE::BF16 -> attn_fwd_args_t<float, bf16, bf16, float>
// Q and dst are always FP32. Unsupported K/V dtypes raise std::invalid_argument.
//
// The caller supplies the BestLA thread pool through `args.threading` (a
// `bestla::parallel::IThreading*`, type-erased as void*); ARK passes
// `CpuWrapper::get_threading()` so the attention path shares the same pool as
// the rest of the CPU kernels. When `args.tmp` is null the wrapper scratch is
// allocated internally (as a float-aligned buffer) for the duration of the
// call. This entry validates PLAIN-strided operands and rejects alibi, tanh and
// padding-right flags; note, however, that the `step_*` stride interface being
// HND/NHD-friendly does NOT mean the wired mixed-precision kernels accept raw
// HND/NHD/PLAIN K/V. Those specializations currently require packed/reordered
// (NTILE24/NTILE48) K/V and throw std::runtime_error for raw PLAIN inputs;
// packed K/V support is deferred to Phase 4.
void bestla_sdpa_forward(const attn_fwd_args_t& args, BTLA_DTYPE kv_dtype);

// ---------------------------------------------------------------------------
// Phase 4 Step 1: raw HND/NHD K/V -> Neural-Speed NTILE packed/reordered cache.
//
// The wired BestLA mixed kernels (`bestla_fusion_attn_forward<float,fp16,...>` /
// `<float,bf16,...>`) consume packed/reordered K/V, not the raw PLAIN
// (HND/NHD-strided) tensors `bestla_sdpa_forward` receives. These helpers build
// the missing bridge: they describe the packed cache geometry and fill it from
// raw K/V so the kernel can be fed NTILE24 (fp16) / NTILE48 (bf16) row-packed
// operands. fp16 K/V map to NTILE24_ROWPACK1, bf16 K/V to NTILE48_ROWPACK2.
// ---------------------------------------------------------------------------

// Per-(NTILE, ROWPACK) packed K/V geometry for a single shape + element type.
struct ReorderKVShape {
  ATTN_FWD_LAYOUT layout = ATTN_FWD_LAYOUT_PLAIN;  // NTILE24/NTILE48 row-pack
  int ntile = 0;                                   // 24 (fp16) or 48 (bf16)
  int rowpack = 0;                                 // 1 (fp16) or 2 (bf16)
  int sl_pad = 0;                                  // seq padded to NTILE
  int hs_pad = 0;                                  // head_size padded to rowpack
  // Per-head element counts (one head = one [B,Hkv] slice).
  size_t k_head_elems = 0;  // packed K bytes/elems per head ([hs_pad][sl_pad])
  size_t v_head_elems = 0;  // packed V bytes/elems per head ([sl_pad][hs_pad])
  int num_heads = 0;        // batch * heads_kv
  // Step strides (in elements) for the resulting packed attn_fwd_args_t.
  int step_k_sl = 0;
  int step_k_head_size = 0;
  int step_v_sl = 0;
  int step_v_head_size = 0;
};

// Compute the packed K/V layout/strides/sizes for the given shape + K/V dtype.
ReorderKVShape reorder_kv_shape(int batch, int num_heads_kv, int seq_len_kv, int head_dim, BTLA_DTYPE kv_dtype);

// Total packed K (or V) cache elements across all heads.
size_t reorder_kv_cache_elems(const ReorderKVShape& shape, bool is_value);

// Reorder raw HND/NHD K -> NTILE row-packed K cache. `src` is the raw K of one
// batch with the provided strides; `dst` is the packed cache (>= K cache size).
void reorder_k_to_packed(void* dst, const void* src, const ReorderKVShape& shape, const AttentionStrides& k_strides,
                         int batch, int num_heads_kv, int seq_len_kv, int head_dim, BTLA_DTYPE kv_dtype);

// Reorder raw HND/NHD V -> NTILE row-packed V cache (NTILE over head_size).
void reorder_v_to_packed(void* dst, const void* src, const ReorderKVShape& shape, const ValueStrides& v_strides,
                         int batch, int num_heads_kv, int seq_len_kv, int head_dim, BTLA_DTYPE kv_dtype);

void kv_cache_update(void* cache_k, void* cache_v, const void* key, const void* value, const AttentionStrides& k_strides,
                     const ValueStrides& v_strides, BTLA_DTYPE dtype, int batch, int num_heads_kv, int append_len,
                     int head_dim, int capacity, int start_pos);

}  // namespace ark::cpu
