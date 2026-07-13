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

// BestLA mixed-precision attention: float32 Q/dst, fp16 or bf16 K/V.
//
// Route 1 (kv_dtype == F16):  f32,f16,f16,f32  — stable fp32-score, AVX2.
// Route 2 (kv_dtype == BF16): f32,bf16,bf16,f32 — stable fp32-score, AVX512F or AMX-BF16.
//
// Exposure: TIER 1 (enabled by default). The scalar Tier-0 fallback handles
// homogeneous dtypes; this entry handles the BestLA mixed-precision route.
//
// Feature support (both routes, all S): causal, GQA (head_num % heads_kv == 0),
// padding-right (n_padding, mutually exclusive with causal), alibi (ALIBI8 flag),
// tanh (TANH30 flag), prefer_fp32 (route-2 selects AVX512F path; no-op for route 1).
// All features are validated before any kernel work; see the matrix in sdpa.cpp.
//
// K/V are received as raw PLAIN-strided operands and reordered internally into
// the NTILE24 (fp16) or NTILE48 (bf16) packed layout the BestLA kernels require.
// Use bestla_sdpa_forward_packed to skip this per-forward reorder when a
// persistent packed KV cache is available. Threading is caller-supplied through
// args.threading (a `bestla::parallel::IThreading*`); args.tmp is allocated
// internally when null.
void bestla_sdpa_forward(const attn_fwd_args_t& args, BTLA_DTYPE kv_dtype);

// ---------------------------------------------------------------------------
// Packed/reordered K/V layout helpers (NS-parity persistent KV cache path).
//
// The BestLA mixed kernels (routes 1/2) consume K/V in NTILE row-packed
// layout, not the raw PLAIN-strided tensors bestla_sdpa_forward receives.
// These helpers build the bridge:
//
//   * fp16 K/V → NTILE24_ROWPACK1  (K: NTILE=24 over seq, ROWPACK=1 over head_size;
//                                   V: NTILE=24 over head_size, ROWPACK=1 over seq)
//   * bf16 K/V → NTILE48_ROWPACK2  (K: NTILE=48 over seq, ROWPACK=2 over head_size;
//                                   V: NTILE=48 over head_size, ROWPACK=2 over seq)
//
// The persistent cache path (`packed_kv_cache_shape` / `update_packed_k/v_cache`
// / `bestla_sdpa_forward_packed`) is the NS-parity runtime-ready path for
// autoregressive decode: a fixed-capacity packed buffer is allocated once, updated
// token-by-token, and passed directly to `bestla_sdpa_forward_packed` without
// any per-forward reorder overhead. This is the internal/experimental tier of
// routes 1/2.
//
// `reorder_kv_shape` / `reorder_kv_cache_elems` / `reorder_k/v_to_packed` are
// used by bestla_sdpa_forward's internal per-forward bridge (raw→packed on every
// call) and are shared with the persistent path.
// ---------------------------------------------------------------------------

// Runtime-ready descriptor for the packed K/V cache layout selected for a single
// [batch, heads_kv, capacity, head_size, dtype] contract.
struct ReorderKVShape {
  BTLA_DTYPE dtype = BTLA_DTYPE::F16;
  ATTN_FWD_LAYOUT layout = ATTN_FWD_LAYOUT_PLAIN;   // legacy common-layout alias
  ATTN_FWD_LAYOUT k_layout = ATTN_FWD_LAYOUT_PLAIN; // explicit K layout
  ATTN_FWD_LAYOUT v_layout = ATTN_FWD_LAYOUT_PLAIN; // explicit V layout
  int ntile = 0;                                    // 24 (fp16) or 48 (bf16)
  int rowpack = 0;                                  // 1 (fp16) or 2 (bf16)
  int batch_size = 0;
  int heads_kv = 0;
  int head_dim = 0;  // logical head_size (unpadded)
  int logical_capacity = 0;
  int num_heads = 0;  // batch_size * heads_kv
  int k_seq_pad = 0;
  int k_head_size_pad = 0;
  int v_seq_pad = 0;
  int v_head_size_pad = 0;
  size_t elem_bytes = 0;
  // Per-head / total element counts.
  size_t k_head_elems = 0;
  size_t v_head_elems = 0;
  size_t k_total_elems = 0;
  size_t v_total_elems = 0;
  // Total storage in bytes across all batch×head slots.
  size_t k_bytes = 0;
  size_t v_bytes = 0;
  // Step strides (in elements) for the resulting packed attn_fwd_args_t.
  int step_k_bs = 0;
  int step_k_head_num = 0;
  int step_k_sl = 0;
  int step_k_head_size = 0;
  int step_v_bs = 0;
  int step_v_head_num = 0;
  int step_v_sl = 0;
  int step_v_head_size = 0;
};

// Compute the packed K/V layout/strides/sizes for the given shape + K/V dtype.
ReorderKVShape reorder_kv_shape(int batch, int num_heads_kv, int seq_len_kv, int head_dim, BTLA_DTYPE kv_dtype);

// Total packed K (or V) cache elements across all heads.
size_t reorder_kv_cache_elems(const ReorderKVShape& shape, bool is_value);

// Reorder raw HND/NHD K -> NTILE row-packed K cache. `src` is the raw K of one
// batch with the provided strides; `dst` is the packed cache (>= K cache size).
void reorder_k_to_packed(void* dst, const void* src, const ReorderKVShape& shape, const AttentionStrides& k_strides);

// Reorder raw HND/NHD V -> NTILE row-packed V cache (NTILE over head_size).
void reorder_v_to_packed(void* dst, const void* src, const ReorderKVShape& shape, const ValueStrides& v_strides);

void kv_cache_update(void* cache_k, void* cache_v, const void* key, const void* value, const AttentionStrides& k_strides,
                     const ValueStrides& v_strides, BTLA_DTYPE dtype, int batch, int num_heads_kv, int append_len,
                     int head_dim, int capacity, int start_pos);

// ---------------------------------------------------------------------------
// Persistent packed KV cache: allocate a fixed-capacity packed buffer, clear
// it, and update it incrementally.
//
// Packed geometry/strides are identical to reorder_kv_shape (fp16→NTILE24_ROWPACK1,
// bf16→NTILE48_ROWPACK2) but the seq dim is padded to `capacity`.  The
// `logical_capacity` field in the returned shape records the real capacity so
// update helpers reject writes past it even when the buffer is padded to a
// NTILE/ROWPACK multiple.  Callers must pass zero-filled buffers (or call
// `clear_packed_*_cache`) so padded/unwritten regions read as zero.
// ---------------------------------------------------------------------------

// Packed cache shape sized for a fixed capacity rather than the current seq.
// k_head_elems / v_head_elems give the packed per-head stride; multiply by
// num_heads for the total K/V cache element count. The returned shape records
// logical_capacity = capacity; update_packed_* reject writes beyond it even when
// the buffer is padded out to a NTILE/ROWPACK multiple, so padded slots stay
// deterministic. Callers must pass zero-filled buffers (or clear_packed_*_cache)
// so padded/unwritten regions read as zero.
ReorderKVShape packed_kv_cache_shape(int batch, int num_heads_kv, int capacity, int head_dim, BTLA_DTYPE kv_dtype);
ReorderKVShape packed_kv_cache_info(int batch, int num_heads_kv, int capacity, int head_dim, BTLA_DTYPE kv_dtype);

// Zero a freshly allocated packed K/V cache so padded regions and future tokens
// are deterministic. Buffers hold reorder_kv_cache_elems(shape, is_value) elems.
void clear_packed_k_cache(void* cache_k, const ReorderKVShape& shape);
void clear_packed_v_cache(void* cache_v, const ReorderKVShape& shape);

// Append raw K tokens -> persistent packed K cache at [start_pos, start_pos+append_len).
void update_packed_k_cache(void* cache_k, const void* key, const ReorderKVShape& shape,
                           const AttentionStrides& k_strides, int append_len, int start_pos, bool no_zeroing = false);

// Append raw V tokens -> persistent packed V cache at [start_pos, start_pos+append_len).
void update_packed_v_cache(void* cache_v, const void* value, const ReorderKVShape& shape,
                           const ValueStrides& v_strides, int append_len, int start_pos, bool no_zeroing = false);

// Copy a logical K/V window [seq_off, seq_off + seq_size) from one packed cache
// to another cache with the same descriptor. With default zero-padding semantics,
// packed padding/alignment slots touched by the copy are also propagated.
void copy_packed_k_cache(void* dst_cache_k, const void* src_cache_k, const ReorderKVShape& shape, int seq_off,
                         int seq_size, bool no_zeroing = false);
void copy_packed_v_cache(void* dst_cache_v, const void* src_cache_v, const ReorderKVShape& shape, int seq_off,
                         int seq_size, bool no_zeroing = false);

// Shift-RoPE packed K in-place using precomputed fp16 cos/sin coefficients.
// Mirrors Neural Speed's packed-K BF16 path; currently only BF16 / NTILE48_ROWPACK2
// is supported.
void shift_packed_k_cache_rope(void* cache_k, const void* cossin, const ReorderKVShape& shape, int seq_keep);

// ---------------------------------------------------------------------------
// Forward over an already-packed persistent K/V cache (NS-parity decode path).
//
// This entry consumes K/V already packed by update_packed_k/v_cache: K/V are
// NTILE24_ROWPACK1 (fp16) or NTILE48_ROWPACK2 (bf16), step_k_*/step_v_* come
// from `shape`, sl_kv is the current valid sequence length
// (<= shape.logical_capacity), and no reorder happens inside.  Q and dst stay
// PLAIN.
//
// Feature support: same as bestla_sdpa_forward (routes 1/2) — causal, GQA,
// padding-right, alibi (ALIBI8), tanh (TANH30), and prefer_fp32 are all
// validated and forwarded to the fp32-score epilogue.
//
// Exposure: internal/experimental, enabled by default alongside the PLAIN entry.
// This is the intended NS-parity persistent-cache forward.
void bestla_sdpa_forward_packed(const attn_fwd_args_t& args, const ReorderKVShape& shape);

// ---------------------------------------------------------------------------
// Homogeneous attention routes (Tier 2 — internal-only by design).
//
// These two routes complete the non-int8 NS-parity surface for homogeneous
// operand types (Q == K == V == dst element type), matching the two distinct
// launcher families Neural Speed uses for this case:
//
//   Route 3 — fp16 stable:   f16,f16,f16,f16  → mha_stable_interface_t over
//             HCoreRowNAvx512fp16 (ISA: AVX512-FP16).  Supports causal + GQA.
//             K/V must be PLAIN or NTILE24_ROWPACK1; Q/dst must be PLAIN.
//
//   Route 4 — bf16 non-stable: bf16,bf16,bf16,bf16 → mha_interface_t (exp-sum)
//             over HCoreRowNAmxbf16 (ISA: AMX-BF16).  Supports causal only
//             (no GQA); all operands must be PLAIN.
//
// padding-right, alibi, tanh, and prefer_fp32 are U for both routes (fp16-score
// and non-stable exp-sum epilogues do not implement them; they are rejected with
// per-route messages before any kernel work).
//
// Exposure: route 3 is now callable from ark.cpp's runtime selector for eligible
// fp16 PLAIN K/V inputs (with silent fallback to Tier-0 scalar when the
// homogeneous contract is not met). Route 4 is also runtime-selectable now, but
// only as a narrow bf16 optimization backend: ark.cpp tries it for homogeneous
// bf16 requests that already satisfy its no-GQA/all-PLAIN/AMX-BF16 contract and
// otherwise silently falls back to Tier-0 scalar.
void bestla_sdpa_forward_homogeneous(const attn_fwd_args_t& args, BTLA_DTYPE dtype);

// Debug-only: call the raw Route 4 (mha_interface_t + AMX-BF16) kernel directly,
// bypassing the public mitigation that redirects to mha_dense_forward.
// Requires ARK_DEBUG_ROUTE4_NAN=1 to enable NaN instrumentation printouts.
void debug_bestla_sdpa_forward_route4_raw(const attn_fwd_args_t& args);

}  // namespace ark::cpu
