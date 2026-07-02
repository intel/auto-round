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
// call. This entry validates PLAIN-strided operands and rejects alibi and tanh
// flags; note, however, that the `step_*` stride interface being
// HND/NHD-friendly does NOT mean the wired mixed-precision kernels accept raw
// HND/NHD/PLAIN K/V. Those specializations require packed/reordered
// (NTILE24/NTILE48) K/V; Phase 4 Step 1 added an internal raw->packed reorder so
// the experimental mixed path can feed them. That reorder bridge stays behind
// ARK_UNSAFE_BESTLA_MIXED_SDPA and the default Python mixed SDPA remains disabled
// until correctness is verified. A persistent packed KV cache/update path and an
// internal already-packed forward (bestla_sdpa_forward_packed) now exist
// alongside this temporary bridge; both stay experimental and gated.
//
// Feature support (Phase 5 Step 1 audit + Phase 5 Step 2 padding-right; see the
// matrix in sdpa.cpp for the full per-route classification): both mixed routes
// support causal (sl_q<=sl_kv), GQA (head_num a multiple of heads_kv), prefer_fp32
// (route 2 uses it to pick the AVX512F fp32-score path over AMX-BF16; route 1 is an
// accepted fp32-score no-op) and padding-right (Phase 5 Step 2 forwards n_padding to
// the fp32-score ScaleTrackMax padding_type==2 epilogue and validates the boundary:
// 0 < n_padding <= sl_kv, mutually exclusive with causal), all validated here.
// alibi/tanh remain a plumbing-gap (the fp32-score stable epilogue implements them
// but this entry does not forward their inputs yet) and are rejected up front.
void bestla_sdpa_forward(const attn_fwd_args_t& args, BTLA_DTYPE kv_dtype);

// ---------------------------------------------------------------------------
// Phase 4 Step 1: raw HND/NHD K/V -> Neural-Speed NTILE packed/reordered cache.
//
// The wired BestLA mixed kernels (`bestla_fusion_attn_forward<float,fp16,...>` /
// `<float,bf16,...>`) consume packed/reordered K/V, not the raw PLAIN
// (HND/NHD-strided) tensors `bestla_sdpa_forward` receives. These helpers build
// the bridge: they describe the packed cache geometry and fill it from raw K/V
// so the kernel can be fed NTILE24 (fp16) / NTILE48 (bf16) row-packed operands.
// fp16 K/V map to NTILE24_ROWPACK1, bf16 K/V to NTILE48_ROWPACK2. Phase 4 Step 2
// validates the reorder layout against the prologue read addresses; the path is
// still experimental and gated by ARK_UNSAFE_BESTLA_MIXED_SDPA only.
// ---------------------------------------------------------------------------

// Per-(NTILE, ROWPACK) packed K/V geometry for a single shape + element type.
struct ReorderKVShape {
  ATTN_FWD_LAYOUT layout = ATTN_FWD_LAYOUT_PLAIN;  // NTILE24/NTILE48 row-pack
  int ntile = 0;                                   // 24 (fp16) or 48 (bf16)
  int rowpack = 0;                                 // 1 (fp16) or 2 (bf16)
  int sl_pad = 0;                                  // seq padded to NTILE
  int hs_pad = 0;                                  // head_size padded to rowpack
  int head_dim = 0;                                // logical head_size (unpadded)
  int logical_capacity = 0;  // logical seq capacity (k_head_elems uses padded cap)
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

// ---------------------------------------------------------------------------
// Phase 4 Step 4: persistent packed K/V cache + in-place update path.
//
// The temporary bridge above reorders the whole raw K/V into a packed cache on
// every forward. To move toward a Neural-Speed-style persistent cache, these
// helpers size a packed cache for a fixed `capacity` (>= sequence length) and
// append raw K/V tokens directly into it at [start_pos, start_pos+append_len),
// without re-reordering the prefix. Packed geometry/strides are identical to
// reorder_kv_shape (fp16->NTILE24_ROWPACK1, bf16->NTILE48_ROWPACK2) but the seq
// dim is padded to `capacity`. Still experimental and gated by
// ARK_UNSAFE_BESTLA_MIXED_SDPA; not default-enabled and not yet routed by the
// Python SDPA path. Source raw tensors are read only through stride fields, so
// HND and NHD layouts work with no hard-coded assumptions.
// ---------------------------------------------------------------------------

// Packed cache shape sized for a fixed capacity rather than the current seq.
// k_head_elems / v_head_elems give the packed per-head stride; multiply by
// num_heads for the total K/V cache element count. The returned shape records
// logical_capacity = capacity; update_packed_* reject writes beyond it even when
// the buffer is padded out to a NTILE/ROWPACK multiple, so padded slots stay
// deterministic. Callers must pass zero-filled buffers (or clear_packed_*_cache)
// so padded/unwritten regions read as zero.
ReorderKVShape packed_kv_cache_shape(int batch, int num_heads_kv, int capacity, int head_dim, BTLA_DTYPE kv_dtype);

// Zero a freshly allocated packed K/V cache so padded regions and future tokens
// are deterministic. Buffers hold reorder_kv_cache_elems(shape, is_value) elems.
void clear_packed_k_cache(void* cache_k, const ReorderKVShape& shape, BTLA_DTYPE kv_dtype);
void clear_packed_v_cache(void* cache_v, const ReorderKVShape& shape, BTLA_DTYPE kv_dtype);

// Append raw K tokens -> persistent packed K cache at [start_pos, start_pos+append_len).
void update_packed_k_cache(void* cache_k, const void* key, const ReorderKVShape& shape,
                           const AttentionStrides& k_strides, int batch, int num_heads_kv, int append_len, int head_dim,
                           int start_pos, BTLA_DTYPE kv_dtype);

// Append raw V tokens -> persistent packed V cache at [start_pos, start_pos+append_len).
void update_packed_v_cache(void* cache_v, const void* value, const ReorderKVShape& shape,
                           const ValueStrides& v_strides, int batch, int num_heads_kv, int append_len, int head_dim,
                           int start_pos, BTLA_DTYPE kv_dtype);

// ---------------------------------------------------------------------------
// Phase 4 Step 5: internal forward over an already-packed persistent K/V cache.
//
// bestla_sdpa_forward (above) keeps the temporary per-forward raw->packed
// reorder bridge. This entry instead consumes a cache already filled by
// update_packed_k_cache / update_packed_v_cache: K/V are NTILE24_ROWPACK1 (fp16)
// or NTILE48_ROWPACK2 (bf16), step_k_*/step_v_* come from `shape`, sl_kv is the
// current valid sequence length (<= shape.logical_capacity), and no reorder
// happens inside. Q and dst stay PLAIN. Internal/experimental only: still gated
// by ARK_UNSAFE_BESTLA_MIXED_SDPA, no default Python path, and true e2e
// numerical validation requires a capable CPU extension build (AVX2/AVX512/AMX).
void bestla_sdpa_forward_packed(const attn_fwd_args_t& args, const ReorderKVShape& shape, BTLA_DTYPE kv_dtype);

// ---------------------------------------------------------------------------
// Phase 4.5 Step 5: internal runtime dispatch for the homogeneous attention
// routes (Q == K == V == dst element type) migrated in steps 3-4.
//
// This is DISTINCT from bestla_sdpa_forward above, which drives the *mixed*
// route (fp32 Q/dst + low-precision K/V). Here every operand shares one element
// type, so dispatch follows the same Neural-Speed-style two-layer model the
// wrapper uses:
//   1. First layer -- the full Q/K/V/dst dtype tuple selects the launcher
//      family via the typed `bestla_fusion_attn_forward<T, T, T, T>` overload:
//        * BTLA_DTYPE::F16  -> `<fp16, fp16, fp16, fp16>`, the *stable*
//          `mha_stable_interface_t` over `gemm::HCoreRowNAvx512fp16` (step 4).
//        * BTLA_DTYPE::BF16 -> `<bf16, bf16, bf16, bf16>`, the *non-stable*
//          `mha_interface_t` exp-sum path over `gemm::HCoreRowNAmxbf16` (step 3).
//      These are two different launcher families -- exactly Neural Speed's
//      structure -- NOT collapsed into one "homogeneous" branch.
//   2. Second layer -- ISA/layout/stride conditions select the concrete kernel
//      inside each dtype branch. That selection already lives in the wrapper
//      overload (each checks the ISA its core needs -- AVX512-FP16 for fp16,
//      AMX-BF16 for bf16 -- and its `weight_base_t` / batch-packer prologue
//      handles the K/V layout at runtime). This entry adds the matching runtime
//      capability gate up front so an unsupported CPU/build fails loudly with a
//      clear message instead of relying on release-mode-stripped asserts. Phase
//      4.5 Step 6 additionally promotes each launcher's layout/stride/GQA
//      contract into explicit std::invalid_argument guards (validated per route,
//      not collapsed) so raw PLAIN shape/stride restrictions are checked before
//      any kernel work -- see the contract matrix in sdpa.cpp.
//
// Unlike the mixed route, the homogeneous prologues pack/convert K/V themselves
// (bf16 batch packers, fp16 plain `weight_base_t`), so NO external raw->packed
// reorder bridge is applied here. Q and dst share the operand dtype. Threading
// is caller-supplied through `args.threading`; `args.tmp` is allocated
// internally (float-aligned) when null, as in the other entries.
//
// Internal/experimental only: this is not routed by the public Python C-ABI
// (ark.cpp) yet -- the homogeneous fp16 stable kernel expects a `weight_base_t`
// K/V layout the raw PLAIN [B,H,S,D] Python inputs do not satisfy -- so the
// default user path stays on the scalar reference kernel. True e2e numerical
// validation requires a capable CPU extension build (AVX512-FP16 / AMX-BF16).
//
// Feature support (Phase 5 Step 1 audit; full matrix in sdpa.cpp): route 3 (fp16
// stable) supports causal and GQA (validated); route 4 (bf16 non-stable) supports
// causal but NOT GQA (requires head_num == heads_kv). prefer_fp32 is unsupported
// for BOTH homogeneous routes and rejected per route (route 3's fp16 core is not
// COMP_FP32; route 4's non-stable exp-sum path asserts prefer_fp32 off).
// alibi/tanh/padding-right are rejected up front (plumbing-gap for the fp16 stable
// route's alibi/tanh, unsupported by the bf16 non-stable launcher).
void bestla_sdpa_forward_homogeneous(const attn_fwd_args_t& args, BTLA_DTYPE dtype);

}  // namespace ark::cpu
