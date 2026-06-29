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
  // results. Phase 4 Step 1 builds the bridge below: raw PLAIN K/V are reordered
  // into the NTILE packed cache the kernels require before dispatch.
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

  // Runtime capability gate: the wired weight prologues are ISA-specialized and
  // return BTLA_CODE::NotSupport (silently, behind asserts) on hardware that
  // lacks the needed extension. Detect that up front and raise a clear error
  // naming the dtype/layout/ISA condition instead of relying on assert (which is
  // a no-op in release builds) or producing wrong results:
  //   * F16 K/V  -> NTILE24_ROWPACK1, fp16->fp32 via F16C, needs AVX2.
  //   * BF16 K/V -> NTILE48_ROWPACK2, bf16->fp32,         needs AVX512F.
  {
    auto* cpu = bestla::device::CpuDevice::getInstance();
    if (kv_dtype == BTLA_DTYPE::F16 && !cpu->AVX2()) {
      throw std::runtime_error(
          "ark::cpu::bestla_sdpa_forward: fp16 K/V (NTILE24_ROWPACK1) mixed SDPA requires AVX2; "
          "this CPU/build does not provide it");
    }
    if (kv_dtype == BTLA_DTYPE::BF16 && !cpu->AVX512F()) {
      throw std::runtime_error(
          "ark::cpu::bestla_sdpa_forward: bf16 K/V (NTILE48_ROWPACK2) mixed SDPA requires AVX512F; "
          "this CPU/build does not provide it");
    }
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
    // attn_fwd_args_t::tmp is char* but the kernel reinterprets it as float*; the
    // backing std::vector<float> guarantees the required alignof(float), so the
    // reinterpret_cast only narrows the element type, not the alignment.
    local.tmp = workspace.empty() ? nullptr : reinterpret_cast<char*>(workspace.data());
  }

  // Phase 4 Step 1: bridge raw PLAIN HND/NHD K/V into the Neural-Speed-style
  // NTILE packed/reordered cache the wired mixed kernels require. The kernel's
  // QK weight is K (NTILE over seq, ROWPACK over head_size) and its PV weight is
  // V (NTILE over head_size, ROWPACK over seq). We allocate per-head packed
  // caches, fill them from the strided inputs, then retarget `local` at the
  // packed layouts/strides. Q and dst stay PLAIN. This path is reached only via
  // the internal/debug ARK_UNSAFE_BESTLA_MIXED_SDPA opt-in (see ark.cpp);
  // default Python mixed SDPA stays disabled until correctness is verified, and
  // persistent packed KV cache/update remains future work.
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
  reorder_k_to_packed(packed_k.data(), local.K, rshape, k_in, local.batch_size, local.heads_kv, local.sl_kv,
                      local.head_size, kv_dtype);
  reorder_v_to_packed(packed_v.data(), local.V, rshape, v_in, local.batch_size, local.heads_kv, local.sl_kv,
                      local.head_size, kv_dtype);
  local.K = packed_k.data();
  local.V = packed_v.data();
  local.K_layout = rshape.layout;
  local.V_layout = rshape.layout;
  local.step_k_head_num = static_cast<int>(rshape.k_head_elems);
  local.step_k_bs = static_cast<int>(rshape.k_head_elems) * local.heads_kv;
  local.step_k_sl = rshape.step_k_sl;
  local.step_k_head_size = rshape.step_k_head_size;
  local.step_v_head_num = static_cast<int>(rshape.v_head_elems);
  local.step_v_bs = static_cast<int>(rshape.v_head_elems) * local.heads_kv;
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

namespace {

// Pad helper.
int pad_up(int v, int p) { return ((v + p - 1) / p) * p; }

}  // namespace

void bestla_sdpa_forward_packed(const attn_fwd_args_t& args, const ReorderKVShape& shape, BTLA_DTYPE kv_dtype) {
  if (!args.Q || !args.K || !args.V || !args.dst) {
    throw std::invalid_argument("ark::cpu::bestla_sdpa_forward_packed: Q/K/V/dst pointers must be non-null");
  }
  // Q and dst stay PLAIN; K/V must already be the NTILE-packed cache for kv_dtype.
  if (args.Q_layout != ATTN_FWD_LAYOUT_PLAIN || args.dst_layout != ATTN_FWD_LAYOUT_PLAIN) {
    throw std::invalid_argument("ark::cpu::bestla_sdpa_forward_packed: Q/dst must be ATTN_FWD_LAYOUT_PLAIN");
  }
  if (args.K_layout != shape.layout || args.V_layout != shape.layout) {
    throw std::invalid_argument("ark::cpu::bestla_sdpa_forward_packed: K/V layout must match packed cache shape");
  }
  if ((kv_dtype == BTLA_DTYPE::F16 && shape.layout != ATTN_FWD_LAYOUT_NTILE24_ROWPACK1) ||
      (kv_dtype == BTLA_DTYPE::BF16 && shape.layout != ATTN_FWD_LAYOUT_NTILE48_ROWPACK2)) {
    throw std::invalid_argument("ark::cpu::bestla_sdpa_forward_packed: dtype/layout mismatch for packed cache");
  }
  // sl_kv is the current valid length, never the padded capacity.
  if (args.sl_kv <= 0 || args.sl_kv > shape.logical_capacity) {
    throw std::invalid_argument("ark::cpu::bestla_sdpa_forward_packed: sl_kv must be in (0, logical_capacity]");
  }
  if (args.head_size != shape.head_dim) {
    throw std::invalid_argument("ark::cpu::bestla_sdpa_forward_packed: head_size must match packed cache head_dim");
  }
  constexpr attn_flags_t kUnsupportedFlags = ATTN_FLAG_IS_ALIBI8 | ATTN_FLAG_IS_TANH30 | ATTN_FLAG_PADDING_RIGHT;
  if ((args.attn_flags & kUnsupportedFlags) != 0) {
    throw std::invalid_argument("ark::cpu::bestla_sdpa_forward_packed: alibi, tanh and padding-right are not wired yet");
  }
  {
    auto* cpu = bestla::device::CpuDevice::getInstance();
    if (kv_dtype == BTLA_DTYPE::F16 && !cpu->AVX2()) {
      throw std::runtime_error("ark::cpu::bestla_sdpa_forward_packed: fp16 K/V mixed SDPA requires AVX2");
    }
    if (kv_dtype == BTLA_DTYPE::BF16 && !cpu->AVX512F()) {
      throw std::runtime_error("ark::cpu::bestla_sdpa_forward_packed: bf16 K/V mixed SDPA requires AVX512F");
    }
  }
  if (args.threading == nullptr) {
    throw std::invalid_argument("ark::cpu::bestla_sdpa_forward_packed: threading pool must be provided");
  }
  auto* th = static_cast<bestla::parallel::IThreading*>(args.threading);

  // Retarget the packed K/V strides from the cache shape (no reorder happens
  // here: K/V are already NTILE-packed). Q/dst pointers/strides are untouched.
  attn_fwd_args_t local = args;
  local.step_k_head_num = static_cast<int>(shape.k_head_elems);
  local.step_k_bs = static_cast<int>(shape.k_head_elems) * local.heads_kv;
  local.step_k_sl = shape.step_k_sl;
  local.step_k_head_size = shape.step_k_head_size;
  local.step_v_head_num = static_cast<int>(shape.v_head_elems);
  local.step_v_bs = static_cast<int>(shape.v_head_elems) * local.heads_kv;
  local.step_v_sl = shape.step_v_sl;
  local.step_v_head_size = shape.step_v_head_size;

  std::vector<float> workspace;
  if (local.tmp == nullptr) {
    attn_shape_t ashape{local.batch_size, local.head_num, local.heads_kv, local.head_size, local.sl_q, local.sl_kv};
    const size_t bytes = bestla_attn_workspace_size(ashape, th->num_threads());
    workspace.resize((bytes + sizeof(float) - 1) / sizeof(float));
    local.tmp = workspace.empty() ? nullptr : reinterpret_cast<char*>(workspace.data());
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
      throw std::invalid_argument("ark::cpu::bestla_sdpa_forward_packed: only F16 and BF16 K/V operands are supported");
  }
}

ReorderKVShape reorder_kv_shape(int batch, int num_heads_kv, int seq_len_kv, int head_dim, BTLA_DTYPE kv_dtype) {
  ReorderKVShape s;
  switch (kv_dtype) {
    case BTLA_DTYPE::F16:
      s.layout = ATTN_FWD_LAYOUT_NTILE24_ROWPACK1;
      s.ntile = 24;
      s.rowpack = 1;
      break;
    case BTLA_DTYPE::BF16:
      s.layout = ATTN_FWD_LAYOUT_NTILE48_ROWPACK2;
      s.ntile = 48;
      s.rowpack = 2;
      break;
    default:
      throw std::invalid_argument("ark::cpu::reorder_kv_shape: only F16 and BF16 K/V are supported");
  }
  if (batch <= 0 || num_heads_kv <= 0 || seq_len_kv <= 0 || head_dim <= 0) {
    throw std::invalid_argument("ark::cpu::reorder_kv_shape: invalid dimensions");
  }
  s.sl_pad = pad_up(seq_len_kv, s.ntile);
  s.hs_pad = pad_up(head_dim, s.rowpack);
  s.head_dim = head_dim;
  s.logical_capacity = seq_len_kv;
  s.num_heads = batch * num_heads_kv;
  // K is the QK weight: NTILE blocks over seq, head_size is ROWPACK-packed.
  const int k_sl_pad = pad_up(seq_len_kv, s.ntile);
  const int k_hs_pad = pad_up(head_dim, s.rowpack);
  s.k_head_elems = static_cast<size_t>(k_sl_pad) * static_cast<size_t>(k_hs_pad);
  s.step_k_sl = k_hs_pad;
  s.step_k_head_size = 1;
  // V is the PV weight: NTILE blocks over head_size, seq is ROWPACK-packed.
  const int v_sl_pad = pad_up(seq_len_kv, s.rowpack);
  const int v_hs_pad = pad_up(head_dim, s.ntile);
  s.v_head_elems = static_cast<size_t>(v_sl_pad) * static_cast<size_t>(v_hs_pad);
  s.step_v_sl = 1;
  s.step_v_head_size = v_sl_pad;
  return s;
}

size_t reorder_kv_cache_elems(const ReorderKVShape& shape, bool is_value) {
  const size_t per_head = is_value ? shape.v_head_elems : shape.k_head_elems;
  return per_head * static_cast<size_t>(std::max(0, shape.num_heads));
}

void reorder_k_to_packed(void* dst, const void* src, const ReorderKVShape& shape, const AttentionStrides& k_strides,
                         int batch, int num_heads_kv, int seq_len_kv, int head_dim, BTLA_DTYPE kv_dtype) {
  if (!dst || !src) {
    throw std::invalid_argument("ark::cpu::reorder_k_to_packed: dst/src must be non-null");
  }
  const int ntile = shape.ntile;
  const int rp = shape.rowpack;
  const int sl_pad = pad_up(seq_len_kv, ntile);   // K: NTILE over seq
  const int hs_pad = pad_up(head_dim, rp);        // K: ROWPACK over head_size
  (void)sl_pad;
  // K element (sl, hs) -> tile of NTILE over sl, ROWPACK over head_size.
  //   tile = sl/NTILE, sl_in = sl%NTILE, kp = hs/rp, rp_i = hs%rp
  //   idx = tile*(hs_pad*NTILE) + kp*(NTILE*rp) + sl_in*rp + rp_i
  std::memset(dst, 0, reorder_kv_cache_elems(shape, /*is_value=*/false) * element_size(kv_dtype));
#pragma omp parallel for collapse(2) schedule(static)
  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < num_heads_kv; ++h) {
      const size_t head_base = (static_cast<size_t>(b) * num_heads_kv + h) * shape.k_head_elems;
      for (int s = 0; s < seq_len_kv; ++s) {
        const int tile = s / ntile, sl_in = s % ntile;
        for (int d = 0; d < head_dim; ++d) {
          const float val = load_scalar(src, qko_offset(k_strides, b, h, s, d), kv_dtype);
          const int kp = d / rp, rp_i = d % rp;
          const size_t idx = static_cast<size_t>(tile) * hs_pad * ntile + static_cast<size_t>(kp) * ntile * rp +
                             static_cast<size_t>(sl_in) * rp + rp_i;
          store_scalar(dst, head_base + idx, kv_dtype, val);
        }
      }
    }
  }
}

void reorder_v_to_packed(void* dst, const void* src, const ReorderKVShape& shape, const ValueStrides& v_strides,
                         int batch, int num_heads_kv, int seq_len_kv, int head_dim, BTLA_DTYPE kv_dtype) {
  if (!dst || !src) {
    throw std::invalid_argument("ark::cpu::reorder_v_to_packed: dst/src must be non-null");
  }
  const int ntile = shape.ntile;
  const int rp = shape.rowpack;
  const int sl_pad = pad_up(seq_len_kv, rp);      // V: ROWPACK over seq
  const int hs_pad = pad_up(head_dim, ntile);     // V: NTILE over head_size
  (void)hs_pad;
  // V element (sl, hs) -> tile of NTILE over head_size, ROWPACK over seq.
  //   tile = hs/NTILE, hs_in = hs%NTILE, kp = sl/rp, rp_i = sl%rp
  //   idx = tile*(sl_pad*NTILE) + kp*(NTILE*rp) + hs_in*rp + rp_i
  std::memset(dst, 0, reorder_kv_cache_elems(shape, /*is_value=*/true) * element_size(kv_dtype));
#pragma omp parallel for collapse(2) schedule(static)
  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < num_heads_kv; ++h) {
      const size_t head_base = (static_cast<size_t>(b) * num_heads_kv + h) * shape.v_head_elems;
      for (int s = 0; s < seq_len_kv; ++s) {
        const int kp = s / rp, rp_i = s % rp;
        for (int d = 0; d < head_dim; ++d) {
          const float val = load_scalar(src, value_offset(v_strides, b, h, s, d), kv_dtype);
          const int tile = d / ntile, hs_in = d % ntile;
          const size_t idx = static_cast<size_t>(tile) * sl_pad * ntile + static_cast<size_t>(kp) * ntile * rp +
                             static_cast<size_t>(hs_in) * rp + rp_i;
          store_scalar(dst, head_base + idx, kv_dtype, val);
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

void clear_packed_k_cache(void* cache_k, const ReorderKVShape& shape, BTLA_DTYPE kv_dtype) {
  if (!cache_k) {
    throw std::invalid_argument("ark::cpu::clear_packed_k_cache: cache must be non-null");
  }
  std::memset(cache_k, 0, reorder_kv_cache_elems(shape, /*is_value=*/false) * element_size(kv_dtype));
}

void clear_packed_v_cache(void* cache_v, const ReorderKVShape& shape, BTLA_DTYPE kv_dtype) {
  if (!cache_v) {
    throw std::invalid_argument("ark::cpu::clear_packed_v_cache: cache must be non-null");
  }
  std::memset(cache_v, 0, reorder_kv_cache_elems(shape, /*is_value=*/true) * element_size(kv_dtype));
}

void update_packed_k_cache(void* cache_k, const void* key, const ReorderKVShape& shape,
                           const AttentionStrides& k_strides, int batch, int num_heads_kv, int append_len, int head_dim,
                           int start_pos, BTLA_DTYPE kv_dtype) {
  if (!cache_k || !key) {
    throw std::invalid_argument("ark::cpu::update_packed_k_cache: cache/src must be non-null");
  }
  if (kv_dtype != BTLA_DTYPE::F16 && kv_dtype != BTLA_DTYPE::BF16) {
    throw std::invalid_argument("ark::cpu::update_packed_k_cache: only F16 and BF16 K are supported");
  }
  const int ntile = shape.ntile, rp = shape.rowpack;
  const int hs_pad = pad_up(head_dim, rp);
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
        for (int d = 0; d < hs_pad; ++d) {
          const float val = d < head_dim ? load_scalar(key, qko_offset(k_strides, b, h, s, d), kv_dtype) : 0.0f;
          const int kp = d / rp, rp_i = d % rp;
          const size_t idx = static_cast<size_t>(tile) * hs_pad * ntile + static_cast<size_t>(kp) * ntile * rp +
                             static_cast<size_t>(sl_in) * rp + rp_i;
          store_scalar(cache_k, head_base + idx, kv_dtype, val);
        }
      }
    }
  }
}

void update_packed_v_cache(void* cache_v, const void* value, const ReorderKVShape& shape,
                           const ValueStrides& v_strides, int batch, int num_heads_kv, int append_len, int head_dim,
                           int start_pos, BTLA_DTYPE kv_dtype) {
  if (!cache_v || !value) {
    throw std::invalid_argument("ark::cpu::update_packed_v_cache: cache/src must be non-null");
  }
  if (kv_dtype != BTLA_DTYPE::F16 && kv_dtype != BTLA_DTYPE::BF16) {
    throw std::invalid_argument("ark::cpu::update_packed_v_cache: only F16 and BF16 V are supported");
  }
  const int ntile = shape.ntile, rp = shape.rowpack;
  const int hs_pad = pad_up(head_dim, ntile);
  const int sl_pad = hs_pad == 0 ? 0 : static_cast<int>(shape.v_head_elems / hs_pad);
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
      for (int s = 0; s < append_len; ++s) {
        const int pos = start_pos + s;
        const int kp = pos / rp, rp_i = pos % rp;
        for (int d = 0; d < hs_pad; ++d) {
          const float val = d < head_dim ? load_scalar(value, value_offset(v_strides, b, h, s, d), kv_dtype) : 0.0f;
          const int tile = d / ntile, hs_in = d % ntile;
          const size_t idx = static_cast<size_t>(tile) * sl_pad * ntile + static_cast<size_t>(kp) * ntile * rp +
                             static_cast<size_t>(hs_in) * rp + rp_i;
          store_scalar(cache_v, head_base + idx, kv_dtype, val);
        }
      }
    }
  }
}

}  // namespace ark::cpu
