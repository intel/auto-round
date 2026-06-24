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

#include <cstddef>
#include <cstdint>
#include "bestla/bestla.h"

namespace ark::cpu {

// ---------------------------------------------------------------------------
// Neural-Speed-style attention base types (Phase 1 migration).
//
// These mirror the public Neural Speed CPU attention interface
// (neural_speed/core/layers/mha_dense.h) so the BestLA flash-attention wrapper
// can be migrated on top of them in Phase 2. The legacy scalar `MhaDenseArgs`
// path below remains the runtime kernel until the wrapper lands.
// ---------------------------------------------------------------------------

// Memory layout of the Q/K/V/dst operands fed to the attention kernel.
enum ATTN_FWD_LAYOUT {
  // Plain (row-major) layout.
  ATTN_FWD_LAYOUT_PLAIN = 0,

  // Reordered K/V layouts produced by the BestLA KV-cache packing. The step of
  // sl/hs only works on indices which are a multiple of 48/4 (NTILE/ROWPACK) on
  // the corresponding dimensions.
  ATTN_FWD_LAYOUT_NTILE48_ROWPACK4,

  // step of sl/hs only works on indices which are a multiple of 48/2.
  ATTN_FWD_LAYOUT_NTILE48_ROWPACK2,

  // step of sl/hs only works on indices which are a multiple of 24/1.
  ATTN_FWD_LAYOUT_NTILE24_ROWPACK1,
};

// Bit flags controlling the attention forward behaviour. Mirrors Neural Speed's
// `ne_attn_flags_t` bit assignments for the shared flags so reordered KV caches
// stay binary-compatible; `PADDING_RIGHT` is an ARK addition reserved for the
// right-padded batch path.
using attn_flags_t = uint32_t;
enum ATTN_FLAG : attn_flags_t {
  ATTN_FLAG_NONE = 0,
  ATTN_FLAG_IS_CAUSAL = 1u << 0,
  ATTN_FLAG_IS_ALIBI8 = 1u << 1,     // only support alibi with 8 now
  ATTN_FLAG_PREFER_FP32 = 1u << 2,   // prefer FP32 as the compute type in attn
  ATTN_FLAG_IS_TANH30 = 1u << 3,     // only support tanh with 30 now
  ATTN_FLAG_PADDING_RIGHT = 1u << 4, // right-padded variable-length batch
};

// Problem shape shared by the workspace-size query and the forward call.
struct attn_shape_t {
  int batch_size;
  int head_num;
  int heads_kv;
  int head_size;
  int sl_q;
  int sl_kv;
};

// Full argument bundle for a single attention forward call. Field naming follows
// Neural Speed's `attn_*_fwd_args_t` so the wrapper migration stays a close port.
// Pointers are kept type-erased here; Phase 2 introduces the dtype-specialized
// wrappers that interpret them.
struct attn_fwd_args_t {
  void* Q = nullptr;
  void* K = nullptr;
  void* V = nullptr;
  void* dst = nullptr;

  // Per-tensor dequant scales (1.0 for non-quantized operands).
  float Q_sc = 1.0f;
  float K_sc = 1.0f;
  float V_sc = 1.0f;
  float dst_sc = 1.0f;

  // Caller-provided scratch buffer (see attn_workspace_size).
  char* tmp = nullptr;

  // Softmax scale applied to the QK^T scores (typically 1/sqrt(head_size)).
  float QK_scale = 1.0f;

  attn_flags_t attn_flags = ATTN_FLAG_NONE;

  int batch_size = 0;
  int head_num = 0;
  int heads_kv = 0;
  int head_size = 0;
  int sl_q = 0;
  int sl_kv = 0;

  ATTN_FWD_LAYOUT Q_layout = ATTN_FWD_LAYOUT_PLAIN;
  ATTN_FWD_LAYOUT K_layout = ATTN_FWD_LAYOUT_PLAIN;
  ATTN_FWD_LAYOUT V_layout = ATTN_FWD_LAYOUT_PLAIN;
  ATTN_FWD_LAYOUT dst_layout = ATTN_FWD_LAYOUT_PLAIN;

  int step_q_bs = 0;
  int step_q_head_num = 0;
  int step_q_sl = 0;

  int step_k_bs = 0;
  int step_k_head_num = 0;
  int step_k_sl = 0;
  int step_k_head_size = 0;

  int step_v_bs = 0;
  int step_v_head_num = 0;
  int step_v_sl = 0;
  int step_v_head_size = 0;

  int step_dst_bs = 0;
  int step_dst_head_num = 0;
  int step_dst_sl = 0;

  // Number of valid (non-padding) K/V positions when PADDING_RIGHT is set.
  int n_padding = 0;

  // Optional BestLA threading context. Type-erased until Phase 2 wires the
  // BestLA parallel runtime in.
  void* threading = nullptr;
};

// Number of scratch bytes required by attn_fwd_args_t::tmp for the given shape.
size_t attn_workspace_size(const attn_shape_t& shape);

// Default number of K/V positions processed per tile in the flash-attention
// (tiled online softmax) inner loop. Mirrors the blocking used by Neural Speed's
// CPU mha_dense kernel.
constexpr int kDefaultKvBlock = 256;

struct AttentionStrides {
  int seq = 0;
  int dim = 1;
  int head = 0;
  int batch = 0;
};

struct ValueStrides {
  int dim = 1;
  int seq = 0;
  int head = 0;
  int batch = 0;
};

struct MhaDenseArgs {
  const void* query = nullptr;
  const void* key = nullptr;
  const void* value = nullptr;
  void* output = nullptr;
  const float* attn_mask = nullptr;
  AttentionStrides q_strides;
  AttentionStrides k_strides;
  ValueStrides v_strides;
  AttentionStrides o_strides;
  BTLA_DTYPE dtype = BTLA_DTYPE::F32;
  int batch = 0;
  int num_heads_q = 0;
  int num_heads_kv = 0;
  int seq_len_q = 0;
  int seq_len_kv = 0;
  int head_dim = 0;
  float softmax_scale = 1.0f;
  bool is_causal = false;
  // K/V tile size for the online-softmax inner loop. Values <= 0 fall back to
  // kDefaultKvBlock.
  int kv_block_size = 0;
  // Optional pre-allocated scratch buffer (FP32). When non-null it must hold at
  // least mha_dense_workspace_size(args) floats; otherwise the kernel allocates
  // per-thread scratch internally.
  float* workspace = nullptr;
};

// Number of FP32 elements required by the workspace buffer for the given args.
size_t mha_dense_workspace_size(const MhaDenseArgs& args);

void mha_dense_forward(const MhaDenseArgs& args);

}  // namespace ark::cpu
