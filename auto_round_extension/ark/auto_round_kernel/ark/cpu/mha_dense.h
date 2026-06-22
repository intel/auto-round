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
