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
// allocated internally for the duration of the call. Only ATTN_FWD_LAYOUT_PLAIN
// operands are accepted; alibi, tanh and padding-right flags are rejected.
void bestla_sdpa_forward(const attn_fwd_args_t& args, BTLA_DTYPE kv_dtype);

void kv_cache_update(void* cache_k, void* cache_v, const void* key, const void* value, const AttentionStrides& k_strides,
                     const ValueStrides& v_strides, BTLA_DTYPE dtype, int batch, int num_heads_kv, int append_len,
                     int head_dim, int capacity, int start_pos);

}  // namespace ark::cpu
