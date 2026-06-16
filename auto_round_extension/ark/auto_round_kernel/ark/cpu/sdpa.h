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

void kv_cache_update(void* cache_k, void* cache_v, const void* key, const void* value, const AttentionStrides& k_strides,
                     const ValueStrides& v_strides, BTLA_DTYPE dtype, int batch, int num_heads_kv, int append_len,
                     int head_dim, int capacity, int start_pos);

}  // namespace ark::cpu
