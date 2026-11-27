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

#include <cstdint>

namespace ark {

void build_sparse_execution_schedule(std::uintptr_t stream, std::uintptr_t lut, std::uintptr_t valid_block_num,
                                     std::uintptr_t execution_schedule, std::uintptr_t execution_schedule_rounds,
                                     int batch, int num_heads, int num_q_blocks, int num_k_blocks, int rows_per_group,
                                     int max_rounds);

}  // namespace ark
