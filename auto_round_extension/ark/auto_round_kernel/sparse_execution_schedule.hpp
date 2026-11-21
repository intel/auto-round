#pragma once

#include <cstdint>

namespace ark {

void build_sparse_execution_schedule(std::uintptr_t stream, std::uintptr_t lut, std::uintptr_t valid_block_num,
                                     std::uintptr_t execution_schedule, std::uintptr_t execution_schedule_rounds,
                                     int batch, int num_heads, int num_q_blocks, int num_k_blocks, int rows_per_group,
                                     int max_rounds);

}  // namespace ark
