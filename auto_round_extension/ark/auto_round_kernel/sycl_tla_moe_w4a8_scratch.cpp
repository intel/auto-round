// Access the extension-wide device scratch pool for the W4A8 MoE path.
//
// Keep `utils.hpp` (and the bestla JIT headers behind it) out of the
// cutlass-free W4A8 translation units. Allocation, growth and synchronization
// of the pool's bookkeeping are handled by DeviceMemoryPool itself.
//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sycl_tla_moe_w4a8_scratch.hpp"

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

#include <stdexcept>

#include "utils.hpp"

namespace ark {
namespace moe_w4a8 {

namespace {

void* acquire_w4a8_slab(sycl::queue* q, size_t bytes, size_t buf_loc) {
  if (q == nullptr) {
    throw std::invalid_argument("moe_gemm_w4a8: device scratch requires a non-null SYCL queue");
  }
  if (bytes == 0) return nullptr;
  return DeviceMemoryPool::Instance()->get_scratch_mem(bytes, buf_loc, q);
}

}  // namespace

uint8_t* acquire_qact_scratch(sycl::queue* q, size_t bytes) {
  return static_cast<uint8_t*>(acquire_w4a8_slab(q, bytes, kW4A8QactScratchLoc));
}

int* acquire_expert_map_scratch(sycl::queue* q, size_t bytes) {
  return static_cast<int*>(acquire_w4a8_slab(q, bytes, kW4A8ExpertMapScratchLoc));
}

}  // namespace moe_w4a8
}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
