// Global device scratch access for the int4 MoE decode fallbacks.
//
// Keep utils.hpp and its bestla JIT dependencies out of the decode headers.
// DeviceMemoryPool owns all allocations, locking and grow-on-demand reuse;
// no decode-specific state or manual create/release lifecycle is needed.

#include "sycl_tla_moe_decode_scratch.hpp"

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

#include <stdexcept>
#include "utils.hpp"

namespace ark {
namespace moe_decode_detail {

namespace {

// Reuse the shared pool's per-queue allocation and growth path. Do not add a
// host wait here: a blocking queue wait is illegal during XPU graph capture.
void* acquire_slab(sycl::queue* q, size_t bytes, size_t buf_loc) {
  if (q == nullptr) {
    throw std::invalid_argument("moe_gemm_decode: device scratch requires a non-null SYCL queue");
  }
  if (bytes == 0) return nullptr;
  auto* pool = DeviceMemoryPool::Instance();
  void* ptr = pool->get_scratch_mem(bytes, buf_loc, q);
  if (ptr == nullptr) {
    // The pool records the slot before checking the result, so a failed
    // allocation leaves a {bytes, nullptr} entry behind that would satisfy
    // every later request of this size or smaller without ever retrying.
    // Drop it so the next call allocates again.
    pool->detach_scratch_mem(buf_loc, q);
    throw std::runtime_error("moe_gemm_decode: failed to allocate device scratch buffer");
  }
  return ptr;
}

}  // namespace

uint8_t* acquire_int4_repack_scratch(sycl::queue* q, size_t bytes) {
  return static_cast<uint8_t*>(acquire_slab(q, bytes, kInt4RepackScratchLoc));
}

float* acquire_act_group_sum_scratch(sycl::queue* q, size_t bytes) {
  return static_cast<float*>(acquire_slab(q, bytes, kActGroupSumScratchLoc));
}

}  // namespace moe_decode_detail
}  // namespace ark

#endif
