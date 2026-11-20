#pragma once

#include <cstddef>
#include <cstdint>
#include "sycl_tla_common.hpp"

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

namespace ark {
namespace moe_decode_detail {

// ----------------------------------------------------------------------------
// Device scratch for the int4 decode fallbacks.
//
// Both buffers are owned by the extension-wide `DeviceMemoryPool`, keyed by
// device UUID, SYCL context and queue identity rather than a raw queue pointer.
// The pool reuses each slot and grows it automatically for larger requests.
// There is no decode-specific allocation state or manual create/release API.
//
// Slabs are intentionally never freed from a static destructor: the SYCL
// context may already be torn down at that point. Allocations persist for the
// lifetime of the process, just like the pool's other scratch slots.
//
// Keying includes queue identity, so concurrent streams in one context use
// independent slabs while still reusing within each stream queue.
// The thin accessors live in `sycl_tla_moe_decode_scratch.cpp` to keep utils.hpp
// and its bestla JIT dependencies out of this include chain.
// ----------------------------------------------------------------------------

// `DeviceMemoryPool` slots owned by the decode path. Slots 0-7 belong to the
// dnnl / xpu / sycl-s8 / cpu wrappers and the SDPA kernels, slot 8 to the DPAS
// work-group counter.
inline constexpr size_t kInt4RepackScratchLoc = 9;
inline constexpr size_t kActGroupSumScratchLoc = 10;

// Acquire the N-tiled int4 weight-repack buffer. Contents are regenerated on
// every call so weight updates and allocator address reuse need no invalidation.
// Both accessors return nullptr for zero bytes and reject a null queue.
uint8_t* acquire_int4_repack_scratch(sycl::queue* q, size_t bytes);

// Acquire the per-(token, K-group) activation-sum buffer (asym int4 only).
float* acquire_act_group_sum_scratch(sycl::queue* q, size_t bytes);

}  // namespace moe_decode_detail
}  // namespace ark

#endif
