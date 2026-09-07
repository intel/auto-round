// Device scratch for the W4A8 MoE path.
//
// The activation-quantization buffers (`[total_tokens, K]` int8 +
// `[total_tokens]` fp32) and the decode expert map (`[total_tokens]` int32) are
// recomputed on every call, so they come from persistent slabs instead of a
// hot-path `malloc_device`.
//
// The slabs are served from the extension-wide `DeviceMemoryPool`, which keys
// on the device UUID rather than on a `sycl::queue*`: a slab therefore follows
// the device and is immune to the caller destroying its queue and to a later
// queue landing on the same address.
//
// Slabs are intentionally never freed from a static destructor -- the SYCL
// context may already be torn down by then. `moe_w4a8_release_scratch` provides
// the explicit teardown (exposed to Python under the same name).
//
// Sharing one slab per device means these entry points must not be driven
// concurrently from two queues on one device, which matches every other
// `DeviceMemoryPool` slot.
//
// Only the declarations live here; the bookkeeping is defined in
// `sycl_tla_moe_w4a8_scratch.cpp` so that the module holds exactly one instance
// of it and -- just as importantly -- so that `utils.hpp` stays out of this
// include chain. `utils.hpp` drags in bestla's AVX512/xbyak JIT headers, which
// would grow every cutlass-free W4A8 translation unit by an order of magnitude
// and undo much of the TU split. This mirrors
// `sycl_tla_moe_decode_scratch.{hpp,cpp}`.
//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef ARK_XPU
#include <sycl/sycl.hpp>
#endif

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

namespace ark {
namespace moe_w4a8 {

// `DeviceMemoryPool` slots owned by the W4A8 path. Slots 0-7 belong to the
// dnnl / xpu / sycl-s8 / cpu wrappers and the SDPA kernels, slot 8 to the DPAS
// work-group counter, and slots 9-10 to the int4 decode scratch.
inline constexpr size_t kW4A8QactScratchLoc = 11;
inline constexpr size_t kW4A8ExpertMapScratchLoc = 12;

// Acquire the quantized-activation slab (int8 activations + per-token scales).
// Returns `nullptr` for a zero-byte request; throws `std::invalid_argument` on
// a null queue and `std::runtime_error` if the pool cannot allocate.
uint8_t* acquire_qact_scratch(sycl::queue* q, size_t bytes);

// Acquire the decode expert map (`[total_tokens]` int32). Same contract.
int* acquire_expert_map_scratch(sycl::queue* q, size_t bytes);

// Release both slabs for every device they were allocated on. Must not overlap
// an acquire on the same device: the acquire entry points hand out a raw
// pointer, so a caller that is between the acquire and its kernel submission
// would have the memory freed underneath it.
void moe_w4a8_release_scratch();

}  // namespace moe_w4a8
}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
