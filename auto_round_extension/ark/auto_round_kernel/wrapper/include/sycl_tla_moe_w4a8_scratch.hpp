// Device scratch for the W4A8 MoE path.
//
// The activation-quantization buffers (`[total_tokens, K]` int8 +
// `[total_tokens]` fp32) and the decode expert map (`[total_tokens]` int32) are
// recomputed on every call, so they come from persistent slabs instead of a
// hot-path `malloc_device`.
//
// The slabs are owned by the extension-wide `DeviceMemoryPool`, keyed by
// device, context and queue identity rather than a raw `sycl::queue*`.
// Repeated requests reuse the existing allocation; larger requests grow it
// automatically. No W4A8-specific state or explicit create/release is needed.
//
// Like the other global scratch slots, allocations persist for the lifetime
// of the process and are not freed from a static destructor, when the SYCL
// context may already be torn down.
//
// Different queues use separate allocations, matching the pool's other slots.
//
// Only the declarations live here; the pool access is defined in
// `sycl_tla_moe_w4a8_scratch.cpp` so that `utils.hpp` stays out of this include
// chain. `utils.hpp` drags in bestla's AVX512/xbyak JIT headers, which
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
// work-group counter, slots 9-10 to the int4 decode scratch, and slot 11 to
// the XMX Hadamard staging buffer. Slot 12 is the SYCL-TLA prefill workspace.
inline constexpr size_t kW4A8QactScratchLoc = 13;
inline constexpr size_t kW4A8ExpertMapScratchLoc = 14;

// Acquire the quantized-activation slab (int8 activations + per-token scales).
// Returns `nullptr` for a zero-byte request; throws `std::invalid_argument` on
// a null queue and `std::runtime_error` if the pool cannot allocate.
uint8_t* acquire_qact_scratch(sycl::queue* q, size_t bytes);

// Acquire the decode expert map (`[total_tokens]` int32). Same contract.
int* acquire_expert_map_scratch(sycl::queue* q, size_t bytes);

}  // namespace moe_w4a8
}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
