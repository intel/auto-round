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
// These buffers used to come from a `DeviceScratchPool` built as a function
// local static *in this header*, keyed by raw `sycl::queue*`. That had three
// problems: the instance was not guaranteed unique across translation units or
// DSOs; a slab outlived the queue it was allocated against, so a destroyed
// queue left a dangling entry (and a recycled queue address could hand a slab
// to an unrelated queue); and two wrappers around the same device allocated two
// slabs. `release_all()` also called `queue->wait()` while holding its mutex,
// putting an unbounded device sync inside the lock.
//
// Both buffers are therefore served from the extension-wide `DeviceMemoryPool`,
// which keys on the device UUID rather than on a queue pointer, so slab
// identity follows the device and is immune to queue lifetime and address
// reuse. The bookkeeping below lives in `sycl_tla_moe_decode_scratch.cpp`, so
// there is exactly one instance in the module; this header only declares it.
//
// Slabs are intentionally never freed from a static destructor: the SYCL
// context may already be torn down at that point. `release_decode_scratch`
// provides explicit teardown for callers that need it (exposed to Python as
// `moe_decode_release_scratch`).
//
// Keying on the device rather than on the queue is what removes the duplicate
// slabs, but it also means two queues on the same device now share one slab
// instead of getting one each. That matches how every other `DeviceMemoryPool`
// slot already behaves, and the decode path runs on the device's current
// queue, but it does mean these entry points must not be driven concurrently
// from two queues on one device.
// ----------------------------------------------------------------------------

// `DeviceMemoryPool` slots owned by the decode path. Slots 0-7 belong to the
// dnnl / xpu / sycl-s8 / cpu wrappers and the SDPA kernels, slot 8 to the DPAS
// work-group counter.
inline constexpr size_t kInt4RepackScratchLoc = 9;
inline constexpr size_t kActGroupSumScratchLoc = 10;

// Acquire the N-tiled int4 weight-repack buffer.
//
// The slab carries an optional *tag* -- the address of the source weight buffer
// plus a caller-supplied key that folds in everything else the repacked bytes
// depend on (expert count, shape). `tag_hit` reports whether the slab already
// holds the repack for that exact tag, which lets the caller skip regenerating
// it. The tag is only consulted when the caller opts in (see
// `moe_decode_int4_repack_cache_enabled`), because the address half of a tag is
// a pointer identity and a freed-then-reallocated buffer can land on the same
// address.
uint8_t* acquire_int4_repack_scratch(sycl::queue* q, size_t bytes, const void* tag_ptr, size_t tag_key, bool use_tag,
                                     bool* tag_hit);

// Acquire the per-(token, K-group) activation-sum buffer (asym int4 only).
float* acquire_act_group_sum_scratch(sycl::queue* q, size_t bytes);

// Release both slabs for every device they were allocated on, dropping any
// cached repack with them.
void release_decode_scratch();

}  // namespace moe_decode_detail
}  // namespace ark

#endif
