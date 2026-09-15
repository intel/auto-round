// Device scratch management for the W4A8 MoE path.
//
// The bookkeeping below is deliberately defined here rather than in a header so
// that the module holds exactly one instance of it, and so that `utils.hpp`
// (and the bestla JIT headers behind it) stays out of the cutlass-free W4A8
// translation units. See `sycl_tla_moe_w4a8_scratch.hpp` for the rationale.
//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sycl_tla_moe_w4a8_scratch.hpp"

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "utils.hpp"

namespace ark {
namespace moe_w4a8 {

namespace {

struct W4A8ScratchState {
  std::mutex mu;
  // Device key (`DeviceMemoryPool::get_device_key`) -> a queue handle on that
  // device, held *by value*: a `sycl::queue` is a reference-counted handle, so
  // keeping a copy guarantees the queue outlives the memory allocated against
  // it.
  std::map<size_t, sycl::queue> queues;
};

// Intentionally leaked, see the header.
W4A8ScratchState& w4a8_scratch_state() {
  static W4A8ScratchState* s = new W4A8ScratchState();
  return *s;
}

// Acquire a slab from the shared pool, synchronizing first when the request
// grows it: `DeviceMemoryPool` frees the old pointer in place when it grows a
// slot, and in-flight kernels may still be reading the old slab, so the wait
// has to happen before the call rather than after.
//
// The caller must hold `W4A8ScratchState::mu`.
void* acquire_w4a8_slab(sycl::queue* q, size_t bytes, size_t buf_loc) {
  auto* pool = DeviceMemoryPool::Instance();
  const size_t held = pool->get_scratch_size(buf_loc, q);
  if (held != 0 && held < bytes) {
    q->wait();
  }
  void* ptr = pool->get_scratch_mem(bytes, buf_loc, q);
  if (ptr == nullptr) {
    // The pool records the slot before checking the result, so a failed
    // allocation leaves a {bytes, nullptr} entry behind that would satisfy
    // every later request of this size or smaller without ever retrying.
    // Drop it so the next call allocates again.
    pool->detach_scratch_mem(buf_loc, q);
    throw std::runtime_error("moe_gemm_w4a8: failed to allocate device scratch buffer");
  }
  auto& st = w4a8_scratch_state();
  const size_t key = pool->get_device_key(q);
  if (st.queues.find(key) == st.queues.end()) {
    st.queues.emplace(key, *q);
  }
  return ptr;
}

}  // namespace

uint8_t* acquire_qact_scratch(sycl::queue* q, size_t bytes) {
  if (q == nullptr) {
    throw std::invalid_argument("moe_gemm_w4a8: device scratch requires a non-null SYCL queue");
  }
  if (bytes == 0) return nullptr;
  auto& st = w4a8_scratch_state();
  std::lock_guard<std::mutex> lock(st.mu);
  return static_cast<uint8_t*>(acquire_w4a8_slab(q, bytes, kW4A8QactScratchLoc));
}

int* acquire_expert_map_scratch(sycl::queue* q, size_t bytes) {
  if (q == nullptr) {
    throw std::invalid_argument("moe_gemm_w4a8: device scratch requires a non-null SYCL queue");
  }
  if (bytes == 0) return nullptr;
  auto& st = w4a8_scratch_state();
  std::lock_guard<std::mutex> lock(st.mu);
  return static_cast<int*>(acquire_w4a8_slab(q, bytes, kW4A8ExpertMapScratchLoc));
}

void moe_w4a8_release_scratch() {
  auto& st = w4a8_scratch_state();

  // Detach everything under the lock, then drop the lock before the device sync
  // and the frees: `wait()` blocks for an unbounded time and must not be held
  // across. Because the slabs are already out of the pool's tables, an acquire
  // that races in behind us allocates fresh ones instead of handing back a
  // pointer we are about to free.
  std::vector<std::pair<sycl::queue, void*>> pending;
  {
    std::lock_guard<std::mutex> lock(st.mu);
    auto* pool = DeviceMemoryPool::Instance();
    for (auto& kv : st.queues) {
      sycl::queue q = kv.second;
      for (size_t loc : {kW4A8QactScratchLoc, kW4A8ExpertMapScratchLoc}) {
        void* ptr = pool->detach_scratch_mem(loc, &q);
        if (ptr != nullptr) pending.emplace_back(q, ptr);
      }
    }
    st.queues.clear();
  }

  for (auto& item : pending) {
    item.first.wait();
    sycl::free(item.second, item.first);
  }
}

}  // namespace moe_w4a8
}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
