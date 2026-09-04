// Device scratch management for the int4 MoE decode fallbacks.
//
// The bookkeeping below is deliberately defined here rather than in a header so
// that the module holds exactly one instance of it, and so that the buffers
// themselves are owned by the extension-wide `DeviceMemoryPool` (keyed by
// device UUID) instead of by a header-local static keyed by raw `sycl::queue*`.
// See `sycl_tla_moe_decode_scratch.hpp` for the rationale.

#include "sycl_tla_moe_decode_scratch.hpp"

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>
#include "utils.hpp"

namespace ark {
namespace moe_decode_detail {

namespace {

// Tag attached to the int4 repack slab: the address of the weight buffer it was
// derived from plus a key folding in everything else its contents depend on.
struct RepackTag {
  bool tagged = false;
  const void* tag_ptr = nullptr;
  size_t tag_key = 0;
};

struct ScratchState {
  std::mutex mu;
  // Device key (`DeviceMemoryPool::get_device_key`) -> a queue handle on that
  // device. The queue is held *by value*: a `sycl::queue` is a reference
  // counted handle, so keeping a copy guarantees the queue outlives the device
  // memory we allocated against it. This is what makes the pool immune to the
  // caller destroying its queue, and to a later queue landing on the same
  // address.
  std::map<size_t, sycl::queue> queues;
  // Device key -> repack tag. The activation-sum slab carries no tag.
  std::map<size_t, RepackTag> repack_tags;
};

// Intentionally leaked: the SYCL context may already be torn down by the time
// static destructors run, so neither the queue handles nor the slabs may be
// released from one. Explicit teardown goes through `release_decode_scratch`.
ScratchState& state() {
  static ScratchState* s = new ScratchState();
  return *s;
}

// Acquire a slab from the shared pool, synchronizing first when the request
// grows it. `DeviceMemoryPool` frees the old pointer in place when it grows a
// slot, and in-flight kernels may still be reading the old slab, so the wait
// has to happen before the call rather than after. Returns the pointer and
// reports whether the slab was reallocated (its previous contents are gone).
//
// The caller must hold `ScratchState::mu`.
void* acquire_slab(sycl::queue* q, size_t bytes, size_t buf_loc, bool* reallocated) {
  auto* pool = DeviceMemoryPool::Instance();
  const size_t held = pool->get_scratch_size(buf_loc, q);
  const bool grows = held < bytes;
  if (held != 0 && grows) {
    q->wait();
  }
  void* ptr = pool->get_scratch_mem(bytes, buf_loc, q);
  if (ptr == nullptr) {
    throw std::runtime_error("moe_gemm_decode: failed to allocate device scratch buffer");
  }
  if (reallocated != nullptr) *reallocated = grows;
  return ptr;
}

// Remember a queue on this device so `release_decode_scratch` can synchronize
// and free without the caller having to hand one back.
//
// The caller must hold `ScratchState::mu`.
void remember_queue(ScratchState& st, size_t key, sycl::queue* q) {
  if (st.queues.find(key) == st.queues.end()) {
    st.queues.emplace(key, *q);
  }
}

}  // namespace

uint8_t* acquire_int4_repack_scratch(sycl::queue* q, size_t bytes, const void* tag_ptr, size_t tag_key, bool use_tag,
                                     bool* tag_hit) {
  if (q == nullptr) {
    throw std::invalid_argument("moe_gemm_decode: device scratch requires a non-null SYCL queue");
  }
  if (bytes == 0) {
    if (tag_hit != nullptr) *tag_hit = false;
    return nullptr;
  }

  ScratchState& st = state();
  std::lock_guard<std::mutex> lock(st.mu);

  bool reallocated = false;
  auto* ptr = static_cast<uint8_t*>(acquire_slab(q, bytes, kInt4RepackScratchLoc, &reallocated));

  const size_t key = DeviceMemoryPool::Instance()->get_device_key(q);
  remember_queue(st, key, q);

  // A reallocated slab holds undefined bytes, so it can never be a cache hit
  // however well the tag matches.
  RepackTag& tag = st.repack_tags[key];
  const bool hit = use_tag && !reallocated && tag.tagged && tag.tag_ptr == tag_ptr && tag.tag_key == tag_key;
  if (tag_hit != nullptr) *tag_hit = hit;
  if (!hit) {
    tag.tagged = use_tag;
    tag.tag_ptr = tag_ptr;
    tag.tag_key = tag_key;
  }
  return ptr;
}

float* acquire_act_group_sum_scratch(sycl::queue* q, size_t bytes) {
  if (q == nullptr) {
    throw std::invalid_argument("moe_gemm_decode: device scratch requires a non-null SYCL queue");
  }
  if (bytes == 0) return nullptr;

  ScratchState& st = state();
  std::lock_guard<std::mutex> lock(st.mu);

  auto* ptr = static_cast<float*>(acquire_slab(q, bytes, kActGroupSumScratchLoc, nullptr));
  remember_queue(st, DeviceMemoryPool::Instance()->get_device_key(q), q);
  return ptr;
}

void release_decode_scratch() {
  ScratchState& st = state();

  // Detach everything under the lock, then drop the lock before the device
  // sync and the frees: `wait()` blocks for an unbounded time and must not be
  // held across. Because the slabs are already out of the pool's tables, an
  // acquire that races in behind us allocates fresh ones instead of handing
  // back a pointer we are about to free.
  std::vector<std::pair<sycl::queue, void*>> pending;
  {
    std::lock_guard<std::mutex> lock(st.mu);
    auto* pool = DeviceMemoryPool::Instance();
    for (auto& kv : st.queues) {
      sycl::queue q = kv.second;
      for (size_t loc : {kInt4RepackScratchLoc, kActGroupSumScratchLoc}) {
        void* ptr = pool->detach_scratch_mem(loc, &q);
        if (ptr != nullptr) pending.emplace_back(q, ptr);
      }
    }
    st.queues.clear();
    st.repack_tags.clear();
  }

  for (auto& item : pending) {
    item.first.wait();
    sycl::free(item.second, item.first);
  }
}

}  // namespace moe_decode_detail
}  // namespace ark

#endif
