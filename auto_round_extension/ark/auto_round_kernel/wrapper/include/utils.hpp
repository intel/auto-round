//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#pragma once

#include <array>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#if ARK_DNNL
#include <dnnl.hpp>
#endif

#include "bestla/bestla_wrapper.h"

#if ARK_XPU
#include "bestla/sycl/sycl_wrapper.h"
#else
namespace sycl {
typedef void queue;
}
#endif

#if ARK_XPU && ARK_DNNL
#include <dnnl_sycl.hpp>
#endif

#define LOG_LINE() printf("%s:L%d\n", __FUNCTION__, __LINE__);

namespace ark {

struct env_params {
  int verbose = 2;  // trace 0/ debug 1/ info 2/ warn 3/ error 4/ critical 5/ off 6
  int auto_s8 = -1;
  int sage_use_mean_bias = 1;
  int sage_print_kbias = 0;
  int sage_disable_packed_hnd_fast = 0;

  static env_params* Instance() {
    static env_params instance;
    return &instance;
  }

  env_params() {
    env_i("ARK_VERBOSE", verbose);
    env_i("ARK_AUTO_S8", auto_s8);
    env_i("ARK_SAGE_USE_MEAN_BIAS", sage_use_mean_bias);
    env_i("ARK_SAGE_PRINT_KBIAS", sage_print_kbias);
    env_i("ARK_SAGE_DISABLE_PACKED_HND_FAST", sage_disable_packed_hnd_fast);
  }

  static inline void env_i(const char* envstr, int& default_) {
    const char* env_value = std::getenv(envstr);
    if (env_value != nullptr) default_ = std::stoi(env_value);
  }
};

using UUIDArray = std::array<unsigned char, 16>;

struct UUIDHasher {
  size_t operator()(const UUIDArray& uuid) const {
    const uint64_t* p = reinterpret_cast<const uint64_t*>(uuid.data());
    uint64_t h1 = p[0];
    uint64_t h2 = p[1];

    h1 ^= h1 >> 33;
    h1 *= 0xff51afd7ed558ccdLLU;
    h2 ^= h2 >> 33;
    h2 *= 0xc4ceb9fe1a85ec53LLU;

    return static_cast<size_t>(h1 ^ h2);
  }
};

#if ARK_XPU
class XpuScratchOutOfMemory : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};
#endif

class DeviceMemoryPool {
 public:
  static DeviceMemoryPool* Instance() {
    static DeviceMemoryPool instance;
    return &instance;
  }

  size_t get_device_key(sycl::queue* q) {
#if ARK_XPU
    if (q != nullptr) {
      std::lock_guard<std::mutex> lock(key_mutex_);
      auto key = get_device_queue_key_locked(q);
      auto it = device_queue_ids_.find(key);
      if (it != device_queue_ids_.end()) {
        return it->second;
      }
      const size_t new_id = next_device_queue_id_++;
      device_queue_ids_.emplace(key, new_id);
      return new_id;
    }
#endif
    return 0;
  }

  size_t get_device_context_key(sycl::queue* q) {
#if ARK_XPU
    if (q != nullptr) {
      std::lock_guard<std::mutex> lock(key_mutex_);
      auto key = get_device_context_key_locked(q);
      auto it = device_context_ids_.find(key);
      if (it != device_context_ids_.end()) {
        return it->second;
      }
      const size_t new_id = next_device_context_id_++;
      device_context_ids_.emplace(key, new_id);
      return new_id;
    }
#endif
    return 0;
  }

  void* get_scratch_mem(size_t size, size_t buf_loc, sycl::queue* q) {
    auto key = get_device_key(q);
    return get_scratch_ptr(size, buf_loc, q, key);
  }

  // Current size of the slab held for ``buf_loc`` on ``q``'s device+context+queue, or 0 when
  // no slab is held.
  size_t get_scratch_size(size_t buf_loc, sycl::queue* q) {
    if (buf_loc >= MaxLocNum) return 0;
    auto key = get_device_key(q);
    std::lock_guard<std::mutex> lock(scratch_mutex_);
    auto it = scratch_buffers_[buf_loc].find(key);
    if (it == scratch_buffers_[buf_loc].end()) return 0;
#if ARK_XPU
    if (q != nullptr && is_recording(*q) && it->second.captured.ptr != nullptr) {
      return it->second.captured.size;
    }
#endif
    return it->second.eager.size;
  }

  // Detach the slab held for ``buf_loc`` on ``q``'s device+context+queue and hand ownership to
  // the caller, which becomes responsible for synchronizing and freeing it.
  // Only eager scratch is detached; captured scratch must remain valid for replay.
  //
  // Detaching rather than freeing in place lets a caller drop the entry while
  // holding its own lock and then perform the (unbounded) `wait()` and
  // `sycl::free()` outside that lock: a concurrent acquire that races in after
  // the detach simply allocates a fresh slab instead of racing on this one.
  void* detach_scratch_mem(size_t buf_loc, sycl::queue* q) {
    if (buf_loc >= MaxLocNum) return nullptr;
    auto key = get_device_key(q);
    std::lock_guard<std::mutex> lock(scratch_mutex_);
    auto& buffers = scratch_buffers_[buf_loc];
    auto it = buffers.find(key);
    if (it == buffers.end()) return nullptr;
    auto* ptr = it->second.eager.ptr;
#if ARK_XPU
    if (it->second.captured.ptr != nullptr) {
      it->second.eager = {};
      return ptr;
    }
#endif
    buffers.erase(it);
    return ptr;
  }

  void* get_scratch_ptr(size_t size, size_t buf_loc, sycl::queue* q, size_t key) {
    if (buf_loc >= MaxLocNum) {
      throw std::invalid_argument("DeviceMemoryPool: invalid scratch slot " + std::to_string(buf_loc));
    }
    if (size == 0) return nullptr;
#if ARK_XPU
    if (q == nullptr) {
      throw std::invalid_argument("DeviceMemoryPool: XPU allocation requires a non-null SYCL queue");
    }
#endif
    std::lock_guard<std::mutex> lock(scratch_mutex_);
#if ARK_XPU
    const bool recording = is_recording(*q);
    if (!recording) reclaim_retired_scratch(*q);
#endif
    auto& buffers = scratch_buffers_[buf_loc];
    auto [it, inserted] = buffers.try_emplace(key);
    auto& slot = it->second;
#if ARK_XPU
    if (recording && slot.captured.ptr == nullptr && slot.eager.size >= size) {
      slot.captured = slot.eager;
      slot.eager = {};
    }
    auto& buffer = recording ? slot.captured : slot.eager;
    if (recording && buffer.ptr != nullptr && buffer.size < size) {
      throw std::runtime_error("ARK XPU cannot grow graph-captured scratch slot " + std::to_string(buf_loc) +
                               " from " + std::to_string(buffer.size) + " to " + std::to_string(size) +
                               " bytes; warm up the largest shape on a fresh queue before capture");
    }
#else
    auto& buffer = slot.eager;
#endif
    if (buffer.size >= size) return buffer.ptr;

    int8_t* ptr = nullptr;
    try {
      ptr = allocate(size, buf_loc, q);
#if ARK_XPU
      if (buffer.ptr != nullptr) retire_scratch(buffer.ptr, *q);
#else
      release(buffer.ptr, q);
#endif
    } catch (...) {
      if (inserted) buffers.erase(it);
      release(ptr, q);
      throw;
    }
    buffer = {size, ptr};
    return ptr;
  }

 private:
  // Slots 0-7 are claimed by the dnnl / xpu / sycl-s8 / cpu wrappers and the
  // SDPA kernels; slot 8 is the MoE DPAS grouped-GEMM work-group counter
  // (`moe_dpas_fp8::kAtomicScratchLoc`); slots 9 and 10 are the int4 decode
  // weight-repack and activation-sum buffers
  // (`moe_decode_detail::kInt4RepackScratchLoc` / `kActGroupSumScratchLoc`);
  // slot 11 is the XMX Hadamard staging buffer
  // (`kHadamardStagingScratchLoc` in `ark.cpp`); slot 12 is the SYCL-TLA
  // prefill workspace (`kPrefillScratchLoc` in `sycl_tla_wrapper.hpp`).
  // slots 13 and 14 are the W4A8
  // activation-quantization and expert-map buffers
  // (`moe_w4a8::kW4A8QactScratchLoc` / `kW4A8ExpertMapScratchLoc`).
  static constexpr int MaxLocNum = 15;

  struct ScratchBuffer {
    size_t size = 0;
    int8_t* ptr = nullptr;
  };

  struct ScratchSlot {
    ScratchBuffer eager;
#if ARK_XPU
    ScratchBuffer captured;
#endif
  };

#if ARK_XPU
  struct RetiredScratch {
    int8_t* ptr;
    sycl::queue queue;
    sycl::event last_use;
  };

  static bool is_recording(const sycl::queue& queue) {
#if defined(SYCL_EXT_ONEAPI_GRAPH)
    return queue.ext_oneapi_get_state() == sycl::ext::oneapi::experimental::queue_state::recording;
#else
    return false;
#endif
  }

  void retire_scratch(int8_t* ptr, sycl::queue& queue) {
    // Python entry points hold the GIL from acquisition through submission.
    retired_scratch_.push_back({ptr, queue, queue.ext_oneapi_submit_barrier()});
  }

  void reclaim_retired_scratch(const sycl::queue& queue) {
    for (auto it = retired_scratch_.begin(); it != retired_scratch_.end();) {
      if (it->queue == queue && it->last_use.get_info<sycl::info::event::command_execution_status>() ==
                                    sycl::info::event_command_status::complete) {
        release(it->ptr, &it->queue);
        it = retired_scratch_.erase(it);
      } else {
        ++it;
      }
    }
  }

  struct DeviceContextKey {
    UUIDArray device_uuid;
    size_t context_id;

    bool operator==(const DeviceContextKey& other) const {
      return context_id == other.context_id && device_uuid == other.device_uuid;
    }
  };

  struct DeviceContextKeyHasher {
    size_t operator()(const DeviceContextKey& key) const {
      size_t h = UUIDHasher{}(key.device_uuid);
      h ^= std::hash<size_t>{}(key.context_id) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
      return h;
    }
  };

  struct DeviceQueueKey {
    UUIDArray device_uuid;
    size_t context_id;
    size_t queue_id;

    bool operator==(const DeviceQueueKey& other) const {
      return queue_id == other.queue_id && context_id == other.context_id && device_uuid == other.device_uuid;
    }
  };

  struct DeviceQueueKeyHasher {
    size_t operator()(const DeviceQueueKey& key) const {
      size_t h = UUIDHasher{}(key.device_uuid);
      h ^= std::hash<size_t>{}(key.context_id) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
      h ^= std::hash<size_t>{}(key.queue_id) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
      return h;
    }
  };

  UUIDArray get_device_uuid_locked(sycl::queue* q) {
    return q->get_device().get_info<sycl::ext::intel::info::device::uuid>();
  }

  DeviceContextKey get_device_context_key_locked(sycl::queue* q) {
    auto context_id = get_context_id_locked(q->get_context());
    return {get_device_uuid_locked(q), context_id};
  }

  size_t get_context_id_locked(const sycl::context& ctx) {
    for (const auto& entry : context_ids_) {
      if (entry.context == ctx) {
        return entry.context_id;
      }
    }
    const size_t new_id = next_context_id_++;
    context_ids_.push_back({ctx, new_id});
    return new_id;
  }

  struct ContextEntry {
    sycl::context context;
    size_t context_id;
  };

  size_t get_queue_id_locked(const sycl::queue& q) {
    for (const auto& entry : queue_ids_) {
      if (entry.queue == q) {
        return entry.queue_id;
      }
    }
    const size_t new_id = next_queue_id_++;
    queue_ids_.push_back({q, new_id});
    return new_id;
  }

  DeviceQueueKey get_device_queue_key_locked(sycl::queue* q) {
    auto context_id = get_context_id_locked(q->get_context());
    auto queue_id = get_queue_id_locked(*q);
    return {get_device_uuid_locked(q), context_id, queue_id};
  }

  struct QueueEntry {
    sycl::queue queue;
    size_t queue_id;
  };
#endif

  int8_t* allocate(size_t size, size_t buf_loc, sycl::queue* q) {
#if ARK_XPU
    auto error = [&](const char* cause) {
      return XpuScratchOutOfMemory("ARK XPU scratch allocation failed for " + std::to_string(size) +
                                   " bytes (slot " + std::to_string(buf_loc) + ", device " +
                                   q->get_device().get_info<sycl::info::device::name>() + "): " + cause);
    };
    int8_t* ptr;
    try {
      ptr = sycl::aligned_alloc_device<int8_t>(128, size, *q);
    } catch (const sycl::exception& err) {
      if (err.code() != sycl::make_error_code(sycl::errc::memory_allocation)) throw;
      throw error(err.what());
    }
    if (ptr == nullptr) throw error("device allocator returned null");
#else
    auto* ptr = static_cast<int8_t*>(std::malloc(size));
    if (ptr == nullptr) throw std::bad_alloc();
#endif
    return ptr;
  }

  void release(void* ptr, sycl::queue* q) {
    if (ptr == nullptr) return;
#if ARK_XPU
    if (q == nullptr) {
      throw std::invalid_argument("DeviceMemoryPool: XPU free requires a non-null SYCL queue");
    }
    sycl::free(ptr, *q);
#else
    std::free(ptr);
#endif
  }

  std::array<std::unordered_map<size_t, ScratchSlot>, MaxLocNum> scratch_buffers_;
  std::mutex scratch_mutex_;
#if ARK_XPU
  std::vector<RetiredScratch> retired_scratch_;
  std::mutex key_mutex_;
  std::vector<ContextEntry> context_ids_;
  std::vector<QueueEntry> queue_ids_;
  std::unordered_map<DeviceContextKey, size_t, DeviceContextKeyHasher> device_context_ids_;
  std::unordered_map<DeviceQueueKey, size_t, DeviceQueueKeyHasher> device_queue_ids_;
  size_t next_context_id_ = 1;
  size_t next_queue_id_ = 1;
  size_t next_device_context_id_ = 1;
  size_t next_device_queue_id_ = 1;
#endif
};

#if ARK_DNNL

template <typename>
struct always_false : std::false_type {};

template <typename T>
static inline constexpr dnnl::memory::data_type to_dt() {
  if constexpr (std::is_same_v<T, float>) {
    return dnnl::memory::data_type::f32;
  } else if constexpr (std::is_same_v<T, bestla::utils::fp16>) {
    return dnnl::memory::data_type::f16;
  } else if constexpr (std::is_same_v<T, int8_t>) {
    return dnnl::memory::data_type::s8;
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return dnnl::memory::data_type::u8;
  } else if constexpr (std::is_same_v<T, bestla::utils::bf16>) {
    return dnnl::memory::data_type::bf16;
  } else {
    static_assert(sizeof(T) == 0, "unsupported data type for to_dt<T>()");
  }
}

static inline constexpr dnnl::memory::data_type to_dt(BTLA_DTYPE bt) {
  switch (bt) {
    case BTLA_DTYPE::F32:
      return dnnl::memory::data_type::f32;
    case BTLA_DTYPE::F16:
      return dnnl::memory::data_type::f16;
    case BTLA_DTYPE::S8:
      return dnnl::memory::data_type::s8;
    case BTLA_DTYPE::U8:
      return dnnl::memory::data_type::u8;
    case BTLA_DTYPE::BF16:
      return dnnl::memory::data_type::bf16;
    default:
      return dnnl::memory::data_type::f32;
  }
}

class DnnlContext {
 public:
  static DnnlContext* Instance() {
    static DnnlContext instance;
    return &instance;
  }

  dnnl::engine* get_eng(sycl::queue* q) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (q == nullptr) {
      if (!cpu_engine_) {
        cpu_engine_ = std::make_unique<dnnl::engine>(dnnl::engine::kind::cpu, 0);
      }
      return cpu_engine_.get();
    }

#if ARK_XPU
    auto key = get_device_context_key_locked(q);
    auto engine_it = xpu_engines_.find(key);
    if (engine_it != xpu_engines_.end()) {
      return engine_it->second.get();
    }
    auto dev = q->get_device();
    auto ctx = q->get_context();
    auto insert_result =
        xpu_engines_.emplace(key, std::make_unique<dnnl::engine>(dnnl::sycl_interop::make_engine(dev, ctx)));
    return insert_result.first->second.get();
#else
    if (!cpu_engine_) {
      cpu_engine_ = std::make_unique<dnnl::engine>(dnnl::engine::kind::cpu, 0);
    }
    return cpu_engine_.get();
#endif
  }

  dnnl::stream get_stream(sycl::queue* q) {
    auto* eng = get_eng(q);
    if (q == nullptr) {
      return dnnl::stream(*eng);
    }
#if ARK_XPU
    return dnnl::sycl_interop::make_stream(*eng, *q);
#else
    return dnnl::stream(*eng);
#endif
  }

  dnnl::memory get_scratch_mem(dnnl::memory::desc md, sycl::queue* q) {
    auto key = get_dnnl_key(q);
    auto ptr = DeviceMemoryPool::Instance()->get_scratch_ptr(md.get_size(), 0, q, key);
    return dnnl::memory(md, *get_eng(q), ptr);
  }

  void* get_scratch_mem(size_t size, size_t buf_loc, sycl::queue* q) {
    auto key = get_dnnl_key(q);
    return DeviceMemoryPool::Instance()->get_scratch_ptr(size, buf_loc, q, key);
  }

  void* get_scratch_ptr(size_t size, size_t buf_loc, sycl::queue* q, size_t key) {
    if (q == nullptr) {
      return DeviceMemoryPool::Instance()->get_scratch_ptr(size, buf_loc, q, key);
    }
#if ARK_XPU
    std::lock_guard<std::mutex> lock(mutex_);
    auto queue_key = DeviceMemoryPool::Instance()->get_device_key(q);
    auto scratch_key = QueueScratchKey{queue_key, key};
    auto it = dnnl_queue_scratch_ids_.find(scratch_key);
    if (it != dnnl_queue_scratch_ids_.end()) {
      return DeviceMemoryPool::Instance()->get_scratch_ptr(size, buf_loc, q, it->second);
    }
    const size_t new_id = next_dnnl_queue_scratch_id_++;
    dnnl_queue_scratch_ids_.emplace(scratch_key, new_id);
    return DeviceMemoryPool::Instance()->get_scratch_ptr(size, buf_loc, q, new_id);
#else
    return DeviceMemoryPool::Instance()->get_scratch_ptr(size, buf_loc, q, key);
#endif
  }

 private:
  struct DeviceContextKey {
    UUIDArray device_uuid;
    size_t context_id;

    bool operator==(const DeviceContextKey& other) const {
      return context_id == other.context_id && device_uuid == other.device_uuid;
    }
  };

  struct DeviceContextKeyHasher {
    size_t operator()(const DeviceContextKey& key) const {
      size_t h = UUIDHasher{}(key.device_uuid);
      h ^= std::hash<size_t>{}(key.context_id) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
      return h;
    }
  };

  size_t get_dnnl_key(sycl::queue* q) {
    if (q == nullptr) return 0;
#if ARK_XPU
    return DeviceMemoryPool::Instance()->get_device_key(q);
#else
    return 0;
#endif
  }

#if ARK_XPU
  UUIDArray get_device_uuid_locked(sycl::queue* q) {
    return q->get_device().get_info<sycl::ext::intel::info::device::uuid>();
  }

  DeviceContextKey get_device_context_key_locked(sycl::queue* q) {
    auto context_id = get_context_id_locked(q->get_context());
    return {get_device_uuid_locked(q), context_id};
  }

  size_t get_context_id_locked(const sycl::context& ctx) {
    for (const auto& entry : context_ids_) {
      if (entry.context == ctx) {
        return entry.context_id;
      }
    }
    const size_t new_id = next_context_id_++;
    context_ids_.push_back({ctx, new_id});
    return new_id;
  }

  struct ContextEntry {
    sycl::context context;
    size_t context_id;
  };

  struct QueueScratchKey {
    size_t queue_key;
    size_t user_key;

    bool operator==(const QueueScratchKey& other) const {
      return queue_key == other.queue_key && user_key == other.user_key;
    }
  };

  struct QueueScratchKeyHasher {
    size_t operator()(const QueueScratchKey& key) const {
      size_t h = std::hash<size_t>{}(key.queue_key);
      h ^= std::hash<size_t>{}(key.user_key) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
      return h;
    }
  };

  std::vector<ContextEntry> context_ids_;
  std::unordered_map<QueueScratchKey, size_t, QueueScratchKeyHasher> dnnl_queue_scratch_ids_;
  std::unordered_map<DeviceContextKey, std::unique_ptr<dnnl::engine>, DeviceContextKeyHasher> xpu_engines_;
  size_t next_dnnl_queue_scratch_id_ = 1;
  size_t next_context_id_ = 1;
#endif

  std::unique_ptr<dnnl::engine> cpu_engine_;
  std::mutex mutex_;
};

#endif  // ARK_DNNL

struct QuantParam {
  int n;
  int k;
  int blocksize;
  BTLA_DTYPE compute_type;
  BTLA_DTYPE weight_type;
  BTLA_DTYPE scale_type;
  bool asym;

  QuantParam(int _n, int _k, int _blocksize, int _ct, int _wt, int _st, bool _asym)
      : n(_n),
        k(_k),
        blocksize(_blocksize),
        compute_type((BTLA_DTYPE)_ct),
        weight_type((BTLA_DTYPE)_wt),
        scale_type((BTLA_DTYPE)_st),
        asym(_asym) {}

  inline int blks() { return k / blocksize; }
};

}  // namespace ark
