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
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

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
  int profile_woq = 0;
  int profile_woq_interval = 1000;
  int profile_woq_skip_unpack = 0;
  int profile_woq_detail = 0;
  int mem_pool = 0;
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
    env_i("ARK_PROFILE_WOQ", profile_woq);
    env_i("ARK_PROFILE_WOQ_INTERVAL", profile_woq_interval);
    env_i("ARK_PROFILE_WOQ_SKIP_UNPACK", profile_woq_skip_unpack);
    env_i("ARK_PROFILE_WOQ_DETAIL", profile_woq_detail);
    env_i("ARK_MEM_POOL", mem_pool);
    env_i("ARK_SAGE_USE_MEAN_BIAS", sage_use_mean_bias);
    env_i("ARK_SAGE_PRINT_KBIAS", sage_print_kbias);
    env_i("ARK_SAGE_DISABLE_PACKED_HND_FAST", sage_disable_packed_hnd_fast);
  }

  static inline void env_i(const char* envstr, int& default_) {
    const char* log_level_env = std::getenv(envstr);
    if (log_level_env != nullptr) default_ = std::stoi(log_level_env);
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

class DeviceMemoryPool {
 public:
  static inline int64_t profile_now_ns() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               clock::now().time_since_epoch())
        .count();
  }

  static inline void profile_mem_pool_record(const char* event, size_t buf_loc, size_t key, size_t request_size,
                                             size_t old_size, size_t cached_size, int64_t op_ns) {
    static std::atomic<int64_t> requests{0};
    static std::atomic<int64_t> hits{0};
    static std::atomic<int64_t> allocs{0};
    static std::atomic<int64_t> reallocs{0};
    static std::atomic<int64_t> releases{0};
    static std::atomic<int64_t> alloc_total_ns{0};
    static std::atomic<int64_t> cached_total_bytes{0};
    static std::atomic<int64_t> peak_cached_total_bytes{0};

    int64_t request_count = requests.fetch_add(1, std::memory_order_relaxed) + 1;
    int64_t hit_count = hits.load(std::memory_order_relaxed);
    int64_t alloc_count = allocs.load(std::memory_order_relaxed);
    int64_t realloc_count = reallocs.load(std::memory_order_relaxed);
    int64_t release_count = releases.load(std::memory_order_relaxed);
    int64_t total_alloc_ns = alloc_total_ns.load(std::memory_order_relaxed);
    int64_t cached_bytes = cached_total_bytes.load(std::memory_order_relaxed);

    if (std::strcmp(event, "hit") == 0) {
      hit_count = hits.fetch_add(1, std::memory_order_relaxed) + 1;
    } else if (std::strcmp(event, "alloc") == 0) {
      alloc_count = allocs.fetch_add(1, std::memory_order_relaxed) + 1;
      total_alloc_ns = alloc_total_ns.fetch_add(op_ns, std::memory_order_relaxed) + op_ns;
      cached_bytes = cached_total_bytes.fetch_add(static_cast<int64_t>(cached_size), std::memory_order_relaxed) +
                     static_cast<int64_t>(cached_size);
    } else if (std::strcmp(event, "realloc") == 0) {
      realloc_count = reallocs.fetch_add(1, std::memory_order_relaxed) + 1;
      release_count = releases.fetch_add(1, std::memory_order_relaxed) + 1;
      total_alloc_ns = alloc_total_ns.fetch_add(op_ns, std::memory_order_relaxed) + op_ns;
      cached_bytes = cached_total_bytes.fetch_add(static_cast<int64_t>(cached_size) - static_cast<int64_t>(old_size),
                                                  std::memory_order_relaxed) +
                     static_cast<int64_t>(cached_size) - static_cast<int64_t>(old_size);
    }

    int64_t peak = peak_cached_total_bytes.load(std::memory_order_relaxed);
    while (cached_bytes > peak &&
           !peak_cached_total_bytes.compare_exchange_weak(peak, cached_bytes, std::memory_order_relaxed)) {
    }
    peak = peak_cached_total_bytes.load(std::memory_order_relaxed);

    int profile = env_params::Instance()->mem_pool;
    if (profile == 0 || (profile == 1 && std::strcmp(event, "hit") == 0)) return;

    std::fprintf(stderr,
                 "[ARK_MEM_POOL] requests=%ld hits=%ld allocs=%ld reallocs=%ld releases=%ld "
                 "event=%s buf_loc=%zu key=%zu request=%.3fMB old=%.3fMB cached=%.3fMB "
                 "cached_total=%.3fMB peak_cached_total=%.3fMB op=%.3fus alloc_total=%.3fms\n",
                 request_count, hit_count, alloc_count, realloc_count, release_count, event, buf_loc, key,
                 static_cast<double>(request_size) / 1048576.0, static_cast<double>(old_size) / 1048576.0,
                 static_cast<double>(cached_size) / 1048576.0, static_cast<double>(cached_bytes) / 1048576.0,
                 static_cast<double>(peak) / 1048576.0, static_cast<double>(op_ns) / 1.0e3,
                 static_cast<double>(total_alloc_ns) / 1.0e6);
  }

  static DeviceMemoryPool* Instance() {
    static DeviceMemoryPool instance;
    return &instance;
  }

  size_t get_device_key(sycl::queue* q) {
#if ARK_XPU
    if (q != nullptr) {
      auto uuid = q->get_device().get_info<sycl::ext::intel::info::device::uuid>();
      return UUIDHasher{}(uuid);
    }
#endif
    return 0;
  }

  void* get_scratch_mem(size_t size, size_t buf_loc, sycl::queue* q) {
    auto key = get_device_key(q);
    return get_scratch_ptr(size, buf_loc, q, key);
  }

  void* get_scratch_ptr(size_t size, size_t buf_loc, sycl::queue* q, size_t key) {
    if (size == 0 || buf_loc >= MaxLocNum) return nullptr;

    bool profile_mem_pool = env_params::Instance()->mem_pool != 0;
    auto it = dev_mem_ptr_map[buf_loc].find(key);
    if (it == dev_mem_ptr_map[buf_loc].end()) {
      int64_t start_ns = profile_mem_pool ? profile_now_ns() : 0;
      auto newptr = allocate(size, q);
      int64_t op_ns = profile_mem_pool ? profile_now_ns() - start_ns : 0;
      dev_mem_size_map[buf_loc][key] = size;
      dev_mem_ptr_map[buf_loc][key] = newptr;
      if (profile_mem_pool) profile_mem_pool_record("alloc", buf_loc, key, size, 0, size, op_ns);
      return newptr;
    }

    auto old_size = dev_mem_size_map[buf_loc][key];
    if (old_size < size) {
      int64_t start_ns = profile_mem_pool ? profile_now_ns() : 0;
      release(it->second, q);
      auto newptr = allocate(size, q);
      int64_t op_ns = profile_mem_pool ? profile_now_ns() - start_ns : 0;
      dev_mem_size_map[buf_loc][key] = size;
      dev_mem_ptr_map[buf_loc][key] = newptr;
      if (profile_mem_pool) profile_mem_pool_record("realloc", buf_loc, key, size, old_size, size, op_ns);
      return newptr;
    }

    if (profile_mem_pool) profile_mem_pool_record("hit", buf_loc, key, size, old_size, old_size, 0);
    return it->second;
  }

 private:
  static constexpr int MaxLocNum = 8;
  using SizeMap = std::unordered_map<size_t, size_t>;
  using PtrMap = std::unordered_map<size_t, int8_t*>;

  int8_t* allocate(size_t size, sycl::queue* q) {
#if ARK_XPU
    if (q == nullptr) {
      throw std::invalid_argument("DeviceMemoryPool: XPU allocation requires a non-null SYCL queue");
    }
    return sycl::aligned_alloc_device<int8_t>(128, size, *q);
#else
    return static_cast<int8_t*>(std::malloc(size));
#endif
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

  std::array<SizeMap, MaxLocNum> dev_mem_size_map;
  std::array<PtrMap, MaxLocNum> dev_mem_ptr_map;
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
    auto key = check_dnnl_device(q);
    return &dev_engine_map[key];
  }

  dnnl::stream* get_stream(sycl::queue* q) {
    auto key = check_dnnl_device(q);
    return &dev_stream_map[key];
  }

  size_t check_dnnl_device(sycl::queue* q) {
    size_t key = 0;

    if (q == nullptr) {
      if (dev_engine_map.find(key) == dev_engine_map.end()) {
        dev_engine_map[key] = dnnl::engine(dnnl::engine::kind::cpu, 0);
        dev_stream_map[key] = dnnl::stream(dev_engine_map[key]);
      }
      return key;
    }

#if ARK_XPU
    key = DeviceMemoryPool::Instance()->get_device_key(q);
    if (dev_engine_map.find(key) == dev_engine_map.end()) {
      sycl::device dev = q->get_device();
      sycl::context ctx = q->get_context();
      dev_engine_map[key] = dnnl::sycl_interop::make_engine(dev, ctx);
      dev_stream_map[key] = dnnl::sycl_interop::make_stream(dev_engine_map[key], *q);
    }
#else
    if (dev_engine_map.find(key) == dev_engine_map.end()) {
      dev_engine_map[key] = dnnl::engine(dnnl::engine::kind::cpu, 0);
      dev_stream_map[key] = dnnl::stream(dev_engine_map[key]);
    }
#endif

    return key;
  }

  dnnl::memory get_scratch_mem(dnnl::memory::desc md, sycl::queue* q) {
    auto key = check_dnnl_device(q);
    auto ptr = DeviceMemoryPool::Instance()->get_scratch_ptr(md.get_size(), 0, q, key);
    return dnnl::memory(md, dev_engine_map[key], ptr);
  }

  void* get_scratch_mem(size_t size, size_t buf_loc, sycl::queue* q) {
    return DeviceMemoryPool::Instance()->get_scratch_mem(size, buf_loc, q);
  }

  void* get_scratch_ptr(size_t size, size_t buf_loc, sycl::queue* q, size_t key) {
    return DeviceMemoryPool::Instance()->get_scratch_ptr(size, buf_loc, q, key);
  }

 private:
  std::unordered_map<size_t, dnnl::engine> dev_engine_map;
  std::unordered_map<size_t, dnnl::stream> dev_stream_map;
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