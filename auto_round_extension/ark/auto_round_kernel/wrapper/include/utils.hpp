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

    auto it = dev_mem_ptr_map[buf_loc].find(key);
    if (it == dev_mem_ptr_map[buf_loc].end()) {
      auto newptr = allocate(size, q);
      dev_mem_size_map[buf_loc][key] = size;
      dev_mem_ptr_map[buf_loc][key] = newptr;
      return newptr;
    }

    auto old_size = dev_mem_size_map[buf_loc][key];
    if (old_size < size) {
      release(it->second, q);
      auto newptr = allocate(size, q);
      dev_mem_size_map[buf_loc][key] = size;
      dev_mem_ptr_map[buf_loc][key] = newptr;
      return newptr;
    }

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
    std::lock_guard<std::mutex> lock(mutex_);

    if (q == nullptr) {
      if (!cpu_engine_) {
        cpu_engine_ = std::make_unique<dnnl::engine>(dnnl::engine::kind::cpu, 0);
      }
      return cpu_engine_.get();
    }

#if ARK_XPU
    auto [dev_key, ctx_id] = get_device_context_ids_locked(q);
    auto engine_it = xpu_engines_.find(EngineKey{dev_key, ctx_id});
    if (engine_it != xpu_engines_.end()) {
      return engine_it->second.get();
    }
    auto dev = q->get_device();
    auto ctx = q->get_context();
    auto insert_result = xpu_engines_.emplace(
        EngineKey{dev_key, ctx_id}, std::make_unique<dnnl::engine>(dnnl::sycl_interop::make_engine(dev, ctx)));
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
    return DeviceMemoryPool::Instance()->get_scratch_ptr(size, buf_loc, q, key);
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
    std::lock_guard<std::mutex> lock(mutex_);
    auto key = get_device_context_key_locked(q);
    auto it = dnnl_scratch_domain_ids_.find(key);
    if (it != dnnl_scratch_domain_ids_.end()) {
      return it->second;
    }
    const size_t new_id = next_dnnl_scratch_domain_id_++;
    dnnl_scratch_domain_ids_.emplace(key, new_id);
    return new_id;
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

  std::pair<size_t, size_t> get_device_context_ids_locked(sycl::queue* q) {
    auto key = get_device_context_key_locked(q);
    return {UUIDHasher{}(key.device_uuid), key.context_id};
  }

  struct ContextEntry {
    sycl::context context;
    size_t context_id;
  };

  struct EngineKey {
    size_t device_key;
    size_t context_id;

    bool operator==(const EngineKey& other) const { return device_key == other.device_key && context_id == other.context_id; }
  };

  std::vector<ContextEntry> context_ids_;
  struct EngineKeyHash {
    size_t operator()(const EngineKey& key) const {
      return key.device_key ^ (key.context_id + 0x9e3779b97f4a7c15ULL + (key.device_key << 6) + (key.device_key >> 2));
    }
  };

  std::unordered_map<DeviceContextKey, size_t, DeviceContextKeyHasher> dnnl_scratch_domain_ids_;
  std::unordered_map<EngineKey, std::unique_ptr<dnnl::engine>, EngineKeyHash> xpu_engines_;
  size_t next_dnnl_scratch_domain_id_ = 1;
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