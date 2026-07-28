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
  int auto_s8 = 0;
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
    static_assert(always_false<T>::value, "unsupported dnnl dtype");
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