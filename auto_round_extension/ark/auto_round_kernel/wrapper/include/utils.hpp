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
#include <cstdlib>
#include <array>
#include <unordered_map>
#include <dnnl.hpp>
#include "bestla/bestla_wrapper.h"
#if ARK_XPU
#include "bestla/sycl/sycl_wrapper.h"
#include <dnnl_sycl.hpp>
#else
namespace sycl {
typedef void queue;
}
#endif

#define LOG_LINE() printf("%s:L%d\n", __FUNCTION__, __LINE__);

namespace ark {

struct env_params {
  int verbose = 2;  // trace 0/ debug 1/ info 2/ warn 3/ error 4/ critical 5/ off 6
  int auto_s8 = 0;
  static env_params* Instance() {
    static env_params instance;
    return &instance;
  }
  env_params() {
    env_i("ARK_VERBOSE", verbose);
    env_i("ARK_AUTO_S8", auto_s8);
  }
  static inline void env_i(const char* envstr, int& default_) {
    const char* log_level_env = getenv(envstr);
    if (log_level_env != nullptr) default_ = std::stoi(log_level_env);
  }
};

using UUIDArray = std::array<unsigned char, 16>;
// 高性能 UUID 哈希器
struct UUIDHasher {
  size_t operator()(const UUIDArray& uuid) const {
    // 将 16 字节视为两个 64 位整数进行处理，性能远高于逐字节遍历
    const uint64_t* p = reinterpret_cast<const uint64_t*>(uuid.data());
    uint64_t h1 = p[0];
    uint64_t h2 = p[1];

    // 使用类似 splitmix64 的混合逻辑
    h1 ^= h1 >> 33;
    h1 *= 0xff51afd7ed558ccdLLU;
    h2 ^= h2 >> 33;
    h2 *= 0xc4ceb9fe1a85ec53LLU;

    return static_cast<size_t>(h1 ^ h2);
  }
};

template <typename T>
static inline constexpr dnnl::memory::data_type to_dt() {
  if constexpr (std::is_same_v<T, float>)
    return dnnl::memory::data_type::f32;
  else if constexpr (std::is_same_v<T, bestla::utils::fp16>)
    return dnnl::memory::data_type::f16;
  else if constexpr (std::is_same_v<T, int8_t>)
    return dnnl::memory::data_type::s8;
  else if constexpr (std::is_same_v<T, uint8_t>)
    return dnnl::memory::data_type::u8;
  else if constexpr (std::is_same_v<T, bestla::utils::bf16>)
    return dnnl::memory::data_type::bf16;
  else
    static_assert(0);
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
    auto uuid = q->get_device().get_info<sycl::ext::intel::info::device::uuid>();
    key = UUIDHasher{}(uuid);
    if (dev_engine_map.find(key) == dev_engine_map.end()) {
      sycl::device dev = q->get_device();
      // Get the context associated with the queue
      sycl::context ctx = q->get_context();
      dev_engine_map[key] = dnnl::sycl_interop::make_engine(dev, ctx);
      dev_stream_map[key] = dnnl::sycl_interop::make_stream(dev_engine_map[key], *q);
    }
#endif
    return key;
  }

  dnnl::memory get_scratch_mem(dnnl::memory::desc _md, sycl::queue* q) {
    auto key = check_dnnl_device(q);
    auto ptr = get_scratch_ptr(_md.get_size(), 0, q, key);
    return dnnl::memory(_md, dev_engine_map[key], ptr);
  }

  void* get_scratch_mem(size_t _size, size_t buf_loc, sycl::queue* q) {
    auto key = check_dnnl_device(q);
    auto ptr = get_scratch_ptr(_size, buf_loc, q, key);
    return ptr;
  }

  void* get_scratch_ptr(size_t _size, size_t buf_loc, sycl::queue* q, size_t key) {
    if (_size <= 0 || buf_loc >= MaxLocNum) return nullptr;

    auto it = dev_mem_ptr_map[buf_loc].find(key);
    if (it == dev_mem_ptr_map[buf_loc].end()) {
      dev_mem_size_map[buf_loc][key] = _size;
#if ARK_XPU
      auto newptr = sycl::aligned_alloc_device<int8_t>(128, _size, *q);
#else
      auto newptr = (int8_t*)malloc(_size);
#endif
      dev_mem_ptr_map[buf_loc][key] = newptr;
      return newptr;
    } else {
      auto pre_size = dev_mem_size_map[buf_loc][key];
      if (pre_size < _size) {
        auto pre_ptr = it->second;
#if ARK_XPU
        sycl::free(pre_ptr, *q);
#else
        free(pre_ptr);
#endif
        dev_mem_size_map[buf_loc][key] = _size;
#if ARK_XPU
        auto newptr = sycl::aligned_alloc_device<int8_t>(128, _size, *q);
#else
        auto newptr = (int8_t*)malloc(_size);
#endif
        dev_mem_ptr_map[buf_loc][key] = newptr;
        return newptr;
      }
    }
    return it->second;
  }
  typedef std::unordered_map<size_t, size_t> msize_t;
  typedef std::unordered_map<size_t, int8_t*> mptr_t;
  static constexpr int MaxLocNum = 8;
  std::array<msize_t, MaxLocNum> dev_mem_size_map;
  std::array<mptr_t, MaxLocNum> dev_mem_ptr_map;
  std::unordered_map<size_t, dnnl::engine> dev_engine_map;
  std::unordered_map<size_t, dnnl::stream> dev_stream_map;
};

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