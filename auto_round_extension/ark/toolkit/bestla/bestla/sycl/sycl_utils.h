//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
#pragma once
#include "sycl_device.h"
#include "bestla/bestla_utils.h"

namespace bestla {
namespace sycl_utils {

struct sycl_deleter {
  sycl::queue* queue_;
  sycl_deleter(sycl::queue* _q) : queue_(_q) {}
  template <class T>
  void operator()(T* obj) const {
    if (obj) {
      sycl::free(obj, *queue_);
    }
  }
};

template <typename _T>
struct sycl_vector {
  sycl_vector(uint64_t _size = 0, sycl::queue* _q = nullptr) : size_(_size) {
    if (_q && _size) {
      resize(_size, _q);
    }
  }

  void resize(uint64_t _size, sycl::queue* _q) {
    size_ = _size;
    _T* tmp = sycl::malloc_device<_T>(_size, *_q);
    ptr_ = std::shared_ptr<_T>(tmp, sycl_deleter(_q));
  }

  inline uint64_t size() { return size_; }

  inline _T* data() { return ptr_.get(); }

  std::shared_ptr<_T> ptr_;
  uint64_t size_;
};

template <typename T>
__inline__ std::vector<T> sycl2host(const T* syclptr, size_t elecount, sycl::queue* q) {
  std::vector<T> tmp(elecount);
  q->memcpy(tmp.data(), syclptr, elecount * sizeof(T)).wait();
  return tmp;
}

class event_helper {
 public:
  static float elapsed_time(sycl::event& evt) {
    float t = 0.f;
    const auto startKernExecutionTimePoint = evt.get_profiling_info<sycl::info::event_profiling::command_submit>();
    const auto endKernExecutionTimePoint = evt.get_profiling_info<sycl::info::event_profiling::command_end>();
    t = (endKernExecutionTimePoint - startKernExecutionTimePoint) / 1e6;
    return t;
  }

  static float execute_time(sycl::event& evt) {
    float t = 0.f;
    const auto startKernExecutionTimePoint = evt.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto endKernExecutionTimePoint = evt.get_profiling_info<sycl::info::event_profiling::command_end>();
    t = (endKernExecutionTimePoint - startKernExecutionTimePoint) / 1e6;
    return t;
  }
};
template <class GemmCoreT>
class nd_item_helper {
 public:
  const sycl::nd_item<2> it;
  const sycl::sub_group sg;
  nd_item_helper(sycl::nd_item<2>& _it) : it(_it), sg(it.get_sub_group()) {}

  constexpr inline void local_barrier() const { it.barrier(sycl::access::fence_space::local_space); }

  constexpr inline int sg_group_id() const { return sg.get_group_id()[0]; }

  constexpr inline int wg_idx_m() const { return it.get_group(0); }
  constexpr inline int wg_size_m() const { return GemmCoreT::WgM * GemmCoreT::TileM; }
  constexpr inline int wg_g_m() const { return wg_idx_m() * wg_size_m(); }

  constexpr inline int wg_idx_n() const { return it.get_group(1); }
  constexpr inline int wg_size_n() const { return GemmCoreT::WgN * GemmCoreT::TileN; }
  constexpr inline int wg_g_n() const { return wg_idx_n() * wg_size_n(); }

  constexpr inline int sg_idx_m() const { return sg_group_id() / GemmCoreT::SgNStride; }
  constexpr inline int sg_g_m() const { return wg_g_m() + sg_idx_m() * GemmCoreT::TileM; }

  constexpr inline int sg_idx_n() const { return sg_group_id() % GemmCoreT::SgNStride; }
  constexpr inline int sg_g_n() const { return wg_g_n() + sg_idx_n() * GemmCoreT::SgSize * GemmCoreT::TileN; }

  constexpr inline int sg_id() const { return sg.get_local_id()[0]; }
  constexpr inline int item_g_m() const { return sg_g_m(); }
  constexpr inline int item_g_n() const { return sg_g_n() + sg_id() * GemmCoreT::TileN; }
};

}  // namespace sycl_utils
}  // namespace bestla
