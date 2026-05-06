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
#include <map>
#include <thread>
#include <vector>
#include <sycl/sycl.hpp>

namespace bestla {

namespace sycl_device {

class SyclDevice {
 public:
  SyclDevice(int gpu_idx, bool profile) {
    // Create an exception handler for asynchronous SYCL exceptions
    static auto exception_handler = [](sycl::exception_list e_list) {
      for (std::exception_ptr const& e : e_list) {
        try {
          std::rethrow_exception(e);
        } catch (std::exception const& e) {
#if _DEBUG
          std::cout << "Failure" << std::endl;
#endif
          std::terminate();
        }
      }
    };
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    assert(gpu_idx < devices.size());

    if (profile) {
      sycl::property_list prop = {sycl::property::queue::enable_profiling(), sycl::property::queue::in_order()};
      mQueue = sycl::queue(devices[gpu_idx], exception_handler, prop);
    } else {
      sycl::property_list prop = {sycl::property::queue::in_order()};
      mQueue = sycl::queue(devices[gpu_idx], exception_handler);
    }
  }

  SyclDevice(bool profile) {
    // Create an exception handler for asynchronous SYCL exceptions
    static auto exception_handler = [](sycl::exception_list e_list) {
      for (std::exception_ptr const& e : e_list) {
        try {
          std::rethrow_exception(e);
        } catch (std::exception const& e) {
#if _DEBUG
          std::cout << "Failure" << std::endl;
#endif
          std::terminate();
        }
      }
    };
    auto d_selector{sycl::default_selector_v};
    if (profile) {
      sycl::property_list prop = {sycl::property::queue::enable_profiling(), sycl::property::queue::in_order()};
      mQueue = sycl::queue(d_selector, exception_handler, prop);
    } else {
      sycl::property_list prop = {sycl::property::queue::in_order()};
      mQueue = sycl::queue(d_selector, exception_handler);
    }
  }

  inline sycl::queue* getQueue() { return &mQueue; }

  inline std::string getName() { return mQueue.get_device().get_info<sycl::info::device::name>(); };

  size_t getGlobalMemSize() { return mQueue.get_device().get_info<sycl::info::device::global_mem_size>(); }
  size_t getMaxMemAllocSize() { return mQueue.get_device().get_info<sycl::info::device::max_mem_alloc_size>(); }

  double getGlobalMemSizeGB() { return double(getGlobalMemSize()) / 1e9; }
  double getMaxMemAllocSizeMB() { return double(getGlobalMemSize()) / 1e6; }

  static inline bool is_cpu(const sycl::device& dev) {
    return dev.get_info<sycl::info::device::device_type>() == sycl::info::device_type::cpu;
  }

  static inline bool is_gpu(const sycl::device& dev) {
    return dev.get_info<sycl::info::device::device_type>() == sycl::info::device_type::gpu;
  }

  static inline bool is_cpu(sycl::queue* q) {
    return q->get_device().get_info<sycl::info::device::device_type>() == sycl::info::device_type::cpu;
  }

  static inline bool is_gpu(sycl::queue* q) {
    return q->get_device().get_info<sycl::info::device::device_type>() == sycl::info::device_type::gpu;
  }

  void print() {
    std::cout << "Running on device: " << mQueue.get_device().get_info<sycl::info::device::name>() << "\n";
    if (is_gpu(mQueue.get_device())) {
      std::cout << "EU count:" << mQueue.get_device().get_info<sycl::ext::intel::info::device::gpu_eu_count>() << "\n";
      std::cout << "EU count per subslice:"
                << mQueue.get_device().get_info<sycl::ext::intel::info::device::gpu_eu_count_per_subslice>() << "\n";
      std::cout << "EU SIMD width:" << mQueue.get_device().get_info<sycl::ext::intel::info::device::gpu_eu_simd_width>()
                << "\n";
      std::cout << "HW threads per EU:"
                << mQueue.get_device().get_info<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>() << "\n";
      std::cout << "GPU slices:" << mQueue.get_device().get_info<sycl::ext::intel::info::device::gpu_slices>() << "\n";
      std::cout << "Subslice per slice:"
                << mQueue.get_device().get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>() << "\n";
    }
    std::cout << "Global Memory size: " << getGlobalMemSizeGB() << "\n";
    std::cout << "Global Memory size: " << getMaxMemAllocSize() << "\n";
  }
  sycl::queue mQueue;
};

}  // namespace sycl_device
}  // namespace bestla
