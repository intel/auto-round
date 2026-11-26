// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <new>
#include <string>
#include <vector>
#include "../include/dnnl_wrapper.hpp"
#include "bestla/bestla_utils.h"

using namespace bestla;
using fp16 = utils::fp16;
using bf16 = utils::bf16;

struct Context {
  Context() {
#if ARK_XPU
    q = new sycl::queue(sycl::gpu_selector_v);
#else
    q = nullptr;
#endif
  }
  sycl::queue* q = nullptr;
  static Context* Instance() {
    static Context inst;
    return &inst;
  }

  void* allocate(size_t size) {
#if ARK_XPU
    auto* ptr = sycl::aligned_alloc_device<int8_t>(128, size, *q);
#else
    auto* ptr = malloc(size);
#endif
    if (size != 0 && ptr == nullptr) throw std::bad_alloc();
    return ptr;
  }

  void deallocate(void* ptr) {
    if (!ptr) return;
#ifdef ARK_XPU
    sycl::free(ptr, *q);
#else
    free(ptr);
#endif
  }
};

#define GETQ()                    \
  auto ctx = Context::Instance(); \
  auto q = Context::Instance()->q;