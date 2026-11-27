// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sycl/sycl_device.h"

namespace bestla {
namespace sycl_ut {

class UT_Device {
 public:
  static bestla::sycl_device::SyclDevice* get() {
    static bestla::sycl_device::SyclDevice Instance(true);
    return &Instance;
  }
};
};  // namespace sycl_ut
}  // namespace bestla
