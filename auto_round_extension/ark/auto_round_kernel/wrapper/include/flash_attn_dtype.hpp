// Flash Attention data type codes shared by the SYCL-TLA wrappers.
//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT

// This enum lives in its own dependency-free header so that translation units
// which only dispatch on dtype (for example ark.cpp, which must include only
// declaration-only headers to keep compiler memory down) can use it without
// pulling in the heavy SYCL-TLA kernel headers or the public API declarations
// in sycl_tla_common.hpp.

#pragma once

namespace ark {

/// Flash Attention data type codes (matches Python side)
enum class FlashAttnDtype : int {
  FP16 = 0,
  BF16 = 1,
  FP32 = 2,
  FP8_E4M3 = 3,
  FP8_E5M2 = 4,
};

}  // namespace ark
