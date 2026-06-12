// SYCL MoE Weight Dequantization Primitives
//
// Device-side dequantization helpers shared between the MoE *decode* (GEMV)
// kernel in `sycl_tla_moe_decode.hpp` and the MoE *prefill* (mixed-input
// Grouped GEMM) kernel in `sycl_tla_moe_mixed.hpp`. Keeping the primitives
// in one place guarantees that both paths produce bit-identical results for
// the same packed weight bytes, which is what the round-trip parity tests
// (decode vs prefill) rely on.
//
// Currently extracted (PR-A1): the FP8 byte->float decoders and the host-
// side `ARK_FP8_DECODE_USE_LUT` env-var reader. INT2/INT4/INT8 decoders are
// still inlined inside the decode kernel; they will be added here when the
// mixed-input prefill mainloop in PR-A2/PR-A3 starts consuming them.
//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>

#include "bestla/sycl/fp8_lut.h"

#ifdef ARK_XPU
#include <sycl/sycl.hpp>
#endif

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

namespace ark {
namespace moe_dequant {

// ----------------------------------------------------------------------------
// FP8 byte -> float decode.
// Matches IEEE-style layout used by torch.float8_e4m3fn / torch.float8_e5m2:
//   E4M3 (finite-only): 1 sign, 4 exp (bias 7),  3 mantissa; 0x7F/0xFF = NaN.
//   E5M2 (IEEE-like):   1 sign, 5 exp (bias 15), 2 mantissa; exp==31 -> Inf/NaN.
//
// Two equivalent (for finite values) implementations are provided:
//   - `_lut`:  read magnitude from the 128-entry constexpr table in
//              `bestla/sycl/fp8_lut.h`, apply sign separately.
//   - `_bits`: fully self-contained inline bit-manipulation, no LUT/SLM.
//
// Selection happens at kernel launch time via a `bool UseLut` template
// parameter sourced from the env var `ARK_FP8_DECODE_USE_LUT` (read once on
// the host by `fp8_decode_use_lut()` below). This keeps the per-element hot
// path branch-free.
// ----------------------------------------------------------------------------

inline float decode_fp8_e4m3_lut(uint8_t byte) {
  const uint32_t mag = byte & 0x7Fu;
  const float v = bestla::sycl_prologue_b::fp8_lut::lut_e4m3_128[mag];
  return (byte & 0x80u) ? -v : v;
}

inline float decode_fp8_e5m2_lut(uint8_t byte) {
  const uint32_t mag = byte & 0x7Fu;
  const float v = bestla::sycl_prologue_b::fp8_lut::lut_e5m2_128[mag];
  return (byte & 0x80u) ? -v : v;
}

inline float decode_fp8_e4m3_bits(uint8_t byte) {
  const uint32_t mag = byte & 0x7Fu;
  const uint32_t sign = byte >> 7;
  float v;
  if (mag == 0u) {
    v = 0.0f;
  } else if (mag == 0x7Fu) {
    v = sycl::nan(0u);
  } else {
    const int exp = static_cast<int>((mag >> 3) & 0xFu);
    const int man = static_cast<int>(mag & 0x7u);
    if (exp == 0) {
      // subnormal: value = man * 2^(1 - bias - mbits) = man / 512
      v = static_cast<float>(man) * (1.0f / 512.0f);
    } else {
      // normal: (1 + man/8) * 2^(exp - bias), bias = 7
      v = (1.0f + static_cast<float>(man) * 0.125f) * sycl::ldexp(1.0f, exp - 7);
    }
  }
  return sign ? -v : v;
}

inline float decode_fp8_e5m2_bits(uint8_t byte) {
  const uint32_t mag = byte & 0x7Fu;
  const uint32_t sign = byte >> 7;
  const int exp = static_cast<int>((mag >> 2) & 0x1Fu);
  const int man = static_cast<int>(mag & 0x3u);
  float v;
  if (exp == 0) {
    // subnormal (incl. zero): value = man * 2^(1 - bias - mbits) = man / 65536
    v = static_cast<float>(man) * (1.0f / 65536.0f);
  } else if (exp == 31) {
    v = (man == 0) ? std::numeric_limits<float>::infinity() : sycl::nan(0u);
  } else {
    // normal: (1 + man/4) * 2^(exp - bias), bias = 15
    v = (1.0f + static_cast<float>(man) * 0.25f) * sycl::ldexp(1.0f, exp - 15);
  }
  return sign ? -v : v;
}

// Compile-time dispatch helper. Both branches are resolved via `if constexpr`,
// so there is no per-element runtime cost regardless of which path is chosen.
template <bool IsE4M3, bool UseLut>
inline float decode_fp8(uint8_t byte) {
  if constexpr (UseLut) {
    if constexpr (IsE4M3) {
      return decode_fp8_e4m3_lut(byte);
    } else {
      return decode_fp8_e5m2_lut(byte);
    }
  } else {
    if constexpr (IsE4M3) {
      return decode_fp8_e4m3_bits(byte);
    } else {
      return decode_fp8_e5m2_bits(byte);
    }
  }
}

// ----------------------------------------------------------------------------
// Host-side env-var reader: cached, defaults to LUT enabled.
//
// `ARK_FP8_DECODE_USE_LUT`:
//   - unset / "1" / "true" / "on" / "yes" (case-insensitive) -> LUT path (default)
//   - "0" / "false" / "off" / "no"        (case-insensitive) -> inline bit-manip
//
// Read once on first call and cached in a function-local static, so it is
// safe (and free) to call this on every launch.
// ----------------------------------------------------------------------------
inline bool fp8_decode_use_lut() {
  static const bool value = []() {
    const char* env = std::getenv("ARK_FP8_DECODE_USE_LUT");
    if (env == nullptr) return true;  // default: LUT on
    std::string s(env);
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (s == "0" || s == "false" || s == "off" || s == "no") return false;
    return true;
  }();
  return value;
}

}  // namespace moe_dequant
}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
