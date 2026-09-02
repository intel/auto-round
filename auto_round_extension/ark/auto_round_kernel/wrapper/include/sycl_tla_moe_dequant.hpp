// SYCL MoE Weight Dequantization Primitives
//
// Device-side dequantization helpers shared between the MoE *decode* (GEMV)
// kernel in `sycl_tla_moe_decode.hpp` and the MoE *prefill* (mixed-input
// Grouped GEMM) kernel in `sycl_tla_moe_mixed.hpp`. Keeping the primitives
// in one place guarantees that both paths produce bit-identical results for
// the same packed weight bytes, which is what the round-trip parity tests
// (decode vs prefill) rely on.
//
// Currently extracted:
//   - FP8 (E4M3 / E5M2) byte->float decoders + host-side
//     `ARK_FP8_DECODE_USE_LUT` env-var reader (PR-A1).
//   - INT2 / INT4 / INT8 packed-byte decoders (PR-A2): return the raw
//     integer field(s) prior to `(q - zp) * scale`. Both the decode (GEMV)
//     and prefill (mixed-input Grouped GEMM) paths call these directly,
//     guaranteeing bit-identical dequantization for the round-trip parity
//     tests in `test_moe_prefill_accuracy.py` / `test_moe_unified.py`.
//   - INT4 / INT2 packed-word decoders (`decode_int4_octet`,
//     `decode_int2_octet`): thin `#pragma unroll` wrappers over
//     `decode_int4_pair` / `decode_int2_quad` that decode 8 K outputs from
//     one 32-bit (INT4) / 16-bit (INT2) little-endian word. Used by the
//     prefill fast paths in `sycl_tla_moe_mixed.hpp` to amortise packed-byte
//     loads and scale/zero broadcasts across 4×/2× more K per work-item;
//     bit-identical to the scalar decoders by construction, so decode↔prefill
//     parity is preserved without any changes to the GEMV path.
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
// INT4 (S4_CLIP) packed-byte decode.
//
// Packing: two 4-bit values per byte:
//   value at k = 2*i     -> LOW nibble  (bits [3:0])
//   value at k = 2*i + 1 -> HIGH nibble (bits [7:4])
//
// Asym=false (sym): signed nibble in [-8, 7]. Sign extension is performed by
// shifting the nibble into the top 4 bits of an int8 and arithmetic-shifting
// right by 4, which fills the upper bits with the sign bit.
// Asym=true         : unsigned nibble in [0, 15]. Callers subtract the
// per-group zero-point before applying the scale.
//
// Returns the two decoded values as ints in `q_lo` (k=2i) and `q_hi` (k=2i+1).
// The exact same bit-level operations are used by the decode (GEMV) kernel
// in `sycl_tla_moe_decode.hpp` and the prefill (Grouped-GEMM) kernel in
// `sycl_tla_moe_mixed.hpp` so the two paths produce bit-identical results
// for identical packed inputs.
// ----------------------------------------------------------------------------
template <bool Asym>
inline void decode_int4_pair(uint8_t packed, int& q_lo, int& q_hi) {
  if constexpr (Asym) {
    q_lo = static_cast<int>(packed & 0x0Fu);
    q_hi = static_cast<int>((packed >> 4) & 0x0Fu);
  } else {
    q_lo = static_cast<int>(static_cast<int8_t>(packed << 4) >> 4);
    q_hi = static_cast<int>(static_cast<int8_t>(packed & 0xF0u) >> 4);
  }
}

// ----------------------------------------------------------------------------
// INT2 (S2_CLIP) packed-byte decode.
//
// Packing: four 2-bit values per byte, byte = q0 | (q1<<2) | (q2<<4) | (q3<<6).
// Field j (0..3) corresponds to K index 4*i + j, i.e. bits [2j+1 : 2j].
//
// Asym=false (sym): signed 2-bit value in [-2, 1]. Sign extension shifts the
// field into bits [7:6] of an int8 and arithmetic-shifts right by 6.
// Asym=true         : unsigned 2-bit value in [0, 3]. Callers subtract the
// per-group zero-point before applying the scale.
//
// The four decoded values are returned in `q[0..3]` in K-index order.
// ----------------------------------------------------------------------------
template <bool Asym>
inline void decode_int2_quad(uint8_t packed, int q[4]) {
  if constexpr (Asym) {
    q[0] = static_cast<int>(packed & 0x3u);
    q[1] = static_cast<int>((packed >> 2) & 0x3u);
    q[2] = static_cast<int>((packed >> 4) & 0x3u);
    q[3] = static_cast<int>((packed >> 6) & 0x3u);
  } else {
    // Shift the target field into bits [7:6] then arithmetic-shift right by 6.
    // Masking with 0xC0 keeps only the top two bits (equivalent to the direct
    // `int8_t(packed << 6) >> 6` used for field 0, where no other bits can
    // survive an 8-bit truncation of a 6-bit left shift of a uint8).
    q[0] = static_cast<int>(static_cast<int8_t>(packed << 6) >> 6);
    q[1] = static_cast<int>(static_cast<int8_t>((packed << 4) & 0xC0u) >> 6);
    q[2] = static_cast<int>(static_cast<int8_t>((packed << 2) & 0xC0u) >> 6);
    q[3] = static_cast<int>(static_cast<int8_t>(packed & 0xC0u) >> 6);
  }
}

// ----------------------------------------------------------------------------
// INT4 (S4_CLIP) packed-word decode: 8 nibbles from one 32-bit little-endian
// word = 4 consecutive packed bytes.
//
// The word is assembled from bytes b0..b3 as `b0 | (b1<<8) | (b2<<16) |
// (b3<<24)` (i.e. little-endian, which matches the memory layout on all
// supported XPUs). Each byte contributes two K outputs via the existing
// `decode_int4_pair`, so the K-index mapping is:
//   q[0] = byte0 low nibble  (k_base + 0)
//   q[1] = byte0 high nibble (k_base + 1)
//   q[2] = byte1 low nibble  (k_base + 2)
//   q[3] = byte1 high nibble (k_base + 3)
//   ...
//   q[7] = byte3 high nibble (k_base + 7)
//
// The decoder is expressed as a `#pragma unroll` loop over `decode_int4_pair`,
// so it is bit-identical by construction to four scalar decodes of the same
// four bytes. This keeps the parity contract with the decode/GEMV path (which
// only ever calls `decode_int4_pair`) trivially satisfied.
// ----------------------------------------------------------------------------
template <bool Asym>
inline void decode_int4_octet(uint32_t packed, int q[8]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint8_t byte = static_cast<uint8_t>((packed >> (i * 8)) & 0xFFu);
    decode_int4_pair<Asym>(byte, q[2 * i], q[2 * i + 1]);
  }
}

// ----------------------------------------------------------------------------
// INT2 (S2_CLIP) packed-word decode: 8 fields from one 16-bit little-endian
// word = 2 consecutive packed bytes.
//
// Word assembly: `b0 | (b1<<8)`. Each byte contributes four K outputs via
// `decode_int2_quad`, so the K-index mapping is:
//   q[0..3] = byte0 fields 0..3 (k_base + 0 .. k_base + 3)
//   q[4..7] = byte1 fields 0..3 (k_base + 4 .. k_base + 7)
//
// Same parity-by-construction argument as `decode_int4_octet`: the semantics
// come entirely from the shared `decode_int2_quad` primitive.
// ----------------------------------------------------------------------------
template <bool Asym>
inline void decode_int2_octet(uint16_t packed, int q[8]) {
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const uint8_t byte = static_cast<uint8_t>((packed >> (i * 8)) & 0xFFu);
    decode_int2_quad<Asym>(byte, &q[4 * i]);
  }
}

// ----------------------------------------------------------------------------
// INT8 (S8) single-byte decode.
//
// The storage buffer is `uint8_t` in both sym and asym modes; only the
// interpretation of the byte differs:
//   Asym=false (sym): reinterpret as signed int8 in [-128, 127].
//   Asym=true         : treat as unsigned in [0, 255]; caller subtracts the
//                       per-group zero-point.
// ----------------------------------------------------------------------------
template <bool Asym>
inline int decode_int8(uint8_t raw) {
  if constexpr (Asym) {
    return static_cast<int>(raw);
  } else {
    return static_cast<int>(static_cast<int8_t>(raw));
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
