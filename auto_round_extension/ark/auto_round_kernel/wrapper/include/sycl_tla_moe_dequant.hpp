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
//   - FP8 word-native decoders (`decode_fp8_half_bits`,
//     `decode_fp8_quad_half_bits`, `fp8_word_scale_bias`) + the
//     `Fp8DecodeMode` selector: convert FP8 bytes to fp16 bit patterns with
//     pure 32-bit field moves (no LUT load, no 8-bit ALU), folding E4M3's
//     residual 2^-8 into the per-K-group scale. Used by the decode GEMV.
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
// Word-native FP8 -> half decode (the `Fp8DecodeMode::kWord` path).
//
// Both LUT and inline-bits decoders above cost real work per weight byte: the
// LUT issues a memory load (plus a sign select) and the bit-manip path runs a
// branchy `ldexp` chain. On the decode hot path -- a pure GEMV that streams one
// weight byte per multiply-add -- that dequant cost is the kernel. Neither is
// necessary, because an FP8 byte is already an IEEE-style float and fp16 is a
// *superset* of both FP8 formats: the whole conversion is a bit-field move.
//
//   E5M2 -> fp16: identical sign position, identical 5-bit exponent with the
//                 same bias 15, mantissa just needs 8 more bits ->
//                     h = byte << 8
//                 Exact for every one of the 256 encodings, specials included
//                 (subnormals stay subnormal, exp==31 stays Inf/NaN).
//
//   E4M3 -> fp16: 4-bit exponent, bias 7. Shifting the 7 magnitude bits up by
//                 7 lands the exponent in fp16's exponent field and the 3
//                 mantissa bits in the top of fp16's mantissa, which yields the
//                 correct value scaled by 2^(7-15) == 2^-8; the sign bit has to
//                 move 8 places instead of 7. Both moves collapse into one
//                 add + one shift, because adding the sign bit to itself
//                 carries it exactly one position further:
//                     h = (byte + (byte & 0x80)) << 7
//                 The residual 2^-8 is constant, so callers fold the reciprocal
//                 (`fp8_word_scale_bias<IsE4M3>()` == 256.0f) into the
//                 per-K-group scale, i.e. it costs nothing per element.
//
// Exactness (verified exhaustively over all 256 byte values / all four
// format-mode combinations): E5M2 is bit-exact including Inf/NaN; E4M3 is
// bit-exact for all 254 finite encodings, including subnormals and both zeros.
// The two E4M3 *NaN* encodings (0x7F / 0xFF -- `torch.float8_e4m3fn` has no
// Inf) decode to +-480 instead of NaN, since fp16 has no NaN pattern reachable
// by a pure field move. auto-round FP8 checkpoints are produced by scaling to
// `finfo(float8_e4m3fn).max == 448` and clamping, so those two encodings cannot
// occur; callers that need NaN propagation can select `Fp8DecodeMode::kLut` or
// `kBits` (see `fp8_decode_mode()`).
// ----------------------------------------------------------------------------
template <bool IsE4M3>
inline uint16_t decode_fp8_half_bits(uint32_t byte) {
  if constexpr (IsE4M3) {
    return static_cast<uint16_t>((byte + (byte & 0x80u)) << 7);
  } else {
    return static_cast<uint16_t>(byte << 8);
  }
}

// Constant the caller must fold into the per-group scale to undo the exponent
// re-bias performed by `decode_fp8_half_bits`. Exact power of two, so the fold
// is a pure exponent bump on the fp32 scale (no rounding).
template <bool IsE4M3>
inline constexpr float fp8_word_scale_bias() {
  return IsE4M3 ? 256.0f : 1.0f;
}

// SWAR form: decode the four FP8 bytes of one little-endian 32-bit word into
// two 32-bit words, each packing two fp16 bit patterns (low 16-bit lane holds
// the lower K index). Bit-identical to calling `decode_fp8_half_bits` on each
// byte, but the whole quad costs a handful of native DWORD ops and -- crucially
// on Xe, whose ALU lanes are 32-bit -- never touches an 8-bit-typed vector,
// which IGC has to expand into narrow-type regioning. This mirrors what
// `decode_int4_octet` does for packed nibbles.
template <bool IsE4M3>
inline void decode_fp8_quad_half_bits(uint32_t word, uint32_t& lo2, uint32_t& hi2) {
  // Spread bytes 0/1 and 2/3 into the two 16-bit lanes of `lo` / `hi`.
  const uint32_t lo = (word & 0x000000FFu) | ((word & 0x0000FF00u) << 8);
  const uint32_t hi = ((word >> 16) & 0x000000FFu) | ((word >> 8) & 0x00FF0000u);
  if constexpr (IsE4M3) {
    // Per-lane `(b + (b & 0x80)) << 7`. A lane's value is <= 0x17F before the
    // shift and <= 0xBF80 after it, so neither the add nor the shift can carry
    // into the neighbouring lane.
    lo2 = (lo + (lo & 0x00800080u)) << 7;
    hi2 = (hi + (hi & 0x00800080u)) << 7;
  } else {
    lo2 = lo << 8;
    hi2 = hi << 8;
  }
}

// ----------------------------------------------------------------------------
// FP8 decode implementation selector.
//
//   kWord : word-native bit-field move + folded scale bias (default; fastest,
//           no memory traffic, no 8-bit ALU ops -- see above).
//   kLut  : 128-entry magnitude table in `bestla/sycl/fp8_lut.h`.
//   kBits : self-contained inline bit manipulation.
//
// `kLut` / `kBits` are kept reachable for A/B measurement, regression escape,
// and the (checkpoint-impossible) E4M3 NaN encodings.
// ----------------------------------------------------------------------------
enum class Fp8DecodeMode { kWord, kLut, kBits };

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
// Collapsing that mapping, field `j` (K offset `j`) is simply bits
// `[4j+3 : 4j]` of the word, so every field can be extracted with a pair of
// *32-bit* ALU ops and the 8-bit datapath is never touched:
//   asym: `(word >> 4j) & 0xF`
//   sym : `(int)(word << (28 - 4j)) >> 28` -- park the nibble in the sign
//         position, then arithmetic-shift it back down. This is exactly the
//         32-bit form of `int8_t(byte << 4) >> 4`, so the decoded integers are
//         bit-identical to `decode_int4_pair` for all inputs and the
//         decode/prefill parity contract is preserved.
//
// The 32-bit form matters on Xe: `sycl::vec<uint8_t, N>` arithmetic and
// per-byte extraction lower to byte-typed regioning that IGC frequently has to
// expand, and that expansion is what made the sym sign-extension look
// inherently more expensive than the asym mask+shift. Both modes now issue the
// same two native DWORD operations per nibble.
// ----------------------------------------------------------------------------
template <bool Asym>
inline void decode_int4_octet(uint32_t packed, int q[8]) {
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    if constexpr (Asym) {
      q[j] = static_cast<int>((packed >> (4 * j)) & 0xFu);
    } else {
      q[j] = static_cast<int>(packed << (28 - 4 * j)) >> 28;
    }
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
//
// NOTE: this only chooses between the two *per-byte* decoders. The decode GEMV
// selects between {word, lut, bits} through `fp8_decode_mode()` below, which
// still honours this variable when it is set explicitly.
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

// ----------------------------------------------------------------------------
// Host-side selector for the FP8 decode implementation.
//
//   `ARK_FP8_DECODE_MODE` = "word" | "lut" | "bits" (case-insensitive) picks a
//   mode explicitly and wins over everything else.
//
//   Otherwise, if the legacy `ARK_FP8_DECODE_USE_LUT` is set, it keeps its old
//   meaning (`kLut` when truthy, `kBits` when falsy) so existing A/B scripts
//   behave exactly as before.
//
//   With neither set, the default is `kWord` -- the word-native bit-field move.
//
// Re-read on every call (not cached) so tests and benchmarks can toggle the
// path in-process; the result is passed into the kernel as a template argument,
// so there is no per-element runtime branch.
// ----------------------------------------------------------------------------
inline Fp8DecodeMode fp8_decode_mode() {
  const char* mode = std::getenv("ARK_FP8_DECODE_MODE");
  if (mode != nullptr) {
    std::string s(mode);
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (s == "word") return Fp8DecodeMode::kWord;
    if (s == "lut") return Fp8DecodeMode::kLut;
    if (s == "bits") return Fp8DecodeMode::kBits;
    // Unrecognised value: fall through to the legacy variable / default.
  }
  if (std::getenv("ARK_FP8_DECODE_USE_LUT") != nullptr) {
    return fp8_decode_use_lut() ? Fp8DecodeMode::kLut : Fp8DecodeMode::kBits;
  }
  return Fp8DecodeMode::kWord;
}

}  // namespace moe_dequant
}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
