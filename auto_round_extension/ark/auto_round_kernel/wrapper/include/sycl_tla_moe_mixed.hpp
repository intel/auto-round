// SYCL MoE Mixed-Input Prefill Kernel
//
// MoE prefill (Grouped GEMM) entry point that accepts the same set of
// quantized weight encodings as the decode kernel in
// `sycl_tla_moe_decode.hpp` (FP16/BF16 baseline, INT8 sym/asym, INT4
// sym/asym, INT2 sym/asym, FP8 E4M3/E5M2 with group-wise scale).
//
// Stage-1 implementation ("function-first"): a single device-side
// dequantization kernel materialises the per-expert weights into a
// `[E, K, N]` fp16/bf16 temporary, after which the existing CUTLASS-SYCL
// grouped GEMM (`moe_gemm` in `sycl_tla_moe.hpp`) is invoked. This keeps
// the dispatch surface, packing convention and numerical behaviour
// bit-identical to the decode path so end-to-end models can be validated
// and profiled. Mainloop fusion (mixed-input grouped GEMM) is the
// follow-up perf-tuning step.
//
// Layout convention (matches `sycl_tla_moe_decode.hpp`):
//   - activations:           [total_tokens, K]           row-major
//   - weights (fp/bf16):     [num_experts, N, K]         row-major
//   - weights (int8):        [num_experts, N, K]         row-major (uint8 buf)
//   - weights (int4 packed): [num_experts, N, K/2]       row-major (uint8)
//   - weights (int2 packed): [num_experts, N, K/4]       row-major (uint8)
//   - weights (fp8):         [num_experts, N, K]         row-major (uint8 buf)
//   - scales:                [num_experts, N, K/group]   in act dtype
//   - zeros (asym only):     [num_experts, N, K/group]   in act dtype
//   - num_tokens_per_expert: [num_experts]               int32
//   - outputs:               [total_tokens, N]           row-major
//
// The dequantized weights are written transposed to `[E, K, N]` so the
// existing prefill grouped GEMM (which expects `[E, K, N]` row-major) can
// consume them directly without an additional transpose pass.
//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "bestla/bestla.h"
#include "sycl_tla_moe_dequant.hpp"
#include "sycl_tla_moe_prefill_fused.hpp"
#include "sycl_tla_moe_prefill_upcast.hpp"
#include "sycl_tla_common.hpp"

#ifdef ARK_XPU
#include <sycl/sycl.hpp>
#endif

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

namespace ark {
namespace moe_mixed_detail {

// ----------------------------------------------------------------------------
// Kernel name tags (one per specialization, required for SYCL kernel naming).
// ----------------------------------------------------------------------------
template <typename ScalarT>
class MoEDequantKernelFP;

template <typename ScalarT, bool Asym>
class MoEDequantKernelInt8;

template <typename ScalarT, bool Asym>
class MoEDequantKernelInt4;

template <typename ScalarT, bool Asym>
class MoEDequantKernelInt2;

// Fast-path (word-load) kernel name tags for the INT4 / INT2 prefill dequant
// paths. Introduced by PR-1 of the int4/int2 MoE dequant throughput plan:
// one work-item handles a full 32-bit (INT4) / 16-bit (INT2) packed word =
// 8 K outputs, amortising the packed-byte load and per-group scale/zero
// loads that the byte-per-work-item path repeats 4×/2× today. Bit-identical
// numerics vs. the scalar path (arithmetic still in fp32; decoders share
// `decode_int4_pair` / `decode_int2_quad`).
template <typename ScalarT, bool Asym>
class MoEDequantKernelInt4Fast;

template <typename ScalarT, bool Asym>
class MoEDequantKernelInt2Fast;

template <typename ScalarT, bool IsE4M3, bool UseLut>
class MoEDequantKernelFP8;

// Tile sizes for the dequant kernels.
//
// Each work-group covers a (PACK_K x WG_N) tile in (k, n) and writes PACK_K
// consecutive K rows for the same column band. N is the inner dimension so
// stores to the `[E, K, N]` workspace stay coalesced across the sub-group.
//
// PACK_K is chosen per weight encoding so a single work-item dequantises an
// entire packed byte (INT4: 2 outputs, INT2: 4 outputs) or a small run of
// elements sharing one scale/zero load (INT8 / FP8 / FP transpose: 4
// outputs). This removes the redundant packed-byte and scale loads that
// the previous "one element per work-item" launch incurred:
//   - INT4: every byte was read twice (one item per nibble) -> now once.
//   - INT2: every byte was read four times -> now once.
//   - All quantized paths: every scale (and zero) was reloaded by every K
//     element in the group -> now once per PACK_K elements.
// Group-wise scale sharing is safe because PACK_K *divides* group_size in
// every supported configuration: PACK_K is 2 (INT4) or 4 (INT2/INT8/FP8),
// and group_size is always a power of two >= 32 (typically 32, 64, or 128).
// As a result `k_base / group_size` yields the same group index for every
// K element in the PACK_K run, and each kernel can hoist a single scale
// (and, for asym, a single zero) load to amortise across the run.
//
// WG_N is the sub-group store width along N. 32 yields a single 64-byte
// coalesced burst per row for FP16/BF16 writes, which matches the L1
// cache-line size on the target XPUs.
constexpr int WG_N = 32;
constexpr int PACK_K_FP = 4;
constexpr int PACK_K_INT8 = 4;
constexpr int PACK_K_FP8 = 4;

// PR-1 fast-path PACK_K: one work-item handles 4 packed bytes for INT4
// (= 8 nibbles = 8 K outputs) and 2 packed bytes for INT2 (= 8 fields =
// 8 K outputs). Enabling this path requires that:
//   * `K % PACK_K_INT{4,2}_FAST == 0`   (so the launch tail is regular), and
//   * `group_size % PACK_K_INT{4,2}_FAST == 0`   (so the 8 K values in one
//     work-item share the same scale/zero and the hoisted-load optimisation
//     stays valid).
// Both hold for every configuration actually used in the repo test/eval
// suite (group_size is 32/64/128 and K is a multiple of group_size), but
// the kernels fall back to the byte-per-work-item path when either
// constraint is violated so short-K unit tests keep working.
// ----------------------------------------------------------------------------
// FP16 / BF16 weight reshape: in-place transpose [E, N, K] -> [E, K, N].
// Implemented as a generic dequant pass with identity scale; used so the
// public entry point can dispatch FP16/BF16 through the same code path
// as the quantized variants.
// ----------------------------------------------------------------------------
template <typename ScalarT>
void launch_dequant_fp(sycl::queue* q, const ScalarT* weights_NK, ScalarT* weights_KN, int E, int N, int K,
                       const int* num_tokens_per_expert = nullptr) {
  if (E == 0 || N == 0 || K == 0) return;

  // Each work-item copies PACK_K_FP consecutive K elements for a single
  // (e, n). The launch grid covers ceil(K / PACK_K_FP) along the middle dim;
  // an inner bounds check guards the K tail when K is not a multiple of
  // PACK_K_FP (e.g. K < PACK_K_FP in tiny unit tests).
  const int k_tiles = (K + PACK_K_FP - 1) / PACK_K_FP;
  sycl::range<3> global{static_cast<size_t>(E), static_cast<size_t>(k_tiles),
                        static_cast<size_t>((N + WG_N - 1) / WG_N) * WG_N};
  sycl::range<3> local{1, 1, static_cast<size_t>(WG_N)};

  q->parallel_for<MoEDequantKernelFP<ScalarT>>(
      sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> it) {
        const int e = static_cast<int>(it.get_global_id(0));
        // Skip experts that receive no tokens in this prefill batch; the
        // grouped GEMM will not read their rows of `weights_KN` either.
        if (num_tokens_per_expert != nullptr && num_tokens_per_expert[e] == 0) return;
        const int k_base = static_cast<int>(it.get_global_id(1)) * PACK_K_FP;
        const int n = static_cast<int>(it.get_global_id(2));
        if (n >= N) return;
        const size_t w_row = (static_cast<size_t>(e) * N + static_cast<size_t>(n)) * K;
        const size_t out_base = static_cast<size_t>(e) * K * N + static_cast<size_t>(n);
#pragma unroll
        for (int j = 0; j < PACK_K_FP; ++j) {
          const int k = k_base + j;
          if (k >= K) break;
          const ScalarT v = weights_NK[w_row + static_cast<size_t>(k)];
          weights_KN[out_base + static_cast<size_t>(k) * N] = v;
        }
      });
}

// ----------------------------------------------------------------------------
// INT8 (S8) dequant: [E, N, K] uint8 -> [E, K, N] ScalarT.
// Asym=false: signed int8 in [-128, 127], dequant = q * scale
// Asym=true : unsigned uint8 in [0, 255],  dequant = (q - zero) * scale
// ----------------------------------------------------------------------------
template <typename ScalarT, bool Asym>
void launch_dequant_int8(sycl::queue* q, const uint8_t* weights_NK, const ScalarT* scales, const ScalarT* zeros,
                         ScalarT* weights_KN, int E, int N, int K, int group_size,
                         const int* num_tokens_per_expert = nullptr) {
  if (E == 0 || N == 0 || K == 0) return;
  if (Asym && zeros == nullptr) {
    throw std::invalid_argument("moe_gemm_prefill(int8): zeros pointer required when asym=true");
  }
  const int num_groups_k = K / group_size;

  // Each work-item dequantises PACK_K_INT8 consecutive K outputs for a
  // single (e, n), sharing one scale/zero load. PACK_K_INT8 (=4) is always
  // <= group_size (>=32 in practice), so the cached scale/zero is valid
  // for every element in the run.
  const int k_tiles = (K + PACK_K_INT8 - 1) / PACK_K_INT8;
  sycl::range<3> global{static_cast<size_t>(E), static_cast<size_t>(k_tiles),
                        static_cast<size_t>((N + WG_N - 1) / WG_N) * WG_N};
  sycl::range<3> local{1, 1, static_cast<size_t>(WG_N)};

  q->parallel_for<MoEDequantKernelInt8<ScalarT, Asym>>(
      sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> it) {
        const int e = static_cast<int>(it.get_global_id(0));
        if (num_tokens_per_expert != nullptr && num_tokens_per_expert[e] == 0) return;
        const int k_base = static_cast<int>(it.get_global_id(1)) * PACK_K_INT8;
        const int n = static_cast<int>(it.get_global_id(2));
        if (n >= N) return;
        const size_t w_row = (static_cast<size_t>(e) * N + static_cast<size_t>(n)) * K;
        const size_t out_base = static_cast<size_t>(e) * K * N + static_cast<size_t>(n);
        // Hoist scale/zero loads: PACK_K_INT8 K values share the same group
        // because PACK_K_INT8 divides group_size (see PACK_K constants
        // above), so `k_base / group_size` is constant across the run.
        const int g = k_base / group_size;
        const size_t s_idx = (static_cast<size_t>(e) * N + static_cast<size_t>(n)) * num_groups_k +
                             static_cast<size_t>(g);
        const float scale = static_cast<float>(scales[s_idx]);
        const float zero = Asym ? static_cast<float>(zeros[s_idx]) : 0.0f;
#pragma unroll
        for (int j = 0; j < PACK_K_INT8; ++j) {
          const int k = k_base + j;
          if (k >= K) break;
          const uint8_t raw = weights_NK[w_row + static_cast<size_t>(k)];
          const int q = moe_dequant::decode_int8<Asym>(raw);
          float w;
          if constexpr (Asym) {
            w = (static_cast<float>(q) - zero) * scale;
          } else {
            w = static_cast<float>(q) * scale;
          }
          weights_KN[out_base + static_cast<size_t>(k) * N] = static_cast<ScalarT>(w);
        }
      });
}

// ----------------------------------------------------------------------------
// INT4 (S4_CLIP) dequant: [E, N, K/2] uint8 packed -> [E, K, N] ScalarT.
// Packing: low nibble at lower K (k = 2*i), high nibble at higher K (k = 2*i+1).
// Asym=false: signed nibble in [-8, 7], dequant = q * scale
// Asym=true : unsigned nibble in [0, 15], dequant = (q - zero) * scale
// ----------------------------------------------------------------------------
template <typename ScalarT, bool Asym>
void launch_dequant_int4(sycl::queue* q, const uint8_t* weights_NKp, const ScalarT* scales, const ScalarT* zeros,
                         ScalarT* weights_KN, int E, int N, int K, int group_size,
                         const int* num_tokens_per_expert = nullptr) {
  if (E == 0 || N == 0 || K == 0) return;
  if (Asym && zeros == nullptr) {
    throw std::invalid_argument("moe_gemm_prefill(int4): zeros pointer required when asym=true");
  }
  if ((K & 1) != 0) {
    throw std::invalid_argument("moe_gemm_prefill(int4): K must be even");
  }
  const int num_groups_k = K / group_size;
  const int k_packed = K / 2;

  // PR-1 fast path: one work-item processes a full 4-byte packed word =
  // 8 nibbles = 8 K outputs for a single (e, n). Prerequisites: K and
  // group_size are both multiples of PACK_K_INT4_FAST (=8) so that (a) the
  // launch grid has no partial-word tail and (b) all 8 K values share the
  // same per-group scale/zero, letting us keep the byte-path's hoist-once
  // load pattern. This cuts packed-byte loads and scale/zero loads by 4×
  // relative to the byte-per-work-item path.
  if ((K % PACK_K_INT4_FAST) == 0 && (group_size % PACK_K_INT4_FAST) == 0) {
    const int k_words = K / PACK_K_INT4_FAST;  // == k_packed / 4
    sycl::range<3> global_fast{static_cast<size_t>(E), static_cast<size_t>(k_words),
                               static_cast<size_t>((N + WG_N - 1) / WG_N) * WG_N};
    sycl::range<3> local_fast{1, 1, static_cast<size_t>(WG_N)};

    q->parallel_for<MoEDequantKernelInt4Fast<ScalarT, Asym>>(
        sycl::nd_range<3>(global_fast, local_fast), [=](sycl::nd_item<3> it) {
          const int e = static_cast<int>(it.get_global_id(0));
          if (num_tokens_per_expert != nullptr && num_tokens_per_expert[e] == 0) return;
          const int kw = static_cast<int>(it.get_global_id(1));
          const int n = static_cast<int>(it.get_global_id(2));
          if (n >= N) return;
          const int k_base = kw * PACK_K_INT4_FAST;
          // PACK_K_INT4_FAST divides group_size (checked above), so all 8
          // K outputs in this work-item share (g, scale, zero).
          const int g = k_base / group_size;
          const size_t s_idx = (static_cast<size_t>(e) * N + static_cast<size_t>(n)) * num_groups_k +
                               static_cast<size_t>(g);
          const float scale = static_cast<float>(scales[s_idx]);
          const float zero = Asym ? static_cast<float>(zeros[s_idx]) : 0.0f;
          // Read 4 consecutive packed bytes = one 32-bit little-endian word.
          // We assemble via byte loads rather than a `reinterpret_cast` to
          // avoid any strict-aliasing / alignment assumption on the input
          // buffer; the DPC++ backend fuses these into a single dword load
          // when the row start is 4-byte aligned (which it is whenever
          // k_packed is a multiple of 4, guaranteed by K%8==0).
          const size_t row_kp_base = (static_cast<size_t>(e) * N + static_cast<size_t>(n)) * k_packed +
                                     static_cast<size_t>(kw) * 4;
          const uint32_t packed =
              static_cast<uint32_t>(weights_NKp[row_kp_base + 0]) |
              (static_cast<uint32_t>(weights_NKp[row_kp_base + 1]) << 8) |
              (static_cast<uint32_t>(weights_NKp[row_kp_base + 2]) << 16) |
              (static_cast<uint32_t>(weights_NKp[row_kp_base + 3]) << 24);
          int qv[8];
          moe_dequant::decode_int4_octet<Asym>(packed, qv);
          const size_t out_base = static_cast<size_t>(e) * K * N + static_cast<size_t>(n);
#pragma unroll
          for (int j = 0; j < PACK_K_INT4_FAST; ++j) {
            float w;
            if constexpr (Asym) {
              w = (static_cast<float>(qv[j]) - zero) * scale;
            } else {
              w = static_cast<float>(qv[j]) * scale;
            }
            weights_KN[out_base + static_cast<size_t>(k_base + j) * N] = static_cast<ScalarT>(w);
          }
        });
    return;
  }

  // Fallback byte-per-work-item path (retained for short-K / small-group
  // configurations where the fast path's PACK_K_INT4_FAST=8 constraint
  // does not hold).
  // Each work-item now dequantises one full packed byte = PACK_K_INT4 (=2)
  // consecutive K outputs for a single (e, n). The middle launch dim is
  // therefore k_packed instead of K, halving the work-item count and
  // eliminating the previous "two items reading the same byte" pattern.
  sycl::range<3> global{static_cast<size_t>(E), static_cast<size_t>(k_packed),
                        static_cast<size_t>((N + WG_N - 1) / WG_N) * WG_N};
  sycl::range<3> local{1, 1, static_cast<size_t>(WG_N)};

  q->parallel_for<MoEDequantKernelInt4<ScalarT, Asym>>(
      sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> it) {
        const int e = static_cast<int>(it.get_global_id(0));
        if (num_tokens_per_expert != nullptr && num_tokens_per_expert[e] == 0) return;
        const int kp = static_cast<int>(it.get_global_id(1));
        const int n = static_cast<int>(it.get_global_id(2));
        if (n >= N) return;
        const int k_base = kp * PACK_K_INT4;
        // PACK_K_INT4 (=2) divides group_size, so all PACK_K_INT4 K values
        // in this run share the same scale/zero (one hoisted load each).
        const int g = k_base / group_size;
        const size_t s_idx = (static_cast<size_t>(e) * N + static_cast<size_t>(n)) * num_groups_k +
                             static_cast<size_t>(g);
        const float scale = static_cast<float>(scales[s_idx]);
        const float zero = Asym ? static_cast<float>(zeros[s_idx]) : 0.0f;
        const uint8_t packed = weights_NKp[(static_cast<size_t>(e) * N + static_cast<size_t>(n)) * k_packed +
                                           static_cast<size_t>(kp)];
        const size_t out_base = static_cast<size_t>(e) * K * N + static_cast<size_t>(n);
        int q_lo, q_hi;
        moe_dequant::decode_int4_pair<Asym>(packed, q_lo, q_hi);
        float w0, w1;
        if constexpr (Asym) {
          w0 = (static_cast<float>(q_lo) - zero) * scale;
          w1 = (static_cast<float>(q_hi) - zero) * scale;
        } else {
          w0 = static_cast<float>(q_lo) * scale;
          w1 = static_cast<float>(q_hi) * scale;
        }
        weights_KN[out_base + static_cast<size_t>(k_base) * N] = static_cast<ScalarT>(w0);
        weights_KN[out_base + static_cast<size_t>(k_base + 1) * N] = static_cast<ScalarT>(w1);
      });
}

// ----------------------------------------------------------------------------
// INT2 (S2_CLIP) dequant: [E, N, K/4] uint8 packed -> [E, K, N] ScalarT.
// Packing: byte = q0 | (q1<<2) | (q2<<4) | (q3<<6); field j corresponds to
// K index 4*i + j.
// Asym=false: signed in [-2, 1]; Asym=true: unsigned in [0, 3].
// ----------------------------------------------------------------------------
template <typename ScalarT, bool Asym>
void launch_dequant_int2(sycl::queue* q, const uint8_t* weights_NKp, const ScalarT* scales, const ScalarT* zeros,
                         ScalarT* weights_KN, int E, int N, int K, int group_size,
                         const int* num_tokens_per_expert = nullptr) {
  if (E == 0 || N == 0 || K == 0) return;
  if (Asym && zeros == nullptr) {
    throw std::invalid_argument("moe_gemm_prefill(int2): zeros pointer required when asym=true");
  }
  if ((K & 3) != 0) {
    throw std::invalid_argument("moe_gemm_prefill(int2): K must be a multiple of 4");
  }
  const int num_groups_k = K / group_size;
  const int k_packed = K / 4;

  // PR-1 fast path: one work-item processes a full 2-byte packed word =
  // 8 2-bit fields = 8 K outputs for a single (e, n). Enables the same
  // load-amortisation win as the int4 fast path (see comment there). The
  // K%8==0 && group_size%8==0 constraints are met by every configuration
  // exercised by the accuracy / perf tests (group_size is 32/64/128).
  if ((K % PACK_K_INT2_FAST) == 0 && (group_size % PACK_K_INT2_FAST) == 0) {
    const int k_words = K / PACK_K_INT2_FAST;  // == k_packed / 2
    sycl::range<3> global_fast{static_cast<size_t>(E), static_cast<size_t>(k_words),
                               static_cast<size_t>((N + WG_N - 1) / WG_N) * WG_N};
    sycl::range<3> local_fast{1, 1, static_cast<size_t>(WG_N)};

    q->parallel_for<MoEDequantKernelInt2Fast<ScalarT, Asym>>(
        sycl::nd_range<3>(global_fast, local_fast), [=](sycl::nd_item<3> it) {
          const int e = static_cast<int>(it.get_global_id(0));
          if (num_tokens_per_expert != nullptr && num_tokens_per_expert[e] == 0) return;
          const int kw = static_cast<int>(it.get_global_id(1));
          const int n = static_cast<int>(it.get_global_id(2));
          if (n >= N) return;
          const int k_base = kw * PACK_K_INT2_FAST;
          const int g = k_base / group_size;
          const size_t s_idx = (static_cast<size_t>(e) * N + static_cast<size_t>(n)) * num_groups_k +
                               static_cast<size_t>(g);
          const float scale = static_cast<float>(scales[s_idx]);
          const float zero = Asym ? static_cast<float>(zeros[s_idx]) : 0.0f;
          const size_t row_kp_base = (static_cast<size_t>(e) * N + static_cast<size_t>(n)) * k_packed +
                                     static_cast<size_t>(kw) * 2;
          const uint16_t packed =
              static_cast<uint16_t>(static_cast<uint32_t>(weights_NKp[row_kp_base + 0]) |
                                    (static_cast<uint32_t>(weights_NKp[row_kp_base + 1]) << 8));
          int qv[8];
          moe_dequant::decode_int2_octet<Asym>(packed, qv);
          const size_t out_base = static_cast<size_t>(e) * K * N + static_cast<size_t>(n);
#pragma unroll
          for (int j = 0; j < PACK_K_INT2_FAST; ++j) {
            float w;
            if constexpr (Asym) {
              w = (static_cast<float>(qv[j]) - zero) * scale;
            } else {
              w = static_cast<float>(qv[j]) * scale;
            }
            weights_KN[out_base + static_cast<size_t>(k_base + j) * N] = static_cast<ScalarT>(w);
          }
        });
    return;
  }

  // Fallback byte-per-work-item path (retained for short-K / small-group
  // configurations where the fast path's constraints do not hold).
  // One work-item handles one full packed byte = PACK_K_INT2 (=4)
  // consecutive K outputs for a single (e, n). This removes the previous
  // 4x duplicated byte loads (one item per 2-bit field) and 4x scale
  // reloads.
  sycl::range<3> global{static_cast<size_t>(E), static_cast<size_t>(k_packed),
                        static_cast<size_t>((N + WG_N - 1) / WG_N) * WG_N};
  sycl::range<3> local{1, 1, static_cast<size_t>(WG_N)};

  q->parallel_for<MoEDequantKernelInt2<ScalarT, Asym>>(
      sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> it) {
        const int e = static_cast<int>(it.get_global_id(0));
        if (num_tokens_per_expert != nullptr && num_tokens_per_expert[e] == 0) return;
        const int kp = static_cast<int>(it.get_global_id(1));
        const int n = static_cast<int>(it.get_global_id(2));
        if (n >= N) return;
        const int k_base = kp * PACK_K_INT2;
        // PACK_K_INT2 (=4) divides group_size, so all PACK_K_INT2 K values
        // in this run share the same scale/zero (one hoisted load each).
        const int g = k_base / group_size;
        const size_t s_idx = (static_cast<size_t>(e) * N + static_cast<size_t>(n)) * num_groups_k +
                             static_cast<size_t>(g);
        const float scale = static_cast<float>(scales[s_idx]);
        const float zero = Asym ? static_cast<float>(zeros[s_idx]) : 0.0f;
        const uint8_t packed = weights_NKp[(static_cast<size_t>(e) * N + static_cast<size_t>(n)) * k_packed +
                                           static_cast<size_t>(kp)];
        const size_t out_base = static_cast<size_t>(e) * K * N + static_cast<size_t>(n);
        int q[4];
        moe_dequant::decode_int2_quad<Asym>(packed, q);
#pragma unroll
        for (int j = 0; j < PACK_K_INT2; ++j) {
          float w;
          if constexpr (Asym) {
            w = (static_cast<float>(q[j]) - zero) * scale;
          } else {
            w = static_cast<float>(q[j]) * scale;
          }
          weights_KN[out_base + static_cast<size_t>(k_base + j) * N] = static_cast<ScalarT>(w);
        }
      });
}

// ----------------------------------------------------------------------------
// FP8 (E4M3 / E5M2) dequant: [E, N, K] uint8 -> [E, K, N] ScalarT.
// Per-group scale applied; no zero-points (caller must enforce sym).
// `UseLut` selects the LUT vs inline-bits decode at compile time
// (driven by `ARK_FP8_DECODE_USE_LUT`, same as the decode path).
// ----------------------------------------------------------------------------
template <typename ScalarT, bool IsE4M3, bool UseLut>
void launch_dequant_fp8(sycl::queue* q, const uint8_t* weights_NK, const ScalarT* scales, ScalarT* weights_KN, int E,
                        int N, int K, int group_size, const int* num_tokens_per_expert = nullptr) {
  if (E == 0 || N == 0 || K == 0) return;
  const int num_groups_k = K / group_size;

  // PACK_K_FP8 (=4) consecutive K outputs per work-item, sharing one scale
  // load. PACK_K_FP8 <= group_size in all supported configurations.
  const int k_tiles = (K + PACK_K_FP8 - 1) / PACK_K_FP8;
  sycl::range<3> global{static_cast<size_t>(E), static_cast<size_t>(k_tiles),
                        static_cast<size_t>((N + WG_N - 1) / WG_N) * WG_N};
  sycl::range<3> local{1, 1, static_cast<size_t>(WG_N)};

  q->parallel_for<MoEDequantKernelFP8<ScalarT, IsE4M3, UseLut>>(
      sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> it) {
        const int e = static_cast<int>(it.get_global_id(0));
        if (num_tokens_per_expert != nullptr && num_tokens_per_expert[e] == 0) return;
        const int k_base = static_cast<int>(it.get_global_id(1)) * PACK_K_FP8;
        const int n = static_cast<int>(it.get_global_id(2));
        if (n >= N) return;
        const size_t w_row = (static_cast<size_t>(e) * N + static_cast<size_t>(n)) * K;
        const size_t out_base = static_cast<size_t>(e) * K * N + static_cast<size_t>(n);
        const int g = k_base / group_size;
        const size_t s_idx = (static_cast<size_t>(e) * N + static_cast<size_t>(n)) * num_groups_k +
                             static_cast<size_t>(g);
        const float scale = static_cast<float>(scales[s_idx]);
#pragma unroll
        for (int j = 0; j < PACK_K_FP8; ++j) {
          const int k = k_base + j;
          if (k >= K) break;
          const uint8_t raw = weights_NK[w_row + static_cast<size_t>(k)];
          const float w = moe_dequant::decode_fp8<IsE4M3, UseLut>(raw) * scale;
          weights_KN[out_base + static_cast<size_t>(k) * N] = static_cast<ScalarT>(w);
        }
      });
}

// ----------------------------------------------------------------------------
// Dispatch helper: dequant any supported weight encoding into `weights_KN`
// (already-allocated `[E, K, N]` ScalarT buffer) using ScalarT == act dtype.
// ----------------------------------------------------------------------------
template <typename ScalarT>
void dequant_to_KN(sycl::queue* q, const void* weights, const void* scales, const void* zeros, ScalarT* weights_KN,
                   BTLA_DTYPE weight_dtype, int E, int N, int K, int group_size, bool asym,
                   const int* num_tokens_per_expert = nullptr) {
  if (weight_dtype == BTLA_DTYPE::F16 || weight_dtype == BTLA_DTYPE::BF16) {
    launch_dequant_fp<ScalarT>(q, static_cast<const ScalarT*>(weights), weights_KN, E, N, K, num_tokens_per_expert);
    return;
  }
  if (weight_dtype == BTLA_DTYPE::S8) {
    if (asym) {
      launch_dequant_int8<ScalarT, true>(q, static_cast<const uint8_t*>(weights), static_cast<const ScalarT*>(scales),
                                         static_cast<const ScalarT*>(zeros), weights_KN, E, N, K, group_size,
                                         num_tokens_per_expert);
    } else {
      launch_dequant_int8<ScalarT, false>(q, static_cast<const uint8_t*>(weights), static_cast<const ScalarT*>(scales),
                                          static_cast<const ScalarT*>(zeros), weights_KN, E, N, K, group_size,
                                          num_tokens_per_expert);
    }
    return;
  }
  if (weight_dtype == BTLA_DTYPE::S4_CLIP) {
    if (asym) {
      launch_dequant_int4<ScalarT, true>(q, static_cast<const uint8_t*>(weights), static_cast<const ScalarT*>(scales),
                                         static_cast<const ScalarT*>(zeros), weights_KN, E, N, K, group_size,
                                         num_tokens_per_expert);
    } else {
      launch_dequant_int4<ScalarT, false>(q, static_cast<const uint8_t*>(weights), static_cast<const ScalarT*>(scales),
                                          static_cast<const ScalarT*>(zeros), weights_KN, E, N, K, group_size,
                                          num_tokens_per_expert);
    }
    return;
  }
  if (weight_dtype == BTLA_DTYPE::S2_CLIP) {
    if (asym) {
      launch_dequant_int2<ScalarT, true>(q, static_cast<const uint8_t*>(weights), static_cast<const ScalarT*>(scales),
                                         static_cast<const ScalarT*>(zeros), weights_KN, E, N, K, group_size,
                                         num_tokens_per_expert);
    } else {
      launch_dequant_int2<ScalarT, false>(q, static_cast<const uint8_t*>(weights), static_cast<const ScalarT*>(scales),
                                          static_cast<const ScalarT*>(zeros), weights_KN, E, N, K, group_size,
                                          num_tokens_per_expert);
    }
    return;
  }
  if (weight_dtype == BTLA_DTYPE::F8_E4M3 || weight_dtype == BTLA_DTYPE::F8_E5M2) {
    if (asym) {
      throw std::invalid_argument("moe_gemm_prefill(fp8): asym mode is not supported");
    }
    const bool is_e4m3 = (weight_dtype == BTLA_DTYPE::F8_E4M3);
    const bool use_lut = moe_dequant::fp8_decode_use_lut();
    // Opt-in SLM-transposed fused dequant path (PR-B1 / Commit 2). This
    // commit only covers fp8-e4m3; e5m2 still takes the legacy path.
    // See `sycl_tla_moe_prefill_fused.hpp` for the tile / SLM design.
    // The flag defaults to OFF, so end-to-end behaviour is unchanged
    // unless `ARK_MOE_PREFILL_FUSED_FP8=1` is set.
    const bool try_fused = is_e4m3 && moe_prefill_fused_detail::moe_prefill_fused_fp8_enabled() &&
                           moe_prefill_fused_detail::moe_prefill_fused_fp8_shape_ok(N, K, group_size);
    if (try_fused) {
      if (use_lut) {
        moe_prefill_fused_detail::launch_dequant_fp8_slm<ScalarT, true, true>(
            q, static_cast<const uint8_t*>(weights), static_cast<const ScalarT*>(scales), weights_KN, E, N, K,
            group_size, num_tokens_per_expert);
      } else {
        moe_prefill_fused_detail::launch_dequant_fp8_slm<ScalarT, true, false>(
            q, static_cast<const uint8_t*>(weights), static_cast<const ScalarT*>(scales), weights_KN, E, N, K,
            group_size, num_tokens_per_expert);
      }
      return;
    }
    if (is_e4m3) {
      if (use_lut) {
        launch_dequant_fp8<ScalarT, true, true>(q, static_cast<const uint8_t*>(weights),
                                                static_cast<const ScalarT*>(scales), weights_KN, E, N, K,
                                                group_size, num_tokens_per_expert);
      } else {
        launch_dequant_fp8<ScalarT, true, false>(q, static_cast<const uint8_t*>(weights),
                                                 static_cast<const ScalarT*>(scales), weights_KN, E, N, K,
                                                 group_size, num_tokens_per_expert);
      }
    } else {
      if (use_lut) {
        launch_dequant_fp8<ScalarT, false, true>(q, static_cast<const uint8_t*>(weights),
                                                 static_cast<const ScalarT*>(scales), weights_KN, E, N, K,
                                                 group_size, num_tokens_per_expert);
      } else {
        launch_dequant_fp8<ScalarT, false, false>(q, static_cast<const uint8_t*>(weights),
                                                  static_cast<const ScalarT*>(scales), weights_KN, E, N, K,
                                                  group_size, num_tokens_per_expert);
      }
    }
    return;
  }
  throw std::invalid_argument(
      "moe_gemm_prefill: unsupported weight_dtype (supported: F16, BF16, S8, S4_CLIP, S2_CLIP, F8_E4M3, F8_E5M2)");
}

}  // namespace moe_mixed_detail

// ----------------------------------------------------------------------------
// Public API
//
// MoE prefill (Grouped GEMM) supporting the same set of weight encodings as
// `moe_gemm_decode`. Activations/scales/zeros/outputs are in `act_dtype`
// (FP16 or BF16). For the unquantized fast path (weight_dtype matches
// act_dtype and is FP16/BF16), the call is forwarded directly to the
// existing `moe_gemm` (which already expects `[E, K, N]` row-major
// weights). For all other dtypes, weights are dequantised on-device into
// a temporary `[E, K, N]` buffer of `act_dtype` and then handed to the
// same grouped GEMM. The temporary buffer must be supplied by the caller
// (see the Python wrapper which allocates it sized
// `E * K * N * sizeof(act_dtype)`); this avoids per-call USM allocations
// and keeps memory ownership in PyTorch's caching allocator.
//
// IMPORTANT layout note:
//   - Quantized `weights` are `[E, N, K_packed]` (decode-style).
//   - The unquantized fast path (`act_dtype == weight_dtype` and FP/BF16)
//     forwards directly to `moe_gemm`, which expects `[E, K, N]` weights.
//     Callers must therefore pass already-`[E, K, N]` weights for the
//     unquantized fast path (matching the existing `moe_gemm` contract).
//     The Python wrapper handles this by exposing the unquantized path
//     under the same shape contract as `moe_gemm` and the quantized paths
//     under the same shape contract as `moe_gemm_decode`.
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// MoE prefill Grouped GEMM -- FP8 per-tensor DPAS (Variant A).
//
// Separate entry point (not multiplexed through `moe_gemm_prefill` because
// the scale layout differs -- `[E]` FP32 per-tensor vs. the
// `[E, N, K/group_size]` act-dtype per-K-group layout `moe_gemm_prefill`
// takes). Weights are `[E, K, N]` row-major (vllm convention).
//
// STATUS: NEEDS-HARDWARE-VALIDATION. See
// `sycl_tla_moe_prefill_fp8_dpas.hpp` for the port's provenance & the
// on-hardware TODOs.
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// MoE prefill Grouped GEMM -- INT8 per-tensor DPAS (Variant A).
//
// Separate entry point (not multiplexed through `moe_gemm_prefill` because
// the scale layout differs -- `[E]` FP32 per-tensor vs. the
// `[E, N, K/group_size]` act-dtype per-K-group layout `moe_gemm_prefill`
// takes for its INT8 branch). Weights are `[E, K, N]` row-major (vllm
// convention, one signed byte per element), scales are `[E]` FP32.
//
// STATUS: NEEDS-HARDWARE-VALIDATION. See
// `sycl_tla_moe_prefill_int_dpas.hpp` for the port's provenance & the
// on-hardware TODOs.
// ----------------------------------------------------------------------------
}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
