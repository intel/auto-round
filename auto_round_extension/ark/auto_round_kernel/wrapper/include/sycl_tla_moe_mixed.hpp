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
#include "sycl_tla_moe.hpp"
#include "sycl_tla_moe_dequant.hpp"

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

template <typename ScalarT, bool IsE4M3, bool UseLut>
class MoEDequantKernelFP8;

// Tile sizes for the dequant kernel: each work-item dequantises one weight
// element, with the work-group covering a 1xWG_N tile along (k, n). N is
// chosen as the inner dimension to keep stores to the [E, K, N] output
// coalesced across the sub-group.
constexpr int WG_N = 16;

// ----------------------------------------------------------------------------
// FP16 / BF16 weight reshape: in-place transpose [E, N, K] -> [E, K, N].
// Implemented as a generic dequant pass with identity scale; used so the
// public entry point can dispatch FP16/BF16 through the same code path
// as the quantized variants.
// ----------------------------------------------------------------------------
template <typename ScalarT>
void launch_dequant_fp(sycl::queue* q, const ScalarT* weights_NK, ScalarT* weights_KN, int E, int N, int K) {
  if (E == 0 || N == 0 || K == 0) return;

  sycl::range<3> global{static_cast<size_t>(E), static_cast<size_t>(K),
                        static_cast<size_t>((N + WG_N - 1) / WG_N) * WG_N};
  sycl::range<3> local{1, 1, static_cast<size_t>(WG_N)};

  q->parallel_for<MoEDequantKernelFP<ScalarT>>(
      sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> it) {
        const int e = static_cast<int>(it.get_global_id(0));
        const int k = static_cast<int>(it.get_global_id(1));
        const int n = static_cast<int>(it.get_global_id(2));
        if (n >= N) return;
        const ScalarT v =
            weights_NK[(static_cast<size_t>(e) * N + static_cast<size_t>(n)) * K + static_cast<size_t>(k)];
        weights_KN[(static_cast<size_t>(e) * K + static_cast<size_t>(k)) * N + static_cast<size_t>(n)] = v;
      });
}

// ----------------------------------------------------------------------------
// INT8 (S8) dequant: [E, N, K] uint8 -> [E, K, N] ScalarT.
// Asym=false: signed int8 in [-128, 127], dequant = q * scale
// Asym=true : unsigned uint8 in [0, 255],  dequant = (q - zero) * scale
// ----------------------------------------------------------------------------
template <typename ScalarT, bool Asym>
void launch_dequant_int8(sycl::queue* q, const uint8_t* weights_NK, const ScalarT* scales, const ScalarT* zeros,
                         ScalarT* weights_KN, int E, int N, int K, int group_size) {
  if (E == 0 || N == 0 || K == 0) return;
  if (Asym && zeros == nullptr) {
    throw std::invalid_argument("moe_gemm_prefill(int8): zeros pointer required when asym=true");
  }
  const int num_groups_k = K / group_size;

  sycl::range<3> global{static_cast<size_t>(E), static_cast<size_t>(K),
                        static_cast<size_t>((N + WG_N - 1) / WG_N) * WG_N};
  sycl::range<3> local{1, 1, static_cast<size_t>(WG_N)};

  q->parallel_for<MoEDequantKernelInt8<ScalarT, Asym>>(
      sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> it) {
        const int e = static_cast<int>(it.get_global_id(0));
        const int k = static_cast<int>(it.get_global_id(1));
        const int n = static_cast<int>(it.get_global_id(2));
        if (n >= N) return;
        const int g = k / group_size;
        const size_t s_idx = (static_cast<size_t>(e) * N + static_cast<size_t>(n)) * num_groups_k +
                             static_cast<size_t>(g);
        const float scale = static_cast<float>(scales[s_idx]);
        const uint8_t raw =
            weights_NK[(static_cast<size_t>(e) * N + static_cast<size_t>(n)) * K + static_cast<size_t>(k)];
        float w;
        if constexpr (Asym) {
          const float zero = static_cast<float>(zeros[s_idx]);
          w = (static_cast<float>(raw) - zero) * scale;
        } else {
          w = static_cast<float>(static_cast<int8_t>(raw)) * scale;
        }
        weights_KN[(static_cast<size_t>(e) * K + static_cast<size_t>(k)) * N + static_cast<size_t>(n)] =
            static_cast<ScalarT>(w);
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
                         ScalarT* weights_KN, int E, int N, int K, int group_size) {
  if (E == 0 || N == 0 || K == 0) return;
  if (Asym && zeros == nullptr) {
    throw std::invalid_argument("moe_gemm_prefill(int4): zeros pointer required when asym=true");
  }
  if ((K & 1) != 0) {
    throw std::invalid_argument("moe_gemm_prefill(int4): K must be even");
  }
  const int num_groups_k = K / group_size;
  const int k_packed = K / 2;

  sycl::range<3> global{static_cast<size_t>(E), static_cast<size_t>(K),
                        static_cast<size_t>((N + WG_N - 1) / WG_N) * WG_N};
  sycl::range<3> local{1, 1, static_cast<size_t>(WG_N)};

  q->parallel_for<MoEDequantKernelInt4<ScalarT, Asym>>(
      sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> it) {
        const int e = static_cast<int>(it.get_global_id(0));
        const int k = static_cast<int>(it.get_global_id(1));
        const int n = static_cast<int>(it.get_global_id(2));
        if (n >= N) return;
        const int g = k / group_size;
        const size_t s_idx = (static_cast<size_t>(e) * N + static_cast<size_t>(n)) * num_groups_k +
                             static_cast<size_t>(g);
        const float scale = static_cast<float>(scales[s_idx]);
        const uint8_t packed =
            weights_NKp[(static_cast<size_t>(e) * N + static_cast<size_t>(n)) * k_packed +
                        static_cast<size_t>(k / 2)];
        const bool is_high = (k & 1) != 0;
        float w;
        if constexpr (Asym) {
          const float zero = static_cast<float>(zeros[s_idx]);
          const int q = static_cast<int>(is_high ? ((packed >> 4) & 0x0F) : (packed & 0x0F));
          w = (static_cast<float>(q) - zero) * scale;
        } else {
          // Sign-extend 4-bit -> 8-bit by shifting into the top nibble.
          const int q = is_high
              ? static_cast<int>(static_cast<int8_t>(packed & 0xF0) >> 4)
              : static_cast<int>(static_cast<int8_t>(packed << 4) >> 4);
          w = static_cast<float>(q) * scale;
        }
        weights_KN[(static_cast<size_t>(e) * K + static_cast<size_t>(k)) * N + static_cast<size_t>(n)] =
            static_cast<ScalarT>(w);
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
                         ScalarT* weights_KN, int E, int N, int K, int group_size) {
  if (E == 0 || N == 0 || K == 0) return;
  if (Asym && zeros == nullptr) {
    throw std::invalid_argument("moe_gemm_prefill(int2): zeros pointer required when asym=true");
  }
  if ((K & 3) != 0) {
    throw std::invalid_argument("moe_gemm_prefill(int2): K must be a multiple of 4");
  }
  const int num_groups_k = K / group_size;
  const int k_packed = K / 4;

  sycl::range<3> global{static_cast<size_t>(E), static_cast<size_t>(K),
                        static_cast<size_t>((N + WG_N - 1) / WG_N) * WG_N};
  sycl::range<3> local{1, 1, static_cast<size_t>(WG_N)};

  q->parallel_for<MoEDequantKernelInt2<ScalarT, Asym>>(
      sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> it) {
        const int e = static_cast<int>(it.get_global_id(0));
        const int k = static_cast<int>(it.get_global_id(1));
        const int n = static_cast<int>(it.get_global_id(2));
        if (n >= N) return;
        const int g = k / group_size;
        const size_t s_idx = (static_cast<size_t>(e) * N + static_cast<size_t>(n)) * num_groups_k +
                             static_cast<size_t>(g);
        const float scale = static_cast<float>(scales[s_idx]);
        const uint8_t packed =
            weights_NKp[(static_cast<size_t>(e) * N + static_cast<size_t>(n)) * k_packed +
                        static_cast<size_t>(k / 4)];
        const int field = k & 3;
        float w;
        if constexpr (Asym) {
          const float zero = static_cast<float>(zeros[s_idx]);
          const int q = static_cast<int>((packed >> (2 * field)) & 0x3);
          w = (static_cast<float>(q) - zero) * scale;
        } else {
          // Sign-extend 2-bit by shifting the field into the top bits of an int8.
          const int shift = 6 - 2 * field;  // 6, 4, 2, 0 for fields 0..3
          const int8_t s8 = static_cast<int8_t>((packed << shift) & 0xC0);
          const int q = static_cast<int>(s8 >> 6);
          w = static_cast<float>(q) * scale;
        }
        weights_KN[(static_cast<size_t>(e) * K + static_cast<size_t>(k)) * N + static_cast<size_t>(n)] =
            static_cast<ScalarT>(w);
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
                        int N, int K, int group_size) {
  if (E == 0 || N == 0 || K == 0) return;
  const int num_groups_k = K / group_size;

  sycl::range<3> global{static_cast<size_t>(E), static_cast<size_t>(K),
                        static_cast<size_t>((N + WG_N - 1) / WG_N) * WG_N};
  sycl::range<3> local{1, 1, static_cast<size_t>(WG_N)};

  q->parallel_for<MoEDequantKernelFP8<ScalarT, IsE4M3, UseLut>>(
      sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> it) {
        const int e = static_cast<int>(it.get_global_id(0));
        const int k = static_cast<int>(it.get_global_id(1));
        const int n = static_cast<int>(it.get_global_id(2));
        if (n >= N) return;
        const int g = k / group_size;
        const size_t s_idx = (static_cast<size_t>(e) * N + static_cast<size_t>(n)) * num_groups_k +
                             static_cast<size_t>(g);
        const float scale = static_cast<float>(scales[s_idx]);
        const uint8_t raw =
            weights_NK[(static_cast<size_t>(e) * N + static_cast<size_t>(n)) * K + static_cast<size_t>(k)];
        const float w = moe_dequant::decode_fp8<IsE4M3, UseLut>(raw) * scale;
        weights_KN[(static_cast<size_t>(e) * K + static_cast<size_t>(k)) * N + static_cast<size_t>(n)] =
            static_cast<ScalarT>(w);
      });
}

// ----------------------------------------------------------------------------
// Dispatch helper: dequant any supported weight encoding into `weights_KN`
// (already-allocated `[E, K, N]` ScalarT buffer) using ScalarT == act dtype.
// ----------------------------------------------------------------------------
template <typename ScalarT>
void dequant_to_KN(sycl::queue* q, const void* weights, const void* scales, const void* zeros, ScalarT* weights_KN,
                   BTLA_DTYPE weight_dtype, int E, int N, int K, int group_size, bool asym) {
  if (weight_dtype == BTLA_DTYPE::F16 || weight_dtype == BTLA_DTYPE::BF16) {
    launch_dequant_fp<ScalarT>(q, static_cast<const ScalarT*>(weights), weights_KN, E, N, K);
    return;
  }
  if (weight_dtype == BTLA_DTYPE::S8) {
    if (asym) {
      launch_dequant_int8<ScalarT, true>(q, static_cast<const uint8_t*>(weights), static_cast<const ScalarT*>(scales),
                                         static_cast<const ScalarT*>(zeros), weights_KN, E, N, K, group_size);
    } else {
      launch_dequant_int8<ScalarT, false>(q, static_cast<const uint8_t*>(weights), static_cast<const ScalarT*>(scales),
                                          static_cast<const ScalarT*>(zeros), weights_KN, E, N, K, group_size);
    }
    return;
  }
  if (weight_dtype == BTLA_DTYPE::S4_CLIP) {
    if (asym) {
      launch_dequant_int4<ScalarT, true>(q, static_cast<const uint8_t*>(weights), static_cast<const ScalarT*>(scales),
                                         static_cast<const ScalarT*>(zeros), weights_KN, E, N, K, group_size);
    } else {
      launch_dequant_int4<ScalarT, false>(q, static_cast<const uint8_t*>(weights), static_cast<const ScalarT*>(scales),
                                          static_cast<const ScalarT*>(zeros), weights_KN, E, N, K, group_size);
    }
    return;
  }
  if (weight_dtype == BTLA_DTYPE::S2_CLIP) {
    if (asym) {
      launch_dequant_int2<ScalarT, true>(q, static_cast<const uint8_t*>(weights), static_cast<const ScalarT*>(scales),
                                         static_cast<const ScalarT*>(zeros), weights_KN, E, N, K, group_size);
    } else {
      launch_dequant_int2<ScalarT, false>(q, static_cast<const uint8_t*>(weights), static_cast<const ScalarT*>(scales),
                                          static_cast<const ScalarT*>(zeros), weights_KN, E, N, K, group_size);
    }
    return;
  }
  if (weight_dtype == BTLA_DTYPE::F8_E4M3 || weight_dtype == BTLA_DTYPE::F8_E5M2) {
    if (asym) {
      throw std::invalid_argument("moe_gemm_prefill(fp8): asym mode is not supported");
    }
    const bool is_e4m3 = (weight_dtype == BTLA_DTYPE::F8_E4M3);
    const bool use_lut = moe_dequant::fp8_decode_use_lut();
    if (is_e4m3) {
      if (use_lut) {
        launch_dequant_fp8<ScalarT, true, true>(q, static_cast<const uint8_t*>(weights),
                                                static_cast<const ScalarT*>(scales), weights_KN, E, N, K,
                                                group_size);
      } else {
        launch_dequant_fp8<ScalarT, true, false>(q, static_cast<const uint8_t*>(weights),
                                                 static_cast<const ScalarT*>(scales), weights_KN, E, N, K,
                                                 group_size);
      }
    } else {
      if (use_lut) {
        launch_dequant_fp8<ScalarT, false, true>(q, static_cast<const uint8_t*>(weights),
                                                 static_cast<const ScalarT*>(scales), weights_KN, E, N, K,
                                                 group_size);
      } else {
        launch_dequant_fp8<ScalarT, false, false>(q, static_cast<const uint8_t*>(weights),
                                                  static_cast<const ScalarT*>(scales), weights_KN, E, N, K,
                                                  group_size);
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
inline void moe_gemm_prefill(sycl::queue* q, void* activations, void* weights, void* scales, void* zeros,
                             void* outputs, void* dequant_workspace, BTLA_DTYPE act_dtype, BTLA_DTYPE weight_dtype,
                             int N, int K, int group_size, int* num_tokens_per_expert, int num_experts,
                             int total_tokens, bool asym) {
  if (total_tokens == 0) return;

  // Unquantized fast path: forward directly to the existing prefill GEMM.
  // The caller is responsible for passing weights in `[E, K, N]` layout
  // (matching the existing `moe_gemm` contract).
  if (weight_dtype == act_dtype && (weight_dtype == BTLA_DTYPE::F16 || weight_dtype == BTLA_DTYPE::BF16)) {
    moe_gemm(q, activations, weights, scales, outputs, act_dtype, N, K, num_tokens_per_expert, num_experts);
    return;
  }

  if (dequant_workspace == nullptr) {
    throw std::invalid_argument("moe_gemm_prefill: dequant_workspace must be non-null for quantized paths");
  }

  if (act_dtype == BTLA_DTYPE::F16) {
    auto* w_kn = static_cast<sycl::half*>(dequant_workspace);
    moe_mixed_detail::dequant_to_KN<sycl::half>(q, weights, scales, zeros, w_kn, weight_dtype, num_experts, N, K,
                                                group_size, asym);
    moe_gemm(q, activations, w_kn, /*scales=*/nullptr, outputs, act_dtype, N, K, num_tokens_per_expert, num_experts);
  } else if (act_dtype == BTLA_DTYPE::BF16) {
    using BF = sycl::ext::oneapi::bfloat16;
    auto* w_kn = static_cast<BF*>(dequant_workspace);
    moe_mixed_detail::dequant_to_KN<BF>(q, weights, scales, zeros, w_kn, weight_dtype, num_experts, N, K, group_size,
                                        asym);
    moe_gemm(q, activations, w_kn, /*scales=*/nullptr, outputs, act_dtype, N, K, num_tokens_per_expert, num_experts);
  } else {
    throw std::invalid_argument("moe_gemm_prefill: act_dtype must be F16 or BF16");
  }
}

}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
