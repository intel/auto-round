// SYCL MoE Prefill — Native FP8 Fused Grouped-GEMM (Variant A: W8A16)
//
// Motivation
// ----------
// The existing quantized MoE prefill path (`sycl_tla_moe_mixed.hpp`) is a
// two-stage pipeline:
//
//     dequant kernel: [E, N, K] fp8  -> [E, K, N] bf16 workspace
//     grouped GEMM  : A[bf16] x W[bf16] -> D[bf16]
//
// The `[E, K, N]` bf16 workspace is enormous (192 * 3072 * 1536 * 2 bytes ~=
// 1.7 GiB for the perf test's minimax shapes) and every element is a
// global-memory round-trip. The SLM-transposed variant in
// `sycl_tla_moe_prefill_fused.hpp` improves the memory pattern of the
// dequant stage but still writes the same amount of bf16 to DRAM.
//
// This header eliminates the workspace entirely. The kernel loads fp8
// weight bytes and the corresponding per-K-group scale straight from DRAM,
// upcasts fp8 -> float **in registers** using the existing
// `moe_dequant::decode_fp8<>` primitives, folds in the group scale, and
// MACs into a float accumulator. Only the final `act_dtype` output row is
// written back. Numerical results share the same `decode_fp8` primitive
// as the two-stage path, so accumulator-ordering differences are the only
// source of drift versus the current implementation.
//
// Path classification
// -------------------
// This is the Variant A ("W8A16 mixed-input") flavour: activations remain
// in bf16/fp16 as the caller supplies them, only the weight operand is
// fp8. A Variant B ("W8A8" — quantize activations to fp8 online and use a
// true fp8xfp8 DPAS) is a follow-up not covered here.
//
// Kernel implementation
// ---------------------
// This is a Stage-1 correctness-first launcher: a hand-rolled sub-group
// GEMM (one lane per output N column, `BM` output rows per lane, SLM-
// staged activation tile). It intentionally does not sit on top of the
// CuTe / XMX-DPAS machinery used by `moe_gemm` in `sycl_tla_moe.hpp`;
// migrating this kernel onto a mixed-input CuTe MMA atom (so the fp8
// upcast lives inside the DPAS prologue) is the natural follow-up and
// is what will unlock true XMX peak throughput. Even in this Stage-1
// form the kernel already:
//   * eliminates the `E * K * N * sizeof(act_dtype)` global workspace,
//   * halves the DRAM traffic of the weight stream (fp8 byte per K
//     element instead of bf16),
//   * folds the per-group scale into the accumulator with a single
//     scale load per (n, K-group) tile,
//
// which are precisely the wins the two-stage pipeline gives up.
//
// Tile / operand layout
// ---------------------
//   Weight `weights_NK`     [E, N, K]                 uint8 (one fp8 byte per K)
//   Scale  `scales`         [E, N, K/group_size]      ScalarT (act dtype)
//   Activ. `activations`    [total_tokens, K]         ScalarT (act dtype)
//   Output `outputs`        [total_tokens, N]         ScalarT (act dtype)
//   ExpTok `expert_offsets` [E + 1]                   int32 (prefix sum of
//                                                            num_tokens_per_expert;
//                                                            computed on host,
//                                                            copied to USM device)
//
// Preconditions (checked at launch):
//   * `N % BN == 0`
//   * `K % BK == 0`
//   * `K % group_size == 0`
//   * `group_size % BK == 0`     (so per-tile scale is constant along K)
//
// Any shape that fails a precondition transparently falls back to the
// two-stage path via the dispatcher in `sycl_tla_moe_mixed.hpp`.
//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "sycl_tla_moe_dequant.hpp"

#ifdef ARK_XPU
#include <sycl/sycl.hpp>
#endif

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

namespace ark {
namespace moe_prefill_native_fp8_detail {

// ----------------------------------------------------------------------------
// Tile constants.
//
//   SG_SIZE = 16  matches the sub-group size used everywhere else in this
//                  module (see `sycl_tla_moe_prefill_fused.hpp`).
//   BN      = SG_SIZE = 16   (one lane per output N column so the final
//                              store phase is a single coalesced burst).
//   BM      = 8              (each lane holds `BM` float accumulators — one
//                              per output row in the tile).
//   BK      = 32             (K reduction chunk; matches the fused-dequant
//                              tile so the shape predicates are compatible).
//
// SLM footprint per work-group:
//   * BM × BK act tile in ScalarT             -> 8 * 32 * 2 = 512 bytes
//   * (no B staging: each lane reads its own BK weight bytes into regs)
// ----------------------------------------------------------------------------
constexpr int SG_SIZE = 16;
constexpr int BN = SG_SIZE;
constexpr int BM = 8;
constexpr int BK = 32;

// Unique kernel-name tag (SYCL requires one per specialization).
template <typename ScalarT, bool IsE4M3, bool UseLut>
class MoEPrefillFP8NativeKernel;

// ----------------------------------------------------------------------------
// Env-var opt-in (cached; defaults to OFF).
//
//   `ARK_MOE_PREFILL_NATIVE_FP8`
//     - unset / "0" / "false" / "off" / "no"  -> use the (fused|v1) dequant path
//     - "1" / "true" / "on" / "yes"           -> use this native path
//
// Kept independent of `ARK_MOE_PREFILL_FUSED_FP8` so the two opt-in paths
// can be A/B benchmarked without recompiling. Precedence in the dispatcher
// is: native -> fused-dequant -> v1 dequant.
// ----------------------------------------------------------------------------
inline bool moe_prefill_native_fp8_enabled() {
  static const bool value = []() {
    const char* env = std::getenv("ARK_MOE_PREFILL_NATIVE_FP8");
    if (env == nullptr) return false;
    std::string s(env);
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (s == "1" || s == "true" || s == "on" || s == "yes") return true;
    return false;
  }();
  return value;
}

// ----------------------------------------------------------------------------
// Shape predicate — silent fallback (matches the fused-path convention in
// `sycl_tla_moe_prefill_fused.hpp::moe_prefill_fused_fp8_shape_ok`).
// ----------------------------------------------------------------------------
inline bool moe_prefill_native_fp8_shape_ok(int N, int K, int group_size) {
  return (N > 0) && (K > 0) && (group_size > 0) && (N % BN == 0) && (K % BK == 0) && (K % group_size == 0) &&
         (group_size % BK == 0);
}

// ----------------------------------------------------------------------------
// Launcher.
//
// Grid layout:
//   global = (E, max_m_tiles, n_tiles * SG_SIZE)
//   local  = (1, 1,           SG_SIZE)              — 1 sub-group per WG
//
// where
//   max_m_tiles = ceil_div(max_e_tokens, BM)   (host-side max over experts)
//   n_tiles     = N / BN
//
// Work-groups whose (e, m_tile) pair falls beyond that expert's token
// count are early-outed. Choosing `max_m_tiles` per-launch (rather than
// per-expert) keeps the launch to a single `parallel_for` call at the
// cost of a small amount of over-provisioning for uneven token
// distributions; the early-out branch is cheap because it's the first
// instruction the WG executes.
// ----------------------------------------------------------------------------
template <typename ScalarT, bool IsE4M3, bool UseLut>
void launch_moe_prefill_fp8_native(sycl::queue* q, const ScalarT* activations, const uint8_t* weights_NK,
                                   const ScalarT* scales, ScalarT* outputs, const int* expert_offsets, int E, int N,
                                   int K, int group_size, int max_e_tokens) {
  if (E == 0 || N == 0 || K == 0 || max_e_tokens == 0) return;
  if (!moe_prefill_native_fp8_shape_ok(N, K, group_size)) {
    throw std::invalid_argument("moe_gemm_prefill(fp8 native): shape preconditions not met");
  }

  const int num_groups_k = K / group_size;
  const int n_tiles = N / BN;
  const int max_m_tiles = (max_e_tokens + BM - 1) / BM;
  const int k_tiles = K / BK;

  sycl::range<3> global{static_cast<size_t>(E), static_cast<size_t>(max_m_tiles),
                        static_cast<size_t>(n_tiles) * static_cast<size_t>(SG_SIZE)};
  sycl::range<3> local{1, 1, static_cast<size_t>(SG_SIZE)};

  q->submit([&](sycl::handler& cgh) {
    // Cooperative A-tile SLM staging: BM * BK ScalarT elements per WG.
    sycl::local_accessor<ScalarT, 1> a_slm(sycl::range<1>(static_cast<size_t>(BM) * static_cast<size_t>(BK)), cgh);

    cgh.parallel_for<MoEPrefillFP8NativeKernel<ScalarT, IsE4M3, UseLut>>(
        sycl::nd_range<3>(global, local),
        [=](sycl::nd_item<3> it) [[intel::reqd_sub_group_size(SG_SIZE)]] {
          const int e = static_cast<int>(it.get_global_id(0));
          const int m_tile = static_cast<int>(it.get_global_id(1));
          const int n_tile = static_cast<int>(it.get_group(2));
          const int lane = static_cast<int>(it.get_local_id(2));

          // -----------------------------------------------------------------
          // Per-expert bounds.
          //
          // `expert_offsets[e]` is the row in `activations` / `outputs`
          // where expert `e`'s slice begins; `expert_offsets[e+1]` is the
          // end. This kernel processes a `BM`-row tile of that slice.
          // -----------------------------------------------------------------
          const int e_start = expert_offsets[e];
          const int e_end = expert_offsets[e + 1];
          const int e_tokens = e_end - e_start;
          const int base_m_local = m_tile * BM;
          if (base_m_local >= e_tokens) return;  // over-provisioned tile

          const int base_m_global = e_start + base_m_local;
          const int base_n = n_tile * BN;
          const int m_valid = (BM < (e_tokens - base_m_local)) ? BM : (e_tokens - base_m_local);

          // Each lane owns output column (base_n + lane); accumulator holds
          // one float per output row in the BM x BN tile.
          float acc[BM];
#pragma unroll
          for (int m = 0; m < BM; ++m) acc[m] = 0.0f;

          // -----------------------------------------------------------------
          // K-reduction loop.
          //
          // For each BK-wide K chunk:
          //   1. Cooperatively stage the A[BM][BK] tile into SLM (each
          //      lane loads BM elements: `BM * BK / SG_SIZE = 16` slots).
          //   2. Load `BK` weight bytes for this lane's N column and
          //      the tile's expert; decode + fold the group scale into
          //      a per-K float in registers.
          //   3. MAC into `acc[m]` for each of the `BM` output rows.
          //
          // Scale hoist: `BK` divides `group_size`, so all `BK` positions
          // in one tile fall in the same scale group `g = base_k / group_size`.
          // Every lane loads its own N column's scale (one broadcast per
          // K-tile boundary, i.e., per `group_size / BK` inner iterations).
          // -----------------------------------------------------------------
          const size_t w_row_stride = static_cast<size_t>(K);              // [E, N, K] row-major
          const size_t w_expert_stride = static_cast<size_t>(N) * w_row_stride;
          const size_t s_row_stride = static_cast<size_t>(num_groups_k);   // [E, N, K/g]
          const size_t s_expert_stride = static_cast<size_t>(N) * s_row_stride;
          const size_t a_row_stride = static_cast<size_t>(K);              // [total_tokens, K]
          const size_t out_row_stride = static_cast<size_t>(N);            // [total_tokens, N]

          const int n_col = base_n + lane;
          const size_t w_row_base = static_cast<size_t>(e) * w_expert_stride + static_cast<size_t>(n_col) * w_row_stride;
          const size_t s_row_base = static_cast<size_t>(e) * s_expert_stride + static_cast<size_t>(n_col) * s_row_stride;

          for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
            const int base_k = k_tile * BK;
            const int g = base_k / group_size;

            // --------- 1. Cooperative A-tile load into SLM ---------------
            // BM * BK = 8 * 32 = 256 elements across 16 lanes -> 16 per lane.
            // We interleave lanes across the flat index so each SG cycle
            // hits SG_SIZE consecutive addresses in `activations` (which
            // is row-major `[total_tokens, K]` — the fast dim is K).
#pragma unroll
            for (int i = 0; i < (BM * BK) / SG_SIZE; ++i) {
              const int flat = i * SG_SIZE + lane;
              const int m = flat / BK;
              const int k = flat % BK;
              ScalarT a_val = static_cast<ScalarT>(0);
              if (m < m_valid) {
                a_val = activations[(static_cast<size_t>(base_m_global) + static_cast<size_t>(m)) * a_row_stride +
                                    static_cast<size_t>(base_k) + static_cast<size_t>(k)];
              }
              a_slm[static_cast<size_t>(flat)] = a_val;
            }
            it.barrier(sycl::access::fence_space::local_space);

            // --------- 2. Load + decode BK weight bytes for this lane ----
            // and fold in the per-group scale (constant across BK).
            const float scale = static_cast<float>(scales[s_row_base + static_cast<size_t>(g)]);
            float w_col[BK];
#pragma unroll
            for (int k = 0; k < BK; ++k) {
              const uint8_t raw = weights_NK[w_row_base + static_cast<size_t>(base_k) + static_cast<size_t>(k)];
              w_col[k] = moe_dequant::decode_fp8<IsE4M3, UseLut>(raw) * scale;
            }

            // --------- 3. MAC ---------------------------------------------
            // For each output row m in this tile, dot-product the length-BK
            // slice of A[m] (staged in SLM) with `w_col`, accumulate into
            // `acc[m]`.
#pragma unroll
            for (int m = 0; m < BM; ++m) {
              float sum = 0.0f;
#pragma unroll
              for (int k = 0; k < BK; ++k) {
                const float a_f = static_cast<float>(a_slm[static_cast<size_t>(m) * BK + static_cast<size_t>(k)]);
                sum += a_f * w_col[k];
              }
              acc[m] += sum;
            }

            // Barrier before the next iteration re-stages A[] into SLM.
            it.barrier(sycl::access::fence_space::local_space);
          }

          // -----------------------------------------------------------------
          // Store phase.
          //
          // Lane `t` writes `outputs[base_m_global + m, base_n + t]` for
          // `m` in `[0, m_valid)`. Consecutive lanes touch consecutive N
          // -> one coalesced burst per row.
          // -----------------------------------------------------------------
#pragma unroll
          for (int m = 0; m < BM; ++m) {
            if (m >= m_valid) break;
            outputs[(static_cast<size_t>(base_m_global) + static_cast<size_t>(m)) * out_row_stride +
                    static_cast<size_t>(n_col)] = static_cast<ScalarT>(acc[m]);
          }
        });
  });
}

// ----------------------------------------------------------------------------
// Host-side driver.
//
// Handles the per-launch prefix-sum of `num_tokens_per_expert` (small —
// num_experts is O(hundreds)) and the USM allocation of the resulting
// expert-offsets buffer. Blocks on the launched event so the caller can
// treat this as a synchronous kernel dispatch (matching the rest of the
// module and `moe_gemm_launcher`'s `event.wait()`).
// ----------------------------------------------------------------------------
template <typename ScalarT, bool IsE4M3, bool UseLut>
void moe_prefill_fp8_native_dispatch(sycl::queue* q, const ScalarT* activations, const uint8_t* weights_NK,
                                     const ScalarT* scales, ScalarT* outputs, const int* num_tokens_per_expert, int E,
                                     int N, int K, int group_size, int total_tokens) {
  if (E == 0 || N == 0 || K == 0 || total_tokens == 0) return;

  // 1) Copy per-expert token counts to host so we can build a prefix sum
  //    and compute the max (needed for `max_m_tiles` in the grid range).
  std::vector<int> h_ntpe(static_cast<size_t>(E));
  q->memcpy(h_ntpe.data(), num_tokens_per_expert, static_cast<size_t>(E) * sizeof(int)).wait();

  std::vector<int> h_offsets(static_cast<size_t>(E) + 1, 0);
  int max_e_tokens = 0;
  for (int e = 0; e < E; ++e) {
    h_offsets[static_cast<size_t>(e) + 1] = h_offsets[static_cast<size_t>(e)] + h_ntpe[static_cast<size_t>(e)];
    if (h_ntpe[static_cast<size_t>(e)] > max_e_tokens) max_e_tokens = h_ntpe[static_cast<size_t>(e)];
  }
  if (max_e_tokens == 0) return;  // all experts empty

  // 2) Push offsets to a USM device buffer (freed at end of dispatch).
  int* d_offsets = sycl::malloc_device<int>(static_cast<size_t>(E) + 1, *q);
  if (d_offsets == nullptr) {
    throw std::runtime_error("moe_gemm_prefill(fp8 native): failed to allocate USM offsets buffer");
  }
  q->memcpy(d_offsets, h_offsets.data(), (static_cast<size_t>(E) + 1) * sizeof(int)).wait();

  // 3) Launch. The launcher submits + returns; we wait on the queue below
  //    before freeing the offsets buffer so the kernel can safely read it.
  launch_moe_prefill_fp8_native<ScalarT, IsE4M3, UseLut>(q, activations, weights_NK, scales, outputs, d_offsets, E, N,
                                                         K, group_size, max_e_tokens);
  q->wait();

  sycl::free(d_offsets, *q);
}

}  // namespace moe_prefill_native_fp8_detail
}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
