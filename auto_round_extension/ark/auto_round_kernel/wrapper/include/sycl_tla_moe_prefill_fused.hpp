// SYCL MoE Prefill — SLM-Transposed Weight Dequantization Kernels (PR-B1)
//
// Path-A staged rewrite of the MoE prefill dequantization stage. The v1
// implementation in `sycl_tla_moe_mixed.hpp` writes the temporary
// `[E, K, N]` fp16/bf16 workspace with each work-item owning a fixed
// (e, n) pair and looping across K. That layout gives coalesced *stores*
// (32 lanes -> 32 consecutive N) but **strided loads**: every lane pulls
// from a different row of `weights_NK[E, N, K]`, so a single sub-group
// cycle triggers up to 32 independent cache-line fetches with only 1–4
// bytes actually consumed each (16×–64× read BW waste depending on
// PACK_K).
//
// This header introduces a new opt-in kernel path that swaps the tile so
// **both** loads and stores are sub-group coalesced:
//
//   - Grid  : (E, N/BN, K/BK) with WG = 1 SG (SG_SIZE lanes)
//   - Load  : lane `t` reads `weights_NK[e, base_n+i, base_k+t]` and
//             `weights_NK[e, base_n+i, base_k+SG_SIZE+t]` for BK = 32.
//             Consecutive lanes touch consecutive K bytes -> one
//             coalesced 16-byte sub-group load per row per K-half.
//   - Xpose : dequant in registers, transpose-store into SLM. The SLM
//             tile is padded (`bf16 slm[BN][BK+1]`) so the subsequent
//             lane-strided read is bank-conflict-free (address stride
//             `BK+1 = 33` is coprime with the 32 SLM banks).
//   - Store : lane `t` writes `weights_KN[e, base_k+it, base_n+t]` for
//             `it` in `[0, BK)`. All 16 lanes at the same `it` hit
//             consecutive N -> one coalesced 32-byte bf16 store per
//             iteration.
//
// Numerical behaviour: identical to v1 (both paths call the same
// `moe_dequant::decode_fp8<IsE4M3, UseLut>` primitive from
// `sycl_tla_moe_dequant.hpp`). The generated `[E, K, N]` workspace is
// consumed by the same downstream `moe_gemm` call, so `test_moe_prefill_
// accuracy.py` and `test_moe_unified.py` must remain bit-identical.
//
// Scope
// -----
// This commit (PR-B1 / Commit 2) implements only fp8-e4m3 with bf16
// activations. The dispatcher in `sycl_tla_moe_mixed.hpp` selects the
// new path when `ARK_MOE_PREFILL_FUSED_FP8=1` is set in the environment;
// otherwise the legacy v1 path is used unchanged. Follow-up commits
// (PR-B2..B5) extend the same tile design to fp8-e5m2, INT8/INT4/INT2
// (sym & asym).
//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "sycl_tla_moe_dequant.hpp"

#ifdef ARK_XPU
#include <sycl/sycl.hpp>
#endif

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

namespace ark {
namespace moe_prefill_fused_detail {

// ----------------------------------------------------------------------------
// Tile / sub-group constants.
//
// SG_SIZE matches the value hard-coded in the decode path
// (`moe_decode_detail::SG_SIZE` in `sycl_tla_moe_decode.hpp`). Keeping the
// same sub-group size across kernels lets the driver reuse the same
// work-group pool.
//
// BN and BK are chosen so a single work-group covers a (BN x BK) tile of
// the input weight matrix:
//   - BK = 2 * SG_SIZE  (so lanes 0..15 fill K positions [0..15] in one
//                        sub-group read and [16..31] in the second)
//   - BN = SG_SIZE      (so the output store phase writes 16 lanes to 16
//                        consecutive N in a single 32-byte bf16 burst)
//
// The SLM tile is `[BN][BK+1]` to add a 2-byte padding column: the
// subsequent lane-strided read (`slm[t][it]`) accesses address
// `t * (BK+1) + it`. With `BK+1 = 33` coprime with the 32 SLM banks,
// consecutive lanes at the same `it` map to distinct banks so the read
// completes in a single cycle without conflicts.
// ----------------------------------------------------------------------------
constexpr int SG_SIZE = 16;
constexpr int BN = SG_SIZE;
constexpr int BK = 2 * SG_SIZE;
constexpr int SLM_BK_STRIDE = BK + 1;

// Unique kernel name tag per specialization (required for SYCL kernel naming).
template <typename ScalarT, bool IsE4M3, bool UseLut>
class MoEDequantFusedKernelFP8;

// ----------------------------------------------------------------------------
// FP8 (E4M3 / E5M2) SLM-transposed dequant.
//
//   Input : `weights_NK` [E, N, K]                       (uint8, one fp8 byte per K)
//           `scales`     [E, N, K/group_size]            (ScalarT)
//   Output: `weights_KN` [E, K, N]                       (ScalarT)
//
// Layout / stride conventions match v1 (`launch_dequant_fp8` in
// `sycl_tla_moe_mixed.hpp`). Callers pre-allocate `weights_KN` (a
// per-launch workspace); this kernel writes every element exactly once.
//
// Preconditions (enforced at the launcher):
//   - N is a multiple of BN.
//   - K is a multiple of BK and of group_size.
//   - BK divides group_size (so the per-group scale is constant across a
//     tile's K span; each tile issues one scale load per row).
//
// Rationale for the tile split:
//   - `weights_NK` is [E, N, K] row-major, so reads are stride-1 along K.
//     Mapping lane `t` -> K position `t` makes the SG load coalesce into
//     a single 16-byte burst per row.
//   - `weights_KN` is [E, K, N] row-major, so writes are stride-1 along N.
//     Mapping lane `t` -> N position `t` in the store phase gives
//     coalesced 32-byte bf16 stores.
//   - The SLM transpose bridges the two: each output row (K slice) needs
//     values from `BN` different input rows (N slice), which no single
//     lane can materialise on its own -> cross-lane exchange via SLM.
// ----------------------------------------------------------------------------
template <typename ScalarT, bool IsE4M3, bool UseLut>
void launch_dequant_fp8_slm(sycl::queue* q, const uint8_t* weights_NK, const ScalarT* scales, ScalarT* weights_KN,
                            int E, int N, int K, int group_size,
                            const int* num_tokens_per_expert = nullptr) {
  if (E == 0 || N == 0 || K == 0) return;
  if (N % BN != 0) {
    throw std::invalid_argument("moe_gemm_prefill(fp8 fused): N must be a multiple of BN (=16)");
  }
  if (K % BK != 0) {
    throw std::invalid_argument("moe_gemm_prefill(fp8 fused): K must be a multiple of BK (=32)");
  }
  if (group_size % BK != 0) {
    throw std::invalid_argument("moe_gemm_prefill(fp8 fused): group_size must be a multiple of BK (=32)");
  }
  if (K % group_size != 0) {
    throw std::invalid_argument("moe_gemm_prefill(fp8 fused): K must be a multiple of group_size");
  }

  const int num_groups_k = K / group_size;
  const int n_tiles = N / BN;
  const int k_tiles = K / BK;

  // Global range: (E, n_tiles, k_tiles * SG_SIZE). The innermost dimension
  // is expanded by SG_SIZE so the local range can pin one work-group to a
  // full sub-group of SG_SIZE lanes.
  sycl::range<3> global{static_cast<size_t>(E), static_cast<size_t>(n_tiles),
                        static_cast<size_t>(k_tiles) * static_cast<size_t>(SG_SIZE)};
  sycl::range<3> local{1, 1, static_cast<size_t>(SG_SIZE)};

  q->submit([&](sycl::handler& cgh) {
    // SLM: bf16 tile[BN][BK+1] (padded to avoid bank conflicts on the
    // lane-strided read in the store phase). See constants above.
    sycl::local_accessor<ScalarT, 1> slm(sycl::range<1>(static_cast<size_t>(BN) * SLM_BK_STRIDE), cgh);

    cgh.parallel_for<MoEDequantFusedKernelFP8<ScalarT, IsE4M3, UseLut>>(
        sycl::nd_range<3>(global, local),
        [=](sycl::nd_item<3> it) [[intel::reqd_sub_group_size(SG_SIZE)]] {
          const int e = static_cast<int>(it.get_global_id(0));
          // Skip experts that receive no tokens in this prefill batch;
          // the downstream Grouped GEMM won't consume their rows.
          if (num_tokens_per_expert != nullptr && num_tokens_per_expert[e] == 0) return;

          const int n_tile = static_cast<int>(it.get_global_id(1));
          const int k_tile = static_cast<int>(it.get_group(2));
          const int lane = static_cast<int>(it.get_local_id(2));

          const int base_n = n_tile * BN;
          const int base_k = k_tile * BK;

          // ------------------------------------------------------------------
          // Load + dequant + transpose-store into SLM.
          //
          // Iteration i covers input row (base_n + i). The row spans K
          // positions [base_k, base_k + BK). We split the row into two
          // SG-coalesced halves (kk = 0 or SG_SIZE) and let lane `t`
          // handle K = base_k + kk + t.
          //
          // Scale hoist: BK divides group_size, so all BK K-positions in
          // this tile fall in the same scale group g = base_k / group_size.
          // Every lane loads the same scale value from the same address
          // (a single broadcast); the compiler folds this into one
          // memory op per iteration.
          // ------------------------------------------------------------------
          const int g = base_k / group_size;
#pragma unroll
          for (int i = 0; i < BN; ++i) {
            const int n_row = base_n + i;
            const size_t w_row_base = (static_cast<size_t>(e) * N + static_cast<size_t>(n_row)) * K;
            const size_t s_idx =
                (static_cast<size_t>(e) * N + static_cast<size_t>(n_row)) * num_groups_k + static_cast<size_t>(g);
            const float scale = static_cast<float>(scales[s_idx]);
#pragma unroll
            for (int kk = 0; kk < BK; kk += SG_SIZE) {
              const int k = base_k + kk + lane;
              const uint8_t raw = weights_NK[w_row_base + static_cast<size_t>(k)];
              const float w = moe_dequant::decode_fp8<IsE4M3, UseLut>(raw) * scale;
              // slm[i][kk + lane] with row stride SLM_BK_STRIDE.
              slm[static_cast<size_t>(i) * SLM_BK_STRIDE + static_cast<size_t>(kk) +
                  static_cast<size_t>(lane)] = static_cast<ScalarT>(w);
            }
          }

          // Publish the tile to the store phase.
          it.barrier(sycl::access::fence_space::local_space);

          // ------------------------------------------------------------------
          // Store phase: lane `t` owns output N-column (base_n + t) and
          // walks its K rows [base_k, base_k + BK). Reading `slm[t][it_k]`
          // maps to address `t * SLM_BK_STRIDE + it_k`; with
          // SLM_BK_STRIDE = 33 (coprime with 32 banks) the sixteen lanes
          // access 16 distinct banks per iteration -> no bank conflicts.
          //
          // Writes go to `weights_KN[e, base_k + it_k, base_n + t]`:
          // consecutive lanes touch consecutive N -> one coalesced
          // 32-byte bf16 burst per iteration.
          // ------------------------------------------------------------------
          const size_t out_n_col = static_cast<size_t>(base_n) + static_cast<size_t>(lane);
          const size_t out_base_kbytes = static_cast<size_t>(e) * static_cast<size_t>(K) * static_cast<size_t>(N);
#pragma unroll
          for (int it_k = 0; it_k < BK; ++it_k) {
            const ScalarT val =
                slm[static_cast<size_t>(lane) * SLM_BK_STRIDE + static_cast<size_t>(it_k)];
            const size_t out_off =
                out_base_kbytes + (static_cast<size_t>(base_k) + static_cast<size_t>(it_k)) * static_cast<size_t>(N) +
                out_n_col;
            weights_KN[out_off] = val;
          }
        });
  });
}

// ----------------------------------------------------------------------------
// Env-var opt-in: cached, defaults to OFF.
//
//   `ARK_MOE_PREFILL_FUSED_FP8`:
//     - unset / "0" / "false" / "off" / "no" (default) -> v1 path
//     - "1" / "true" / "on" / "yes"                    -> new SLM path
//
// Re-read on every call (host-side cold path) so benchmarks / tests can
// toggle the path in-process. Was previously a function-local static,
// which prevented `test_moe_prefill_perf.py` from independently measuring
// v1 vs fused after the first FP8 launch had frozen the value.
//
// The opt-in default (OFF) is deliberate: the fused path only covers
// fp8 + bf16 in this commit, and BMG performance verification is still
// pending. Landing the new kernel behind a flag lets end-to-end MoE
// runs continue to exercise the unchanged v1 path while allowing
// benchmark suites (`test_moe_prefill_perf.py`) to A/B-compare v1 vs v2
// by toggling a single env var.
// ----------------------------------------------------------------------------
inline bool moe_prefill_fused_fp8_enabled() {
  const char* env = std::getenv("ARK_MOE_PREFILL_FUSED_FP8");
  if (env == nullptr) return false;
  std::string s(env);
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "1" || s == "true" || s == "on" || s == "yes") return true;
  return false;
}

// ----------------------------------------------------------------------------
// Precondition helper: the SLM-fused kernel only applies to a subset of
// (N, K, group_size) shapes. When the flag is on but the shape isn't
// covered we transparently fall back to v1 rather than throwing, so
// enabling the flag globally in a mixed workload is safe.
// ----------------------------------------------------------------------------
inline bool moe_prefill_fused_fp8_shape_ok(int N, int K, int group_size) {
  return (N > 0) && (K > 0) && (group_size > 0) && (N % BN == 0) && (K % BK == 0) && (K % group_size == 0) &&
         (group_size % BK == 0);
}

}  // namespace moe_prefill_fused_detail
}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
