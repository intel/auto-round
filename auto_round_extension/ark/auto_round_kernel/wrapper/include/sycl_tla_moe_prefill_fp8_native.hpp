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
// K-loop structure (Phase 1 optimization)
// ---------------------------------------
// The K reduction is organised as a two-level nest:
//
//   for g in [0, K/group_size):          // scale group; barrier + scale reload
//     stage A[BM][group_size] into SLM   // ONE cooperative load + barrier
//     scale = scales[e, n_col, g]        // per-lane scale, loaded ONCE
//     for sub in [0, group_size/BK):     // BK sub-tile inside the group
//       load BK fp8 bytes for this lane  // 4-byte chunked, unrolled
//       w_col[k] = decode_fp8(byte) * scale
//       acc[m]  += sum_k a_slm[m, sub*BK+k] * w_col[k]
//
// vs. the original one-level loop that reloaded A + issued a barrier
// once per BK-wide K-tile and reloaded the scale on every iteration.
// The default group_size = 128 (BK = 32) drops the per-WG barrier count
// by 4x and the per-WG global scale-load count by 4x with no change to
// the per-BK partial-sum accumulation order (numerics preserved within
// the FP8 accuracy tolerance of 7e-2 in `test_moe_prefill_accuracy.py`).
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
//   * BM x group_size ScalarT elements  (A tile staged once per scale group)
//     = 8 * 128 * 2 = 2 KiB for the default group_size = 128 / bf16 acts.
//     Scales with group_size up to a few KiB of SLM; well below the per-WG
//     SLM budget on BMG/PVC.
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
// Env-var opt-in (defaults to OFF).
//
//   `ARK_MOE_PREFILL_NATIVE_FP8`
//     - unset / "0" / "false" / "off" / "no"  -> use the (fused|v1) dequant path
//     - "1" / "true" / "on" / "yes"           -> use this native path
//
// Re-read on every call (host-side cold path) so benchmarks / tests can
// toggle the path in-process. Kept independent of `ARK_MOE_PREFILL_FUSED_FP8`
// so the two opt-in paths can be A/B benchmarked without recompiling.
// Precedence in the dispatcher is: dpas -> native -> fused-dequant -> v1 dequant.
// ----------------------------------------------------------------------------
inline bool moe_prefill_native_fp8_enabled() {
  const char* env = std::getenv("ARK_MOE_PREFILL_NATIVE_FP8");
  if (env == nullptr) return false;
  std::string s(env);
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "1" || s == "true" || s == "on" || s == "yes") return true;
  return false;
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
sycl::event launch_moe_prefill_fp8_native(sycl::queue* q, const ScalarT* activations, const uint8_t* weights_NK,
                                          const ScalarT* scales, ScalarT* outputs, const int* expert_offsets, int E,
                                          int N, int K, int group_size, int max_e_tokens) {
  if (E == 0 || N == 0 || K == 0 || max_e_tokens == 0) return sycl::event{};
  if (!moe_prefill_native_fp8_shape_ok(N, K, group_size)) {
    throw std::invalid_argument("moe_gemm_prefill(fp8 native): shape preconditions not met");
  }

  const int num_groups_k = K / group_size;
  const int n_tiles = N / BN;
  const int max_m_tiles = (max_e_tokens + BM - 1) / BM;
  // Per-scale-group inner sub-tile count. `group_size % BK == 0` is a
  // launch precondition so this divides evenly.
  const int gs_per_tile = group_size / BK;
  const int stage_k = group_size;  // A tile is BM x stage_k, staged once per group.

  sycl::range<3> global{static_cast<size_t>(E), static_cast<size_t>(max_m_tiles),
                        static_cast<size_t>(n_tiles) * static_cast<size_t>(SG_SIZE)};
  sycl::range<3> local{1, 1, static_cast<size_t>(SG_SIZE)};

  return q->submit([&](sycl::handler& cgh) {
    // Cooperative A-tile SLM staging: BM x stage_k ScalarT elements per WG.
    // Sized at submit time; stage_k == group_size is a kernel argument.
    sycl::local_accessor<ScalarT, 1> a_slm(sycl::range<1>(static_cast<size_t>(BM) * static_cast<size_t>(stage_k)), cgh);

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
          // K-reduction loop (Phase 1 layout).
          //
          // Outer loop iterates once per scale group `g in [0, K/group_size)`:
          //   1. Cooperatively stage the [BM][stage_k] A tile into SLM.
          //      ONE barrier per scale group (vs. one per BK-tile before).
          //   2. Load this lane's scale ONCE per group (vs. once per BK-tile).
          //   3. Inner loop over `sub in [0, group_size/BK)` runs BK-wide
          //      sub-tiles fully from SLM + registers with no extra barrier:
          //        a. Fetch BK fp8 weight bytes (4-byte chunked, unrolled).
          //        b. Decode + fold scale in registers into w_col[BK].
          //        c. MAC into acc[m] for each of the BM output rows.
          //
          // Per-BK partial-sum accumulation order is preserved bit-for-bit,
          // so the FP8 parity tests (7e-2 tolerance) remain valid.
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

          for (int g = 0; g < num_groups_k; ++g) {
            const int base_gk = g * stage_k;

            // --------- 1. Cooperative A-tile load into SLM --------------
            // Load BM * stage_k ScalarT elements across SG_SIZE lanes with
            // a stride of SG_SIZE, so within one SG cycle the lanes touch
            // SG_SIZE consecutive addresses in `activations` (K is the fast
            // dim of the row-major [total_tokens, K] layout).
            const int a_total = BM * stage_k;
            for (int i = lane; i < a_total; i += SG_SIZE) {
              const int m = i / stage_k;
              const int k = i % stage_k;
              ScalarT a_val = static_cast<ScalarT>(0);
              if (m < m_valid) {
                a_val = activations[(static_cast<size_t>(base_m_global) + static_cast<size_t>(m)) * a_row_stride +
                                    static_cast<size_t>(base_gk) + static_cast<size_t>(k)];
              }
              a_slm[static_cast<size_t>(i)] = a_val;
            }
            it.barrier(sycl::access::fence_space::local_space);

            // --------- 2. Hoisted per-lane scale load --------------------
            // One global load per lane per scale group (vs. one per BK-tile
            // in the prior revision).
            const float scale = static_cast<float>(scales[s_row_base + static_cast<size_t>(g)]);

            // --------- 3. Inner BK-sub-tile loop -------------------------
            for (int sub = 0; sub < gs_per_tile; ++sub) {
              const int base_k_sub = sub * BK;

              // 3a. Load BK weight bytes for this lane's column, vectorized
              //     as (BK/4) x uint32_t chunks. The pointer is 4-byte
              //     aligned: `weights_NK` is 4-byte aligned (tensor storage)
              //     and the offset `w_row_base + base_gk + base_k_sub` is a
              //     multiple of BK = 32 (K % BK == 0, base_gk multiple of
              //     group_size which is a multiple of BK).
              const size_t w_off = w_row_base + static_cast<size_t>(base_gk) + static_cast<size_t>(base_k_sub);
              const uint32_t* w_u32 =
                  reinterpret_cast<const uint32_t*>(weights_NK + w_off);
              float w_col[BK];
              constexpr int W_WORDS = BK / 4;  // 8 for BK = 32
#pragma unroll
              for (int wi = 0; wi < W_WORDS; ++wi) {
                const uint32_t w = w_u32[wi];
                const uint8_t b0 = static_cast<uint8_t>(w & 0xFFu);
                const uint8_t b1 = static_cast<uint8_t>((w >> 8) & 0xFFu);
                const uint8_t b2 = static_cast<uint8_t>((w >> 16) & 0xFFu);
                const uint8_t b3 = static_cast<uint8_t>((w >> 24) & 0xFFu);
                w_col[wi * 4 + 0] = moe_dequant::decode_fp8<IsE4M3, UseLut>(b0) * scale;
                w_col[wi * 4 + 1] = moe_dequant::decode_fp8<IsE4M3, UseLut>(b1) * scale;
                w_col[wi * 4 + 2] = moe_dequant::decode_fp8<IsE4M3, UseLut>(b2) * scale;
                w_col[wi * 4 + 3] = moe_dequant::decode_fp8<IsE4M3, UseLut>(b3) * scale;
              }

              // 3b. MAC. For each output row m in this tile, dot-product
              //     the length-BK slice of A[m] (staged in SLM) with
              //     `w_col`, accumulate into `acc[m]`. Same per-BK partial-
              //     sum shape as the original kernel to preserve numerics.
              const size_t a_col_base = static_cast<size_t>(base_k_sub);
#pragma unroll
              for (int m = 0; m < BM; ++m) {
                float sum = 0.0f;
#pragma unroll
                for (int k = 0; k < BK; ++k) {
                  const size_t a_off =
                      static_cast<size_t>(m) * static_cast<size_t>(stage_k) + a_col_base + static_cast<size_t>(k);
                  const float a_f = static_cast<float>(a_slm[a_off]);
                  sum += a_f * w_col[k];
                }
                acc[m] += sum;
              }
            }

            // Barrier before the next scale group re-stages A[].
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
// expert-offsets buffer. Returns *asynchronously*: the kernel event is
// captured and used to chain an async `sycl::free` of the offsets buffer
// via a `host_task` (Phase 1 optimization — eliminates the trailing
// `q->wait()` that the initial revision needed to safely release the
// USM buffer). Ordering with respect to subsequent ops on the same XPU
// queue is preserved through the queue itself, matching how the rest of
// the module composes with the PyTorch XPU stream.
// ----------------------------------------------------------------------------
template <typename ScalarT, bool IsE4M3, bool UseLut>
void moe_prefill_fp8_native_dispatch(sycl::queue* q, const ScalarT* activations, const uint8_t* weights_NK,
                                     const ScalarT* scales, ScalarT* outputs, const int* num_tokens_per_expert, int E,
                                     int N, int K, int group_size, int total_tokens) {
  if (E == 0 || N == 0 || K == 0 || total_tokens == 0) return;

  // 1) Copy per-expert token counts to host so we can build a prefix sum
  //    and compute the max (needed for `max_m_tiles` in the grid range).
  //    This `.wait()` is mandatory: we read `h_ntpe` on the host below.
  std::vector<int> h_ntpe(static_cast<size_t>(E));
  q->memcpy(h_ntpe.data(), num_tokens_per_expert, static_cast<size_t>(E) * sizeof(int)).wait();

  std::vector<int> h_offsets(static_cast<size_t>(E) + 1, 0);
  int max_e_tokens = 0;
  for (int e = 0; e < E; ++e) {
    h_offsets[static_cast<size_t>(e) + 1] = h_offsets[static_cast<size_t>(e)] + h_ntpe[static_cast<size_t>(e)];
    if (h_ntpe[static_cast<size_t>(e)] > max_e_tokens) max_e_tokens = h_ntpe[static_cast<size_t>(e)];
  }
  if (max_e_tokens == 0) return;  // all experts empty

  // 2) Push offsets to a USM device buffer. This `.wait()` is kept: the
  //    source (`h_offsets`) is a stack `std::vector` that dies when this
  //    function returns, and dropping the host block here would race with
  //    the async H2D copy. The async release below still buys us the
  //    tail `q->wait()` savings (dominant on repeated benchmark calls).
  int* d_offsets = sycl::malloc_device<int>(static_cast<size_t>(E) + 1, *q);
  if (d_offsets == nullptr) {
    throw std::runtime_error("moe_gemm_prefill(fp8 native): failed to allocate USM offsets buffer");
  }
  q->memcpy(d_offsets, h_offsets.data(), (static_cast<size_t>(E) + 1) * sizeof(int)).wait();

  // 3) Launch. The launcher submits and returns the kernel event; ordering
  //    with the H2D copy above is established by the (in-order) XPU queue.
  sycl::event kernel_evt = launch_moe_prefill_fp8_native<ScalarT, IsE4M3, UseLut>(
      q, activations, weights_NK, scales, outputs, d_offsets, E, N, K, group_size, max_e_tokens);

  // 4) Async release of `d_offsets`. A `host_task` submission chained on
  //    the kernel event frees the USM buffer after the kernel has fully
  //    consumed it, without blocking the host. Subsequent ops on the
  //    same queue observe the kernel result via queue ordering, so the
  //    caller-visible semantics match the previous synchronous dispatch
  //    modulo the host-side wait latency we're removing.
  q->submit([q, d_offsets, kernel_evt](sycl::handler& cgh) {
    cgh.depends_on(kernel_evt);
    cgh.host_task([q, d_offsets]() { sycl::free(d_offsets, *q); });
  });
}

}  // namespace moe_prefill_native_fp8_detail
}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
