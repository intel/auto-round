// SYCL MoE Prefill — INT8 mixed-input DPAS Grouped GEMM (Variants A & B)
// (Mirrors `sycl_tla_moe_prefill_fp8_dpas.hpp`)
//
// STATUS: NEEDS-HARDWARE-VALIDATION -- this header is a mode-2 port
// ---------------------------------------------------------------------------
// This file adapts the FP8 DPAS grouped-GEMM path
// (`sycl_tla_moe_prefill_fp8_dpas.hpp`) to INT8 **symmetric** weight
// storage. The design intent is deliberately narrow: **weights are stored
// as one signed byte per element (`int8_t`)**, **activations stay
// FP16/BF16**, and the DPAS atom keeps its `bf16/fp16 × bf16/fp16 → fp32`
// shape. The mainloop upcasts each INT8 weight byte to the activation
// dtype in-register (via `cute::reorder(tBrB, tCrB)`, which delegates to
// `cutlass::NumericArrayConverter` for the int→float leg), then feeds the
// upcast fragment into the same DPAS MMA as FP8 / bf16. The per-expert /
// per-group scale is folded either once per output element in the epilogue
// (per-tensor) or once per K-group in the deferred fold (per-group),
// exactly as the FP8 header's `xe_gemm<>` / `xe_gemm_fp8_pergroup<>` do.
//
// **Asym is intentionally not supported here.** The initial asym port
// used a per-M-row per-K-group activation-sum precompute to fold
// `Σ_g s · (Σ w·a − z · Σ a)` at each group boundary, but the on-hardware
// perf tail (extra global load per group boundary + a dedicated asum
// prepass) regressed the fused path below the dequant fallback. Asym S8
// callers fall through to the generic dequant + `moe_gemm` path in
// `sycl_tla_moe_mixed.hpp` (which handles zero-point folding at dequant
// time and stays competitive).
//
// This is Path A of the discussion attached to the auto-round issue that
// introduced this header: "INT8 storage + bf16/fp16 DPAS compute". It is
// also the compute kernel that the INT4-sym / INT2-sym prefill paths
// call after an on-device upcast to INT8-sym: those callers materialise a
// `[E, N, K]` `int8_t` view of the packed nibble/crumb weights into the
// dequant workspace and hand it to `moe_prefill_int_dpas_per_group_dispatch`
// unchanged.
//
// Path taxonomy
// -------------
// This header exposes TWO fused INT8 grouped-GEMM entry points, matching
// the FP8 header's Variants A / B:
//
//   * Variant A (int8, per-tensor) -- weight scales are `[E]` (one FP32
//     scalar per expert). Weight layout `[E, K, N]` row-major `int8_t`
//     (vllm-xpu-kernels convention, transposed relative to auto-round's
//     default per-group `[E, N, K]` int8 layout so callers that already
//     produce `[E, K, N]` FP8 weights need only swap dtype). Epilogue is
//     `tCrC(i) *= Scales[expert_id]`.
//
//   * Variant B (int8, per-K-group, sym) -- weight scales are
//     `[E, N, K/GS]` stored in the activation dtype (half / bfloat16),
//     same layout as the FP8 per-group path so the scale-load logic is
//     bit-identical. Weights are `[E, N, K]` row-major `int8_t`
//     (auto-round's default INT8 sym layout, stored as `uint8_t` on the
//     Python side and reinterpret-cast on entry). The launcher passes
//     `LayoutKindB='C'` so `MoEGEMM_int<>` XOR-flips to `'R'` inside
//     `make_moe_tensor`. Sym only -- see `xe_gemm_int_pergroup<>`.
//
// The scheduler / launcher are forked here (rather than re-templating the
// FP8 header) to keep the FP8 hot path untouched while this INT8 branch
// stabilises on hardware; once verified, the two files can be unified
// under a single `MoEGEMM<... MainloopKind>` template if desired.
//
// On-hardware open questions (must be resolved on first build):
//   1. Whether `cute::reorder(tBrB, tCrB)` upcasts `int8_t` -> `act_dtype`
//      (bfloat16_t / half_t). `cutlass::NumericArrayConverter<half_t,
//      int8_t, ...>` and `<bfloat16_t, int8_t, ...>` both exist upstream
//      as of cutlass 3.5+, and `cute::reorder` dispatches through
//      `NumericArrayConverter`; but the version of cutlass-sycl pinned by
//      auto-round may need a spot-check.
//   2. Whether `XE_DPAS_TT<8, float, ElementA, ElementA>` (the atom used
//      by the FP8 header's launcher, keeping A/B as the SAME activation
//      dtype after the upcast) is still the right atom here. It should
//      be -- once `reorder` has upcast the INT8 fragment to `ElementA`
//      the atom's operand dtypes match A exactly. If a future cutlass-sycl
//      requires the atom's operand dtype to reflect the STORAGE dtype
//      pre-reorder, adjust here.
//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#ifdef ARK_XPU
#include <sycl/sycl.hpp>
#endif

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

// The FP8 DPAS header pulls in the full cutlass-sycl / CuTe include set and
// defines the policy classes (`dpas_w8a16_policy*`), scalar-type mapping
// (`cute_scalar`), and per-tensor `apply_scale` primitive we can reuse
// verbatim. Include it first so we inherit all of that machinery; the
// INT8 driver lives in a sibling namespace to avoid ODR clashes.
#include "sycl_tla_moe_prefill_fp8_dpas.hpp"

namespace ark {
namespace moe_dpas_int {

using namespace cute;

// Re-export the FP8 header's policy classes into the INT8 namespace so
// callsites read cleanly. The classes carry no per-dtype state; they only
// describe the WG/SG tile shape and the D-store copy atom.
using ::ark::moe_dpas_fp8::dpas_policy_base;
using ::ark::moe_dpas_fp8::dpas_w16a16_policy;
using ::ark::moe_dpas_fp8::dpas_w8a16_policy;
using ::ark::moe_dpas_fp8::dpas_w8a16_policy_m_16;
using ::ark::moe_dpas_fp8::dpas_w8a16_policy_m_32;
using ::ark::moe_dpas_fp8::ScaleMode;
using ::ark::moe_dpas_fp8::cute_scalar;
using ::ark::moe_dpas_fp8::cute_scalar_t;
using ::ark::moe_dpas_fp8::make_moe_tensor;

// ---------------------------------------------------------------------------
// Variant A -- per-tensor INT8 mainloop.
//
// Adapted from `moe_dpas_fp8::xe_gemm<>`. Structural differences:
//
//   * `ElementB` is required to be `int8_t` (sym INT8 storage). The upcast
//     from `int8_t` to `ElementA` (bf16/fp16) happens inside
//     `reorder(tBrB, tCrB)`, which delegates to
//     `cutlass::NumericArrayConverter<ElementA, int8_t, N>` -- this is
//     the SAME machinery `moe_dpas_fp8::xe_gemm<>` relies on for FP8.
//   * `is_B_fp8_type` (the FP8 header's static gate on the epilogue
//     scale multiply) is replaced with `is_B_int8_type`. The epilogue
//     `tCrC(i) *= B_scale` fires unconditionally for INT8 weights (there
//     is no un-scaled INT8 GEMM in this kernel -- the per-tensor scale
//     is always meaningful).
//   * Bias handling is preserved verbatim (Bias is currently unused by
//     the Python entry point but the parameter is retained for
//     interface symmetry with the FP8 path).
// ---------------------------------------------------------------------------

template <class GmemTiledCopyA, class GmemTiledCopyB, class GmemTiledCopyC,
          class ATensor, class BTensor, class DTensor, class TiledMMA,
          typename ElementS, typename ElementBI>
CUTE_DEVICE void xe_gemm_int_pertensor(ATensor const& A,   // (M,K)
                                       BTensor const& B,   // (N,K)
                                       const ElementS* Scales,
                                       const ElementBI* Bias,
                                       DTensor& C,         // (M,N)
                                       Coord<int, int, cute::Underscore, int> blk_coord,
                                       TiledMMA const& mma) {
  using TA = typename ATensor::element_type;
  using TB = typename BTensor::element_type;
  static_assert(std::is_same_v<TB, int8_t>,
                "xe_gemm_int_pertensor: ElementB must be int8_t");

  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto wg_m = get<0>(blk_coord);
  auto wg_n = get<1>(blk_coord);
  int local_id = item.get_local_linear_id();

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B.shape());
  Tensor cC = make_identity_tensor(C.shape());

  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));
  Tensor gC = local_tile(cC, wg_tile, wg_coord, Step<_1, _1, X>{});

  auto copy_a = get_block_2d_copy_A<GmemTiledCopyA>(mma, A);
  auto copy_b = get_block_2d_copy_B<GmemTiledCopyB>(mma, B);
  auto copy_c = get_block_2d_copy_D<GmemTiledCopyC>(mma, C);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);
  auto thr_copy_c = copy_c.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  auto tCrC = thr_mma.partition_sg_fragment_C(gC);
  auto tCrC_out = thr_copy_c.partition_sg_fragment_S(gC);
  auto tCgC = thr_copy_c.partition_D(gC);

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  const int prefetch_dist = 3;
  constexpr auto barrier_scope = ScopeWorkgroup;
  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  clear(tCrC);

  static constexpr bool is_B_int8_type = std::is_same_v<TB, int8_t>;

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }

    // `reorder` performs the in-register `int8_t -> ElementA` upcast via
    // `cutlass::NumericArrayConverter`. Once tCrB carries bf16/fp16
    // values it is compatible with the same DPAS atom used by the FP8
    // and W16A16 paths.
    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    cute::gemm(mma, tCrA, tCrB, tCrC);

    barrier_wait(barrier_scope);
  }

  if constexpr (is_B_int8_type) {
    float B_scale = Scales[0];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tCrC.size(); ++i) {
      tCrC(i) *= B_scale;
    }
  }

  if (Bias != nullptr) {
    static constexpr auto ATOM_M = get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
    static constexpr auto ATOM_N = get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());

    auto sg_local_n_coord = cutlass::get_sub_group_id() % ATOM_N;

    static constexpr auto tile_m = get<0>(wg_tile);
    static constexpr auto tile_n = get<1>(wg_tile);

    static constexpr auto SG_M = tile_m / ATOM_M;
    static constexpr auto SG_N = tile_n / ATOM_N;

    int sg_local_id = cutlass::get_sub_group_local_id();
    static constexpr int sg_local_range = 16;

    int n_tile_start = wg_n * tile_n;
    int n_sg_start = sg_local_n_coord * SG_N;

    CUTLASS_PRAGMA_UNROLL
    for (int sn = 0; sn < SG_N / sg_local_range; ++sn) {
      int sg_local_n = sn * sg_local_range + sg_local_id;
      float b_float = Bias[n_tile_start + n_sg_start + sg_local_n];
      CUTLASS_PRAGMA_UNROLL
      for (int sm = 0; sm < SG_M; ++sm) {
        tCrC(sn * SG_M + sm) += b_float;
      }
    }
  }

  reorder(tCrC, tCrC_out);
  copy(copy_c, tCrC_out, tCgC);
}

// ---------------------------------------------------------------------------
// Variant B -- per-K-group INT8 mainloop (sym only).
//
// Adapted from `moe_dpas_fp8::xe_gemm_fp8_pergroup<>`. Structural notes:
//
//   * `ElementB` is required to be `int8_t`. As in `xe_gemm_int_pertensor`,
//     the `int8 -> ElementA` upcast happens inside `reorder(tBrB, tCrB)`
//     via `cutlass::NumericArrayConverter`; the DPAS atom itself remains
//     `bf16/fp16 x bf16/fp16 -> fp32`.
//   * `ElementS` is the activation dtype (half / bfloat16). Scales are
//     loaded once per group boundary into a per-SG `sg_scale[]` cache and
//     cast to `float` there, matching the FP8 per-group path's cache
//     shape / prefetch schedule bit-for-bit. This differs from the
//     Variant A INT8 per-tensor path which reads FP32 scalars.
//   * The mainloop accumulates the current scale group's partial sum into
//     `tCrC_group`, then folds `tCrC_group * s` into `tCrC` at the group
//     boundary (deferred fold). At the last k_tile the fold fires
//     unconditionally so the tail group is not dropped.
//   * Bias epilogue is identical to the FP8 per-group path and the INT8
//     per-tensor path above.
//
// Asym is deliberately not templated in: the previous port carried an
// `Asym` bool + `Zeros` / `Asum` pointers, but the on-hardware cost of
// the per-M-row per-K-group asum precompute + the extra group-boundary
// loads regressed the fused path below the dequant fallback. Asym
// callers dequantise to bf16/fp16 in `sycl_tla_moe_mixed.hpp` instead.
// ---------------------------------------------------------------------------

template <class GmemTiledCopyA, class GmemTiledCopyB, class GmemTiledCopyC,
          int GroupSize, class ATensor, class BTensor, class DTensor,
          class TiledMMA, typename ElementS, typename ElementBI>
CUTE_DEVICE void xe_gemm_int_pergroup(
    ATensor const& A,   // (M,K)
    BTensor const& B,   // (N,K)
    const ElementS* Scales,
    const ElementBI* Bias,
    DTensor& C,         // (M,N)
    Coord<int, int, cute::Underscore, int> blk_coord,
    TiledMMA const& mma) {
  using TA = typename ATensor::element_type;
  using TB = typename BTensor::element_type;
  static_assert(std::is_same_v<TB, int8_t>,
                "xe_gemm_int_pergroup: ElementB must be int8_t (sym only)");
  static constexpr int group_size = GroupSize;
  static constexpr int sg_local_range = 16;
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto wg_m = get<0>(blk_coord);
  auto wg_n = get<1>(blk_coord);
  int local_id = item.get_local_linear_id();

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B.shape());
  Tensor cC = make_identity_tensor(C.shape());

  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));
  Tensor gC = local_tile(cC, wg_tile, wg_coord, Step<_1, _1, X>{});

  auto copy_a = get_block_2d_copy_A<GmemTiledCopyA>(mma, A);
  auto copy_b = get_block_2d_copy_B<GmemTiledCopyB>(mma, B);
  auto copy_c = get_block_2d_copy_D<GmemTiledCopyC>(mma, C);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);
  auto thr_copy_c = copy_c.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  auto tCrC = thr_mma.partition_sg_fragment_C(gC);
  auto tCrC_out = thr_copy_c.partition_sg_fragment_S(gC);
  auto tCgC = thr_copy_c.partition_D(gC);

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  // Prefetch distance mirrors `xe_gemm_fp8_pergroup<>` (which mirrors
  // upstream `xe_gemm<>` per-tensor now that scales are hoisted out of
  // the k_tile loop). Scales are prefetched separately once per group
  // boundary a few groups ahead.
  const int prefetch_dist = 3;
  const int prefetch_dist_scale = 3;
  constexpr auto barrier_scope = ScopeWorkgroup;
  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  static constexpr auto ATOM_M = get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_N = get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_K = get<3>(typename TiledMMA::ThrLayoutVMNK{}.shape());

  static constexpr auto tile_m = get<0>(wg_tile);
  static constexpr auto tile_n = get<1>(wg_tile);
  static constexpr auto tile_k = get<2>(wg_tile);

  static constexpr auto SG_M = tile_m / ATOM_M;
  static constexpr auto SG_N = tile_n / ATOM_N;
  // NOTE: SG_K (tile_k / ATOM_K) is intentionally not materialised here.
  // The FP8 per-group mainloop keeps it around for symmetry with the
  // per-tensor variant, but it is unused inside the pergroup loop — the
  // K-slicing granularity is driven by `group_size` / `tile_k`, not by
  // sub-group K-tiles. Kept out to avoid an `-Wunused-const-variable`.

  static constexpr int sg_n_strides = SG_N / sg_local_range;

  auto n_tile_start = wg_n * tile_n;
  auto sg_local_n_coord = cutlass::get_sub_group_id() % ATOM_N;
  auto sg_local_m_coord = cutlass::get_sub_group_id() / ATOM_N;
  int sg_local_id = cutlass::get_sub_group_local_id();
  int n_sg_start = sg_local_n_coord * SG_N;
  int m_sg_start = sg_local_m_coord * SG_M;
  int m_tile_start = wg_m * tile_m;
  int group_num = get<1>(A.shape()) / group_size;

  // Group-local accumulator: same fragment shape as `tCrC`, cleared at
  // every scale-group boundary and folded into `tCrC` with a per-N-column
  // scale before being reset. Mirrors the FP8 per-group path exactly.
  auto tCrC_group = thr_mma.partition_sg_fragment_C(gC);

  clear(tCrC);
  clear(tCrC_group);

  // Per-SG per-N scale cache. One float per SG-owned N lane stride, loaded
  // once per group boundary from an ElementS (act-dtype) tile and cast to
  // float. Not pre-initialised: the first mainloop iteration (`k_tile == 0`)
  // trips the group-boundary load unconditionally.
  float sg_scale[sg_n_strides];

  // Warm-up prefetch: A/B for `prefetch_dist` k_tiles ahead. Scales for
  // the first `prefetch_dist_scale` groups are prefetched separately so
  // they overlap with the first few MMAs of the mainloop.
  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }
  CUTLASS_PRAGMA_UNROLL
  for (int pg = 0; pg < prefetch_dist_scale; ++pg) {
    if (pg * group_size < shape<1>(A)) {
      auto next_scales_tensor = make_tensor(
          make_gmem_ptr(reinterpret_cast<const ElementS*>(
              Scales + (n_tile_start + n_sg_start) * group_num + pg)),
          make_layout(make_shape(Int<SG_N>{}, Int<1>{}),
                      make_stride(group_num, Int<1>{})));
      auto prefetch_scales = make_block_2d_prefetch<1>(
          make_shape(Int<SG_N>{}, Int<1>{}), next_scales_tensor);
      auto thr_prefetch_scales = prefetch_scales.get_slice(sg_local_id);
      auto pSgS = thr_prefetch_scales.partition_S(
          make_identity_tensor(make_shape(Int<SG_N>{}, Int<1>{})));
      prefetch(prefetch_scales, pSgS(_, 0, 0));
    }
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

    // At group start, load this SG's per-N-column scales into `sg_scale`
    // and cast from the storage dtype (ElementS = act dtype) to float.
    // ONE global load per SG-owned N-lane per group -- the deferred fold
    // below reuses these values every k_tile until the next boundary.
    if (k_tile * tile_k % group_size == 0) {
      int group_idx = (k_tile * tile_k) / group_size;
      CUTLASS_PRAGMA_UNROLL
      for (int sn = 0; sn < sg_n_strides; ++sn) {
        int sg_local_n = sn * sg_local_range + sg_local_id;
        sg_scale[sn] = static_cast<float>(
            Scales[(n_tile_start + n_sg_start + sg_local_n) * group_num + group_idx]);
      }

      // Prefetch the scales for the group `prefetch_dist_scale` groups
      // ahead so the load above hits L2 by the time we get there.
      if ((group_idx + prefetch_dist_scale) * group_size < shape<1>(A)) {
        auto next_scales_tensor = make_tensor(
            make_gmem_ptr(reinterpret_cast<const ElementS*>(
                Scales + (n_tile_start + n_sg_start) * group_num +
                group_idx + prefetch_dist_scale)),
            make_layout(make_shape(Int<SG_N>{}, Int<1>{}),
                        make_stride(group_num, Int<1>{})));
        auto prefetch_scales = make_block_2d_prefetch<1>(
            make_shape(Int<SG_N>{}, Int<1>{}), next_scales_tensor);
        auto thr_prefetch_scales = prefetch_scales.get_slice(sg_local_id);
        auto pSgS = thr_prefetch_scales.partition_S(
            make_identity_tensor(make_shape(Int<SG_N>{}, Int<1>{})));
        prefetch(prefetch_scales, pSgS(_, 0, 0));
      }
    }

    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }

    // `reorder` performs the in-register `int8_t -> ElementA` upcast via
    // `cutlass::NumericArrayConverter`. Once `tCrB` carries bf16/fp16
    // values it is compatible with the same DPAS atom used by the FP8
    // per-group path.
    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    // HOT MAINLOOP -- MMA accumulates into `tCrC_group`. Per-N scale is
    // applied ONCE at the end of the group in the fold block below.
    cute::gemm(mma, tCrA, tCrB, tCrC_group);

    // Group-boundary fold. Fires when either (a) the NEXT k_tile would
    // start a new scale group, or (b) we've reached the last k_tile of
    // the K reduction (tail-group protection).
    const bool is_group_end = (((k_tile + 1) * tile_k) % group_size == 0) ||
                              (k_tile + 1 == k_tile_count);
    if (is_group_end) {
      CUTLASS_PRAGMA_UNROLL
      for (int sn = 0; sn < sg_n_strides; ++sn) {
        float s = sg_scale[sn];
        CUTLASS_PRAGMA_UNROLL
        for (int sm = 0; sm < SG_M; ++sm) {
          const int idx = sn * SG_M + sm;
          tCrC(idx) += tCrC_group(idx) * s;
          tCrC_group(idx) = 0.0f;
        }
      }
    }

    barrier_wait(barrier_scope);
  }

  if (Bias != nullptr) {
    CUTLASS_PRAGMA_UNROLL
    for (int sn = 0; sn < sg_n_strides; ++sn) {
      int sg_local_n = sn * sg_local_range + sg_local_id;
      float b_float = Bias[n_tile_start + n_sg_start + sg_local_n];
      CUTLASS_PRAGMA_UNROLL
      for (int sm = 0; sm < SG_M; ++sm) {
        tCrC(sn * SG_M + sm) += b_float;
      }
    }
  }

  reorder(tCrC, tCrC_out);
  copy(copy_c, tCrC_out, tCgC);
}

// ---------------------------------------------------------------------------
// Persistent scheduler (adapted from `moe_dpas_fp8::MoEGEMM<>`).
//
// Structural notes vs. the FP8 scheduler:
//
//   * `ScaleMode` template parameter (`kPerTensor` vs. `kPerGroup`) picks
//     between the two per-expert scale-pointer offset conventions:
//       - kPerTensor : `Scales + expert_id`
//                      (one FP32 scalar per expert, Variant A)
//       - kPerGroup  : `Scales + expert_id * gemm_n * (gemm_k / group_size)`
//                      (auto-round `[E, N, K/group_size]` layout, Variant B)
//   * `B_offset` uses `int8_t` element units (one byte per element) --
//     identical arithmetic to the FP8 branch, but noted explicitly for
//     future readers importing this file for INT4 / INT2 where the
//     element-vs-byte distinction matters.
//   * The mainloop dispatch is `kPerTensor -> xe_gemm_int_pertensor<>`
//     (per-tensor FP32 scalar epilogue) or
//     `kPerGroup -> xe_gemm_int_pergroup<GS>` (per-group deferred fold).
//     `group_size` is a runtime argument used only in the kPerGroup case;
//     the switch inside `MoEGEMM_int` selects one of the compile-time
//     specialisations covered by the FP8 header (32 / 64 / 128 / 256).
// ---------------------------------------------------------------------------

template <class GmemTiledCopyA, class GmemTiledCopyB, class GmemTiledCopyD,
          char LayoutKindA, char LayoutKindB, char LayoutKindD, ScaleMode Mode,
          class TiledMMA, typename ElementA, typename ElementB,
          typename ElementS, typename ElementBI, typename ElementD>
CUTE_DEVICE void MoEGEMM_int(const ElementA* Activations,
                             const ElementB* Weights,
                             const ElementS* Scales,
                             const ElementBI* Bias,
                             ElementD* Outputs, TiledMMA const& mma,
                             const int* rows_per_expert,
                             const int32_t num_experts,
                             const int32_t group_size, const int32_t gemm_n,
                             const int32_t gemm_k, int32_t* atomic_buffer,
                             const sycl::local_accessor<int32_t, 1>& slm_mem_const) {
  constexpr char actual_layout_of_B = LayoutKindB ^ ('R' ^ 'C');

  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto wg_tile = mma.tile_mnk();
  auto wg_tile_m = get<0>(wg_tile);
  auto wg_tile_n = get<1>(wg_tile);

  int group_id = item.get_group_linear_id();
  int gemm_n_pad = (gemm_n + wg_tile_n - 1) / wg_tile_n * wg_tile_n;
  int group_m_id = (group_id * wg_tile_n) / gemm_n_pad;
  int group_range = item.get_group_range(1);
  int local_id = item.get_local_linear_id();

  if (group_id == 0 && local_id == 0) {
    auto atm = sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>(
        atomic_buffer[0]);
    atm.store(0);
  }

  int pre_rows = 0;
  int pre_tiles = 0;

  int32_t* slm_mem = static_cast<int32_t*>(
      slm_mem_const.template get_multi_ptr<sycl::access::decorated::no>().get());

  for (int i = 0; i < num_experts; ++i) {
    int gemm_m = rows_per_expert[i];
    int cumsum_rows_for_experts = pre_rows + gemm_m;
    int cumsum_tiles_for_experts =
        (gemm_m + wg_tile_m - 1) / wg_tile_m + pre_tiles;

    if (group_m_id >= cumsum_tiles_for_experts) {
      pre_rows = cumsum_rows_for_experts;
      pre_tiles = cumsum_tiles_for_experts;
      continue;
    }

    int expert_id = i;
    int64_t B_offset = static_cast<int64_t>(expert_id) *
                       static_cast<int64_t>(gemm_n) *
                       static_cast<int64_t>(gemm_k);

    ElementA* ptr_A_curr_batch =
        const_cast<ElementA*>(Activations) + pre_rows * gemm_k;
    ElementB* ptr_B_curr_batch = const_cast<ElementB*>(Weights) + B_offset;
    ElementD* ptr_D_curr_batch = Outputs + pre_rows * gemm_n;

    ElementS* ptr_Scales_curr_batch = nullptr;
    if constexpr (Mode == ScaleMode::kPerTensor) {
      // [E] -- one scalar per expert.
      ptr_Scales_curr_batch = const_cast<ElementS*>(Scales) + expert_id;
    } else {
      // [E, N, K/group_size] -- auto-round layout. Advance one expert.
      int64_t scale_expert_stride =
          static_cast<int64_t>(gemm_n) * (gemm_k / group_size);
      ptr_Scales_curr_batch =
          const_cast<ElementS*>(Scales) + expert_id * scale_expert_stride;
    }

    ElementBI* ptr_Bias_curr_batch = nullptr;
    if (Bias != static_cast<ElementBI*>(nullptr)) {
      ptr_Bias_curr_batch = const_cast<ElementBI*>(Bias) + expert_id * gemm_n;
    }

    auto A_tensor = make_moe_tensor<ElementA, LayoutKindA>(ptr_A_curr_batch,
                                                           gemm_m, gemm_k);
    auto B_tensor = make_moe_tensor<ElementB, actual_layout_of_B>(
        ptr_B_curr_batch, gemm_n, gemm_k);
    auto D_tensor = make_moe_tensor<ElementD, LayoutKindD>(ptr_D_curr_batch,
                                                           gemm_m, gemm_n);

    while (group_m_id < cumsum_tiles_for_experts) {
      int n_coord = (group_id * wg_tile_n) % gemm_n_pad / wg_tile_n;
      int m_coord = (group_m_id - pre_tiles);
      auto tile_coord = make_coord(m_coord, n_coord, _, 0);

      if constexpr (Mode == ScaleMode::kPerTensor) {
        xe_gemm_int_pertensor<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD>(
            A_tensor, B_tensor, ptr_Scales_curr_batch, ptr_Bias_curr_batch,
            D_tensor, tile_coord, mma);
      } else {
// Per-K-group dispatch on the runtime group_size (mirrors the FP8
// header's `ARK_MOE_DPAS_FP8_GROUP_CALLER` macro; group_size is not a
// compile-time constant at this level).
#define ARK_MOE_DPAS_INT_GROUP_CALLER(GS)                                     \
  xe_gemm_int_pergroup<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD, GS>(   \
      A_tensor, B_tensor, ptr_Scales_curr_batch, ptr_Bias_curr_batch,         \
      D_tensor, tile_coord, mma);
        if (group_size == 32) {
          ARK_MOE_DPAS_INT_GROUP_CALLER(32)
        } else if (group_size == 64) {
          ARK_MOE_DPAS_INT_GROUP_CALLER(64)
        } else if (group_size == 128) {
          ARK_MOE_DPAS_INT_GROUP_CALLER(128)
        } else if (group_size == 256) {
          ARK_MOE_DPAS_INT_GROUP_CALLER(256)
        }
#undef ARK_MOE_DPAS_INT_GROUP_CALLER
      }

      if (local_id == 0) {
        slm_mem[0] = cutlass::atomicAdd(atomic_buffer, 1);
      }
      item.barrier(sycl::access::fence_space::local_space);
      group_id = group_range + slm_mem[0];
      group_m_id = (group_id * wg_tile_n) / gemm_n_pad;
    }
    pre_rows = cumsum_rows_for_experts;
    pre_tiles = cumsum_tiles_for_experts;
  }
}

// ---------------------------------------------------------------------------
// Launcher (fork of `moe_dpas_fp8::MoEGEMMLauncher<>`).
// ---------------------------------------------------------------------------

template <typename, typename, typename, typename, char, char, class, ScaleMode>
class DpasGemmIntName;

template <char layoutA, char layoutB, class policy, ScaleMode Mode,
          typename ElementA, typename ElementB, typename ElementS,
          typename ElementBI, typename ElementD>
void MoEGEMMLauncher_int(sycl::queue& stream, const ElementA* activations,
                         const ElementB* weights, const ElementS* scales,
                         const ElementBI* bias, ElementD* outputs,
                         const int gemm_n, const int gemm_k,
                         const int* rows_per_expert, const int num_experts,
                         const int group_size, int32_t* atomic_buffer) {
  using ElementA_non_CV = cutlass::platform::remove_cv_t<ElementA>;
  // DPAS operates on same-dtype A/B pairs; the INT8 B tensor is upcast to
  // the activation dtype (ElementA) via `reorder(tBrB, tCrB)` in the
  // mainloop before entering the MMA atom. See open question #2 above.
  auto op = XE_DPAS_TT<8, float, ElementA_non_CV, ElementA_non_CV>{};

  using WGTile = typename policy::WGTile;
  using SGLayout = typename policy::SGLayout;
  using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>,
                                      SGLayout>::TiledMMA;
  auto mma = MMA{};

  int sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  auto MaxThreadsPerWorkgroup = size(mma);

  static constexpr int MaxThreadsPerSM = 512;
  if (MaxThreadsPerSM % MaxThreadsPerWorkgroup != 0) {
    throw std::runtime_error(
        "moe_prefill_int_dpas: MaxThreadsPerSM must be divisible by "
        "MaxThreadsPerWorkgroup");
  }

  sycl::range<3> local(1, 1, MaxThreadsPerWorkgroup);
  sycl::range<3> global(1, sm_count * MaxThreadsPerSM / MaxThreadsPerWorkgroup, 1);

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props{syclex::sub_group_size<16>,
                                  intelex::grf_size<256>};

  using GmemTiledCopyA = typename policy::GmemTiledCopyA;
  using GmemTiledCopyB = typename policy::GmemTiledCopyB;
  using GmemTiledCopyD = typename policy::GmemTiledCopyD;

  auto event = stream.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> local_mem(sycl::range<1>(1), cgh);
    cgh.parallel_for<DpasGemmIntName<ElementA, ElementB, ElementS, ElementD,
                                     layoutA, layoutB, policy, Mode>>(
        sycl::nd_range<3>{global * local, local}, kernel_props, [=](auto) {
          MoEGEMM_int<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD, layoutA,
                      layoutB, 'R', Mode>(
              activations, weights, scales, bias, outputs, mma,
              rows_per_expert, num_experts, group_size, gemm_n, gemm_k,
              atomic_buffer, local_mem);
        });
  });

  EventManager::getInstance().addEvent(event);
  event.wait();
}

// ---------------------------------------------------------------------------
// Host-side driver: INT8 Variant A (per-tensor).
//
// Weight layout `[E, K, N]` row-major, one signed byte per element.
// Scale pointer `[E]` FP32 (one per-expert scalar).
// ---------------------------------------------------------------------------

template <typename ScalarT>
void moe_prefill_int_dpas_per_tensor_dispatch(
    sycl::queue* q, const ScalarT* activations, const int8_t* weights_KN,
    const float* scales_e, ScalarT* outputs, const int* num_tokens_per_expert,
    int E, int N, int K, int total_tokens) {
  if (E == 0 || N == 0 || K == 0 || total_tokens == 0) return;

  compat::set_default_queue(*q);

  using ElementB = int8_t;
  // Map the caller-facing SYCL native half/bfloat16 to the CUTLASS type CUTE
  // has DPAS-atom specializations for. See `cute_scalar` in the FP8 header.
  using ElementA = cute_scalar_t<ScalarT>;
  const auto* activations_ca =
      reinterpret_cast<const ElementA*>(activations);
  auto* outputs_ca = reinterpret_cast<ElementA*>(outputs);

  int A_avg_M = total_tokens / E;

  int32_t* atomic_buffer = sycl::malloc_device<int32_t>(1, *q);
  if (atomic_buffer == nullptr) {
    throw std::runtime_error(
        "moe_prefill_int_dpas(per-tensor): failed to allocate atomic buffer");
  }

#define ARK_DPAS_INT_PT_LAUNCH(policy)                                         \
  MoEGEMMLauncher_int<'R', 'R', policy, ScaleMode::kPerTensor>(                \
      *q, activations_ca, weights_KN, scales_e,                                \
      static_cast<const ElementA*>(nullptr), outputs_ca, N, K,                 \
      num_tokens_per_expert, E, /*group_size=*/0, atomic_buffer);

  if (A_avg_M <= 8) {
    ARK_DPAS_INT_PT_LAUNCH(dpas_w8a16_policy_m_16);
  } else if (A_avg_M <= 32) {
    ARK_DPAS_INT_PT_LAUNCH(dpas_w8a16_policy_m_32);
  } else {
    ARK_DPAS_INT_PT_LAUNCH(dpas_w8a16_policy);
  }
#undef ARK_DPAS_INT_PT_LAUNCH

  sycl::free(atomic_buffer, *q);
}

// ---------------------------------------------------------------------------
// Host-side driver: INT8 Variant B (per-K-group, sym only).
//
// Weight layout `[E, N, K]` row-major (auto-round convention -- the launcher
// passes `LayoutKindB='C'` so `MoEGEMM_int<>` XOR-flips to `'R'` inside
// `make_moe_tensor`, matching the physical `[N, K]` row-major storage).
// One signed byte per element: `int8_t` for sym storage (auto-round
// `_pack_int8_sym` packs signed `[-127, 127]` values into a `torch.uint8`
// buffer that is reinterpret-cast to `int8_t` here). The same entry point
// is also used by the INT4-sym / INT2-sym prefill paths after an on-device
// nibble/crumb -> int8 upcast pass fills the dequant workspace.
// Scales `[E, N, K/group_size]` in act dtype (half / bfloat16), same layout
// as the FP8 per-group path.
// ---------------------------------------------------------------------------

template <typename ScalarT>
void moe_prefill_int_dpas_per_group_dispatch(
    sycl::queue* q, const ScalarT* activations, const int8_t* weights_NK,
    const ScalarT* scales, ScalarT* outputs,
    const int* num_tokens_per_expert, int E, int N, int K, int group_size,
    int total_tokens) {
  if (E == 0 || N == 0 || K == 0 || total_tokens == 0) return;
  if (K % group_size != 0) {
    throw std::invalid_argument(
        "moe_prefill_int_dpas(per-group): K must be a multiple of group_size");
  }

  compat::set_default_queue(*q);

  // Map the caller-facing SYCL native half/bfloat16 to the CUTLASS type CUTE
  // has DPAS-atom specializations for. See `cute_scalar` in the FP8 header.
  using ElementA = cute_scalar_t<ScalarT>;
  const auto* activations_ca =
      reinterpret_cast<const ElementA*>(activations);
  const auto* scales_ca = reinterpret_cast<const ElementA*>(scales);
  auto* outputs_ca = reinterpret_cast<ElementA*>(outputs);

  int A_avg_M = total_tokens / E;

  int32_t* atomic_buffer = sycl::malloc_device<int32_t>(1, *q);
  if (atomic_buffer == nullptr) {
    throw std::runtime_error(
        "moe_prefill_int_dpas(per-group): failed to allocate atomic buffer");
  }

#define ARK_DPAS_INT_PG_LAUNCH_SYM(policy)                                     \
  MoEGEMMLauncher_int<'R', 'C', policy, ScaleMode::kPerGroup>(                 \
      *q, activations_ca, weights_NK, scales_ca,                               \
      static_cast<const ElementA*>(nullptr), outputs_ca, N, K,                 \
      num_tokens_per_expert, E, group_size, atomic_buffer);

  if (A_avg_M <= 8) {
    ARK_DPAS_INT_PG_LAUNCH_SYM(dpas_w8a16_policy_m_16);
  } else if (A_avg_M <= 32) {
    ARK_DPAS_INT_PG_LAUNCH_SYM(dpas_w8a16_policy_m_32);
  } else {
    ARK_DPAS_INT_PG_LAUNCH_SYM(dpas_w8a16_policy);
  }
#undef ARK_DPAS_INT_PG_LAUNCH_SYM

  sycl::free(atomic_buffer, *q);
}

// ---------------------------------------------------------------------------
// Env-flag helper -- `ARK_MOE_PREFILL_DPAS_INT8` (default ON, semantics
// identical to `moe_prefill_dpas_fp8_enabled`).
//
// Truthy values (case-insensitive): "1", "true", "on", "yes".
// Explicit "0" / "false" / "off" / "no" disable. Re-read on every call so
// benchmarks / tests can toggle the path in-process.
// ---------------------------------------------------------------------------
inline bool moe_prefill_dpas_int_enabled() {
  const char* env = std::getenv("ARK_MOE_PREFILL_DPAS_INT8");
  if (env == nullptr) return true;  // default ON
  std::string s(env);
  for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "0" || s == "false" || s == "off" || s == "no") return false;
  return true;
}

// ---------------------------------------------------------------------------
// Shape preconditions for the per-K-group INT8 dispatcher branch. Matches
// the FP8 per-group predicate exactly (same policy tiles, same group_size
// specialisations).
// ---------------------------------------------------------------------------
inline bool moe_prefill_dpas_int_pergroup_shape_ok(int N, int K,
                                                    int group_size) {
  if (N <= 0 || K <= 0 || group_size <= 0) return false;
  if (N % 64 != 0) return false;
  if (K % 32 != 0) return false;
  if (K % group_size != 0) return false;
  if (group_size != 32 && group_size != 64 && group_size != 128 &&
      group_size != 256) {
    return false;
  }
  return true;
}

}  // namespace moe_dpas_int
}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
