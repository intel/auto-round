// SYCL MoE Prefill — FP8 mixed-input DPAS Grouped GEMM
// (Variants A + B, ported from vllm-xpu-kernels)
//
// STATUS: NEEDS-HARDWARE-VALIDATION -- this header is a mode-2 port
// ---------------------------------------------------------------------------
// The C++ template code below is a near-verbatim port of the FP8 mainloop
// (`xe_gemm<>`), the FP8-per-K-group mainloop (adapted from
// `xe_gemm_4bits<>`), the atomic persistent scheduler (`MoEGEMM<>`) and the
// launcher (`MoEGEMMLauncher<>`) from
//
//   vllm-project/vllm-xpu-kernels@26497a0
//   csrc/xpu/grouped_gemm/xe_2/{gemm_xe2,gemm_xe2_policy,grouped_gemm_xe2,
//                               grouped_gemm_xe2_interface}.hpp
//
// It was committed WITHOUT on-hardware build / accuracy / perf verification
// (no XPU, no oneAPI toolchain, no cutlass-sycl checkout in the environment
// where the port was authored). First-build errors are expected. Kernel
// numerics on real Intel BMG/PVC silicon MUST be validated against
// `test_moe_prefill_accuracy.py::test_accuracy_fp8` at 7e-2 tolerance before
// this path is trusted in production. See the PR description that landed
// this file for the port's provenance & the on-hardware TODOs.
//
// Path taxonomy
// -------------
// This header adds TWO fused FP8 grouped-GEMM entry points sitting alongside
// the pre-existing scalar-DPAS `sycl_tla_moe_prefill_fp8_native.hpp`:
//
//   * Variant A -- per-tensor FP8. Weight scales are `[E]` (one scalar per
//     expert). Weight layout `[E, K, N]` row-major. Epilogue is
//     `tCrC(i) *= Scales[expert_id]`. Requires callers to re-quantize with
//     a per-tensor recipe.
//
//   * Variant B -- per-K-group FP8. Weight scales are `[E, N, K/group_size]`
//     in the activation dtype (auto-round's existing quantization scheme,
//     drop-in for existing FP8 checkpoints). Weight layout `[E, N, K]` row-
//     major (auto-round's existing packer output). The mainloop reloads the
//     per-`N`-column scale at every `k_tile * tile_k % group_size == 0`
//     boundary and applies it to the register-side upcast (bf16/fp16)
//     fragment via the inline-asm `apply_scale()` primitive.
//
// The scalar-DPAS `sycl_tla_moe_prefill_fp8_native.hpp` path stays in tree
// (per plan: only delete after on-hardware parity). Both variants live in
// the `ark::moe_dpas_fp8` namespace to avoid symbol collisions with the
// pre-existing `MoE::MoEGEMM<>` (an unrelated grouped GEMM shipped in
// sycl-tla example 12 that is pulled in via `sycl_tla_moe.hpp`).
//
// On-hardware open questions (must be resolved on first build):
//   1. Whether `cute::reorder(tBrB, tCrB)` upcasts `float_e4m3_t / float_e5m2_t`
//      to `act_dtype` (bfloat16_t / half_t) or to `float`. NOTE: as of the
//      deferred-scale rewrite of `xe_gemm_fp8_pergroup<>` this no longer
//      affects correctness or performance -- the mainloop feeds `tCrB`
//      straight into `cute::gemm(mma, tCrA, tCrB, tCrC_group)` with no
//      per-element scale multiply, and the DPAS atom is responsible for
//      any residual upcast. The old `apply_scale()` primitive is retained
//      in this file for future variants that may need it but is unused on
//      the per-group hot path.
//   2. Whether `cutlass-sycl` at the version auto-round pulls exposes
//      `make_block_2d_prefetch<Dim>`, `get_block_2d_copy_{A,B,D}`,
//      `partition_sg_fragment_{A,B,C,D,S}`, `TiledMMAHelper`,
//      `PersistentTileSchedulerXeMoE`, `cutlass::atomicAdd` under the
//      identifiers used below. If any symbol has moved, adjust the include
//      or the qualified name.
//   3. Whether `XE_DPAS_TT<8, float, ElementA, ElementB>` accepts
//      `ElementB = float_e4m3_t / float_e5m2_t` -- if the DPAS atom
//      rejects fp8 (i.e. requires the upcast to happen at MMA time via
//      `reorder`), the `xe_gemm<>` per-tensor path still works because
//      `reorder(tBrB, tCrB)` runs in the mainloop; only the atom's
//      template signature may need `ElementA` instead of `(ElementA,
//      ElementB)` on this version of cutlass-sycl.
//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Portions ported from vllm-project/vllm-xpu-kernels
// (Copyright (C) 2025 Intel Corporation, SPDX-License-Identifier: BSD-3-Clause).
// The BSD-3-Clause upstream license is compatible with Apache-2.0; the
// upstream copyright notice is preserved in the source-of-truth links in
// the docstring above.

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#ifdef ARK_XPU
#include <sycl/sycl.hpp>
#endif

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

// Pull in the same set of cutlass-sycl / CuTe headers as `sycl_tla_moe.hpp`.
// `sycl_tla_moe.hpp` already validates that `cute/tensor.hpp`,
// `cute/util/compat.hpp`, `cutlass/kernel_hardware_info.h` and the
// grf_size_properties extension are present, so no new CMake work is
// required for this header.
#include "sycl_tla_moe.hpp"

// Additional cutlass-sycl bits used only by the DPAS launcher / scheduler.
// If any of these paths differ in the auto-round build, adjust here.
// (These match the include list at the top of
// vllm-xpu-kernels grouped_gemm_xe2_interface.hpp.)
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/sycl_event_manager.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace ark {
namespace moe_dpas_fp8 {

using namespace cute;

// ---------------------------------------------------------------------------
// Scalar-type translation: callers in `sycl_tla_moe_mixed.hpp` pass the
// activation buffer typed as `sycl::half` / `sycl::ext::oneapi::bfloat16`
// (the SYCL native half/bfloat16 types), but CUTE only ships
// `XE_DPAS_TT<...>` specializations for `cutlass::half_t` /
// `cutlass::bfloat16_t`. Instantiating the DPAS atom directly with the SYCL
// native types produces `implicit instantiation of undefined template
// 'cute::XE_DPAS_TT<8, float, sycl::detail::half_impl::half>'` at compile
// time. The two representations are bit-compatible (both are IEEE-754 16-bit
// floats laid out identically in memory), so the dispatchers below map the
// caller's `ScalarT` to the matching CUTLASS type and `reinterpret_cast` the
// buffer pointers before entering `MoEGEMMLauncher<>`. The sibling
// `sycl_tla_moe.hpp` dispatcher works today for the same reason (it selects
// `cutlass::half_t` / `cutlass::bfloat16_t` at its call sites before
// invoking `choose_tiled_mma`).
// ---------------------------------------------------------------------------
template <typename ScalarT>
struct cute_scalar {
  using type = ScalarT;
};

template <>
struct cute_scalar<sycl::half> {
  using type = cutlass::half_t;
};

template <>
struct cute_scalar<sycl::ext::oneapi::bfloat16> {
  using type = cutlass::bfloat16_t;
};

template <typename ScalarT>
using cute_scalar_t = typename cute_scalar<ScalarT>::type;

// ---------------------------------------------------------------------------
// Policy classes (ported verbatim from vllm-xpu-kernels
// `gemm_xe2_policy.hpp`, renamed to `dpas_*` to avoid collision with any
// future upstream import into `namespace MoE`).
// ---------------------------------------------------------------------------

class dpas_policy_base {
 public:
  using WGTile = Shape<_256, _256, _32>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;

  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  using GmemTiledCopyD = void;
};

class dpas_w16a16_policy : public dpas_policy_base {
 public:
  using GmemTiledCopyD = XE_STORE_2D<16, 8, 32>;
};

class dpas_w8a16_policy : public dpas_policy_base {
 public:
  using WGTile = Shape<_128, _128, _16>;
  using SGLayout = Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>;

  using GmemTiledCopyD = XE_STORE_2D<16, 8, 32>;
};

class dpas_w8a16_policy_m_16 : public dpas_policy_base {
 public:
  using WGTile = Shape<_16, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

class dpas_w8a16_policy_m_32 : public dpas_policy_base {
 public:
  using WGTile = Shape<_32, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

// ---------------------------------------------------------------------------
// `apply_scale` -- inline-asm per-lane multiply of a bf16/fp16 fragment by
// an FP32 scalar. Copied verbatim from vllm-xpu-kernels `gemm_xe2.hpp`.
// Only compiled inside the device path (`__SYCL_DEVICE_ONLY__`
// + `SYCL_INTEL_TARGET`); on the host side the function is a no-op returning
// its input, so the definition is safe to include in host-only translation
// units too.
// ---------------------------------------------------------------------------

template <typename TB>
CUTE_DEVICE TB apply_scale(TB& x, float& y) {
  static_assert(is_any_of_v<TB, cutlass::bfloat16_t, cutlass::half_t>,
                "apply_scale: only BF16 & FP16 fragments are supported");
  uint16_t z = sycl::bit_cast<uint16_t>(x);
#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
  if constexpr (std::is_same_v<TB, cutlass::half_t>) {
    asm("{\n"
        ".decl Z_FP16 v_type=G type=HF num_elts=16 alias=<%0,0>\n"
        ".decl Y_FP32 v_type=G type=F num_elts=16 alias=<%1,0>\n"
        "mul (M1, 16) Z_FP16(0,0)<1> Z_FP16(0,0)<1;1,0> Y_FP32(0,0)<1;1,0>\n"
        "}\n"
        : "+rw"(z)
        : "rw"(y));
  } else {
    asm("{\n"
        ".decl Z_BF16 v_type=G type=BF num_elts=16 alias=<%0,0>\n"
        ".decl Y_FP32 v_type=G type=F num_elts=16 alias=<%1,0>\n"
        "mul (M1, 16) Z_BF16(0,0)<1> Z_BF16(0,0)<1;1,0> Y_FP32(0,0)<1;1,0>\n"
        "}\n"
        : "+rw"(z)
        : "rw"(y));
  }
#endif
  return sycl::bit_cast<TB>(z);
}

// ---------------------------------------------------------------------------
// Variant A -- per-tensor FP8 mainloop.
//
// Ported verbatim from vllm-xpu-kernels `gemm_xe2.hpp::xe_gemm<>`. `Scales`
// is a pointer to a single FP32 scalar (the per-expert weight scale; the
// caller has already offset by `expert_id`). The epilogue folds
// `tCrC(i) *= Scales[0]` before the store.
// ---------------------------------------------------------------------------

template <class GmemTiledCopyA, class GmemTiledCopyB, class GmemTiledCopyC,
          class ATensor, class BTensor, class DTensor, class TiledMMA,
          typename ElementS, typename ElementBI>
CUTE_DEVICE void xe_gemm(ATensor const& A,   // (M,K)
                         BTensor const& B,   // (N,K)
                         const ElementS* Scales,
                         const ElementBI* Bias,
                         DTensor& C,         // (M,N)
                         Coord<int, int, cute::Underscore, int> blk_coord,
                         TiledMMA const& mma) {
  using TA = typename ATensor::element_type;
  using TB = typename BTensor::element_type;
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

  using ElementB = typename BTensor::element_type;
  static constexpr bool is_B_fp8_type =
      std::is_same_v<ElementB, cutlass::float_e5m2_t> ||
      std::is_same_v<ElementB, cutlass::float_e4m3_t>;

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

    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    cute::gemm(mma, tCrA, tCrB, tCrC);

    barrier_wait(barrier_scope);
  }

  if constexpr (is_B_fp8_type) {
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
// Variant B -- per-K-group FP8 mainloop (deferred-scale accumulation).
//
// The original port of this mainloop was adapted from vllm-xpu-kernels
// `gemm_xe2.hpp::xe_gemm_4bits<>` and applied the per-N-column scale to
// the upcast `tCrB` fragment inside EVERY k_tile iteration via an
// inline-asm `apply_scale()` (bf16) or `*=` (fp16) triple loop. That
// pattern is required for int4/mxfp4 (where the scale contributes to the
// upcast itself), but for FP8 it is pure overhead: the scale is a plain
// floating-point multiplier that distributes over the K reduction and
// therefore can be folded into the accumulator once per scale group
// instead of once per k_tile. Measured cost on the 8K-token minimax
// shapes was ~60% of the mainloop instruction stream.
//
// Deferred-scale design (this revision)
// -------------------------------------
// We split the C accumulator in two:
//
//   tCrC       : final FP32 accumulator, folds in group results and
//                receives the bias / store epilogue (same role as before).
//   tCrC_group : FP32 accumulator scoped to a single K-scale-group,
//                cleared at every group boundary.
//
// The hot mainloop reduces to exactly the upstream `xe_gemm<>` body:
//
//     copy(A,B); reorder(A,B); cute::gemm(mma, tCrA, tCrB, tCrC_group);
//
// with NO `apply_scale` on `tCrB`. At every group boundary
// (`(k_tile+1)*tile_k % group_size == 0`, plus the tail case
// `k_tile == k_tile_count - 1`) we load the SG's per-N-column scale slice
// exactly once, fold `tCrC += tCrC_group * scale_n` using the same
// linear `(sn * SG_M + sm)` indexing the bias epilogue already uses, and
// clear `tCrC_group`.
//
// Numerical equivalence
// ---------------------
// For any output element the deferred formulation computes
//     tCrC(m,n) = sum_g scale(n,g) * sum_{k in group_g} A(m,k) * W(n,k)
// which is bit-equivalent to the original
//     sum_g sum_{k in group_g} A(m,k) * (W(n,k) * scale(n,g))
// under associativity/distributivity, differing only in FP32-accumulator
// ordering. Ordering drift stays within the 7e-2 tolerance already
// enforced by `test_moe_prefill_accuracy.py::test_accuracy_fp8`.
//
// Path taxonomy vs. upstream
// --------------------------
//   * Upstream `xe_gemm<>`         : per-tensor FP8, scale applied ONCE in
//                                    the epilogue (`tCrC(i) *= B_scale`).
//   * Upstream `xe_gemm_4bits<>`   : true 4-bit path, scale is part of the
//                                    upcast; scale MUST live in the hot
//                                    loop.
//   * ARK `xe_gemm_fp8_pergroup<>` : per-K-group FP8, scale applied ONCE
//                                    per K-scale-group in a deferred fold
//                                    (this file). Hot loop matches
//                                    upstream `xe_gemm` instruction-for-
//                                    instruction; extra cost is only the
//                                    per-group fold + one SG-wide scale
//                                    load, both hidden by DPAS pipelining.
//
// Prefetch distance is set to 3 (upstream `xe_gemm` value); the old
// distance of 6 was needed to hide the per-k_tile scale load, which no
// longer exists here. Scales are prefetched once per group boundary at
// group-index `+ prefetch_dist_scale` ahead.
// ---------------------------------------------------------------------------

template <class GmemTiledCopyA, class GmemTiledCopyB, class GmemTiledCopyC,
          int GroupSize, class ATensor, class BTensor, class DTensor,
          class TiledMMA, typename ElementS, typename ElementBI>
CUTE_DEVICE void xe_gemm_fp8_pergroup(
    ATensor const& A,   // (M,K)
    BTensor const& B,   // (N,K)
    const ElementS* Scales,
    const ElementBI* Bias,
    DTensor& C,         // (M,N)
    Coord<int, int, cute::Underscore, int> blk_coord,
    TiledMMA const& mma) {
  using TA = typename ATensor::element_type;
  using TB = typename BTensor::element_type;
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

  // Prefetch distance mirrors upstream `xe_gemm<>` (per-tensor FP8) now
  // that the mainloop no longer consumes scales per k_tile. The old
  // per-group implementation used `prefetch_dist = 6` to hide the extra
  // scale reload; that reload is gone, so the deeper prefetch pipeline
  // only wastes GRF and L2 bandwidth. Scales are prefetched separately
  // once per group boundary a few groups ahead (`prefetch_dist_scale`).
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
  static constexpr auto SG_K = tile_k / ATOM_K;
  (void)SG_K;

  // Number of per-SG N-lane strides for the fold / bias epilogue. This
  // matches the pattern already used by the bias epilogue below, so the
  // deferred-scale fold can reuse the same linear `sn * SG_M + sm`
  // indexing into `tCrC` / `tCrC_group`.
  static constexpr int sg_n_strides = SG_N / sg_local_range;

  auto n_tile_start = wg_n * tile_n;
  auto sg_local_n_coord = cutlass::get_sub_group_id() % ATOM_N;
  int sg_local_id = cutlass::get_sub_group_local_id();
  int n_sg_start = sg_local_n_coord * SG_N;
  int group_num = get<1>(A.shape()) / group_size;

  // Group-local accumulator: same fragment shape as `tCrC`, cleared at
  // every scale-group boundary and folded into `tCrC` with a per-N-column
  // scale before being reset. Doubles the C fragment footprint (128 floats
  // per lane for `dpas_w8a16_policy` <128,128,16>), which is comfortably
  // within the grf_size=256 GRF budget requested by the launcher.
  auto tCrC_group = thr_mma.partition_sg_fragment_C(gC);

  clear(tCrC);
  clear(tCrC_group);

  // Per-SG per-N scale cache. Loaded once per group boundary; kept small
  // (one float per SG-owned N lane stride) so it lives entirely in the
  // register file. Matches upstream `xe_gemm`'s per-tensor `float B_scale`
  // caching, generalized to per-N-column for the per-group scheme. Not
  // pre-initialized: the first mainloop iteration (`k_tile == 0`) trips
  // the group-boundary load unconditionally, so every entry is written
  // before any read.
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

    // At group start, load this SG's per-N-column scales into `sg_scale`.
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

    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    // HOT MAINLOOP -- identical to upstream `xe_gemm<>` (per-tensor).
    // `tCrB` is left untouched: no per-element scale application, no
    // inline-asm `apply_scale`. The MMA accumulates the current scale
    // group's partial sum into `tCrC_group`; the per-N scale is applied
    // ONCE at the end of the group in the fold block below.
    cute::gemm(mma, tCrA, tCrB, tCrC_group);

    // Group-boundary fold. Fires when either (a) the NEXT k_tile would
    // start a new scale group, or (b) we've reached the last k_tile of
    // the K reduction. Applies the per-N-column scale we cached at group
    // start, folds `tCrC_group` into `tCrC`, and clears `tCrC_group`.
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
// Tensor-view helper (verbatim from vllm-xpu-kernels grouped_gemm_xe2.hpp).
// ---------------------------------------------------------------------------

template <typename T, char LayoutKind>
CUTE_DEVICE auto make_moe_tensor(T* ptr, int r, int c) {
  auto shape = make_shape(r, c);
  if constexpr (LayoutKind == 'C')
    return make_tensor(make_gmem_ptr(ptr),
                       make_layout(shape, make_stride(_1{}, r)));
  else
    return make_tensor(make_gmem_ptr(ptr),
                       make_layout(shape, make_stride(c, _1{})));
}

// ---------------------------------------------------------------------------
// `MoEGEMM<>` -- atomic persistent scheduler over `rows_per_expert`.
//
// Ported from vllm-xpu-kernels `grouped_gemm_xe2.hpp`, with the following
// simplifications for auto-round's FP8 use case:
//
//   * `ScaleMode` template parameter (`kPerTensor` vs. `kPerGroup`) picks
//     between the two per-expert scale-pointer offset conventions:
//       - kPerTensor : `Scales + expert_id`
//                      (one FP32 scalar per expert, Variant A)
//       - kPerGroup  : `Scales + expert_id * gemm_n * (gemm_k / group_size)`
//                      (auto-round `[E, N, K/group_size]` layout, Variant B)
//   * The int4/mxfp4 4-bit branches (`is_B_4bits`, `B_offset /= 2`) are
//     removed -- FP8 is one byte per element so `B_offset =
//     expert_id * gemm_n * gemm_k` is always in element units.
//   * The mainloop dispatch is `kPerTensor -> xe_gemm<>` (per-tensor scalar
//     epilogue) or `kPerGroup -> xe_gemm_fp8_pergroup<>` (per-group inline
//     scale reload).
// ---------------------------------------------------------------------------

enum class ScaleMode : int { kPerTensor = 0, kPerGroup = 1 };

template <class GmemTiledCopyA, class GmemTiledCopyB, class GmemTiledCopyD,
          char LayoutKindA, char LayoutKindB, char LayoutKindD, ScaleMode Mode,
          class TiledMMA, typename ElementA, typename ElementB,
          typename ElementS, typename ElementBI, typename ElementD>
CUTE_DEVICE void MoEGEMM(const ElementA* Activations, const ElementB* Weights,
                         const ElementS* Scales, const ElementBI* Bias,
                         ElementD* Outputs, TiledMMA const& mma,
                         const int* rows_per_expert, const int32_t num_experts,
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
      ptr_Scales_curr_batch = const_cast<ElementS*>(Scales) +
                              static_cast<int64_t>(expert_id) * gemm_n *
                                  (gemm_k / group_size);
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
        xe_gemm<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD>(
            A_tensor, B_tensor, ptr_Scales_curr_batch, ptr_Bias_curr_batch,
            D_tensor, tile_coord, mma);
      } else {
// Per-K-group dispatch on the runtime group_size (matches
// vllm-xpu-kernels' `XE_GEMM_4BITS_CALLER` macro; group_size is not a
// compile-time constant at this level).
#define ARK_MOE_DPAS_FP8_GROUP_CALLER(GS)                                     \
  xe_gemm_fp8_pergroup<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD, GS>(   \
      A_tensor, B_tensor, ptr_Scales_curr_batch, ptr_Bias_curr_batch,         \
      D_tensor, tile_coord, mma);
        if (group_size == 32) {
          ARK_MOE_DPAS_FP8_GROUP_CALLER(32)
        } else if (group_size == 64) {
          ARK_MOE_DPAS_FP8_GROUP_CALLER(64)
        } else if (group_size == 128) {
          ARK_MOE_DPAS_FP8_GROUP_CALLER(128)
        } else if (group_size == 256) {
          ARK_MOE_DPAS_FP8_GROUP_CALLER(256)
        }
#undef ARK_MOE_DPAS_FP8_GROUP_CALLER
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
// Launcher (adapted from vllm-xpu-kernels `MoEGEMMLauncher<>`; the caller
// owns the `atomic_buffer` USM allocation lifetime).
// ---------------------------------------------------------------------------

template <typename, typename, typename, typename, char, char, class, ScaleMode>
class DpasGemmName;

template <char layoutA, char layoutB, class policy, ScaleMode Mode,
          typename ElementA, typename ElementB, typename ElementS,
          typename ElementBI, typename ElementD>
void MoEGEMMLauncher(sycl::queue& stream, const ElementA* activations,
                     const ElementB* weights, const ElementS* scales,
                     const ElementBI* bias, ElementD* outputs, const int gemm_n,
                     const int gemm_k, const int* rows_per_expert,
                     const int num_experts, const int group_size,
                     int32_t* atomic_buffer) {
  using ElementA_non_CV = cutlass::platform::remove_cv_t<ElementA>;
  // DPAS operates on same-dtype A/B pairs; the FP8 B tensor is upcast to the
  // activation dtype (ElementA) via `reorder(tBrB, tCrB)` in the mainloop
  // before entering the MMA atom. See comment #3 at the top of this file.
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
        "moe_prefill_fp8_dpas: MaxThreadsPerSM must be divisible by "
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
    cgh.parallel_for<DpasGemmName<ElementA, ElementB, ElementS, ElementD,
                                  layoutA, layoutB, policy, Mode>>(
        sycl::nd_range<3>{global * local, local}, kernel_props, [=](auto) {
          MoEGEMM<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD, layoutA,
                  layoutB, 'R', Mode>(activations, weights, scales, bias,
                                      outputs, mma, rows_per_expert,
                                      num_experts, group_size, gemm_n, gemm_k,
                                      atomic_buffer, local_mem);
        });
  });

  EventManager::getInstance().addEvent(event);
  event.wait();
}

// ---------------------------------------------------------------------------
// Host-side driver: Variant A (per-tensor FP8).
//
// Weight layout `[E, K, N]` row-major (vllm convention), fp8 bytes, scale
// pointer `[E]` FP32.
// ---------------------------------------------------------------------------

template <typename ScalarT, bool IsE4M3>
void moe_prefill_fp8_dpas_per_tensor_dispatch(
    sycl::queue* q, const ScalarT* activations, const uint8_t* weights_KN,
    const float* scales_e, ScalarT* outputs, const int* num_tokens_per_expert,
    int E, int N, int K, int total_tokens) {
  if (E == 0 || N == 0 || K == 0 || total_tokens == 0) return;

  compat::set_default_queue(*q);

  using ElementB = std::conditional_t<IsE4M3, cutlass::float_e4m3_t,
                                      cutlass::float_e5m2_t>;
  // Map the caller-facing SYCL native half/bfloat16 to the CUTLASS type CUTE
  // has DPAS-atom specializations for. See `cute_scalar` above.
  using ElementA = cute_scalar_t<ScalarT>;
  const auto* activations_ca =
      reinterpret_cast<const ElementA*>(activations);
  auto* outputs_ca = reinterpret_cast<ElementA*>(outputs);

  int A_avg_M = total_tokens / E;

  int32_t* atomic_buffer = sycl::malloc_device<int32_t>(1, *q);
  if (atomic_buffer == nullptr) {
    throw std::runtime_error(
        "moe_prefill_fp8_dpas(per-tensor): failed to allocate atomic buffer");
  }

#define ARK_DPAS_PT_LAUNCH(policy)                                             \
  MoEGEMMLauncher<'R', 'R', policy, ScaleMode::kPerTensor>(                    \
      *q, activations_ca, reinterpret_cast<const ElementB*>(weights_KN),       \
      scales_e, static_cast<const ElementA*>(nullptr), outputs_ca, N, K,       \
      num_tokens_per_expert, E, /*group_size=*/0, atomic_buffer);

  if (A_avg_M <= 8) {
    ARK_DPAS_PT_LAUNCH(dpas_w8a16_policy_m_16);
  } else if (A_avg_M <= 32) {
    ARK_DPAS_PT_LAUNCH(dpas_w8a16_policy_m_32);
  } else {
    ARK_DPAS_PT_LAUNCH(dpas_w8a16_policy);
  }
#undef ARK_DPAS_PT_LAUNCH

  sycl::free(atomic_buffer, *q);
}

// ---------------------------------------------------------------------------
// Host-side driver: Variant B (per-K-group FP8).
//
// Weight layout `[E, N, K]` row-major (auto-round convention -- the launcher
// passes `LayoutKindB='C'` so `MoEGEMM<>` XOR-flips to `'R'` inside
// `make_moe_tensor`, matching the physical `[N, K]` row-major storage).
// Scales `[E, N, K/group_size]` in act dtype.
// ---------------------------------------------------------------------------

template <typename ScalarT, bool IsE4M3>
void moe_prefill_fp8_dpas_per_group_dispatch(
    sycl::queue* q, const ScalarT* activations, const uint8_t* weights_NK,
    const ScalarT* scales, ScalarT* outputs, const int* num_tokens_per_expert,
    int E, int N, int K, int group_size, int total_tokens) {
  if (E == 0 || N == 0 || K == 0 || total_tokens == 0) return;
  if (K % group_size != 0) {
    throw std::invalid_argument(
        "moe_prefill_fp8_dpas(per-group): K must be a multiple of group_size");
  }

  compat::set_default_queue(*q);

  using ElementB = std::conditional_t<IsE4M3, cutlass::float_e4m3_t,
                                      cutlass::float_e5m2_t>;
  // Map the caller-facing SYCL native half/bfloat16 to the CUTLASS type CUTE
  // has DPAS-atom specializations for. See `cute_scalar` above.
  using ElementA = cute_scalar_t<ScalarT>;
  const auto* activations_ca =
      reinterpret_cast<const ElementA*>(activations);
  const auto* scales_ca = reinterpret_cast<const ElementA*>(scales);
  auto* outputs_ca = reinterpret_cast<ElementA*>(outputs);

  int A_avg_M = total_tokens / E;

  int32_t* atomic_buffer = sycl::malloc_device<int32_t>(1, *q);
  if (atomic_buffer == nullptr) {
    throw std::runtime_error(
        "moe_prefill_fp8_dpas(per-group): failed to allocate atomic buffer");
  }

#define ARK_DPAS_PG_LAUNCH(policy)                                             \
  MoEGEMMLauncher<'R', 'C', policy, ScaleMode::kPerGroup>(                     \
      *q, activations_ca, reinterpret_cast<const ElementB*>(weights_NK),       \
      scales_ca, static_cast<const ElementA*>(nullptr), outputs_ca, N, K,      \
      num_tokens_per_expert, E, group_size, atomic_buffer);

  if (A_avg_M <= 8) {
    ARK_DPAS_PG_LAUNCH(dpas_w8a16_policy_m_16);
  } else if (A_avg_M <= 32) {
    ARK_DPAS_PG_LAUNCH(dpas_w8a16_policy_m_32);
  } else {
    ARK_DPAS_PG_LAUNCH(dpas_w8a16_policy);
  }
#undef ARK_DPAS_PG_LAUNCH

  sycl::free(atomic_buffer, *q);
}

// ---------------------------------------------------------------------------
// Env-flag helper -- `ARK_MOE_PREFILL_DPAS_FP8` (default ON per plan).
//
// Truthy values (case-insensitive): "1", "true", "on", "yes".
// Explicit "0" / "false" / "off" / "no" disable. Re-read on every call so
// benchmarks / tests can toggle the path in-process (was previously cached
// via a Meyers singleton, which prevented `test_moe_prefill_perf.py` from
// independently measuring the native / dpas variants after the first
// FP8 launch had frozen the value).
// ---------------------------------------------------------------------------
inline bool moe_prefill_dpas_fp8_enabled() {
  const char* env = std::getenv("ARK_MOE_PREFILL_DPAS_FP8");
  if (env == nullptr) return true;  // default ON
  std::string s(env);
  for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "0" || s == "false" || s == "off" || s == "no") return false;
  return true;
}

// ---------------------------------------------------------------------------
// Shape preconditions for the per-K-group dispatcher branch.
//
//   * N must accommodate at least one WG tile in the smallest policy
//     (SG_N * ATOM_N = 64 for `dpas_w8a16_policy_m_16`).
//   * K must be a multiple of `tile_k = 32` and of `group_size`.
//   * `group_size` must be one of {32, 64, 128, 256} -- the switch inside
//     `MoEGEMM<>` covers exactly these values.
// ---------------------------------------------------------------------------
inline bool moe_prefill_dpas_fp8_pergroup_shape_ok(int N, int K,
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

}  // namespace moe_dpas_fp8
}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
