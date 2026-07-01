// SYCL MoE Prefill — INT8 per-tensor mixed-input DPAS Grouped GEMM
// (Variant A, mirrors `sycl_tla_moe_prefill_fp8_dpas.hpp`)
//
// STATUS: NEEDS-HARDWARE-VALIDATION -- this header is a mode-2 port
// ---------------------------------------------------------------------------
// This file adapts the FP8 Variant A DPAS grouped-GEMM path
// (`sycl_tla_moe_prefill_fp8_dpas.hpp`) to INT8 weight storage. The design
// intent is deliberately narrow: **weights are stored as one signed byte
// per element** (`int8_t`, sym per-tensor), **activations stay FP16/BF16**,
// and the DPAS atom keeps its `bf16/fp16 × bf16/fp16 → fp32` shape. The
// mainloop upcasts each INT8 weight byte to the activation dtype in-register
// (via `cute::reorder(tBrB, tCrB)`, which delegates to
// `cutlass::NumericArrayConverter` for the int→float leg), then feeds the
// upcast fragment into the same DPAS MMA as FP8 / bf16. The per-expert
// FP32 scale is folded once per output element in the epilogue, exactly as
// the FP8 Variant A `xe_gemm<>` does with `tCrC(i) *= B_scale`.
//
// This is Path A of the discussion attached to the auto-round issue that
// introduced this header: "INT8 storage + bf16/fp16 DPAS compute". It is
// the correct precursor for INT4 / INT2 per-group DPAS (Phase 2 & 3 of the
// plan): those paths reuse this file's mainloop / launcher / scheduler
// skeleton and only add a bit-unpack step ahead of `reorder(tBrB, tCrB)`.
//
// Path taxonomy
// -------------
// This header adds ONE fused INT8 grouped-GEMM entry point alongside the
// pre-existing FP8 Variant A / B in `sycl_tla_moe_prefill_fp8_dpas.hpp`:
//
//   * Variant A (int8) -- per-tensor INT8. Weight scales are `[E]`
//     (one FP32 scalar per expert). Weight layout `[E, K, N]` row-major
//     `int8_t` (vllm-xpu-kernels convention, transposed relative to
//     auto-round's default per-group `[E, N, K]` int8 layout so callers
//     that already produce `[E, K, N]` FP8 weights need only swap dtype).
//     Epilogue is `tCrC(i) *= Scales[expert_id]`.
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
// Persistent scheduler (adapted from `moe_dpas_fp8::MoEGEMM<>`).
//
// Structural differences relative to the FP8 scheduler:
//
//   * Only `ScaleMode::kPerTensor` is instantiated; INT8 per-group DPAS is
//     a follow-up phase and gets its own mainloop template. To keep the
//     hot path identical (and reduce the review surface for Phase 1) we
//     drop the runtime `group_size` switch here entirely.
//   * `B_offset` uses `int8_t` element units (one byte per element) --
//     identical arithmetic to the FP8 branch, but noted explicitly for
//     future readers importing this file for INT4 / INT2 where the
//     element-vs-byte distinction matters.
//   * Mainloop dispatch is `xe_gemm_int_pertensor<>` (no fp8 fallback).
// ---------------------------------------------------------------------------

template <class GmemTiledCopyA, class GmemTiledCopyB, class GmemTiledCopyD,
          char LayoutKindA, char LayoutKindB, char LayoutKindD,
          class TiledMMA, typename ElementA, typename ElementB,
          typename ElementS, typename ElementBI, typename ElementD>
CUTE_DEVICE void MoEGEMM_int(const ElementA* Activations,
                             const ElementB* Weights,
                             const ElementS* Scales, const ElementBI* Bias,
                             ElementD* Outputs, TiledMMA const& mma,
                             const int* rows_per_expert,
                             const int32_t num_experts, const int32_t gemm_n,
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

    // Per-tensor: `[E]` -- one FP32 scalar per expert.
    ElementS* ptr_Scales_curr_batch =
        const_cast<ElementS*>(Scales) + expert_id;

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

      xe_gemm_int_pertensor<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD>(
          A_tensor, B_tensor, ptr_Scales_curr_batch, ptr_Bias_curr_batch,
          D_tensor, tile_coord, mma);

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

template <typename, typename, typename, typename, char, char, class>
class DpasGemmIntName;

template <char layoutA, char layoutB, class policy,
          typename ElementA, typename ElementB, typename ElementS,
          typename ElementBI, typename ElementD>
void MoEGEMMLauncher_int(sycl::queue& stream, const ElementA* activations,
                         const ElementB* weights, const ElementS* scales,
                         const ElementBI* bias, ElementD* outputs,
                         const int gemm_n, const int gemm_k,
                         const int* rows_per_expert, const int num_experts,
                         int32_t* atomic_buffer) {
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
                                     layoutA, layoutB, policy>>(
        sycl::nd_range<3>{global * local, local}, kernel_props, [=](auto) {
          MoEGEMM_int<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD, layoutA,
                      layoutB, 'R'>(activations, weights, scales, bias,
                                    outputs, mma, rows_per_expert,
                                    num_experts, gemm_n, gemm_k,
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
  MoEGEMMLauncher_int<'R', 'R', policy>(                                       \
      *q, activations_ca, weights_KN, scales_e,                                \
      static_cast<const ElementA*>(nullptr), outputs_ca, N, K,                 \
      num_tokens_per_expert, E, atomic_buffer);

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

}  // namespace moe_dpas_int
}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
