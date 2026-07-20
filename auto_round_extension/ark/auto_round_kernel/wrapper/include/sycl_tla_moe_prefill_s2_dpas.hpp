// SYCL MoE Prefill — S2 (sym) mixed-input DPAS Grouped GEMM (Variant B)
// (Fork of `sycl_tla_moe_prefill_s4_dpas.hpp` per-group mainloop)
//
// STATUS: NEEDS-HARDWARE-VALIDATION -- untested single-pass port. Precedence
// and gating in `sycl_tla_moe_mixed.hpp` fall back to the S2->S8 upcast +
// INT8 DPAS path when this header's env gate is off, so a regression can
// be neutralised at runtime without a rebuild.
// ---------------------------------------------------------------------------
// Design rationale
// ----------------
// The prior S2-sym prefill path materialised `[E, N, K]` `int8_t` upcast
// bytes into the `dequant_workspace` (`launch_upcast_int2_sym_to_int8`)
// and then handed the buffer to `moe_prefill_int_dpas_per_group_dispatch`.
// The upcast pass writes `E * N * K` bytes and the mainloop then reads
// them back through L2 -- essentially quadrupling the B-side global-memory
// traffic vs. a direct packed-crumb read (a 2-bit weight is one quarter of
// an int8 byte). This header removes the round-trip: the mainloop reads
// packed `[E, N, K/4]` `uint8_t` (four 2-bit fields per byte, sym-signed
// [-2, 1]) and upcasts to the activation dtype in registers via the same
// `cute::reorder(tBrB, tCrB)` machinery the S4 header uses -- the only
// substantive difference is that CuTe/cutlass-sycl's
// `NumericArrayConverter<ElementA, cutlass::int2b_t, N>` unpacks 2-bit
// fields four-per-byte from the loaded fragment. The B-side global load
// is a quarter of the bytes (`E * N * K / 4`) so the mainloop is
// bandwidth-bound on a quarter of the INT8-upcast traffic.
//
// Numerical parity
// ----------------
// The auto-round S2_CLIP encoding packs four 2-bit sym-signed fields per
// byte: field j (0..3) at K index 4*i+j occupies bits [2j+1 : 2j] and is
// sign-extended from [-2, 1]. This is bit-identical to `cutlass::int2b_t`
// (== `integer_subbyte<2, true>`) storage, so reinterpret-casting the
// packed `uint8_t` pointer to `cutlass::int2b_t*` and letting the
// `NumericArrayConverter` sign-extend each field reproduces exactly the
// values that `moe_dequant::decode_int2_quad<Asym=false>` computes on the
// decode / dequant paths. Asym S2 is intentionally NOT supported here (see
// the S4 header's asym rationale) and falls through to the generic dequant
// + `moe_gemm` path in `sycl_tla_moe_mixed.hpp`.
//
// Path taxonomy
// -------------
// This header exposes ONE fused S2 grouped-GEMM entry point:
//
//   * Variant B (s2, per-K-group, sym) -- weight scales `[E, N, K/GS]`
//     in the activation dtype (half / bfloat16). Weights `[E, N, K/4]`
//     packed uint8_t (4 crumbs per byte, sym-signed [-2, 1]). The
//     launcher passes `LayoutKindB='C'` so `MoEGEMM_s2<>` XOR-flips to
//     `'R'` inside `make_moe_tensor`, matching the physical `[N, K/4]`
//     row-major storage.
//
// Variant A (per-tensor) is intentionally not implemented -- auto-round
// never ships a per-tensor S2 scale.
//
// On-hardware open questions (must be resolved on first build):
//   1. Whether the pinned cutlass-sycl exposes a
//      `NumericArrayConverter<half_t / bfloat16_t, int2b_t, N>`
//      specialisation. Upstream cutlass ships `int2b_t` under
//      `cutlass/integer_subbyte.h`; cutlass-sycl may need the same
//      pull-in for the 2-bit converter. If the converter is missing,
//      `reorder(tBrB, tCrB)` will fail to instantiate at compile time --
//      the failure mode is a "no matching function" error localised to
//      the `reorder(tBrB, tCrB)` line in `xe_gemm_s2_pergroup`. In that
//      case disable `ARK_MOE_PREFILL_DPAS_S2` (falls back to the S2->S8
//      upcast + INT8 DPAS path) until the converter is available.
//   2. Whether `XE_DPAS_TT<8, float, ElementA, ElementA>` (the atom used
//      by the S4 / INT8 headers, keeping A/B as the SAME activation
//      dtype after the upcast) is still the right atom here. It should
//      be -- once `reorder` has upcast the packed-crumb fragment to
//      `ElementA` the atom's operand dtypes match A exactly.
//   3. Whether CuTe's block-2d copy atom for a 2-bit storage type loads
//      the packed bytes correctly (tile_k = 32 elements -> 8 bytes per
//      SG per k_tile, deduced from `sizeof_bits<int2b_t>::value == 2`).
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

// The S4 DPAS header pulls in the full cutlass-sycl / CuTe include set
// (via `sycl_tla_moe_prefill_int_dpas.hpp` -> `..._fp8_dpas.hpp`) and
// defines the policy classes, scalar-type mapping (`cute_scalar`), and
// per-group scale machinery we need. Include it first so we inherit all
// of that; the S2 driver lives in a sibling namespace to avoid ODR
// clashes.
#include "sycl_tla_moe_prefill_s4_dpas.hpp"

// `cutlass::int2b_t` is the storage-narrow signed 2-bit type upstream
// cutlass / cutlass-sycl uses to trigger the packed-crumb copy /
// `NumericArrayConverter` code paths. Its `sizeof` is defined as 1 byte
// (one storage byte holds four elements); CuTe uses the type's
// `bits_per_element` trait, not `sizeof`, to derive copy/atom strides.
#include "cutlass/integer_subbyte.h"

namespace ark {
namespace moe_dpas_s2 {

using namespace cute;

// Re-export the FP8/S4 policy classes / helpers into the S2 namespace so
// callsites read cleanly. The classes carry no per-dtype state.
using ::ark::moe_dpas_fp8::cute_scalar;
using ::ark::moe_dpas_fp8::cute_scalar_t;
using ::ark::moe_dpas_fp8::dpas_w8a16_policy;
using ::ark::moe_dpas_fp8::dpas_w8a16_policy_m_16;
using ::ark::moe_dpas_fp8::dpas_w8a16_policy_m_32;
using ::ark::moe_dpas_fp8::make_moe_tensor;

// ---------------------------------------------------------------------------
// Variant B -- per-K-group S2 (sym) mainloop.
//
// Structurally identical to `moe_dpas_s4::xe_gemm_s4_pergroup<>` except
// `ElementB` is required to be `cutlass::int2b_t`. The 2-bit-per-element
// storage triggers CuTe's packed-crumb copy atom and the
// `NumericArrayConverter<ElementA, int2b_t, N>` specialisation inside
// `reorder(tBrB, tCrB)`, which decodes each byte into four sign-extended
// `ElementA` (bf16/fp16) values in-register. The B fragment size is
// unchanged in element units -- CuTe deduces it from the MMA tile shape
// and the `bits_per_element` trait on `int2b_t`, so a `tile_k` of 32
// loads 8 bytes per SG per k_tile. Per-group scale reload / fold cadence
// is bit-identical to the S4 per-group path.
// ---------------------------------------------------------------------------

template <class GmemTiledCopyA, class GmemTiledCopyB, class GmemTiledCopyC,
          int GroupSize, class ATensor, class BTensor, class DTensor,
          class TiledMMA, typename ElementS, typename ElementBI>
CUTE_DEVICE void xe_gemm_s2_pergroup(
    ATensor const& A,   // (M,K)   -- ElementA (bf16/fp16)
    BTensor const& B,   // (N,K)   -- cutlass::int2b_t (packed crumbs)
    const ElementS* Scales,
    const ElementBI* Bias,
    DTensor& C,         // (M,N)   -- ElementA
    Coord<int, int, cute::Underscore, int> blk_coord,
    TiledMMA const& mma) {
  using TA = typename ATensor::element_type;
  using TB = typename BTensor::element_type;
  static_assert(std::is_same_v<TB, cutlass::int2b_t>,
                "xe_gemm_s2_pergroup: ElementB must be cutlass::int2b_t (sym only)");
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

  // Prefetch distance mirrors `xe_gemm_s4_pergroup<>` for now. On-hardware
  // perf tuning may want to grow `prefetch_dist` further on the packed S2
  // path since the B stream is a quarter of the INT8-upcast bandwidth.
  const int prefetch_dist = 3;
  const int prefetch_dist_scale = 3;
  constexpr auto barrier_scope = ScopeWorkgroup;
  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  static constexpr auto ATOM_M = get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_N = get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());

  static constexpr auto tile_m = get<0>(wg_tile);
  static constexpr auto tile_n = get<1>(wg_tile);
  static constexpr auto tile_k = get<2>(wg_tile);

  static constexpr auto SG_M = tile_m / ATOM_M;
  static constexpr auto SG_N = tile_n / ATOM_N;

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
  // scale before being reset. Mirrors the S4 per-group path exactly.
  auto tCrC_group = thr_mma.partition_sg_fragment_C(gC);

  clear(tCrC);
  clear(tCrC_group);

  // Per-SG per-N scale cache. Same layout / semantics as the S4 path.
  float sg_scale[sg_n_strides];

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

    // Group-boundary scale reload. Same math as the S4 per-group path:
    // `tile_k` is expressed in element units (crumbs), not bytes, so
    // `k_tile * tile_k` is the reduction position in *element* space and
    // the modulo test against `group_size` matches the scale-tensor
    // layout `[E, N, K/group_size]` unchanged.
    if (k_tile * tile_k % group_size == 0) {
      int group_idx = (k_tile * tile_k) / group_size;
      CUTLASS_PRAGMA_UNROLL
      for (int sn = 0; sn < sg_n_strides; ++sn) {
        int sg_local_n = sn * sg_local_range + sg_local_id;
        sg_scale[sn] = static_cast<float>(
            Scales[(n_tile_start + n_sg_start + sg_local_n) * group_num + group_idx]);
      }

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

    // `reorder` performs the in-register `int2b_t -> ElementA` unpack
    // + sign-extend + cast via `cutlass::NumericArrayConverter<
    // ElementA, cutlass::int2b_t, N>`. Once `tCrB` carries bf16/fp16
    // values it is compatible with the same DPAS atom used by the S4 /
    // INT8 per-group paths. See the header preamble open-question (1) --
    // if the pinned cutlass-sycl is missing this converter specialisation
    // this line is where the build fails.
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
// Persistent scheduler (fork of `moe_dpas_s4::MoEGEMM_s4<>`).
//
// The only substantive difference vs. the S4 scheduler is the per-expert
// `B_offset` computation: `int2b_t` storage is 2 bits per element, so
// `expert_id * gemm_n * gemm_k` elements = `expert_id * gemm_n * gemm_k /
// 4` bytes. Pointer arithmetic on `ElementB*` (where
// `sizeof(cutlass::int2b_t) == 1` byte and each byte holds four elements)
// means adding `B_offset` in *element* units to the raw pointer would
// quadruple-count -- so we divide by 4 here.
// ---------------------------------------------------------------------------

template <class GmemTiledCopyA, class GmemTiledCopyB, class GmemTiledCopyD,
          char LayoutKindA, char LayoutKindB, char LayoutKindD,
          class TiledMMA, typename ElementA, typename ElementB,
          typename ElementS, typename ElementBI, typename ElementD>
CUTE_DEVICE void MoEGEMM_s2(const ElementA* Activations,
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
    // 2-bit branch: `gemm_n * gemm_k` is the *element* count for one
    // expert; the packed byte count is a quarter of that. `ElementB*`
    // pointer arithmetic advances by whole bytes, so we quarter the
    // offset here. `gemm_k` is guaranteed a multiple of 4 by the shape
    // gate below.
    int64_t B_offset = static_cast<int64_t>(expert_id) *
                       static_cast<int64_t>(gemm_n) *
                       static_cast<int64_t>(gemm_k) / 4;

    ElementA* ptr_A_curr_batch =
        const_cast<ElementA*>(Activations) + pre_rows * gemm_k;
    ElementB* ptr_B_curr_batch = const_cast<ElementB*>(Weights) + B_offset;
    ElementD* ptr_D_curr_batch = Outputs + pre_rows * gemm_n;

    // Per-group scale offset in ElementS units. Layout `[E, N, K/GS]`.
    ElementS* ptr_Scales_curr_batch = nullptr;
    int64_t scale_expert_stride =
        static_cast<int64_t>(gemm_n) * (gemm_k / group_size);
    ptr_Scales_curr_batch =
        const_cast<ElementS*>(Scales) + expert_id * scale_expert_stride;

    ElementBI* ptr_Bias_curr_batch = nullptr;
    if (Bias != static_cast<ElementBI*>(nullptr)) {
      ptr_Bias_curr_batch = const_cast<ElementBI*>(Bias) + expert_id * gemm_n;
    }

    auto A_tensor = make_moe_tensor<ElementA, LayoutKindA>(ptr_A_curr_batch,
                                                           gemm_m, gemm_k);
    // B tensor extents are in *element* units (crumbs) -- CuTe deduces
    // the byte-space stride from `sizeof_bits<int2b_t>::value == 2`.
    auto B_tensor = make_moe_tensor<ElementB, actual_layout_of_B>(
        ptr_B_curr_batch, gemm_n, gemm_k);
    auto D_tensor = make_moe_tensor<ElementD, LayoutKindD>(ptr_D_curr_batch,
                                                           gemm_m, gemm_n);

    while (group_m_id < cumsum_tiles_for_experts) {
      int n_coord = (group_id * wg_tile_n) % gemm_n_pad / wg_tile_n;
      int m_coord = (group_m_id - pre_tiles);
      auto tile_coord = make_coord(m_coord, n_coord, _, 0);

// Per-K-group dispatch on the runtime group_size, mirrors the S4 header's
// macro. Only per-group is supported for S2 (no per-tensor variant is
// instantiated here).
#define ARK_MOE_DPAS_S2_GROUP_CALLER(GS)                                      \
  xe_gemm_s2_pergroup<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD, GS>(    \
      A_tensor, B_tensor, ptr_Scales_curr_batch, ptr_Bias_curr_batch,         \
      D_tensor, tile_coord, mma);
      if (group_size == 32) {
        ARK_MOE_DPAS_S2_GROUP_CALLER(32)
      } else if (group_size == 64) {
        ARK_MOE_DPAS_S2_GROUP_CALLER(64)
      } else if (group_size == 128) {
        ARK_MOE_DPAS_S2_GROUP_CALLER(128)
      } else if (group_size == 256) {
        ARK_MOE_DPAS_S2_GROUP_CALLER(256)
      }
#undef ARK_MOE_DPAS_S2_GROUP_CALLER

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
// Launcher (fork of `moe_dpas_s4::MoEGEMMLauncher_s4<>`).
// ---------------------------------------------------------------------------

template <typename, typename, typename, typename, char, char, class>
class DpasGemmS2Name;

template <char layoutA, char layoutB, class policy,
          typename ElementA, typename ElementB, typename ElementS,
          typename ElementBI, typename ElementD>
void MoEGEMMLauncher_s2(sycl::queue& stream, const ElementA* activations,
                        const ElementB* weights, const ElementS* scales,
                        const ElementBI* bias, ElementD* outputs,
                        const int gemm_n, const int gemm_k,
                        const int* rows_per_expert, const int num_experts,
                        const int group_size, int32_t* atomic_buffer) {
  using ElementA_non_CV = cutlass::platform::remove_cv_t<ElementA>;
  // DPAS atom keeps its bf16/fp16 x bf16/fp16 -> fp32 shape; the S2 B
  // tensor is upcast to ElementA in `reorder(tBrB, tCrB)` in the mainloop
  // before entering the MMA atom. See open question #2 in the header
  // preamble.
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
        "moe_prefill_s2_dpas: MaxThreadsPerSM must be divisible by "
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
    cgh.parallel_for<DpasGemmS2Name<ElementA, ElementB, ElementS, ElementD,
                                    layoutA, layoutB, policy>>(
        sycl::nd_range<3>{global * local, local}, kernel_props, [=](auto) {
          MoEGEMM_s2<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD, layoutA,
                     layoutB, 'R'>(
              activations, weights, scales, bias, outputs, mma,
              rows_per_expert, num_experts, group_size, gemm_n, gemm_k,
              atomic_buffer, local_mem);
        });
  });

  EventManager::getInstance().addEvent(event);
  event.wait();
}

// ---------------------------------------------------------------------------
// Host-side driver: S2 Variant B (per-K-group, sym only).
//
// Weight layout `[E, N, K/4]` row-major packed uint8_t -- four sym-signed
// 2-bit fields per byte, matching the auto-round S2_CLIP encoding. On
// entry the raw uint8_t pointer is reinterpret-cast to
// `cutlass::int2b_t*` so CuTe's copy atom / `NumericArrayConverter` can
// decode the packed crumbs directly. `LayoutKindB='C'` at the launcher
// level so `MoEGEMM_s2<>` XOR-flips to `'R'` inside `make_moe_tensor`,
// matching the physical `[N, K/4]` row-major storage.
// Scales `[E, N, K/group_size]` in act dtype (half / bfloat16).
// ---------------------------------------------------------------------------

template <typename ScalarT>
void moe_prefill_s2_dpas_per_group_dispatch(
    sycl::queue* q, const ScalarT* activations, const uint8_t* weights_NKp,
    const ScalarT* scales, ScalarT* outputs,
    const int* num_tokens_per_expert, int E, int N, int K, int group_size,
    int total_tokens) {
  if (E == 0 || N == 0 || K == 0 || total_tokens == 0) return;
  if (K % group_size != 0) {
    throw std::invalid_argument(
        "moe_prefill_s2_dpas(per-group): K must be a multiple of group_size");
  }
  if ((K & 0x3) != 0) {
    throw std::invalid_argument(
        "moe_prefill_s2_dpas(per-group): K must be a multiple of 4 (packed crumbs)");
  }

  compat::set_default_queue(*q);

  // Map the caller-facing SYCL native half/bfloat16 to the CUTLASS type CUTE
  // has DPAS-atom specializations for. See `cute_scalar` in the FP8 header.
  using ElementA = cute_scalar_t<ScalarT>;
  const auto* activations_ca =
      reinterpret_cast<const ElementA*>(activations);
  const auto* scales_ca = reinterpret_cast<const ElementA*>(scales);
  auto* outputs_ca = reinterpret_cast<ElementA*>(outputs);
  // `cutlass::int2b_t` is a 2-bit-per-element storage type whose sizeof
  // is 1 (one byte holds four elements). CuTe reads the bit-width from
  // `sizeof_bits<int2b_t>::value == 2` when computing copy strides, so
  // reinterpret-casting the packed uint8_t pointer is safe.
  const auto* weights_i2 =
      reinterpret_cast<const cutlass::int2b_t*>(weights_NKp);

  int A_avg_M = total_tokens / E;

  int32_t* atomic_buffer = sycl::malloc_device<int32_t>(1, *q);
  if (atomic_buffer == nullptr) {
    throw std::runtime_error(
        "moe_prefill_s2_dpas(per-group): failed to allocate atomic buffer");
  }

#define ARK_DPAS_S2_PG_LAUNCH_SYM(policy)                                      \
  MoEGEMMLauncher_s2<'R', 'C', policy>(                                        \
      *q, activations_ca, weights_i2, scales_ca,                               \
      static_cast<const ElementA*>(nullptr), outputs_ca, N, K,                 \
      num_tokens_per_expert, E, group_size, atomic_buffer);

  if (A_avg_M <= 8) {
    ARK_DPAS_S2_PG_LAUNCH_SYM(dpas_w8a16_policy_m_16);
  } else if (A_avg_M <= 32) {
    ARK_DPAS_S2_PG_LAUNCH_SYM(dpas_w8a16_policy_m_32);
  } else {
    ARK_DPAS_S2_PG_LAUNCH_SYM(dpas_w8a16_policy);
  }
#undef ARK_DPAS_S2_PG_LAUNCH_SYM

  sycl::free(atomic_buffer, *q);
}

// ---------------------------------------------------------------------------
// Env-flag helper -- `ARK_MOE_PREFILL_DPAS_S2` (default ON, semantics
// identical to `moe_prefill_dpas_s4_enabled`). Decoupled from
// `ARK_MOE_PREFILL_DPAS_S4` / `ARK_MOE_PREFILL_DPAS_INT8` so this new
// single-pass path can be disabled in isolation if it regresses --
// switching S2 off falls back to the S2->S8 upcast + INT8 DPAS path which
// is itself gated by `ARK_MOE_PREFILL_DPAS_INT8`.
//
// Truthy values (case-insensitive): "1", "true", "on", "yes". Explicit
// "0" / "false" / "off" / "no" disable. Re-read on every call so
// benchmarks / tests can toggle the path in-process.
// ---------------------------------------------------------------------------
inline bool moe_prefill_dpas_s2_enabled() {
  const char* env = std::getenv("ARK_MOE_PREFILL_DPAS_S2");
  if (env == nullptr) return true;  // default ON
  std::string s(env);
  for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "0" || s == "false" || s == "off" || s == "no") return false;
  return true;
}

// ---------------------------------------------------------------------------
// Shape preconditions for the per-K-group S2 dispatcher. Mirrors the S4
// per-group predicate but requires K to be a multiple of 4 so packed
// crumbs never straddle a byte boundary -- guaranteed for auto-round
// S2_CLIP callers by construction, checked here for defensive parity with
// the launcher's throw.
// ---------------------------------------------------------------------------
inline bool moe_prefill_dpas_s2_pergroup_shape_ok(int N, int K,
                                                  int group_size) {
  if (N <= 0 || K <= 0 || group_size <= 0) return false;
  if (N % 64 != 0) return false;
  if (K % 32 != 0) return false;
  if ((K & 0x3) != 0) return false;
  if (K % group_size != 0) return false;
  if ((group_size & 0x3) != 0) return false;
  if (group_size != 32 && group_size != 64 && group_size != 128 &&
      group_size != 256) {
    return false;
  }
  return true;
}

}  // namespace moe_dpas_s2
}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
