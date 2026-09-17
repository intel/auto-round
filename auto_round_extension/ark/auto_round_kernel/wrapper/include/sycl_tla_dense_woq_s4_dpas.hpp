// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// SYCL-TLA Dense WOQ S4 DPAS Wrapper

#pragma once

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#ifdef ARK_XPU
#include <sycl/sycl.hpp>
#endif

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "cute/tensor.hpp"
#include "cute/util/compat.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/integer_subbyte.h"
#include "cutlass/platform/platform.h"
#include "cutlass/util/sycl_event_manager.hpp"
#include "sycl_tla_common.hpp"

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#endif

namespace ark {

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

namespace dense_woq_s4_dpas {

using namespace cute;

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

class dpas_policy_base {
 public:
  using WGTile = Shape<_256, _256, _32>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;

  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  using GmemTiledCopyD = void;
};

class dpas_w4a16_policy : public dpas_policy_base {
 public:
  using WGTile = Shape<_128, _256, _32>;
  using SGLayout = Layout<Shape<_4, _8, _1>, Stride<_8, _1, _0>>;

  using GmemTiledCopyD = XE_STORE_2D<16, 8, 32>;
};

class dpas_w4a16_policy_m_8 : public dpas_policy_base {
 public:
  using WGTile = Shape<_8, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

class dpas_w4a16_policy_m_16 : public dpas_policy_base {
 public:
  using WGTile = Shape<_16, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

class dpas_w4a16_policy_m_32 : public dpas_policy_base {
 public:
  using WGTile = Shape<_32, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

class dpas_w4a16_dense_policy_m_4_n128 : public dpas_policy_base {
 public:
  using WGTile = Shape<_4, _128, _32>;
  using SGLayout = Layout<Shape<_1, _8, _1>, Stride<_8, _1, _0>>;
};

class dpas_w4a16_dense_policy_m_8_n128 : public dpas_policy_base {
 public:
  using WGTile = Shape<_8, _128, _32>;
  using SGLayout = Layout<Shape<_1, _8, _1>, Stride<_8, _1, _0>>;
};

class dpas_w4a16_dense_policy_m_16 : public dpas_policy_base {
 public:
  using WGTile = Shape<_16, _128, _32>;
  using SGLayout = Layout<Shape<_1, _8, _1>, Stride<_8, _1, _0>>;
};

class dpas_w4a16_dense_policy_m_32 : public dpas_policy_base {
 public:
  using WGTile = Shape<_32, _128, _32>;
  using SGLayout = Layout<Shape<_1, _8, _1>, Stride<_8, _1, _0>>;
};

class dpas_w4a16_dense_policy_m_32_n256 : public dpas_policy_base {
 public:
  using WGTile = Shape<_32, _256, _32>;
  using SGLayout = Layout<Shape<_1, _8, _1>, Stride<_8, _1, _0>>;
};

class dpas_w4a16_dense_policy_m_64_n256 : public dpas_policy_base {
 public:
  using WGTile = Shape<_64, _256, _32>;
  using SGLayout = Layout<Shape<_2, _8, _1>, Stride<_8, _1, _0>>;
};

class dpas_w4a16_dense_policy_m_128 : public dpas_policy_base {
 public:
  using WGTile = Shape<_128, _256, _32>;
  using SGLayout = Layout<Shape<_4, _8, _1>, Stride<_8, _1, _0>>;
};

inline constexpr int kMinGroupSize = 32;
inline constexpr int kMaxGroupSize = 4096;

inline bool is_supported_group_size(int group_size) {
  return group_size >= kMinGroupSize && group_size <= kMaxGroupSize &&
         (group_size & (group_size - 1)) == 0;
}

template <typename ElementA, typename ElementB, typename ElementS,
          typename ElementBI, typename ElementD, char layoutA, char layoutB,
          class policy, int GroupSize, bool ScaleGroupMajor>
class DenseWoqS4DpasName;

template <typename T, char LayoutKind>
CUTE_DEVICE auto make_dense_tensor(T* ptr, int r, int c) {
  auto shape = make_shape(r, c);
  auto gmem_ptr = make_gmem_ptr(ptr);
  if constexpr (LayoutKind == 'C') {
    return make_tensor(gmem_ptr, make_layout(shape, make_stride(_1{}, r)));
  } else {
    return make_tensor(gmem_ptr, make_layout(shape, make_stride(c, _1{})));
  }
}

template <class GmemTiledCopyA, class GmemTiledCopyB, class GmemTiledCopyC,
          class ATensor, class BTensor, class DTensor, class TiledMMA,
          typename ElementS, typename ElementBI>
CUTE_DEVICE void dense_gemm_s4_single_group(
    ATensor const& A,
    BTensor const& B,
    const ElementS* Scales,
    const ElementBI* Bias,
    DTensor& C,
    Coord<int, int, cute::Underscore, int> blk_coord,
    TiledMMA const& mma) {
  using TA = typename ATensor::element_type;
  using TB = typename BTensor::element_type;
  static_assert(std::is_same_v<TB, cutlass::uint4b_t>,
                "dense_gemm_s4_single_group: ElementB must be cutlass::uint4b_t");
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

  const int prefetch_dist = 3;
  constexpr auto barrier_scope = ScopeWorkgroup;
  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  static constexpr auto ATOM_M = get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_N = get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr auto tile_m = get<0>(wg_tile);
  static constexpr auto tile_n = get<1>(wg_tile);
  static constexpr auto SG_M = tile_m / ATOM_M;
  static constexpr auto SG_N = tile_n / ATOM_N;
  static constexpr int sg_n_strides = SG_N / sg_local_range;

  auto n_tile_start = wg_n * tile_n;
  auto sg_local_n_coord = cutlass::get_sub_group_id() % ATOM_N;
  int sg_local_id = cutlass::get_sub_group_local_id();
  int n_sg_start = sg_local_n_coord * SG_N;

  clear(tCrC);

  float sg_scale[sg_n_strides];
  CUTLASS_PRAGMA_UNROLL
  for (int sn = 0; sn < sg_n_strides; ++sn) {
    int sg_local_n = sn * sg_local_range + sg_local_id;
    sg_scale[sn] = static_cast<float>(Scales[n_tile_start + n_sg_start + sg_local_n]);
  }

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist && k_tile_prefetch < k_tile_count; k_tile_prefetch++) {
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
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tCrB.size(); ++i) {
      tCrB(i) = static_cast<TA>(static_cast<float>(tCrB(i)) - 8.0f);
    }

    cute::gemm(mma, tCrA, tCrB, tCrC);

    barrier_wait(barrier_scope);
  }

  CUTLASS_PRAGMA_UNROLL
  for (int sn = 0; sn < sg_n_strides; ++sn) {
    float s = sg_scale[sn];
    CUTLASS_PRAGMA_UNROLL
    for (int sm = 0; sm < SG_M; ++sm) {
      tCrC(sn * SG_M + sm) *= s;
    }
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

template <class GmemTiledCopyA, class GmemTiledCopyB, class GmemTiledCopyC,
          int GroupSize, bool TileAlignedGroup, bool ScaleGroupMajor,
          class ATensor, class BTensor, class DTensor, class TiledMMA,
          typename ElementS, typename ElementBI>
CUTE_DEVICE void dense_gemm_s4_pergroup(
    ATensor const& A,   // (M,K)   -- ElementA (bf16/fp16)
    BTensor const& B,   // (N,K)   -- cutlass::uint4b_t (packed nibbles)
    const ElementS* Scales,
    const ElementBI* Bias,
    DTensor& C,         // (M,N)   -- ElementA
    Coord<int, int, cute::Underscore, int> blk_coord,
    TiledMMA const& mma) {
  using TA = typename ATensor::element_type;
  using TB = typename BTensor::element_type;
  static_assert(std::is_same_v<TB, cutlass::uint4b_t>,
                "dense_gemm_s4_pergroup: ElementB must be cutlass::uint4b_t (BestLA S4_CLIP)");
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

  // Prefetch distance mirrors `xe_gemm_int_pergroup<>` for now.
  // On-hardware perf tuning may want to grow `prefetch_dist` on the
  // packed path since the B stream is half the bandwidth.
  const int prefetch_dist = 3;
  constexpr int prefetch_dist_scale =
      ScaleGroupMajor ? (GroupSize == 32 ? 5 : 4) : 3;
  constexpr auto barrier_scope = ScopeWorkgroup;
  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  static constexpr auto ATOM_M = get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_N = get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());

  static constexpr auto tile_m = get<0>(wg_tile);
  static constexpr auto tile_n = get<1>(wg_tile);
  static constexpr auto tile_k = get<2>(wg_tile);
  static constexpr int tile_k_size = int(tile_k);
  static constexpr int tiles_per_group = GroupSize / tile_k_size;

  static constexpr auto SG_M = tile_m / ATOM_M;
  static constexpr auto SG_N = tile_n / ATOM_N;

  static constexpr int sg_n_strides = SG_N / sg_local_range;

  auto n_tile_start = wg_n * tile_n;
  auto sg_local_n_coord = cutlass::get_sub_group_id() % ATOM_N;
  int sg_local_id = cutlass::get_sub_group_local_id();
  int n_sg_start = sg_local_n_coord * SG_N;
  int group_num = get<1>(A.shape()) / group_size;
  int gemm_n = shape<0>(B);

  // Group-local accumulator: same fragment shape as `tCrC`, cleared at
  // every scale-group boundary and folded into `tCrC` with a per-N-column
  // scale before being reset. Mirrors the INT8 per-group path exactly.
  auto tCrC_group = thr_mma.partition_sg_fragment_C(gC);

  clear(tCrC);
  clear(tCrC_group);

  // Per-SG per-N scale cache. Same layout / semantics as the INT8
  // per-group path.
  float sg_scale[sg_n_strides];

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist && k_tile_prefetch < k_tile_count; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }
  CUTLASS_PRAGMA_UNROLL
  for (int pg = 0; pg < prefetch_dist_scale; ++pg) {
    if (pg * group_size < shape<1>(A)) {
      if constexpr (ScaleGroupMajor) {
        auto next_scales_tensor = make_tensor(
            make_gmem_ptr(reinterpret_cast<const ElementS*>(
                Scales + pg * gemm_n + n_tile_start + n_sg_start)),
            make_layout(make_shape(Int<SG_N>{}, Int<1>{}),
                        make_stride(Int<1>{}, gemm_n)));
        auto prefetch_scales = make_block_2d_prefetch<1>(
            make_shape(Int<SG_N>{}, Int<1>{}), next_scales_tensor);
        auto thr_prefetch_scales = prefetch_scales.get_slice(sg_local_id);
        auto pSgS = thr_prefetch_scales.partition_S(
            make_identity_tensor(make_shape(Int<SG_N>{}, Int<1>{})));
        prefetch(prefetch_scales, pSgS(_, 0, 0));
      } else {
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
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

    bool is_group_start;
    int group_idx;
    if constexpr (TileAlignedGroup) {
      is_group_start = k_tile % tiles_per_group == 0;
      group_idx = k_tile / tiles_per_group;
    } else {
      is_group_start = k_tile * tile_k % group_size == 0;
      group_idx = (k_tile * tile_k) / group_size;
    }

    if (is_group_start) {
      CUTLASS_PRAGMA_UNROLL
      for (int sn = 0; sn < sg_n_strides; ++sn) {
        int sg_local_n = sn * sg_local_range + sg_local_id;
        if constexpr (ScaleGroupMajor) {
          sg_scale[sn] = static_cast<float>(
            Scales[group_idx * gemm_n + n_tile_start + n_sg_start + sg_local_n]);
        } else {
          sg_scale[sn] = static_cast<float>(
            Scales[(n_tile_start + n_sg_start + sg_local_n) * group_num + group_idx]);
        }
      }

      if ((group_idx + prefetch_dist_scale) * group_size < shape<1>(A)) {
        if constexpr (ScaleGroupMajor) {
          auto next_scales_tensor = make_tensor(
              make_gmem_ptr(reinterpret_cast<const ElementS*>(
                  Scales + (group_idx + prefetch_dist_scale) * gemm_n +
                  n_tile_start + n_sg_start)),
              make_layout(make_shape(Int<SG_N>{}, Int<1>{}),
                          make_stride(Int<1>{}, gemm_n)));
          auto prefetch_scales = make_block_2d_prefetch<1>(
              make_shape(Int<SG_N>{}, Int<1>{}), next_scales_tensor);
          auto thr_prefetch_scales = prefetch_scales.get_slice(sg_local_id);
          auto pSgS = thr_prefetch_scales.partition_S(
              make_identity_tensor(make_shape(Int<SG_N>{}, Int<1>{})));
          prefetch(prefetch_scales, pSgS(_, 0, 0));
        } else {
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
    }

    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }

    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tCrB.size(); ++i) {
      tCrB(i) = static_cast<TA>(static_cast<float>(tCrB(i)) - 8.0f);
    }

    // HOT MAINLOOP -- MMA accumulates into `tCrC_group`. Per-N scale
    // is applied ONCE at the end of the group in the fold block below.
    cute::gemm(mma, tCrA, tCrB, tCrC_group);

    bool is_group_end;
    if constexpr (TileAlignedGroup) {
      is_group_end = ((k_tile + 1) % tiles_per_group == 0) ||
                     (k_tile + 1 == k_tile_count);
    } else {
      is_group_end = (((k_tile + 1) * tile_k) % group_size == 0) ||
                     (k_tile + 1 == k_tile_count);
    }
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

template <class GmemTiledCopyA, class GmemTiledCopyB, class GmemTiledCopyD,
          char LayoutKindA, char LayoutKindB, char LayoutKindD,
          class TiledMMA, int GroupSize, bool ScaleGroupMajor,
          typename ElementA, typename ElementB, typename ElementS,
          typename ElementBI, typename ElementD>
CUTE_DEVICE void DenseWoqS4GEMM(const ElementA* Activations,
                                const ElementB* Weights,
                                const ElementS* Scales,
                                const ElementBI* Bias,
                                ElementD* Outputs,
                                TiledMMA const& mma,
                                const int32_t gemm_m,
                                const int32_t gemm_n,
                                const int32_t gemm_k) {
  constexpr char actual_layout_of_B = LayoutKindB ^ ('R' ^ 'C');

  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int wg_n = item.get_group(0);
  int wg_m = item.get_group(1);

  auto A_tensor = make_dense_tensor<ElementA, LayoutKindA>(
      const_cast<ElementA*>(Activations), gemm_m, gemm_k);
  auto B_tensor = make_dense_tensor<ElementB, actual_layout_of_B>(
      const_cast<ElementB*>(Weights), gemm_n, gemm_k);
  auto D_tensor = make_dense_tensor<ElementD, LayoutKindD>(Outputs, gemm_m,
                                                        gemm_n);
  auto tile_coord = make_coord(wg_m, wg_n, _, 0);

  if (gemm_k == GroupSize) {
    dense_gemm_s4_single_group<GmemTiledCopyA, GmemTiledCopyB,
                               GmemTiledCopyD>(A_tensor, B_tensor, Scales,
                                               Bias, D_tensor, tile_coord,
                                               mma);
  } else if constexpr (GroupSize % 32 == 0) {
    dense_gemm_s4_pergroup<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD,
                           GroupSize, true, ScaleGroupMajor>(
        A_tensor, B_tensor, Scales, Bias, D_tensor, tile_coord, mma);
  } else {
    dense_gemm_s4_pergroup<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD,
                           GroupSize, false, ScaleGroupMajor>(
        A_tensor, B_tensor, Scales, Bias, D_tensor, tile_coord, mma);
  }
}

template <char layoutA, char layoutB, class policy, int GroupSize,
          bool ScaleGroupMajor, typename ElementA, typename ElementB,
          typename ElementS, typename ElementBI, typename ElementD>
void DenseWoqS4GEMMLauncherGroup(sycl::queue& stream,
                                 const ElementA* activations,
                                 const ElementB* weights,
                                 const ElementS* scales,
                                 const ElementBI* bias,
                                 ElementD* outputs,
                                 const int gemm_m,
                                 const int gemm_n,
                                 const int gemm_k,
                                 bool wait = true) {
  compat::set_default_queue(stream);

  using ElementA_non_CV = cutlass::platform::remove_cv_t<ElementA>;
  auto op = XE_DPAS_TT<8, float, ElementA_non_CV, ElementA_non_CV>{};

  using WGTile = typename policy::WGTile;
  using SGLayout = typename policy::SGLayout;
  using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>,
                                      SGLayout>::TiledMMA;
  auto mma = MMA{};

  auto wg_tile = mma.tile_mnk();
  const int tile_m = int(get<0>(wg_tile));
  const int tile_n = int(get<1>(wg_tile));
  const int m_tiles = (gemm_m + tile_m - 1) / tile_m;
  const int n_tiles = (gemm_n + tile_n - 1) / tile_n;

  auto max_threads_per_workgroup = size(mma);
  sycl::range<3> local(1, 1, max_threads_per_workgroup);
  sycl::range<3> groups(n_tiles, m_tiles, 1);

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props{syclex::sub_group_size<16>,
                                  intelex::grf_size<256>};

  using GmemTiledCopyA = typename policy::GmemTiledCopyA;
  using GmemTiledCopyB = typename policy::GmemTiledCopyB;
  using GmemTiledCopyD = typename policy::GmemTiledCopyD;

  auto event = stream.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<DenseWoqS4DpasName<ElementA, ElementB, ElementS,
                                        ElementBI, ElementD, layoutA,
                                        layoutB, policy, GroupSize,
                                        ScaleGroupMajor>>(
        sycl::nd_range<3>{groups * local, local}, kernel_props, [=](auto) {
          DenseWoqS4GEMM<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD,
                         layoutA, layoutB, 'R', MMA, GroupSize,
                         ScaleGroupMajor>(
              activations, weights, scales, bias, outputs, mma, gemm_m, gemm_n,
              gemm_k);
        });
  });

  EventManager::getInstance().addEvent(event);
  if (wait) event.wait();
}

template <int GroupSize, char layoutA, char layoutB, class policy,
          bool ScaleGroupMajor, typename ElementA, typename ElementB,
          typename ElementS, typename ElementBI, typename ElementD>
bool DenseWoqS4GEMMLauncherDispatch(sycl::queue& stream,
                                    const ElementA* activations,
                                    const ElementB* weights,
                                    const ElementS* scales,
                                    const ElementBI* bias,
                                    ElementD* outputs,
                                    const int gemm_m,
                                    const int gemm_n,
                                    const int gemm_k,
                                    const int group_size,
                                    bool wait) {
  if (group_size == GroupSize) {
    DenseWoqS4GEMMLauncherGroup<layoutA, layoutB, policy, GroupSize,
                                ScaleGroupMajor>(
        stream, activations, weights, scales, bias, outputs, gemm_m, gemm_n,
        gemm_k, wait);
    return true;
  }
  if constexpr (GroupSize < kMaxGroupSize) {
    return DenseWoqS4GEMMLauncherDispatch<GroupSize * 2, layoutA, layoutB,
                                          policy, ScaleGroupMajor>(
        stream, activations, weights, scales, bias, outputs, gemm_m, gemm_n,
        gemm_k, group_size, wait);
  }
  return false;
}

template <char layoutA, char layoutB, class policy, bool ScaleGroupMajor,
          typename ElementA, typename ElementB, typename ElementS,
          typename ElementBI, typename ElementD>
void DenseWoqS4GEMMLauncher(sycl::queue& stream,
                            const ElementA* activations,
                            const ElementB* weights,
                            const ElementS* scales,
                            const ElementBI* bias,
                            ElementD* outputs,
                            const int gemm_m,
                            const int gemm_n,
                            const int gemm_k,
                            const int group_size,
                            bool wait = true) {
  if (!is_supported_group_size(group_size) ||
      !DenseWoqS4GEMMLauncherDispatch<kMinGroupSize, layoutA, layoutB,
                                      policy, ScaleGroupMajor>(
          stream, activations, weights, scales, bias, outputs, gemm_m, gemm_n,
          gemm_k, group_size, wait)) {
    throw std::runtime_error("dense_woq_s4_dpas: unsupported group size");
  }
}

}  // namespace dense_woq_s4_dpas

#endif  // ARK_XPU && ARK_SYCL_TLA

}  // namespace ark
