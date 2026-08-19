// SYCL-TLA S8 GEMM Wrapper
//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <stdexcept>

#ifdef ARK_XPU
#include <sycl/sycl.hpp>
#endif

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "cute/tensor.hpp"
#include "cute/util/compat.hpp"
#endif

#include "utils.hpp"

namespace ark {

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

namespace sycl_tla_s8_detail {

using namespace cute;

template <bool AccumBlock, bool HasBias, bool FullTile, class ElementOut, int TileM, int TileN, class SGLayout>
class S8DequantKernelName;

template <bool HasBias, bool FullTile, class ElementOut, int TileM, int TileN, class SGLayout>
class S8KBlockDequantKernelName;

static constexpr int S8_GEMM_TILE_K = 64;

using SmallTileSG = Layout<Shape<_1, _4, _1>, Stride<_0, _1, _0>>;
using SmallMidTileSG = Layout<Shape<_2, _4, _1>, Stride<_4, _1, _0>>;
using MediumTileSG = Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>;
using LargeTileSG = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;

template <bool HasBias, bool FullTile, class TiledMMA, class ElementOut>
void igemm_kblock_device_impl(TiledMMA const& mma, const int8_t* a, const int8_t* b, ElementOut* c,
                              const ElementOut* scale_a, const ElementOut* scale_b, const ElementOut* bias, int m,
                              int n, int k, int blocksize, int blks) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
  int wg_m = int(item.get_group(1));
  int wg_n = int(item.get_group(0));
  int local_id = int(item.get_local_id(0));

  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor cC = make_identity_tensor(make_shape(m, n));
  Tensor gC = local_tile(cC, wg_tile, wg_coord, Step<_1, _1, X>{});

  auto thr_mma = mma.get_slice(local_id);

  Tensor tCrC = partition_fragment_C(mma, select<0, 1>(wg_tile));
  Tensor tFrC = make_tensor_like<float>(tCrC);
  Tensor tCgC = thr_mma.partition_C(gC);

  CUTE_UNROLL
  for (int i = 0; i < size(tFrC); ++i) {
    tFrC(i) = 0.0f;
  }

  constexpr SPIRVScope barrier_scope = ScopeWorkgroup;
  constexpr int prefetch_dist = 3;

  constexpr int k_tile_size = S8_GEMM_TILE_K;

  auto A = make_tensor(make_gmem_ptr(const_cast<int8_t*>(a)), make_shape(m, k), make_stride(k, _1{}));
  auto B = make_tensor(make_gmem_ptr(const_cast<int8_t*>(b)), make_shape(n, k), make_stride(k, _1{}));

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B.shape());

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));

  auto copy_a = make_block_2d_copy_A(mma, A);
  auto copy_b = make_block_2d_copy_B(mma, B);

  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  int k_tiles_per_block = blocksize / k_tile_size;
  int k_tile_count = blks * k_tiles_per_block;
  int k_tile_prefetch = 0;

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist && k_tile_prefetch < k_tile_count; ++k_tile_prefetch) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }

  for (int ib = 0; ib < blks; ++ib) {
    clear(tCrC);

    for (int bk = 0; bk < k_tiles_per_block; ++bk) {
      int k_tile = ib * k_tiles_per_block + bk;

      barrier_arrive(barrier_scope);

      copy(copy_a, tAgA(_, _, _, k_tile), tArA);
      copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

      if (k_tile_prefetch < k_tile_count) {
        prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
        prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
      }
      ++k_tile_prefetch;

      reorder(tArA, tCrA);
      reorder(tBrB, tCrB);
      cute::gemm(mma, tCrA, tCrB, tCrC);

      barrier_wait(barrier_scope);
    }

    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
      auto coord = tCgC(i);
      int row = int(get<0>(coord));
      int col = int(get<1>(coord));

      if constexpr (!FullTile) {
        if (row >= m || col >= n) continue;
      }

      float sb = static_cast<float>(scale_b[col * blks + ib]);
      tFrC(i) += static_cast<float>(tCrC(i)) * sb;
    }
  }
  

  CUTE_UNROLL
  for (int i = 0; i < size(tFrC); ++i) {
    auto coord = tCgC(i);
    int row = int(get<0>(coord));
    int col = int(get<1>(coord));

    if constexpr (!FullTile) {
      if (row >= m || col >= n) continue;
    }

    float value = tFrC(i) * static_cast<float>(scale_a[row]);
    if constexpr (HasBias) {
      value += static_cast<float>(bias[col]);
    }
    c[row * n + col] = static_cast<ElementOut>(value);
  }
}

template <bool AccumBlock, bool HasBias, bool FullTile, class ATensor, class BTensor, class TiledMMA, class ElementOut>
void igemm_device_impl(ATensor const& A, BTensor const& B, TiledMMA const& mma, ElementOut* c, float* accum,
                       const ElementOut* scale_a, const ElementOut* scale_b, const ElementOut* bias, int m, int n,
                       int block_idx, int scale_b_stride) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
  int wg_m = int(item.get_group(1));
  int wg_n = int(item.get_group(0));
  int local_id = int(item.get_local_id(0));

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B.shape());
  Tensor cC = make_identity_tensor(make_shape(m, n));

  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));
  Tensor gC = local_tile(cC, wg_tile, wg_coord, Step<_1, _1, X>{});

  auto copy_a = make_block_2d_copy_A(mma, A);
  auto copy_b = make_block_2d_copy_B(mma, B);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  Tensor tCrC = partition_fragment_C(mma, select<0, 1>(wg_tile));
  Tensor tCgC = thr_mma.partition_C(gC);

  constexpr SPIRVScope barrier_scope = ScopeWorkgroup;
  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));

  clear(tCrC);
  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  constexpr int prefetch_dist = 3;
  int k_tile_prefetch = 0;

    CUTE_UNROLL
    for (; k_tile_prefetch < prefetch_dist; ++k_tile_prefetch) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }

    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile, ++k_tile_prefetch) {
    barrier_arrive(barrier_scope);

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));

    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);
    cute::gemm(mma, tCrA, tCrB, tCrC);

    barrier_wait(barrier_scope);
    }

    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
      auto coord = tCgC(i);
      int row = int(get<0>(coord));
      int col = int(get<1>(coord));

      if constexpr (!FullTile) {
        if (row >= m || col >= n) continue;
      }

      float acc = static_cast<float>(tCrC(i));
      if constexpr (AccumBlock) {
        float sb = static_cast<float>(scale_b[col * scale_b_stride + block_idx]);
        accum[row * n + col] += acc * sb;
      } else {
        float sa = static_cast<float>(scale_a[row]);
        float sb = static_cast<float>(scale_b[col]);
        float value = acc * sa * sb;
        if constexpr (HasBias) {
          value += static_cast<float>(bias[col]);
        }
        c[row * n + col] = static_cast<ElementOut>(value);
      }
    }
}

template <bool HasBias, class ElementOut, int TileM, int TileN, class SGLayout>
void launch_igemm_kblock_tile(sycl::queue* q, int m, int n, int k, const int8_t* a, const int8_t* b, ElementOut* c,
                              const ElementOut* scale_a, const ElementOut* scale_b, const ElementOut* bias,
                              int blocksize, int blks) {
  compat::set_default_queue(*q);

  using Op = XE_DPAS_TT<8, int32_t, int8_t, int8_t>;
  using WGTile = Shape<Int<TileM>, Int<TileN>, _64>;
  using MMA = typename TiledMMAHelper<MMA_Atom<Op>, Layout<WGTile>, SGLayout>::TiledMMA;
  MMA mma{};

  sycl::range<2> local = {size(mma), 1};
  sycl::range<2> global = {local[0] * ceil_div(n, get<1>(mma.tile_mnk())),
                           local[1] * ceil_div(m, get<0>(mma.tile_mnk()))};

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties props{syclex::sub_group_size<16>, intelex::grf_size<256>};

  bool full_tile = (m % TileM == 0) && (n % TileN == 0);

  if (full_tile) {
    q->parallel_for<S8KBlockDequantKernelName<HasBias, true, ElementOut, TileM, TileN, SGLayout>>(
        sycl::nd_range<2>(global, local), props, [=](auto) {
          igemm_kblock_device_impl<HasBias, true>(mma, a, b, c, scale_a, scale_b, bias, m, n, k, blocksize, blks);
        });
  } else {
    q->parallel_for<S8KBlockDequantKernelName<HasBias, false, ElementOut, TileM, TileN, SGLayout>>(
        sycl::nd_range<2>(global, local), props, [=](auto) {
          igemm_kblock_device_impl<HasBias, false>(mma, a, b, c, scale_a, scale_b, bias, m, n, k, blocksize, blks);
        });
  }
}

template <bool AccumBlock, class ElementOut, int TileM, int TileN, class SGLayout>
void launch_igemm_tile(sycl::queue* q, int m, int n, int gemm_k, int lda, int ldb, const int8_t* a, const int8_t* b,
                       ElementOut* c, float* accum, const ElementOut* scale_a, const ElementOut* scale_b,
                       const ElementOut* bias, int block_idx, int scale_b_stride) {
  compat::set_default_queue(*q);

  auto A = make_tensor(make_gmem_ptr(const_cast<int8_t*>(a)), make_shape(m, gemm_k), make_stride(lda, _1{}));
  auto B = make_tensor(make_gmem_ptr(const_cast<int8_t*>(b)), make_shape(n, gemm_k), make_stride(ldb, _1{}));

  using Op = XE_DPAS_TT<8, int32_t, int8_t, int8_t>;
  using WGTile = Shape<Int<TileM>, Int<TileN>, _64>;
  using MMA = typename TiledMMAHelper<MMA_Atom<Op>, Layout<WGTile>, SGLayout>::TiledMMA;
  MMA mma{};

  sycl::range<2> local = {size(mma), 1};
  sycl::range<2> global = {local[0] * ceil_div(n, get<1>(mma.tile_mnk())),
                           local[1] * ceil_div(m, get<0>(mma.tile_mnk()))};

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties props{syclex::sub_group_size<16>, intelex::grf_size<256>};

  bool full_tile = (m % TileM == 0) && (n % TileN == 0);
  bool has_bias = (!AccumBlock) && (bias != nullptr);

  if (full_tile) {
    if (has_bias) {
      q->parallel_for<S8DequantKernelName<AccumBlock, true, true, ElementOut, TileM, TileN, SGLayout>>(
          sycl::nd_range<2>(global, local), props, [=](auto) {
            igemm_device_impl<AccumBlock, true, true>(A, B, mma, c, accum, scale_a, scale_b, bias, m, n, block_idx,
                                                      scale_b_stride);
          });
    } else {
      q->parallel_for<S8DequantKernelName<AccumBlock, false, true, ElementOut, TileM, TileN, SGLayout>>(
          sycl::nd_range<2>(global, local), props, [=](auto) {
            igemm_device_impl<AccumBlock, false, true>(A, B, mma, c, accum, scale_a, scale_b, bias, m, n, block_idx,
                                                       scale_b_stride);
          });
    }
  } else {
    if (has_bias) {
      q->parallel_for<S8DequantKernelName<AccumBlock, true, false, ElementOut, TileM, TileN, SGLayout>>(
          sycl::nd_range<2>(global, local), props, [=](auto) {
            igemm_device_impl<AccumBlock, true, false>(A, B, mma, c, accum, scale_a, scale_b, bias, m, n, block_idx,
                                                       scale_b_stride);
          });
    } else {
      q->parallel_for<S8DequantKernelName<AccumBlock, false, false, ElementOut, TileM, TileN, SGLayout>>(
          sycl::nd_range<2>(global, local), props, [=](auto) {
            igemm_device_impl<AccumBlock, false, false>(A, B, mma, c, accum, scale_a, scale_b, bias, m, n, block_idx,
                                                        scale_b_stride);
          });
    }
  }
}

template <class ElementOut, int TileM, int TileN, class SGLayout>
void launch_igemm_kblock_tile_dispatch(sycl::queue* q, int m, int n, int k, const int8_t* a, const int8_t* b,
                                       ElementOut* c, const ElementOut* scale_a, const ElementOut* scale_b,
                                       const ElementOut* bias, int blocksize, int blks) {
  if (bias) {
    launch_igemm_kblock_tile<true, ElementOut, TileM, TileN, SGLayout>(
        q, m, n, k, a, b, c, scale_a, scale_b, bias, blocksize, blks);
  } else {
    launch_igemm_kblock_tile<false, ElementOut, TileM, TileN, SGLayout>(
        q, m, n, k, a, b, c, scale_a, scale_b, bias, blocksize, blks);
  }
}

template <class ElementOut>
void launch_igemm_kblock(sycl::queue* q, int m, int n, int k, const int8_t* a, const int8_t* b, ElementOut* c,
                         const ElementOut* scale_a, const ElementOut* scale_b, const ElementOut* bias, int blocksize,
                         int blks) {
  if (m < 16) {
    launch_igemm_kblock_tile_dispatch<ElementOut, 8, 128, SmallTileSG>(
        q, m, n, k, a, b, c, scale_a, scale_b, bias, blocksize, blks);
  } else if (m < 128) {
    launch_igemm_kblock_tile_dispatch<ElementOut, 64, 128, SmallMidTileSG>(
        q, m, n, k, a, b, c, scale_a, scale_b, bias, blocksize, blks);
  } else if (m <= 1024) {
    launch_igemm_kblock_tile_dispatch<ElementOut, 128, 128, MediumTileSG>(
        q, m, n, k, a, b, c, scale_a, scale_b, bias, blocksize, blks);
  } else {
    launch_igemm_kblock_tile_dispatch<ElementOut, 256, 128, LargeTileSG>(
        q, m, n, k, a, b, c, scale_a, scale_b, bias, blocksize, blks);
  }
}


template <bool AccumBlock, class ElementOut>
void launch_igemm(sycl::queue* q, int m, int n, int gemm_k, int lda, int ldb,
                  const int8_t* a, const int8_t* b, ElementOut* c, float* accum,
                  const ElementOut* scale_a, const ElementOut* scale_b,
                  const ElementOut* bias, int block_idx, int scale_b_stride) {
  if (m < 16) {
    launch_igemm_tile<AccumBlock, ElementOut, 8, 128, SmallTileSG>(
        q, m, n, gemm_k, lda, ldb, a, b, c, accum, scale_a, scale_b, bias, block_idx, scale_b_stride);
  } else if (m < 128) {
    launch_igemm_tile<AccumBlock, ElementOut, 64, 128, SmallMidTileSG>(
        q, m, n, gemm_k, lda, ldb, a, b, c, accum, scale_a, scale_b, bias, block_idx, scale_b_stride);
  } else if (m <= 1024) {
    launch_igemm_tile<AccumBlock, ElementOut, 128, 128, MediumTileSG>(
        q, m, n, gemm_k, lda, ldb, a, b, c, accum, scale_a, scale_b, bias, block_idx, scale_b_stride);
  } else {
    launch_igemm_tile<AccumBlock, ElementOut, 256, 256, LargeTileSG>(
        q, m, n, gemm_k, lda, ldb, a, b, c, accum, scale_a, scale_b, bias, block_idx, scale_b_stride);
  }
}

static inline void validate_s8_kblock_args(int k, int blocksize) {
  if (blocksize <= 0 || k % blocksize != 0) {
    throw std::invalid_argument("sycl_tla_igemm_s8s8_dequant: blocksize must divide k");
  }
  if (blocksize % S8_GEMM_TILE_K != 0) {
    throw std::invalid_argument("sycl_tla_igemm_s8s8_dequant: k-block blocksize must be a multiple of 64");
  }
}

template <class ElementOut>
void run_typed(sycl::queue* q, int m, int n, int k, const int8_t* a, const int8_t* b, ElementOut* c,
               const ElementOut* scale_a, const ElementOut* scale_b, const ElementOut* bias, int blocksize) {
  bool k_block = !(blocksize == k || blocksize == -1);

  if (!k_block) {
    launch_igemm<false, ElementOut>(q, m, n, k, k, k, a, b, c, nullptr, scale_a, scale_b, bias, 0, 1);
    return;
  }

  validate_s8_kblock_args(k, blocksize);

  int blks = k / blocksize;
  launch_igemm_kblock<ElementOut>(q, m, n, k, a, b, c, scale_a, scale_b, bias, blocksize, blks);
}

}  // namespace sycl_tla_s8_detail

inline void sycl_tla_igemm_s8s8_dequant(sycl::queue* q, int m, int n, int k, const void* a, const void* b, void* c,
                                         BTLA_DTYPE ct, const void* scale_a, const void* scale_b, const void* bias,
                                         int blocksize) {
  if (!q) throw std::invalid_argument("sycl_tla_igemm_s8s8_dequant: queue must not be null");
  if (!a || !b || !c || !scale_a || !scale_b) {
    throw std::invalid_argument("sycl_tla_igemm_s8s8_dequant: input pointers must not be null");
  }
  if (m <= 0 || n <= 0 || k <= 0) return;

  bool k_block = !(blocksize == k || blocksize == -1);

  switch (ct) {
    case BTLA_DTYPE::F32:
      sycl_tla_s8_detail::run_typed(q, m, n, k, static_cast<const int8_t*>(a), static_cast<const int8_t*>(b),
                                    static_cast<float*>(c), static_cast<const float*>(scale_a),
                                    static_cast<const float*>(scale_b), static_cast<const float*>(bias), blocksize);
      return;
    case BTLA_DTYPE::F16:
      sycl_tla_s8_detail::run_typed(q, m, n, k, static_cast<const int8_t*>(a), static_cast<const int8_t*>(b),
                                    static_cast<cute::half_t*>(c), static_cast<const cute::half_t*>(scale_a),
                                    static_cast<const cute::half_t*>(scale_b), static_cast<const cute::half_t*>(bias),
                                    blocksize);
      return;
    case BTLA_DTYPE::BF16:
      if (k_block) {
        throw std::invalid_argument("sycl_tla_igemm_s8s8_dequant: k-block path supports only F32/F16 output");
      }
      sycl_tla_s8_detail::run_typed(q, m, n, k, static_cast<const int8_t*>(a), static_cast<const int8_t*>(b),
                                    static_cast<cute::bfloat16_t*>(c),
                                    static_cast<const cute::bfloat16_t*>(scale_a),
                                    static_cast<const cute::bfloat16_t*>(scale_b),
                                    static_cast<const cute::bfloat16_t*>(bias), blocksize);
      return;
    default:
      throw std::invalid_argument("sycl_tla_igemm_s8s8_dequant: unsupported output dtype");
  }
}

#endif  // ARK_XPU && ARK_SYCL_TLA

}  // namespace ark