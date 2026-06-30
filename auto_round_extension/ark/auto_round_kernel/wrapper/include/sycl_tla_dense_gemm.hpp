// SYCL-TLA Dense GEMM Wrapper
// Based on sycl-tla/examples/cute/tutorial/xe_gemm.cpp
//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
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

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#endif

#include "utils.hpp"

namespace ark {

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

namespace dense_gemm_detail {

using namespace cute;

template <class ATensor, class BTensor, class CTensor, class TiledMMA>
void gemm_device(ATensor const& A, BTensor const& B, CTensor& C, TiledMMA const& mma) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
  auto wg_m = int(item.get_group(1));
  auto wg_n = int(item.get_group(0));
  auto local_id = int(item.get_local_id(0));

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B.shape());
  Tensor cC = make_identity_tensor(C.shape());

  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));
  Tensor gC = local_tile(cC, wg_tile, wg_coord, Step<_1, _1, X>{});

  auto copy_a = make_block_2d_copy_A(mma, A);
  auto copy_b = make_block_2d_copy_B(mma, B);
  auto copy_c = make_block_2d_copy_D(mma, C);

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

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  constexpr int prefetch_dist = 3;
  constexpr SPIRVScope barrier_scope = ScopeWorkgroup;

  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  clear(tCrC);

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
    gemm(mma, tCrA, tCrB, tCrC);

    barrier_wait(barrier_scope);
  }

  copy(copy_c, tCrC, tCgC);
}

template <typename TA, typename TB, typename TC>
auto choose_mma_op() {
  if constexpr (is_complete_v<XE_DPAS_TT<8, TC, TA, TB>>) {
    return XE_DPAS_TT<8, TC, TA, TB>{};
  } else if constexpr (is_same_v<TA, cute::bfloat16_t>) {
    return XE_DPAS_TT<8, float, cute::bfloat16_t>{};
  } else {
    return XE_DPAS_TT<8, float, cute::half_t>{};
  }
}

template <class ATensor, class BTensor, class CTensor>
auto choose_tiled_mma(ATensor const& A, BTensor const& B, CTensor const&) {
  using TA = typename ATensor::element_type;
  using TB = typename BTensor::element_type;
  using TC = typename CTensor::element_type;

  auto op = choose_mma_op<TA, TB, TC>();

  constexpr bool byte = (cute::max(sizeof_bits_v<TA>, sizeof_bits_v<TB>) <= 8);
  constexpr bool a_t = is_constant_v<1, decltype(stride<0>(A))>;
  constexpr bool b_n = is_constant_v<1, decltype(stride<0>(B))>;

  constexpr bool use_1x_dpas_per_k = a_t || (byte && b_n);
  constexpr bool use_4x8_sg =
      ((sizeof_bits_v<TB> < sizeof_bits_v<TA>) && !(is_same_v<TB, cute::float_e5m2_t>)) ||
      (b_n && sizeof_bits_v<TB> < 8);

  using _K = conditional_t<use_1x_dpas_per_k, C<op.K>, C<op.K * 2>>;
  using WGTile = Shape<_256, _256, _K>;
  using SGLayout8x4 = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
  using SGLayout4x8 = Layout<Shape<_4, _8, _1>, Stride<_8, _1, _0>>;
  using SGLayout = conditional_t<use_4x8_sg, SGLayout4x8, SGLayout8x4>;
  using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>, SGLayout>::TiledMMA;

  return MMA{};
}

template <class, class, char, char>
class DenseGemmKernelName;

template <class ATensor, class BTensor, class CTensor, typename TA, typename TB, char layoutA, char layoutB>
void gemm_cute(sycl::queue* q, ATensor const& A, BTensor const& B, CTensor& C) {
  compat::set_default_queue(*q);
  auto mma = choose_tiled_mma(A, B, C);

  sycl::range<2> local = {size(mma), 1};
  sycl::range<2> global = {local[0] * ceil_div(shape<0>(B), get<1>(mma.tile_mnk())),
                           local[1] * ceil_div(shape<0>(A), get<0>(mma.tile_mnk()))};

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props{syclex::sub_group_size<16>, intelex::grf_size<256>};

  auto event = q->parallel_for<DenseGemmKernelName<TA, TB, layoutA, layoutB>>(
      sycl::nd_range<2>(global, local), kernel_props, [=](auto) { gemm_device(A, B, C, mma); });
  event.wait_and_throw();
}

template <typename Element>
void postprocess(sycl::queue* q, int m, int n, const float* accum_ptr, const Element* bias_ptr, Element* out_ptr) {
  std::size_t elems = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
  auto event = q->parallel_for(sycl::range<1>(elems), [=](sycl::id<1> idx) {
    std::size_t linear = idx[0];
    float value = accum_ptr[linear];
    if (bias_ptr != nullptr) {
      value += static_cast<float>(bias_ptr[linear % static_cast<std::size_t>(n)]);
    }
    out_ptr[linear] = static_cast<Element>(value);
  });
  event.wait_and_throw();
}

template <typename Element>
void run_dense_gemm(sycl::queue* q, int m, int n, int k, const Element* a_ptr, const Element* b_ptr, Element* c_ptr,
                    const Element* bias_ptr) {
  auto A = make_tensor(make_gmem_ptr(const_cast<Element*>(a_ptr)), make_shape(m, k), make_stride(k, _1{}));
  auto B = make_tensor(make_gmem_ptr(const_cast<Element*>(b_ptr)), make_shape(n, k), make_stride(k, _1{}));

  std::size_t accum_bytes = static_cast<std::size_t>(m) * static_cast<std::size_t>(n) * sizeof(float);
  auto* accum_ptr = static_cast<float*>(DnnlContext::Instance()->get_scratch_mem(accum_bytes, 6, q));
  if (!accum_ptr) {
    throw std::runtime_error("sycl_tla_dense_gemm: failed to allocate scratch buffer");
  }

  auto C = make_tensor(make_gmem_ptr(accum_ptr), make_shape(m, n), make_stride(n, _1{}));

  gemm_cute<decltype(A), decltype(B), decltype(C), Element, Element, 'R', 'R'>(q, A, B, C);
  postprocess(q, m, n, accum_ptr, bias_ptr, c_ptr);
}

}  // namespace dense_gemm_detail

inline void sycl_tla_dense_gemm(sycl::queue* q, int m, int n, int k, const void* a, BTLA_DTYPE at, const void* b,
                                BTLA_DTYPE bt, void* c, BTLA_DTYPE ct, const void* bias, bool BT) {
  if (!q) {
    throw std::invalid_argument("sycl_tla_dense_gemm: stream must be a valid SYCL queue");
  }
  if (!BT) {
    throw std::invalid_argument("sycl_tla_dense_gemm: only the A @ B.T contract is supported");
  }
  if (!a || !b || !c) {
    throw std::invalid_argument("sycl_tla_dense_gemm: input and output pointers must not be null");
  }
  if (m <= 0 || n <= 0 || k <= 0) {
    return;
  }
  if (at != bt) {
    throw std::invalid_argument("sycl_tla_dense_gemm: A and B must use the same dtype");
  }
  if (ct != at) {
    throw std::invalid_argument("sycl_tla_dense_gemm: output dtype must match input dtype");
  }

  switch (at) {
    case BTLA_DTYPE::F16:
      dense_gemm_detail::run_dense_gemm(q, m, n, k, static_cast<const cute::half_t*>(a),
                                        static_cast<const cute::half_t*>(b), static_cast<cute::half_t*>(c),
                                        static_cast<const cute::half_t*>(bias));
      return;
    case BTLA_DTYPE::BF16:
      dense_gemm_detail::run_dense_gemm(q, m, n, k, static_cast<const cute::bfloat16_t*>(a),
                                        static_cast<const cute::bfloat16_t*>(b),
                                        static_cast<cute::bfloat16_t*>(c),
                                        static_cast<const cute::bfloat16_t*>(bias));
      return;
    default:
      throw std::invalid_argument("sycl_tla_dense_gemm: only FP16 and BF16 are supported");
  }
}

#endif  // ARK_XPU && ARK_SYCL_TLA

}  // namespace ark