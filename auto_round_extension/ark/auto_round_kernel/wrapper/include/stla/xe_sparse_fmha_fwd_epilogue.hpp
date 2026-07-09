/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#pragma once

#include <sycl/sycl.hpp>
#include "cutlass/cutlass.h"
#include "cutlass/detail/layout.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"

#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/algorithm/tensor_algorithms.hpp"
#include "flash_attention_v2/collective/copy_block_slm.hpp"

namespace cutlass::fmha::collective {

using namespace cute;

// Sparse kernels do not consume LSE today. Keep a local stateless epilogue so
// the sparse instantiations preserve the older no-LSE codegen shape.
template <class CollectiveMainloop, class TileShapeO_, class TensorO_, class TiledCopyO_ = void>
class SparseFMHAFwdEpilogue {
 public:
  using TiledMMAPV = typename CollectiveMainloop::TiledMMAPV;
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  using TileShapeO = TileShapeO_;
  using SGPerWG = decltype(product(take<1, 4>(shape(typename TiledMMAPV::ThrLayoutVMNK{}))));

  using TensorO = TensorO_;
  using TensorO2D = decltype(TensorO_{}(append<rank_v<TensorO_>>(make_coord(_, _), 0)));
  using ElementO = typename TensorO_::value_type;

  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;
  using ElementA = typename FragA::value_type;

  using ReduceK = decltype(size<3>(typename TiledMMAPV::ThrLayoutVMNK{}));

  static auto reduce_sg_v_helper() {
    constexpr auto v_total_sg = get<1>(SGTileShapeA{}) / intel::_SGSize{};
    constexpr auto v_avail_sg = ReduceK{} / ReduceSGQ{};
    return Int<(v_total_sg > v_avail_sg) ? cute::gcd(v_total_sg, v_avail_sg) : v_total_sg>{};
  }

  using SGTileShapeA = decltype(atuple_coshape(FragA{}.tv_layout()));
  using ReduceSGQ = decltype(cute::gcd(get<0>(SGTileShapeA{}), ReduceK{}));
  using ReduceSGV = decltype(reduce_sg_v_helper());
  using ReduceSGLayout = decltype(make_identity_layout(Shape<ReduceSGQ, ReduceSGV>{}));

  using SGTileShapeO = decltype(shape_div(take<0, 2>(SGTileShapeA{}), shape(ReduceSGLayout{})));

  using ReduceFragA =
      decltype(make_subgroup_tensor<ElementA>(make_layout(select<1, 0>(SGTileShapeO{}), Stride<E<1>, E<0>>{})));
  using ReduceFragARow = decltype(reduce<1>(ReduceFragA{}, sycl::plus<void>{}));

  static auto default_tiled_copy_O_helper() {
    if constexpr (ReduceK{} == _1{}) {
      return make_block_2d_copy_D(TiledMMAPV{}, TensorO2D{});
    } else {
      return make_block_2d_copy_D_subtiled(TiledMMAPV{}, ReduceFragA{}.tv_layout(), ReduceSGLayout{}, TensorO2D{});
    }
  }

  using DefaultTiledCopyO = decltype(default_tiled_copy_O_helper());
  using TiledCopyO = conditional_t<is_void_v<TiledCopyO_>, DefaultTiledCopyO, TiledCopyO_>;

  struct Arguments {};
  struct Params {};

  using AlignedSGTileA_Q = C<((size<0>(SGTileShapeA{}) + intel::sg_size - 1) / intel::sg_size) * intel::sg_size>;

  struct SharedStorageNone {};
  struct SharedStorageReduceK {
    cute::array<ElementA, size(SGTileShapeA{}) * SGPerWG{}> a_data;
    cute::array<ElementA, AlignedSGTileA_Q{} * SGPerWG{}> a_sum_data, a_max_data;
  };

  using SharedStorage = conditional_t<(ReduceK{} > _1{}), SharedStorageReduceK, SharedStorageNone>;

 private:
  SharedStorage& shared;

 public:
  static constexpr Params to_underlying_arguments(Arguments const&, void*) { return {}; }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) { return true; }

  CUTLASS_HOST_DEVICE
  SparseFMHAFwdEpilogue(Params const&, SharedStorage& shared_) : shared(shared_) {}

  template <typename QVCoord>
  CUTLASS_DEVICE void operator()(TensorO2D const& O, FragA& tArA, FragARow& tA_max, FragARow& tA_sum, QVCoord blk_qv,
                                 int thr_id) {
    using namespace cute;
    using ElementAcc = typename FragA::element_type;

    auto [rA, rA_sum, active] = reduce_A(tArA, tA_max, tA_sum, thr_id);
    if (!active) {
      return;
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA_sum.size(); i++) {
      rA_sum(i) = ElementAcc(1) / rA_sum(i);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA.size(); i++) {
      rA(i) *= broadcast<0>(rA_sum, rA, i);
    }

    Tensor cO = make_identity_tensor(O.shape());
    Tensor gO = local_tile(cO, TileShapeO{}, blk_qv);

    TiledCopyO copy_o{O};
    auto thr_copy_o = copy_o.get_slice(thr_id);
    auto tOrO = thr_copy_o.partition_sg_fragment_S(gO);
    auto tOgO = thr_copy_o.partition_D(gO);

    reorder(rA, tOrO);
    copy(copy_o, tOrO, tOgO);
  }

  template <typename FragA_, typename FragARow_>
  CUTLASS_DEVICE decltype(auto) reduce_A(FragA_& tArA, FragARow_& tA_max, FragARow_& tA_sum, int thr_id) {
    using namespace sycl::ext::oneapi::this_work_item;

    if constexpr (ReduceK{} == _1{}) {
      return std::make_tuple(tArA, tA_sum, true);
    } else {
      auto thr_vak = group<1, 3>(TiledMMAPV{}.get_thr_layout_vmnk()).get_flat_coord(assert_uniform(thr_id));
      auto a_tile = get<1>(thr_vak);
      auto k_blk = get<2>(thr_vak);

      auto shape_A = append(append(SGTileShapeA{}, ReduceK{}), SGPerWG{} / ReduceK{});
      auto shape_A_row = make_shape(get<0>(SGTileShapeO{}), shape(ReduceSGLayout{}), ReduceK{}, SGPerWG{} / ReduceK{});

      auto sA_layout = group<2, 4>(flat_divide(make_ordered_layout(shape_A, Step<_1, _0, _2, _3>{}), SGTileShapeO{}));
      auto sA_row_stride = make_stride(_1{}, make_stride(get<0>(shape_A_row), _0{}), AlignedSGTileA_Q{},
                                       AlignedSGTileA_Q{} * ReduceK{});
      auto sA_row_layout = make_layout(shape_A_row, sA_row_stride);

      auto basis2 = make_basis_like(SGTileShapeO{});
      auto sA_coords = make_layout(append(SGTileShapeO{}, shape(ReduceSGLayout{})),
                                   append(basis2, product_each(zip(SGTileShapeO{}, basis2))));

      auto sA = make_tensor(make_smem_ptr<ElementA>(&shared.a_data), sA_layout);
      auto sA_max = make_tensor(make_smem_ptr<ElementA>(&shared.a_max_data), sA_row_layout);
      auto sA_sum = make_tensor(make_smem_ptr<ElementA>(&shared.a_sum_data), sA_row_layout);

      copy_block_r2s(tA_max, sA_max(_, _, k_blk, a_tile));
      barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);
      copy_block_r2s(tA_sum, sA_sum(_, _, k_blk, a_tile));
      copy_block_r2s(tArA, sA(_, _, _, k_blk, a_tile), sA_coords);

      bool active = (k_blk < size(ReduceSGLayout{})) || (ReduceK{} == size(ReduceSGLayout{}));

      barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);
      barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);

      ReduceFragA rA;
      ReduceFragARow rA_sum, rA_max, rA_kmax[ReduceK{}];

      if (active) {
        CUTLASS_PRAGMA_UNROLL
        for (int kr = 0; kr < ReduceK{}; kr++) {
          copy_block_s2r(sA_max(_, k_blk, kr, a_tile), rA_kmax[kr]);
        }

        rA_max = rA_kmax[0];
        for (int kr = 1; kr < ReduceK{}; kr++) {
          cute::transform(rA_max, rA_kmax[kr], rA_max, cute::max_fn{});
        }

        for (int kr = 0; kr < ReduceK{}; kr++) {
          cute::transform(rA_max, rA_kmax[kr], rA_kmax[kr],
                          [](auto gmax, auto kmax) { return sycl::native::exp2(kmax - gmax); });
        }
      }

      barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);

      if (active) {
        clear(rA_sum);

        CUTLASS_PRAGMA_UNROLL
        for (int kr = 0; kr < ReduceK{}; kr++) {
          ReduceFragARow rA_sum_read;
          copy_block_s2r(sA_sum(_, k_blk, kr, a_tile), rA_sum_read);

          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < rA_sum_read.size(); i++) {
            rA_sum(i) += rA_sum_read(i) * rA_kmax[kr](i);
          }
        }

        clear(rA);

        CUTLASS_PRAGMA_UNROLL
        for (int kr = 0; kr < ReduceK{}; kr++) {
          ReduceFragA rA_read;
          copy_block_s2r(sA(_, _, k_blk, kr, a_tile), sA_coords(_, _, 0), rA_read);

          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < rA_read.size(); i++) {
            rA(i) += rA_read(i) * broadcast<0>(rA_kmax[kr], rA, i);
          }
        }
      }

      return std::make_tuple(rA, rA_sum, active);
    }
  }
};

}  // namespace cutlass::fmha::collective
