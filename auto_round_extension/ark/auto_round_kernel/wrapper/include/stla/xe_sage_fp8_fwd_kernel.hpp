/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#pragma once

#include "cute/tensor.hpp"
#include "flash_attention_v2/kernel/xe_fmha_fwd_kernel.hpp"

namespace cutlass::fmha::kernel {

using namespace cute;

/**
 * Sage FP8 attention built on the unmodified SYCL-TLA FA2 FP8 kernel.
 *
 * K smoothing is performed before quantization:
 *
 *   K_smooth = K - mean(K, sequence)
 *
 * It changes every logit in a query row by the same constant, so softmax does
 * not require a correction. V smoothing is also performed before quantization:
 *
 *   V_smooth = V - mean(V, sequence)
 *
 * Since every normalized attention row sums to one, its exact correction is:
 *
 *   O = softmax(Q K_smooth^T) V_smooth + mean(V, sequence)
 *
 * This wrapper deliberately leaves the FA2 mainloop untouched and fuses the V
 * correction into the same kernel launch after FA2 stores its output tile.
 * VMean uses dense [batch, num_heads_kv, head_size_vo] FP32 layout.
 */
template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_,
          bool SmoothV_>
class XeSageFP8FwdKernel {
 public:
  using BaseKernel =
      XeFMHAFwdKernel<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileScheduler_>;

  using ProblemShape = ProblemShape_;
  using CollectiveMainloop = CollectiveMainloop_;
  using CollectiveEpilogue = CollectiveEpilogue_;
  using TileScheduler = TileScheduler_;
  using BaseArguments = typename BaseKernel::Arguments;
  using BaseParams = typename BaseKernel::Params;
  using ElementQ = typename BaseKernel::ElementQ;
  using ElementK = typename BaseKernel::ElementK;
  using ElementV = typename BaseKernel::ElementV;
  using ElementO = typename BaseKernel::ElementO;
  using StrideQ = typename BaseKernel::StrideQ;
  using StrideK = typename BaseKernel::StrideK;
  using StrideV = typename BaseKernel::StrideV;
  using StrideO = typename BaseKernel::StrideO;
  using SGPerWG = typename BaseKernel::SGPerWG;
  using TileShapeO = typename BaseKernel::TileShapeO;
  using SharedStorage = typename BaseKernel::SharedStorage;

  static constexpr bool SmoothV = SmoothV_;
  static constexpr bool is_var_len = BaseKernel::is_var_len;
  static constexpr int SharedStorageSize = BaseKernel::SharedStorageSize;

  struct Arguments {
    BaseArguments fa2{};
    float const* v_mean = nullptr;
    float const* q_scale = nullptr;
    float const* k_scale = nullptr;
    float const* v_scale = nullptr;
  };

  struct Params {
    BaseParams fa2;
    float const* v_mean;
    float const* q_scale;
    float const* k_scale;
    float const* v_scale;
  };

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    return {BaseKernel::to_underlying_arguments(args.fa2, workspace), args.v_mean,
            args.q_scale, args.k_scale, args.v_scale};
  }

  static bool can_implement(Arguments const& args) {
    if constexpr (SmoothV) {
      if (args.v_mean == nullptr) {
        return false;
      }
    }
    return BaseKernel::can_implement(args.fa2);
  }

  static int get_workspace_size(Arguments const& args) {
    return BaseKernel::get_workspace_size(args.fa2);
  }

  static cutlass::Status initialize_workspace(Arguments const& args, void* workspace = nullptr,
                                              cudaStream_t stream = nullptr,
                                              CudaHostAdapter* cuda_adapter = nullptr) {
    return BaseKernel::initialize_workspace(args.fa2, workspace, stream, cuda_adapter);
  }

  static dim3 get_grid_shape(Params const& params) {
    return BaseKernel::get_grid_shape(params.fa2);
  }

  static dim3 get_block_shape() {
    return BaseKernel::get_block_shape();
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    BaseParams fa2 = params.fa2;
    if (params.q_scale != nullptr) {
      fa2.kernel.scale_q = *params.q_scale;
      fa2.kernel.scale_k = *params.k_scale;
      fa2.kernel.scale_v = *params.v_scale;
    }
    BaseKernel{}(fa2, smem_buf);

    if constexpr (SmoothV) {
      sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_work_group<3>());
      add_v_mean(params);
    }
  }

 private:
  CUTLASS_DEVICE
  void add_v_mean(Params const& params) {
    static_assert(!is_var_len, "Sage FP8 V smoothing currently supports fixed-length tensors only");

    auto const& kernel = params.fa2.kernel;
    auto const& shape = kernel.shape;
    int const head_group_q = shape.num_heads_q / shape.num_heads_kv;
    int const thread_idx = int(ThreadIdxX());
    int const threads = SGPerWG::value * intel::sg_size;
    int const tile_q = int(get<0>(TileShapeO{}));
    int const tile_v = int(get<1>(TileShapeO{}));

    TileScheduler scheduler{params.fa2.scheduler};
    CUTLASS_PRAGMA_NO_UNROLL
    for (; scheduler.is_valid(); ++scheduler) {
      auto [blk_q, blk_v, head_q, batch] = scheduler.get_block_coord();
      int const head_kv = head_q / head_group_q;
      int const q_begin = blk_q * tile_q;
      int const v_begin = blk_v * tile_v;

      auto shape_o = make_shape(shape.seq_len_qo, shape.head_size_vo, shape.num_heads_q, shape.batch);
      auto output = make_tensor(make_gmem_ptr(kernel.O), make_layout(shape_o, kernel.dO));
      float const* v_mean = params.v_mean + (batch * shape.num_heads_kv + head_kv) * shape.head_size_vo;

      for (int linear = thread_idx; linear < tile_q * tile_v; linear += threads) {
        int const q = q_begin + linear / tile_v;
        int const v = v_begin + linear % tile_v;
        if (q < shape.seq_len_qo && v < shape.head_size_vo) {
          output(q, v, head_q, batch) =
              static_cast<ElementO>(static_cast<float>(output(q, v, head_q, batch)) + v_mean[v]);
        }
      }
    }
  }
};

}  // namespace cutlass::fmha::kernel