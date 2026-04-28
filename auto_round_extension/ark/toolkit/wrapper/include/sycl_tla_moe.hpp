// SYCL-TLA MOE GEMM Wrapper
// Based on sycl-tla/examples/12_xe20_moe_gemm_cute_interface
//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#ifdef ARK_XPU
#include <sycl/sycl.hpp>
#endif

#if defined(ARK_SYCL_TLA)

#include <type_traits>

#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/sycl_event_manager.hpp"

// CUTLASS-SYCL/CuTe compat layer
#include "cute/util/compat.hpp"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "cute/tensor.hpp"

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"

// MOE kernel headers from sycl-tla example 12
#include "moe_grouped_gemm.hpp"
#include "moe_tile_scheduler.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#endif  // ARK_SYCL_TLA

#include "sycl_tla_common.hpp"

namespace ark {

#if defined(ARK_SYCL_TLA)

namespace moe_detail {

using namespace cute;
using namespace MoE;

// Helper to choose TiledMMA based on element types
template <class TA, class TB>
auto choose_tiled_mma(TA* A, TB* B) {
  using TA_non_CV = cutlass::platform::remove_cv_t<TA>;
  using TB_non_CV = cutlass::platform::remove_cv_t<TB>;
  auto op = XE_DPAS_TT<8, float, TA_non_CV, TB_non_CV>{};

  using WGTile = Shape<_256, _128, _32>;                           // 256x128 WG tile size
  using SGLayout = Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>;  // 8x2 SG tiling, n-major

  using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>, SGLayout>::TiledMMA;

  return MMA{};
}

// Unique kernel name tag
template <typename EA, typename EB, typename ED, char layoutA, char layoutB>
class MoEGemmKernel;

// MOE GEMM launcher template
template <char layoutA, char layoutB, typename ElementA, typename ElementB, typename ElementS, typename ElementD>
void moe_gemm_launcher(sycl::queue* q, const ElementA* activations, const ElementB* weights, const ElementS* scales,
                       ElementD* outputs, const int gemm_n, const int gemm_k, int* num_rows_per_expert_device,
                       const int num_experts) {
  compat::set_default_queue(*q);

  int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  cutlass::KernelHardwareInfo hw_info{0, sm_count};

  auto dummy_problem_shape = cute::Shape<int, int, int>{1, gemm_k, gemm_n};
  auto dummy_group_problem_shape =
      cutlass::gemm::GroupProblemShape<Shape<int, int, int>>{1, &dummy_problem_shape, nullptr};

  using TileShape = Shape<_256, _128, _32>;
  using ClusterShape = Shape<_1, _1, _1>;

  auto scheduler_params = PersistentTileSchedulerXeMoE<ProblemShape>::to_underlying_arguments(
      dummy_group_problem_shape, TileShape{}, ClusterShape{}, hw_info,
      PersistentTileSchedulerXeMoE<ProblemShape>::Arguments{1, RasterOrderOptions::AlongN});

  auto group_distribution = PersistentTileSchedulerXeMoE<ProblemShape>::get_grid_shape(
      scheduler_params, dummy_group_problem_shape, TileShape{}, ClusterShape{}, hw_info,
      PersistentTileSchedulerXeMoE<ProblemShape>::Arguments{1, RasterOrderOptions::AlongN});

  auto mma = choose_tiled_mma(activations, weights);
  auto MaxThreadsPerWorkgroup = size(mma);
  dim3 local_range{static_cast<unsigned int>(MaxThreadsPerWorkgroup), 1, 1};

  sycl::range<3> local = {local_range.z, local_range.y, local_range.x};
  sycl::range<3> groups = {group_distribution.z, group_distribution.y, group_distribution.x};
  sycl::range<3> global = {local[0] * groups[0], local[1] * groups[1], local[2] * groups[2]};

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props{syclex::sub_group_size<16>, intelex::grf_size<256>};

  auto event = q->parallel_for<MoEGemmKernel<ElementA, ElementB, ElementD, layoutA, layoutB>>(
      sycl::nd_range<3>(global, local), kernel_props, [=](auto) {
        MoE::MoEGEMM<XE_LOAD_2D<16, 32, 32, 16>, XE_LOAD_2D_VNNI<16, 32, 16, 16>, XE_STORE_2D<16, 8, 32>, 'R', 'R',
                     'R'>(activations, weights, scales, outputs, mma, num_rows_per_expert_device, num_experts, gemm_n,
                          gemm_k, scheduler_params);
      });

  EventManager::getInstance().addEvent(event);
  event.wait();
}

}  // namespace moe_detail

// Public MOE GEMM API
inline void moe_gemm(sycl::queue* q, void* activations, void* weights, void* scales, void* outputs, BTLA_DTYPE dtype,
                     int N, int K, int* num_tokens_per_expert, int num_experts) {
  switch (dtype) {
    case BTLA_DTYPE::BF16: {
      using Element = cutlass::bfloat16_t;
      moe_detail::moe_gemm_launcher<'R', 'R', Element, Element, Element, Element>(
          q, static_cast<const Element*>(activations), static_cast<const Element*>(weights),
          static_cast<const Element*>(scales), static_cast<Element*>(outputs), N, K, num_tokens_per_expert,
          num_experts);
      break;
    }
    case BTLA_DTYPE::F16: {
      using Element = cutlass::half_t;
      moe_detail::moe_gemm_launcher<'R', 'R', Element, Element, Element, Element>(
          q, static_cast<const Element*>(activations), static_cast<const Element*>(weights),
          static_cast<const Element*>(scales), static_cast<Element*>(outputs), N, K, num_tokens_per_expert,
          num_experts);
      break;
    }
    default:
      throw std::runtime_error("moe_gemm: unsupported dtype, only BF16/FP16 supported");
  }
}

#endif  // ARK_SYCL_TLA

}  // namespace ark
