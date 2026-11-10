// MOE GEMM Benchmarks using sycl-tla
// Based on sycl-tla/examples/12_xe20_moe_gemm_cute_interface
//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT

#include <stdio.h>
#include <random>
#include <type_traits>
#include "bestla_ut.h"
#include "bestla_utils.h"
#include "sycl_ut.h"
#include "sycl/sycl_utils.h"

#if defined(ARK_SYCL_TLA)

#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/initialize_block.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"

#include "cute/util/compat.hpp"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "cute/tensor.hpp"

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"

// MOE kernel headers from sycl-tla
#include "moe_grouped_gemm.hpp"
#include "moe_tile_scheduler.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace bestla {
using namespace ut;
using namespace utils;
using namespace sycl_utils;
namespace sycl_ut {

using namespace cute;
using namespace MoE;

int constexpr TestMs = 1000;

using ElementAccumulator = float;

// ========================================================================
// Helper to choose TiledMMA based on element types
// ========================================================================
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

// Unique kernel name tags
template <typename, typename, typename, char, char>
class MoEGemmCuteName;

template <typename, typename, typename, char, char>
class MoEGemmCuteNameWarmup;

// ========================================================================
// MOE GEMM Launcher
// ========================================================================
template <char layoutA, char layoutB, typename ElementA, typename ElementB, typename ElementS, typename ElementD>
void MoEGEMMLauncher(const ElementA* activations, const ElementB* weights, const ElementS* scales, ElementD* outputs,
                     const int gemm_n, const int gemm_k, const int* num_rows_per_expert_device,
                     const int* num_tokens_per_expert_host, const int num_experts, int iterations,
                     double& avg_time_ms) {
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
  sycl::queue Q = compat::get_default_queue();

  // Warmup
  for (int i = 0; i < 5; i++) {
    Q.parallel_for<MoEGemmCuteNameWarmup<ElementA, ElementB, ElementD, layoutA,
                                         layoutB>>(sycl::nd_range<3>(global, local), kernel_props, [=](auto) {
       MoE::MoEGEMM<XE_LOAD_2D<16, 32, 32, 16>, XE_LOAD_2D_VNNI<16, 32, 16, 16>, XE_STORE_2D<16, 8, 32>, 'R', 'R', 'R'>(
           activations, weights, scales, outputs, mma, num_rows_per_expert_device, num_experts, gemm_n, gemm_k,
           scheduler_params);
     }).wait();
  }

  // Timing
  GPU_Clock timer;
  timer.start();
  for (int i = 0; i < iterations; ++i) {
    Q.parallel_for<MoEGemmCuteName<ElementA, ElementB, ElementD, layoutA,
                                   layoutB>>(sycl::nd_range<3>(global, local), kernel_props, [=](auto) {
       MoE::MoEGEMM<XE_LOAD_2D<16, 32, 32, 16>, XE_LOAD_2D_VNNI<16, 32, 16, 16>, XE_STORE_2D<16, 8, 32>, 'R', 'R', 'R'>(
           activations, weights, scales, outputs, mma, num_rows_per_expert_device, num_experts, gemm_n, gemm_k,
           scheduler_params);
     }).wait();
  }
  avg_time_ms = timer.seconds() * 1000.0 / iterations;
}

// ========================================================================
// Verification Helper
// ========================================================================
struct VerificationHelper {
  int m = 0, n = 0, k = 0, groups;
  std::vector<typename MoE::ProblemShape::UnderlyingProblemShape> problem_sizes_host;

  void parse(const int num_experts, const int* num_tokens_per_expert_host, int moe_n, int moe_k) {
    n = moe_n;
    k = moe_k;
    groups = num_experts;
    problem_sizes_host.clear();
    problem_sizes_host.reserve(groups);
    for (int i = 0; i < groups; i++) {
      problem_sizes_host.push_back({num_tokens_per_expert_host[i], n, k});
      m += num_tokens_per_expert_host[i];
    }
  }

  std::tuple<double, double> gflops(double runtime_s) const {
    uint64_t fmas = 0;
    uint64_t bytes_loaded = 0;

    for (auto const& problem : problem_sizes_host) {
      auto M = static_cast<uint64_t>(get<0>(problem));
      auto N = static_cast<uint64_t>(get<1>(problem));
      auto K = static_cast<uint64_t>(get<2>(problem));
      fmas += M * N * K;
      bytes_loaded += 2 * (2 * M * N + N * K + M * K);  // sizeof(bfloat16_t) = 2
    }

    uint64_t flop = uint64_t(2) * fmas;
    double gflop = double(flop) / double(1.0e9);
    double gbytes = double(bytes_loaded) / double(1.0e9);

    return std::make_tuple(gflop / runtime_s, gbytes / runtime_s);
  }

  template <class ElementA, class ElementB, class ElementD>
  bool verify(const ElementA* activations, const ElementB* weights, ElementD* outputs) {
    cutlass::DeviceAllocation<ElementD> output_ref;
    cutlass::DeviceAllocation<ElementD> unused_c_matrix;
    output_ref.reset(m * n);
    unused_c_matrix.reset(m * n);

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    bool passed = true;
    int cumulative_sum = 0;
    for (int32_t i = 0; i < groups; ++i) {
      auto problem = problem_sizes_host.at(i);
      auto M = get<0>(problem);
      cutlass::TensorRef ref_A(const_cast<ElementA*>(activations) + cumulative_sum * k, LayoutA::packed({M, k}));
      cutlass::TensorRef ref_B(const_cast<ElementB*>(weights) + i * n * k, LayoutB::packed({k, n}));
      cutlass::TensorRef ref_C(unused_c_matrix.get() + cumulative_sum * n, LayoutC::packed({M, n}));
      cutlass::TensorRef ref_D(output_ref.get() + cumulative_sum * n, LayoutD::packed({M, n}));

      cutlass::reference::device::GemmComplex({M, n, k}, 1.0, ref_A, cutlass::ComplexTransform::kNone, ref_B,
                                              cutlass::ComplexTransform::kNone, 0.0, ref_C, ref_D,
                                              ElementAccumulator(0), 1, M * k, k * n, M * n, M * n);
      compat::wait();

      passed &= cutlass::reference::device::BlockCompareEqual(output_ref.get() + cumulative_sum * n,
                                                              outputs + cumulative_sum * n, M * n);
      if (!passed) {
        break;
      }
      cumulative_sum += M;
    }
    return passed;
  }
};

// ========================================================================
// MOE GEMM Benchmark Class
// ========================================================================
class Benchmark_SyclTla_MoEGemm {
 public:
  Benchmark_SyclTla_MoEGemm() {
    UT_START();

    // Test cases similar to sycl-tla example
    // Typical MOE configurations: num_experts=32, N=5760/2880, K=2880
    printf("\n[sycl-tla MOE GEMM] Starting benchmarks...\n");

    // Case 1: Small experts, typical shape
    {
      std::vector<int> tokens_per_expert = {256, 256, 256, 256, 256, 256, 256, 256};
      run_moe_gemm(tokens_per_expert, 2880, 2880);
    }

    // Case 2: Medium experts
    {
      std::vector<int> tokens_per_expert = {512, 384, 640, 256, 448, 320, 576, 416};
      run_moe_gemm(tokens_per_expert, 2880, 2880);
    }

    // Case 3: Larger scale (32 experts)
    {
      std::vector<int> tokens_per_expert = {148, 231, 404, 180, 127, 244, 224, 244, 110, 617, 289,
                                            845, 191, 424, 30,  97,  57,  324, 62,  77,  75,  144,
                                            250, 287, 629, 370, 161, 101, 215, 113, 224, 35};
      run_moe_gemm(tokens_per_expert, 5760, 2880);
    }

    // Case 4: Same experts with different N/K
    {
      std::vector<int> tokens_per_expert = {148, 231, 404, 180, 127, 244, 224, 244, 110, 617, 289,
                                            845, 191, 424, 30,  97,  57,  324, 62,  77,  75,  144,
                                            250, 287, 629, 370, 161, 101, 215, 113, 224, 35};
      run_moe_gemm(tokens_per_expert, 2880, 2880);
    }
  }

 private:
  void run_moe_gemm(const std::vector<int>& tokens_per_expert, int N, int K, bool verify = false) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    compat::set_default_queue(*q);

    int num_experts = static_cast<int>(tokens_per_expert.size());
    int num_tokens_total = 0;
    for (int m : tokens_per_expert) {
      num_tokens_total += m;
    }

    // Calculate M-occupancy
    float actual_num_units = 0.0f;
    int total_num_M_tiles = 0;
    for (int m : tokens_per_expert) {
      total_num_M_tiles += (m + 255) / 256;
      actual_num_units += m / 256.0f;
    }
    float M_occupancy = actual_num_units / total_num_M_tiles;

    printf("\n[sycl-tla MOE GEMM] experts=%d, total_tokens=%d, N=%d, K=%d, M_occupancy=%.2f\n", num_experts,
           num_tokens_total, N, K, M_occupancy);

    // Allocate memory
    using ElementA = bfloat16_t;
    using ElementB = bfloat16_t;
    using ElementD = bfloat16_t;

    size_t A_size = static_cast<size_t>(num_tokens_total) * K;
    size_t B_size = static_cast<size_t>(num_experts) * N * K;
    size_t D_size = static_cast<size_t>(num_tokens_total) * N;

    cutlass::DeviceAllocation<int32_t> num_rows_per_expert_device;
    cutlass::DeviceAllocation<ElementA> activations_data;
    cutlass::DeviceAllocation<ElementB> weights_data;
    cutlass::DeviceAllocation<ElementD> output_data;

    num_rows_per_expert_device.reset(num_experts);
    num_rows_per_expert_device.copy_from_host(tokens_per_expert.data());
    activations_data.reset(A_size);
    weights_data.reset(B_size);
    output_data.reset(D_size);

    uint64_t seed = 2023;
    initialize_block(activations_data, seed + 2023);
    initialize_block(weights_data, seed + 2022);
    initialize_block(output_data, seed + 2021);

    int iterations = 50;
    double avg_time_ms = 0.0;

    MoEGEMMLauncher<'R', 'R'>(activations_data.get(), weights_data.get(), static_cast<void*>(nullptr),
                              output_data.get(), N, K, num_rows_per_expert_device.get(), tokens_per_expert.data(),
                              num_experts, iterations, avg_time_ms);

    // Verify if requested
    if (verify) {
      VerificationHelper helper;
      helper.parse(num_experts, tokens_per_expert.data(), N, K);
      bool passed = helper.verify(activations_data.get(), weights_data.get(), output_data.get());
      printf("  Verification: %s\n", passed ? "PASSED" : "FAILED");
    }

    // Calculate performance
    VerificationHelper perf_helper;
    perf_helper.parse(num_experts, tokens_per_expert.data(), N, K);
    auto [gflops, gbps] = perf_helper.gflops(avg_time_ms / 1000.0);

    printf("  Performance: %.3f GFlop/s, %.3f GB/s, %.4f ms\n", gflops, gbps, avg_time_ms);
  }
};

#if 0
static Benchmark_SyclTla_MoEGemm sBenchmark_SyclTla_MoEGemm;
#endif
#endif  // ARK_SYCL_TLA
}  // namespace sycl_ut
}  // namespace bestla
