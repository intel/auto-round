// Flash Attention Prefill Benchmarks using sycl-tla
// Separated from sycl_benchmark.cpp to avoid header conflicts with decode kernels.
//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT

#include <stdio.h>
#include <type_traits>
#include "bestla_ut.h"
#include "bestla_utils.h"
#include "sycl_ut.h"
#include "sycl/sycl_utils.h"

#if defined(ARK_SYCL_TLA)

#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

// FlashAttention-v2 (Xe) building blocks (from sycl-tla).
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "flash_attention_v2/kernel/tile_scheduler.hpp"

// FlashAttention-v2 Prefill kernels only
#include "flash_attention_v2/kernel/xe_flash_attn_prefill.hpp"
#include "flash_attention_v2/collective/xe_flash_attn_prefill_epilogue.hpp"
#include "flash_attention_v2/collective/xe_flash_attn_prefill_softmax_epilogue.hpp"

// CUTLASS-SYCL/CuTe compat layer
#include "cute/util/compat.hpp"

// sycl-tla example utilities
#include "helper.h"
#include "sycl_common.hpp"

#include "cute/tensor.hpp"

// sycl-tla example 06 (SDPA) runner header
#if !defined(HEAD_DIM)
#define HEAD_DIM 128
#endif
#include "xe_fmha_fwd_runner.hpp"

namespace bestla {
using namespace ut;
using namespace utils;
using namespace sycl_utils;
namespace sycl_ut {

int constexpr TestMs = 1000;

template <typename Kernel, typename = void>
struct BestlaBenchSubgroupSize {
  static constexpr int value = Kernel::DispatchPolicy::SubgroupSize;
};

template <typename Kernel>
struct BestlaBenchSubgroupSize<Kernel, std::void_t<decltype(Kernel::SubgroupSize)>> {
  static constexpr int value = Kernel::SubgroupSize;
};

// ========================================================================
// Flash Attention Prefill Benchmarks (FP16)
// ========================================================================

class Benchmark_SyclTla_FlashAttnPrefill {
 public:
  Benchmark_SyclTla_FlashAttnPrefill() {
    UT_START();
    // SDPA/FlashAttention forward (fp16) using sycl-tla example-06 runner.
    // Include a couple larger shapes to better expose bandwidth effects.
    run_fp16_prefill(/*batch=*/1, /*num_heads_q=*/32, /*num_heads_kv=*/32, /*seq_qo=*/1024, /*seq_kv=*/1024,
                     /*head_dim=*/128,
                     /*is_causal=*/true,
                     /*seq_kv_cache=*/0);
    // Non-causal baseline for fair comparison with FP8 non-causal cases.
    run_fp16_prefill(/*batch=*/1, /*num_heads_q=*/32, /*num_heads_kv=*/32, /*seq_qo=*/1024, /*seq_kv=*/1024,
                     /*head_dim=*/128,
                     /*is_causal=*/false,
                     /*seq_kv_cache=*/0);
    run_fp16_prefill(/*batch=*/4, /*num_heads_q=*/32, /*num_heads_kv=*/32, /*seq_qo=*/1024, /*seq_kv=*/1024,
                     /*head_dim=*/128,
                     /*is_causal=*/false,
                     /*seq_kv_cache=*/0);
    run_fp16_prefill(/*batch=*/4, /*num_heads_q=*/32, /*num_heads_kv=*/32, /*seq_qo=*/2048, /*seq_kv=*/2048,
                     /*head_dim=*/128,
                     /*is_causal=*/true,
                     /*seq_kv_cache=*/0);
  }

  static void run_fp16_prefill(int batch, int num_heads_q, int num_heads_kv, int seq_len_qo, int seq_len_kv,
                               int head_dim, bool is_causal, int seq_kv_cache) {
    if (head_dim == 64) {
      if (is_causal) {
        run_fp16_prefill_impl</*HeadDim=*/64, /*Causal=*/true>(batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv,
                                                               seq_kv_cache);
      } else {
        run_fp16_prefill_impl</*HeadDim=*/64, /*Causal=*/false>(batch, num_heads_q, num_heads_kv, seq_len_qo,
                                                                seq_len_kv, seq_kv_cache);
      }
    } else if (head_dim == 128) {
      if (is_causal) {
        run_fp16_prefill_impl</*HeadDim=*/128, /*Causal=*/true>(batch, num_heads_q, num_heads_kv, seq_len_qo,
                                                                seq_len_kv, seq_kv_cache);
      } else {
        run_fp16_prefill_impl</*HeadDim=*/128, /*Causal=*/false>(batch, num_heads_q, num_heads_kv, seq_len_qo,
                                                                 seq_len_kv, seq_kv_cache);
      }
    } else {
      printf("[sycl-tla flash_attn prefill fp16] Skip: unsupported head_dim=%d (supported 64/128)\n", head_dim);
    }
  }

 private:
  template <int HeadDim, bool Causal>
  static void run_fp16_prefill_impl(int batch, int num_heads_q, int num_heads_kv, int seq_len_qo, int seq_len_kv,
                                    int seq_kv_cache) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    compat::set_default_queue(*q);

    Options options;
    options.is_causal = Causal;
    options.varlen = false;
    options.use_paged_kv = false;
    options.scheduler = std::string("Individual");
    options.batch = batch;
    options.num_heads_q = num_heads_q;
    options.num_heads_kv = num_heads_kv;
    options.seq_len_qo = seq_len_qo;
    options.seq_len_kv = seq_len_kv;
    options.seq_len_kv_cache = seq_kv_cache;
    options.page_size = 128;
    options.head_size_qk = HeadDim;
    options.head_size_vo = HeadDim;
    // Keep runtime reasonable for larger shapes.
    if (batch >= 4 || seq_len_qo >= 2048 || seq_len_kv >= 2048) {
      options.warmup = 5;
      options.iterations = 20;
    } else {
      options.warmup = 10;
      options.iterations = 50;
    }
    options.verify = 0;
    options.softmax_scale = 1.0f;

    using namespace cute;
    using ShapeQK = Shape<_128, _64, _32>;
    using ShapePV = Shape<_128, _32, _64>;
    using ShapeOut = Shape<_128, Int<HeadDim>>;
    using SubgroupLayoutQK = std::conditional_t<HeadDim == 64, Layout<Shape<_8, _1, _1>>, Layout<Shape<_16, _1, _1>>>;

    // BMG does not support native BF16; use FP16 instead
    using ElemQ = cutlass::half_t;
    using ElemK = cutlass::half_t;
    using ElemV = cutlass::half_t;

    int rc = FMHAConfig<Causal, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK,
                        /*SubgroupLayoutPV_*/ void,
                        /*PipelineStages*/ 2,
                        /*persistent*/ false, ElemQ, ElemK, ElemV>::run(options);
    (void)rc;

    printf("[sycl-tla flash_attn prefill fp16] B=%d Hq=%d Hkv=%d Sq=%d Skv=%d Skv_cache=%d D=%d causal=%d\n", batch,
           num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, seq_kv_cache, HeadDim, int(Causal));
  }
};

#if 0
static Benchmark_SyclTla_FlashAttnPrefill sBenchmark_SyclTla_FlashAttnPrefill;
#endif
// ========================================================================
// Flash Attention Prefill with Cached KV
// ========================================================================

class Benchmark_SyclTla_FlashAttnPrefillCachedKV {
 public:
  Benchmark_SyclTla_FlashAttnPrefillCachedKV() {
    UT_START();
    // Prefill with KV-cache (simulate attention over cached KV + current KV).
    // Keep non-causal to avoid example constraints; this is intended as a perf microbench.
    Benchmark_SyclTla_FlashAttnPrefill::run_fp16_prefill(
        /*batch=*/1, /*num_heads_q=*/32, /*num_heads_kv=*/32, /*seq_len_qo=*/128, /*seq_len_kv=*/128,
        /*head_dim=*/128, /*is_causal=*/false, /*seq_kv_cache=*/4096);
    // Larger batch to expose cache bandwidth.
    Benchmark_SyclTla_FlashAttnPrefill::run_fp16_prefill(
        /*batch=*/4, /*num_heads_q=*/32, /*num_heads_kv=*/32, /*seq_len_qo=*/128, /*seq_len_kv=*/128,
        /*head_dim=*/128, /*is_causal=*/false, /*seq_kv_cache=*/4096);
  }
};

#if 0
static Benchmark_SyclTla_FlashAttnPrefillCachedKV sBenchmark_SyclTla_FlashAttnPrefillCachedKV;
#endif

// ========================================================================
// Flash Attention Prefill FP8
// ========================================================================

class Benchmark_SyclTla_FlashAttnPrefillFP8 {
 public:
  Benchmark_SyclTla_FlashAttnPrefillFP8() {
    UT_START();
    // Kernel-only microbench for the FP8 FMHA prefill path.
    run_all(1, 32, 32, 1024, 1024, 64, false);
    run_all(1, 32, 32, 1024, 1024, 128, false);
    run_all(1, 32, 32, 2048, 2048, 128, true);
    // Larger batch/seq cases to better expose bandwidth effects.
    run_all(4, 32, 32, 1024, 1024, 128, false);
    run_all(4, 32, 32, 2048, 2048, 128, true);
  }

 private:
  template <typename T>
  static constexpr const char* dtype_str() {
    if constexpr (std::is_same_v<T, cutlass::float_e4m3_t> || std::is_same_v<T, cute::float_e4m3_t>) return "f8_e4m3";
    if constexpr (std::is_same_v<T, cutlass::float_e5m2_t> || std::is_same_v<T, cute::float_e5m2_t>) return "f8_e5m2";
    return "unknown";
  }

  template <typename KernelT>
  static sycl::event launch_prefill(typename KernelT::Params const& kernel_params) {
    compat::dim3 const block = KernelT::get_block_shape();
    compat::dim3 const grid = KernelT::get_grid_shape(kernel_params);
    int smem_size = KernelT::SharedStorageSize;

#if !defined(SYCL_EXT_ONEAPI_WORK_GROUP_SCRATCH_MEMORY)
    using namespace compat::experimental;
    return launch<cutlass::device_kernel<KernelT>>(
        launch_policy{grid, block, local_mem_size{static_cast<std::size_t>(smem_size)},
                      kernel_properties{sycl_exp::sub_group_size<BestlaBenchSubgroupSize<KernelT>::value>}},
        kernel_params);
#else
    compat::experimental::launch_properties launch_props{
        sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
        sycl::ext::oneapi::experimental::sub_group_size<BestlaBenchSubgroupSize<KernelT>::value>,
    };
    compat::experimental::launch_policy policy{grid, block, launch_props, kernel_props};
    return compat::experimental::launch<cutlass::device_kernel<KernelT>, KernelT>(policy, kernel_params);
#endif
  }

  template <typename ElementInput, int HeadDim, bool Causal>
  static void bench_one(int batch, int num_heads_q, int num_heads_kv, int seq_len_qo, int seq_len_kv,
                        float softmax_scale, float timems) {
    using namespace cute;

    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    compat::set_default_queue(*q);

    // Match sycl-tla example-06 fp8 prefill defaults.
    using MMAOperation = XE_8x16x16_F32F16F16F32_TT;
    using GmemTiledCopyQ = XE_2D_U8x8x32_LD_N;
    using GmemTiledCopyK = XE_2D_U8x16x16_LD_T;
    using GmemTiledCopyV = XE_2D_U8x32x32_LD_V;
    using GmemTiledCopyO = XE_2D_U32x8x16_ST_N;

    using ElementAccumulator = float;
    using ElementComputeEpilogue = float;
    // Keep output as float for this FP8 kernel microbench; changing element size
    // affects the sycl-tla epilogue tiled store configuration.
    using ElementOutput = float;
    constexpr int PipelineStages = 2;

    using ShapeQK = Shape<_128, _64, _64>;
    using ShapePV = Shape<_128, _32, _64>;
    using ShapeOutput = std::conditional_t<HeadDim == 64, Shape<_128, _64, _64>, Shape<_128, _128, _64>>;
    using SubgroupLayout = std::conditional_t<HeadDim == 64, Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>,
                                              Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>>;

    using LayoutQ = cutlass::layout::RowMajor;
    using LayoutK = cutlass::layout::ColumnMajor;
    using LayoutV = cutlass::layout::RowMajor;
    using LayoutO = cutlass::layout::RowMajor;

    using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;
    using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;

    using CollectiveEpilogue = cutlass::flash_attention::collective::FlashPrefillEpilogue<
        EpilogueDispatchPolicy, MMAOperation, ShapeOutput, SubgroupLayout, ElementComputeEpilogue, ElementOutput,
        cutlass::gemm::TagToStrideC_t<LayoutO>, ElementOutput, GmemTiledCopyO>;

    using CollectiveSoftmaxEpilogue =
        cutlass::flash_attention::collective::FlashPrefillSoftmaxEpilogue<Causal, EpilogueDispatchPolicy,
                                                                          ElementAccumulator>;

    using ProblemShape = cute::tuple<int, int, int, int, int, int, int>;

    using CollectiveMainloop = cutlass::flash_attention::collective::FlashPrefillMma<
        GEMMDispatchPolicy, ProblemShape, ElementInput, cutlass::gemm::TagToStrideA_t<LayoutQ>, ElementInput,
        cutlass::gemm::TagToStrideB_t<LayoutK>, ElementInput, cutlass::gemm::TagToStrideB_t<LayoutV>, MMAOperation,
        ShapeQK, ShapePV, SubgroupLayout, GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, Causal>;

    using Kernel = cutlass::flash_attention::kernel::FMHAPrefill<ProblemShape, CollectiveMainloop,
                                                                 CollectiveSoftmaxEpilogue, CollectiveEpilogue,
                                                                 cutlass::flash_attention::IndividualScheduler>;

    const size_t q_elems = size_t(seq_len_qo) * size_t(HeadDim) * size_t(batch) * size_t(num_heads_q);
    const size_t kv_elems = size_t(seq_len_kv) * size_t(HeadDim) * size_t(batch) * size_t(num_heads_kv);
    const size_t o_elems = q_elems;

    cutlass::DeviceAllocation<ElementInput> dQ(q_elems);
    cutlass::DeviceAllocation<ElementInput> dK(kv_elems);
    cutlass::DeviceAllocation<ElementInput> dV(kv_elems);
    cutlass::DeviceAllocation<ElementOutput> dO(o_elems);

    q->memset(dQ.get(), 0, q_elems * sizeof(ElementInput)).wait();
    q->memset(dK.get(), 0, kv_elems * sizeof(ElementInput)).wait();
    q->memset(dV.get(), 0, kv_elems * sizeof(ElementInput)).wait();
    q->memset(dO.get(), 0, o_elems * sizeof(ElementOutput)).wait();

    typename Kernel::StrideQ stride_Q = cutlass::make_cute_packed_stride(
        typename Kernel::StrideQ{}, make_shape(seq_len_qo, HeadDim, batch * num_heads_q));
    typename Kernel::StrideK stride_K = cutlass::make_cute_packed_stride(
        typename Kernel::StrideK{}, make_shape(seq_len_kv, HeadDim, batch * num_heads_kv));
    typename Kernel::StrideV stride_V = cutlass::make_cute_packed_stride(
        typename Kernel::StrideV{}, make_shape(HeadDim, seq_len_kv, batch * num_heads_kv));
    typename Kernel::StrideO stride_O = cutlass::make_cute_packed_stride(
        typename Kernel::StrideO{}, make_shape(seq_len_qo, HeadDim, batch * num_heads_q));

    typename Kernel::ProblemShape problem_size =
        make_tuple(batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, HeadDim, HeadDim);
    cutlass::KernelHardwareInfo hw_info;
    typename Kernel::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                         problem_size,
                                         {dQ.get(), stride_Q, dK.get(), stride_K, dV.get(), stride_V},
                                         {softmax_scale},
                                         {dO.get(), stride_O},
                                         hw_info};

    if (!Kernel::can_implement(arguments)) {
      printf("[sycl-tla flash_attn prefill fp8] Skip: can_implement failed\n");
      return;
    }

    size_t workspace_size = Kernel::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    auto status = Kernel::initialize_workspace(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      printf("[sycl-tla flash_attn prefill fp8] Skip: initialize_workspace failed\n");
      return;
    }

    auto params = Kernel::to_underlying_arguments(arguments, workspace.get());
    (void)launch_prefill<Kernel>(params).wait();

    using LOG = timer_statistics_logger<TestMs * 2>;
    LOG log;
    utils::timer<std::chrono::milliseconds> wall;
    wall.start();
    while (wall.stop() < timems) {
      auto ev = launch_prefill<Kernel>(params);
      ev.wait();
      log.add(event_helper::execute_time(ev) * 1000);
      if (wall.stop() >= timems) break;
    }
    log.record();

    // Print sycl-tla runner style metrics for easy comparison.
    // Runner FLOPs convention (counts both QK and PV GEMM-like work):
    //   flops_full = 4 * B * Hq * Sq * Skv * D
    // For causal attention, effective work is reduced by a triangular mask.
    const double time_ms = static_cast<double>(log.avg_val);
    const double time_s = time_ms * 1e-3;
    const double sq = static_cast<double>(seq_len_qo);
    const double skv = static_cast<double>(seq_len_kv);

    double causal_ratio = 1.0;
    if constexpr (Causal) {
      // Total attended keys = sum_{i=1..Sq} min(i, Skv)
      const double denom = sq * skv;
      const double attended = (sq <= skv) ? (sq * (sq + 1.0) * 0.5) : (skv * (skv + 1.0) * 0.5 + (sq - skv) * skv);
      causal_ratio = (denom > 0.0) ? (attended / denom) : 1.0;
    }

    const double flops = 4.0 * static_cast<double>(batch) * static_cast<double>(num_heads_q) * sq * skv *
                         static_cast<double>(HeadDim) * causal_ratio;

    // Bytes: count Q and O fully; count K/V proportionally to attended keys.
    const double kv_read_elems_equiv =
        static_cast<double>(batch) * static_cast<double>(num_heads_q) * skv * static_cast<double>(HeadDim);
    const double bytes = static_cast<double>(q_elems) * sizeof(ElementInput) +
                         2.0 * kv_read_elems_equiv * sizeof(ElementInput) * causal_ratio +
                         static_cast<double>(o_elems) * sizeof(ElementOutput);
    const double tflops = (time_s > 0.0) ? (flops / time_s / 1.0e12) : 0.0;
    const double gbs = (time_s > 0.0) ? (bytes / time_s / 1.0e9) : 0.0;
    printf("Performance:   %.3f  GB/s,    %.3f  TFlop/s,   %.4f  ms (dtype=%s)\n", gbs, tflops, time_ms,
           dtype_str<ElementInput>());

    printf("[sycl-tla flash_attn prefill fp8] B=%d Hq=%d Hkv=%d Sq=%d Skv=%d D=%d causal=%d  %s\n", batch, num_heads_q,
           num_heads_kv, seq_len_qo, seq_len_kv, HeadDim, int(Causal), log.get_log_str());
  }

  static void run_all(int batch, int num_heads_q, int num_heads_kv, int seq_len_qo, int seq_len_kv, int head_dim,
                      bool is_causal) {
    const float softmax_scale = 1.0f;
    printf("\n[sycl-tla flash_attn prefill fp8] case: B=%d Hq=%d Hkv=%d Sq=%d Skv=%d D=%d causal=%d\n", batch,
           num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, head_dim, int(is_causal));

    // Note: Only E5M2 is supported. E4M3 is not optimized in the original sycl-tla FP8 prefill kernel
    // and causes severe register spill, resulting in ~10x slower performance.
    if (head_dim == 64) {
      if (is_causal) {
        bench_one<cute::float_e5m2_t, 64, true>(batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, softmax_scale,
                                                float(TestMs));
      } else {
        bench_one<cute::float_e5m2_t, 64, false>(batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv,
                                                 softmax_scale, float(TestMs));
      }
    } else if (head_dim == 128) {
      if (is_causal) {
        bench_one<cute::float_e5m2_t, 128, true>(batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv,
                                                 softmax_scale, float(TestMs));
      } else {
        bench_one<cute::float_e5m2_t, 128, false>(batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv,
                                                  softmax_scale, float(TestMs));
      }
    } else {
      printf("[sycl-tla flash_attn prefill fp8] Skip: unsupported head_dim=%d (supported 64/128)\n", head_dim);
    }
  }
};

#if 0
static Benchmark_SyclTla_FlashAttnPrefillFP8 sBenchmark_SyclTla_FlashAttnPrefillFP8;
#endif
#endif  // ARK_SYCL_TLA
}  // namespace sycl_ut
}  // namespace bestla
