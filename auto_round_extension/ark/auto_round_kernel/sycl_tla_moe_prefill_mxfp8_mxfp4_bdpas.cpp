#include "sycl_tla_moe_prefill_mxfp8_mxfp4_bdpas.hpp"

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA) && defined(ARK_MXFP_BDPAS_CUTLASS)

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "sycl_tla_moe_dequant.hpp"
#include "utils.hpp"

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/collective/xe_array_mma_blockscaled_native.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/xe_gemm_array_cooperative.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/util/packed_stride.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace ark {
namespace moe_mxfp_bdpas_detail {

using namespace cute;

inline constexpr size_t kMetadataScratchLoc = 11;
inline constexpr size_t kWorkspaceScratchLoc = 12;

class UnpackMXFP4WeightToFP8Kernel;
class TransposeActivationScaleKernel;
class TransposeWeightScaleKernel;

bool env_enabled(const char* env_name) {
  const char* env = std::getenv(env_name);
  if (env == nullptr) return true;
  std::string value(env);
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return !(value == "0" || value == "false" || value == "off" || value == "no");
}

template <class T>
T* align_ptr(uint8_t*& cursor) {
  const uintptr_t raw = reinterpret_cast<uintptr_t>(cursor);
  const uintptr_t aligned = (raw + alignof(T) - 1) & ~(uintptr_t(alignof(T)) - 1);
  cursor = reinterpret_cast<uint8_t*>(aligned);
  return reinterpret_cast<T*>(cursor);
}

template <class T>
void reserve_bytes(size_t count, size_t& bytes) {
  bytes = (bytes + alignof(T) - 1) & ~(size_t(alignof(T)) - 1);
  bytes += sizeof(T) * count;
}

template <class T>
T* carve(uint8_t*& cursor, size_t count) {
  T* ptr = align_ptr<T>(cursor);
  cursor += sizeof(T) * count;
  return ptr;
}

void launch_unpack_mxfp4_to_fp8(sycl::queue* queue, const uint8_t* weights_nkp, uint8_t* weights_nk,
                                int num_experts, int N, int K) {
  const int k_packed = K / 2;
  sycl::range<3> global{static_cast<size_t>(num_experts), static_cast<size_t>(N), static_cast<size_t>(k_packed)};
  queue->parallel_for<UnpackMXFP4WeightToFP8Kernel>(global, [=](sycl::id<3> index) {
    const int expert_index = static_cast<int>(index[0]);
    const int n_coord = static_cast<int>(index[1]);
    const int k_pair = static_cast<int>(index[2]);
    const uint8_t packed = weights_nkp[(static_cast<size_t>(expert_index) * N + n_coord) * k_packed + k_pair];
    const size_t out_base = static_cast<size_t>(expert_index) * N * K + static_cast<size_t>(k_pair) * 2 * N + n_coord;
    weights_nk[out_base] = moe_dequant::encode_fp4_e2m1_as_fp8_e4m3(packed & 0x0Fu);
    weights_nk[out_base + N] = moe_dequant::encode_fp4_e2m1_as_fp8_e4m3((packed >> 4) & 0x0Fu);
  });
}

void launch_transpose_activation_scales(sycl::queue* queue, const uint8_t* input_scales, uint8_t* output_scales,
                                        const int32_t* row_offsets, const int32_t* token_counts,
                                        const int32_t* scale_offsets, const int32_t* padded_token_counts,
                                        int active_count, int scale_groups, int max_padded_tokens) {
  sycl::range<3> global{static_cast<size_t>(active_count), static_cast<size_t>(scale_groups),
                        static_cast<size_t>(max_padded_tokens)};
  queue->parallel_for<TransposeActivationScaleKernel>(global, [=](sycl::id<3> index) {
    const int active_index = static_cast<int>(index[0]);
    const int scale_group = static_cast<int>(index[1]);
    const int token_index = static_cast<int>(index[2]);
    const int token_count = token_counts[active_index];
    const int padded_tokens = padded_token_counts[active_index];
    if (token_index >= padded_tokens) return;
    const int out_offset = scale_offsets[active_index] + scale_group * padded_tokens + token_index;
    if (token_index < token_count) {
      const int row_offset = row_offsets[active_index] + token_index;
      output_scales[out_offset] = input_scales[static_cast<size_t>(row_offset) * scale_groups + scale_group];
    } else {
      output_scales[out_offset] = 127;
    }
  });
}

void launch_transpose_weight_scales(sycl::queue* queue, const uint8_t* input_scales, uint8_t* output_scales,
                                    int num_experts, int N, int scale_groups) {
  sycl::range<3> global{static_cast<size_t>(num_experts), static_cast<size_t>(scale_groups), static_cast<size_t>(N)};
  queue->parallel_for<TransposeWeightScaleKernel>(global, [=](sycl::id<3> index) {
    const int expert_index = static_cast<int>(index[0]);
    const int scale_group = static_cast<int>(index[1]);
    const int n_coord = static_cast<int>(index[2]);
    const size_t in_offset = (static_cast<size_t>(expert_index) * N + n_coord) * scale_groups + scale_group;
    const size_t out_offset = (static_cast<size_t>(expert_index) * scale_groups + scale_group) * N + n_coord;
    output_scales[out_offset] = input_scales[in_offset];
  });
}

template <typename ElementInputA, bool NativeMxfp4 = false>
bool run_grouped_bdpas(sycl::queue* queue, void* activations, void* activation_scales, void* weights,
                       void* weight_scales, void* outputs, void* activation_workspace, void* weight_workspace,
                       int N, int K, int group_size, int* num_tokens_per_expert, int num_experts, int total_tokens,
                       bool refresh_weight_staging, int* num_tokens_per_expert_host, bool refresh_metadata) {
  using ElementScale = cutlass::float_ue8m0_t;
  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = std::conditional_t<NativeMxfp4, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using ElementInputB = ElementInputA;
  using ElementOutput = cutlass::bfloat16_t;
  using TileShape = std::conditional_t<NativeMxfp4, Shape<_256, _512, _128>, Shape<Int<64>, _512, _64>>;
  using ThreadLayout = std::conditional_t<NativeMxfp4, Layout<Shape<_4, _8, _1>, Stride<_8, _1, _0>>,
                                         Layout<Shape<_1, Int<32>, _1>, Stride<Int<32>, _1, _0>>>;
  using TiledMma = typename TiledMMAHelper<MMA_Atom<XE_BDPAS_TT<8, float, ElementInputA>>,
                                          Layout<TileShape>, ThreadLayout>::TiledMMA;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGenericGroup;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementOutput, ElementComputeEpilogue,
                                                                  ElementAccumulator, ElementAccumulator,
                                                                  cutlass::FloatRoundStyle::round_to_nearest>;
  using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape,
                                                                     decltype(tile_shape(TiledMma()))>;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy, TileShape, void, ElementAccumulator, cutlass::gemm::TagToStrideC_t<LayoutC*>, ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutD*>, FusionCallbacks, void, void>;
  using StrideScale = Stride<_1, int64_t, int64_t>;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16BlockScaledGroup<2, Int<32>>;
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy, TileShape, tuple<ElementInputA, ElementScale>,
      tuple<cutlass::gemm::TagToStrideA_t<LayoutA*>, StrideScale*>, tuple<ElementInputB, ElementScale>,
      tuple<cutlass::gemm::TagToStrideB_t<LayoutB*>, StrideScale*>, TiledMma, tuple<void, void>, void, void,
      identity, tuple<void, void>, void, void, identity>;
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue,
                                                          cutlass::gemm::GroupScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;
  using StrideScaleA = typename CollectiveMainloop::InternalStrideScaleA;
  using StrideScaleB = typename CollectiveMainloop::InternalStrideScaleB;

  if (group_size != 32 || K % 64 != 0 || N % 16 != 0 || (K & 1) != 0) return false;
  if (activation_workspace == nullptr || weight_workspace == nullptr) return false;

  std::vector<int32_t> token_counts_all(static_cast<size_t>(num_experts));
  if (num_tokens_per_expert_host != nullptr) {
    std::copy(num_tokens_per_expert_host, num_tokens_per_expert_host + num_experts, token_counts_all.begin());
  } else {
    queue->memcpy(token_counts_all.data(), num_tokens_per_expert, sizeof(int32_t) * num_experts).wait();
  }

  std::vector<int32_t> active_experts_host;
  std::vector<int32_t> row_offsets_host;
  std::vector<int32_t> token_counts_host;
  std::vector<int32_t> padded_token_counts_host;
  std::vector<int32_t> scale_a_offsets_host;
  std::vector<int32_t> scale_b_offsets_host;
  std::vector<UnderlyingProblemShape> problem_shapes_host;

  active_experts_host.reserve(static_cast<size_t>(num_experts));
  int32_t running_row_offset = 0;
  int32_t running_scale_a_offset = 0;
  int32_t max_padded_tokens = 0;
  const int scale_groups = K / group_size;
  for (int expert_index = 0; expert_index < num_experts; ++expert_index) {
    const int32_t token_count = token_counts_all[static_cast<size_t>(expert_index)];
    if (token_count > 0) {
      const int32_t padded_tokens = ((token_count + 3) / 4) * 4;
      active_experts_host.push_back(expert_index);
      row_offsets_host.push_back(running_row_offset);
      token_counts_host.push_back(token_count);
      padded_token_counts_host.push_back(padded_tokens);
      scale_a_offsets_host.push_back(running_scale_a_offset);
      problem_shapes_host.push_back(make_shape(static_cast<int>(token_count), N, K));
      running_scale_a_offset += padded_tokens * scale_groups;
      max_padded_tokens = std::max<int32_t>(max_padded_tokens, padded_tokens);
    }
    running_row_offset += token_count;
  }

  const int active_count = static_cast<int>(active_experts_host.size());
  if (active_count == 0) return true;

  auto* activation_workspace_bytes = static_cast<uint8_t*>(activation_workspace);
  auto* weight_workspace_bytes = static_cast<uint8_t*>(weight_workspace);
  auto* weights_fp8 = weight_workspace_bytes;
  auto* scales_bdpas_a = activation_workspace_bytes;
  auto* scales_bdpas_b = NativeMxfp4 ? weight_workspace_bytes
                                     : weights_fp8 + static_cast<size_t>(num_experts) * N * K;
  if (static_cast<size_t>(num_experts) * N * scale_groups > static_cast<size_t>(num_experts) * N * K) return false;
  size_t metadata_bytes = 0;
  reserve_bytes<UnderlyingProblemShape>(active_count, metadata_bytes);
  reserve_bytes<const ElementInputA*>(active_count, metadata_bytes);
  reserve_bytes<const ElementInputB*>(active_count, metadata_bytes);
  reserve_bytes<const ElementScale*>(active_count, metadata_bytes);
  reserve_bytes<const ElementScale*>(active_count, metadata_bytes);
  reserve_bytes<const ElementAccumulator*>(active_count, metadata_bytes);
  reserve_bytes<ElementOutput*>(active_count, metadata_bytes);
  reserve_bytes<StrideA>(active_count, metadata_bytes);
  reserve_bytes<StrideB>(active_count, metadata_bytes);
  reserve_bytes<StrideC>(active_count, metadata_bytes);
  reserve_bytes<StrideD>(active_count, metadata_bytes);
  reserve_bytes<StrideScaleA>(active_count, metadata_bytes);
  reserve_bytes<StrideScaleB>(active_count, metadata_bytes);
  reserve_bytes<int32_t>(active_count, metadata_bytes);
  reserve_bytes<int32_t>(active_count, metadata_bytes);
  reserve_bytes<int32_t>(active_count, metadata_bytes);
  reserve_bytes<int32_t>(active_count, metadata_bytes);
  reserve_bytes<int32_t>(active_count, metadata_bytes);
  reserve_bytes<int32_t>(active_count, metadata_bytes);

  auto* metadata = static_cast<uint8_t*>(DeviceMemoryPool::Instance()->get_scratch_mem(metadata_bytes,
                                                                                       kMetadataScratchLoc, queue));
  if (metadata == nullptr) throw std::runtime_error("mxfp8_mxfp4_bdpas: failed to allocate metadata scratch");
  uint8_t* cursor = metadata;
  auto* problem_shapes_device = carve<UnderlyingProblemShape>(cursor, active_count);
  auto* ptr_a_device = carve<const ElementInputA*>(cursor, active_count);
  auto* ptr_b_device = carve<const ElementInputB*>(cursor, active_count);
  auto* ptr_scale_a_device = carve<const ElementScale*>(cursor, active_count);
  auto* ptr_scale_b_device = carve<const ElementScale*>(cursor, active_count);
  auto* ptr_c_device = carve<const ElementAccumulator*>(cursor, active_count);
  auto* ptr_d_device = carve<ElementOutput*>(cursor, active_count);
  auto* stride_a_device = carve<StrideA>(cursor, active_count);
  auto* stride_b_device = carve<StrideB>(cursor, active_count);
  auto* stride_c_device = carve<StrideC>(cursor, active_count);
  auto* stride_d_device = carve<StrideD>(cursor, active_count);
  auto* stride_scale_a_device = carve<StrideScaleA>(cursor, active_count);
  auto* stride_scale_b_device = carve<StrideScaleB>(cursor, active_count);
  auto* active_experts_device = carve<int32_t>(cursor, active_count);
  auto* row_offsets_device = carve<int32_t>(cursor, active_count);
  auto* token_counts_device = carve<int32_t>(cursor, active_count);
  auto* padded_token_counts_device = carve<int32_t>(cursor, active_count);
  auto* scale_a_offsets_device = carve<int32_t>(cursor, active_count);

  std::vector<const ElementInputA*> ptr_a_host(static_cast<size_t>(active_count));
  std::vector<const ElementInputB*> ptr_b_host(static_cast<size_t>(active_count));
  std::vector<const ElementScale*> ptr_scale_a_host(static_cast<size_t>(active_count));
  std::vector<const ElementScale*> ptr_scale_b_host(static_cast<size_t>(active_count));
  std::vector<const ElementAccumulator*> ptr_c_host(static_cast<size_t>(active_count), nullptr);
  std::vector<ElementOutput*> ptr_d_host(static_cast<size_t>(active_count));
  std::vector<StrideA> stride_a_host;
  std::vector<StrideB> stride_b_host;
  std::vector<StrideC> stride_c_host;
  std::vector<StrideD> stride_d_host;
  std::vector<StrideScaleA> stride_scale_a_host;
  std::vector<StrideScaleB> stride_scale_b_host;
  stride_a_host.reserve(static_cast<size_t>(active_count));
  stride_b_host.reserve(static_cast<size_t>(active_count));
  stride_c_host.reserve(static_cast<size_t>(active_count));
  stride_d_host.reserve(static_cast<size_t>(active_count));
  stride_scale_a_host.reserve(static_cast<size_t>(active_count));
  stride_scale_b_host.reserve(static_cast<size_t>(active_count));

  for (int active_index = 0; active_index < active_count; ++active_index) {
    const int expert_index = active_experts_host[static_cast<size_t>(active_index)];
    const int token_count = token_counts_host[static_cast<size_t>(active_index)];
    const int row_offset = row_offsets_host[static_cast<size_t>(active_index)];
    const int padded_tokens = padded_token_counts_host[static_cast<size_t>(active_index)];
    if constexpr (NativeMxfp4) {
      ptr_a_host[static_cast<size_t>(active_index)] = reinterpret_cast<const ElementInputA*>(
          static_cast<const uint8_t*>(activations) + static_cast<size_t>(row_offset) * (K / 2));
      ptr_b_host[static_cast<size_t>(active_index)] = reinterpret_cast<const ElementInputB*>(
          static_cast<const uint8_t*>(weights) + static_cast<size_t>(expert_index) * N * (K / 2));
    } else {
      ptr_a_host[static_cast<size_t>(active_index)] = reinterpret_cast<const ElementInputA*>(
          static_cast<const uint8_t*>(activations) + static_cast<size_t>(row_offset) * K);
      ptr_b_host[static_cast<size_t>(active_index)] = reinterpret_cast<const ElementInputB*>(
          weights_fp8 + static_cast<size_t>(expert_index) * N * K);
    }
    ptr_scale_a_host[static_cast<size_t>(active_index)] = reinterpret_cast<const ElementScale*>(
        scales_bdpas_a + scale_a_offsets_host[static_cast<size_t>(active_index)]);
    ptr_scale_b_host[static_cast<size_t>(active_index)] = reinterpret_cast<const ElementScale*>(
      scales_bdpas_b + static_cast<size_t>(expert_index) * N * scale_groups);
    ptr_d_host[static_cast<size_t>(active_index)] = reinterpret_cast<ElementOutput*>(outputs) +
                            static_cast<size_t>(row_offset) * N;

    auto shape_a = make_shape(token_count, K, 1);
    auto shape_b = make_shape(N, K, 1);
    auto shape_d = make_shape(token_count, N, 1);
    auto shape_scale_a = make_shape(padded_tokens, scale_groups, 1);
    auto shape_scale_b = make_shape(N, scale_groups, 1);
    stride_a_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, shape_a));
    stride_b_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, shape_b));
    stride_c_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, shape_d));
    stride_d_host.push_back(cutlass::make_cute_packed_stride(StrideD{}, shape_d));
    stride_scale_a_host.push_back(cutlass::make_cute_packed_stride(StrideScaleA{}, shape_scale_a));
    stride_scale_b_host.push_back(cutlass::make_cute_packed_stride(StrideScaleB{}, shape_scale_b));
  }

  if (refresh_metadata) {
    std::vector<uint8_t> metadata_host(metadata_bytes, 0);
    auto copy_metadata = [&](auto* device_dst, auto const* host_src, size_t count) {
      using Value = std::remove_pointer_t<decltype(device_dst)>;
      const size_t offset = reinterpret_cast<uint8_t*>(device_dst) - metadata;
      std::memcpy(metadata_host.data() + offset, host_src, sizeof(Value) * count);
    };
    copy_metadata(problem_shapes_device, problem_shapes_host.data(), active_count);
    copy_metadata(ptr_a_device, ptr_a_host.data(), active_count);
    copy_metadata(ptr_b_device, ptr_b_host.data(), active_count);
    copy_metadata(ptr_scale_a_device, ptr_scale_a_host.data(), active_count);
    copy_metadata(ptr_scale_b_device, ptr_scale_b_host.data(), active_count);
    copy_metadata(ptr_c_device, ptr_c_host.data(), active_count);
    copy_metadata(ptr_d_device, ptr_d_host.data(), active_count);
    copy_metadata(stride_a_device, stride_a_host.data(), active_count);
    copy_metadata(stride_b_device, stride_b_host.data(), active_count);
    copy_metadata(stride_c_device, stride_c_host.data(), active_count);
    copy_metadata(stride_d_device, stride_d_host.data(), active_count);
    copy_metadata(stride_scale_a_device, stride_scale_a_host.data(), active_count);
    copy_metadata(stride_scale_b_device, stride_scale_b_host.data(), active_count);
    copy_metadata(active_experts_device, active_experts_host.data(), active_count);
    copy_metadata(row_offsets_device, row_offsets_host.data(), active_count);
    copy_metadata(token_counts_device, token_counts_host.data(), active_count);
    copy_metadata(padded_token_counts_device, padded_token_counts_host.data(), active_count);
    copy_metadata(scale_a_offsets_device, scale_a_offsets_host.data(), active_count);
    queue->memcpy(metadata, metadata_host.data(), metadata_bytes);
  }
  if (refresh_weight_staging) {
    if constexpr (!NativeMxfp4) {
      launch_unpack_mxfp4_to_fp8(queue, static_cast<const uint8_t*>(weights), weights_fp8, num_experts, N, K);
    }
    launch_transpose_weight_scales(queue, static_cast<const uint8_t*>(weight_scales), scales_bdpas_b, num_experts, N,
                                   scale_groups);
  }
  launch_transpose_activation_scales(queue, static_cast<const uint8_t*>(activation_scales), scales_bdpas_a,
                                     row_offsets_device, token_counts_device, scale_a_offsets_device,
                                     padded_token_counts_device, active_count, scale_groups, max_padded_tokens);

  cutlass::KernelHardwareInfo hardware_info;
  hardware_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hardware_info.device_id);
  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerXeGroup<ProblemShape>::RasterOrderOptions;
  typename Gemm::GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      ProblemShape{active_count, problem_shapes_device, problem_shapes_host.data()},
      typename Gemm::GemmKernel::MainloopArguments{ptr_a_device, stride_a_device, ptr_b_device, stride_b_device,
                                                   ptr_scale_a_device, stride_scale_a_device, ptr_scale_b_device,
                                                   stride_scale_b_device},
      typename Gemm::GemmKernel::EpilogueArguments{{ElementAccumulator(1), ElementAccumulator(0)}, ptr_c_device,
                                                   stride_c_device, ptr_d_device, stride_d_device},
      hardware_info,
      typename Gemm::GemmKernel::TileSchedulerArguments{1, RasterOrderOptions::AlongN}};

  Gemm gemm_op;
  if (gemm_op.can_implement(arguments) != cutlass::Status::kSuccess) return false;
  const size_t workspace_size = Gemm::get_workspace_size(arguments);
  void* workspace = nullptr;
  if (workspace_size > 0) {
    workspace = DeviceMemoryPool::Instance()->get_scratch_mem(workspace_size, kWorkspaceScratchLoc, queue);
    if (workspace == nullptr) throw std::runtime_error("mxfp8_mxfp4_bdpas: failed to allocate scheduler workspace");
  }
  if (gemm_op.initialize(arguments, workspace, queue) != cutlass::Status::kSuccess) return false;
  if (gemm_op.run(queue) != cutlass::Status::kSuccess) return false;
  return true;
}

}  // namespace moe_mxfp_bdpas_detail

bool sycl_tla_moe_prefill_mxfp8_mxfp4_bdpas(sycl::queue* q, void* activations, void* activation_scales,
                                            void* weights, void* weight_scales, void* outputs,
                                            void* activation_workspace, void* weight_workspace,
                                            BTLA_DTYPE output_dtype, BTLA_DTYPE activation_dtype, int N, int K,
                                            int group_size, int* num_tokens_per_expert, int num_experts,
                                            int total_tokens, bool refresh_weight_staging,
                                            int* num_tokens_per_expert_host, bool refresh_metadata) {
  if (!moe_mxfp_bdpas_detail::env_enabled("ARK_MOE_PREFILL_BDPAS_MXFP8_MXFP4")) return false;
  if (activation_dtype == BTLA_DTYPE::F8_E4M3 && output_dtype == BTLA_DTYPE::BF16) {
    return moe_mxfp_bdpas_detail::run_grouped_bdpas<cutlass::float_e4m3_t, false>(
        q, activations, activation_scales, weights, weight_scales, outputs, activation_workspace, weight_workspace,
      N, K, group_size, num_tokens_per_expert, num_experts, total_tokens, refresh_weight_staging,
      num_tokens_per_expert_host, refresh_metadata);
  }
  if (activation_dtype == BTLA_DTYPE::F8_E5M2 && output_dtype == BTLA_DTYPE::BF16) {
    return false;
  }
  return false;
}

bool sycl_tla_moe_prefill_mxfp4_mxfp4_bdpas(sycl::queue* q, void* activations, void* activation_scales,
                                            void* weights, void* weight_scales, void* outputs,
                                            void* activation_workspace, void* weight_workspace,
                                            BTLA_DTYPE output_dtype, BTLA_DTYPE activation_dtype, int N, int K,
                                            int group_size, int* num_tokens_per_expert, int num_experts,
                                            int total_tokens, bool refresh_weight_staging,
                                            int* num_tokens_per_expert_host, bool refresh_metadata) {
  if (!moe_mxfp_bdpas_detail::env_enabled("ARK_MOE_PREFILL_BDPAS_MXFP4_MXFP4")) return false;
  if (activation_dtype == BTLA_DTYPE::F4_E2M1 && output_dtype == BTLA_DTYPE::BF16) {
    return moe_mxfp_bdpas_detail::run_grouped_bdpas<cutlass::float_e2m1_t, true>(
        q, activations, activation_scales, weights, weight_scales, outputs, activation_workspace, weight_workspace,
        N, K, group_size, num_tokens_per_expert, num_experts, total_tokens, refresh_weight_staging,
      num_tokens_per_expert_host, refresh_metadata);
  }
  return false;
}

}  // namespace ark

#endif