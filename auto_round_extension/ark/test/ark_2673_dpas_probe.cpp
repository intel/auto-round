#include <cstdint>
#include <stdexcept>

#include <pybind11/pybind11.h>

#include "sycl_tla_dense_woq_s4_dpas.hpp"

namespace ark::dense_woq_s4_dpas::probe {

constexpr int kGroupSize = 128;
using Half = cute_scalar_t<sycl::half>;
using Weight = cutlass::uint4b_t;

template <class Policy, int Splits, int GroupSize>
class SplitKS4ProbeName;

template <class Policy, int Splits, int GroupSize>
class ReduceS4ProbeName;

template <class ATensor, class BTensor, class CTensor, class Mma>
CUTE_DEVICE void group32_tile64(const ATensor& input, const BTensor& weight, const Half* scales, CTensor& output,
                                int tile_row, int tile_column, const Mma& mma) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int thread_id = item.get_local_linear_id();
  const auto tile = mma.tile_mnk();
  auto tile_a = local_tile(make_identity_tensor(input.shape()), select<0, 2>(tile), make_coord(tile_row, _));
  auto tile_b = local_tile(make_identity_tensor(weight.shape()), select<1, 2>(tile), make_coord(tile_column, _));
  auto tile_c = local_tile(make_identity_tensor(output.shape()), tile, make_coord(tile_row, tile_column, 0),
                           Step<_1, _1, X>{});
  auto copy_a = get_block_2d_copy_A<void>(mma, input);
  auto copy_b = get_block_2d_copy_B<void>(mma, weight);
  auto copy_c = get_block_2d_copy_D<void>(mma, output);
  auto thread_a = copy_a.get_slice(thread_id);
  auto thread_b = copy_b.get_slice(thread_id);
  auto thread_c = copy_c.get_slice(thread_id);
  auto thread_mma = mma.get_slice(thread_id);
  auto indices_a = thread_a.partition_S(tile_a);
  auto indices_b = thread_b.partition_S(tile_b);
  auto registers_a = thread_a.partition_sg_fragment_D(tile_a(_, _, 0));
  auto registers_b = thread_b.partition_sg_fragment_D(tile_b(_, _, 0));
  auto frag_a = thread_mma.partition_sg_fragment_A(tile_a(_, _, 0));
  auto frag_b = thread_mma.partition_sg_fragment_B(tile_b(_, _, 0));
  auto group_sum = thread_mma.partition_sg_fragment_C(tile_c);
  auto result = thread_mma.partition_sg_fragment_C(tile_c);
  auto store = thread_c.partition_sg_fragment_S(tile_c);
  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);
  auto prefetch_indices_a = prefetch_a.get_slice(thread_id).partition_S(tile_a);
  auto prefetch_indices_b = prefetch_b.get_slice(thread_id).partition_S(tile_b);
  constexpr auto thread_layout = typename Mma::ThrLayoutVMNK{}.shape();
  constexpr int tile_k = int(get<2>(decltype(tile){}));
  constexpr int tile_n = int(get<1>(decltype(tile){}));
  constexpr int subgroup_m = int(get<0>(decltype(tile){})) / int(get<1>(thread_layout));
  constexpr int subgroup_n = tile_n / int(get<2>(thread_layout));
  constexpr int groups_per_tile = 2;
  constexpr int atom_steps = decltype(size<2>(frag_a.tensor()))::value;
  static_assert(tile_k == 64 && subgroup_n == 16 && atom_steps % groups_per_tile == 0);
  constexpr int steps_per_group = atom_steps / groups_per_tile;
  constexpr int prefetch_distance = 3;
  const int scale_column = tile_column * tile_n +
                           (cutlass::get_sub_group_id() % int(get<2>(thread_layout))) * subgroup_n +
                           cutlass::get_sub_group_local_id();
  const int columns = shape<0>(weight);
  const int tile_count = shape<1>(input) / tile_k;
  clear(result);

  CUTE_UNROLL
  for (int next = 0; next < prefetch_distance && next < tile_count; ++next) {
    prefetch(prefetch_a, prefetch_indices_a(_, _, _, next));
    prefetch(prefetch_b, prefetch_indices_b(_, _, _, next));
  }
  for (int k_tile = 0; k_tile < tile_count; ++k_tile) {
    barrier_arrive(ScopeWorkgroup);
    copy(copy_a, indices_a(_, _, _, k_tile), registers_a);
    copy(copy_b, indices_b(_, _, _, k_tile), registers_b);
    if (k_tile + prefetch_distance < tile_count) {
      prefetch(prefetch_a, prefetch_indices_a(_, _, _, k_tile + prefetch_distance));
      prefetch(prefetch_b, prefetch_indices_b(_, _, _, k_tile + prefetch_distance));
    }
    reorder(registers_a, frag_a);
    reorder(registers_b, frag_b);
    CUTE_UNROLL
    for (int nibble = 0; nibble < frag_b.size(); ++nibble) {
      frag_b(nibble) = static_cast<Half>(static_cast<float>(frag_b(nibble)) - 8.0f);
    }
    CUTE_UNROLL
    for (int group = 0; group < groups_per_tile; ++group) {
      clear(group_sum);
      CUTE_UNROLL
      for (int step = 0; step < steps_per_group; ++step) {
        const int atom = group * steps_per_group + step;
        cute::gemm(mma, frag_a.tensor()(_, _, atom), frag_b.tensor()(_, _, atom), group_sum);
      }
      const float scale = scales[(k_tile * groups_per_tile + group) * columns + scale_column];
      CUTE_UNROLL
      for (int row = 0; row < subgroup_m; ++row) {
        result(row) += group_sum(row) * scale;
      }
    }
    barrier_wait(ScopeWorkgroup);
  }
  reorder(result, store);
  copy(copy_c, store, thread_c.partition_D(tile_c));
}

template <class Policy, int Splits, int GroupSize = kGroupSize>
void launch(sycl::queue& queue, const Half* input, const uint8_t* weights, const Half* scales, Half* output,
            float* partials, int rows, int columns, int inner, const Half* bias = nullptr) {
  if constexpr (Splits == 1) {
    DenseWoqS4GEMMLauncherGroup<'R', 'C', Policy, kGroupSize, true>(
        queue, input, reinterpret_cast<const Weight*>(weights), scales, static_cast<const Half*>(nullptr), output,
        rows, columns, inner);
  } else {
    compat::set_default_queue(queue);
    using Atom = MMA_Atom<XE_DPAS_TT<8, float, Half, Half>>;
    using Mma = typename TiledMMAHelper<Atom, Layout<typename Policy::WGTile>, typename Policy::SGLayout>::TiledMMA;
    Mma mma;
    const int slice_k = inner / Splits;
    const size_t output_size = size_t(rows) * columns;
    const sycl::range<3> local(1, 1, size(mma));
    const int tile_m = get<0>(mma.tile_mnk());
    const sycl::range<3> groups(columns / 128, (rows + tile_m - 1) / tile_m, Splits);
    const sycl::ext::oneapi::experimental::properties props{
        sycl::ext::oneapi::experimental::sub_group_size<16>, sycl::ext::intel::experimental::grf_size<256>};
    queue.submit([&](sycl::handler& handler) {
      handler.parallel_for<SplitKS4ProbeName<Policy, Splits, GroupSize>>(
          sycl::nd_range<3>{groups * local, local}, props, [=](sycl::nd_item<3> item) {
            const int split = item.get_group(2);
            const int offset = split * slice_k;
            const auto shape_a = make_shape(rows, slice_k);
            const auto shape_b = make_shape(columns, slice_k);
            const auto stride = make_stride(inner, _1{});
            auto input_slice = make_tensor(make_gmem_ptr(const_cast<Half*>(input + offset)), make_layout(shape_a, stride));
            auto weight_slice = make_tensor(
                make_gmem_ptr(reinterpret_cast<Weight*>(const_cast<uint8_t*>(weights + offset / 2))),
                make_layout(shape_b, stride));
            auto partial_tile = make_dense_tensor<float, 'R'>(partials + split * output_size, rows, columns);
            const Half* slice_scales = scales + (offset / GroupSize) * columns;
            const int tile_row = item.get_group(1);
            const int tile_column = item.get_group(0);
            if constexpr (GroupSize == 32) {
              group32_tile64(input_slice, weight_slice, slice_scales, partial_tile, tile_row, tile_column, mma);
            } else {
              dense_gemm_s4_pergroup<void, void, void, GroupSize, true, true>(
                  input_slice, weight_slice, slice_scales, static_cast<const Half*>(nullptr), partial_tile,
                  make_coord(tile_row, tile_column, _, 0), mma);
            }
          });
    });
    queue.parallel_for<ReduceS4ProbeName<Policy, Splits, GroupSize>>(sycl::range<1>(output_size), [=](sycl::id<1> item) {
      const size_t index = item[0];
      float sum = partials[index];
      for (int split = 1; split < Splits; ++split) {
        sum += partials[split * output_size + index];
      }
      if (bias != nullptr) sum += static_cast<float>(bias[index % columns]);
      output[index] = static_cast<Half>(sum);
    });
  }
}

template <class Policy>
void dispatch(sycl::queue& queue, const Half* input, const uint8_t* weights, const Half* scales, Half* output,
              float* partials, int rows, int columns, int inner, int splits) {
  if (splits == 1) {
    launch<Policy, 1>(queue, input, weights, scales, output, partials, rows, columns, inner);
  } else if (splits == 2) {
    launch<Policy, 2>(queue, input, weights, scales, output, partials, rows, columns, inner);
  } else {
    launch<Policy, 4>(queue, input, weights, scales, output, partials, rows, columns, inner);
  }
}

void run(uintptr_t queue_ptr, uintptr_t input, uintptr_t weights, uintptr_t scales, uintptr_t output,
         uintptr_t partials, int rows, int columns, int inner, int tile_k, int splits) {
  if (!queue_ptr || !input || !weights || !scales || !output || rows < 1 || rows > 2 || columns <= 0 ||
      columns % 128 != 0 || inner <= 0 || (tile_k != 32 && tile_k != 64) ||
      (splits != 1 && splits != 2 && splits != 4) || (splits != 1 && !partials) ||
      inner % (splits * kGroupSize) != 0) {
    throw std::invalid_argument("expected M=1/2, aligned N, whole K groups, K32/K64 and split 1/2/4");
  }
  auto& queue = *reinterpret_cast<sycl::queue*>(queue_ptr);
  if (!queue.is_in_order()) throw std::invalid_argument("the DPAS probe requires an in-order queue");
  auto launcher = tile_k == 32 ? dispatch<dpas_w4a16_dense_policy_m_4_n128>
                               : dispatch<dpas_w4a16_dense_policy_m_4_n128_k64>;
  launcher(queue, reinterpret_cast<const Half*>(input), reinterpret_cast<const uint8_t*>(weights),
           reinterpret_cast<const Half*>(scales), reinterpret_cast<Half*>(output), reinterpret_cast<float*>(partials),
           rows, columns, inner, splits);
}

void run_k64_split4(uintptr_t queue_ptr, uintptr_t input, uintptr_t weights, uintptr_t scales, uintptr_t bias,
                   uintptr_t output, uintptr_t partials, int rows, int columns, int inner, int group_size) {
  const int slice_alignment = group_size == 32 ? 64 : kGroupSize;
  if (!queue_ptr || !input || !weights || !scales || !output || !partials || rows < 1 || rows > 128 ||
      columns <= 0 || columns > 16384 || columns % 128 != 0 || inner <= 0 || inner % (4 * slice_alignment) != 0 ||
      (group_size != 32 && group_size != kGroupSize)) {
    throw std::invalid_argument("K64 split4 requires group32/128, M=1..128, aligned N <= 16384, and whole K tiles and groups");
  }
  auto& queue = *reinterpret_cast<sycl::queue*>(queue_ptr);
  if (!queue.is_in_order()) throw std::invalid_argument("the DPAS probe requires an in-order queue");
  using Policy = dpas_w4a16_dense_policy_m_4_n128_k64;
  auto launcher = group_size == 32 ? launch<Policy, 4, 32> : launch<Policy, 4, kGroupSize>;
  launcher(
      queue, reinterpret_cast<const Half*>(input), reinterpret_cast<const uint8_t*>(weights),
      reinterpret_cast<const Half*>(scales), reinterpret_cast<Half*>(output), reinterpret_cast<float*>(partials),
      rows, columns, inner, reinterpret_cast<const Half*>(bias));
}

void run_group32_bias(uintptr_t queue_ptr, uintptr_t input, uintptr_t weights, uintptr_t scales, uintptr_t bias,
                      uintptr_t output, uintptr_t partials, int rows, int columns, int inner) {
  if (!bias) throw std::invalid_argument("run_group32_bias requires a bias pointer");
  run_k64_split4(queue_ptr, input, weights, scales, bias, output, partials, rows, columns, inner, 32);
}

}  // namespace ark::dense_woq_s4_dpas::probe

PYBIND11_MODULE(ark_2673_dpas_probe, module) {
  module.def("run", &ark::dense_woq_s4_dpas::probe::run);
  module.def("run_group32_bias", &ark::dense_woq_s4_dpas::probe::run_group32_bias);
  module.def("run_k64_split4", &ark::dense_woq_s4_dpas::probe::run_k64_split4);
}