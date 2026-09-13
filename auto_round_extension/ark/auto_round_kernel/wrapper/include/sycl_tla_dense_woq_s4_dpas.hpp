// SYCL-TLA Dense WOQ S4 DPAS Wrapper

#pragma once

#include <stdexcept>
#include <type_traits>

#ifdef ARK_XPU
#include <sycl/sycl.hpp>
#endif

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "sycl_tla_moe_prefill_s4_dpas.hpp"
#endif

namespace ark {

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

namespace dense_woq_s4_dpas {

using namespace cute;

using ::ark::moe_dpas_s4::cute_scalar_t;
using ::ark::moe_dpas_s4::dpas_w4a16_policy;
using ::ark::moe_dpas_s4::dpas_w4a16_policy_m_8;
using ::ark::moe_dpas_s4::dpas_w4a16_policy_m_16;
using ::ark::moe_dpas_s4::dpas_w4a16_policy_m_32;
using ::ark::moe_dpas_s4::make_moe_tensor;
using ::ark::moe_dpas_s4::xe_gemm_s4_pergroup;

inline constexpr int kMinGroupSize = 32;
inline constexpr int kMaxGroupSize = 4096;

inline bool is_supported_group_size(int group_size) {
  return group_size >= kMinGroupSize && group_size <= kMaxGroupSize &&
         (group_size & (group_size - 1)) == 0;
}

template <typename ElementA, typename ElementB, typename ElementS,
          typename ElementBI, typename ElementD, char layoutA, char layoutB,
          class policy, int GroupSize>
class DenseWoqS4DpasName;

template <class GmemTiledCopyA, class GmemTiledCopyB, class GmemTiledCopyD,
          char LayoutKindA, char LayoutKindB, char LayoutKindD,
          class TiledMMA, int GroupSize, typename ElementA, typename ElementB,
          typename ElementS, typename ElementBI, typename ElementD>
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

  auto A_tensor = make_moe_tensor<ElementA, LayoutKindA>(
      const_cast<ElementA*>(Activations), gemm_m, gemm_k);
  auto B_tensor = make_moe_tensor<ElementB, actual_layout_of_B>(
      const_cast<ElementB*>(Weights), gemm_n, gemm_k);
  auto D_tensor = make_moe_tensor<ElementD, LayoutKindD>(Outputs, gemm_m,
                                                        gemm_n);
  auto tile_coord = make_coord(wg_m, wg_n, _, 0);

  xe_gemm_s4_pergroup<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD,
                       GroupSize, true>(A_tensor, B_tensor, Scales, Bias,
                                        D_tensor, tile_coord, mma);
}

template <char layoutA, char layoutB, class policy, int GroupSize,
          typename ElementA, typename ElementB, typename ElementS,
          typename ElementBI, typename ElementD>
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
                                        layoutB, policy, GroupSize>>(
        sycl::nd_range<3>{groups * local, local}, kernel_props, [=](auto) {
          DenseWoqS4GEMM<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD,
                         layoutA, layoutB, 'R', MMA, GroupSize>(
              activations, weights, scales, bias, outputs, mma, gemm_m, gemm_n,
              gemm_k);
        });
  });

  EventManager::getInstance().addEvent(event);
  if (wait) event.wait();
}

template <int GroupSize, char layoutA, char layoutB, class policy,
          typename ElementA, typename ElementB, typename ElementS,
          typename ElementBI, typename ElementD>
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
    DenseWoqS4GEMMLauncherGroup<layoutA, layoutB, policy, GroupSize>(
        stream, activations, weights, scales, bias, outputs, gemm_m, gemm_n,
        gemm_k, wait);
    return true;
  }
  if constexpr (GroupSize < kMaxGroupSize) {
    return DenseWoqS4GEMMLauncherDispatch<GroupSize * 2, layoutA, layoutB,
                                          policy>(
        stream, activations, weights, scales, bias, outputs, gemm_m, gemm_n,
        gemm_k, group_size, wait);
  }
  return false;
}

template <char layoutA, char layoutB, class policy, typename ElementA,
          typename ElementB, typename ElementS, typename ElementBI,
          typename ElementD>
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
                                      policy>(
          stream, activations, weights, scales, bias, outputs, gemm_m, gemm_n,
          gemm_k, group_size, wait)) {
    throw std::runtime_error("dense_woq_s4_dpas: unsupported group size");
  }
}

}  // namespace dense_woq_s4_dpas

#endif  // ARK_XPU && ARK_SYCL_TLA

}  // namespace ark
