#pragma once

#include "sycl_tla_common.hpp"

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

namespace ark {
namespace dense_woq_s4_dpas {

inline constexpr int kMinGroupSize = 32;
inline constexpr int kMaxGroupSize = 4096;
inline constexpr int kPackedScaleMaxGroupSize = 128;

inline bool is_supported_group_size(int group_size) {
  return group_size >= kMinGroupSize && group_size <= kMaxGroupSize &&
         (group_size & (group_size - 1)) == 0;
}

namespace detail {

#define ARK_DECLARE_DENSE_WOQ_S4_DPAS_ROUTE(prefix)                                                           \
  void prefix##_group_n(sycl::queue* q, const void* activations, const void* weights, const void* scales,     \
                        const void* bias, void* outputs, int gemm_m, int gemm_n, int gemm_k, int group_size); \
  void prefix##_n_group(sycl::queue* q, const void* activations, const void* weights, const void* scales,     \
                        const void* bias, void* outputs, int gemm_m, int gemm_n, int gemm_k, int group_size)

ARK_DECLARE_DENSE_WOQ_S4_DPAS_ROUTE(run_m4_n128);
ARK_DECLARE_DENSE_WOQ_S4_DPAS_ROUTE(run_m8_n128);
ARK_DECLARE_DENSE_WOQ_S4_DPAS_ROUTE(run_m16);
ARK_DECLARE_DENSE_WOQ_S4_DPAS_ROUTE(run_m32);
ARK_DECLARE_DENSE_WOQ_S4_DPAS_ROUTE(run_m32_n256);
ARK_DECLARE_DENSE_WOQ_S4_DPAS_ROUTE(run_m64_n256);
ARK_DECLARE_DENSE_WOQ_S4_DPAS_ROUTE(run_m128);

#undef ARK_DECLARE_DENSE_WOQ_S4_DPAS_ROUTE

}  // namespace detail
}  // namespace dense_woq_s4_dpas
}  // namespace ark

#endif