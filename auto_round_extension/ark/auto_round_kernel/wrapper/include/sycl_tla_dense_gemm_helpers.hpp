#pragma once

#include "sycl_tla_common.hpp"

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

namespace ark {
namespace dense_gemm_detail {

#define ARK_DECLARE_DENSE_GEMM_ROUTES(prefix) \
    void prefix##_with_bias_small(sycl::queue* q, int m, int n, int k, const void* a, const void* b, void* c, \
                                                                const void* bias); \
    void prefix##_with_bias_mid(sycl::queue* q, int m, int n, int k, const void* a, const void* b, void* c, \
                                                            const void* bias); \
    void prefix##_with_bias_medium(sycl::queue* q, int m, int n, int k, const void* a, const void* b, void* c, \
                                                                 const void* bias); \
    void prefix##_with_bias_large(sycl::queue* q, int m, int n, int k, const void* a, const void* b, void* c, \
                                                                const void* bias); \
    void prefix##_without_bias_small(sycl::queue* q, int m, int n, int k, const void* a, const void* b, void* c); \
    void prefix##_without_bias_mid(sycl::queue* q, int m, int n, int k, const void* a, const void* b, void* c); \
    void prefix##_without_bias_medium(sycl::queue* q, int m, int n, int k, const void* a, const void* b, void* c); \
    void prefix##_without_bias_large(sycl::queue* q, int m, int n, int k, const void* a, const void* b, void* c)

ARK_DECLARE_DENSE_GEMM_ROUTES(run_f16);
ARK_DECLARE_DENSE_GEMM_ROUTES(run_bf16);
ARK_DECLARE_DENSE_GEMM_ROUTES(run_f32);

#undef ARK_DECLARE_DENSE_GEMM_ROUTES

}  // namespace dense_gemm_detail
}  // namespace ark

#endif