#pragma once

#include "sycl_tla_common.hpp"

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

#include <stdexcept>

namespace ark {
namespace sycl_tla_s8_detail {

#define ARK_DECLARE_S8_NORMAL_ROUTES(prefix) \
  void prefix##_normal_small(sycl::queue* q, int m, int n, int k, const void* a, const void* b, void* c, \
                             const void* scale_a, const void* scale_b, const void* bias); \
  void prefix##_normal_mid(sycl::queue* q, int m, int n, int k, const void* a, const void* b, void* c, \
                           const void* scale_a, const void* scale_b, const void* bias); \
  void prefix##_normal_medium(sycl::queue* q, int m, int n, int k, const void* a, const void* b, void* c, \
                              const void* scale_a, const void* scale_b, const void* bias); \
  void prefix##_normal_large(sycl::queue* q, int m, int n, int k, const void* a, const void* b, void* c, \
                             const void* scale_a, const void* scale_b, const void* bias)

#define ARK_DECLARE_S8_KBLOCK_ROUTES(prefix) \
  void prefix##_kblock_small(sycl::queue* q, int m, int n, int k, const void* a, const void* b, void* c, \
                             const void* scale_a, const void* scale_b, const void* bias, int blocksize); \
  void prefix##_kblock_mid(sycl::queue* q, int m, int n, int k, const void* a, const void* b, void* c, \
                           const void* scale_a, const void* scale_b, const void* bias, int blocksize); \
  void prefix##_kblock_medium(sycl::queue* q, int m, int n, int k, const void* a, const void* b, void* c, \
                              const void* scale_a, const void* scale_b, const void* bias, int blocksize); \
  void prefix##_kblock_large(sycl::queue* q, int m, int n, int k, const void* a, const void* b, void* c, \
                             const void* scale_a, const void* scale_b, const void* bias, int blocksize)

ARK_DECLARE_S8_NORMAL_ROUTES(run_f16);
ARK_DECLARE_S8_NORMAL_ROUTES(run_bf16);
ARK_DECLARE_S8_NORMAL_ROUTES(run_f32);
ARK_DECLARE_S8_KBLOCK_ROUTES(run_f16);
ARK_DECLARE_S8_KBLOCK_ROUTES(run_f32);

#undef ARK_DECLARE_S8_NORMAL_ROUTES
#undef ARK_DECLARE_S8_KBLOCK_ROUTES

inline void validate_kblock_args(int k, int blocksize) {
  if (blocksize <= 0 || k % blocksize != 0) {
    throw std::invalid_argument("sycl_tla_igemm_s8s8_dequant: blocksize must divide k");
  }
  if (blocksize % 64 != 0) {
    throw std::invalid_argument("sycl_tla_igemm_s8s8_dequant: k-block blocksize must be a multiple of 64");
  }
}

}  // namespace sycl_tla_s8_detail
}  // namespace ark

#endif