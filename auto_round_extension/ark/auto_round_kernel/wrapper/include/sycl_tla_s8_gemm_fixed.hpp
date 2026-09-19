#pragma once

#include "sycl_tla_s8_gemm.hpp"

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

#define ARK_DEFINE_S8_NORMAL_ROUTE(name, element, tile_m, tile_n, layout) \
  namespace ark { \
  namespace sycl_tla_s8_detail { \
  void name(sycl::queue* q, int m, int n, int k, const void* a, const void* b, void* c, const void* scale_a, \
            const void* scale_b, const void* bias) { \
    run_igemm_normal_fixed_tile<element, tile_m, tile_n, layout>( \
        q, m, n, k, static_cast<const int8_t*>(a), static_cast<const int8_t*>(b), static_cast<element*>(c), \
        static_cast<const element*>(scale_a), static_cast<const element*>(scale_b), static_cast<const element*>(bias)); \
  } \
  } \
  }

#define ARK_DEFINE_S8_KBLOCK_ROUTE(name, element, tile_m, tile_n, layout) \
  namespace ark { \
  namespace sycl_tla_s8_detail { \
  void name(sycl::queue* q, int m, int n, int k, const void* a, const void* b, void* c, const void* scale_a, \
            const void* scale_b, const void* bias, int blocksize) { \
    run_igemm_kblock_fixed_tile<element, tile_m, tile_n, layout>( \
        q, m, n, k, static_cast<const int8_t*>(a), static_cast<const int8_t*>(b), static_cast<element*>(c), \
        static_cast<const element*>(scale_a), static_cast<const element*>(scale_b), static_cast<const element*>(bias), \
        blocksize); \
  } \
  } \
  }

#endif