// SYCL MoE prefill low-bit to INT8 upcast kernels

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#ifdef ARK_XPU
#include <sycl/sycl.hpp>
#endif

#include "sycl_tla_moe_dequant.hpp"

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

namespace ark {
namespace moe_mixed_detail {

class MoEUpcastInt4SymToInt8Kernel;
class MoEUpcastInt2SymToInt8Kernel;
class MoEUpcastInt4SymToInt8KernelFast;
class MoEUpcastInt2SymToInt8KernelFast;

constexpr int UPCAST_WG_N = 32;
constexpr int PACK_K_INT4 = 2;
constexpr int PACK_K_INT2 = 4;
constexpr int PACK_K_INT4_FAST = 8;
constexpr int PACK_K_INT2_FAST = 8;

inline void launch_upcast_int4_sym_to_int8(sycl::queue* q, const uint8_t* weights_NKp, int8_t* weights_i8_NK,
                                           int E, int N, int K,
                                           const int* num_tokens_per_expert = nullptr) {
  if (E == 0 || N == 0 || K == 0) return;
  if ((K & 1) != 0) {
    throw std::invalid_argument("moe_gemm_prefill(int4->int8 upcast): K must be even");
  }
  const int k_packed = K / 2;

  if ((K % PACK_K_INT4_FAST) == 0) {
    const int k_words = K / PACK_K_INT4_FAST;
    sycl::range<3> global_fast{static_cast<size_t>(E), static_cast<size_t>(k_words),
                               static_cast<size_t>((N + UPCAST_WG_N - 1) / UPCAST_WG_N) * UPCAST_WG_N};
    sycl::range<3> local_fast{1, 1, static_cast<size_t>(UPCAST_WG_N)};

    q->parallel_for<MoEUpcastInt4SymToInt8KernelFast>(
        sycl::nd_range<3>(global_fast, local_fast), [=](sycl::nd_item<3> it) {
          const int e = static_cast<int>(it.get_global_id(0));
          if (num_tokens_per_expert != nullptr && num_tokens_per_expert[e] == 0) return;
          const int kw = static_cast<int>(it.get_global_id(1));
          const int n = static_cast<int>(it.get_global_id(2));
          if (n >= N) return;
          const int k_base = kw * PACK_K_INT4_FAST;
          const size_t row_kp_base = (static_cast<size_t>(e) * N + static_cast<size_t>(n)) * k_packed +
                                     static_cast<size_t>(kw) * 4;
          const uint32_t packed =
              static_cast<uint32_t>(weights_NKp[row_kp_base + 0]) |
              (static_cast<uint32_t>(weights_NKp[row_kp_base + 1]) << 8) |
              (static_cast<uint32_t>(weights_NKp[row_kp_base + 2]) << 16) |
              (static_cast<uint32_t>(weights_NKp[row_kp_base + 3]) << 24);
          int qv[8];
          moe_dequant::decode_int4_octet<false>(packed, qv);
          const size_t out_row = (static_cast<size_t>(e) * N + static_cast<size_t>(n)) * K;
#pragma unroll
          for (int j = 0; j < PACK_K_INT4_FAST; ++j) {
            weights_i8_NK[out_row + static_cast<size_t>(k_base + j)] = static_cast<int8_t>(qv[j]);
          }
        });
    return;
  }

  sycl::range<3> global{static_cast<size_t>(E), static_cast<size_t>(k_packed),
                        static_cast<size_t>((N + UPCAST_WG_N - 1) / UPCAST_WG_N) * UPCAST_WG_N};
  sycl::range<3> local{1, 1, static_cast<size_t>(UPCAST_WG_N)};

  q->parallel_for<MoEUpcastInt4SymToInt8Kernel>(
      sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> it) {
        const int e = static_cast<int>(it.get_global_id(0));
        if (num_tokens_per_expert != nullptr && num_tokens_per_expert[e] == 0) return;
        const int kp = static_cast<int>(it.get_global_id(1));
        const int n = static_cast<int>(it.get_global_id(2));
        if (n >= N) return;
        const int k_base = kp * PACK_K_INT4;
        const uint8_t packed =
            weights_NKp[(static_cast<size_t>(e) * N + static_cast<size_t>(n)) * k_packed + static_cast<size_t>(kp)];
        int q_lo, q_hi;
        moe_dequant::decode_int4_pair<false>(packed, q_lo, q_hi);
        const size_t out_row = (static_cast<size_t>(e) * N + static_cast<size_t>(n)) * K;
        weights_i8_NK[out_row + static_cast<size_t>(k_base)] = static_cast<int8_t>(q_lo);
        weights_i8_NK[out_row + static_cast<size_t>(k_base + 1)] = static_cast<int8_t>(q_hi);
      });
}

inline void launch_upcast_int2_sym_to_int8(sycl::queue* q, const uint8_t* weights_NKp, int8_t* weights_i8_NK,
                                           int E, int N, int K,
                                           const int* num_tokens_per_expert = nullptr) {
  if (E == 0 || N == 0 || K == 0) return;
  if ((K & 3) != 0) {
    throw std::invalid_argument("moe_gemm_prefill(int2->int8 upcast): K must be a multiple of 4");
  }
  const int k_packed = K / 4;

  if ((K % PACK_K_INT2_FAST) == 0) {
    const int k_words = K / PACK_K_INT2_FAST;
    sycl::range<3> global_fast{static_cast<size_t>(E), static_cast<size_t>(k_words),
                               static_cast<size_t>((N + UPCAST_WG_N - 1) / UPCAST_WG_N) * UPCAST_WG_N};
    sycl::range<3> local_fast{1, 1, static_cast<size_t>(UPCAST_WG_N)};

    q->parallel_for<MoEUpcastInt2SymToInt8KernelFast>(
        sycl::nd_range<3>(global_fast, local_fast), [=](sycl::nd_item<3> it) {
          const int e = static_cast<int>(it.get_global_id(0));
          if (num_tokens_per_expert != nullptr && num_tokens_per_expert[e] == 0) return;
          const int kw = static_cast<int>(it.get_global_id(1));
          const int n = static_cast<int>(it.get_global_id(2));
          if (n >= N) return;
          const int k_base = kw * PACK_K_INT2_FAST;
          const size_t row_kp_base = (static_cast<size_t>(e) * N + static_cast<size_t>(n)) * k_packed +
                                     static_cast<size_t>(kw) * 2;
          const uint16_t packed = static_cast<uint16_t>(static_cast<uint32_t>(weights_NKp[row_kp_base + 0]) |
                                                        (static_cast<uint32_t>(weights_NKp[row_kp_base + 1]) << 8));
          int qv[8];
          moe_dequant::decode_int2_octet<false>(packed, qv);
          const size_t out_row = (static_cast<size_t>(e) * N + static_cast<size_t>(n)) * K;
#pragma unroll
          for (int j = 0; j < PACK_K_INT2_FAST; ++j) {
            weights_i8_NK[out_row + static_cast<size_t>(k_base + j)] = static_cast<int8_t>(qv[j]);
          }
        });
    return;
  }

  sycl::range<3> global{static_cast<size_t>(E), static_cast<size_t>(k_packed),
                        static_cast<size_t>((N + UPCAST_WG_N - 1) / UPCAST_WG_N) * UPCAST_WG_N};
  sycl::range<3> local{1, 1, static_cast<size_t>(UPCAST_WG_N)};

  q->parallel_for<MoEUpcastInt2SymToInt8Kernel>(
      sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> it) {
        const int e = static_cast<int>(it.get_global_id(0));
        if (num_tokens_per_expert != nullptr && num_tokens_per_expert[e] == 0) return;
        const int kp = static_cast<int>(it.get_global_id(1));
        const int n = static_cast<int>(it.get_global_id(2));
        if (n >= N) return;
        const int k_base = kp * PACK_K_INT2;
        const uint8_t packed =
            weights_NKp[(static_cast<size_t>(e) * N + static_cast<size_t>(n)) * k_packed + static_cast<size_t>(kp)];
        int qv[4];
        moe_dequant::decode_int2_quad<false>(packed, qv);
        const size_t out_row = (static_cast<size_t>(e) * N + static_cast<size_t>(n)) * K;
#pragma unroll
        for (int j = 0; j < PACK_K_INT2; ++j) {
          weights_i8_NK[out_row + static_cast<size_t>(k_base + j)] = static_cast<int8_t>(qv[j]);
        }
      });
}

}  // namespace moe_mixed_detail
}  // namespace ark

#endif