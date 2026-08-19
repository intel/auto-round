#pragma once

#include <sycl/sycl.hpp>

#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "../include/sycl_tla_dense_gemm.hpp"

struct TestSyclTlaDenseGemmPerf {
  TestSyclTlaDenseGemmPerf() {
#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)
    run<sycl::half>(4096, 4096, 4096, 50, 100, false);
    run<sycl::half>(4096, 4096, 4096, 50, 100, true);
    run<sycl::ext::oneapi::bfloat16>(4096, 4096, 4096, 50, 100, false);
    run<sycl::ext::oneapi::bfloat16>(4096, 4096, 4096, 50, 100, true);
    
    run<sycl::half>(5120, 4096, 4096, 50, 100, false);
    run<sycl::half>(5120, 4096, 4096, 50, 100, true);
    run<sycl::ext::oneapi::bfloat16>(5120, 4096, 4096, 50, 100, false);
    run<sycl::ext::oneapi::bfloat16>(5120, 4096, 4096, 50, 100, true);
#else
    std::cout << "[sycl_tla_dense_gemm][perf] skipped: requires ARK_XPU=ON and ARK_SYCL_TLA=ON\n";
#endif
  }

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)
  template <typename T>
  static BTLA_DTYPE dtype_of();

  template <typename T>
  static const char* dtype_name();

  template <typename T>
  static std::vector<T> make_random_fp(size_t count, float low, float high, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(low, high);
    std::vector<T> values(count);
    for (auto& value : values) {
      value = T(dist(rng));
    }
    return values;
  }

  static double elapsed_ms(const sycl::event& start_event, const sycl::event& stop_event) {
    const auto start_ns = start_event.get_profiling_info<sycl::info::event_profiling::command_end>();
    const auto stop_ns = stop_event.get_profiling_info<sycl::info::event_profiling::command_start>();
    return static_cast<double>(stop_ns - start_ns) * 1.0e-6;
  }

  template <typename T>
  static void run(int m, int n, int k, int warmup, int iterations, bool has_bias) {
    sycl::property_list props{
        sycl::property::queue::enable_profiling(),
        sycl::property::queue::in_order(),
    };
    sycl::queue q(sycl::gpu_selector_v, props);

    const size_t a_count = static_cast<size_t>(m) * static_cast<size_t>(k);
    const size_t b_count = static_cast<size_t>(n) * static_cast<size_t>(k);
    const size_t c_count = static_cast<size_t>(m) * static_cast<size_t>(n);
    const size_t bias_count = static_cast<size_t>(n);

    auto host_a = make_random_fp<T>(a_count, -0.5f, 0.5f, 2026);
    auto host_b = make_random_fp<T>(b_count, -0.5f, 0.5f, 2027);
    std::vector<T> host_bias;
    if (has_bias) {
      host_bias = make_random_fp<T>(bias_count, -0.1f, 0.1f, 2028);
    }

    auto* A = sycl::aligned_alloc_device<T>(128, a_count, q);
    auto* B = sycl::aligned_alloc_device<T>(128, b_count, q);
    auto* C = sycl::aligned_alloc_device<T>(128, c_count, q);
    T* bias = nullptr;
    if (has_bias) {
      bias = sycl::aligned_alloc_device<T>(128, bias_count, q);
    }

    if (!A || !B || !C || (has_bias && !bias)) {
      throw std::runtime_error("test_sycl_tla_dense_gemm_perf: device allocation failed");
    }

    q.memcpy(A, host_a.data(), a_count * sizeof(T));
    q.memcpy(B, host_b.data(), b_count * sizeof(T));
    if (has_bias) {
      q.memcpy(bias, host_bias.data(), bias_count * sizeof(T));
    }
    q.wait();

    const BTLA_DTYPE dtype = dtype_of<T>();
    const T* bias_arg = has_bias ? bias : nullptr;

    for (int i = 0; i < warmup; ++i) {
      ark::sycl_tla_dense_gemm(&q, m, n, k, A, dtype, B, dtype, C, dtype, bias_arg, true);
    }
    q.wait();

    auto start_event = q.ext_oneapi_submit_barrier();
    for (int i = 0; i < iterations; ++i) {
      ark::sycl_tla_dense_gemm(&q, m, n, k, A, dtype, B, dtype, C, dtype, bias_arg, true);
    }
    auto stop_event = q.ext_oneapi_submit_barrier();
    stop_event.wait();

    const double total_ms = elapsed_ms(start_event, stop_event);
    const double avg_ms = total_ms / static_cast<double>(iterations);
    const double tflops =
        (2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k)) /
        (avg_ms * 1.0e-3) / 1.0e12;

    std::cout << std::fixed << std::setprecision(4)
              << "[sycl_tla_dense_gemm][perf] dtype=" << dtype_name<T>()
              << " bias=" << (has_bias ? "yes" : "no")
              << " shape=" << m << "x" << n << "x" << k
              << " warmup=" << warmup
              << " iterations=" << iterations
              << " avg_ms=" << avg_ms
              << " TFLOP/s=" << tflops << "\n";

    sycl::free(A, q);
    sycl::free(B, q);
    sycl::free(C, q);
    if (bias) {
      sycl::free(bias, q);
    }
  }
#endif
};

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)
template <>
inline BTLA_DTYPE TestSyclTlaDenseGemmPerf::dtype_of<float>() {
  return BTLA_DTYPE::F32;
}

template <>
inline BTLA_DTYPE TestSyclTlaDenseGemmPerf::dtype_of<sycl::half>() {
  return BTLA_DTYPE::F16;
}

template <>
inline BTLA_DTYPE TestSyclTlaDenseGemmPerf::dtype_of<sycl::ext::oneapi::bfloat16>() {
  return BTLA_DTYPE::BF16;
}

template <>
inline const char* TestSyclTlaDenseGemmPerf::dtype_name<float>() {
  return "fp32";
}

template <>
inline const char* TestSyclTlaDenseGemmPerf::dtype_name<sycl::half>() {
  return "fp16";
}

template <>
inline const char* TestSyclTlaDenseGemmPerf::dtype_name<sycl::ext::oneapi::bfloat16>() {
  return "bf16";
}
#endif