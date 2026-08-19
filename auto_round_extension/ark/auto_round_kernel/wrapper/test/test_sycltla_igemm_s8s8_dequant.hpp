#pragma once

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../include/sycl_tla_s8_gemm.hpp"

struct TestSyclTlaIgemmS8S8Dequant {
  TestSyclTlaIgemmS8S8Dequant() {
#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)
    run_case<float>("full_k", 16, 128, 128, -1, false);
    run_case<float>("full_k", 16, 128, 128, -1, true);
    run_case<float>("k_block", 32, 128, 256, 128, false);
    run_case<float>("k_block", 32, 128, 256, 128, true);

    run_case<sycl::half>("full_k", 16, 128, 128, -1, false);
    run_case<sycl::half>("full_k", 16, 128, 128, -1, true);
    run_case<sycl::half>("k_block", 32, 128, 256, 128, false);
    run_case<sycl::half>("k_block", 32, 128, 256, 128, true);

    run_case<sycl::ext::oneapi::bfloat16>("full_k", 16, 128, 128, -1, false);
    run_case<sycl::ext::oneapi::bfloat16>("full_k", 16, 128, 128, -1, true);

    constexpr int perf_warmup = 20;
    constexpr int perf_iterations = 100;

    run_perf_case<float>("perf_full_k", 4096, 4096, 4096, -1, false, perf_warmup, perf_iterations);
    run_perf_case<float>("perf_full_k", 4096, 4096, 4096, -1, true, perf_warmup, perf_iterations);
    run_perf_case<sycl::half>("perf_full_k", 4096, 4096, 4096, -1, false, perf_warmup, perf_iterations);
    run_perf_case<sycl::half>("perf_full_k", 4096, 4096, 4096, -1, true, perf_warmup, perf_iterations);

    run_perf_case<sycl::ext::oneapi::bfloat16>("perf_full_k", 4096, 4096, 4096, -1, false, perf_warmup,
                                               perf_iterations);
    run_perf_case<sycl::ext::oneapi::bfloat16>("perf_full_k", 4096, 4096, 4096, -1, true, perf_warmup,
                                               perf_iterations);
    
    
    run_perf_case<float>("perf_full_k", 5120, 4096, 4096, -1, false, perf_warmup, perf_iterations);
    run_perf_case<float>("perf_full_k", 5120, 4096, 4096, -1, true, perf_warmup, perf_iterations);
    run_perf_case<sycl::half>("perf_full_k", 5120, 4096, 4096, -1, false, perf_warmup, perf_iterations);
    run_perf_case<sycl::half>("perf_full_k", 5120, 4096, 4096, -1, true, perf_warmup, perf_iterations);

    run_perf_case<sycl::ext::oneapi::bfloat16>("perf_full_k", 5120, 4096, 4096, -1, false, perf_warmup,
                                               perf_iterations);
    run_perf_case<sycl::ext::oneapi::bfloat16>("perf_full_k", 5120, 4096, 4096, -1, true, perf_warmup,
                                               perf_iterations);
#else
    std::cout << "[sycl_tla_igemm_s8s8_dequant][UT] skipped: requires ARK_XPU=ON and ARK_SYCL_TLA=ON\n";
#endif
  }

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)
  template <typename T>
  static BTLA_DTYPE dtype_of();

  template <typename T>
  static const char* dtype_name();

  template <typename T>
  static T from_float(float value) {
    return static_cast<T>(value);
  }

  template <typename T>
  static float to_float(T value) {
    return static_cast<float>(value);
  }

  static double elapsed_ms(const sycl::event& start_event, const sycl::event& stop_event) {
    const auto start_ns = start_event.get_profiling_info<sycl::info::event_profiling::command_end>();
    const auto stop_ns = stop_event.get_profiling_info<sycl::info::event_profiling::command_start>();
    return static_cast<double>(stop_ns - start_ns) * 1.0e-6;
  }

  static std::vector<int8_t> make_random_s8(size_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-8, 7);
    std::vector<int8_t> values(count);
    for (auto& value : values) {
      value = static_cast<int8_t>(dist(rng));
    }
    return values;
  }

  template <typename T>
  static std::vector<T> make_random_fp(size_t count, float low, float high, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(low, high);
    std::vector<T> values(count);
    for (auto& value : values) {
      value = from_float<T>(dist(rng));
    }
    return values;
  }

  template <typename T>
  static std::vector<T> reference(const std::vector<int8_t>& a, const std::vector<int8_t>& b,
                                  const std::vector<T>& scale_a, const std::vector<T>& scale_b,
                                  const std::vector<T>& bias, int m, int n, int k, int blocksize, bool has_bias) {
    const bool k_block = !(blocksize == k || blocksize == -1);
    const int blks = k_block ? k / blocksize : 1;
    std::vector<T> ref(size_t(m) * size_t(n));

    for (int row = 0; row < m; ++row) {
      for (int col = 0; col < n; ++col) {
        float value = 0.0f;

        if (k_block) {
          for (int ib = 0; ib < blks; ++ib) {
            int32_t acc = 0;
            const int begin = ib * blocksize;
            const int end = begin + blocksize;

            for (int kk = begin; kk < end; ++kk) {
              acc += int32_t(a[size_t(row) * k + kk]) * int32_t(b[size_t(col) * k + kk]);
            }

            value += static_cast<float>(acc) * to_float(scale_b[size_t(col) * blks + ib]);
          }

          value *= to_float(scale_a[row]);
        } else {
          int32_t acc = 0;

          for (int kk = 0; kk < k; ++kk) {
            acc += int32_t(a[size_t(row) * k + kk]) * int32_t(b[size_t(col) * k + kk]);
          }

          value = static_cast<float>(acc) * to_float(scale_a[row]) * to_float(scale_b[col]);
        }

        if (has_bias) {
          value += to_float(bias[col]);
        }

        ref[size_t(row) * n + col] = from_float<T>(value);
      }
    }

    return ref;
  }

  template <typename T>
  static void check_result(const std::string& name, const std::vector<T>& got, const std::vector<T>& ref) {
    float atol = 1e-3f;
    float rtol = 1e-3f;

    if constexpr (std::is_same_v<T, sycl::half>) {
      atol = 5e-2f;
      rtol = 5e-2f;
    } else if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
      atol = 2e-1f;
      rtol = 8e-2f;
    }

    float max_abs = 0.0f;
    float max_rel = 0.0f;
    size_t bad_count = 0;

    for (size_t i = 0; i < got.size(); ++i) {
      const float actual = to_float(got[i]);
      const float expected = to_float(ref[i]);
      const float abs_diff = std::fabs(actual - expected);
      const float rel_diff = abs_diff / std::max(std::fabs(expected), 1e-6f);

      max_abs = std::max(max_abs, abs_diff);
      max_rel = std::max(max_rel, rel_diff);

      if (abs_diff > atol + rtol * std::fabs(expected)) {
        ++bad_count;
      }
    }

    if (bad_count != 0) {
      std::cerr << "[sycl_tla_igemm_s8s8_dequant][UT] " << name << " failed bad_count=" << bad_count
                << " max_abs=" << max_abs << " max_rel=" << max_rel << "\n";
      throw std::runtime_error("sycl_tla_igemm_s8s8_dequant UT failed");
    }

    std::cout << "[sycl_tla_igemm_s8s8_dequant][UT] " << name << " passed max_abs=" << max_abs
              << " max_rel=" << max_rel << "\n";
  }

  template <typename T>
  static void run_case(const std::string& name, int m, int n, int k, int blocksize, bool has_bias, bool verify = true,
                       int warmup = 0, int iterations = 0) {
    const bool k_block = !(blocksize == k || blocksize == -1);
    const int blks = k_block ? k / blocksize : 1;

    const size_t a_count = size_t(m) * size_t(k);
    const size_t b_count = size_t(n) * size_t(k);
    const size_t c_count = size_t(m) * size_t(n);
    const size_t scale_a_count = size_t(m);
    const size_t scale_b_count = k_block ? size_t(n) * size_t(blks) : size_t(n);
    const size_t bias_count = size_t(n);

    auto host_a = make_random_s8(a_count, 2026);
    auto host_b = make_random_s8(b_count, 2027);
    auto host_scale_a = make_random_fp<T>(scale_a_count, 0.001f, 0.008f, 2028);
    auto host_scale_b = make_random_fp<T>(scale_b_count, 0.001f, 0.008f, 2029);
    auto host_bias = has_bias ? make_random_fp<T>(bias_count, -0.5f, 0.5f, 2030) : std::vector<T>();

    std::vector<T> host_ref;
    if (verify) {
      host_ref = reference(host_a, host_b, host_scale_a, host_scale_b, host_bias, m, n, k, blocksize, has_bias);
    }

    sycl::property_list props{
        sycl::property::queue::enable_profiling(),
        sycl::property::queue::in_order(),
    };
    sycl::queue q(sycl::gpu_selector_v, props);

    auto* dev_a = sycl::aligned_alloc_device<int8_t>(128, a_count, q);
    auto* dev_b = sycl::aligned_alloc_device<int8_t>(128, b_count, q);
    auto* dev_c = sycl::aligned_alloc_device<T>(128, c_count, q);
    auto* dev_scale_a = sycl::aligned_alloc_device<T>(128, scale_a_count, q);
    auto* dev_scale_b = sycl::aligned_alloc_device<T>(128, scale_b_count, q);
    T* dev_bias = has_bias ? sycl::aligned_alloc_device<T>(128, bias_count, q) : nullptr;

    if (!dev_a || !dev_b || !dev_c || !dev_scale_a || !dev_scale_b || (has_bias && !dev_bias)) {
      throw std::runtime_error("sycl_tla_igemm_s8s8_dequant UT allocation failed");
    }

    q.memcpy(dev_a, host_a.data(), a_count * sizeof(int8_t));
    q.memcpy(dev_b, host_b.data(), b_count * sizeof(int8_t));
    q.memcpy(dev_scale_a, host_scale_a.data(), scale_a_count * sizeof(T));
    q.memcpy(dev_scale_b, host_scale_b.data(), scale_b_count * sizeof(T));
    if (has_bias) {
      q.memcpy(dev_bias, host_bias.data(), bias_count * sizeof(T));
    }
    q.wait();

    if (verify) {
      ark::sycl_tla_igemm_s8s8_dequant(&q, m, n, k, dev_a, dev_b, dev_c, dtype_of<T>(), dev_scale_a, dev_scale_b,
                                       dev_bias, blocksize);
      q.wait();

      std::vector<T> host_c(c_count);
      q.memcpy(host_c.data(), dev_c, c_count * sizeof(T)).wait();

      check_result(name + " dtype=" + dtype_name<T>() + " bias=" + (has_bias ? "yes" : "no"), host_c, host_ref);
    }

    if (iterations > 0) {
      for (int i = 0; i < warmup; ++i) {
        ark::sycl_tla_igemm_s8s8_dequant(&q, m, n, k, dev_a, dev_b, dev_c, dtype_of<T>(), dev_scale_a, dev_scale_b,
                                         dev_bias, blocksize);
      }
      q.wait();

      auto start_event = q.ext_oneapi_submit_barrier();
      for (int i = 0; i < iterations; ++i) {
        ark::sycl_tla_igemm_s8s8_dequant(&q, m, n, k, dev_a, dev_b, dev_c, dtype_of<T>(), dev_scale_a, dev_scale_b,
                                         dev_bias, blocksize);
      }
      auto stop_event = q.ext_oneapi_submit_barrier();
      stop_event.wait();

      const double avg_ms = elapsed_ms(start_event, stop_event) / static_cast<double>(iterations);
      const double tops = (2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k)) /
                          (avg_ms * 1.0e-3) / 1.0e12;

      std::cout << std::fixed << std::setprecision(4)
                << "[sycl_tla_igemm_s8s8_dequant][PERF] " << name << " out_dtype=" << dtype_name<T>()
                << " bias=" << (has_bias ? "yes" : "no") << " blocksize=" << blocksize
                << " shape=" << m << "x" << n << "x" << k << " warmup=" << warmup
                << " iterations=" << iterations << " avg_ms=" << avg_ms << " TOPS=" << tops << "\n";
    }

    sycl::free(dev_a, q);
    sycl::free(dev_b, q);
    sycl::free(dev_c, q);
    sycl::free(dev_scale_a, q);
    sycl::free(dev_scale_b, q);
    if (dev_bias) {
      sycl::free(dev_bias, q);
    }
  }

  template <typename T>
  static void run_perf_case(const std::string& name, int m, int n, int k, int blocksize, bool has_bias, int warmup,
                            int iterations) {
    run_case<T>(name, m, n, k, blocksize, has_bias, false, warmup, iterations);
  }
#endif
};

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)
template <>
inline BTLA_DTYPE TestSyclTlaIgemmS8S8Dequant::dtype_of<float>() {
  return BTLA_DTYPE::F32;
}

template <>
inline BTLA_DTYPE TestSyclTlaIgemmS8S8Dequant::dtype_of<sycl::half>() {
  return BTLA_DTYPE::F16;
}

template <>
inline BTLA_DTYPE TestSyclTlaIgemmS8S8Dequant::dtype_of<sycl::ext::oneapi::bfloat16>() {
  return BTLA_DTYPE::BF16;
}

template <>
inline const char* TestSyclTlaIgemmS8S8Dequant::dtype_name<float>() {
  return "fp32";
}

template <>
inline const char* TestSyclTlaIgemmS8S8Dequant::dtype_name<sycl::half>() {
  return "fp16";
}

template <>
inline const char* TestSyclTlaIgemmS8S8Dequant::dtype_name<sycl::ext::oneapi::bfloat16>() {
  return "bf16";
}
#endif