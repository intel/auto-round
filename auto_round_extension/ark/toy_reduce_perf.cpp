#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct ReduceAdd {
  template <typename T>
  static inline T eval(sycl::sub_group sg, T x) {
    return sycl::reduce_over_group(sg, x, sycl::plus<T>());
  }
  static const char* name() { return "add"; }
};

struct ReduceMax {
  template <typename T>
  static inline T eval(sycl::sub_group sg, T x) {
    return sycl::reduce_over_group(sg, x, sycl::maximum<T>());
  }
  static const char* name() { return "max"; }
};

template <typename T, typename Op, int Repeat>
class SubgroupReduceKernel;

template <typename T, typename Op, int Repeat>
sycl::event launch_subgroup_reduce_kernel(sycl::queue& q, const T* in, T* out, std::size_t n, std::size_t local_size) {
  return q.submit([&](sycl::handler& h) {
    h.parallel_for<SubgroupReduceKernel<T, Op, Repeat>>(
        sycl::nd_range<1>(sycl::range<1>(n), sycl::range<1>(local_size)), [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
          const std::size_t i = item.get_global_linear_id();
          sycl::sub_group sg = item.get_sub_group();
          // One global load, many register/subgroup reductions, one global store.
          // Keep the benchmark instruction-bound instead of memory-bound.
          T base = in[i];
          T x = base;
          T acc = T(0.0f);
#pragma unroll 16
          for (int r = 0; r < Repeat; ++r) {
            T y = x + T(float((r & 7) + 1) * 0.001f);
            T red = Op::eval(sg, y);
            acc += red * T(0.0009765625f);
            // Make the next reduction depend on the previous one, but keep values bounded.
            x = T(red * T(0.0009765625f) + base);
          }
          out[i] = acc;
        });
  });
}

double event_ms(const sycl::event& e) {
  const auto start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
  const auto end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
  return double(end - start) * 1e-6;
}

template <typename T, typename Op, int Repeat>
void run_case(sycl::queue& q, const std::string& name, std::size_t n, std::size_t local_size, int warmup, int iters) {
  T* in = sycl::malloc_shared<T>(n, q);
  T* out = sycl::malloc_shared<T>(n, q);
  if (!in || !out) throw std::runtime_error("USM allocation failed");

  for (std::size_t i = 0; i < n; ++i) {
    float x = 0.5f + float(i % 1024) / 4096.0f;
    in[i] = T(x);
    out[i] = T(0.0f);
  }
  q.wait();

  for (int i = 0; i < warmup; ++i) {
    launch_subgroup_reduce_kernel<T, Op, Repeat>(q, in, out, n, local_size).wait();
  }

  std::vector<double> times;
  times.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    auto e = launch_subgroup_reduce_kernel<T, Op, Repeat>(q, in, out, n, local_size);
    e.wait();
    times.push_back(event_ms(e));
  }

  std::sort(times.begin(), times.end());
  double median_ms = times[times.size() / 2];
  double avg_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
  double greduce_per_s = double(n) * double(Repeat) / (median_ms * 1e-3) / 1e9;

  volatile float sink = float(out[n / 2]);
  (void)sink;

  std::cout << std::fixed << std::setprecision(3) << name << " op=" << Op::name() << " n=" << n
            << " local=" << local_size << " repeat=" << Repeat << " median_ms=" << median_ms
            << " avg_ms=" << avg_ms << " Greduce/s=" << greduce_per_s << "\n";

  sycl::free(in, q);
  sycl::free(out, q);
}

template <int Repeat>
void prewarm_device(sycl::queue& q, std::size_t n, std::size_t local_size) {
  float* in = sycl::malloc_shared<float>(n, q);
  float* out = sycl::malloc_shared<float>(n, q);
  if (!in || !out) throw std::runtime_error("USM allocation failed");
  for (std::size_t i = 0; i < n; ++i) {
    in[i] = 0.5f + float(i % 1024) / 4096.0f;
    out[i] = 0.0f;
  }
  q.wait();
  for (int i = 0; i < 5; ++i) {
    launch_subgroup_reduce_kernel<float, ReduceAdd, Repeat>(q, in, out, n, local_size).wait();
  }
  sycl::free(in, q);
  sycl::free(out, q);
}

}  // namespace

int main() {
  constexpr std::size_t N = 1 << 16;
  constexpr std::size_t LocalSize = 256;
  constexpr int Repeat = 32768;
  constexpr int Warmup = 3;
  constexpr int Iters = 10;

  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
  std::cout << "device=" << q.get_device().get_info<sycl::info::device::name>() << "\n";
  std::cout << "one load + one store per work-item, subgroup_size=16, repeated subgroup reductions per load=" << Repeat
            << "\n";
  prewarm_device<Repeat>(q, N, LocalSize);
  run_case<float, ReduceAdd, Repeat>(q, "reduce_float", N, LocalSize, Warmup, Iters);
  run_case<sycl::half, ReduceAdd, Repeat>(q, "reduce_half", N, LocalSize, Warmup, Iters);
  run_case<float, ReduceMax, Repeat>(q, "reduce_float", N, LocalSize, Warmup, Iters);
  run_case<sycl::half, ReduceMax, Repeat>(q, "reduce_half", N, LocalSize, Warmup, Iters);
  return 0;
}
