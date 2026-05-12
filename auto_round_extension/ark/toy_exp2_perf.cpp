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

template <typename T>
struct Exp2Native {
  static constexpr const char* label = "native_exp2";
  static inline T eval(T x) { return sycl::native::exp2(x); }
};

struct Exp2HalfViaFloat {
  static constexpr const char* label = "native_exp2_half_via_float";
  static inline sycl::half eval(sycl::half x) { return sycl::half(sycl::native::exp2(float(x))); }
};

struct Exp2HalfVisaAsm {
  static constexpr const char* label = "visa_exp_hf_inline_asm";
  static inline sycl::half eval(sycl::half x) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
    sycl::half y;
  asm(".decl OUT_HF v_type=G type=HF num_elts=1 alias=<%0,0>\n"
    ".decl IN_HF v_type=G type=HF num_elts=1 alias=<%1,0>\n"
    "exp (M1_NM, 1) OUT_HF(0,0)<1> IN_HF(0,0)<1;1,0>"
    : "=rw"(y)
    : "rw"(x));
    return y;
#else
    return sycl::native::exp2(x);
#endif
  }
};

template <typename T>
struct Exp2PolySage {
  static constexpr const char* label = "poly_sage_1+x+x2/2";
  static inline T eval(T x) { return T(1.0f) + x + x * x * T(0.5f); }
};

template <typename T>
struct Exp2PolyTaylor {
  static constexpr const char* label = "poly_exp2_taylor_ln2";
  static inline T eval(T x) {
    T y = x * T(0.6931471805599453f);
    return T(1.0f) + y + y * y * T(0.5f);
  }
};

template <typename T, typename Op, int Repeat>
class Exp2Kernel;

template <typename T, typename Op, int Repeat>
sycl::event launch_exp2_kernel(sycl::queue& q, const T* in, T* out, std::size_t n) {
  return q.submit([&](sycl::handler& h) {
    h.parallel_for<Exp2Kernel<T, Op, Repeat>>(sycl::range<1>(n), [=](sycl::id<1> id) {
      const std::size_t i = id[0];
      // One global load, many register-only math operations, one global store.
      // Arithmetic intensity is roughly Repeat ops per sizeof(T)*2 bytes.
      T x = in[i];
      T acc = T(0.0f);
#pragma unroll 16
      for (int r = 0; r < Repeat; ++r) {
        // Keep inputs in [-1, 0) and make every operation depend on the loaded value.
        T y = x - T(0.00390625f * float(r & 15));
        acc += Op::eval(y);
        x = x + T(0.000244140625f);
        if ((r & 4095) == 4095) {
          x = in[i];
        }
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
void run_case(sycl::queue& q, const std::string& name, std::size_t n, int warmup, int iters) {
  T* in = sycl::malloc_shared<T>(n, q);
  T* out = sycl::malloc_shared<T>(n, q);
  if (!in || !out) {
    throw std::runtime_error("USM allocation failed");
  }

  for (std::size_t i = 0; i < n; ++i) {
    float x = -1.0f + float(i % 1024) / 1024.0f;  // [-1, 0)
    in[i] = T(x);
    out[i] = T(0.0f);
  }
  q.wait();

  for (int i = 0; i < warmup; ++i) {
    launch_exp2_kernel<T, Op, Repeat>(q, in, out, n).wait();
  }

  std::vector<double> times;
  times.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    auto e = launch_exp2_kernel<T, Op, Repeat>(q, in, out, n);
    e.wait();
    times.push_back(event_ms(e));
  }

  std::sort(times.begin(), times.end());
  double median_ms = times[times.size() / 2];
  double avg_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
  double gop_per_s = double(n) * double(Repeat) / (median_ms * 1e-3) / 1e9;

  // Read a value to keep outputs observable.
  volatile float sink = float(out[n / 2]);
  (void)sink;

  std::cout << std::fixed << std::setprecision(3) << name << " n=" << n << " repeat=" << Repeat
            << " median_ms=" << median_ms << " avg_ms=" << avg_ms << " Gop/s=" << gop_per_s << "\n";

  sycl::free(in, q);
  sycl::free(out, q);
}

template <int Repeat>
void prewarm_device(sycl::queue& q, std::size_t n) {
  float* in = sycl::malloc_shared<float>(n, q);
  float* out = sycl::malloc_shared<float>(n, q);
  if (!in || !out) throw std::runtime_error("USM allocation failed");
  for (std::size_t i = 0; i < n; ++i) {
    in[i] = -1.0f + float(i % 1024) / 1024.0f;
    out[i] = 0.0f;
  }
  q.wait();
  for (int i = 0; i < 5; ++i) {
    launch_exp2_kernel<float, Exp2Native<float>, Repeat>(q, in, out, n).wait();
  }
  sycl::free(in, q);
  sycl::free(out, q);
}

}  // namespace

int main() {
  constexpr std::size_t N = 1 << 16;
  constexpr int Repeat = 32768;
  constexpr int Warmup = 3;
  constexpr int Iters = 10;

  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
  std::cout << "device=" << q.get_device().get_info<sycl::info::device::name>() << "\n";
  std::cout << "one load + one store per work-item, repeated register-only ops per load=" << Repeat << "\n";
  prewarm_device<Repeat>(q, N);
  run_case<float, Exp2Native<float>, Repeat>(q, "float_native_exp2", N, Warmup, Iters);
  run_case<float, Exp2PolySage<float>, Repeat>(q, "float_poly_sage_1+x+x2/2", N, Warmup, Iters);
  run_case<float, Exp2PolyTaylor<float>, Repeat>(q, "float_poly_exp2_taylor_ln2", N, Warmup, Iters);
  run_case<sycl::half, Exp2Native<sycl::half>, Repeat>(q, "half_native_exp2", N, Warmup, Iters);
  run_case<sycl::half, Exp2HalfViaFloat, Repeat>(q, "half_native_exp2_via_float", N, Warmup, Iters);
  run_case<sycl::half, Exp2HalfVisaAsm, Repeat>(q, "half_visa_exp_hf_inline_asm", N, Warmup, Iters);
  run_case<sycl::half, Exp2PolySage<sycl::half>, Repeat>(q, "half_poly_sage_1+x+x2/2", N, Warmup, Iters);
  run_case<sycl::half, Exp2PolyTaylor<sycl::half>, Repeat>(q, "half_poly_exp2_taylor_ln2", N, Warmup, Iters);
  return 0;
}
