#include <stdio.h>
#include <sycl/detail/builtins/builtins.hpp>
#include <sycl/functional.hpp>
#include <type_traits>
#include "bestla_wrapper.h"
#include "bestla_ut.h"
#include "sycl_ut.h"
#include "sycl/sycl_wrapper.h"

#if 0
namespace c10 {
using DeviceIndex = int8_t;
}
#define ARK_XPU 1
#include "../../../dispatcher/include/dnnl_wrapper.hpp"
using namespace ark;
#endif

namespace bestla {
using namespace ut;
using namespace utils;
using namespace sycl_utils;
using namespace sycl_gemm;
namespace sycl_ut {
namespace {
template <bool IsE4M3>
inline uint8_t sanitize_fp8_byte(uint8_t raw) {
  if constexpr (IsE4M3) {
    uint8_t exp = (raw >> 3) & 0xF;
    uint8_t mant = raw & 0x7;
    if (exp == 0xF && mant == 0x7) {
      raw = (raw & 0xF8) | 0x6;
    }
  } else {
    uint8_t exp = (raw >> 2) & 0x1F;
    if (exp == 0x1F) {
      raw = (raw & 0x83) | (0x1E << 2);
    }
  }
  return raw;
}

template <bool IsE4M3>
inline void sanitize_fp8_buffer(uint8_t* data, size_t size) {
#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    data[i] = sanitize_fp8_byte<IsE4M3>(data[i]);
  }
}

template <bool IsE4M3>
inline float decode_fp8_byte(uint8_t raw) {
  const uint8_t mag = raw & 0x7F;
  const float v = IsE4M3 ? sycl_prologue_b::fp8_lut::lut_e4m3_128[mag] : sycl_prologue_b::fp8_lut::lut_e5m2_128[mag];
  return (raw & 0x80) ? -v : v;
}
}  // namespace

int constexpr TestMs = 1000;

class Benchmark_Fp32Fp32 {
 public:
  Benchmark_Fp32Fp32() {
    UT_START();
    benchmark_all(1, 2, 3, true);
    benchmark_all(1, 1024, 768, true);
    benchmark_all(500, 1024, 768, true);
    // benchmark_all(1, 4096, 4096);
    // benchmark_all(4096, 4096, 4096);
  }

  using AType = float;
  using BType = float;
  using CType = float;

  void benchmark(int m, int n, int k, int batch, AType* A, BType* B, CType* C, CType* Bias, bool verify) {
    using LOG = timer_statistics_logger<TestMs * 2>;
    LOG log;
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    float timems = TestMs;
    utils::timer<std::chrono::milliseconds> tm;
    auto A_d = A;
    auto B_d = B;
    auto C_d = C;
    auto psize = (size_t)m * n * k * 2;
    using CFG = xve::SGemmCfg;
    q->wait();
    for (size_t i = 0; i < batch; i++) {
      Launcher<CFG, xve::GemmCore>::run(
          q, {A_d + i * m * k, B_d + i * n * k, C_d + i * m * n, m, n, k, k, k, n, Bias + i * n});
    }
    if (verify) {
      return;
    }
    tm.start();
    utils::timer<std::chrono::microseconds> tm1;
    while (tm.stop() < timems) {
      tm1.start();
      for (size_t i = 0; i < batch; i++) {
        Launcher<CFG, xve::GemmCore>::run(
            q, {A_d + i * m * k, B_d + i * n * k, C_d + i * m * n, m, n, k, k, k, n, Bias + i * n});
      }
      q->wait();
      log.add(tm1.stop() / batch);
      if (tm.stop() >= timems) {
        break;
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::F32, BTLA_DTYPE::F32, BTLA_DTYPE::F32);
    double mms = double(memsize) / log.min_val / 1e6;
    printf(" %s Flops:%.3f GOPS Bandwidth:%.3f GB/s\n", log.get_log_str(), flops, mms);
  }

  void benchmark_all(int m, int n, int k, bool verify = false) {
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::F32, BTLA_DTYPE::F32, BTLA_DTYPE::F32);
    auto batch = verify ? 16 : auto_batch(memsize);
    printf("%d %d %d %d %s %s %s\n", m, n, k, batch, bestla_dtype_str(BTLA_DTYPE::F32),
           bestla_dtype_str(BTLA_DTYPE::F32), bestla_dtype_str(BTLA_DTYPE::F32));
    float testtime = float(TestMs);
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    sycl_vector<AType> dA(size_t(m) * k * batch, q);
    sycl_vector<BType> dB(size_t(n) * k * batch, q);
    sycl_vector<CType> dC(size_t(m) * n * batch, q);
    sycl_vector<CType> dBias(n * batch, q);
    avector<float> matC(m * n), ref(m * n), bias(n);
    avector<float> matA(m * k), matBT(k * n);
    if (verify) {
      fill_buffer_randn(matA.data(), matA.size(), (-0.5f), (0.5f));
      fill_buffer_randn(matBT.data(), matBT.size(), (-0.5f), (0.5f));
      fill_buffer_randn(bias.data(), bias.size(), (-0.5f), (0.5f));
      for (size_t i = 0; i < batch; i++) {
        q->memcpy(dA.data() + i * m * k, matA.data(), m * k * 4);
        q->memcpy(dB.data() + i * n * k, matBT.data(), n * k * 4);
        q->memcpy(dBias.data() + i * n, bias.data(), n * 4);
      }
      q->wait();
    }

    benchmark(m, n, k, batch, dA.data(), dB.data(), dC.data(), dBias.data(), verify);

    if (verify) {
      gemmref_fp32fp32fp32(m, n, k, matA.data(), matBT.data(), ref.data(), k, k, n, bias.data(), true);
      q->memcpy(matC.data(), dC.data(), m * n * 4).wait();
      buffer_error(ref.data(), matC.data(), matC.size(), (1.f));
    }
  }
};
#ifdef BTLA_UT_SYCL
static Benchmark_Fp32Fp32 sBenchmark_Fp32Fp32;
#endif

class Benchmark_Fp16Fp16 {
 public:
  Benchmark_Fp16Fp16() {
    UT_START();
    benchmark_all(500, 1024, 768, true);
    benchmark_all(1, 1024, 768, true);
    benchmark_all(1, 4096, 4096);
    benchmark_all(4096, 4096, 4096);
  }

  using AType = sycl::half;
  using BType = sycl::half;
  using CType = sycl::half;

  void benchmark(int m, int n, int k, int batch, AType* A, BType* B, CType* C, CType* Bias, bool verify) {
    using LOG = timer_statistics_logger<TestMs * 2>;
    LOG log;
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    float timems = TestMs;
    utils::timer<std::chrono::milliseconds> tm;
    auto A_d = A;
    auto B_d = B;
    auto C_d = C;
    auto psize = (size_t)m * n * k * 2;
    using CFG = xmx::HGemmCfg;
    q->wait();
    for (size_t i = 0; i < batch; i++) {
      Launcher<CFG, xmx::GemmCore>::run(
          q, {A_d + i * m * k, B_d + i * n * k, C_d + i * m * n, m, n, k, k, k, n, Bias + i * n});
    }
    if (verify) {
      return;
    }
    tm.start();
    utils::timer<std::chrono::microseconds> tm1;
    while (tm.stop() < timems) {
      tm1.start();
      for (size_t i = 0; i < batch; i++) {
        Launcher<CFG, xmx::GemmCore>::run(
            q, {A_d + i * m * k, B_d + i * n * k, C_d + i * m * n, m, n, k, k, k, n, Bias + i * n});
      }
      q->wait();
      log.add(tm1.stop() / batch);
      if (tm.stop() >= timems) {
        break;
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::F16, BTLA_DTYPE::F16, BTLA_DTYPE::F16);
    double mms = double(memsize) / log.min_val / 1e6;
    printf(" %s Flops:%.3f GOPS Bandwidth:%.3f GB/s\n", log.get_log_str(), flops, mms);
  }

  void benchmark_all(int m, int n, int k, bool verify = false) {
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::F16, BTLA_DTYPE::F16, BTLA_DTYPE::F16);
    auto batch = verify ? 16 : auto_batch(memsize);
    auto dev = UT_Device::get();
    printf("%s %d %d %d %d %s %s %s\n", dev->getName().c_str(), m, n, k, batch, bestla_dtype_str(BTLA_DTYPE::F16),
           bestla_dtype_str(BTLA_DTYPE::F16), bestla_dtype_str(BTLA_DTYPE::F16));
    float testtime = float(TestMs);

    auto q = dev->getQueue();

    sycl_vector<AType> dA(size_t(m) * k * batch, q);
    sycl_vector<BType> dB(size_t(n) * k * batch, q);
    sycl_vector<CType> dC(size_t(m) * n * batch, q);
    sycl_vector<CType> dBias(n * batch, q);
    avector<utils::fp16> matC(m * n), ref(m * n), bias(n);
    avector<utils::fp16> matA(m * k), matBT(k * n);
    if (verify) {
      fill_buffer_randn(matA.data(), matA.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
      fill_buffer_randn(matBT.data(), matBT.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
      fill_buffer_randn(bias.data(), bias.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
      for (size_t i = 0; i < batch; i++) {
        q->memcpy(dA.data() + i * m * k, matA.data(), m * k * 2);
        q->memcpy(dB.data() + i * n * k, matBT.data(), n * k * 2);
        q->memcpy(dBias.data() + i * n, bias.data(), n * 2);
      }
      q->wait();
    }

    benchmark(m, n, k, batch, dA.data(), dB.data(), dC.data(), dBias.data(), verify);

    if (verify) {
      gemmref_fp16fp16fp16(m, n, k, matA.data(), matBT.data(), ref.data(), k, k, n, bias.data(), true);
      q->memcpy(matC.data(), dC.data(), m * n * 2).wait();
      buffer_error(ref.data(), matC.data(), matC.size(), utils::fp16(1.f));
    }
  }
};
#ifdef BTLA_UT_SYCL
static Benchmark_Fp16Fp16 sBenchmark_Fp16Fp16;
#endif

class Benchmark_Bf16Bf16 {
 public:
  Benchmark_Bf16Bf16() {
    UT_START();
    benchmark_all(500, 1024, 768, true);
    benchmark_all(1, 1024, 768, true);
    benchmark_all(1, 4096, 4096);
    benchmark_all(4096, 4096, 4096);
  }

  using AType = sycl::ext::oneapi::bfloat16;
  using BType = sycl::ext::oneapi::bfloat16;
  using CType = sycl::ext::oneapi::bfloat16;

  void benchmark(int m, int n, int k, int batch, AType* A, BType* B, CType* C, CType* Bias, bool verify) {
    using LOG = timer_statistics_logger<TestMs * 2>;
    LOG log;
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    float timems = TestMs;
    utils::timer<std::chrono::milliseconds> tm;
    auto A_d = A;
    auto B_d = B;
    auto C_d = C;
    auto psize = (size_t)m * n * k * 2;
    using CFG = xmx::HGemmBf16Cfg;
    q->wait();
    for (size_t i = 0; i < batch; i++) {
      Launcher<CFG, xmx::GemmCore>::run(
          q, {A_d + i * m * k, B_d + i * n * k, C_d + i * m * n, m, n, k, k, k, n, Bias + i * n});
    }
    if (verify) {
      return;
    }
    tm.start();
    utils::timer<std::chrono::microseconds> tm1;
    while (tm.stop() < timems) {
      tm1.start();
      for (size_t i = 0; i < batch; i++) {
        Launcher<CFG, xmx::GemmCore>::run(
            q, {A_d + i * m * k, B_d + i * n * k, C_d + i * m * n, m, n, k, k, k, n, Bias + i * n});
      }
      q->wait();
      log.add(tm1.stop() / batch);
      if (tm.stop() >= timems) {
        break;
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::BF16, BTLA_DTYPE::BF16, BTLA_DTYPE::BF16);
    double mms = double(memsize) / log.min_val / 1e6;
    printf(" %s Flops:%.3f GOPS Bandwidth:%.3f GB/s\n", log.get_log_str(), flops, mms);
  }

  void benchmark_all(int m, int n, int k, bool verify = false) {
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::BF16, BTLA_DTYPE::BF16, BTLA_DTYPE::BF16);
    auto batch = verify ? 16 : auto_batch(memsize);
    printf("%d %d %d %d %s %s %s\n", m, n, k, batch, bestla_dtype_str(BTLA_DTYPE::BF16),
           bestla_dtype_str(BTLA_DTYPE::BF16), bestla_dtype_str(BTLA_DTYPE::BF16));
    float testtime = float(TestMs);
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    sycl_vector<AType> dA(size_t(m) * k * batch, q);
    sycl_vector<BType> dB(size_t(n) * k * batch, q);
    sycl_vector<CType> dC(size_t(m) * n * batch, q);
    sycl_vector<CType> dBias(n * batch, q);
    avector<utils::bf16> matC(m * n), ref(m * n), bias(n);
    avector<utils::bf16> matA(m * k), matBT(k * n);
    if (verify) {
      fill_buffer_randn(matA.data(), matA.size(), utils::bf16(-0.5f), utils::bf16(0.5f));
      fill_buffer_randn(matBT.data(), matBT.size(), utils::bf16(-0.5f), utils::bf16(0.5f));
      fill_buffer_randn(bias.data(), bias.size(), utils::bf16(-0.5f), utils::bf16(0.5f));
      for (size_t i = 0; i < batch; i++) {
        q->memcpy(dA.data() + i * m * k, matA.data(), m * k * 2);
        q->memcpy(dB.data() + i * n * k, matBT.data(), n * k * 2);
        q->memcpy(dBias.data() + i * n, bias.data(), n * 2);
      }
      q->wait();
    }

    benchmark(m, n, k, batch, dA.data(), dB.data(), dC.data(), dBias.data(), verify);

    if (verify) {
      gemmref_bf16bf16bf16(m, n, k, matA.data(), matBT.data(), ref.data(), k, k, n, bias.data(), true);
      q->memcpy(matC.data(), dC.data(), m * n * 2).wait();
      buffer_error(ref.data(), matC.data(), matC.size(), utils::bf16(1.f));
    }
  }
};
#ifdef BTLA_UT_SYCL
static Benchmark_Bf16Bf16 sBenchmark_Bf16Bf16;
#endif

#if 0


class Benchmark_DequantF8 {
 public:
  Benchmark_DequantF8() {
    UT_START();
    benchmark_all<true>(4096, 4096, 64);
    benchmark_all<false>(4096, 4096, 64);
  }

 private:
  template <bool IsE4M3>
  void benchmark_all(int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Benchmark FP8 %s dequant: n=%d k=%d blk=%d Device:%s\n", IsE4M3 ? "E4M3" : "E5M2", n, k, blocksize,
           dev->getName().c_str());
    int blks = k / blocksize;
    avector<uint8_t> rawB(size_t(k) * n);
    avector<float> scale(size_t(blks) * n);
    fill_buffer_randn(scale.data(), scale.size(), 0.01f, 0.03f);
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    sanitize_fp8_buffer<IsE4M3>(rawB.data(), rawB.size());

    sycl_vector<uint8_t> dB(rawB.size(), q);
    sycl_vector<float> dS(scale.size(), q);
    sycl_vector<float> dDst(size_t(n) * k, q);
    q->memcpy(dB.data(), rawB.data(), rawB.size()).wait();
    q->memcpy(dS.data(), scale.data(), scale.size() * sizeof(float)).wait();

    using LOG = timer_statistics_logger<TestMs * 2>;
    float testtime = float(TestMs);
    printf(" Layout:NT\n");
    benchmark_impl<IsE4M3, false, LOG>(n, k, blocksize, blks, dB.data(), dS.data(), dDst.data(), q, testtime);
    printf(" Layout:T\n");
    benchmark_impl<IsE4M3, true, LOG>(n, k, blocksize, blks, dB.data(), dS.data(), dDst.data(), q, testtime);
#if 0
    printf(" oneDNN reorder\n");
    benchmark_onednn<IsE4M3, LOG>(n, k, blocksize, blks, dB.data(), dS.data(), dDst.data(), q, testtime);
#endif
  }

  template <bool IsE4M3, bool IsTrans, typename LOG_T>
  void benchmark_impl(int n, int k, int blocksize, int blks, uint8_t* B, float* S, float* Out, sycl::queue* q,
                      float timems) {
    using ProB = sycl_prologue_b::WeightF8T<xve::DefaultSGemmCore, float, IsE4M3>;
    LOG_T log;
    utils::timer<std::chrono::milliseconds> tm;
    tm.start();
    while (tm.stop() < timems) {
      sycl::event ev;
      if constexpr (IsTrans) {
        ev = ProB::template dequant_T<sycl_prologue_b::KernelConfigTrans>(n, k, blocksize, {B, S, blks}, Out, q);
      } else {
        ev = ProB::template dequant<sycl_prologue_b::KernelConfigTrans>(n, k, blocksize, {B, S, blks}, Out, q);
      }
      ev.wait();
      log.add(event_helper::execute_time(ev) * 1000);
      if (tm.stop() >= timems) {
        break;
      }
    }
    log.record();
    size_t memsize = size_t(n) * k * sizeof(float) + size_t(n) * blks * sizeof(float) + size_t(n) * k;
    double band = double(memsize) / log.min_val / 1e6;
    printf(" %s MemoryBandwidth:%.3fGB/s\n", log.get_log_str(), band);
  }
#if 0
  template <bool IsE4M3, typename LOG_T>
  void benchmark_onednn(int n, int k, int blocksize, int blks, uint8_t* B, float* S, float* Out, sycl::queue* q,
                        float timems) {
    LOG_T log;
    utils::timer<std::chrono::milliseconds> tm;
    auto bt = IsE4M3 ? DnnlWrapper::dt::f8_e4m3 : DnnlWrapper::dt::f8_e5m2;
    tm.start();
    while (tm.stop() < timems) {
      utils::timer<utils::microseconds> kernel_tm;
      kernel_tm.start();
      DnnlWrapper::dequant_fp8(*q, 0, k, n, B, bt, Out, DnnlWrapper::dt::f32, S, DnnlWrapper::dt::f32, blocksize, true);
      log.add(kernel_tm.stop());
      if (tm.stop() >= timems) {
        break;
      }
    }
    log.record();
    size_t memsize = size_t(n) * k * sizeof(float) + size_t(n) * blks * sizeof(float) + size_t(n) * k;
    double band = double(memsize) / log.min_val / 1e6;
    printf(" %s [oneDNN] MemoryBandwidth:%.3fGB/s\n", log.get_log_str(), band);
  }
#endif
};
#ifdef BTLA_UT_SYCL
static Benchmark_DequantF8 sBenchmark_DequantF8;
#endif

#endif

class Benchmark_S4 {
 public:
  Benchmark_S4() {
    UT_START();
    gemv_next<float>(1, 4096, 4096, 32);
    gemv_next<float, true>(1, 4096, 4096, 32);
    gemv_next<utils::fp16>(1, 4096, 4096, 32);
    gemv_next<utils::fp16, true>(1, 4096, 4096, 32);

    dequant_next_s8<float, true>(4096, 4096, 32, 0);
    dequant_next_s8<float>(4096, 4096, 32, 0);
    dequant_next_s8<utils::fp16, true>(4096, 4096, 32, 0);
    dequant_next_s8<utils::fp16>(4096, 4096, 32, 0);
    dequant_next_s8<float, true>(4096, 4096, 32, 1);
    dequant_next_s8<float>(4096, 4096, 32, 1);
    dequant_next_s8<utils::fp16, true>(4096, 4096, 32, 1);
    dequant_next_s8<utils::fp16>(4096, 4096, 32, 1);

    dequant_next<float>(4096, 4096, 32);
    dequant_next<utils::fp16>(4096, 4096, 32);
    dequant_next<float, true>(4096, 4096, 32);
    dequant_next<utils::fp16, true>(4096, 4096, 32);
  }
  static constexpr size_t nbits = 4;

  template <typename T, bool asym = false>
  void gemv_next(int m, int n, int k, int blocksize) {
    int blks = k / blocksize;
    auto memsize = (size_t)(n * k * nbits / 8 + n * blks * sizeof(T)) + (m * k + m * n) * sizeof(T);
    if (asym) memsize += n * blks;
    auto batch = auto_batch(memsize);
    using LOG = timer_statistics_logger<TestMs * 2>;
    LOG log;
    auto dev = UT_Device::get();
    printf("Test Case %s: %d %d %d Device:%s\n", __FUNCTION__, n, k, blocksize, dev->getName().c_str());
    auto q = dev->getQueue();
    using ST = std::conditional_t<std::is_same_v<T, utils::fp16>, sycl::half, float>;
    avector<uint8_t> rawB(k * n / 2);
    avector<T> scale(blks * n), A(m * k), C(n * m);
    avector<float> Bf32(n * k), Af32(m * k), ref(n * m);
    avector<int8_t> zp(n * blks);
    fill_buffer_randn(zp.data(), zp.size(), int8_t(-8), int8_t(7));
    fill_buffer_randn(A.data(), A.size(), T(-.5f), T(0.5f));
    fill_buffer_randn(scale.data(), scale.size(), T(0.01f), T(0.03f));
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    auto srcptr = (utils::int4x2*)rawB.data();
    Af32 = A.template to<float>();
    for (int j = 0; j < n; j += 1) {
      for (int i = 0; i < k; i += 2) {
        auto tmp = srcptr[i / 2 + j * k / 2];
        auto noffset = i / blocksize + j * blks;
        if constexpr (asym) {
          Bf32[i * n + j] = static_cast<float>(static_cast<int8_t>(tmp.x) - 8 - zp[noffset]) * (float)scale[noffset];
          Bf32[(i + 1) * n + j] =
              static_cast<float>(static_cast<int8_t>(tmp.y) - 8 - zp[noffset]) * (float)scale[noffset];
        } else {
          Bf32[i * n + j] = static_cast<float>(static_cast<int8_t>(tmp.x) - 8) * (float)scale[noffset];
          Bf32[(i + 1) * n + j] = static_cast<float>(static_cast<int8_t>(tmp.y) - 8) * (float)scale[noffset];
        }
      }
    }
    gemmref_fp32fp32fp32(m, n, k, Af32.data(), Bf32.data(), ref.data(), k, n, n);
    sycl_vector<ST> dS(scale.size() * batch, q), dA(A.size() * batch, q), dC((size_t)n * k * batch, q);
    sycl_vector<uint8_t> dB(rawB.size() * batch, q);
    sycl_vector<int8_t> dZP(zp.size() * batch, q);
    for (size_t i = 0; i < batch; i++) {
      q->memcpy(dS.data() + i * scale.size(), scale.data(), scale.size() * sizeof(T));
      q->memcpy(dB.data() + i * rawB.size(), rawB.data(), rawB.size() * 1);
      q->memcpy(dA.data() + i * A.size(), A.data(), A.size() * sizeof(T));
      q->memcpy(dZP.data() + i * zp.size(), zp.data(), zp.size() * 1);
    }
    q->wait();
    utils::timer<std::chrono::milliseconds> tm;
    utils::timer<std::chrono::microseconds> tm1;
    using ProB = sycl_prologue_b::WeightS4T<ST>;
    using Cfg = std::conditional_t<std::is_same_v<T, float>, typename ProB::CfgGemvF32, typename ProB::CfgGemvF16>;
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();
    auto S_d = dS.data();
    auto psize = (size_t)m * n * k * 2;
    tm.start();
    while (tm.stop() < TestMs) {
      tm1.start();
      for (size_t i = 0; i < batch; i++) {
        ProB::template gemv<Cfg, ST>(
            A_d + i * m * k,
            {B_d + i * n * k / 2, S_d + i * n * blks, blks, nullptr, asym ? dZP.data() + i * zp.size() : nullptr},
            C_d + i * m * n, n, k, blocksize, q);
      }
      q->wait();
      log.add(tm1.stop() / (float)batch);
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;

    double band = double(memsize) / log.min_val / 1e6;
    printf(" %s Flops:%.3f MemoryBandwidth:%.3fGB/s\n", log.get_log_str(), flops, band);
    q->memcpy(C.data(), C_d, C.size() * sizeof(T)).wait();
    auto Cf32 = C.template to<float>();
    buffer_error(ref.data(), Cf32.data(), ref.size(), float(0.1f));
  }

  template <typename T, bool asym = false>
  void dequant_next(int n, int k, int blocksize) {
    int blks = updiv(k, blocksize);
    auto psize = (size_t)n * k * sizeof(T) + n * k / 2 + n * k / blocksize * sizeof(T);
    if (asym) psize += n * blks;
    size_t batch = auto_batch(psize);
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case %s: %d %d %d Device:%s\n", __FUNCTION__, n, k, blocksize, dev->getName().c_str());
    avector<uint8_t> rawB(k * n / 2);
    using ST = std::conditional_t<std::is_same_v<T, utils::fp16>, sycl::half, float>;
    avector<T> scale(blks * n), dequant(n * k), ref(n * k);
    avector<int8_t> zp(n * blks);
    fill_buffer_randn(zp.data(), zp.size(), int8_t(-8), int8_t(7));
    fill_buffer_randn(scale.data(), scale.size(), T(0.01f), T(0.03f));
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    auto srcptr = (utils::int4x2*)rawB.data();
    for (int j = 0; j < n; j += 1) {
      for (int i = 0; i < k; i += 2) {
        auto tmp = srcptr[i / 2 + j * k / 2];
        auto noffset = i / blocksize + j * blks;
        if constexpr (asym) {
          ref[i + j * k] = static_cast<float>(static_cast<int8_t>(tmp.x) - 8 - zp[noffset]) * (float)scale[noffset];
          ref[i + 1 + j * k] = static_cast<float>(static_cast<int8_t>(tmp.y) - 8 - zp[noffset]) * (float)scale[noffset];
        } else {
          ref[i + j * k] = static_cast<float>(static_cast<int8_t>(tmp.x) - 8) * (float)scale[noffset];
          ref[i + 1 + j * k] = static_cast<float>(static_cast<int8_t>(tmp.y) - 8) * (float)scale[noffset];
        }
      }
    }
    sycl_vector<ST> dS(scale.size() * batch, q), dequantB((size_t)n * k * batch, q);
    sycl_vector<uint8_t> dB(rawB.size() * batch, q);
    sycl_vector<int8_t> dZP(zp.size() * batch, q);
    for (size_t i = 0; i < batch; i++) {
      q->memcpy(dS.data() + i * scale.size(), scale.data(), scale.size() * sizeof(T)).wait();
      q->memcpy(dB.data() + i * rawB.size(), rawB.data(), rawB.size() * 1).wait();
      q->memcpy(dZP.data() + i * zp.size(), zp.data(), zp.size() * 1).wait();
    }

    auto S_d = dS.data();
    auto B_d = dB.data();
    auto DB_d = dequantB.data();
    using LOG = timer_statistics_logger<TestMs * 2>;
    LOG log;
    using ProB = sycl_prologue_b::WeightS4T<ST>;
    utils::timer<std::chrono::milliseconds> tm;
    utils::timer<std::chrono::microseconds> tm1;
    tm.start();
    while (tm.stop() < TestMs) {
      tm1.start();
      for (size_t i = 0; i < batch; i++) {
        if constexpr (sizeof(T) == 4) {
          auto ev = ProB::template dequant<typename ProB::CfgDequantF32>(
              n, k, blocksize,
              {B_d + i * rawB.size(), S_d + i * scale.size(), blks, nullptr,
               asym ? dZP.data() + i * zp.size() : nullptr},
              DB_d + i * n * k, q);
        } else {
          auto ev = ProB::template dequant<typename ProB::CfgDequantF16>(
              n, k, blocksize,
              {B_d + i * rawB.size(), S_d + i * scale.size(), blks, nullptr,
               asym ? dZP.data() + i * zp.size() : nullptr},
              DB_d + i * n * k, q);
        }
      }
      q->wait();

      log.add(tm1.stop() / (float)batch);
      if (tm.stop() >= TestMs) {
        break;
      }
    }

    q->memcpy(dequant.data(), DB_d, dequant.size() * sizeof(T)).wait();
    buffer_error(ref.data(), dequant.data(), dequant.size(), T(0.1f));
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    printf(" %s Memory Bandwidth:%.3f\n", log.get_log_str(), flops);
  }

  template <typename T, bool asym = false>
  void dequant_next_s8(int n, int k, int blocksize, bool rescale) {
    int blks = updiv(k, blocksize);
    auto psize = (size_t)n * k + n * k / 2;
    int newblocksize = 256;
    int newblks = k / newblocksize;
    if (rescale) psize += n * blks * sizeof(T) + n * newblks * sizeof(T);
    if (asym) psize += n * blks;
    size_t batch = auto_batch(psize);
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case %s: %d %d %d, batch %d Device:%s\n", __FUNCTION__, n, k, blocksize, batch,
           dev->getName().c_str());
    avector<uint8_t> rawB(k * n / 2);
    using ST = std::conditional_t<std::is_same_v<T, utils::fp16>, sycl::half, float>;
    avector<T> scale(blks * n), scaleN(n * newblks);
    avector<float> dqs4(n * k), dqs8(n * k);
    avector<int8_t> dequant(n * k), ref(n * k);
    avector<int8_t> zp(n * blks);
    fill_buffer_randn(scale.data(), scale.size(), T(0.01f), T(0.03f));
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(zp.data(), zp.size(), int8_t(-8), int8_t(7));
    auto srcptr = (utils::int4x2*)rawB.data();
    for (int j = 0; j < n; j += 1) {
      for (size_t i = 0; i < newblks; i++) {
        float maxv = 0.f;
        int start_blk = i * newblocksize / blocksize;
        int end_blk = (i + 1) * newblocksize / blocksize;
        for (int ii = start_blk; ii < end_blk; ii += 1) {
          maxv = std::max((float)scale[ii + j * blks], maxv);
        }
        scaleN[j * newblks + i] = asym ? maxv * 16.f / 127.f : maxv * 8.f / 127.f;
      }
    }
#pragma omp parallel for
    for (int j = 0; j < n; j += 1) {
      for (int i = 0; i < k; i += 2) {
        auto tmp = srcptr[i / 2 + j * k / 2];
        auto noffset = i / blocksize + j * blks;
        float sn = (float)scaleN[j * newblks + i / newblocksize];
        float _scale = (float)scale[noffset] / sn;
        if constexpr (asym) {
          dqs4[i + j * k] = (static_cast<int8_t>(tmp.x) - 8 - zp[noffset]) * (float)scale[noffset];
          dqs4[i + 1 + j * k] = (static_cast<int8_t>(tmp.y) - 8 - zp[noffset]) * (float)scale[noffset];
        } else {
          dqs4[i + j * k] = (static_cast<int8_t>(tmp.x) - 8) * (float)scale[noffset];
          dqs4[i + 1 + j * k] = (static_cast<int8_t>(tmp.y) - 8) * (float)scale[noffset];
        }
        if (rescale) {
          if constexpr (asym) {
            ref[i + j * k] = std::round(static_cast<float>(static_cast<int8_t>(tmp.x) - 8 - zp[noffset]) * _scale);
            ref[i + 1 + j * k] = std::round(static_cast<float>(static_cast<int8_t>(tmp.y) - 8 - zp[noffset]) * _scale);
          } else {
            ref[i + j * k] = std::round(static_cast<float>(static_cast<int8_t>(tmp.x) - 8) * _scale);
            ref[i + 1 + j * k] = std::round(static_cast<float>(static_cast<int8_t>(tmp.y) - 8) * _scale);
          }
          dqs8[i + j * k] = ref[i + j * k] * sn;
          dqs8[i + 1 + j * k] = ref[i + 1 + j * k] * sn;
        } else {
          ref[i + j * k] = static_cast<int8_t>(tmp.x) - 8;
          ref[i + 1 + j * k] = static_cast<int8_t>(tmp.y) - 8;
          if constexpr (asym) {
            ref[i + j * k] -= zp[noffset];
            ref[i + 1 + j * k] -= zp[noffset];
          }
        }
      }
    }
    if (rescale) buffer_error(dqs4.data(), dqs8.data(), dqs8.size(), float(1));
    sycl_vector<ST> dS(scale.size() * batch, q), dSInv(scaleN.size() * batch, q);
    sycl_vector<uint8_t> dB(rawB.size() * batch, q);
    sycl_vector<int8_t> dequantB((size_t)n * k * batch, q), dZP(zp.size() * batch, q);
    for (size_t i = 0; i < batch; i++) {
      q->memcpy(dS.data() + i * scale.size(), scale.data(), scale.size() * sizeof(T));
      q->memcpy(dSInv.data() + i * scaleN.size(), scaleN.data(), scaleN.size() * sizeof(T));
      q->memcpy(dB.data() + i * rawB.size(), rawB.data(), rawB.size() * 1);
      q->memcpy(dZP.data() + i * zp.size(), zp.data(), zp.size() * 1);
    }
    q->wait();
    auto S_d = dS.data();
    auto SInv_d = rescale ? dSInv.data() : nullptr;
    auto B_d = dB.data();
    auto DB_d = dequantB.data();
    using LOG = timer_statistics_logger<TestMs * 2>;
    LOG log;
    using ProB = sycl_prologue_b::WeightS4T<ST>;
    utils::timer<std::chrono::milliseconds> tm;
    utils::timer<std::chrono::microseconds> tm1;
    tm.start();
    while (tm.stop() < TestMs) {
      tm1.start();
      for (size_t i = 0; i < batch; i++) {
        if (rescale) {
          auto ev = ProB::template dequantS8<typename ProB::CfgDequantS8Rescale>(
              n, k, blocksize,
              {B_d + i * rawB.size(), S_d + i * scale.size(), blks, nullptr,
               asym ? dZP.data() + i * zp.size() : nullptr, SInv_d ? SInv_d + i * n * newblks : nullptr},
              DB_d + i * n * k, q, newblocksize);
        } else {
          auto ev = ProB::template dequantS8<typename ProB::CfgDequantS8>(
              n, k, blocksize,
              {B_d + i * rawB.size(), nullptr, blks, nullptr, asym ? dZP.data() + i * zp.size() : nullptr, nullptr},
              DB_d + i * n * k, q);
        }
      }
      q->wait();

      log.add(tm1.stop() / (float)batch);
      if (tm.stop() >= TestMs) {
        break;
      }
    }

    q->memcpy(dequant.data(), DB_d, dequant.size()).wait();
    buffer_error(ref.data(), dequant.data(), dequant.size(), int8_t(1));  // Why?
    log.record();

    double flops = double(psize) / log.min_val / 1e6;
    printf(" %s Memory Bandwidth:%.3f\n", log.get_log_str(), flops);
  }
};
#ifdef BTLA_UT_SYCL
static Benchmark_S4 sBenchmark_S4;
#endif

class Benchmark_S2 {
 public:
  Benchmark_S2() {
    UT_START();
    gemv_next<float>(1, 4096, 4096, 32);
    gemv_next<float, true>(1, 4096, 4096, 32);
    gemv_next<utils::fp16>(1, 4096, 4096, 32);
    gemv_next<utils::fp16, true>(1, 4096, 4096, 32);

    dequant_next_s8<float, true>(4096, 4096, 32, 0);
    dequant_next_s8<float>(4096, 4096, 32, 0);
    dequant_next_s8<utils::fp16, true>(4096, 4096, 32, 0);
    dequant_next_s8<utils::fp16>(4096, 4096, 32, 0);
    dequant_next_s8<float, true>(4096, 4096, 32, 1);
    dequant_next_s8<float>(4096, 4096, 32, 1);
    dequant_next_s8<utils::fp16, true>(4096, 4096, 32, 1);
    dequant_next_s8<utils::fp16>(4096, 4096, 32, 1);

    dequant_next<float>(4096, 4096, 32);
    dequant_next<utils::fp16>(4096, 4096, 32);
    dequant_next<float, true>(4096, 4096, 32);
    dequant_next<utils::fp16, true>(4096, 4096, 32);
  }
  static constexpr size_t nbits = 2;

  template <typename T, bool asym = false>
  void gemv_next(int m, int n, int k, int blocksize) {
    int blks = k / blocksize;
    auto memsize = (size_t)(n * k * nbits / 8 + n * blks * sizeof(T)) + (m * k + m * n) * sizeof(T);
    if (asym) memsize += n * blks;
    auto batch = auto_batch(memsize);
    using LOG = timer_statistics_logger<TestMs * 2>;
    LOG log;
    auto dev = UT_Device::get();
    printf("Test Case %s: %d %d %d Device:%s\n", __FUNCTION__, n, k, blocksize, dev->getName().c_str());
    auto q = dev->getQueue();
    using ST = std::conditional_t<std::is_same_v<T, utils::fp16>, sycl::half, float>;
    avector<uint8_t> rawB(k * n / 4);
    avector<T> scale(blks * n), A(m * k), C(n * m);
    avector<float> Bf32(n * k), Af32(m * k), ref(n * m);
    avector<int8_t> zp(n * blks);
    fill_buffer_randn(zp.data(), zp.size(), int8_t(-2), int8_t(1));
    fill_buffer_randn(A.data(), A.size(), T(-.5f), T(0.5f));
    fill_buffer_randn(scale.data(), scale.size(), T(0.01f), T(0.03f));
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    auto srcptr = (utils::bit2x4*)rawB.data();
    Af32 = A.template to<float>();
    for (int j = 0; j < n; j += 1) {
      for (int i = 0; i < k; i += 4) {
        auto tmp = srcptr[i / 4 + j * k / 4];
        auto noffset = i / blocksize + j * blks;
        int8_t q0 = static_cast<int8_t>(tmp.a) - 2;
        int8_t q1 = static_cast<int8_t>(tmp.b) - 2;
        int8_t q2 = static_cast<int8_t>(tmp.c) - 2;
        int8_t q3 = static_cast<int8_t>(tmp.d) - 2;
        if constexpr (asym) {
          Bf32[(i + 0) * n + j] = static_cast<float>(q0 - zp[noffset]) * (float)scale[noffset];
          Bf32[(i + 1) * n + j] = static_cast<float>(q1 - zp[noffset]) * (float)scale[noffset];
          Bf32[(i + 2) * n + j] = static_cast<float>(q2 - zp[noffset]) * (float)scale[noffset];
          Bf32[(i + 3) * n + j] = static_cast<float>(q3 - zp[noffset]) * (float)scale[noffset];
        } else {
          Bf32[(i + 0) * n + j] = static_cast<float>(q0) * (float)scale[noffset];
          Bf32[(i + 1) * n + j] = static_cast<float>(q1) * (float)scale[noffset];
          Bf32[(i + 2) * n + j] = static_cast<float>(q2) * (float)scale[noffset];
          Bf32[(i + 3) * n + j] = static_cast<float>(q3) * (float)scale[noffset];
        }
      }
    }
    gemmref_fp32fp32fp32(m, n, k, Af32.data(), Bf32.data(), ref.data(), k, n, n);
    sycl_vector<ST> dS(scale.size() * batch, q), dA(A.size() * batch, q), dC((size_t)n * k * batch, q);
    sycl_vector<uint8_t> dB(rawB.size() * batch, q);
    sycl_vector<int8_t> dZP(zp.size() * batch, q);
    for (size_t i = 0; i < batch; i++) {
      q->memcpy(dS.data() + i * scale.size(), scale.data(), scale.size() * sizeof(T));
      q->memcpy(dB.data() + i * rawB.size(), rawB.data(), rawB.size() * 1);
      q->memcpy(dA.data() + i * A.size(), A.data(), A.size() * sizeof(T));
      q->memcpy(dZP.data() + i * zp.size(), zp.data(), zp.size() * 1);
    }
    q->wait();
    utils::timer<std::chrono::milliseconds> tm;
    utils::timer<std::chrono::microseconds> tm1;
    using ProB = sycl_prologue_b::WeightS2T<ST>;
    using Cfg = std::conditional_t<std::is_same_v<T, float>, typename ProB::CfgGemvF32, typename ProB::CfgGemvF16>;
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();
    auto S_d = dS.data();
    auto psize = (size_t)m * n * k * 2;
    tm.start();
    while (tm.stop() < TestMs) {
      tm1.start();
      for (size_t i = 0; i < batch; i++) {
        ProB::template gemv<Cfg, ST>(
            A_d + i * m * k,
            {B_d + i * n * k / 4, S_d + i * n * blks, blks, nullptr, asym ? dZP.data() + i * zp.size() : nullptr},
            C_d + i * m * n, n, k, blocksize, q);
      }
      q->wait();
      log.add(tm1.stop() / (float)batch);
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;

    double band = double(memsize) / log.min_val / 1e6;
    printf(" %s Flops:%.3f MemoryBandwidth:%.3fGB/s\n", log.get_log_str(), flops, band);
    q->memcpy(C.data(), C_d, C.size() * sizeof(T)).wait();
    auto Cf32 = C.template to<float>();
    buffer_error(ref.data(), Cf32.data(), ref.size(), float(0.1f));
  }

  template <typename T, bool asym = false>
  void dequant_next(int n, int k, int blocksize) {
    int blks = updiv(k, blocksize);
    auto psize = (size_t)n * k * sizeof(T) + n * k / 4 + n * k / blocksize * sizeof(T);
    if (asym) psize += n * blks;
    size_t batch = auto_batch(psize);
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case %s: %d %d %d Device:%s\n", __FUNCTION__, n, k, blocksize, dev->getName().c_str());
    avector<uint8_t> rawB(k * n / 4);
    using ST = std::conditional_t<std::is_same_v<T, utils::fp16>, sycl::half, float>;
    avector<T> scale(blks * n), dequant(n * k), ref(n * k);
    avector<int8_t> zp(n * blks);
    fill_buffer_randn(zp.data(), zp.size(), int8_t(-2), int8_t(1));
    fill_buffer_randn(scale.data(), scale.size(), T(0.01f), T(0.03f));
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    auto srcptr = (utils::bit2x4*)rawB.data();
    for (int j = 0; j < n; j += 1) {
      for (int i = 0; i < k; i += 4) {
        auto tmp = srcptr[i / 4 + j * k / 4];
        auto noffset = i / blocksize + j * blks;
        int8_t q0 = static_cast<int8_t>(tmp.a) - 2;
        int8_t q1 = static_cast<int8_t>(tmp.b) - 2;
        int8_t q2 = static_cast<int8_t>(tmp.c) - 2;
        int8_t q3 = static_cast<int8_t>(tmp.d) - 2;
        if constexpr (asym) {
          ref[i + 0 + j * k] = static_cast<float>(q0 - zp[noffset]) * (float)scale[noffset];
          ref[i + 1 + j * k] = static_cast<float>(q1 - zp[noffset]) * (float)scale[noffset];
          ref[i + 2 + j * k] = static_cast<float>(q2 - zp[noffset]) * (float)scale[noffset];
          ref[i + 3 + j * k] = static_cast<float>(q3 - zp[noffset]) * (float)scale[noffset];
        } else {
          ref[i + 0 + j * k] = static_cast<float>(q0) * (float)scale[noffset];
          ref[i + 1 + j * k] = static_cast<float>(q1) * (float)scale[noffset];
          ref[i + 2 + j * k] = static_cast<float>(q2) * (float)scale[noffset];
          ref[i + 3 + j * k] = static_cast<float>(q3) * (float)scale[noffset];
        }
      }
    }
    sycl_vector<ST> dS(scale.size() * batch, q), dequantB((size_t)n * k * batch, q);
    sycl_vector<uint8_t> dB(rawB.size() * batch, q);
    sycl_vector<int8_t> dZP(zp.size() * batch, q);
    for (size_t i = 0; i < batch; i++) {
      q->memcpy(dS.data() + i * scale.size(), scale.data(), scale.size() * sizeof(T)).wait();
      q->memcpy(dB.data() + i * rawB.size(), rawB.data(), rawB.size() * 1).wait();
      q->memcpy(dZP.data() + i * zp.size(), zp.data(), zp.size() * 1).wait();
    }

    auto S_d = dS.data();
    auto B_d = dB.data();
    auto DB_d = dequantB.data();
    using LOG = timer_statistics_logger<TestMs * 2>;
    LOG log;
    using ProB = sycl_prologue_b::WeightS2T<ST>;
    utils::timer<std::chrono::milliseconds> tm;
    utils::timer<std::chrono::microseconds> tm1;
    tm.start();
    while (tm.stop() < TestMs) {
      tm1.start();
      for (size_t i = 0; i < batch; i++) {
        if constexpr (sizeof(T) == 4) {
          auto ev = ProB::template dequant<typename ProB::CfgDequantF32>(
              n, k, blocksize,
              {B_d + i * rawB.size(), S_d + i * scale.size(), blks, nullptr,
               asym ? dZP.data() + i * zp.size() : nullptr},
              DB_d + i * n * k, q);
        } else {
          auto ev = ProB::template dequant<typename ProB::CfgDequantF16>(
              n, k, blocksize,
              {B_d + i * rawB.size(), S_d + i * scale.size(), blks, nullptr,
               asym ? dZP.data() + i * zp.size() : nullptr},
              DB_d + i * n * k, q);
        }
      }
      q->wait();

      log.add(tm1.stop() / (float)batch);
      if (tm.stop() >= TestMs) {
        break;
      }
    }

    q->memcpy(dequant.data(), DB_d, dequant.size() * sizeof(T)).wait();
    buffer_error(ref.data(), dequant.data(), dequant.size(), T(0.1f));
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    printf(" %s Memory Bandwidth:%.3f\n", log.get_log_str(), flops);
  }

  template <typename T, bool asym = false>
  void dequant_next_s8(int n, int k, int blocksize, bool rescale) {
    int blks = updiv(k, blocksize);
    auto psize = (size_t)n * k + n * k / 4;
    int newblocksize = 256;
    int newblks = k / newblocksize;
    if (rescale) psize += n * blks * sizeof(T) + n * newblks * sizeof(T);
    if (asym) psize += n * blks;
    size_t batch = auto_batch(psize);
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case %s: %d %d %d, batch %d Device:%s\n", __FUNCTION__, n, k, blocksize, batch,
           dev->getName().c_str());
    avector<uint8_t> rawB(k * n / 4);
    using ST = std::conditional_t<std::is_same_v<T, utils::fp16>, sycl::half, float>;
    avector<T> scale(blks * n), scaleN(n * newblks);
    avector<float> dqs2(n * k), dqs8(n * k);
    avector<int8_t> dequant(n * k), ref(n * k);
    avector<int8_t> zp(n * blks);
    fill_buffer_randn(scale.data(), scale.size(), T(0.01f), T(0.03f));
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(zp.data(), zp.size(), int8_t(-2), int8_t(1));
    auto srcptr = (utils::bit2x4*)rawB.data();
    for (int j = 0; j < n; j += 1) {
      for (size_t i = 0; i < newblks; i++) {
        float maxv = 0.f;
        int start_blk = i * newblocksize / blocksize;
        int end_blk = (i + 1) * newblocksize / blocksize;
        for (int ii = start_blk; ii < end_blk; ii += 1) {
          maxv = std::max((float)scale[ii + j * blks], maxv);
        }
        scaleN[j * newblks + i] = asym ? maxv * 4.f / 127.f : maxv * 2.f / 127.f;
      }
    }
#pragma omp parallel for
    for (int j = 0; j < n; j += 1) {
      for (int i = 0; i < k; i += 4) {
        auto tmp = srcptr[i / 4 + j * k / 4];
        auto noffset = i / blocksize + j * blks;
        float sn = (float)scaleN[j * newblks + i / newblocksize];
        float _scale = (float)scale[noffset] / sn;
        int8_t q0 = static_cast<int8_t>(tmp.a) - 2;
        int8_t q1 = static_cast<int8_t>(tmp.b) - 2;
        int8_t q2 = static_cast<int8_t>(tmp.c) - 2;
        int8_t q3 = static_cast<int8_t>(tmp.d) - 2;
        if constexpr (asym) {
          dqs2[i + 0 + j * k] = (q0 - zp[noffset]) * (float)scale[noffset];
          dqs2[i + 1 + j * k] = (q1 - zp[noffset]) * (float)scale[noffset];
          dqs2[i + 2 + j * k] = (q2 - zp[noffset]) * (float)scale[noffset];
          dqs2[i + 3 + j * k] = (q3 - zp[noffset]) * (float)scale[noffset];
        } else {
          dqs2[i + 0 + j * k] = q0 * (float)scale[noffset];
          dqs2[i + 1 + j * k] = q1 * (float)scale[noffset];
          dqs2[i + 2 + j * k] = q2 * (float)scale[noffset];
          dqs2[i + 3 + j * k] = q3 * (float)scale[noffset];
        }
        if (rescale) {
          if constexpr (asym) {
            ref[i + 0 + j * k] = std::round(static_cast<float>(q0 - zp[noffset]) * _scale);
            ref[i + 1 + j * k] = std::round(static_cast<float>(q1 - zp[noffset]) * _scale);
            ref[i + 2 + j * k] = std::round(static_cast<float>(q2 - zp[noffset]) * _scale);
            ref[i + 3 + j * k] = std::round(static_cast<float>(q3 - zp[noffset]) * _scale);
          } else {
            ref[i + 0 + j * k] = std::round(static_cast<float>(q0) * _scale);
            ref[i + 1 + j * k] = std::round(static_cast<float>(q1) * _scale);
            ref[i + 2 + j * k] = std::round(static_cast<float>(q2) * _scale);
            ref[i + 3 + j * k] = std::round(static_cast<float>(q3) * _scale);
          }
          dqs8[i + 0 + j * k] = ref[i + 0 + j * k] * sn;
          dqs8[i + 1 + j * k] = ref[i + 1 + j * k] * sn;
          dqs8[i + 2 + j * k] = ref[i + 2 + j * k] * sn;
          dqs8[i + 3 + j * k] = ref[i + 3 + j * k] * sn;
        } else {
          ref[i + 0 + j * k] = q0;
          ref[i + 1 + j * k] = q1;
          ref[i + 2 + j * k] = q2;
          ref[i + 3 + j * k] = q3;
          if constexpr (asym) {
            ref[i + 0 + j * k] -= zp[noffset];
            ref[i + 1 + j * k] -= zp[noffset];
            ref[i + 2 + j * k] -= zp[noffset];
            ref[i + 3 + j * k] -= zp[noffset];
          }
        }
      }
    }
    if (rescale) buffer_error(dqs2.data(), dqs8.data(), dqs8.size(), float(1));
    sycl_vector<ST> dS(scale.size() * batch, q), dSInv(scaleN.size() * batch, q);
    sycl_vector<uint8_t> dB(rawB.size() * batch, q);
    sycl_vector<int8_t> dequantB((size_t)n * k * batch, q), dZP(zp.size() * batch, q);
    for (size_t i = 0; i < batch; i++) {
      q->memcpy(dS.data() + i * scale.size(), scale.data(), scale.size() * sizeof(T));
      q->memcpy(dSInv.data() + i * scaleN.size(), scaleN.data(), scaleN.size() * sizeof(T));
      q->memcpy(dB.data() + i * rawB.size(), rawB.data(), rawB.size() * 1);
      q->memcpy(dZP.data() + i * zp.size(), zp.data(), zp.size() * 1);
    }
    q->wait();
    auto S_d = dS.data();
    auto SInv_d = rescale ? dSInv.data() : nullptr;
    auto B_d = dB.data();
    auto DB_d = dequantB.data();
    using LOG = timer_statistics_logger<TestMs * 2>;
    LOG log;
    using ProB = sycl_prologue_b::WeightS2T<ST>;
    utils::timer<std::chrono::milliseconds> tm;
    utils::timer<std::chrono::microseconds> tm1;
    tm.start();
    while (tm.stop() < TestMs) {
      tm1.start();
      for (size_t i = 0; i < batch; i++) {
        if (rescale) {
          auto ev = ProB::template dequantS8<typename ProB::CfgDequantS8Rescale>(
              n, k, blocksize,
              {B_d + i * rawB.size(), S_d + i * scale.size(), blks, nullptr,
               asym ? dZP.data() + i * zp.size() : nullptr, SInv_d ? SInv_d + i * n * newblks : nullptr},
              DB_d + i * n * k, q, newblocksize);
        } else {
          auto ev = ProB::template dequantS8<typename ProB::CfgDequantS8>(
              n, k, blocksize,
              {B_d + i * rawB.size(), nullptr, blks, nullptr, asym ? dZP.data() + i * zp.size() : nullptr, nullptr},
              DB_d + i * n * k, q);
        }
      }
      q->wait();

      log.add(tm1.stop() / (float)batch);
      if (tm.stop() >= TestMs) {
        break;
      }
    }

    q->memcpy(dequant.data(), DB_d, dequant.size()).wait();
    buffer_error(ref.data(), dequant.data(), dequant.size(), int8_t(1));
    log.record();

    double flops = double(psize) / log.min_val / 1e6;
    printf(" %s Memory Bandwidth:%.3f\n", log.get_log_str(), flops);
  }
};
#ifdef BTLA_UT_SYCL
#endif
static Benchmark_S2 sBenchmark_S2;

class Benchmark_S8 {
 public:
  Benchmark_S8() {
    UT_START();
    gemv_next<float, true>(1, 4096, 4096, 32);
    gemv_next<float>(1, 4096, 4096, 32);
    // gemv_next<float>(1, 4096, 768, 32);
    gemv_next<utils::fp16, true>(1, 4096, 4096, 32);
    gemv_next<utils::fp16>(1, 4096, 4096, 32);
    // gemv_next<utils::fp16>(1, 4096, 768, 32);

    dequant_next<float, true>(4096, 4096, 32);
    dequant_next<float, false>(4096, 4096, 32);
    // dequant_next<float>(4096, 768, 32);
    dequant_next<utils::fp16, true>(4096, 4096, 32);
    dequant_next<utils::fp16, false>(4096, 4096, 32);
    // dequant_next<utils::fp16>(4096, 768, 32);
  }
  static constexpr size_t nbits = 8;

  template <typename T, bool asym = false>
  void gemv_next(int m, int n, int k, int blocksize) {
    int blks = k / blocksize;
    auto memsize = (size_t)(n * k * nbits / 8 + n * blks * sizeof(T)) + (m * k + m * n) * sizeof(T);
    auto batch = auto_batch(memsize);
    using LOG = timer_statistics_logger<TestMs * 2>;
    LOG log;
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    using ST = std::conditional_t<std::is_same_v<T, utils::fp16>, sycl::half, float>;
    avector<int8_t> rawB(k * n);
    avector<T> scale(blks * n), A(m * k), C(n * m);
    avector<float> Bf32(n * k), Af32(m * k), ref(n * m);
    avector<int8_t> zp(n * blks);
    fill_buffer_randn(zp.data(), zp.size(), int8_t(-8), int8_t(7));
    fill_buffer_randn(A.data(), A.size(), T(-.5f), T(0.5f));
    fill_buffer_randn(scale.data(), scale.size(), T(0.01f), T(0.03f));
    fill_buffer_randn(rawB.data(), rawB.size(), int8_t(-128), int8_t(127));
    auto srcptr = (int8_t*)rawB.data();
    Af32 = A.template to<float>();

    for (int j = 0; j < n; j += 1) {
      for (int i = 0; i < k; i += 2) {
        auto tmp = srcptr[i + j * k];
        auto tmp1 = srcptr[i + 1 + j * k];
        auto noffset = i / blocksize + j * blks;
        if constexpr (asym) {
          Bf32[i * n + j] = static_cast<float>((int)tmp - zp[noffset]) * (float)scale[noffset];
          Bf32[(i + 1) * n + j] = static_cast<float>((int)tmp1 - zp[noffset]) * (float)scale[noffset];
        } else {
          Bf32[i * n + j] = static_cast<float>(tmp) * (float)scale[noffset];
          Bf32[(i + 1) * n + j] = static_cast<float>(tmp1) * (float)scale[noffset];
        }
      }
    }
    gemmref_fp32fp32fp32(m, n, k, Af32.data(), Bf32.data(), ref.data(), k, n, n);
    sycl_vector<ST> dS(scale.size() * batch, q), dA(A.size() * batch, q), dC((size_t)n * k * batch, q);
    sycl_vector<int8_t> dB(rawB.size() * batch, q);
    sycl_vector<int8_t> dZP(zp.size() * batch, q);
    for (size_t i = 0; i < batch; i++) {
      q->memcpy(dS.data() + i * scale.size(), scale.data(), scale.size() * sizeof(T));
      q->memcpy(dZP.data() + i * zp.size(), zp.data(), zp.size() * 1);
      q->memcpy(dB.data() + i * rawB.size(), rawB.data(), rawB.size() * 1);
      q->memcpy(dA.data() + i * A.size(), A.data(), A.size() * sizeof(T));
    }
    q->wait();
    utils::timer<std::chrono::milliseconds> tm;
    utils::timer<std::chrono::microseconds> tm1;
    using ProB = sycl_prologue_b::WeightS8T<ST>;
    using Cfg = std::conditional_t<std::is_same_v<T, float>, typename ProB::CfgGemvF32, typename ProB::CfgGemvF16>;
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();
    auto S_d = dS.data();
    auto psize = (size_t)m * n * k * 2;
    tm.start();
    while (tm.stop() < TestMs) {
      tm1.start();
      for (size_t i = 0; i < batch; i++) {
        auto ev = ProB::template gemv<Cfg, ST>(
            A_d + i * m * k,
            {B_d + i * n * k, S_d + i * n * blks, blks, nullptr, asym ? dZP.data() + i * zp.size() : nullptr},
            C_d + i * m * n, n, k, blocksize, q);
        // ev.wait();
        // log.add(event_helper::execute_time(ev) * 1000);
      }
      q->wait();
      log.add(tm1.stop() / (float)batch);
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;

    double band = double(memsize) / log.min_val / 1e6;
    printf(" %s Flops:%.3f MemoryBandwidth:%.3fGB/s\n", log.get_log_str(), flops, band);
    q->memcpy(C.data(), C_d, C.size() * sizeof(T)).wait();
    auto Cf32 = C.template to<float>();
    buffer_error(ref.data(), Cf32.data(), ref.size(), float(0.3f));
  }

  template <typename T, bool asym = false>
  void dequant_next(int n, int k, int blocksize) {
    int blks = updiv(k, blocksize);
    auto psize = (size_t)n * k * sizeof(T) + n * k * nbits / 8 + n * k / blocksize * sizeof(T);
    size_t batch = auto_batch(psize);
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case %s: %d %d %d Device:%s\n", __FUNCTION__, n, k, blocksize, dev->getName().c_str());
    avector<int8_t> rawB(k * n);
    using ST = std::conditional_t<std::is_same_v<T, utils::fp16>, sycl::half, float>;
    avector<T> scale(blks * n), dequant(n * k), ref(n * k);
    avector<int8_t> zp(n * blks);
    fill_buffer_randn(zp.data(), zp.size(), int8_t(-8), int8_t(7));
    fill_buffer_randn(scale.data(), scale.size(), T(0.01f), T(0.03f));
    fill_buffer_randn(rawB.data(), rawB.size(), int8_t(-128), int8_t(127));
    auto srcptr = (int8_t*)rawB.data();
    for (int j = 0; j < n; j += 1) {
      for (int i = 0; i < k; i += 2) {
        auto tmp = srcptr[i + j * k];
        auto tmp1 = srcptr[i + 1 + j * k];
        auto noffset = i / blocksize + j * blks;
        if constexpr (asym) {
          ref[i + j * k] = static_cast<float>((int)tmp - zp[noffset]) * (float)scale[noffset];
          ref[i + 1 + j * k] = static_cast<float>((int)tmp1 - zp[noffset]) * (float)scale[noffset];
        } else {
          ref[i + j * k] = static_cast<float>(tmp) * (float)scale[noffset];
          ref[i + 1 + j * k] = static_cast<float>(tmp1) * (float)scale[noffset];
        }
      }
    }

    sycl_vector<ST> dS(scale.size() * batch, q), dequantB((size_t)n * k * batch, q);
    sycl_vector<int8_t> dB(rawB.size() * batch, q);
    sycl_vector<int8_t> dZP(zp.size() * batch, q);
    for (size_t i = 0; i < batch; i++) {
      q->memcpy(dZP.data() + i * zp.size(), zp.data(), zp.size() * 1).wait();
      q->memcpy(dS.data() + i * scale.size(), scale.data(), scale.size() * sizeof(T)).wait();
      q->memcpy(dB.data() + i * rawB.size(), rawB.data(), rawB.size() * 1).wait();
    }

    auto S_d = dS.data();
    auto B_d = dB.data();
    auto DB_d = dequantB.data();
    using LOG = timer_statistics_logger<TestMs * 2>;
    LOG log;
    using ProB = sycl_prologue_b::WeightS8T<ST>;
    utils::timer<std::chrono::milliseconds> tm;
    utils::timer<std::chrono::microseconds> tm1;
    tm.start();
    while (tm.stop() < TestMs) {
      tm1.start();
      for (size_t i = 0; i < batch; i++) {
        if constexpr (sizeof(T) == 4) {
          auto ev = ProB::template dequant<typename ProB::CfgDequantF32>(
              n, k, blocksize,
              {B_d + i * rawB.size(), S_d + i * scale.size(), blks, nullptr,
               asym ? dZP.data() + i * zp.size() : nullptr},
              DB_d + i * n * k, q);
        } else {
          auto ev = ProB::template dequant<typename ProB::CfgDequantF16>(
              n, k, blocksize,
              {B_d + i * rawB.size(), S_d + i * scale.size(), blks, nullptr,
               asym ? dZP.data() + i * zp.size() : nullptr},
              DB_d + i * n * k, q);
        }
      }
      q->wait();

      log.add(tm1.stop() / (float)batch);
      if (tm.stop() >= TestMs) {
        break;
      }
    }

    q->memcpy(dequant.data(), DB_d, dequant.size() * sizeof(T)).wait();
    buffer_error(ref.data(), dequant.data(), dequant.size(), T(0.1f));
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    printf(" %s Memory Bandwidth:%.3f\n", log.get_log_str(), flops);
  }
};
#ifdef BTLA_UT_SYCL
static Benchmark_S8 sBenchmark_S8;
#endif

class Benchmark_F8 {
 public:
  Benchmark_F8() {
    UT_START();
    gemv_next<float>(1, 4096, 4096, 32);
    gemv_next<float, false>(1, 4096, 4096, 32);
    gemv_next<utils::fp16>(1, 4096, 4096, 32);
    // gemv_next<utils::fp16, false>(1, 4096, 4096, 32);
    // gemv_next<float>(1, 4096, 768, 32);
    // gemv_next<utils::fp16>(1, 16384, 4096, 32);
    // gemv_next<utils::fp16>(1, 4096, 768, 32);

    // dequant_next<float>(256, 768, 128, true);
    // dequant_next<float>(256, 768, 32, true);
    // dequant_next<float>(256, 768, 16, true);
    // dequant_next<utils::fp16>(256, 768, 128, true);
    // dequant_next<utils::fp16>(256, 768, 32, true);
    // dequant_next<utils::fp16>(256, 768, 16, true);
    dequant_next<float>(4096, 4096, 32);
    dequant_next<float, false>(4096, 4096, 32);
    dequant_next<utils::fp16>(4096, 4096, 32);
    dequant_next<utils::fp16, false>(4096, 4096, 32);
  }
  static constexpr size_t nbits = 8;

  template <typename T, bool IsE4M3 = true>
  void gemv_next(int m, int n, int k, int blocksize) {
    int blks = k / blocksize;
    auto memsize = (size_t)(n * k * nbits / 8 + n * blks * sizeof(T)) + (m * k + m * n) * sizeof(T);
    auto batch = auto_batch(memsize);
    auto dev = UT_Device::get();
    printf("Test Case %s: %d %d %d %d, %s batch %d Device:%s\n", __FUNCTION__, m, n, k, blocksize,
           IsE4M3 ? "E4M3" : "E5M2", batch, dev->getName().c_str());
    using LOG = timer_statistics_logger<TestMs * 2>;
    LOG log;
    auto q = dev->getQueue();
    using ST = std::conditional_t<std::is_same_v<T, utils::fp16>, sycl::half, float>;
    avector<uint8_t> rawB(k * n);
    avector<T> scale(blks * n), A(m * k), C(n * m);
    avector<float> Bf32(n * k), Af32(m * k), ref(n * m);
    fill_buffer_randn(A.data(), A.size(), T(-.5f), T(0.5f));
    fill_buffer_randn(scale.data(), scale.size(), T(0.01f), T(0.03f));
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));

    sanitize_fp8_buffer<IsE4M3>(rawB.data(), rawB.size());
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j++) {
        auto noffset = i * blks + j / blocksize;
        Bf32[i + j * n] = decode_fp8_byte<IsE4M3>(rawB[i * k + j]) * (float)scale[noffset];
      }
    }
    Af32 = A.template to<float>();

    gemmref_fp32fp32fp32(m, n, k, Af32.data(), Bf32.data(), ref.data(), k, n, n);
    sycl_vector<ST> dS(scale.size() * batch, q), dA(A.size() * batch, q), dC((size_t)n * k * batch, q);
    sycl_vector<uint8_t> dB(rawB.size() * batch, q);
    for (size_t i = 0; i < batch; i++) {
      q->memcpy(dS.data() + i * scale.size(), scale.data(), scale.size() * sizeof(T));
      q->memcpy(dB.data() + i * rawB.size(), rawB.data(), rawB.size() * 1);
      q->memcpy(dA.data() + i * A.size(), A.data(), A.size() * sizeof(T));
    }
    q->wait();
    utils::timer<std::chrono::milliseconds> tm;
    utils::timer<std::chrono::microseconds> tm1;
    using ProB = sycl_prologue_b::WeightF8T<ST, IsE4M3>;
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();
    auto S_d = dS.data();
    auto psize = (size_t)m * n * k * 2;
    tm.start();
    while (tm.stop() < TestMs) {
      tm1.start();
      for (size_t i = 0; i < batch; i++) {
        auto ev = ProB::template gemv<ST>(A_d + i * m * k, {B_d + i * n * k, S_d + i * n * blks, blks}, C_d + i * m * n,
                                          n, k, blocksize, q);
      }
      q->wait();
      log.add(tm1.stop() / (float)batch);
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;

    double band = double(memsize) / log.min_val / 1e6;
    printf(" %s Flops:%.3f MemoryBandwidth:%.3fGB/s\n", log.get_log_str(), flops, band);
    q->memcpy(C.data(), C_d, C.size() * sizeof(T)).wait();
    auto Cf32 = C.template to<float>();
    buffer_error(ref.data(), Cf32.data(), ref.size(), 0.3f, 0.1f);
  }

  template <typename T, bool IsE4M3 = true>
  void dequant_next(int n, int k, int blocksize) {
    int blks = updiv(k, blocksize);
    auto psize = (size_t)n * k * sizeof(T) + n * k * nbits / 8 + n * k / blocksize * sizeof(T);
    size_t batch = auto_batch(psize);
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case %s: %d %d %d, %s batch %d Device:%s\n", __FUNCTION__, n, k, blocksize, IsE4M3 ? "E4M3" : "E5M2",
           batch, dev->getName().c_str());
    avector<uint8_t> rawB(k * n);
    using ST = std::conditional_t<std::is_same_v<T, utils::fp16>, sycl::half, float>;
    avector<T> scale(blks * n), dequant(n * k), ref(n * k);
    fill_buffer_randn(scale.data(), scale.size(), T(0.01f), T(0.03f));
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    sanitize_fp8_buffer<IsE4M3>(rawB.data(), rawB.size());
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j++) {
        auto noffset = i * blks + j / blocksize;
        ref[i * k + j] = decode_fp8_byte<IsE4M3>(rawB[i * k + j]) * (float)scale[noffset];
      }
    }

    sycl_vector<ST> dS(scale.size() * batch, q), dequantB((size_t)n * k * batch, q);
    sycl_vector<uint8_t> dB(rawB.size() * batch, q);
    for (size_t i = 0; i < batch; i++) {
      q->memcpy(dS.data() + i * scale.size(), scale.data(), scale.size() * sizeof(T));
      q->memcpy(dB.data() + i * rawB.size(), rawB.data(), rawB.size() * 1);
    }
    q->wait();

    auto S_d = dS.data();
    auto B_d = dB.data();
    auto DB_d = dequantB.data();
    using LOG = timer_statistics_logger<TestMs * 2>;
    LOG log;
    using ProB = sycl_prologue_b::WeightF8T<ST, IsE4M3>;
    utils::timer<std::chrono::milliseconds> tm;
    utils::timer<std::chrono::microseconds> tm1;
    tm.start();
    while (tm.stop() < TestMs) {
      tm1.start();
      for (size_t i = 0; i < batch; i++) {
        if constexpr (sizeof(T) == 4) {
          auto ev = ProB::template dequant<typename ProB::Cfg>(
              n, k, blocksize, {B_d + i * rawB.size(), S_d + i * scale.size(), blks}, DB_d + i * n * k, q);
        } else {
          auto ev = ProB::template dequant<typename ProB::Cfg>(
              n, k, blocksize, {B_d + i * rawB.size(), S_d + i * scale.size(), blks}, DB_d + i * n * k, q);
        }
      }
      q->wait();
      log.add(tm1.stop() / (float)batch);
      if (tm.stop() >= TestMs) {
        break;
      }
    }
    q->memcpy(dequant.data(), DB_d, dequant.size() * sizeof(T)).wait();
    buffer_error(ref.data(), dequant.data(), dequant.size(), T(0.3f), T(0.1f));
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    printf(" %s Memory Bandwidth:%.3f\n", log.get_log_str(), flops);
  }
};
#ifdef BTLA_UT_SYCL
static Benchmark_F8 sBenchmark_F8;
#endif

class Benchmark_SDPAFp16Fp16 {
 public:
  Benchmark_SDPAFp16Fp16() {
    UT_START();
    // benchmark_all(512, 1024, 20, 20, 128, 128, true);
    benchmark_all(512, 512, 16, 16, 128, 128, true);
    benchmark_all(512, 512, 16 * 32, 16 * 32, 128, 128, false);
    // benchmark_all(1, 1024, 768, true);
    // benchmark_all(1, 4096, 4096);
    // benchmark_all(4096, 4096, 4096);
  }

  using AType = sycl::half;
  using BType = sycl::half;
  using CType = sycl::half;

  void benchmark(int head_num, int head_kv_num, int seq_cur, int seq_kv, int head_idim, int head_odim, float scale,
                 AType* q_p, BType* k_p, CType* v_p, CType* out, CType* mask, bool verify) {
    using LOG = timer_statistics_logger<TestMs * 2>;
    LOG log;
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    float timems = TestMs;
    utils::timer<std::chrono::milliseconds> tm;
    // auto A_d = A;
    // auto B_d = B;
    // auto C_d = C;
    // auto psize = (size_t)m * n * k * 2;
    // using CFG = xmx::HGemmCfg;
    q->wait();
    size_t constexpr sg_size = 16;
    size_t constexpr TM = 16, TN = 16, TK = 16;
    size_t constexpr SGM = 1, SGN = 2, SGN2 = 8;
    size_t constexpr TileSKV = TN * SGN;
    size_t constexpr TileSQ = TM * SGM;
    size_t constexpr TileHO = TN * SGN2;
    size_t constexpr SG_SQ = 1;
    int constexpr UnrollK = 2;
    int sg_col = head_odim / TileHO;
    int sg_count = sg_col * SG_SQ;
    size_t wg_size = sg_size * sg_col * SG_SQ;
    int constexpr WG_SQ = TileSQ * SG_SQ;
    int sq_wg_count = seq_cur / WG_SQ;
    size_t wg_repeat = head_kv_num * sq_wg_count;
    static int constexpr PrefetchDis = 3;
    using sycl::ext::oneapi::experimental::matrix::layout;
    using sycl::ext::oneapi::experimental::matrix::use;
    CType hscale = scale;
    auto ker = [&](sycl::handler& cgh) {
      // sycl::stream outs(65536, 256, cgh);
      cgh.parallel_for(sycl::nd_range<2>({wg_repeat, wg_size}, {1, wg_size}), [=](sycl::nd_item<2>
                                                                                      it) [[sycl::reqd_sub_group_size(
                                                                                  sg_size)]] {
        auto g = it.get_group();
        auto sg = it.get_sub_group();
        int g_id = it.get_group(0);
        int g_sq = g_id % sq_wg_count;
        int g_hkv = g_id / sq_wg_count;
        // int g_hkv = g_id % head_kv_num;
        // int g_sq = g_id / head_kv_num;
        int sgSize = sg.get_local_range()[0];
        int sgGroupId = sg.get_group_id()[0];
        int sggid_col = sgGroupId % sg_col;
        int sggid_row = sgGroupId / sg_col;
        int sg_sq = g_sq * WG_SQ + sggid_row * TileSQ;
        if (sg_sq >= seq_cur || g_hkv >= head_kv_num) return;
        int sgId = sg.get_local_id()[0];
        auto g_q = q_p + g_hkv * head_idim;
        auto g_k = k_p + g_hkv * head_idim;
        auto g_v = v_p + g_hkv * head_odim + sggid_col * TileHO;
        auto g_o = out + g_hkv * head_odim + sggid_col * TileHO;
        auto pQ = syclex::annotated_ptr{
            g_q, syclex::properties{syclintelex::read_hint<
                     syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L1>,
                     syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L3>>}};
        auto pK = syclex::annotated_ptr{
            g_k, syclex::properties{syclintelex::read_hint<
                     syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L1>,
                     syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L3>>}};
        auto pV = syclex::annotated_ptr{
            g_v, syclex::properties{syclintelex::read_hint<
                     syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L1>,
                     syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L3>>}};
        auto pO = syclex::annotated_ptr{
            g_o, syclex::properties{syclintelex::write_hint<syclintelex::cache_control<
                     syclintelex::cache_mode::write_through, syclex::cache_level::L1, syclex::cache_level::L3>>}};

        using SA = sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, AType, use::a, TM, TK,
                                                                         layout::row_major>;
        sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, AType, use::a, TM, TK, layout::row_major>
            sub_Q[SGM], subSA[SGM * SGN];
        sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, BType, use::b, TK, TN, layout::col_major>
            sub_K[SGN];
        sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, CType, use::accumulator, TM, TN>
            sub_S[SGM * SGN];
        sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, BType, use::b, TK, TN, layout::row_major>
            sub_V[SGN2];
        sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, CType, use::accumulator, TM, TN>
            sub_O[SGM * SGN2], gsub_O[SGM * SGN2];
        CType g_m[SGM];
        CType g_l[SGM];
#pragma unroll
        for (int i = 0; i < SGM; i++) {
          g_m[i] = CType(0);
          g_l[i] = CType(0);
        }
        int ldq = head_num * head_idim;
        int ldk = head_kv_num * head_idim;
        int ldv = head_kv_num * head_odim;
        int ldo = head_num * head_odim;
#pragma unroll
        for (int i = 0; i < SGM * SGN2; i++) joint_matrix_fill(sg, gsub_O[i], 0);
        sycl::group_barrier(sg);
        // iskv: N loop for 1st mm, K loop for 2nd mm
        for (int iskv = 0; iskv < seq_kv; iskv += TileSKV) {
#pragma unroll
          for (int i = 0; i < SGM * SGN; i++) joint_matrix_fill(sg, sub_S[i], 0);
          for (int ik = 0; ik < head_idim; ik += TK * UnrollK) {
#pragma unroll
            for (int ikk = 0; ikk < UnrollK; ikk++) {
#pragma unroll
              for (int in = 0; in < SGN; in++) {
                sycl::ext::intel::experimental::matrix::joint_matrix_load_checked(
                    sg, sub_K[in], pK, ldk, seq_kv, head_idim, iskv + in * TN, ik + ikk * TK);
              }
#pragma unroll
              for (int im = 0; im < SGM; im++) {
                sycl::ext::intel::experimental::matrix::joint_matrix_load_checked(
                    sg, sub_Q[im], pQ, ldq, seq_cur, head_idim, sg_sq + im * TM, ik + ikk * TK);
#pragma unroll
                for (int in = 0; in < SGN; in++) {
                  joint_matrix_mad(sg, sub_S[im * SGN + in], sub_Q[im], sub_K[in], sub_S[im * SGN + in]);
                }
              }
            }
            // if (iskv + TileSKV <= seq_kv) {
            //   for (int in = sgGroupId; in < SGN; in += sg_count)
            //     joint_matrix_prefetch<TN, TK * UnrollK>(sg, g_k + (in * TN) * ldk + ik + TK * UnrollK * PrefetchDis,
            //                                             ldk, layout::row_major,
            //                                             syclex::properties{syclex::prefetch_hint_L1});
            //   for (int in = sgGroupId; in < UnrollK; in += sg_count)
            //     joint_matrix_prefetch<TM, TK>(sg, g_q + sg_sq * ldq + ik + TK * UnrollK * PrefetchDis + in * TK, ldq,
            //                                   layout::row_major, syclex::properties{syclex::prefetch_hint_L1});
            // }
          }
          sycl::group_barrier(sg);

          CType maxS[SGM];
#pragma unroll
          for (int im = 0; im < SGM; im++) {
#pragma unroll
            for (int in = 0; in < SGN; in++) {
              auto wi_data_c = sycl::ext::oneapi::detail::get_wi_data(sg, sub_S[im * SGN + in]);
#pragma unroll
              for (int imm = 0; imm < TM; imm++) {
                auto element = wi_data_c[imm];
                CType tmp = element * hscale + mask[(sg_sq + im * TM + imm) * seq_kv + iskv + in * TN + sgId];
                CType _max = sycl::reduce_over_group(sg, tmp, sycl::maximum<CType>());
                if (imm == sgId) maxS[im] = std::max(maxS[im], _max);
                wi_data_c[imm] = tmp;
              }
            }
          }
          sycl::group_barrier(sg);

          CType sumExp[SGM];
#pragma unroll
          for (int i = 0; i < SGM; i++) {
            sumExp[i] = CType(0);
          }
#pragma unroll
          for (int im = 0; im < SGM; im++) {
#pragma unroll
            for (int in = 0; in < SGN; in++) {
              auto wi_data_c = sycl::ext::oneapi::detail::get_wi_data(sg, sub_S[im * SGN + in]);
#pragma unroll
              for (int imm = 0; imm < TM; imm++) {
                auto element = wi_data_c[imm];
                CType tmp = sycl::exp(element - sycl::select_from_group(sg, maxS[im], imm));
                CType sumtmp = sycl::reduce_over_group(sg, tmp, sycl::plus<CType>());
                if (imm == sgId) sumExp[im] += sumtmp;
                wi_data_c[imm] = tmp;
              }
            }
          }
          CType blk_scale0[SGM], blk_scale1[SGM];
          CType maxS_[SGM], sumExp_[SGM];

#pragma unroll
          for (int i = 0; i < SGM; i++) {
            maxS_[i] = std::max(g_m[i], maxS[i]);
            blk_scale0[i] = sycl::exp(g_m[i] - maxS_[i]) * g_l[i];
            blk_scale1[i] = sycl::exp(maxS[i] - maxS_[i]);
            sumExp_[i] = blk_scale0[i] + blk_scale1[i] * sumExp[i];
            blk_scale0[i] = blk_scale0[i] / sumExp_[i];
            blk_scale1[i] = blk_scale1[i] / sumExp_[i];
            g_m[i] = maxS_[i];
            g_l[i] = sumExp_[i];
          }
          sycl::group_barrier(sg);

#pragma unroll
          for (int i = 0; i < SGM * SGN2; i++) joint_matrix_fill(sg, sub_O[i], 0);
#pragma unroll
          for (int ik2 = 0; ik2 < SGN; ik2++) {
#pragma unroll
            for (int in = 0; in < SGN2; in++) {
              sycl::ext::intel::experimental::matrix::joint_matrix_load_checked(
                  sg, sub_V[in], pV, head_kv_num * head_odim, seq_kv, head_odim, iskv + ik2 * TK, in * TN);
#pragma unroll
              for (int im = 0; im < SGM; im++) {
                joint_matrix_mad(sg, sub_O[im * SGN2 + in], *(SA*)&sub_S[im * SGN + ik2], sub_V[in],
                                 sub_O[im * SGN2 + in]);
              }
            }
          }
          sycl::group_barrier(sg);

          for (int im = 0; im < SGM; im++) {
            for (int in = 0; in < SGN2; in++) {
              auto wi_data_c0 = sycl::ext::oneapi::detail::get_wi_data(sg, gsub_O[im * SGN2 + in]);
              auto wi_data_c1 = sycl::ext::oneapi::detail::get_wi_data(sg, sub_O[im * SGN2 + in]);
              for (int imm = 0; imm < TM; imm++) {
                auto element0 = wi_data_c0[imm];
                auto element1 = wi_data_c1[imm];
                wi_data_c0[imm] = element0 * sycl::select_from_group(sg, blk_scale0[im], imm) +
                                  element1 * sycl::select_from_group(sg, blk_scale1[im], imm);
              }
            }
          }
          sycl::group_barrier(sg);
          // if (iskv + TileSKV * (PrefetchDis + 1) <= seq_kv) {
          //   for (int im = sgGroupId; im < SGN; im += sg_count)
          //     joint_matrix_prefetch<TM, TK>(sg, pK + (im * TM) * lda + ik + TK * UnrollK * PrefetchDis,
          //                                             lda, layout::row_major,
          //                                             syclex::properties{syclex::prefetch_hint_L1});
          //   for (int in = sggid_row; in < SGN; in += sg_row)
          //     if ((g_idn + in * TN) <= n)
          //       joint_matrix_prefetch<TN, TK * UnrollK>(sg, Bwg_d + (in * TN) * ldb + ik + TK * UnrollK *
          //       PrefetchDis,
          //                                               ldb, layout::row_major,
          //                                               syclex::properties{syclex::prefetch_hint_L1});
          // }
        }
        sycl::group_barrier(sg);

#pragma unroll
        for (int im = 0; im < SGM; im++) {
#pragma unroll
          for (int in = 0; in < SGN2; in++) {
            sycl::ext::intel::experimental::matrix::joint_matrix_store_checked(
                sg, gsub_O[im * SGN2 + in], pO, head_num * head_odim, layout::row_major, seq_cur, head_odim,
                sg_sq + im * TM, in * TN);
          }
        }
      });  // parallel for
    };
    auto ker2 = [&](sycl::handler& cgh) {
      // sycl::stream outs(65536, 256, cgh);
      cgh.parallel_for(sycl::nd_range<2>({wg_repeat, wg_size}, {1, wg_size}), [=](sycl::nd_item<2>
                                                                                      it) [[sycl::reqd_sub_group_size(
                                                                                  sg_size)]] {
        auto g = it.get_group();
        auto sg = it.get_sub_group();
        int g_id = it.get_group(0);
        // int g_sq = g_id % sq_wg_count;
        // int g_hkv = g_id / sq_wg_count;
        int g_hkv = g_id % head_kv_num;
        int g_sq = g_id / head_kv_num;
        int sgSize = sg.get_local_range()[0];
        int sgGroupId = sg.get_group_id()[0];
        int sggid_col = sgGroupId % sg_col;
        int sggid_row = sgGroupId / sg_col;
        int sg_sq = g_sq * WG_SQ + sggid_row * TileSQ;
        if (sg_sq >= seq_cur || g_hkv >= head_kv_num) return;
        int sgId = sg.get_local_id()[0];
        auto g_q = q_p + g_hkv * head_idim;
        auto g_k = k_p + g_hkv * head_idim;
        auto g_v = v_p + g_hkv * head_odim + sggid_col * TileHO;
        auto g_o = out + g_hkv * head_odim + sggid_col * TileHO;
        auto pQ = syclex::annotated_ptr{
            g_q, syclex::properties{syclintelex::read_hint<
                     syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L1>,
                     syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L3>>}};
        auto pK = syclex::annotated_ptr{
            g_k, syclex::properties{syclintelex::read_hint<
                     syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L1>,
                     syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L3>>}};
        auto pV = syclex::annotated_ptr{
            g_v, syclex::properties{syclintelex::read_hint<
                     syclintelex::cache_control<syclintelex::cache_mode::cached, syclex::cache_level::L1>,
                     syclintelex::cache_control<syclintelex::cache_mode::streaming, syclex::cache_level::L3>>}};
        auto pO = syclex::annotated_ptr{
            g_o, syclex::properties{syclintelex::write_hint<syclintelex::cache_control<
                     syclintelex::cache_mode::write_through, syclex::cache_level::L1, syclex::cache_level::L3>>}};

        using SA = sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, AType, use::a, TM, TK,
                                                                         layout::row_major>;
        sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, AType, use::a, TM, TK, layout::row_major>
            sub_Q[SGM];
        sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, BType, use::b, TK, TN, layout::col_major>
            sub_K[SGN];
        sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, CType, use::accumulator, TM, TN>
            sub_S[SGM * SGN];
        sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, BType, use::b, TK, TN, layout::row_major>
            sub_V[SGN2];
        sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, CType, use::accumulator, TM, TN>
            sub_O[SGM * SGN2], gsub_O[SGM * SGN2];

        int ldq = head_num * head_idim;
        int ldk = head_kv_num * head_idim;
        int ldv = head_kv_num * head_odim;
        int ldo = head_num * head_odim;
#pragma unroll
        for (int i = 0; i < SGM * SGN2; i++) joint_matrix_fill(sg, gsub_O[i], 0);

        sycl::group_barrier(sg);
        // iskv: N loop for 1st mm, K loop for 2nd mm
        for (int iskv = 0; iskv < seq_kv; iskv += TileSKV) {
          // if (iskv + TileSKV * (PrefetchDis + 1) <= seq_kv) {
          //   for (int in = sgGroupId; in < SGN * 4; in += sg_count) {
          //     int _in = in % SGN;
          //     int _ih = in / SGN;
          //     joint_matrix_prefetch<TN, TK * UnrollK>(
          //         sg, g_k + (iskv + TileSKV * PrefetchDis + _in * TN) * ldk + _ih * TK * UnrollK, ldk,
          //         layout::row_major, syclex::properties{syclex::prefetch_hint_L1});
          //   }
          // }
          // for (int in = sgGroupId; in < SGN2; in += sg_count) {
          //   joint_matrix_prefetch<TN, TK * SGN>(sg, g_v + (iskv + TileSKV * 1) * ldq + in * TK * SGN, ldq,
          //                                       layout::row_major, syclex::properties{syclex::prefetch_hint_L1});
          // }
#pragma unroll
          for (int i = 0; i < SGM * SGN; i++) joint_matrix_fill(sg, sub_S[i], 0);
          sycl::group_barrier(sg);
          for (int ik = 0; ik < head_idim; ik += TK * UnrollK) {
            // if (ik + TK * UnrollK * (PrefetchDis + 1) <= head_idim) {
            //   for (int in = sgGroupId; in < SGN; in += sg_count) {
            //     joint_matrix_prefetch<TN, TK * UnrollK>(
            //         sg, g_k + (iskv + in * TN) * ldk + ik + TK * UnrollK * PrefetchDis, ldk, layout::row_major,
            //         syclex::properties{syclex::prefetch_hint_L1});
            //   }
            //   for (int in = sgGroupId; in < SGM; in += sg_count) {
            //     joint_matrix_prefetch<TM, TK * UnrollK>(
            //         sg, g_q + (sg_sq + in * TM) * ldq + ik + TK * UnrollK * PrefetchDis, ldq, layout::row_major,
            //         syclex::properties{syclex::prefetch_hint_L1});
            //   }
            // } else if (iskv + TileSKV < seq_kv) {
            //   for (int in = sgGroupId; in < SGN; in += sg_count) {
            //     joint_matrix_prefetch<TN, TK * UnrollK>(
            //         sg, g_k + (iskv + TileSKV + in * TN) * ldk + ik + TK * UnrollK * PrefetchDis - head_idim, ldk,
            //         layout::row_major, syclex::properties{syclex::prefetch_hint_L1});
            //   }
            // }

#pragma unroll
            for (int ikk = 0; ikk < UnrollK; ikk++) {
#pragma unroll
              for (int in = 0; in < SGN; in++) {
                sycl::ext::intel::experimental::matrix::joint_matrix_load_checked(
                    sg, sub_K[in], pK, ldk, seq_kv, head_idim, iskv + in * TN, ik + ikk * TK);
              }
#pragma unroll
              for (int im = 0; im < SGM; im++) {
                sycl::ext::intel::experimental::matrix::joint_matrix_load_checked(
                    sg, sub_Q[im], pQ, ldq, seq_cur, head_idim, sg_sq + im * TM, ik + ikk * TK);
#pragma unroll
                for (int in = 0; in < SGN; in++) {
                  joint_matrix_mad(sg, sub_S[im * SGN + in], sub_Q[im], sub_K[in], sub_S[im * SGN + in]);
                }
              }
            }
          }
          sycl::group_barrier(sg);

#pragma unroll
          for (int i = 0; i < SGM * SGN2; i++) joint_matrix_fill(sg, sub_O[i], 0);
#pragma unroll
          for (int ik2 = 0; ik2 < SGN; ik2++) {
#pragma unroll
            for (int in = 0; in < SGN2; in++) {
              sycl::ext::intel::experimental::matrix::joint_matrix_load_checked(
                  sg, sub_V[in], pV, head_kv_num * head_odim, seq_kv, head_odim, iskv + ik2 * TK, in * TN);
#pragma unroll
              for (int im = 0; im < SGM; im++) {
                joint_matrix_mad(sg, sub_O[im * SGN2 + in], *(SA*)&sub_S[im * SGN + ik2], sub_V[in],
                                 sub_O[im * SGN2 + in]);
              }
            }
          }
          sycl::group_barrier(sg);

          for (int im = 0; im < SGM; im++) {
            for (int in = 0; in < SGN2; in++) {
              auto wi_data_c0 = sycl::ext::oneapi::detail::get_wi_data(sg, gsub_O[im * SGN2 + in]);
              auto wi_data_c1 = sycl::ext::oneapi::detail::get_wi_data(sg, sub_O[im * SGN2 + in]);
              for (int imm = 0; imm < TM; imm++) {
                auto element0 = wi_data_c0[imm];
                auto element1 = wi_data_c1[imm];
                wi_data_c0[imm] = element0 + element1;
              }
            }
          }
          sycl::group_barrier(sg);
        }

#pragma unroll
        for (int im = 0; im < SGM; im++) {
#pragma unroll
          for (int in = 0; in < SGN2; in++) {
            sycl::ext::intel::experimental::matrix::joint_matrix_store_checked(
                sg, gsub_O[im * SGN2 + in], pO, head_num * head_odim, layout::row_major, seq_cur, head_odim,
                sg_sq + im * TM, in * TN);
          }
        }
      });  // parallel for
    };

    q->wait();
    for (size_t i = 0; i < 1; i++) {
      q->submit(ker2);
    }
    q->wait();
    if (verify) {
      return;
    }
    tm.start();
    utils::timer<std::chrono::microseconds> tm1;
    while (tm.stop() < timems) {
      tm1.start();
      for (size_t i = 0; i < 10; i++) {
        q->submit(ker2);
      }
      q->wait();
      log.add(tm1.stop() / 10);
      if (tm.stop() >= timems) {
        break;
      }
    }
    log.record();
    size_t group = 1;
    size_t OPS = group * head_kv_num * seq_cur * head_idim * seq_kv * 2;     // q*k
    OPS += group * head_kv_num * seq_cur * seq_kv * 2;                       //  # sfmx
    OPS += group * head_kv_num * seq_cur * head_odim * seq_kv * 2;           //  # s*v
    size_t MEM = group * head_kv_num * seq_cur * head_idim * sizeof(AType);  // # q
    MEM += head_kv_num * seq_kv * head_idim * sizeof(AType);                 //  # k
    MEM += head_kv_num * seq_kv * head_odim * sizeof(AType);                 // # v

    double flops = double(OPS) / log.min_val / 1e6;
    double mms = double(MEM) / log.min_val / 1e6;
    printf(" %s Flops:%.3f GOPS Bandwidth:%.3f GB/s\n", log.get_log_str(), flops, mms);
  }

  void benchmark_all(int seq, int seq_kv, int hn_q, int hn_kv, int h_dim, int h_vdim, bool verify = false) {
    auto batch = verify ? 1 : 32;
    // printf("%d %d %d %d %s %s %s\n", m, n, k, batch, bestla_dtype_str(BTLA_DTYPE::F16),
    //        bestla_dtype_str(BTLA_DTYPE::F16), bestla_dtype_str(BTLA_DTYPE::F16));
    float testtime = float(TestMs);
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    float scale = 1 / sqrt(h_dim);
    sycl_vector<AType> dQ(size_t(seq) * hn_q * h_dim * batch, q);
    sycl_vector<BType> dK(size_t(seq_kv) * hn_kv * h_dim * batch, q);
    sycl_vector<CType> dV(size_t(seq_kv) * hn_kv * h_vdim * batch, q);
    sycl_vector<CType> dO(size_t(seq) * hn_q * h_vdim * batch, q);
    sycl_vector<CType> dMask(seq * seq_kv * batch, q);
    avector<utils::fp16> V(size_t(seq_kv) * hn_kv * h_vdim), O(size_t(seq) * hn_q * h_vdim), Mask(seq * seq_kv);
    avector<utils::fp16> Q(size_t(seq) * hn_q * h_dim), Ref16(size_t(seq) * hn_q * h_vdim),
        K(size_t(seq_kv) * hn_kv * h_dim);
    avector<float> Ref(size_t(seq) * hn_q * h_vdim), Of32(size_t(seq) * hn_q * h_vdim);
    if (verify) {
      fill_buffer_randn(Q.data(), Q.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
      fill_buffer_randn(K.data(), K.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
      fill_buffer_randn(V.data(), V.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
      // for (int i = 0; i < seq_kv / 2; i++) {
      //   K[i * hn_kv * h_dim + (i % h_dim)] = float(K[i * hn_kv * h_dim + (i % h_dim)]) + 10.f;
      //   V[i * hn_kv * h_vdim + (i % h_vdim)] = float(V[i * hn_kv * h_vdim + (i % h_vdim)]) + 10.f;
      // }
      fill_buffer_randn(Mask.data(), Mask.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
      for (size_t i = 0; i < batch; i++) {
        q->memcpy(dQ.data() + i * Q.size(), Q.data(), Q.size() * sizeof(Q[0]));
        q->memcpy(dK.data() + i * K.size(), K.data(), K.size() * sizeof(K[0]));
        q->memcpy(dV.data() + i * V.size(), V.data(), V.size() * sizeof(V[0]));
        q->memcpy(dMask.data() + i * Mask.size(), Mask.data(), Mask.size() * sizeof(Mask[0]));
      }
      q->wait();
    }

    benchmark(hn_q, hn_kv, seq, seq_kv, h_dim, h_vdim, scale, dQ.data(), dK.data(), dV.data(), dO.data(), dMask.data(),
              verify);
    if (verify) {
      auto Qf32 = Q.to<float>();
      auto Kf32 = K.to<float>();
      sdpa_ref(hn_q, hn_kv, seq, seq_kv, h_dim, h_vdim, scale, Qf32.data(), Kf32.data(), V.to<float>().data(),
               Ref.data(), Mask.to<float>().data(), true);
      q->memcpy(O.data(), dO.data(), O.size() * sizeof(O[0])).wait();
      Of32 = O.to<float>();
      buffer_error(Ref.data(), Of32.data(), Of32.size(), 1.f);
      print_buffer(Ref.data(), 16);
      print_buffer(O.data(), 16);
    }
  }
};
#ifdef BTLA_UT_SYCL
static Benchmark_SDPAFp16Fp16 sBenchmark_SDPAFp16Fp16;
#endif
}  // namespace sycl_ut
}  // namespace bestla
