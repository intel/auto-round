#pragma once
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include "common.hpp"
#include "../include/xpu_wrapper.hpp"

struct TestGemm {
  TestGemm() {
    test<float>(128, 128, 128);
#ifdef ARK_XPU
    test<bf16>(128, 128, 128);
    test<fp16>(128, 128, 128);
    test_s8s8<float>(128, 128, 128);
    test_woqs8<float>(128, 128, 128);
    test_woq_s3<float>(128, 128, 128);
    test_woq_s3<float>(256, 96, 32);
    test_woq_s3<float>(64, 256, 64);
    test_woq_s3_gemm<float>(4, 128, 128, 128);
    test_woq_s3_gemm<float>(8, 256, 128, 128);
    test_woq_s3_gemm<float>(16, 64, 256, 64);
#endif
  }

  template <typename T>
  void test(size_t m, size_t n, size_t k) {
    GETQ();
    LOG_LINE();
    auto dt = ark::to_dt<T>();
    auto A = ctx->allocate(m * k * sizeof(T));
    auto B = ctx->allocate(n * k * sizeof(T));
    auto C = ctx->allocate(m * n * sizeof(T));
    ark::DnnlWrapper::gemm(q, m, n, k, A, dt, B, dt, false, C, dt, nullptr);
#ifdef ARK_XPU
    q->wait();
#endif
    ctx->deallocate(A);
    ctx->deallocate(B);
    ctx->deallocate(C);
  }

  template <typename T>
  void test_s8s8(size_t m, size_t n, size_t k) {
    GETQ();
    LOG_LINE();
    auto dt = ark::to_dt<T>();
    auto A = ctx->allocate(m * k * sizeof(int8_t));
    auto B = ctx->allocate(n * k * sizeof(int8_t));
    auto C = ctx->allocate(m * n * sizeof(T));
    auto scaleA = ctx->allocate(1 * sizeof(T));
    auto scaleB = ctx->allocate(n * sizeof(T));
    ark::DnnlWrapper::igemm_s8s8(q, m, n, k, A, B, true, C, dt, scaleA, scaleB, nullptr);
    ctx->deallocate(A);
    ctx->deallocate(B);
    ctx->deallocate(C);
    ctx->deallocate(scaleA);
    ctx->deallocate(scaleB);
  }

  template <typename T>
  void test_woqs8(size_t m, size_t n, size_t k) {
    GETQ();
    LOG_LINE();
    auto dt = ark::to_dt<T>();
    auto A = ctx->allocate(m * k * sizeof(T));
    auto B = ctx->allocate(n * k * sizeof(int8_t));
    auto C = ctx->allocate(m * n * sizeof(T));
    auto scaleB = ctx->allocate(n * sizeof(T));
    ark::DnnlWrapper::woq_s8(q, m, n, k, A, B, true, C, dt, scaleB, nullptr, k);
    ctx->deallocate(A);
    ctx->deallocate(B);
    ctx->deallocate(C);
    ctx->deallocate(scaleB);
  }

#ifdef ARK_XPU
  // m==1 int3 (S3) symmetric WOQ GEMV accuracy check: pack random 3-bit weights via the kernel
  // packer, run woq_gemv, and compare against a host fp reference dequant-then-matvec.
  template <typename T>
  void test_woq_s3(size_t n, size_t k, int blocksize) {
    GETQ();
    LOG_LINE();
    if (k % 32 != 0 || k % blocksize != 0 || blocksize % 32 != 0) {
      throw std::runtime_error("test_woq_s3 requires k % 32 == 0, k % blocksize == 0, blocksize % 32 == 0");
    }
    int blks = int(k) / blocksize;

    std::mt19937 rng(7u + uint32_t(n) + uint32_t(k));
    std::uniform_int_distribution<int> wdist(-4, 3);   // symmetric int3 range
    std::uniform_real_distribution<float> adist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> sdist(0.01f, 0.05f);

    // raw signed weights, row-major [k, n]; activations [k]; per-(n,blk) scales.
    std::vector<int8_t> raw(k * n);
    for (auto& w : raw) w = int8_t(wdist(rng));
    std::vector<float> hostA(k);
    for (auto& a : hostA) a = adist(rng);
    // scale source layout for packscale: row-major [blks, n].
    std::vector<float> hostScale(size_t(blks) * n);
    for (auto& s : hostScale) s = sdist(rng);

    // Host reference: C[j] = sum_k A[k] * raw[k,j] * scale[k/blocksize, j].
    std::vector<float> refC(n, 0.0f);
    for (size_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (size_t kk = 0; kk < k; ++kk) {
        float sc = hostScale[(kk / blocksize) * n + j];
        acc += hostA[kk] * float(raw[kk * n + j]) * sc;
      }
      refC[j] = acc;
    }

    ark::QuantParam param{int(n), int(k), blocksize, (int)BTLA_DTYPE::F32, (int)BTLA_DTYPE::S3, (int)BTLA_DTYPE::F32,
                          false};
    size_t blob_size = ark::XpuWrapper::get_packw_size(&param);

    auto* dRaw = reinterpret_cast<int8_t*>(ctx->allocate(raw.size() * sizeof(int8_t)));
    auto* dScale = reinterpret_cast<float*>(ctx->allocate(hostScale.size() * sizeof(float)));
    auto* dBlob = reinterpret_cast<int8_t*>(ctx->allocate(blob_size));
    auto* dA = reinterpret_cast<T*>(ctx->allocate(k * sizeof(T)));
    auto* dC = reinterpret_cast<T*>(ctx->allocate(n * sizeof(T)));

    std::vector<T> hostAt(k);
    for (size_t i = 0; i < k; ++i) hostAt[i] = T(hostA[i]);

    q->memcpy(dRaw, raw.data(), raw.size() * sizeof(int8_t)).wait();
    q->memcpy(dScale, hostScale.data(), hostScale.size() * sizeof(float)).wait();
    q->memcpy(dA, hostAt.data(), k * sizeof(T)).wait();

    // Pack (raw int8 + scale) into the device blob, then run the m==1 GEMV.
    ark::XpuWrapper::packq(dRaw, (void*)dScale, nullptr, dBlob, &param, q);
    q->wait();
    int ret = ark::XpuWrapper::woq_gemv(q, 1, &param, dA, dBlob, dC, nullptr, BTLA_DTYPE::F32);
    q->wait();
    if (ret != 0) {
      throw std::runtime_error("woq_gemv(S3) returned non-zero: " + std::to_string(ret));
    }

    std::vector<T> hostC(n);
    q->memcpy(hostC.data(), dC, n * sizeof(T)).wait();

    float max_diff = 0.0f;
    for (size_t j = 0; j < n; ++j) {
      max_diff = std::max(max_diff, std::fabs(float(hostC[j]) - refC[j]));
    }
    std::cout << "[woq_s3][accuracy] n=" << n << " k=" << k << " blk=" << blocksize << " max_diff=" << max_diff << "\n";
    if (max_diff > 1e-2f) {
      throw std::runtime_error("woq_s3 accuracy check failed");
    }

    ctx->deallocate(dRaw);
    ctx->deallocate(dScale);
    ctx->deallocate(dBlob);
    ctx->deallocate(dA);
    ctx->deallocate(dC);
  }

  // m>1 int3 (S3) symmetric WOQ GEMM accuracy check. Drives the full woq_gemm path with m>1 and
  // compute_type=S8 (matching the Python qlinear cdt="int8") to exercise the fp-compute fallback
  // that decodes the dense 3-bit blob via WeightS3T::dequant and runs a DNNL fp GEMM. Compares
  // against a host fp reference dequant-then-matmul.
  template <typename T>
  void test_woq_s3_gemm(size_t m, size_t n, size_t k, int blocksize) {
    GETQ();
    LOG_LINE();
    if (k % 32 != 0 || k % blocksize != 0 || blocksize % 32 != 0) {
      throw std::runtime_error("test_woq_s3_gemm requires k % 32 == 0, k % blocksize == 0, blocksize % 32 == 0");
    }
    int blks = int(k) / blocksize;

    std::mt19937 rng(11u + uint32_t(m) + uint32_t(n) + uint32_t(k));
    std::uniform_int_distribution<int> wdist(-4, 3);  // symmetric int3 range
    std::uniform_real_distribution<float> adist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> sdist(0.01f, 0.05f);

    // raw signed weights, row-major [k, n]; activations [m, k]; per-(blk, n) scales [blks, n].
    std::vector<int8_t> raw(k * n);
    for (auto& w : raw) w = int8_t(wdist(rng));
    std::vector<float> hostA(m * k);
    for (auto& a : hostA) a = adist(rng);
    std::vector<float> hostScale(size_t(blks) * n);
    for (auto& s : hostScale) s = sdist(rng);

    // Host reference: C[i,j] = sum_k A[i,k] * raw[k,j] * scale[k/blocksize, j].
    std::vector<float> refC(m * n, 0.0f);
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j) {
        float acc = 0.0f;
        for (size_t kk = 0; kk < k; ++kk) {
          float sc = hostScale[(kk / blocksize) * n + j];
          acc += hostA[i * k + kk] * float(raw[kk * n + j]) * sc;
        }
        refC[i * n + j] = acc;
      }
    }

    // compute_type=S8 mirrors the Python XPU path (cdt="int8"); woq_gemm must force S3 to fp.
    ark::QuantParam param{int(n), int(k), blocksize, (int)BTLA_DTYPE::S8, (int)BTLA_DTYPE::S3, (int)BTLA_DTYPE::F32,
                          false};
    size_t blob_size = ark::XpuWrapper::get_packw_size(&param);

    auto* dRaw = reinterpret_cast<int8_t*>(ctx->allocate(raw.size() * sizeof(int8_t)));
    auto* dScale = reinterpret_cast<float*>(ctx->allocate(hostScale.size() * sizeof(float)));
    auto* dBlob = reinterpret_cast<int8_t*>(ctx->allocate(blob_size));
    auto* dA = reinterpret_cast<T*>(ctx->allocate(m * k * sizeof(T)));
    auto* dC = reinterpret_cast<T*>(ctx->allocate(m * n * sizeof(T)));

    std::vector<T> hostAt(m * k);
    for (size_t i = 0; i < m * k; ++i) hostAt[i] = T(hostA[i]);

    q->memcpy(dRaw, raw.data(), raw.size() * sizeof(int8_t)).wait();
    q->memcpy(dScale, hostScale.data(), hostScale.size() * sizeof(float)).wait();
    q->memcpy(dA, hostAt.data(), m * k * sizeof(T)).wait();

    ark::XpuWrapper::packq(dRaw, (void*)dScale, nullptr, dBlob, &param, q);
    q->wait();
    ark::XpuWrapper::woq_gemm(int(m), dA, dBlob, dC, nullptr, BTLA_DTYPE::F32, &param, q);
    q->wait();

    std::vector<T> hostC(m * n);
    q->memcpy(hostC.data(), dC, m * n * sizeof(T)).wait();

    float max_diff = 0.0f;
    for (size_t idx = 0; idx < m * n; ++idx) {
      max_diff = std::max(max_diff, std::fabs(float(hostC[idx]) - refC[idx]));
    }
    std::cout << "[woq_s3][gemm] m=" << m << " n=" << n << " k=" << k << " blk=" << blocksize
              << " max_diff=" << max_diff << "\n";
    if (max_diff > 1e-2f) {
      throw std::runtime_error("woq_s3 gemm accuracy check failed");
    }

    ctx->deallocate(dRaw);
    ctx->deallocate(dScale);
    ctx->deallocate(dBlob);
    ctx->deallocate(dA);
    ctx->deallocate(dC);
  }
#endif
};