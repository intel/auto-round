#pragma once
#include "common.hpp"

struct TestGemm {
  TestGemm() {
    test<float>(128, 128, 128);
#ifdef ARK_XPU
    test<bf16>(128, 128, 128);
    test<fp16>(128, 128, 128);
    test_s8s8<float>(128, 128, 128);
    test_woqs8<float>(128, 128, 128);
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
};