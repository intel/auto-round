//
// Route B torch-free harness: calls ARK's REAL joint_matrix kernel
// (bestla::sycl_gemm::xmx::IGemmDQCore via Launcher::run), the exact m>1
// int8 WOQ GEMM path that DnnlWrapper::sycl_igemm_s8s8 instantiates.
// No torch, no oneDNN. Purpose: prove the "joint_matrix not supported" runtime
// failure is determined ONLY by libsycl, by building once and running under
// 2025.3.x (.so.8, expect FAIL) vs 2026.0.0 (.so.9, expect PASS).
//
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <sycl/sycl.hpp>
#include "bestla/sycl/sycl_wrapper.h"

using namespace bestla;
using namespace bestla::sycl_gemm;

int main(int argc, char** argv) {
  int m = argc > 1 ? atoi(argv[1]) : 128;
  int n = argc > 2 ? atoi(argv[2]) : 128;
  int k = argc > 3 ? atoi(argv[3]) : 128;
  printf("[harness] ARK IGemmDQCore (joint_matrix) m=%d n=%d k=%d\n", m, n, k);

  sycl::queue q(sycl::gpu_selector_v);
  auto dev = q.get_device();
  printf("[harness] device: %s\n", dev.get_info<sycl::info::device::name>().c_str());

  // A: m*k int8 ; B: n*k int8 (col-major weight) ; C: m*n float
  // scaleA: m float ; scaleB: n float
  auto* A = sycl::malloc_device<int8_t>((size_t)m * k, q);
  auto* B = sycl::malloc_device<int8_t>((size_t)n * k, q);
  auto* C = sycl::malloc_device<float>((size_t)m * n, q);
  auto* sA = sycl::malloc_device<float>((size_t)m, q);
  auto* sB = sycl::malloc_device<float>((size_t)n, q);

  // init (values irrelevant; we only need the kernel to JIT+launch)
  q.memset(A, 1, (size_t)m * k * sizeof(int8_t));
  q.memset(B, 1, (size_t)n * k * sizeof(int8_t));
  std::vector<float> ones_m(m, 1.0f), ones_n(n, 1.0f);
  q.memcpy(sA, ones_m.data(), m * sizeof(float));
  q.memcpy(sB, ones_n.data(), n * sizeof(float));
  q.wait();

  // Param layout mirrors sycl_igemm_s8s8's call:
  //   {A, B, C, m, n, k, lda=k, ldb=k, ldc=n, Bias=nullptr, scaleA, scaleB}
  using T = float;
  xmx::IGemmDQParam param;
  param.A_d = A; param.B_d = B; param.C_d = C;
  param.m = m; param.n = n; param.k = k;
  param.lda = k; param.ldb = k; param.ldc = n;
  param.Bias = nullptr;
  param.scaleA = sA; param.scaleB = sB;

  printf("[harness] launching joint_matrix kernel...\n");
  fflush(stdout);
  try {
    auto ev = Launcher<xmx::IGemmDQCfg<T>, xmx::IGemmDQCore>::run(&q, param);
    ev.wait_and_throw();
  } catch (sycl::exception const& e) {
    printf("[harness] FAIL: sycl::exception: %s\n", e.what());
    return 2;
  } catch (std::exception const& e) {
    printf("[harness] FAIL: std::exception: %s\n", e.what());
    return 3;
  }
  printf("[harness] PASS: joint_matrix kernel launched and completed.\n");

  sycl::free(A, q); sycl::free(B, q); sycl::free(C, q);
  sycl::free(sA, q); sycl::free(sB, q);
  return 0;
}
