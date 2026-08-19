#include <iostream>
#include "test_gemm.hpp"
#include "test_quant.hpp"
#include "test_sdpa.hpp"
#include "test_sycltla_dense_gemm_perf.hpp"
#include "test_sycltla_igemm_s8s8_dequant.hpp"

int main() {
  printf("Welcome to ARK TEST\n");
  // TestGemm test_gemm;
  // TestQuant test_quant;
  // TestSDPA test_sdpa;
  TestSyclTlaDenseGemmPerf test_sycl_tla_dense_gemm_perf;
  TestSyclTlaIgemmS8S8Dequant test_sycl_tla_igemm_s8s8_dequant;
  return 0;
}