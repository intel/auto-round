#include <iostream>
#include "test_gemm.hpp"
#include "test_quant.hpp"
#include "test_sdpa.hpp"

int main() {
  printf("Welcome to ARK TEST\n");
  TestGemm test_gemm;
  // TestQuant test_quant;
  if (ark_s3_bench_only()) return 0;  // ARK_S3_BENCH=1: int3 benchmarks only, skip the SDPA suite.
  TestSDPA test_sdpa;
  return 0;
}