#include <iostream>
#include "test_gemm.hpp"
#include "test_quant.hpp"
#include "test_sdpa.hpp"

int main() {
  printf("Welcome to ARK TEST\n");
  // TestGemm test_gemm;
  // TestQuant test_quant;
  TestSDPA test_sdpa;
  return 0;
}