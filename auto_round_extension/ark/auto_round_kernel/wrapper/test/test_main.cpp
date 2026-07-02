#include <iostream>
#include "test_gemm.hpp"
#include "test_quant.hpp"
#include "test_reorder_kv.hpp"
#include "test_sdpa.hpp"

int main() {
  printf("Welcome to ARK TEST\n");
  // TestGemm test_gemm;
  // TestQuant test_quant;
  ark::cpu::TestReorderKV test_reorder_kv;  // CPU packed K/V reorder layout checks
  ark::cpu::TestPersistentPackedKV test_persistent_packed_kv;  // persistent packed K/V update checks
  ark::cpu::TestPackedForwardSetup test_packed_forward_setup;  // logical-cap/zero-fill/packed-forward checks
  ark::cpu::TestHomogeneousForwardSetup test_homogeneous_forward_setup;  // homogeneous SDPA dispatch validation
  TestSDPA test_sdpa;
  return 0;
}