#include <iostream>
#include "test_core_attention_e2e.hpp"
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
  ark::cpu::TestMixedPaddingRight test_mixed_padding_right;  // mixed SDPA padding-right plumbing/validation
  ark::cpu::TestMixedAlibiTanh test_mixed_alibi_tanh;  // mixed SDPA alibi/tanh wiring; homogeneous rejection
  ark::cpu::TestCoreAttentionE2E test_core_attention_e2e;  // four core dtype tuples e2e dispatch validation
  TestSDPA test_sdpa;
  return 0;
}