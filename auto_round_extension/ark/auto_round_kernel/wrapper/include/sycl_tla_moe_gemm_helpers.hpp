#pragma once

#include <cctype>
#include <cstdlib>
#include <string>
#include "sycl_tla_common.hpp"

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

namespace ark {
namespace moe_gemm_detail {

// Declaration-only view of the bf16/fp16 grouped MoE GEMM, mirroring
// `sycl_tla_moe_prefill_fp8_helpers.hpp`.
//
// `moe_gemm_dispatch` picks one of three work-group tile policies from the
// output width `N`. Expanding all three inside `sycl_tla_moe_<dtype>.cpp` put
// three full cutlass grouped-GEMM instantiations in a single translation unit
// and pushed its peak compiler RSS to ~2.2 GiB, while every other cutlass TU in
// this build instantiates exactly one policy and stays well under that. Each
// `dispatch_*` below is defined in its own generated translation unit
// (`sycl_tla_moe_<dtype>_<policy>.cpp`) and instantiates exactly one policy.
//
// The policy is selected on the host by `select_tile_policy()`, so the split is
// invisible to callers and numerically identical to the fused version.
enum class TilePolicy {
  kN64,   // 256x64x32,  SGLayout 8x1
  kN128,  // 256x128x32, SGLayout 8x2 (historical default)
  kN256,  // 256x256x32, SGLayout 8x4
};

// Whether the N-based tile-policy heuristic is disabled (default: enabled).
//
// Set ``ARK_MOE_GEMM_FIXED_TILE`` to a truthy value ("1"/"true"/"on"/"yes")
// to always use the historical fixed 256x128 (8x2) tile regardless of N.
// This provides an escape hatch should a specific device regress with the
// wider tiles.
inline bool moe_gemm_fixed_tile() {
  const char* env = std::getenv("ARK_MOE_GEMM_FIXED_TILE");
  if (env == nullptr) {
    return false;
  }
  std::string v(env);
  for (auto& c : v) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return !(v == "0" || v == "false" || v == "off" || v == "no" || v.empty());
}

// Select the work-group tile policy from the output width ``gemm_n``,
// mirroring the ``w16a16`` large-M heuristic in vllm-xpu-kernels grouped GEMM:
//
//   * N <= 64  -> 256x64x32,  SGLayout 8x1
//   * N <= 512 -> 256x128x32, SGLayout 8x2 (historical default)
//   * N >  512 -> 256x256x32, SGLayout 8x4
//
// Prefill routes many tokens per expert (large M), so the taller/wider N tile
// increases sub-group utilization and reduces the number of work-group tiles
// launched for the large-N up/down projections. All three policies share the
// same per-sub-group tile (32x64x32), so the copy atoms in
// ``moe_gemm_launcher`` remain valid.
inline TilePolicy select_tile_policy(int gemm_n) {
  if (moe_gemm_fixed_tile()) {
    return TilePolicy::kN128;
  }
  if (gemm_n <= 64) {
    return TilePolicy::kN64;
  }
  if (gemm_n <= 512) {
    return TilePolicy::kN128;
  }
  return TilePolicy::kN256;
}

void dispatch_f16_n64(sycl::queue* q, const void* activations, const void* weights, const void* scales, void* outputs,
                      int N, int K, int* num_tokens_per_expert, int num_experts);
void dispatch_f16_n128(sycl::queue* q, const void* activations, const void* weights, const void* scales, void* outputs,
                       int N, int K, int* num_tokens_per_expert, int num_experts);
void dispatch_f16_n256(sycl::queue* q, const void* activations, const void* weights, const void* scales, void* outputs,
                       int N, int K, int* num_tokens_per_expert, int num_experts);
void dispatch_bf16_n64(sycl::queue* q, const void* activations, const void* weights, const void* scales, void* outputs,
                       int N, int K, int* num_tokens_per_expert, int num_experts);
void dispatch_bf16_n128(sycl::queue* q, const void* activations, const void* weights, const void* scales, void* outputs,
                        int N, int K, int* num_tokens_per_expert, int num_experts);
void dispatch_bf16_n256(sycl::queue* q, const void* activations, const void* weights, const void* scales, void* outputs,
                        int N, int K, int* num_tokens_per_expert, int num_experts);

}  // namespace moe_gemm_detail
}  // namespace ark

#endif
