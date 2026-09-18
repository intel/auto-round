#pragma once

#include "sycl_tla_common.hpp"

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

namespace ark {

#if defined(ARK_MXFP_BDPAS_CUTLASS)
bool sycl_tla_moe_prefill_mxfp8_mxfp4_bdpas(sycl::queue* q, void* activations, void* activation_scales,
                                            void* weights, void* weight_scales, void* outputs,
                                            void* activation_workspace, void* weight_workspace,
                                            BTLA_DTYPE output_dtype, BTLA_DTYPE activation_dtype, int N, int K,
                                            int group_size, int* num_tokens_per_expert, int num_experts,
                                              int total_tokens, bool refresh_weight_staging,
                                              int* num_tokens_per_expert_host, bool refresh_metadata);
bool sycl_tla_moe_prefill_mxfp4_mxfp4_bdpas(sycl::queue* q, void* activations, void* activation_scales,
                                            void* weights, void* weight_scales, void* outputs,
                                            void* activation_workspace, void* weight_workspace,
                                            BTLA_DTYPE output_dtype, BTLA_DTYPE activation_dtype, int N, int K,
                                            int group_size, int* num_tokens_per_expert, int num_experts,
                                            int total_tokens, bool refresh_weight_staging,
                                            int* num_tokens_per_expert_host, bool refresh_metadata);
bool sycl_tla_moe_prefill_hmt_mxfp4_mxfp4_bdpas(sycl::queue* q, void* activations, void* hadamard,
                                                void* weights, void* weight_scales, void* outputs,
                                                void* activation_workspace, void* weight_workspace,
                                                BTLA_DTYPE output_dtype, BTLA_DTYPE activation_dtype, int N, int K,
                                                int group_size, int* num_tokens_per_expert, int num_experts,
                                                int total_tokens, bool use_fwht, int hadamard_dim,
                                                bool refresh_weight_staging, int* num_tokens_per_expert_host,
                                                bool refresh_metadata);
#else
inline bool sycl_tla_moe_prefill_mxfp8_mxfp4_bdpas(sycl::queue*, void*, void*, void*, void*, void*, void*, void*,
                                                BTLA_DTYPE, BTLA_DTYPE, int, int, int, int*, int, int, bool, int*,
                                                bool) {
  return false;
}
inline bool sycl_tla_moe_prefill_mxfp4_mxfp4_bdpas(sycl::queue*, void*, void*, void*, void*, void*, void*, void*,
                                                BTLA_DTYPE, BTLA_DTYPE, int, int, int, int*, int, int, bool, int*,
                                                bool) {
  return false;
}
inline bool sycl_tla_moe_prefill_hmt_mxfp4_mxfp4_bdpas(sycl::queue*, void*, void*, void*, void*, void*, void*, void*,
                                                       BTLA_DTYPE, BTLA_DTYPE, int, int, int, int*, int, int, bool,
                                                       int, bool, int*, bool) {
  return false;
}
#endif

}  // namespace ark

#endif