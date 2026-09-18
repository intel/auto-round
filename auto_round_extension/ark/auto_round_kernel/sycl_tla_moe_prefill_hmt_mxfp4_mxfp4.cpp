#include "sycl_tla_common.hpp"
#include "xpu_mxfp4_hadamard.hpp"

namespace ark {

void moe_gemm_prefill_mxfp4_mxfp4(sycl::queue* q, void* activations, void* activation_scales, void* weights,
                                  void* weight_scales, void* outputs, void* activation_workspace,
                                  void* weight_workspace, BTLA_DTYPE output_dtype, BTLA_DTYPE activation_dtype,
                                  int N, int K, int group_size, int* num_tokens_per_expert, int num_experts,
                                  int total_tokens, bool refresh_weight_staging, int* num_tokens_per_expert_host,
                                  bool refresh_metadata);

template <typename ActivationT>
void moe_gemm_prefill_hmt_mxfp4_mxfp4_impl(sycl::queue* q, void* activations, void* hadamard, void* weights,
                                           void* weight_scales, void* outputs, void* activation_workspace,
                                           void* weight_workspace, BTLA_DTYPE output_dtype, int N, int K,
                                           int group_size, int* num_tokens_per_expert, int num_experts,
                                           int total_tokens, bool use_fwht, int hadamard_dim,
                                           bool refresh_weight_staging, int* num_tokens_per_expert_host,
                                           bool refresh_metadata) {
  if (group_size != ark::XpuMxfp4Hadamard::kGroupSize) {
    throw std::invalid_argument("moe_gemm_prefill_hmt_mxfp4_mxfp4: group_size must be 32");
  }
  if (K % 64 != 0 || K % hadamard_dim != 0) {
    throw std::invalid_argument("moe_gemm_prefill_hmt_mxfp4_mxfp4: unsupported K/hadamard_dim shape");
  }
  if (!ark::XpuMxfp4Hadamard::is_supported_hadamard_dim(hadamard_dim)) {
    throw std::invalid_argument("moe_gemm_prefill_hmt_mxfp4_mxfp4: unsupported hadamard_dim");
  }
  if (!use_fwht && hadamard_dim != ark::XpuMxfp4Hadamard::kHadamardDim) {
    throw std::invalid_argument("moe_gemm_prefill_hmt_mxfp4_mxfp4: custom Hadamard is supported only for D=32");
  }

  auto* workspace_bytes = static_cast<uint8_t*>(activation_workspace);
  auto* packed_activations = workspace_bytes;
  auto* activation_scales = packed_activations + static_cast<size_t>(total_tokens) * (K / 2);
  auto* bdpas_activation_workspace = activation_scales + static_cast<size_t>(total_tokens) * (K / group_size);

  ark::XpuMxfp4Hadamard::mxfp4_hadamard_quant<ActivationT>(
      q, static_cast<const ActivationT*>(activations), static_cast<const float*>(hadamard), packed_activations,
      activation_scales, total_tokens, K, use_fwht, /*quant_only=*/false, hadamard_dim, /*stream_only=*/false);

  moe_gemm_prefill_mxfp4_mxfp4(q, packed_activations, activation_scales, weights, weight_scales, outputs,
                               bdpas_activation_workspace, weight_workspace, output_dtype, BTLA_DTYPE::F4_E2M1, N, K,
                               group_size, num_tokens_per_expert, num_experts, total_tokens, refresh_weight_staging,
                               num_tokens_per_expert_host, refresh_metadata);
}

void moe_gemm_prefill_hmt_mxfp4_mxfp4(sycl::queue* q, void* activations, void* hadamard, void* weights,
                                      void* weight_scales, void* outputs, void* activation_workspace,
                                      void* weight_workspace, BTLA_DTYPE output_dtype, BTLA_DTYPE activation_dtype,
                                      int N, int K, int group_size, int* num_tokens_per_expert, int num_experts,
                                      int total_tokens, bool use_fwht, int hadamard_dim, bool refresh_weight_staging,
                                      int* num_tokens_per_expert_host, bool refresh_metadata) {
  if (total_tokens == 0) return;
  if (activation_workspace == nullptr || weight_workspace == nullptr) {
    throw std::invalid_argument("moe_gemm_prefill_hmt_mxfp4_mxfp4: workspace pointers must be non-null");
  }
  if (hadamard == nullptr) {
    throw std::invalid_argument("moe_gemm_prefill_hmt_mxfp4_mxfp4: hadamard pointer must be non-null");
  }
  if (activation_dtype != BTLA_DTYPE::F16 && activation_dtype != BTLA_DTYPE::BF16) {
    throw std::invalid_argument("moe_gemm_prefill_hmt_mxfp4_mxfp4: activations must be F16 or BF16");
  }
  if (output_dtype != BTLA_DTYPE::BF16) {
    throw std::invalid_argument("moe_gemm_prefill_hmt_mxfp4_mxfp4: output_dtype must be BF16");
  }
  if (K % group_size != 0) {
    throw std::invalid_argument("moe_gemm_prefill_hmt_mxfp4_mxfp4: K must be a multiple of group_size");
  }

  if (activation_dtype == BTLA_DTYPE::F16) {
    moe_gemm_prefill_hmt_mxfp4_mxfp4_impl<sycl::half>(
        q, activations, hadamard, weights, weight_scales, outputs, activation_workspace, weight_workspace,
        output_dtype, N, K, group_size, num_tokens_per_expert, num_experts, total_tokens, use_fwht, hadamard_dim,
        refresh_weight_staging, num_tokens_per_expert_host, refresh_metadata);
    return;
  }
  moe_gemm_prefill_hmt_mxfp4_mxfp4_impl<sycl::ext::oneapi::bfloat16>(
      q, activations, hadamard, weights, weight_scales, outputs, activation_workspace, weight_workspace, output_dtype,
      N, K, group_size, num_tokens_per_expert, num_experts, total_tokens, use_fwht, hadamard_dim,
      refresh_weight_staging, num_tokens_per_expert_host, refresh_metadata);
}

}  // namespace ark